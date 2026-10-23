"""Correlation-aware grouped permutation importance: clusters, change vs level, domains (the `grouped_importance` stage, Supplement S5.4).

Permutation importance with features shuffled in groups using one shared row
permutation. This removes the group's joint relationship with the outcome
while keeping the correlations among its members intact, so correlated
features cannot stand in for one another.

Inputs: the fitted stacking ensemble and held-out test set of each target
archived by the `main` stage (_batch_archive, or the current run in
single-target mode). The train/test matrices are rebuilt exactly as the
runner prepared them for the ensemble, and each target is used only if the
rebuilt test set reproduces the archived held-out score. Nothing is refit.
Across-category models only.

1. Correlation clusters among the top-K features by weighted consensus
importance (CatBoost importance if the runner's consensus analysis was
not run), using average-linkage clustering on |Spearman r| in the
training partition. For each multi-feature cluster: drop in held-out R²/AUC
when the cluster is shuffled together vs the sum and the largest of its
members' single-feature drops. A joint drop well above the single drops
means credit is shared among correlated items, so importance is best
read at the cluster level.
2. Change-based (delta_*) features vs all remaining level (non-change)
features, and the Delta Other Psychopathology / Parent Delta
Psychopathology category sets on their own.
3. Optional: the measurement domains defined by the `ablation` stage, each
shuffled with a shared permutation. Requires the `ablation` stage to have
run in this session (its four run_* switches can be off).

Cost: (top-K features + clusters + groups) x gpi_n_repeats predictions of
the ensemble on the test set per target; the estimated time is printed.
"""

import logging
import os
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from bpsp.context import ctx

logger = logging.getLogger(__name__)


def _gpi_entries(selection):
    arch = ctx.get("_batch_archive")
    if isinstance(arch, dict) and len(arch) > 0:
        entries = {k: v for k, v in arch.items() if isinstance(v, dict)}
    elif ctx.get("X_test") is not None and ctx.get("y_test") is not None:
        g = ctx
        e = {
            k: g.get(k)
            for k in [
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "last_fitted_model",
                "split",
                "consensus_df",
                "metrics_dict_full",
            ]
        }
        e["target"], e["timepoint"] = g.get("target_options"), g.get("tp_option")
        entries = {f"{e['target']}__tp{e['timepoint']}": e}
    else:
        entries = {}
    if selection and selection.strip().lower() != "all":
        wanted = [s.strip() for s in selection.split(",") if s.strip()]
        entries = {k: v for k, v in entries.items() if k in wanted or str(v.get("target", k)) in wanted}
    return entries


def _gpi_model_inputs(e):
    """Rebuild the train/test matrices exactly as run_catboost_analysis() prepared them before fitting
    and scoring the ensemble (label-encode -> drop train-constant columns -> handle_outliers per
    partition -> clean_data per partition -> de-duplicate columns -> StandardScaler fit on train).
    The archived X_train/X_test are the matrices passed into run_catboost_analysis()."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    g = ctx
    Xtr, Xte = e["X_train"].copy(), e["X_test"].copy()
    ytr, yte = e["y_train"].copy(), e["y_test"].copy()
    le = LabelEncoder()
    for c in Xtr.select_dtypes(include=["object", "category"]).columns:
        Xtr[c] = le.fit_transform(Xtr[c].astype(str))
        Xte[c] = le.transform(Xte[c].astype(str))
    try:
        const = [c for c in Xtr.columns if Xtr[c].nunique() == 1]
        Xtr, Xte = Xtr.drop(columns=const), Xte.drop(columns=const)
    except Exception:
        pass
    try:
        Xtr, Xte = g["handle_outliers"](Xtr), g["handle_outliers"](Xte)
    except Exception:
        pass
    method, strategy = {
        "fill_with_zero": ("fill_0", None),
        "drop": ("remove", None),
        "impute_mean": ("impute", "mean"),
        "impute_median": ("impute", "median"),
    }[g.get("deal_with_na_values", "impute_median")]
    Xtr, ytr = g["clean_data"](Xtr, ytr, method=method, imputer_strategy=strategy)
    Xte, yte = g["clean_data"](Xte, yte, method=method, imputer_strategy=strategy)
    Xtr = Xtr.loc[:, ~Xtr.columns.duplicated(keep="first")]
    Xte = Xte.loc[:, ~Xte.columns.duplicated(keep="first")]
    sc = StandardScaler()
    Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns)
    Xte = pd.DataFrame(sc.transform(Xte), columns=Xte.columns)
    return Xtr, Xte, np.asarray(yte, dtype=float).ravel()


def _gpi_reference_score(e, is_clf):
    """Held-out meta-model score stored by evaluate_models() for this (across-category) model."""
    try:
        meta = e["metrics_dict_full"]["metrics"][-1]["meta_model"]
        return float(meta["auc" if is_clf else "r2"])
    except Exception:
        return None


def _gpi_prepare(key, e):
    """Returns (model, X_train, X_test, y_test, is_clf, predictions) for an across-category archived
    model, after confirming the rebuilt test matrix reproduces the archived held-out score."""
    if e.get("split") == "within_categories":
        logger.info(f"  [{key}] skipped: within-category entry (across-category models only)")
        return None
    m = e.get("last_fitted_model")
    if m is None or any(e.get(k) is None for k in ("X_train", "X_test", "y_train", "y_test")):
        logger.info(f"  [{key}] skipped: no archived model / train-test data")
        return None
    if str(ctx.get("selection_method", "None")) != "None":
        logger.info(f"  [{key}] skipped: selection_method = {ctx.selection_method!r} cannot be rebuilt here")
        return None
    task = getattr(m, "task_type", None)
    try:
        Xtr, Xte, y = _gpi_model_inputs(e)
        is_clf = (
            (task == "classification")
            if task in ("classification", "regression")
            else (len(np.unique(y)) == 2 and set(np.unique(y).tolist()) <= {0, 1})
        )
        if is_clf and not (len(np.unique(y)) == 2 and set(np.unique(y).tolist()) <= {0, 1}):
            logger.info(f"  [{key}] skipped: multi-class target")
            return None
        p = (np.asarray(m.predict_proba(Xte))[:, 1] if is_clf else np.asarray(m.predict(Xte), float).ravel()).astype(
            float
        )
    except Exception as err:
        logger.info(f"  [{key}] skipped: could not rebuild model inputs or predict ({type(err).__name__}: {err})")
        return None
    ref = _gpi_reference_score(e, is_clf)
    if is_clf:
        from sklearn.metrics import roc_auc_score

        got = roc_auc_score(y, p)
    else:
        got = 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    if ref is not None and abs(got - ref) > 1e-6:
        logger.info(f"  [{key}] skipped: rebuilt test set gives {got:.6f} but the archived held-out score is {ref:.6f}")
        return None
    return m, Xtr, Xte, y, is_clf, p


def _gpi_auc(y, p):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _gpi_score(model, X, y, is_clf):
    if is_clf:
        return _gpi_auc(y, np.asarray(model.predict_proba(X))[:, 1])
    p = np.asarray(model.predict(X), float).ravel()
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def _gpi_drop(model, X, y, cols, is_clf, base, n_rep, rng):
    """Mean and SD of the score drop when `cols` are shuffled together (one shared row permutation)."""
    Xp = X.copy()
    block = X[cols]
    drops = []
    for _ in range(n_rep):
        shuffled = block.iloc[rng.permutation(len(X))]
        shuffled.index = X.index
        Xp[cols] = shuffled  # column-wise assignment keeps each column's dtype
        drops.append(base - _gpi_score(model, Xp, y, is_clf))
    return float(np.mean(drops)), (float(np.std(drops, ddof=1)) if n_rep > 1 else np.nan)


def _gpi_label(f):
    dn = ctx.get("DISPLAY_NAMES")
    return dn.get(f, f) if isinstance(dn, dict) else f


def _gpi_top_features(e, cols, k):
    """Top-k raw column names by weighted consensus importance (consensus_df is indexed by display name)."""
    cdf = e.get("consensus_df")
    if isinstance(cdf, pd.DataFrame) and "weighted_score" in cdf.columns:
        inv = {}
        for raw, disp in (ctx.get("DISPLAY_NAMES") or {}).items():
            inv.setdefault(disp, raw)
        out = []
        for f in cdf.sort_values("weighted_score", ascending=False).index:
            raw = f if f in cols else inv.get(f)
            if raw in cols and raw not in out:
                out.append(raw)
            if len(out) >= k:
                break
        if len(out) >= min(k, 5):
            return out, "weighted consensus importance"
    cb = getattr(e.get("last_fitted_model"), "base_models", {}).get("catboost")
    fi = getattr(cb, "feature_importances_", None)
    if fi is not None and len(fi) == len(cols):
        return list(pd.Series(fi, index=cols).sort_values(ascending=False).index[:k]), "CatBoost importance"
    return cols[:k], "column order (no importance found)"


def _gpi_change_groups(cols, tp):
    groups = {}
    delta = [c for c in cols if c.startswith("delta_")]
    if delta:
        groups["Change-based (all delta_ features)"] = delta
        groups["Level / non-change (all other features)"] = [c for c in cols if c not in delta]
    for dict_name, cat in [
        ("child_variables_for_each_time_point", "Delta Other Psychopathology"),
        ("parent_variables_for_each_time_point", "Parent Delta Psychopathology"),
    ]:
        reg = ctx.get(dict_name)
        if not isinstance(reg, dict):
            continue
        # Category at the target wave; otherwise (e.g. after the objective/subjective spectrum override) the
        # same-named category from the wave whose definition overlaps most with the model's columns
        # (ties: nearest wave), restricted to columns present.
        cand = {w: [c for c in ((reg.get(w) or {}).get(cat) or []) if c in cols] for w in reg}
        cand = {w: v for w, v in cand.items() if v}
        if not cand:
            continue
        if any(str(w) == str(tp) for w in cand):
            w = [w for w in cand if str(w) == str(tp)][0]
        else:
            dist = (lambda w: abs(int(w) - int(tp))) if str(tp).isdigit() else (lambda w: 0)
            w = sorted(cand, key=lambda w: (-len(cand[w]), dist(w), int(w)))[0]
        lbl = f"{cat} (category set)" if str(w) == str(tp) else f"{cat} (category set as defined at T{w})"
        groups[lbl] = cand[w]
    return groups


_GPI_INK, _GPI_INK2, _GPI_MUTED, _GPI_GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
_GPI_BLUE, _GPI_LIGHT, _GPI_ORANGE = "#2a78d6", "#9cc3ef", "#eb6834"


def _gpi_target_label(tgt):
    names = ctx.get("TARGET_DISPLAY_NAMES")
    return names.get(tgt, tgt) if isinstance(names, dict) else tgt


def _gpi_style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GPI_MUTED)
    ax.tick_params(axis="x", colors=_GPI_INK2, labelsize=8, length=3)
    ax.tick_params(axis="y", length=0, labelsize=8.5, colors=_GPI_INK2)
    ax.grid(False)
    ax.grid(axis="x", color=_GPI_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=_GPI_MUTED, lw=0.8)


def _gpi_plot(tgt, tp, metric, base, ct, gt, out_dir):
    """Left: correlated clusters (largest single drop, sum of single drops, joint drop).
    Right: feature groups shuffled together."""
    from matplotlib.lines import Line2D

    heights = [
        h for h in (min(len(ct), 12) * 0.62 + 1.3 if len(ct) else 0, len(gt) * 0.36 + 1.3 if len(gt) else 0) if h
    ]
    fig, axes = plt.subplots(
        len(heights), 1, figsize=(8.2, sum(heights) + 0.6), squeeze=False, gridspec_kw=dict(height_ratios=heights)
    )
    k = 0
    if len(ct):
        ax = axes[k, 0]
        k += 1
        c = ct.head(12).iloc[::-1]
        yy = np.arange(len(c))
        for i, (_, r) in enumerate(c.iterrows()):
            ax.plot(
                [r["Max_single"], r["Joint_drop"]], [i, i], color=_GPI_GRID, lw=2.4, solid_capstyle="round", zorder=1
            )
            ax.plot(r["Max_single"], i, "o", ms=6.5, color=_GPI_MUTED, mec="white", mew=1, zorder=2)
            ax.plot(r["Sum_single"], i, "o", ms=6.5, color=_GPI_LIGHT, mec="white", mew=1, zorder=2)
            ax.plot(r["Joint_drop"], i, "o", ms=7.5, color=_GPI_BLUE, mec="white", mew=1, zorder=3)
        ax.set_yticks(yy)

        def _wrap(m):
            parts = m.split("; ")
            txt = "; ".join(parts[:3]) + (f"; +{len(parts) - 3} more" if len(parts) > 3 else "")
            return textwrap.fill(txt, 46, max_lines=3, placeholder=" …")

        ax.set_yticklabels([_wrap(m) for m in c["Members"]], linespacing=1.15)
        _gpi_style(ax)
        ax.set_xlabel(f"Drop in held-out {metric} when shuffled", fontsize=9, color=_GPI_INK2)
        ax.set_title(
            f"Correlated clusters (|r| ≥ {ctx.gpi_cluster_r_threshold})",
            loc="left",
            fontsize=10,
            fontweight="bold",
            color=_GPI_INK,
            pad=22,
        )
        handles = [
            Line2D([], [], ls="", marker="o", ms=6.5, color=_GPI_MUTED, mec="white", label="Largest single feature"),
            Line2D([], [], ls="", marker="o", ms=6.5, color=_GPI_LIGHT, mec="white", label="Sum of single features"),
            Line2D([], [], ls="", marker="o", ms=7.5, color=_GPI_BLUE, mec="white", label="Cluster shuffled together"),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=3,
            frameon=False,
            fontsize=7.5,
            labelcolor=_GPI_INK2,
            handletextpad=0.2,
            columnspacing=1,
        )
    if len(gt):
        ax = axes[k, 0]

        def order(gname):
            if gname.startswith("Domain:"):
                return 3
            return 0 if gname.startswith("Change") else 1 if "Delta" in gname else 2

        g = gt.assign(_o=gt["Group"].map(order)).sort_values(["_o", "Joint_drop"], ascending=[True, False]).iloc[::-1]
        yy = np.arange(len(g))
        cols = [(_GPI_ORANGE, _GPI_ORANGE, _GPI_MUTED, _GPI_BLUE)[order(n)] for n in g["Group"]]
        ax.barh(yy, g["Joint_drop"], height=0.62, color=cols, edgecolor="white", lw=0.8)
        ax.errorbar(
            g["Joint_drop"], yy, xerr=g["Joint_SD"].fillna(0), fmt="none", ecolor=_GPI_INK2, elinewidth=0.9, capsize=2
        )
        ends = g["Joint_drop"] + g["Joint_SD"].fillna(0)
        xmax = max(ends.max(), 0)
        for y, v, e in zip(yy, g["Joint_drop"], ends):
            ax.text(max(e, 0) + 0.02 * (xmax or 1), y, f"{v:.3f}", va="center", fontsize=7.5, color=_GPI_INK2)
        ax.set_xlim(min(g["Joint_drop"].min(), 0) * 1.1, xmax * 1.18 + 1e-9)
        ax.set_yticks(yy)
        ax.set_yticklabels([n.replace("(category set", "(set").replace(" features)", ")") for n in g["Group"]])
        _gpi_style(ax)
        ax.set_xlabel(f"Drop in held-out {metric} when the group is shuffled together", fontsize=9, color=_GPI_INK2)
        from matplotlib.patches import Patch

        present = [
            (c_, l_)
            for c_, l_ in [(_GPI_ORANGE, "Change features"), (_GPI_MUTED, "Non-change features"), (_GPI_BLUE, "Domain")]
            if c_ in cols
        ]
        ax.legend(
            handles=[Patch(color=c_, label=l_) for c_, l_ in present],
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=len(present),
            frameon=False,
            fontsize=7.5,
            labelcolor=_GPI_INK2,
            handlelength=1,
            handleheight=0.8,
        )
        ax.set_title("Feature groups", loc="left", fontsize=10, fontweight="bold", color=_GPI_INK, pad=22)
    fig.suptitle(
        f"{_gpi_target_label(tgt)}, T{tp}: grouped permutation importance (held-out {metric} = {base:.3f})",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
        color=_GPI_INK,
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"grouped_permutation_{tgt}_T{tp}.png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.show()


def _run_grouped_permutation():
    out_dir = os.path.join(os.getcwd(), "grouped_permutation_importance")
    os.makedirs(out_dir, exist_ok=True)
    feat_rows, clus_rows, grp_rows = [], [], []
    sel = _gpi_entries(ctx.gpi_targets)
    logger.info(f"Grouped permutation importance: {len(sel)} target(s) selected")

    for key, e in sel.items():
        tgt = e.get("target", key)
        tp = e.get("timepoint", str(key).split("__tp")[-1] if "__tp" in str(key) else ctx.get("tp_option", "?"))
        prep = _gpi_prepare(key, e)
        if prep is None:
            continue
        m, Xtr, X, y, is_clf, _ = prep
        metric = "AUC" if is_clf else "R²"
        rng = np.random.RandomState(42)
        t0 = time.time()
        base = _gpi_score(m, X, y, is_clf)
        t_pred = time.time() - t0

        top, src = _gpi_top_features(e, list(X.columns), ctx.gpi_top_k_features)
        R = Xtr[top].apply(lambda c: pd.Series(rankdata(c), index=c.index)).corr().abs().fillna(0).values
        if len(top) > 1:
            D = np.clip(1 - R, 0, None)
            np.fill_diagonal(D, 0)
            lab = fcluster(
                linkage(squareform(D, checks=False), method="average"),
                t=1 - ctx.gpi_cluster_r_threshold,
                criterion="distance",
            )
        else:
            lab = np.array([1])
        clusters = {}
        for f, c in zip(top, lab):
            clusters.setdefault(int(c), []).append(f)
        multi = {c: fs for c, fs in clusters.items() if len(fs) > 1}
        groups = _gpi_change_groups(list(X.columns), tp)
        if ctx.gpi_include_domain_groups and callable(ctx.get("_build_feature_domain_map")):
            try:
                for f, d in ctx["_build_feature_domain_map"](list(X.columns)).items():
                    groups.setdefault(f"Domain: {d}", []).append(f)
            except Exception as err:
                logger.info(f"  [{key}] ablation domain map unavailable ({err})")
        elif ctx.gpi_include_domain_groups:
            logger.info(f"  [{key}] domain groups skipped: run the ablation stage first to add them")

        n_calls = (len(top) + len(multi) + len(groups)) * ctx.gpi_n_repeats
        logger.info(
            f"\n{'=' * 78}\n  {tgt} (T{tp})  held-out {metric} = {base:.4f}   top {len(top)} by {src}\n"
            f"  {len(multi)} multi-feature cluster(s) (|r| ≥ {ctx.gpi_cluster_r_threshold}), {len(groups)} group(s); "
            f"~{n_calls} predictions, est. {n_calls * t_pred / 60:.1f} min\n{'=' * 78}"
        )

        single = {}
        for f in top:
            single[f] = _gpi_drop(m, X, y, [f], is_clf, base, ctx.gpi_n_repeats, rng)
            feat_rows.append(
                dict(
                    Target=tgt,
                    TP=tp,
                    Metric=metric,
                    Feature=f,
                    Label=_gpi_label(f),
                    Cluster=int(lab[top.index(f)]),
                    Drop=single[f][0],
                    SD=single[f][1],
                )
            )
        for c, fs in multi.items():
            jd, jsd = _gpi_drop(m, X, y, fs, is_clf, base, ctx.gpi_n_repeats, rng)
            idx = [top.index(f) for f in fs]
            sub = R[np.ix_(idx, idx)]
            clus_rows.append(
                dict(
                    Target=tgt,
                    TP=tp,
                    Metric=metric,
                    Cluster=c,
                    Size=len(fs),
                    Mean_abs_r=float(sub[np.triu_indices(len(fs), 1)].mean()),
                    Members="; ".join(_gpi_label(f) for f in sorted(fs, key=lambda f: -single[f][0])),
                    Joint_drop=jd,
                    Joint_SD=jsd,
                    Sum_single=float(sum(single[f][0] for f in fs)),
                    Max_single=float(max(single[f][0] for f in fs)),
                )
            )
        for gname, fs in groups.items():
            jd, jsd = _gpi_drop(m, X, y, fs, is_clf, base, ctx.gpi_n_repeats, rng)
            signal = (base - 0.5) if is_clf else base  # AUC above chance, or R²
            grp_rows.append(
                dict(
                    Target=tgt,
                    TP=tp,
                    Metric=metric,
                    Group=gname,
                    N_features=len(fs),
                    Joint_drop=jd,
                    Joint_SD=jsd,
                    Pct_signal_lost=100 * jd / signal if signal > 0 else np.nan,
                )
            )

        ct = pd.DataFrame([r for r in clus_rows if r["Target"] == tgt and r["TP"] == tp])
        if len(ct):
            ct = ct.sort_values("Joint_drop", ascending=False)
            ct["Joint / max single"] = ct["Joint_drop"] / ct["Max_single"].where(ct["Max_single"] > 0)
            logger.info("  Correlated clusters among the top features (shuffled together):")
            display(
                ct[
                    [
                        "Size",
                        "Mean_abs_r",
                        "Joint_drop",
                        "Joint_SD",
                        "Sum_single",
                        "Max_single",
                        "Joint / max single",
                        "Members",
                    ]
                ]
                .style.format(
                    {
                        "Mean_abs_r": "{:.2f}",
                        "Joint_drop": "{:.4f}",
                        "Joint_SD": "{:.4f}",
                        "Sum_single": "{:.4f}",
                        "Max_single": "{:.4f}",
                        "Joint / max single": "{:.1f}",
                    }
                )
                .hide(axis="index")
            )
        else:
            logger.info(
                f"  No two top features reach |r| ≥ {ctx.gpi_cluster_r_threshold}; each top feature is its own cluster."
            )
        gt = pd.DataFrame([r for r in grp_rows if r["Target"] == tgt and r["TP"] == tp])
        if len(gt):
            logger.info(
                "  Feature groups (shuffled together). Pct_signal_lost = drop as % of held-out R², or of AUC above 0.5; "
                "> 100 = shuffled model worse than predicting the mean / than chance:"
            )
            display(
                gt[["Group", "N_features", "Joint_drop", "Joint_SD", "Pct_signal_lost"]]
                .sort_values("Joint_drop", ascending=False)
                .style.format({"Joint_drop": "{:.4f}", "Joint_SD": "{:.4f}", "Pct_signal_lost": "{:.1f}"})
                .hide(axis="index")
            )

        if ctx.gpi_make_plots and (len(ct) or len(gt)):
            _gpi_plot(tgt, tp, metric, base, ct, gt, out_dir)

    feat_df, clus_df, grp_df = pd.DataFrame(feat_rows), pd.DataFrame(clus_rows), pd.DataFrame(grp_rows)
    for d, name in [
        (feat_df, "single_feature_permutation.csv"),
        (clus_df, "cluster_permutation.csv"),
        (grp_df, "group_permutation.csv"),
    ]:
        if len(d):
            d.to_csv(os.path.join(out_dir, name), index=False)
    if len(grp_df):
        chg = grp_df[
            grp_df["Group"].str.startswith(("Change-based", "Level / non-change", "Delta Other", "Parent Delta"))
        ]
        if len(chg):
            w = chg.pivot_table(index=["Target", "TP", "Metric"], columns="Group", values="Joint_drop", aggfunc="first")
            logger.info(
                f"\n{'=' * 78}\n  Change-based vs level features, all targets (drop when shuffled together)\n{'=' * 78}"
            )
            display(w.reset_index().style.format({c: "{:.4f}" for c in w.columns}).hide(axis="index"))
    logger.info(f"\nSaved to {out_dir}/")
    return feat_df, clus_df, grp_df


def run() -> None:
    """Run this stage (notebook cell 24)."""

    ctx.grouped_perm_features_df, ctx.grouped_perm_clusters_df, ctx.grouped_perm_groups_df = _run_grouped_permutation()
