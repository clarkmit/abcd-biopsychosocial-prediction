"""Race/ethnicity and religion fairness, with a partial-correlation confound check (the `fairness` stage, Supplement S5.1).

Held-out performance stratified by race/ethnicity and religion, plus a
partial-correlation check of whether predictions depend on these (and other
demographic) variables.

Inputs: the fitted stacking ensemble and held-out test set of each target
archived by the `main` stage (_batch_archive, or the current run in
single-target mode). The test matrix is rebuilt exactly as the runner
prepared it for the ensemble, and each target is used only if the rebuilt
set reproduces the archived held-out score. Nothing is refit and no other
validation stage needs to run first. Across-category models only.

Part 1  Held-out R² (regression) or AUC (classification) within each
race/ethnicity group, religious affiliation and level of religious
importance, with a bootstrap 95% CI and the difference from all
remaining test participants (bootstrap 95% CI). '*' = that
difference CI excludes 0 (uncorrected for the number of groups).
Part 2  Correlation between observed and predicted outcome before and
after both are residualized on (a) race/ethnicity + religion
(affiliation, importance, attendance) and
(b) these plus child sex, parent income, parent education and
(optionally) site. For classification the prediction is
logit(predicted probability).

Race/ethnicity: Hispanic/Latino (any race), then non-Hispanic White, Black
and Asian (single race), and Other/Multiracial (American Indian/Alaska
Native, more than one race, or none of the listed categories). Built from
the desc_* race indicators and desc_latin_B (ABCD demo_ethn_v2: 1 = yes).
If ABCD's race_ethnicity column has been merged into the panel it is used
instead. Religion: child's religious preference (child_religion, ABCD
demo_relig_v2) as named affiliations (affiliations with fewer than
frr_min_group_n test participants pooled) or as broad traditions; importance
of religion (relig_importance); and attendance at religious services
(religious_service_frequency, ABCD demo_yrs_1). Codes 777 (refused) and 999
(don't know) are treated as missing.
"""

import logging

from bpsp.context import ctx

logger = logging.getLogger(__name__)


# ABCD response codes.
FRR_RELIGION_LABELS = {  # child_religion (demo_relig_v2)
    1: "Mainline Protestant",
    2: "Evangelical Protestant",
    3: "Historically Black Church",
    4: "Roman Catholic",
    5: "Jewish",
    6: "Mormon (LDS)",
    7: "Jehovah's Witness",
    8: "Muslim",
    9: "Buddhist",
    10: "Hindu",
    11: "Orthodox Christian",
    12: "Unitarian Universalist",
    13: "Other Christian",
    14: "Atheist",
    15: "Agnostic",
    16: "Something else",
    17: "Nothing in particular",
}
FRR_RELIGION_BROAD = {  # used when frr_religion_grouping = "broad tradition"
    **{k: "Christian" for k in (1, 2, 3, 4, 6, 7, 11, 13)},
    **{k: "Other religion" for k in (5, 8, 9, 10, 12, 16)},
    14: "Atheist or agnostic",
    15: "Atheist or agnostic",
    17: "Nothing in particular",
}
FRR_ATTENDANCE_LABELS = {  # religious_service_frequency (demo_yrs_1)
    0: "Never",
    1: "Less than monthly",
    2: "1 to 3 times a month",
    3: "About weekly",
    4: "More than weekly",
}
FRR_IMPORTANCE_LABELS = {}  # relig_importance codes 1 to 4; add labels here, e.g. {1: '...'}
FRR_NONRESPONSE = (777, 999)

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import rankdata

_FRR_RACE_COLS = {
    "white": ["desc_european_EUR_B"],
    "black": ["desc_african_AFR_B"],
    "asian": [
        "desc_chinese_EAS_B",
        "desc_japanese_EAS_B",
        "desc_korean_EAS_B",
        "desc_vietnamese_EAS_B",
        "desc_asian_indian_SAS_B",
        "desc_other_south_asian_SAS_B",
    ],
    "aian": ["desc_native_american_AMR_B", "desc_alaska_native_AMR_B"],
}
_FRR_HISP_COL = "desc_latin_B"
_FRR_FIXED = [c for v in _FRR_RACE_COLS.values() for c in v] + [
    _FRR_HISP_COL,
    "race_ethnicity",
    "child_religion",
    "relig_importance",
    "religious_service_frequency",
    "sex",
    "parent_income",
    "parent_education",
    "L_site_id",
]


def _frr_entries(selection):
    """Archived targets (batch mode) or the current globals (single-target mode)."""
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
                "sample_valid_test",
                "split",
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


def _frr_tp(key, e):
    if e.get("timepoint") is not None:
        return e["timepoint"]
    return str(key).split("__tp")[-1] if "__tp" in str(key) else ctx.get("tp_option", "?")


def _frr_model_inputs(e):
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


def _frr_reference_score(e, is_clf):
    """Held-out meta-model score stored by evaluate_models() for this (across-category) model."""
    try:
        meta = e["metrics_dict_full"]["metrics"][-1]["meta_model"]
        return float(meta["auc" if is_clf else "r2"])
    except Exception:
        return None


def _frr_prepare(key, e):
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
        Xtr, Xte, y = _frr_model_inputs(e)
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
    ref = _frr_reference_score(e, is_clf)
    if is_clf:
        from sklearn.metrics import roc_auc_score

        got = roc_auc_score(y, p)
    else:
        got = 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    if ref is not None and abs(got - ref) > 1e-6:
        logger.info(f"  [{key}] skipped: rebuilt test set gives {got:.6f} but the archived held-out score is {ref:.6f}")
        return None
    return m, Xtr, Xte, y, is_clf, p


def _frr_sample_frame(key, e):
    """Raw demographics aligned row-for-row with X_test."""
    n = len(e["y_test"])
    sv = e.get("sample_valid_test")
    if isinstance(sv, pd.DataFrame) and len(sv) == n:
        sv = sv.reset_index(drop=True).copy()
    else:
        sv = None
        smp = ctx.get("sample")
        try:  # same reconstruction as the Main Analysis Runner (full sample only; length-checked)
            tgt, tp = e.get("target"), int(_frr_tp(key, e))
            f = smp[smp["time"] == tp]
            f = f[f[tgt].notna()].reset_index(drop=True)
            if len(f) == len(e["X_train"]) + len(e["X_test"]):
                sv = f.iloc[e["X_test"].index.values].reset_index(drop=True).copy()
        except Exception:
            sv = None
    if sv is None:
        return None
    # Race, ethnicity and religion are fixed per child: fill any gaps from the child's other waves.
    smp = ctx.get("sample")
    if isinstance(smp, pd.DataFrame) and "subject" in sv.columns and "subject" in smp.columns:
        cols = [c for c in _FRR_FIXED if c in smp.columns]
        if cols:
            first = smp.groupby("subject")[cols].first()
            for c in cols:
                filled = sv["subject"].map(first[c])
                sv[c] = sv[c].fillna(filled) if c in sv.columns else filled
    return sv


def _frr_race_ethnicity(sv):
    if "race_ethnicity" in sv.columns and pd.to_numeric(sv["race_ethnicity"], errors="coerce").notna().any():
        m = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other/Multiracial"}
        return pd.to_numeric(sv["race_ethnicity"], errors="coerce").map(m)
    present = [c for c in [c for v in _FRR_RACE_COLS.values() for c in v] + [_FRR_HISP_COL] if c in sv.columns]
    if not present:
        return None

    def _any(cols):
        out = pd.Series(False, index=sv.index)
        for c in cols:
            if c in sv.columns:
                out |= pd.to_numeric(sv[c], errors="coerce") == 1
        return out

    flags = {k: _any(v) for k, v in _FRR_RACE_COLS.items()}
    n_races = sum(f.astype(int) for f in flags.values())
    out = pd.Series("Other/Multiracial", index=sv.index, dtype=object)
    for k, lbl in [("white", "White"), ("black", "Black"), ("asian", "Asian")]:
        out[(n_races == 1) & flags[k]] = lbl
    out[_any([_FRR_HISP_COL])] = "Hispanic"
    out[sv[present].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)] = np.nan
    return out


def _frr_codes(sv, col):
    if col not in sv.columns:
        return None
    s = pd.to_numeric(sv[col], errors="coerce")
    return s.where(~s.isin(FRR_NONRESPONSE))


def _frr_religion_affiliation(sv, min_n):
    s = _frr_codes(sv, "child_religion")
    if s is None:
        return None
    if ctx.frr_religion_grouping == "broad tradition":
        return s.map(lambda v: np.nan if pd.isna(v) else FRR_RELIGION_BROAD.get(int(v), f"Code {int(v)}"))
    counts = s.value_counts()
    keep = set(counts[counts >= min_n].index.tolist())
    return s.map(
        lambda v: (
            np.nan
            if pd.isna(v)
            else (
                FRR_RELIGION_LABELS.get(int(v), f"Code {int(v)}")
                if v in keep
                else f"Other affiliations (each n < {min_n})"
            )
        )
    )


def _frr_levels(sv, col, labels, prefix):
    s = _frr_codes(sv, col)
    if s is None:
        return None
    return s.map(lambda v: np.nan if pd.isna(v) else labels.get(int(v), f"{prefix} {int(v)}"))


def _frr_level_order(dim, groups):
    """Keep ordinal scales (importance, attendance) in code order; other dimensions by size."""
    lookup = {"Religious importance": FRR_IMPORTANCE_LABELS, "Service attendance": FRR_ATTENDANCE_LABELS}.get(dim)
    if lookup is None:
        return None
    inv = {v: k for k, v in lookup.items()}
    return sorted(groups, key=lambda g: inv.get(g, int(str(g).split()[-1]) if str(g).split()[-1].isdigit() else 99))


def _frr_auc(y, p):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _frr_r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan


def _frr_ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) >= 20 else (np.nan, np.nan)


def _frr_group_stats(y, p, mask, fn, n_boot, rng):
    yg, pg, yr, pr = y[mask], p[mask], y[~mask], p[~mask]
    est, rest = fn(yg, pg), fn(yr, pr)
    bg, bd = [], []
    for _ in range(n_boot):
        ig, ir = rng.randint(0, len(yg), len(yg)), rng.randint(0, len(yr), len(yr))
        a, b = fn(yg[ig], pg[ig]), fn(yr[ir], pr[ir])
        if np.isfinite(a) and np.isfinite(b):
            bg.append(a)
            bd.append(a - b)
    return est, _frr_ci(bg), est - rest, _frr_ci(bd)


def _frr_dummies(s, name):
    s = s.astype(object).where(s.notna(), "Missing")
    return pd.get_dummies(s, prefix=name, drop_first=True, dtype=float)


def _frr_design(groups, sv, extended, include_site):
    parts = [_frr_dummies(s, k) for k, s in groups.items() if s is not None]
    if extended:
        if "sex" in sv.columns:
            parts.append(pd.DataFrame({"female": (pd.to_numeric(sv["sex"], errors="coerce") == 2).astype(float)}))
        for c in ["parent_income", "parent_education"]:
            if c in sv.columns:
                v = pd.to_numeric(sv[c], errors="coerce")
                parts.append(pd.DataFrame({c: v.fillna(v.median()), c + "_missing": v.isna().astype(float)}))
        if include_site and "L_site_id" in sv.columns:
            parts.append(_frr_dummies(sv["L_site_id"], "site"))
    C = pd.concat([d.reset_index(drop=True) for d in parts], axis=1) if parts else pd.DataFrame()
    if C.shape[1]:
        C = C.loc[:, C.std() > 0]
    return np.column_stack([np.ones(len(sv)), C.values]) if C.shape[1] else np.ones((len(sv), 1))


def _frr_resid(v, C):
    beta, *_ = np.linalg.lstsq(C, v, rcond=None)
    return v - C @ beta


def _frr_partial(y, yhat, C, n_boot, rng):
    def _stats(idx):
        yy, hh, CC = y[idx], yhat[idx], C[idx]
        return np.corrcoef(yy, hh)[0, 1], np.corrcoef(_frr_resid(yy, CC), _frr_resid(hh, CC))[0, 1]

    r, pr = _stats(np.arange(len(y)))
    br, bp, bd = [], [], []
    for _ in range(n_boot):
        a, b = _stats(rng.randint(0, len(y), len(y)))
        br.append(a)
        bp.append(b)
        bd.append(a - b)
    return dict(
        r=r,
        r_ci=_frr_ci(br),
        pr=pr,
        pr_ci=_frr_ci(bp),
        d=r - pr,
        d_ci=_frr_ci(bd),
        k=C.shape[1] - 1,
        r2_pred=1 - np.var(_frr_resid(yhat, C)) / np.var(yhat),
        r2_obs=1 - np.var(_frr_resid(y, C)) / np.var(y),
    )


_FRR_INK, _FRR_INK2, _FRR_MUTED = "#0b0b0b", "#52514e", "#9a9893"
_FRR_GRID, _FRR_BAND = "#e6e5e1", "#ecebe7"
_FRR_BLUE, _FRR_ORANGE = "#2a78d6", "#eb6834"


def _frr_target_label(tgt):
    names = ctx.get("TARGET_DISPLAY_NAMES")
    return names.get(tgt, tgt) if isinstance(names, dict) else tgt


def _frr_axes_style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_FRR_MUTED)
    ax.tick_params(axis="x", colors=_FRR_INK2, labelsize=8.5, length=3)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color=_FRR_GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


def _frr_forest(tab, metric, tgt, tp, out_dir):
    """Forest plot for one target: groups under dimension headers, overall estimate as a band."""
    ov = tab[tab["Dimension"] == "Overall"].iloc[0]
    rows = []  # (label, row or None, kind)
    for dim in [d for d in dict.fromkeys(tab["Dimension"]) if d != "Overall"]:
        rows.append((dim, None, "header"))
        for _, r in tab[(tab["Dimension"] == dim) & (tab["Group"] != "(missing)")].iterrows():
            rows.append((str(r["Group"]), r, "group"))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * n + 1.5))
    ypos = np.arange(n)[::-1]
    ax.axvspan(ov["CI_lo"], ov["CI_hi"], color=_FRR_BAND, lw=0, zorder=0)
    ax.axvline(ov["Estimate"], color=_FRR_INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    scored = [r for _, r, k in rows if k == "group" and np.isfinite(r["Estimate"])]
    ests = [r["Estimate"] for r in scored] + [ov["CI_lo"], ov["CI_hi"]]
    span = max(ests) - min(ests) if len(ests) > 1 else 0.2
    lim_lo = max(min([r["CI_lo"] for r in scored] + [ov["CI_lo"]]), min(ests) - max(0.15, 0.6 * span))
    lim_hi = min(max([r["CI_hi"] for r in scored] + [ov["CI_hi"]]), max(ests) + max(0.15, 0.6 * span))
    pad = 0.04 * (lim_hi - lim_lo)
    lim_lo, lim_hi = lim_lo - pad, lim_hi + pad
    for y, (lbl, r, kind) in zip(ypos, rows):
        if kind != "group" or not np.isfinite(r["Estimate"]):
            continue
        col = _FRR_ORANGE if r["Note"] == "*" else _FRR_BLUE
        a, b = max(r["CI_lo"], lim_lo), min(r["CI_hi"], lim_hi)
        ax.plot([a, b], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        if r["CI_lo"] < lim_lo:  # interval runs past the axis: mark the cut end
            ax.plot(a, y, marker="<", ms=5, color=col, zorder=3, clip_on=False)
        if r["CI_hi"] > lim_hi:
            ax.plot(b, y, marker=">", ms=5, color=col, zorder=3, clip_on=False)
        ax.plot(r["Estimate"], y, "o", ms=6, color=col, mec="white", mew=1.2, zorder=3)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lbl for lbl, _, _ in rows], fontsize=8.5)
    for tick, (lbl, r, kind) in zip(ax.get_yticklabels(), rows):
        if kind == "header":
            tick.set_fontweight("bold")
            tick.set_color(_FRR_INK)
        else:
            tick.set_color(_FRR_INK2 if np.isfinite(r["Estimate"]) else _FRR_MUTED)
    tr = ax.get_yaxis_transform()
    ax.text(1.02, ypos[0], "n", transform=tr, ha="left", va="center", fontsize=8, color=_FRR_INK2, fontweight="bold")
    for y, (lbl, r, kind) in zip(ypos, rows):
        if kind == "group":
            note = (
                ""
                if np.isfinite(r["Estimate"])
                else ("  not scored" if r["Note"] == "too few to score" else "  no variance")
            )
            ax.text(
                1.02,
                y,
                f"{int(r['N']):,}{note}",
                transform=tr,
                ha="left",
                va="center",
                fontsize=8,
                color=_FRR_INK2 if np.isfinite(r["Estimate"]) else _FRR_MUTED,
            )
    _frr_axes_style(ax)
    ax.set_xlabel(f"Held-out {metric} (95% bootstrap CI)", fontsize=9, color=_FRR_INK2)
    ax.set_title(f"{_frr_target_label(tgt)}, T{tp}", loc="left", fontsize=11, fontweight="bold", color=_FRR_INK, pad=34)
    ax.text(
        0,
        1.012,
        f"Dashed line and band: overall {metric} {ov['Estimate']:.2f} "
        f"[{ov['CI_lo']:.2f}, {ov['CI_hi']:.2f}], n = {int(ov['N']):,}\n"
        "Orange: 95% CI of the difference from all other participants excludes 0 (uncorrected)",
        transform=ax.transAxes,
        fontsize=8,
        color=_FRR_INK2,
        va="bottom",
        linespacing=1.5,
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"forest_{tgt}_T{tp}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def _frr_difference_map(df, out_dir):
    """Across targets: each group's difference from all other participants, diverging around 0."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("frr_div", [_FRR_ORANGE, "#f4f3f0", _FRR_BLUE])
    for metric in df["Metric"].unique():
        sub = df[(df["Metric"] == metric) & (df["Dimension"] != "Overall") & df["Diff_vs_rest"].notna()].copy()
        if sub.empty:
            continue
        sub["row"] = [f"{_frr_target_label(t)}, T{tp}" for t, tp in zip(sub["Target"], sub["TP"])]
        dim_order = list(dict.fromkeys(sub["Dimension"]))
        cols = [c for d in dim_order for c in dict.fromkeys(zip(sub["Dimension"], sub["Group"])) if c[0] == d]
        rows = list(dict.fromkeys(sub["row"]))
        M = np.full((len(rows), len(cols)), np.nan)
        F = np.zeros_like(M, dtype=bool)
        for _, r in sub.iterrows():
            i, j = rows.index(r["row"]), cols.index((r["Dimension"], r["Group"]))
            M[i, j], F[i, j] = r["Diff_vs_rest"], r["Note"] == "*"
        vmax = np.nanmax(np.abs(M)) or 1
        fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(cols) + 3.2), 0.42 * len(rows) + 2.8))
        ax.grid(False)
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isfinite(M[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{M[i, j]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color=_FRR_INK,
                        fontweight="bold" if F[i, j] else "normal",
                    )
                    if F[i, j]:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=_FRR_INK, lw=1.2))
        dims = [d for d, _ in cols]
        for j in range(1, len(cols)):
            if dims[j] != dims[j - 1]:
                ax.axvline(j - 0.5, color="white", lw=4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([g for _, g in cols], rotation=55, ha="right", fontsize=7.5, color=_FRR_INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8, color=_FRR_INK)
        for j0 in [0] + [j for j in range(1, len(cols)) if dims[j] != dims[j - 1]]:
            width = max(9, 6 * dims.count(dims[j0]))
            ax.text(
                j0 - 0.4,
                -0.7,
                textwrap.fill(dims[j0], width),
                fontsize=8,
                fontweight="bold",
                color=_FRR_INK,
                ha="left",
                va="bottom",
                linespacing=1.1,
            )
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
        cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.015)
        cb.set_label(f"{metric}: group minus all others", fontsize=8, color=_FRR_INK2)
        cb.ax.grid(False)
        cb.ax.tick_params(labelsize=7, colors=_FRR_INK2, length=2)
        cb.outline.set_visible(False)
        ax.set_title(
            f"Held-out {metric} by group relative to all other participants",
            loc="left",
            fontsize=11,
            fontweight="bold",
            color=_FRR_INK,
            pad=36,
        )
        fig.text(
            0.01,
            0.005,
            "Boxed, bold cells: 95% CI of the difference excludes 0 (uncorrected). Blank: group too small to score.",
            fontsize=7.5,
            color=_FRR_INK2,
        )
        fig.patch.set_facecolor("white")
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        plt.savefig(
            os.path.join(out_dir, f"group_differences_{'auc' if metric == 'AUC' else 'r2'}.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.show()


def _run_fairness_race_religion():
    out_dir = os.path.join(os.getcwd(), "fairness_race_religion")
    os.makedirs(out_dir, exist_ok=True)
    rows, pc_rows = [], []
    sel = _frr_entries(ctx.frr_targets)
    logger.info(f"Race/ethnicity & religion fairness: {len(sel)} target(s) selected")

    for key, e in sel.items():
        tgt, tp = e.get("target", key), _frr_tp(key, e)
        prep = _frr_prepare(key, e)
        if prep is None:
            continue
        _, _, _, y, is_clf, p = prep
        sv = _frr_sample_frame(key, e)
        if sv is None or len(sv) != len(y):
            logger.info(f"  [{key}] skipped: demographics could not be aligned with the test set")
            continue

        groups = {
            "Race/ethnicity": _frr_race_ethnicity(sv),
            "Religious affiliation": _frr_religion_affiliation(sv, ctx.frr_min_group_n),
            "Religious importance": _frr_levels(sv, "relig_importance", FRR_IMPORTANCE_LABELS, "Importance"),
        }
        if ctx.frr_include_service_attendance:
            groups["Service attendance"] = _frr_levels(
                sv, "religious_service_frequency", FRR_ATTENDANCE_LABELS, "Attendance"
            )
        groups = {k: v for k, v in groups.items() if v is not None}

        metric, fn = ("AUC", _frr_auc) if is_clf else ("R²", _frr_r2)
        rng = np.random.RandomState(42)
        overall = fn(y, p)
        ob = [fn(y[i], p[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(ctx.frr_n_bootstrap))]
        olo, ohi = _frr_ci(ob)
        base = dict(Target=tgt, TP=tp, Metric=metric)
        rows.append(
            dict(
                base,
                Dimension="Overall",
                Group="Overall",
                N=len(y),
                Pos=int(y.sum()) if is_clf else np.nan,
                Estimate=overall,
                CI_lo=olo,
                CI_hi=ohi,
                Diff_vs_rest=np.nan,
                Diff_lo=np.nan,
                Diff_hi=np.nan,
                Note="",
            )
        )
        for dim, s in groups.items():
            present = list(s.dropna().unique())
            order = _frr_level_order(dim, present) or sorted(present, key=lambda g: (-int((s == g).sum()), str(g)))
            for g in order:
                mask = (s == g).values
                n = int(mask.sum())
                row = dict(base, Dimension=dim, Group=g, N=n, Pos=int(y[mask].sum()) if is_clf else np.nan)
                small = n < ctx.frr_min_group_n
                if is_clf:
                    npos = int(y[mask].sum())
                    small = small or npos < ctx.frr_min_group_pos or (n - npos) < ctx.frr_min_group_pos
                if small:
                    row.update(
                        Estimate=np.nan,
                        CI_lo=np.nan,
                        CI_hi=np.nan,
                        Diff_vs_rest=np.nan,
                        Diff_lo=np.nan,
                        Diff_hi=np.nan,
                        Note="too few to score",
                    )
                elif not is_clf and np.var(y[mask]) == 0:
                    row.update(
                        Estimate=np.nan,
                        CI_lo=np.nan,
                        CI_hi=np.nan,
                        Diff_vs_rest=np.nan,
                        Diff_lo=np.nan,
                        Diff_hi=np.nan,
                        Note="no outcome variance in group",
                    )
                else:
                    est, (lo, hi), d, (dlo, dhi) = _frr_group_stats(y, p, mask, fn, ctx.frr_n_bootstrap, rng)
                    flag = "*" if (np.isfinite(dlo) and (dlo > 0 or dhi < 0)) else ""
                    row.update(Estimate=est, CI_lo=lo, CI_hi=hi, Diff_vs_rest=d, Diff_lo=dlo, Diff_hi=dhi, Note=flag)
                rows.append(row)
            nmiss = int(s.isna().sum())
            if nmiss:
                rows.append(
                    dict(
                        base,
                        Dimension=dim,
                        Group="(missing)",
                        N=nmiss,
                        Pos=np.nan,
                        Estimate=np.nan,
                        CI_lo=np.nan,
                        CI_hi=np.nan,
                        Diff_vs_rest=np.nan,
                        Diff_lo=np.nan,
                        Diff_hi=np.nan,
                        Note="not scored",
                    )
                )

        tab = pd.DataFrame([r for r in rows if r["Target"] == tgt and r["TP"] == tp])
        show = tab.copy()
        show[f"{metric} [95% CI]"] = [
            ("" if not np.isfinite(a) else f"{a:.3f} [{b:.3f}, {c:.3f}]")
            for a, b, c in zip(show["Estimate"], show["CI_lo"], show["CI_hi"])
        ]
        show["Diff vs rest [95% CI]"] = [
            ("" if not np.isfinite(a) else f"{a:+.3f} [{b:+.3f}, {c:+.3f}]")
            for a, b, c in zip(show["Diff_vs_rest"], show["Diff_lo"], show["Diff_hi"])
        ]
        cols = (
            ["Dimension", "Group", "N"]
            + (["Pos"] if is_clf else [])
            + [f"{metric} [95% CI]", "Diff vs rest [95% CI]", "Note"]
        )
        logger.info(f"\n{'=' * 78}\n  {tgt} (T{tp}): held-out {metric} by group\n{'=' * 78}")
        display(show[cols].style.hide(axis="index"))
        if ctx.frr_make_plots:
            _frr_forest(tab, metric, tgt, tp, out_dir)

        if ctx.frr_run_partial_corr:
            yhat = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) if is_clf else p
            pc_groups = {
                "race": groups.get("Race/ethnicity"),
                "rel_aff": groups.get("Religious affiliation"),
                "rel_imp": groups.get("Religious importance"),
                "rel_att": groups.get("Service attendance"),
            }
            ext_lbl = "+ sex, parent income, parent education" + (", site" if ctx.frr_include_site_in_partial else "")
            for set_name, ext in [("Race/ethnicity + religion", False), (ext_lbl, True)]:
                C = _frr_design(pc_groups, sv, ext, ctx.frr_include_site_in_partial)
                res = _frr_partial(y, yhat, C, ctx.frr_n_bootstrap, np.random.RandomState(7))
                pc_rows.append(
                    dict(
                        Target=tgt,
                        TP=tp,
                        Outcome="binary (logit p)" if is_clf else "continuous",
                        Confounds=set_name,
                        k=res["k"],
                        r=res["r"],
                        r_lo=res["r_ci"][0],
                        r_hi=res["r_ci"][1],
                        partial_r=res["pr"],
                        partial_lo=res["pr_ci"][0],
                        partial_hi=res["pr_ci"][1],
                        diff=res["d"],
                        diff_lo=res["d_ci"][0],
                        diff_hi=res["d_ci"][1],
                        R2_pred_on_confounds=res["r2_pred"],
                        R2_obs_on_confounds=res["r2_obs"],
                    )
                )

    fair_df = pd.DataFrame(rows)
    pc_df = pd.DataFrame(pc_rows)
    if len(fair_df):
        fair_df.to_csv(os.path.join(out_dir, "fairness_race_religion_long.csv"), index=False)
        scored = fair_df[(fair_df["Dimension"] != "Overall") & fair_df["Estimate"].notna()]
        gap = (
            scored.groupby(["Target", "TP", "Metric", "Dimension"])["Estimate"]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        gap["max_gap"] = gap["max"] - gap["min"]
        flags = (
            fair_df[fair_df["Note"] == "*"]
            .groupby(["Target", "TP", "Dimension"])
            .size()
            .rename("groups_differing_from_rest")
            .reset_index()
        )
        gap = gap[gap["count"] >= 2]
        gap = gap.merge(flags, on=["Target", "TP", "Dimension"], how="left").fillna({"groups_differing_from_rest": 0})
        gap = gap.rename(columns={"min": "lowest", "max": "highest", "count": "groups_scored"})
        logger.info(f"\n{'=' * 78}\n  Largest gap between scored groups within each dimension\n{'=' * 78}")
        display(
            gap.style.format(
                {"lowest": "{:.3f}", "highest": "{:.3f}", "max_gap": "{:.3f}", "groups_differing_from_rest": "{:.0f}"}
            ).hide(axis="index")
        )
        gap.to_csv(os.path.join(out_dir, "fairness_race_religion_gaps.csv"), index=False)
        logger.info(
            "  '*' in the per-target tables: the group's 95% CI for the difference from all other test "
            "participants excludes 0 (uncorrected for the number of groups and targets)."
        )
        if ctx.frr_make_plots:
            _frr_difference_map(fair_df, out_dir)
    if len(pc_df):
        pc_df.to_csv(os.path.join(out_dir, "partial_correlation_confounds.csv"), index=False)
        s = pc_df.copy()
        s["r [95% CI]"] = [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(s["r"], s["r_lo"], s["r_hi"])]
        s["partial r [95% CI]"] = [
            f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(s["partial_r"], s["partial_lo"], s["partial_hi"])
        ]
        s["r − partial r [95% CI]"] = [
            f"{a:+.3f} [{b:+.3f}, {c:+.3f}]" for a, b, c in zip(s["diff"], s["diff_lo"], s["diff_hi"])
        ]
        logger.info(f"\n{'=' * 78}\n  Partial-correlation confound check (held-out observed vs predicted)\n{'=' * 78}")
        display(
            s[
                [
                    "Target",
                    "TP",
                    "Outcome",
                    "Confounds",
                    "k",
                    "r [95% CI]",
                    "partial r [95% CI]",
                    "r − partial r [95% CI]",
                    "R2_pred_on_confounds",
                    "R2_obs_on_confounds",
                ]
            ]
            .style.format({"R2_pred_on_confounds": "{:.3f}", "R2_obs_on_confounds": "{:.3f}"})
            .hide(axis="index")
        )
        logger.info(
            "  partial r: observed and predicted both residualised on the confound set (k = confound columns). "
            "R2_*_on_confounds: variance in predictions / outcomes explained by the confounds."
        )
    logger.info(f"\nSaved to {out_dir}/")
    return fair_df, pc_df


def run() -> None:
    """Run this stage (notebook cell 21)."""

    ctx.fairness_race_religion_df, ctx.fairness_partial_corr_df = _run_fairness_race_religion()
