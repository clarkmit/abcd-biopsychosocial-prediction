"""Calibration check and Platt/isotonic recalibration (the `calibration` stage, Supplement S5.2).

Reliability of predicted probabilities for classification targets, before
and after post-hoc recalibration, plus calibration slope/intercept for
regression targets.

Inputs: the fitted stacking ensemble and held-out test set of each target
archived by the `main` stage (_batch_archive, or the current run in
single-target mode). The test matrix is rebuilt exactly as the runner
prepared it for the ensemble, and each target is used only if the rebuilt
set reproduces the archived held-out score. The ensemble is not refit.
Across-category models only.

1. Held-out test set: Brier score (bootstrap 95% CI), count-weighted ECE,
calibration-in-the-large (CITL, ideal 0) and calibration slope (ideal
1) for the original probabilities and after Platt scaling and isotonic
regression. By default the calibrators are cross-fitted within the
held-out set (fit on k-1 folds, applied to the remaining fold), so each
case is recalibrated by a map learned without its own label, on the
same model's probabilities. Option: fit them on the training
partition's out-of-fold probabilities from the `nested_cv` stage; this is
only appropriate when those probabilities are on the same scale as the
final model's (a distribution check is printed).
2. Nested-CV out-of-fold probabilities (only if `nested_cv` has run in this
session): the same metrics, with leave-one-outer-fold-out
recalibration.
3. Regression targets: observed regressed on predicted (slope ideal 1,
intercept ideal 0) and observed vs predicted by decile of prediction.

Reliability diagrams here omit empty bins, plot each bin at its mean
predicted probability and weight ECE by bin size. (The `nested_cv`
calibration panels place bins at fixed bin centers and draw empty bins at an observed
rate of 0, which also enter its unweighted ECE; with low base rates the
upper bins are often empty. Both ECE versions are printed for comparison.)
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from bpsp.context import ctx

logger = logging.getLogger(__name__)

_RCAL_EPS = 1e-6


def _rcal_entries(selection):
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
                "split",
                "metrics_dict_full",
                "reporter_point1",
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


def _rcal_tp(key, e):
    if e.get("timepoint") is not None:
        return e["timepoint"]
    return str(key).split("__tp")[-1] if "__tp" in str(key) else ctx.get("tp_option", "?")


def _rcal_model_inputs(e):
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


def _rcal_reference_score(e, is_clf):
    """Held-out meta-model score stored by evaluate_models() for this (across-category) model."""
    try:
        meta = e["metrics_dict_full"]["metrics"][-1]["meta_model"]
        return float(meta["auc" if is_clf else "r2"])
    except Exception:
        return None


def _rcal_prepare(key, e):
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
        Xtr, Xte, y = _rcal_model_inputs(e)
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
    ref = _rcal_reference_score(e, is_clf)
    if is_clf:
        from sklearn.metrics import roc_auc_score

        got = roc_auc_score(y, p)
    else:
        got = 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    if ref is not None and abs(got - ref) > 1e-6:
        logger.info(f"  [{key}] skipped: rebuilt test set gives {got:.6f} but the archived held-out score is {ref:.6f}")
        return None
    return m, Xtr, Xte, y, is_clf, p


def _rcal_logit(p):
    p = np.clip(np.asarray(p, float), _RCAL_EPS, 1 - _RCAL_EPS)
    return np.log(p / (1 - p))


def _rcal_auc(y, p):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _rcal_bins(p, n_bins, strategy):
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            return np.zeros(len(p), dtype=int)
        return np.searchsorted(edges[1:-1], p, side="right")
    edges = np.linspace(0, 1, n_bins + 1)
    return np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)


def _rcal_ece(y, p, n_bins, strategy):
    """Count-weighted ECE over non-empty bins."""
    b = _rcal_bins(p, n_bins, strategy)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in np.unique(b)))


def _rcal_ece_cell12(y, p, n_bins):
    """ECE as computed by the nested_cv stage (unweighted mean over all equal-width bins, bin centers, empty bins = 0)."""
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    obs = np.zeros(n_bins)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() > 0:
            obs[i] = y[m].mean()
    return float(np.mean(np.abs(obs - centres)))


def _rcal_citl(y, p):
    """Calibration-in-the-large: a in logit P(y=1) = a + logit(p), solved by bisection. Ideal 0."""
    lp = _rcal_logit(p)
    f = lambda a: float((y - expit(a + lp)).sum())  # decreasing in a
    lo, hi = -30.0, 30.0
    if f(lo) < 0 or f(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return float((lo + hi) / 2)


def _rcal_slope(y, p):
    """Calibration slope: b in logit P(y=1) = c + b * logit(p). Ideal 1."""
    lr = LogisticRegression(C=1e10, max_iter=5000)
    lr.fit(_rcal_logit(p).reshape(-1, 1), y)
    return float(lr.coef_.ravel()[0])


def _rcal_metrics(y, p):
    return dict(
        AUC=_rcal_auc(y, p),
        Brier=float(np.mean((p - y) ** 2)),
        ECE=_rcal_ece(y, p, ctx.rcal_n_bins, ctx.rcal_bin_strategy),
        CITL=_rcal_citl(y, p),
        Slope=_rcal_slope(y, p),
    )


def _rcal_brier_ci(y, p, n_boot, seed=42):
    rng = np.random.RandomState(seed)
    bb = [np.mean((p[i] - y[i]) ** 2) for i in (rng.randint(0, len(y), len(y)) for _ in range(n_boot))]
    return (np.percentile(bb, 2.5), np.percentile(bb, 97.5)) if len(bb) >= 20 else (np.nan, np.nan)


def _rcal_fit_apply(p_cal, y_cal, p_new, method):
    if method == "platt":
        lr = LogisticRegression(C=1e10, max_iter=5000)
        lr.fit(_rcal_logit(p_cal).reshape(-1, 1), y_cal)
        return lr.predict_proba(_rcal_logit(p_new).reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    return iso.predict(p_new)


def _rcal_crossfit(p, y, method, k, seed=42):
    out = np.zeros(len(p), dtype=float)
    for tr, va in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(p.reshape(-1, 1), y):
        out[va] = _rcal_fit_apply(p[tr], y[tr], p[va], method)
    return out


def _rcal_leave_fold_out(p_list, y_list, method):
    out = []
    for i in range(len(p_list)):
        pc = np.concatenate([p for j, p in enumerate(p_list) if j != i])
        yc = np.concatenate([y for j, y in enumerate(y_list) if j != i])
        out.append(_rcal_fit_apply(pc, yc, p_list[i], method))
    return np.concatenate(out)


def _rcal_recalibrate(key, p, y, p_cal=None, y_cal=None):
    """Platt- and isotonic-recalibrated held-out probabilities (or None, None if not possible)."""
    try:
        if ctx.rcal_heldout_calibrator == "nested_oof" and p_cal is not None:
            from scipy.stats import ks_2samp

            q = [50, 90, 99]
            ks = ks_2samp(p, p_cal).statistic
            logger.info(
                f"  [{key}] probability scale check: held-out p50/p90/p99 = "
                f"{'/'.join(f'{v:.3f}' for v in np.percentile(p, q))}; nested-CV OOF = "
                f"{'/'.join(f'{v:.3f}' for v in np.percentile(p_cal, q))}; KS = {ks:.3f}"
            )
            if ks > 0.2:
                logger.info(
                    "     Warning: the two sets of probabilities are on different scales; "
                    "rcal_heldout_calibrator = 'crossfit' is more appropriate for this target."
                )
            mode = "calibrators fit on training-partition nested-CV out-of-fold probabilities (Cell 12)"
            return _rcal_fit_apply(p_cal, y_cal, p, "platt"), _rcal_fit_apply(p_cal, y_cal, p, "isotonic"), mode
        if ctx.rcal_heldout_calibrator == "nested_oof":
            logger.info(f"  [{key}] no nested_cv out-of-fold probabilities in this session; using cross-fitting")
        mode = f"calibrators cross-fitted within the held-out set ({ctx.rcal_crossfit_folds}-fold)"
        return (
            _rcal_crossfit(p, y, "platt", ctx.rcal_crossfit_folds),
            _rcal_crossfit(p, y, "isotonic", ctx.rcal_crossfit_folds),
            mode,
        )
    except Exception as err:  # e.g. too few positives for the folds
        logger.info(f"  [{key}] recalibration not possible ({type(err).__name__}: {err}); reporting original only")
        return None, None, "no recalibration"


_RCAL_INK, _RCAL_INK2, _RCAL_MUTED, _RCAL_GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
_RCAL_COLORS = {"Original": "#52514e", "Platt": "#2a78d6", "Isotonic": "#eb6834"}


def _rcal_target_label(tgt):
    names = ctx.get("TARGET_DISPLAY_NAMES")
    return names.get(tgt, tgt) if isinstance(names, dict) else tgt


def _rcal_clean_axes(ax, grid_axis="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_RCAL_MUTED)
    ax.tick_params(colors=_RCAL_INK2, labelsize=8, length=3)
    ax.grid(False)
    ax.grid(axis=grid_axis, color=_RCAL_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


def _rcal_reliability(ax, y, p, label, color):
    """Observed vs mean predicted per bin; empty and sparse bins are not drawn. Returns the plotted extent."""
    b = _rcal_bins(p, ctx.rcal_n_bins, ctx.rcal_bin_strategy)
    xs, ys, lo, hi, ns = [], [], [], [], []
    for k in np.unique(b):
        m = b == k
        n, ph = int(m.sum()), y[m].mean()
        if n < ctx.rcal_min_bin_n:
            continue
        z = 1.96
        den = 1 + z**2 / n
        cen = (ph + z**2 / (2 * n)) / den
        mar = z * np.sqrt((ph * (1 - ph) + z**2 / (4 * n)) / n) / den
        xs.append(p[m].mean())
        ys.append(ph)
        lo.append(max(0, cen - mar))
        hi.append(min(1, cen + mar))
        ns.append(n)
    if not xs:
        return 0.1
    xs, ys, lo, hi = map(np.asarray, (xs, ys, lo, hi))
    ax.fill_between(xs, lo, hi, color=color, alpha=0.10, lw=0)
    ax.plot(xs, ys, "-", color=color, lw=1.8, label=label)
    ax.plot(xs, ys, "o", color=color, ms=4.5, mec="white", mew=0.8)
    return float(max(xs.max(), hi.max()))


def _rcal_panel(fig, gs_col, sets, title):
    """Reliability diagram (top) and histogram of original probabilities (bottom), shared x."""
    ax = fig.add_subplot(gs_col[0])
    axh = fig.add_subplot(gs_col[1], sharex=ax)
    ext = [
        _rcal_reliability(
            ax,
            yy,
            pp,
            f"{meth}  ECE {_rcal_ece(yy, pp, ctx.rcal_n_bins, ctx.rcal_bin_strategy):.3f}",
            _RCAL_COLORS[meth],
        )
        for _, meth, pp, yy in sets
    ]
    lim = float(min(1.0, max(0.1, np.ceil(max(ext) * 1.1 * 20) / 20)))  # zoom to the data for rare outcomes
    ax.plot([0, lim], [0, lim], ls=(0, (4, 3)), color=_RCAL_MUTED, lw=1, zorder=0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_ylabel("Observed proportion", fontsize=9, color=_RCAL_INK2)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=_RCAL_INK)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, labelcolor=_RCAL_INK2, handlelength=1.6)
    plt.setp(ax.get_xticklabels(), visible=False)
    _rcal_clean_axes(ax)
    p0 = sets[0][2]
    axh.hist(p0[p0 <= lim], bins=np.linspace(0, lim, 31), color="#c9c8c3", edgecolor="white", lw=0.5)
    axh.set_yscale("log")
    n_out = int((p0 > lim).sum())
    if n_out:
        axh.text(
            0.99,
            0.92,
            f"{n_out:,} above {lim:.2f} not shown",
            transform=axh.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=_RCAL_INK2,
        )
    axh.set_xlabel(
        f"Mean predicted probability ({ctx.rcal_bin_strategy} bins, n ≥ {ctx.rcal_min_bin_n} per bin)",
        fontsize=8.5,
        color=_RCAL_INK2,
    )
    axh.set_ylabel("Count", fontsize=8, color=_RCAL_INK2)
    _rcal_clean_axes(axh, grid_axis="y")


def _rcal_figure(tgt, tp, base_rate, n, sets, oof_sets, out_dir):
    from matplotlib.gridspec import GridSpec

    ncol = 2 if oof_sets else 1
    fig = plt.figure(figsize=(4.9 * ncol + 0.4, 5.6))
    gs = GridSpec(2, ncol, height_ratios=[3.2, 1], hspace=0.08, wspace=0.28, figure=fig)
    _rcal_panel(fig, [gs[0, 0], gs[1, 0]], sets, "Held-out test set")
    if oof_sets:
        _rcal_panel(fig, [gs[0, 1], gs[1, 1]], oof_sets, "Nested-CV out-of-fold (training partition)")
    fig.suptitle(
        f"{_rcal_target_label(tgt)}, T{tp}", x=0.06, ha="left", fontsize=11.5, fontweight="bold", color=_RCAL_INK
    )
    fig.text(
        0.06,
        0.925,
        f"n = {n:,}, base rate {base_rate:.3f}. Shading: 95% Wilson interval per bin. "
        "Histogram: original predicted probabilities.",
        fontsize=8,
        color=_RCAL_INK2,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.98)
    plt.savefig(os.path.join(out_dir, f"calibration_{tgt}_T{tp}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def _run_calibration_recalibration():
    out_dir = os.path.join(os.getcwd(), "calibration_recalibration")
    os.makedirs(out_dir, exist_ok=True)
    rows, reg_rows, reg_plot = [], [], []
    sel = _rcal_entries(ctx.rcal_targets)
    logger.info(f"Calibration & recalibration: {len(sel)} target(s) selected")
    colors = {"Original": "#455A64", "Platt": "#1565C0", "Isotonic": "#C62828"}

    for key, e in sel.items():
        tgt, tp = e.get("target", key), _rcal_tp(key, e)
        if getattr(e.get("last_fitted_model"), "task_type", None) == "regression" and not ctx.rcal_include_regression:
            continue
        prep = _rcal_prepare(key, e)
        if prep is None:
            continue
        _, _, _, y, is_clf, p = prep

        if not is_clf:  # regression
            if not ctx.rcal_include_regression:
                continue
            yh = p
            b1, b0 = np.polyfit(yh, y, 1)
            reg_rows.append(
                dict(
                    Target=tgt,
                    TP=tp,
                    N=len(y),
                    R2=1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2),
                    Mean_obs_minus_pred=float(np.mean(y - yh)),
                    Calibration_slope=b1,
                    Calibration_intercept=b0,
                )
            )
            reg_plot.append((f"{_rcal_target_label(tgt)}, T{tp}", y, yh))
            continue

        # Classification targets
        ofr = getattr(e.get("reporter_point1"), "outer_fold_results", None)
        has_oof = (
            isinstance(ofr, dict)
            and len(ofr.get("y_proba", [])) > 1
            and len(ofr.get("y_true", [])) == len(ofr.get("y_proba", []))
        )
        if has_oof:
            p_list = [np.asarray(v, float).ravel() for v in ofr["y_proba"]]
            y_list = [np.asarray(v, float).ravel() for v in ofr["y_true"]]
            p_cal, y_cal = np.concatenate(p_list), np.concatenate(y_list)
        p_platt, p_iso, mode = _rcal_recalibrate(key, p, y, p_cal if has_oof else None, y_cal if has_oof else None)

        n_empty = ctx.rcal_n_bins - len(np.unique(_rcal_bins(p, ctx.rcal_n_bins, "uniform")))
        logger.info(f"\n{'=' * 78}\n  {tgt} (T{tp})  N={len(y)}  base rate={y.mean():.3f}\n  {mode}\n{'=' * 78}")
        logger.info(
            f"  Held-out original: {n_empty} of {ctx.rcal_n_bins} equal-width bins empty; ECE as in nested_cv = "
            f"{_rcal_ece_cell12(y, p, ctx.rcal_n_bins):.4f}; count-weighted ECE = "
            f"{_rcal_ece(y, p, ctx.rcal_n_bins, ctx.rcal_bin_strategy):.4f} ({ctx.rcal_bin_strategy} bins)"
        )

        sets = [("Held-out", "Original", p, y)]
        if p_platt is not None:
            sets += [("Held-out", "Platt", p_platt, y), ("Held-out", "Isotonic", p_iso, y)]
        oof_sets = []
        if has_oof and ctx.rcal_include_nested_oof:
            oof_sets = [("Nested-CV OOF", "Original", p_cal, y_cal)]
            try:
                oof_sets += [
                    ("Nested-CV OOF", "Platt", _rcal_leave_fold_out(p_list, y_list, "platt"), y_cal),
                    ("Nested-CV OOF", "Isotonic", _rcal_leave_fold_out(p_list, y_list, "isotonic"), y_cal),
                ]
            except Exception as err:
                logger.info(f"  [{key}] nested-CV OOF recalibration not possible ({type(err).__name__}: {err})")
            n_empty_o = ctx.rcal_n_bins - len(np.unique(_rcal_bins(p_cal, ctx.rcal_n_bins, "uniform")))
            logger.info(
                f"  Nested-CV OOF original: {n_empty_o} of {ctx.rcal_n_bins} equal-width bins empty; ECE as in nested_cv = "
                f"{_rcal_ece_cell12(y_cal, p_cal, ctx.rcal_n_bins):.4f}; count-weighted ECE = "
                f"{_rcal_ece(y_cal, p_cal, ctx.rcal_n_bins, ctx.rcal_bin_strategy):.4f} ({ctx.rcal_bin_strategy} bins)"
            )

        for set_name, meth, pp, yy in sets + oof_sets:
            blo, bhi = _rcal_brier_ci(yy, pp, ctx.rcal_n_bootstrap)
            rows.append(
                dict(
                    Target=tgt,
                    TP=tp,
                    Set=set_name,
                    Method=meth,
                    N=len(yy),
                    Base_rate=yy.mean(),
                    **_rcal_metrics(yy, pp),
                    Brier_lo=blo,
                    Brier_hi=bhi,
                    Calibrator=mode if set_name == "Held-out" else "leave-one-outer-fold-out",
                )
            )

        t = pd.DataFrame([r for r in rows if r["Target"] == tgt and r["TP"] == tp])
        t["Brier [95% CI]"] = [
            f"{a:.4f} [{b:.4f}, {c:.4f}]" for a, b, c in zip(t["Brier"], t["Brier_lo"], t["Brier_hi"])
        ]
        display(
            t[["Set", "Method", "N", "AUC", "Brier [95% CI]", "ECE", "CITL", "Slope"]]
            .style.format({"AUC": "{:.3f}", "ECE": "{:.4f}", "CITL": "{:+.3f}", "Slope": "{:.3f}"})
            .hide(axis="index")
        )

        if ctx.rcal_make_plots:
            _rcal_figure(tgt, tp, y.mean(), len(y), sets, oof_sets, out_dir)

    clf_df, reg_df = pd.DataFrame(rows), pd.DataFrame(reg_rows)
    if len(clf_df):
        clf_df.to_csv(os.path.join(out_dir, "calibration_classification.csv"), index=False)
        w = clf_df.pivot_table(
            index=["Target", "TP", "Set"], columns="Method", values=["Brier", "ECE"], aggfunc="first"
        )
        w = w.reindex(
            columns=[(a, b) for a in ["Brier", "ECE"] for b in ["Original", "Platt", "Isotonic"] if (a, b) in w.columns]
        )
        w.columns = [f"{a} {b}" for a, b in w.columns]
        logger.info(f"\n{'=' * 78}\n  Summary (classification): Brier and ECE by method\n{'=' * 78}")
        display(w.reset_index().style.format({c: "{:.4f}" for c in w.columns}).hide(axis="index"))
        logger.info("  CITL = calibration-in-the-large (ideal 0); Slope = calibration slope (ideal 1).")
    if len(reg_df):
        reg_df.to_csv(os.path.join(out_dir, "calibration_regression.csv"), index=False)
        logger.info(
            f"\n{'=' * 78}\n  Regression targets: held-out calibration (observed regressed on predicted)\n{'=' * 78}"
        )
        display(
            reg_df.style.format(
                {
                    "R2": "{:.3f}",
                    "Mean_obs_minus_pred": "{:+.3f}",
                    "Calibration_slope": "{:.3f}",
                    "Calibration_intercept": "{:+.3f}",
                }
            ).hide(axis="index")
        )
        logger.info("  Ideal: calibration slope 1 and mean(observed - predicted) 0.")
        if ctx.rcal_make_plots:
            nc = min(4, len(reg_plot))
            nr = int(np.ceil(len(reg_plot) / nc))
            fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc + 0.4, 3.2 * nr + 0.6), squeeze=False)
            for ax, (lbl, yy, hh) in zip(axes.ravel(), reg_plot):
                d = pd.DataFrame({"q": pd.qcut(hh, 10, labels=False, duplicates="drop"), "y": yy, "h": hh})
                g = d.groupby("q").agg(h=("h", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
                lo, hi = min(g["h"].min(), g["y"].min()), max(g["h"].max(), g["y"].max())
                pad = 0.05 * (hi - lo)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls=(0, (4, 3)), color=_RCAL_MUTED, lw=1)
                se = 1.96 * g["sd"] / np.sqrt(g["n"])
                ax.fill_between(g["h"], g["y"] - se, g["y"] + se, color="#2a78d6", alpha=0.12, lw=0)
                ax.plot(g["h"], g["y"], "-o", color="#2a78d6", lw=1.8, ms=4.5, mec="white", mew=0.8)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_title(lbl, loc="left", fontsize=9.5, fontweight="bold", color=_RCAL_INK)
                ax.set_xlabel("Mean predicted", fontsize=8.5, color=_RCAL_INK2)
                ax.set_ylabel("Mean observed", fontsize=8.5, color=_RCAL_INK2)
                _rcal_clean_axes(ax)
            for ax in axes.ravel()[len(reg_plot) :]:
                ax.axis("off")
            fig.suptitle(
                "Regression targets: observed vs predicted by decile of prediction (held-out, 95% CI)",
                x=0.02,
                ha="left",
                fontsize=11,
                fontweight="bold",
                color=_RCAL_INK,
            )
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, "calibration_regression_deciles.png"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.show()
    logger.info(f"\nSaved to {out_dir}/")
    return clf_df, reg_df


def run() -> None:
    """Run this stage (notebook cell 22)."""

    ctx.calibration_classification_df, ctx.calibration_regression_df = _run_calibration_recalibration()
