"""Extended demographic-stratified performance: sex, SES, ALE and parent education (the `stratified` stage, Supplement S1.17, S1.11.7).

Per-subgroup performance evaluation stratified by child sex, SES tertile,
ALE exposure, parent sex, and parent education. Produces per-subgroup
performance tables with bootstrap 95% CIs, ROC curve overlays,
calibration curves, forest plots, and formal fairness gap analysis
quantifying performance differentials across strata.
"""

import logging

from bpsp.context import ctx
from bpsp.data.preprocessing import DISPLAY_NAMES, TARGET_DISPLAY_NAMES, detect_task_type
from bpsp.models.stacking import create_stacking_ensemble
from bpsp.reporting import (
    _RS,
    _pipeline_cleanup,
    cache_load,
    cache_save,
    fold_load_all,
    make_run_key,
    report_save_state,
)

logger = logging.getLogger(__name__)


"""
Stratified Performance & Fairness Analysis
------------------------------------------
Per-subgroup evaluation of predictive performance to assess fairness
and generalisability regarding differential
performance across demographic strata (sex, SES tertile, ALE exposure).

Stratification variables
------------------------
  sex             : Male (1) / Female (2)
  SES             : Tertile-based Low / Mid / High
  high_ALE / low_ALE : Adverse life-event exposure groups
  sex_P           : Parent sex (1 / 2)
  parent_education: Low / Mid / High (tertiles)

Outputs
  Per-subgroup performance tables with bootstrap 95% CIs,
  ROC curve overlay, calibration curves, forest plots,
  and formal fairness gap analysis.
"""

import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold

# Style constants
_SLATE = "#1F2D3D"
_CHARCOAL = "#34495E"
_STEEL = "#5D6D7E"
_SILVER = "#BDC3C7"
_SNOW = "#FAFBFC"

_SUBGROUP_COLORS = {
    "Overall": "#455A64",
    "Sex: Male": "#1565C0",
    "Sex: Female": "#E91E63",
    "SES: Low": "#E65100",
    "SES: Mid": "#F9A825",
    "SES: High": "#2E7D32",
    "ALE: Low ALE": "#7B1FA2",
    "ALE: Not Low ALE": "#CE93D8",
    "ALE: High ALE": "#4A148C",
    "ALE: Not High ALE": "#BA68C8",
    "ParSex: Parent Male": "#00838F",
    "ParSex: Parent Female": "#4DD0E1",
    "Edu: Low Edu": "#795548",
    "Edu: Mid Edu": "#A1887F",
    "Edu: High Edu": "#3E2723",
}


def _sg_color(label):
    if label in _SUBGROUP_COLORS:
        return _SUBGROUP_COLORS[label]
    for prefix, col in [
        ("Sex:", "#1565C0"),
        ("SES:", "#E65100"),
        ("ALE:", "#7B1FA2"),
        ("ParSex:", "#00838F"),
        ("Edu:", "#2E7D32"),
    ]:
        if label.startswith(prefix):
            return col
    return "#78909C"


# Classification metrics with bootstrap CIs


def _compute_subgroup_metrics_classification(y_true, y_proba, label, threshold=0.5, n_boot=None):
    """
    Classification metrics with bootstrap 95% CIs.
    Returns both display strings and raw values for plotting.
    """
    if n_boot is None:
        n_boot = ctx._VAL_N_BOOTSTRAP
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    n = len(y_true)

    # Point estimates
    y_pred = (y_proba >= threshold).astype(int)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    base_rate = n_pos / n if n > 0 else 0

    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    auc_pr = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan

    _cm = confusion_matrix(y_true, y_pred)
    if _cm.shape == (2, 2):
        tn, fp, fn, tp = _cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
    acc = accuracy_score(y_true, y_pred)

    # Optimal threshold (F1-maximizing on PR curve)
    try:
        _pr, _rc, _thr = precision_recall_curve(y_true, y_proba)
        _f1c = 2 * (_pr[:-1] * _rc[:-1]) / (_pr[:-1] + _rc[:-1] + 1e-10)
        _best_idx = np.argmax(_f1c)
        opt_thresh = float(_thr[_best_idx])
        y_pred_opt = (y_proba >= opt_thresh).astype(int)
        _cm_opt = confusion_matrix(y_true, y_pred_opt)
        if _cm_opt.shape == (2, 2):
            tn_o, fp_o, fn_o, tp_o = _cm_opt.ravel()
        else:
            tn_o, fp_o, fn_o, tp_o = 0, 0, 0, 0
        sens_opt = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0
        ppv_opt = tp_o / (tp_o + fp_o) if (tp_o + fp_o) > 0 else 0
        f1_opt = 2 * ppv_opt * sens_opt / (ppv_opt + sens_opt) if (ppv_opt + sens_opt) > 0 else 0
    except Exception:
        opt_thresh, sens_opt, ppv_opt, f1_opt = 0.5, sens, ppv, f1

    # Bootstrap CIs
    rng = np.random.RandomState(42)
    boot = {"auc": [], "auc_pr": [], "sens": [], "spec": [], "ppv": [], "npv": [], "f1": [], "brier": []}
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb_t = y_true[idx]
        yb_p = y_proba[idx]
        if len(np.unique(yb_t)) < 2:
            continue
        yb_pred = (yb_p >= threshold).astype(int)
        _cmb = confusion_matrix(yb_t, yb_pred)
        if _cmb.shape != (2, 2):
            continue
        tn_b, fp_b, fn_b, tp_b = _cmb.ravel()
        boot["auc"].append(roc_auc_score(yb_t, yb_p))
        boot["auc_pr"].append(average_precision_score(yb_t, yb_p))
        boot["brier"].append(brier_score_loss(yb_t, yb_p))
        boot["sens"].append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)
        boot["spec"].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
        boot["ppv"].append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0)
        boot["npv"].append(tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0)
        _s = boot["sens"][-1]
        _p = boot["ppv"][-1]
        boot["f1"].append(2 * _p * _s / (_p + _s) if (_p + _s) > 0 else 0)

    def _ci(vals):
        if len(vals) < 10:
            return (np.nan, np.nan)
        return (np.percentile(vals, 2.5), np.percentile(vals, 97.5))

    ci = {k: _ci(v) for k, v in boot.items()}

    return {
        "Subgroup": label,
        "N": n,
        "Pos": n_pos,
        "Neg": n_neg,
        "Base Rate": base_rate,
        # Raw values for plotting
        "auc_val": auc,
        "auc_lo": ci["auc"][0],
        "auc_hi": ci["auc"][1],
        "auc_pr_val": auc_pr,
        "auc_pr_lo": ci["auc_pr"][0],
        "auc_pr_hi": ci["auc_pr"][1],
        "sens_val": sens,
        "sens_lo": ci["sens"][0],
        "sens_hi": ci["sens"][1],
        "spec_val": spec,
        "spec_lo": ci["spec"][0],
        "spec_hi": ci["spec"][1],
        "ppv_val": ppv,
        "ppv_lo": ci["ppv"][0],
        "ppv_hi": ci["ppv"][1],
        "npv_val": npv,
        "npv_lo": ci["npv"][0],
        "npv_hi": ci["npv"][1],
        "f1_val": f1,
        "f1_lo": ci["f1"][0],
        "f1_hi": ci["f1"][1],
        "brier_val": brier,
        "brier_lo": ci["brier"][0],
        "brier_hi": ci["brier"][1],
        # Optimal threshold
        "opt_thresh": opt_thresh,
        "sens_opt": sens_opt,
        "ppv_opt": ppv_opt,
        "f1_opt": f1_opt,
        # For ROC/PR curves
        "_y_true": y_true,
        "_y_proba": y_proba,
    }


# Regression metrics with bootstrap CIs


def _compute_subgroup_metrics_regression(y_true, y_pred, label, n_boot=None):
    """Regression metrics with bootstrap 95% CIs."""
    if n_boot is None:
        n_boot = ctx._VAL_N_BOOTSTRAP
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rng = np.random.RandomState(42)
    boot_r2, boot_rmse, boot_mae = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        if np.var(y_t) > 0:
            boot_r2.append(r2_score(y_t, y_p))
            boot_rmse.append(np.sqrt(mean_squared_error(y_t, y_p)))
            boot_mae.append(mean_absolute_error(y_t, y_p))
    r2_lo = np.percentile(boot_r2, 2.5) if boot_r2 else np.nan
    r2_hi = np.percentile(boot_r2, 97.5) if boot_r2 else np.nan
    rmse_lo = np.percentile(boot_rmse, 2.5) if boot_rmse else np.nan
    rmse_hi = np.percentile(boot_rmse, 97.5) if boot_rmse else np.nan
    mae_lo = np.percentile(boot_mae, 2.5) if boot_mae else np.nan
    mae_hi = np.percentile(boot_mae, 97.5) if boot_mae else np.nan
    return {
        "Subgroup": label,
        "N": len(y_true),
        "R²": f"{r2:.3f} [{r2_lo:.3f}-{r2_hi:.3f}]",
        "RMSE": f"{rmse:.3f} [{rmse_lo:.3f}-{rmse_hi:.3f}]",
        "MAE": f"{mae:.3f} [{mae_lo:.3f}-{mae_hi:.3f}]",
        "r2_val": r2,
        "r2_lo": r2_lo,
        "r2_hi": r2_hi,
        "rmse_val": rmse,
        "rmse_lo": rmse_lo,
        "rmse_hi": rmse_hi,
        "mae_val": mae,
        "mae_lo": mae_lo,
        "mae_hi": mae_hi,
        "_residuals": y_true - y_pred,
    }


# VISUALIZATION: Classification, 4-panel figure


def _plot_classification_stratified(all_rows, results_dir, target_lbl, tp_lbl):
    """
    Publication-quality 4-panel figure for classification stratification:
      Panel A: Forest plot: AUC-ROC with bootstrap 95% CIs
      Panel B: Forest plot: AUC-PR with bootstrap 95% CIs
      Panel C: ROC curves overlaid per subgroup
      Panel D: Calibration curves per subgroup
    """
    # Filter to rows with valid AUC
    rows = [r for r in all_rows if not np.isnan(r.get("auc_val", np.nan))]
    if len(rows) < 2:
        logger.info("  Warning: Insufficient subgroups for classification plots.")
        return

    n_sg = len(rows)
    fig = plt.figure(figsize=(22, max(8, n_sg * 0.55 + 3)))
    fig.patch.set_facecolor("white")

    # GridSpec: top row = forest plots, bottom row = curves
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30, height_ratios=[max(1.0, n_sg * 0.12), 1.0])
    ax_forest_roc = fig.add_subplot(gs[0, 0])
    ax_forest_pr = fig.add_subplot(gs[0, 1])
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_cal = fig.add_subplot(gs[1, 1])

    # Panel A: AUC-ROC Forest Plot
    plot_rows = list(reversed(rows))
    labels = [r["Subgroup"] for r in plot_rows]
    y_pos = np.arange(len(labels))

    overall_auc = next((r["auc_val"] for r in rows if r["Subgroup"] == "Overall"), None)

    for i, r in enumerate(plot_rows):
        col = _sg_color(r["Subgroup"])
        val, lo, hi = r["auc_val"], r["auc_lo"], r["auc_hi"]
        ax_forest_roc.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax_forest_roc.plot(val, i, "o", color=col, ms=9, mec="white", mew=1.5, zorder=5)
        ax_forest_roc.text(
            hi + 0.008, i, f"{val:.3f} [{lo:.3f}-{hi:.3f}]", va="center", fontsize=8, fontweight="bold", color=col
        )
        # N annotation on left
        ax_forest_roc.text(lo - 0.008, i, f"n={r['N']:,}", va="center", ha="right", fontsize=9.5, color=_STEEL)

    if overall_auc is not None:
        ax_forest_roc.axvline(overall_auc, color=_SILVER, ls=":", lw=1, alpha=0.6)

    ax_forest_roc.set_yticks(y_pos)
    ax_forest_roc.set_yticklabels(labels, fontsize=9)
    ax_forest_roc.set_xlabel("AUC-ROC (bootstrap 95% CI)", fontsize=10)
    ax_forest_roc.set_title("A.  AUC-ROC by Subgroup", fontsize=12, fontweight="bold", color=_SLATE, loc="left")
    ax_forest_roc.grid(axis="x", alpha=0.15, ls="--")

    # Panel B: AUC-PR Forest Plot
    overall_pr = next((r["auc_pr_val"] for r in rows if r["Subgroup"] == "Overall"), None)

    for i, r in enumerate(plot_rows):
        col = _sg_color(r["Subgroup"])
        val, lo, hi = r["auc_pr_val"], r["auc_pr_lo"], r["auc_pr_hi"]
        ax_forest_pr.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax_forest_pr.plot(val, i, "s", color=col, ms=8, mec="white", mew=1.5, zorder=5)
        ax_forest_pr.text(
            hi + 0.008, i, f"{val:.3f} [{lo:.3f}-{hi:.3f}]", va="center", fontsize=8, fontweight="bold", color=col
        )

    # Base rate reference line
    base_rate = next((r["Base Rate"] for r in rows if r["Subgroup"] == "Overall"), None)
    if base_rate is not None:
        ax_forest_pr.axvline(base_rate, color="#C62828", ls="--", lw=1, alpha=0.5)
        ax_forest_pr.text(
            base_rate + 0.003,
            len(plot_rows) - 0.3,
            f"chance ({base_rate:.3f})",
            fontsize=9.5,
            color="#C62828",
            style="italic",
        )
    if overall_pr is not None:
        ax_forest_pr.axvline(overall_pr, color=_SILVER, ls=":", lw=1, alpha=0.6)

    ax_forest_pr.set_yticks(y_pos)
    ax_forest_pr.set_yticklabels(labels, fontsize=9)
    ax_forest_pr.set_xlabel("AUC-PR (bootstrap 95% CI)", fontsize=10)
    ax_forest_pr.set_title("B.  AUC-PR by Subgroup", fontsize=12, fontweight="bold", color=_SLATE, loc="left")
    ax_forest_pr.grid(axis="x", alpha=0.15, ls="--")

    # Panel C: ROC Curves Overlay
    for r in rows:
        col = _sg_color(r["Subgroup"])
        lw = 2.5 if r["Subgroup"] == "Overall" else 1.8
        ls = "-" if r["Subgroup"] == "Overall" else "-"
        alpha = 1.0 if r["Subgroup"] == "Overall" else 0.75
        fpr, tpr, _ = roc_curve(r["_y_true"], r["_y_proba"])
        ax_roc.plot(fpr, tpr, color=col, lw=lw, ls=ls, alpha=alpha, label=f"{r['Subgroup']}  (AUC={r['auc_val']:.3f})")

    ax_roc.plot([0, 1], [0, 1], "--", color=_SILVER, lw=1)
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.set_title("C.  ROC Curves by Subgroup", fontsize=12, fontweight="bold", color=_SLATE, loc="left")
    ax_roc.legend(fontsize=8, loc="lower right", framealpha=0.95, edgecolor=_SILVER)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.grid(alpha=0.12, ls="--")

    # Panel D: Calibration Curves
    for r in rows:
        col = _sg_color(r["Subgroup"])
        lw = 2.5 if r["Subgroup"] == "Overall" else 1.8
        alpha = 1.0 if r["Subgroup"] == "Overall" else 0.75
        try:
            prob_true, prob_pred = calibration_curve(r["_y_true"], r["_y_proba"], n_bins=8, strategy="quantile")
            ax_cal.plot(
                prob_pred,
                prob_true,
                "o-",
                color=col,
                lw=lw,
                alpha=alpha,
                ms=5,
                mec="white",
                mew=0.8,
                label=f"{r['Subgroup']}  (Brier={r['brier_val']:.3f})",
            )
        except Exception:
            pass

    ax_cal.plot([0, 1], [0, 1], "--", color=_SILVER, lw=1, label="Perfect")
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax_cal.set_ylabel("Fraction of Positives", fontsize=10)
    ax_cal.set_title("D.  Calibration by Subgroup", fontsize=12, fontweight="bold", color=_SLATE, loc="left")
    ax_cal.legend(fontsize=8, loc="lower right", framealpha=0.95, edgecolor=_SILVER)
    ax_cal.set_xlim([-0.02, 1.02])
    ax_cal.set_ylim([-0.02, 1.02])
    ax_cal.grid(alpha=0.12, ls="--")

    fig.suptitle(
        f"Stratified Classification Performance  ·  {target_lbl}  (T{tp_lbl})",
        fontsize=14,
        fontweight="bold",
        color=_SLATE,
        y=1.01,
    )

    out_path = f"{results_dir}/stratified_classification_4panel.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    logger.info(f"   Saved: {out_path}")


# VISUALIZATION: Classification, Threshold-optimized comparison


def _plot_threshold_comparison(all_rows, results_dir, target_lbl, tp_lbl):
    """
    Bar chart comparing default (0.50) vs optimal threshold metrics
    per subgroup: Recall(1), Precision(1), F1(1).
    """
    rows = [r for r in all_rows if not np.isnan(r.get("auc_val", np.nan)) and r["Subgroup"] != "Overall"]
    if len(rows) < 1:
        return

    labels = [r["Subgroup"] for r in rows]
    n = len(labels)
    x = np.arange(n)
    w = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(min(20, 6 + n * 1.2), 5))
    fig.patch.set_facecolor("white")

    metrics = [
        ("Recall (Class 1)", "sens_val", "sens_opt"),
        ("Precision (Class 1)", "ppv_val", "ppv_opt"),
        ("F1 (Class 1)", "f1_val", "f1_opt"),
    ]

    for ax, (title, key_def, key_opt) in zip(axes, metrics):
        vals_def = [r[key_def] for r in rows]
        vals_opt = [r[key_opt] for r in rows]

        bars1 = ax.bar(x - w / 2, vals_def, w, label="Default (0.50)", color="#90A4AE", edgecolor="white", lw=0.5)
        bars2 = ax.bar(x + w / 2, vals_opt, w, label="Optimised", color="#1565C0", edgecolor="white", lw=0.5)

        for b in bars1:
            h = b.get_height()
            if h > 0.01:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", fontsize=9.5, color="#546E7A")
        for b in bars2:
            h = b.get_height()
            if h > 0.01:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    fontsize=9.5,
                    color="#1565C0",
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_title(title, fontsize=11, fontweight="bold", color=_SLATE)
        ax.set_ylim(0, max(max(vals_def + vals_opt) * 1.25, 0.1))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.15, ls="--")

    fig.suptitle(
        f"Default vs Optimised Threshold  ·  {target_lbl}  (T{tp_lbl})",
        fontsize=13,
        fontweight="bold",
        color=_SLATE,
        y=1.02,
    )
    plt.tight_layout()
    out_path = f"{results_dir}/stratified_threshold_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    logger.info(f"   Saved: {out_path}")


# TABLE: Styled HTML table for classification


def _display_classification_table(all_rows, target_lbl, tp_lbl):
    """Display a publication-quality styled table with all classification metrics."""
    rows_display = []
    for r in all_rows:

        def _fmt(val, lo, hi):
            if np.isnan(val):
                return "--"
            return f"{val:.3f} [{lo:.3f}-{hi:.3f}]"

        rows_display.append(
            {
                "Subgroup": r["Subgroup"],
                "N": r["N"],
                "Pos / Neg": f"{r['Pos']} / {r['Neg']}",
                "Base Rate": f"{r['Base Rate']:.3f}",
                "AUC-ROC": _fmt(r["auc_val"], r["auc_lo"], r["auc_hi"]),
                "AUC-PR": _fmt(r["auc_pr_val"], r["auc_pr_lo"], r["auc_pr_hi"]),
                "Brier": f"{r['brier_val']:.4f}" if not np.isnan(r["brier_val"]) else "--",
                "Sens (0.50)": f"{r['sens_val']:.3f}",
                "Spec (0.50)": f"{r['spec_val']:.3f}",
                "PPV (0.50)": f"{r['ppv_val']:.3f}",
                "F1 (0.50)": f"{r['f1_val']:.3f}",
                "Opt Thresh": f"{r['opt_thresh']:.3f}",
                "Sens (opt)": f"{r['sens_opt']:.3f}",
                "PPV (opt)": f"{r['ppv_opt']:.3f}",
                "F1 (opt)": f"{r['f1_opt']:.3f}",
            }
        )

    df = pd.DataFrame(rows_display)
    # Numeric columns for gradient
    _num_cols = [
        "AUC-ROC",
        "AUC-PR",
        "Brier",
        "Sens (0.50)",
        "Spec (0.50)",
        "PPV (0.50)",
        "F1 (0.50)",
        "Sens (opt)",
        "PPV (opt)",
        "F1 (opt)",
    ]

    styler = (
        df.style.set_caption(f"Stratified Classification Performance  ·  {target_lbl}  (T{tp_lbl})")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "14px"),
                        ("font-weight", "700"),
                        ("color", _SLATE),
                        ("padding-bottom", "10px"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#ECEFF1"),
                        ("color", _CHARCOAL),
                        ("font-size", "9.5px"),
                        ("font-weight", "600"),
                        ("text-transform", "uppercase"),
                        ("padding", "7px 8px"),
                    ],
                },
                {"selector": "td", "props": [("padding", "5px 8px"), ("font-size", "10.5px")]},
                {"selector": "tr:first-child td", "props": [("background-color", "#E8EAF6"), ("font-weight", "600")]},
            ]
        )
        .hide(axis="index")
    )
    display(styler)
    return df


# Fairness summary


def _print_fairness_summary(all_rows):
    """Print formal fairness gap analysis."""
    subgroup_rows = [r for r in all_rows if r["Subgroup"] != "Overall"]
    if len(subgroup_rows) < 2:
        return

    logger.info("\n" + "=" * 72)
    logger.info("  FAIRNESS GAP ANALYSIS")
    logger.info("=" * 72)

    # Group subgroups by demographic dimension
    dims = {}
    for r in subgroup_rows:
        prefix = r["Subgroup"].split(":")[0].strip()
        dims.setdefault(prefix, []).append(r)

    for dim, dim_rows in dims.items():
        if len(dim_rows) < 2:
            continue
        aucs = [r["auc_val"] for r in dim_rows if not np.isnan(r["auc_val"])]
        auc_prs = [r["auc_pr_val"] for r in dim_rows if not np.isnan(r["auc_pr_val"])]
        senses = [r["sens_val"] for r in dim_rows]
        f1_opts = [r["f1_opt"] for r in dim_rows]

        logger.info(f"\n  {dim}")
        logger.info(f"  {'-' * 60}")

        if len(aucs) >= 2:
            gap_auc = max(aucs) - min(aucs)
            best = [r["Subgroup"] for r in dim_rows if r["auc_val"] == max(aucs)][0]
            worst = [r["Subgroup"] for r in dim_rows if r["auc_val"] == min(aucs)][0]
            flag = "  WARN" if gap_auc > 0.05 else "  ok"
            logger.info(f"    AUC-ROC gap:  {gap_auc:.4f}  (best: {best}, worst: {worst}){flag}")

        if len(auc_prs) >= 2:
            gap_pr = max(auc_prs) - min(auc_prs)
            flag = "  WARN" if gap_pr > 0.05 else "  ok"
            logger.info(f"    AUC-PR gap:   {gap_pr:.4f}{flag}")

        if len(senses) >= 2:
            gap_sens = max(senses) - min(senses)
            flag = "  WARN" if gap_sens > 0.10 else "  ok"
            logger.info(f"    Sens gap:     {gap_sens:.4f}  (equalized opportunity){flag}")

        if len(f1_opts) >= 2:
            gap_f1 = max(f1_opts) - min(f1_opts)
            flag = "  WARN" if gap_f1 > 0.10 else "  ok"
            logger.info(f"    F1(opt) gap:  {gap_f1:.4f}{flag}")

    logger.info("\n  Thresholds: AUC/AUC-PR gap > 0.05 WARN, Sens/F1 gap > 0.10 WARN")


# VISUALIZATION: Regression (preserved from original)


def _plot_regression_stratified(all_rows, results_dir, target_lbl, tp_lbl):
    """Forest plot + residual boxplot for regression."""
    rows = [r for r in all_rows if not np.isnan(r.get("r2_val", np.nan))]
    if len(rows) < 2:
        return

    n_sg = len(rows)
    fig_height = max(6, 0.55 * n_sg + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_height))

    plot_rows = list(reversed(rows))
    labels = [r["Subgroup"] for r in plot_rows]
    y_pos = np.arange(len(labels))

    overall_r2 = next((r["r2_val"] for r in rows if r["Subgroup"] == "Overall"), None)

    for i, r in enumerate(plot_rows):
        col = _sg_color(r["Subgroup"])
        val, lo, hi = r["r2_val"], r["r2_lo"], r["r2_hi"]
        axes[0].plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        axes[0].plot(val, i, "o", color=col, ms=9, mec="white", mew=1.5, zorder=5)
        axes[0].text(hi + 0.003, i, f"{val:.3f}", va="center", fontsize=9, fontweight="bold", color=col)

    if overall_r2 is not None:
        axes[0].axvline(overall_r2, color=_SILVER, ls=":", lw=1, alpha=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("R² (bootstrap 95% CI)", fontsize=11)
    axes[0].set_title("R² by Demographic Subgroup", fontsize=12, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.2)

    # Residual boxplot
    _box_data = [r["_residuals"] for r in plot_rows if "_residuals" in r]
    _box_labels = [r["Subgroup"] for r in plot_rows if "_residuals" in r]
    if _box_data:
        bp = axes[1].boxplot(
            _box_data,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#E3F2FD", edgecolor="#1565C0"),
            medianprops=dict(color="#C62828", linewidth=2),
            whiskerprops=dict(color="#455A64"),
            capprops=dict(color="#455A64"),
            flierprops=dict(markerfacecolor="#90A4AE", markersize=3),
        )
        axes[1].set_yticks(range(1, len(_box_labels) + 1), _box_labels)
        axes[1].axvline(0, color="#C62828", ls="--", lw=1.5, alpha=0.5)
        axes[1].set_xlabel("Residual (Actual − Predicted)", fontsize=11)
        axes[1].set_title("Residual Distribution by Subgroup", fontsize=12, fontweight="bold")

    plt.suptitle(f"Stratified Holdout Performance  ·  {target_lbl}  (T{tp_lbl})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = f"{results_dir}/stratified_forest_residual.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"   Saved: {out_path}")


# Create Validation Model


def create_validation_model_p2(is_classification=True):
    """
    Creates the same ensemble model used in main pipeline.
    Matches run_catboost_analysis() with stacking ensemble.
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression

    best_params = dict(ctx._CB_PARAMS)

    model_configs = {
        "catboost": best_params,
        "random_forest": dict(ctx._RF_PARAMS),
        "xgboost": dict(ctx._XGB_PARAMS),
        "tabpfn": dict(ctx.PIPELINE_MODEL_CONFIGS["tabpfn"]),
    }
    # Respect _SKIP_TABPFN gate (across-category / manual skip)
    if ctx.get("_SKIP_TABPFN", False):
        model_configs.pop("tabpfn", None)

    if is_classification:
        meta_model = LogisticRegression(random_state=42)
    else:
        meta_model = LinearRegression()

    model = create_stacking_ensemble(
        task_type="regression" if not is_classification else "classification",
        model_configs=model_configs,
        meta_model=meta_model,
        cv=5,
        random_state=42,
    )

    return model


# Cache system: stratified

# Batch-aware wrapper: Extended Stratified Performance
# When _batch_archive holds multiple targets with archived state, this
# wrapper iterates over each, restoring globals before each pass.
# In single-target mode the validation runs exactly once (no change).


def _run_for_current_target():
    """Run this stage's validation for the current target."""
    # Validation stage gate (reads the main stage settings)
    _skip_demog = not ctx.get("run_validation_cells", True)

    # Dependency guard
    if "make_run_key" not in ctx:
        logger.info("\nWarning: setup has not been run in this session.")
        logger.info("   Run setup and the main stage first.")
        _RUN_13 = False
    elif _skip_demog:
        logger.info("stratified skipped: main.run_validation_cells is False")
        _RUN_13 = False
    else:
        _CTAG_13 = "stratified"
        _CKEY_13 = make_run_key({"cell": _CTAG_13})
        _CACHED_13 = cache_load(_CTAG_13, _CKEY_13)
        if _CACHED_13:
            logger.info(f"  {_CTAG_13} loaded from cache -- skipping execution.")
            _RUN_13 = False
        else:
            _RS["section"] = "stratified"
            _done_folds_13 = fold_load_all(_CKEY_13, _CTAG_13)
            _RUN_13 = True

    if _RUN_13:
        try:
            # Task-type helpers
            if "is_regression" not in ctx or not callable(ctx.get("is_regression")):
                ctx["is_regression"] = lambda target: target in (ctx.get("continuous_outcome_variables", []))
            if "is_classification" not in ctx or not callable(ctx.get("is_classification")):
                ctx["is_classification"] = lambda target: (
                    target in (ctx.get("binary_or_categorical_outcome_variables", []))
                )

            # Mode detection
            _is_within_mode = (
                "exported_within_dict" in ctx
                and isinstance(ctx.exported_within_dict, dict)
                and len(ctx.exported_within_dict) > 0
                and "split" in ctx
                and ctx.split == "within_categories"
            )

            # Top-K categories by performance (from metrics_dict_full)
            _top_cats_ordered = []
            if _is_within_mode and "metrics_dict_full" in ctx:
                _cat_scores = {}
                for _cat, _mdict in zip(
                    ctx.metrics_dict_full.get("category", []), ctx.metrics_dict_full.get("metrics", [])
                ):
                    _m = _mdict.get("ensemble", _mdict.get("meta_model", {}))
                    if _m:
                        _s = _m.get("r2", _m.get("auc", _m.get("accuracy", np.nan)))
                        if not np.isnan(_s):
                            _cat_scores[_cat] = _s
                _top_cats_ordered = sorted(_cat_scores.keys(), key=lambda c: _cat_scores[c], reverse=True)

            _has_demo = "sample_valid_test" in ctx and ctx.sample_valid_test is not None

            if _is_within_mode and _has_demo:
                # Within-categories mode (preserved from original)

                _TOP_K = min(5, len(_top_cats_ordered))
                _cats_to_run = _top_cats_ordered[:_TOP_K]

                logger.info("\n" + "=" * 80)
                logger.info(f" Running Stratified Holdout (Within-Categories, Top {_TOP_K})")
                logger.info("=" * 80)

                task_type = detect_task_type(list(ctx.exported_within_dict.values())[0][2])
                is_classification = task_type == "classification"
                logger.info(f"   Task type: {task_type}")

                _sv_test = ctx.sample_valid_test

                _demo_labels = {}
                if "sex" in _sv_test.columns:
                    _demo_labels["Sex"] = _sv_test["sex"].map({1: "Male", 2: "Female"}).reset_index(drop=True)
                if "parent_income" in _sv_test.columns:
                    _demo_labels["SES"] = pd.cut(
                        _sv_test["parent_income"],
                        bins=[0, 4, 7, 10],
                        labels=["Low", "Mid", "High"],
                        include_lowest=True,
                    ).reset_index(drop=True)
                if "high_ale" in _sv_test.columns:
                    _demo_labels["ALE"] = (
                        _sv_test["high_ale"]
                        .astype(bool)
                        .map({True: "High ALE", False: "Not High ALE"})
                        .reset_index(drop=True)
                    )

                logger.info(f"   Demographics available: {list(_demo_labels.keys())}")

                results_dir = os.path.join(os.getcwd(), "nestedcv_results_stratified")
                os.makedirs(results_dir, exist_ok=True)

                def _compute_subgroup_r2(y_true, y_pred, label):
                    r2 = r2_score(y_true, y_pred)
                    rng = np.random.RandomState(42)
                    boot_r2 = []
                    for _ in range(ctx._VAL_N_BOOTSTRAP):
                        idx = rng.choice(len(y_true), len(y_true), replace=True)
                        y_t, y_p = np.array(y_true)[idx], np.array(y_pred)[idx]
                        if np.var(y_t) > 0:
                            boot_r2.append(r2_score(y_t, y_p))
                    r2_lo = np.percentile(boot_r2, 2.5) if boot_r2 else np.nan
                    r2_hi = np.percentile(boot_r2, 97.5) if boot_r2 else np.nan
                    return {"Subgroup": label, "N": len(y_true), "r2": r2, "r2_lo": r2_lo, "r2_hi": r2_hi}

                _all_cat_results = {}

                for _cat_name in _cats_to_run:
                    if _cat_name not in ctx.exported_within_dict:
                        continue
                    _Xtr, _Xte, _ytr, _yte = ctx.exported_within_dict[_cat_name]
                    logger.info(f"\n  [{_cat_name}] ({_Xtr.shape[1]} features)")

                    try:
                        _model = create_validation_model_p2(is_classification=is_classification)
                        _model.fit(_Xtr, _ytr)

                        if is_classification:
                            # Use predict_proba and classification metrics for clf targets.
                            _Xte_df = _Xte if isinstance(_Xte, pd.DataFrame) else pd.DataFrame(np.array(_Xte))
                            _y_proba_cat = (
                                _model.predict_proba(_Xte_df)[:, 1]
                                if hasattr(_model, "predict_proba")
                                else _model.predict(_Xte_df).astype(float)
                            )
                            _cat_rows = [
                                _compute_subgroup_metrics_classification(np.array(_yte), _y_proba_cat, "Overall")
                            ]

                            for _demo_name, _demo_series in _demo_labels.items():
                                for _grp in _demo_series.dropna().unique():
                                    _mask = (_demo_series == _grp).values[: len(_yte)]
                                    if _mask.sum() > 20:
                                        _cat_rows.append(
                                            _compute_subgroup_metrics_classification(
                                                np.array(_yte)[_mask], _y_proba_cat[_mask], f"{_demo_name}: {_grp}"
                                            )
                                        )

                            _all_cat_results[_cat_name] = _cat_rows
                            _overall = _cat_rows[0]
                            logger.info(
                                f"    Overall AUC: {_overall['auc_val']:.4f} "
                                f"[{_overall['auc_lo']:.4f}-{_overall['auc_hi']:.4f}]"
                            )

                        else:
                            _y_pred = _model.predict(_Xte)

                            _cat_rows = [_compute_subgroup_r2(np.array(_yte), _y_pred, "Overall")]

                            for _demo_name, _demo_series in _demo_labels.items():
                                for _grp in _demo_series.dropna().unique():
                                    _mask = (_demo_series == _grp).values[: len(_yte)]
                                    if _mask.sum() > 20:
                                        _cat_rows.append(
                                            _compute_subgroup_r2(
                                                np.array(_yte)[_mask], _y_pred[_mask], f"{_demo_name}: {_grp}"
                                            )
                                        )

                            _all_cat_results[_cat_name] = _cat_rows
                            _overall = _cat_rows[0]
                            logger.info(
                                f"    Overall R²: {_overall['r2']:.4f} "
                                f"[{_overall['r2_lo']:.4f}-{_overall['r2_hi']:.4f}]"
                            )

                    except Exception as _e:
                        logger.info(f"    Warning: Failed: {_e}")
                    finally:
                        try:
                            del _model
                        except NameError:
                            pass
                        _pipeline_cleanup(label=f"P2 {_cat_name}", verbose=False)

                if _all_cat_results:
                    logger.info("\n" + "=" * 80)
                    logger.info("Stratified Performance Summary")
                    logger.info("=" * 80)

                    _subgroups = list({r["Subgroup"] for rows in _all_cat_results.values() for r in rows})
                    _subgroups.sort(key=lambda x: (0 if x == "Overall" else 1, x))

                    # Use the correct metric key and vmin for clf vs regression.
                    _within_metric_key = "auc_val" if is_classification else "r2"
                    _within_metric_lbl = "AUC" if is_classification else "R²"
                    _within_vmin = 0.5 if is_classification else 0

                    _wide_rows = []
                    for _sg in _subgroups:
                        _row = {"Subgroup": _sg}
                        for _cat, _rows in _all_cat_results.items():
                            _match = [r for r in _rows if r["Subgroup"] == _sg]
                            _short = _cat[:25] + "…" if len(_cat) > 25 else _cat
                            if _match:
                                _row[_short] = _match[0].get(_within_metric_key, np.nan)
                            else:
                                _row[_short] = np.nan
                        _wide_rows.append(_row)

                    _wide_df = pd.DataFrame(_wide_rows)
                    _num_cols = [c for c in _wide_df.columns if c != "Subgroup"]
                    _wide_styler = (
                        _wide_df.style.format({c: "{:.4f}" for c in _num_cols})
                        .background_gradient(
                            cmap="RdYlGn",
                            subset=_num_cols,
                            vmin=_within_vmin,
                            axis=None,
                            vmax=max(_within_vmin, _wide_df[_num_cols].max().max()),
                        )
                        .set_caption(
                            f"Subgroup {_within_metric_lbl} by Category -- {ctx.target_options} (T{ctx.tp_option})"
                        )
                        .set_table_styles(
                            [
                                {
                                    "selector": "caption",
                                    "props": [
                                        ("font-size", "14px"),
                                        ("font-weight", "700"),
                                        ("color", "#455A64"),
                                        ("padding-bottom", "8px"),
                                    ],
                                },
                                {
                                    "selector": "th",
                                    "props": [
                                        ("background-color", "#ECEFF1"),
                                        ("color", "#455A64"),
                                        ("font-size", "10px"),
                                        ("font-weight", "600"),
                                        ("padding", "6px 8px"),
                                    ],
                                },
                                {"selector": "td", "props": [("padding", "5px 8px"), ("font-size", "11px")]},
                            ]
                        )
                        .hide(axis="index")
                    )
                    display(_wide_styler)

                    _wide_df.to_csv(f"{results_dir}/stratified_within_categories.csv", index=False)
                    logger.info(f"\nSaved: {results_dir}/stratified_within_categories.csv")

                    # Use the correct metric key for clf vs regression.
                    logger.info(f"\n  Fairness summary (max {_within_metric_lbl} gap within each category):")
                    for _cat, _rows in _all_cat_results.items():
                        _gap_vals = [
                            r.get(_within_metric_key, np.nan)
                            for r in _rows
                            if r["Subgroup"] != "Overall" and not np.isnan(r.get(_within_metric_key, np.nan))
                        ]
                        if len(_gap_vals) >= 2:
                            _gap = max(_gap_vals) - min(_gap_vals)
                            _short = _cat[:30]
                            logger.info(f"    {_short}: max gap = {_gap:.4f}")

            elif "X_train" in ctx and "y_train" in ctx and _has_demo:
                # Across-categories mode: Stratified holdout analysis

                logger.info("\n" + "=" * 80)
                logger.info("  STRATIFIED HOLDOUT PERFORMANCE ANALYSIS")
                logger.info("=" * 80)

                task_type = detect_task_type(ctx.y_train)
                is_classification = task_type == "classification"
                logger.info(f"   Task type: {task_type}")

                # Train model once, predict on holdout
                logger.info("   Training stacking ensemble on X_train...")
                holdout_model = create_validation_model_p2(is_classification=is_classification)
                holdout_model.fit(ctx.X_train, ctx.y_train)

                if is_classification and hasattr(holdout_model, "predict_proba"):
                    y_pred_test = holdout_model.predict(ctx.X_test)
                    y_proba_test = holdout_model.predict_proba(ctx.X_test)[:, 1]
                else:
                    y_pred_test = holdout_model.predict(ctx.X_test)
                    y_proba_test = None

                # Extract demographic labels for test set
                _sv_test = ctx.sample_valid_test

                # Demographic label extraction
                _demo_map = {}

                if "sex" in _sv_test.columns:
                    _demo_map["Sex"] = {
                        "series": _sv_test["sex"].map({1: "Male", 2: "Female"}).reset_index(drop=True),
                        "groups": ["Male", "Female"],
                        "prefix": "Sex",
                    }
                    logger.info(f"   Sex: {_demo_map['Sex']['series'].value_counts().to_dict()}")

                if "parent_income" in _sv_test.columns:
                    _demo_map["SES"] = {
                        "series": pd.cut(
                            _sv_test["parent_income"],
                            bins=[0, 4, 7, 10],
                            labels=["Low", "Mid", "High"],
                            include_lowest=True,
                        ).reset_index(drop=True),
                        "groups": ["Low", "Mid", "High"],
                        "prefix": "SES",
                    }
                    logger.info(f"   SES: {_demo_map['SES']['series'].value_counts().to_dict()}")

                if "low_ale_children_p" in _sv_test.columns:
                    _s = (
                        _sv_test["low_ale_children_p"]
                        .astype(bool)
                        .map({True: "Low ALE", False: "Not Low ALE"})
                        .reset_index(drop=True)
                    )
                    if _s.isna().sum() == 0:
                        _demo_map["low_ALE"] = {
                            "series": _s,
                            "groups": ["Low ALE", "Not Low ALE"],
                            "prefix": "ALE",
                        }
                        logger.info(f"   low_ALE: {_s.value_counts().to_dict()}")

                if "high_ale" in _sv_test.columns:
                    _s = (
                        _sv_test["high_ale"]
                        .astype(bool)
                        .map({True: "High ALE", False: "Not High ALE"})
                        .reset_index(drop=True)
                    )
                    if _s.isna().sum() == 0:
                        _demo_map["high_ALE"] = {
                            "series": _s,
                            "groups": ["High ALE", "Not High ALE"],
                            "prefix": "ALE",
                        }
                        logger.info(f"   high_ALE: {_s.value_counts().to_dict()}")

                if "sex_P" in _sv_test.columns:
                    _demo_map["sex_P"] = {
                        "series": _sv_test["sex_P"].map({1: "Parent Male", 2: "Parent Female"}).reset_index(drop=True),
                        "groups": ["Parent Male", "Parent Female"],
                        "prefix": "ParSex",
                    }
                    logger.info(f"   sex_P: {_demo_map['sex_P']['series'].value_counts().to_dict()}")

                if "parent_education" in _sv_test.columns:
                    try:
                        _s = pd.qcut(
                            _sv_test["parent_education"],
                            q=3,
                            labels=["Low Edu", "Mid Edu", "High Edu"],
                            duplicates="drop",
                        ).reset_index(drop=True)
                        _demo_map["Edu"] = {
                            "series": _s,
                            "groups": ["Low Edu", "Mid Edu", "High Edu"],
                            "prefix": "Edu",
                        }
                        logger.info(f"   parent_education: {_s.value_counts().to_dict()}")
                    except Exception:
                        logger.info("   Warning: parent_education: qcut failed -- skipping")

                # Compute metrics per subgroup
                results_dir = os.path.join(os.getcwd(), "nestedcv_results_stratified")
                os.makedirs(results_dir, exist_ok=True)

                _target_lbl = ctx.target_options if "target_options" in ctx else "?"
                _tp_lbl = ctx.tp_option if "tp_option" in ctx else "?"

                logger.info("\n   Computing subgroup metrics (1000 bootstrap iterations each)...\n")

                ctx.all_rows = []
                subgroup_masks = {}

                # Overall
                if is_classification:
                    overall = _compute_subgroup_metrics_classification(np.array(ctx.y_test), y_proba_test, "Overall")
                else:
                    overall = _compute_subgroup_metrics_regression(np.array(ctx.y_test), y_pred_test, "Overall")
                ctx.all_rows.append(overall)
                subgroup_masks["Overall"] = np.ones(len(ctx.X_test), dtype=bool)
                logger.info(f"   Overall (n={len(ctx.y_test):,})")

                # Per demographic
                for _dim_name, _dim in _demo_map.items():
                    for _grp in _dim["groups"]:
                        mask = (_dim["series"] == _grp).values
                        if mask.sum() > 10:
                            label = f"{_dim['prefix']}: {_grp}"
                            if is_classification:
                                row = _compute_subgroup_metrics_classification(
                                    np.array(ctx.y_test)[mask], y_proba_test[mask], label
                                )
                            else:
                                row = _compute_subgroup_metrics_regression(
                                    np.array(ctx.y_test)[mask], y_pred_test[mask], label
                                )
                            ctx.all_rows.append(row)
                            subgroup_masks[label] = mask
                            logger.info(f"   {label} (n={mask.sum():,})")

                # Display results

                if is_classification:
                    # 1. Styled table
                    logger.info("\n" + "=" * 72)
                    logger.info("  PERFORMANCE TABLE")
                    logger.info("=" * 72)
                    results_df = _display_classification_table(ctx.all_rows, _target_lbl, _tp_lbl)
                    results_df.to_csv(f"{results_dir}/stratified_holdout_classification.csv", index=False)
                    logger.info(f"\n  Saved: {results_dir}/stratified_holdout_classification.csv")

                    # 2. Four-panel figure (forest plots + ROC + calibration)
                    _plot_classification_stratified(ctx.all_rows, results_dir, _target_lbl, _tp_lbl)

                    # 3. Threshold comparison bars
                    _plot_threshold_comparison(ctx.all_rows, results_dir, _target_lbl, _tp_lbl)

                    # 4. Fairness gap analysis
                    _print_fairness_summary(ctx.all_rows)

                else:
                    # Regression path (preserved from original)
                    _internal = {
                        "r2_val",
                        "rmse_val",
                        "mae_val",
                        "r2_lo",
                        "r2_hi",
                        "rmse_lo",
                        "rmse_hi",
                        "mae_lo",
                        "mae_hi",
                        "_residuals",
                    }
                    display_cols = [k for k in ctx.all_rows[0].keys() if k not in _internal]
                    results_df = pd.DataFrame(ctx.all_rows)[display_cols]
                    logger.info(results_df.to_string(index=False))
                    results_df.to_csv(f"{results_dir}/stratified_holdout_results.csv", index=False)
                    logger.info(f"\n  Saved: {results_dir}/stratified_holdout_results.csv")

                    _plot_regression_stratified(ctx.all_rows, results_dir, _target_lbl, _tp_lbl)

                # Feature importance per subgroup
                # Set _run_fi = False to skip on re-runs if computation is slow.
                _run_fi = True

                def _plot_feature_importance_subgroups(
                    model,
                    X_test_df,
                    masks,
                    y_test_arr,
                    is_cls=False,
                    top_n=10,
                    results_dir=".",
                ):
                    """Compute and plot per-subgroup feature importance."""
                    import warnings as _w

                    feature_names = list(X_test_df.columns)
                    rng = np.random.RandomState(42)
                    method_used = None

                    def _get_tree_base(m):
                        for attr in ("base_models", "estimators_", "named_estimators_"):
                            obj = getattr(m, attr, None)
                            if obj is None:
                                continue
                            items = obj.items() if isinstance(obj, dict) else enumerate(obj)
                            for _, bm in items:
                                t = str(type(bm)).lower()
                                if any(k in t for k in ("catboost", "xgb", "forest", "gradient")):
                                    return bm
                        t = str(type(m)).lower()
                        if any(k in t for k in ("catboost", "xgb", "forest")):
                            return m
                        return None

                    tree_bm = _get_tree_base(model)
                    bg_size = min(100, len(X_test_df))
                    bg_idx = rng.choice(len(X_test_df), bg_size, replace=False)
                    background = X_test_df.iloc[bg_idx]

                    # Probe strategy
                    probe_X = X_test_df.iloc[: min(10, len(X_test_df))]
                    strategy = None
                    explainer_obj = None

                    if tree_bm is not None:
                        try:
                            import shap

                            with _w.catch_warnings():
                                _w.simplefilter("ignore")
                                _exp = shap.TreeExplainer(tree_bm, check_additivity=False)
                                _sv = _exp.shap_values(probe_X, check_additivity=False)
                            if isinstance(_sv, list):
                                _sv = _sv[1] if len(_sv) > 1 else _sv[0]
                            _sv = np.array(_sv)
                            if _sv.shape[-1] == len(feature_names):
                                strategy = "tree"
                                explainer_obj = _exp
                                method_used = "SHAP TreeExplainer"
                        except Exception:
                            pass

                    if strategy is None:
                        try:
                            import shap

                            predict_fn = (
                                (lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1])
                                if is_cls
                                else (lambda x: model.predict(pd.DataFrame(x, columns=feature_names)))
                            )
                            with _w.catch_warnings():
                                _w.simplefilter("ignore")
                                _exp = shap.KernelExplainer(predict_fn, background, silent=True)
                                _sv = _exp.shap_values(probe_X, nsamples=30)
                            if isinstance(_sv, list):
                                _sv = _sv[1] if len(_sv) > 1 else _sv[0]
                            _sv = np.array(_sv)
                            if _sv.shape[-1] == len(feature_names):
                                strategy = "kernel"
                                explainer_obj = _exp
                                method_used = "SHAP KernelExplainer"
                        except Exception:
                            pass

                    if strategy is None:
                        try:
                            model.predict(probe_X)
                            strategy = "permutation"
                            method_used = "Permutation Importance"
                        except Exception:
                            logger.info("   Warning: No importance strategy succeeded.")
                            return

                    logger.info(f"\n Feature importance ({method_used}, {len(masks)} subgroups, top {top_n})...")

                    def _importance_for_mask(mask, label):
                        X_sub = X_test_df[mask]
                        y_sub = y_test_arr[mask]
                        n_sub = len(X_sub)
                        if n_sub < 10:
                            return None

                        shap_cap = 150 if strategy == "kernel" else 300
                        if n_sub > shap_cap:
                            idx = rng.choice(n_sub, shap_cap, replace=False)
                            X_shap = X_sub.iloc[idx]
                        else:
                            X_shap = X_sub

                        fi = None

                        if strategy == "tree":
                            try:
                                with _w.catch_warnings():
                                    _w.simplefilter("ignore")
                                    sv = explainer_obj.shap_values(X_shap, check_additivity=False)
                                if isinstance(sv, list):
                                    sv = sv[1] if len(sv) > 1 else sv[0]
                                sv = np.array(sv)
                                if sv.ndim == 1:
                                    sv = sv.reshape(1, -1)
                                fi = pd.Series(np.abs(sv).mean(axis=0), index=feature_names[: sv.shape[1]])
                                if (
                                    "DISPLAY_NAMES" in ctx
                                    and isinstance(DISPLAY_NAMES, dict)
                                    and len(DISPLAY_NAMES) > 0
                                ):
                                    fi.index = fi.index.map(lambda x: DISPLAY_NAMES.get(x, x))
                            except Exception as e:
                                logger.info(f"   Warning: TreeExplainer failed for {label}: {e}")
                        elif strategy == "kernel":
                            try:
                                with _w.catch_warnings():
                                    _w.simplefilter("ignore")
                                    sv = explainer_obj.shap_values(X_shap, nsamples=50)
                                if isinstance(sv, list):
                                    sv = sv[1] if len(sv) > 1 else sv[0]
                                sv = np.array(sv)
                                if sv.ndim == 1:
                                    sv = sv.reshape(1, -1)
                                fi = pd.Series(np.abs(sv).mean(axis=0), index=feature_names[: sv.shape[1]])
                                if (
                                    "DISPLAY_NAMES" in ctx
                                    and isinstance(DISPLAY_NAMES, dict)
                                    and len(DISPLAY_NAMES) > 0
                                ):
                                    fi.index = fi.index.map(lambda x: DISPLAY_NAMES.get(x, x))
                            except Exception as e:
                                logger.info(f"   Warning: KernelExplainer failed for {label}: {e}")
                        elif strategy == "permutation":
                            # Custom scorer used to bypass sklearn's
                            # response-method detection, which fails on
                            # stacking ensembles that lack _estimator_type.
                            try:
                                from sklearn.inspection import permutation_importance as _pi

                                n_rep = max(5, min(20, n_sub // 30))
                                if is_cls:

                                    def _auc_scorer(estimator, X, y):
                                        """Custom AUC scorer that calls predict_proba directly."""
                                        try:
                                            proba = estimator.predict_proba(X)[:, 1]
                                        except Exception:
                                            proba = estimator.predict(X)
                                        return roc_auc_score(y, proba)

                                    scoring = _auc_scorer
                                else:

                                    def _r2_scorer(estimator, X, y):
                                        """Custom R² scorer that calls .predict() directly."""
                                        return r2_score(y, estimator.predict(X))

                                    scoring = _r2_scorer
                                perm = _pi(
                                    model, X_sub, y_sub, n_repeats=n_rep, random_state=42, scoring=scoring, n_jobs=-1
                                )
                                vals = np.maximum(perm.importances_mean, 0)
                                fi = pd.Series(vals, index=feature_names[: len(vals)])
                                if (
                                    "DISPLAY_NAMES" in ctx
                                    and isinstance(DISPLAY_NAMES, dict)
                                    and len(DISPLAY_NAMES) > 0
                                ):
                                    fi.index = fi.index.map(lambda x: DISPLAY_NAMES.get(x, x))
                            except Exception as e:
                                logger.info(f"   Warning: Permutation failed for {label}: {e}")

                        if fi is None or fi.sum() == 0:
                            return None
                        return fi.nlargest(top_n)

                    group_fi = {}
                    for label, mask in masks.items():
                        fi_top = _importance_for_mask(mask, label)
                        if fi_top is not None:
                            group_fi[label] = fi_top
                            logger.info(f"   {label} (n={mask.sum()})")
                        else:
                            logger.info(f"   {label}: skipped")

                    if not group_fi:
                        logger.info("   Warning: No importances computed.")
                        return

                    # Plot
                    n_groups = len(group_fi)
                    ncols = min(3, n_groups)
                    nrows = int(np.ceil(n_groups / ncols))
                    subplot_h = max(6.5, top_n * 0.45 + 2.2)
                    subplot_w = 8.0

                    fig, axes = plt.subplots(
                        nrows, ncols, figsize=(subplot_w * ncols, subplot_h * nrows), squeeze=False
                    )
                    axes_flat = axes.flatten()

                    for ax_idx, (label, fi_top) in enumerate(group_fi.items()):
                        ax = axes_flat[ax_idx]
                        color = _sg_color(label)
                        fi_norm = fi_top / fi_top.max() if fi_top.max() > 0 else fi_top
                        n_bars = len(fi_norm)
                        vals_plot = fi_norm.values[::-1]
                        raw_plot = fi_top.values[::-1]
                        feats_plot = list(fi_norm.index[::-1])
                        y_pos = np.arange(n_bars)
                        alphas = np.linspace(0.40, 0.92, n_bars)

                        for j, (yp, val, raw, alpha) in enumerate(zip(y_pos, vals_plot, raw_plot, alphas)):
                            ax.barh(yp, val, color=color, alpha=alpha, edgecolor="none", height=0.72)
                            ax.text(
                                val + 0.015,
                                yp,
                                f"{raw:.4f}",
                                va="center",
                                ha="left",
                                fontsize=9.5,
                                color="#333333",
                                fontweight="bold",
                            )

                        feat_labels = [(f[:28] + "…" if len(f) > 28 else f) for f in feats_plot]
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(feat_labels, fontsize=8.5)
                        n_obs = int(masks[label].sum())
                        ax.set_title(f"{label}\n(n = {n_obs:,})", fontsize=10, fontweight="bold", color=color, pad=7)
                        ax.set_xlabel("Relative importance", fontsize=8)
                        ax.set_xlim([0, 1.35])
                        ax.axvline(0, color=color, lw=4, alpha=0.5)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.spines["left"].set_visible(False)
                        ax.tick_params(axis="y", length=0, pad=4)
                        ax.grid(axis="x", alpha=0.18, ls="--", lw=0.8)

                    for ax_idx in range(n_groups, len(axes_flat)):
                        axes_flat[ax_idx].set_visible(False)

                    fig.suptitle(
                        f"Feature Importance per Subgroup\nTop {top_n} features  |  {method_used}",
                        fontsize=13,
                        fontweight="bold",
                        y=1.015,
                    )
                    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4.0, w_pad=3.5)
                    out_path = f"{results_dir}/stratified_feature_importance.png"
                    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
                    plt.show()
                    logger.info(f"   Saved: {out_path}")

                if _run_fi:
                    _plot_feature_importance_subgroups(
                        model=holdout_model,
                        X_test_df=ctx.X_test if isinstance(ctx.X_test, pd.DataFrame) else pd.DataFrame(ctx.X_test),
                        masks=subgroup_masks,
                        y_test_arr=np.array(ctx.y_test),
                        is_cls=is_classification,
                        top_n=10,
                        results_dir=results_dir,
                    )
                else:
                    logger.info("  Feature importance skipped (_run_fi = False)")

            else:
                if not _has_demo:
                    logger.info("\nError: sample_valid_test not found")
                else:
                    logger.info("\nError: Required variables not found")

        except Exception as e:
            logger.info(f"\nError in stratified analysis: {e}")
            import traceback

            traceback.print_exc()

        logger.info("\n" + "=" * 72)
        logger.info("  Stratified Performance Analysis Complete")
        logger.info("=" * 72)

        # [Cache + report save]
        _RS["section"] = "uncategorized"
        try:
            _pipeline_cleanup(label="P2 complete")

            cache_save(
                {"reporter": locals().get("reporter") or locals().get("stratified_reporter")},
                cell_tag=_CTAG_13,
                key=_CKEY_13,
                label=f"{ctx.target_options}_tp{ctx.tp_option}_Stratified Performance",
            )
            report_save_state(make_run_key())
        except Exception as _ce_13:
            logger.info(f"[Cache] Could not save stratified: {_ce_13}")


def run() -> None:
    """Run this stage (notebook cell 13)."""
    global LinearSegmentedColormap, copy, np, pd
    warnings.filterwarnings("ignore")

    # Read the bootstrap count from the stage settings
    ctx._VAL_N_BOOTSTRAP = ctx.n_bootstrap

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    # Batch dispatch
    _is_bv_16 = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_16:
        try:
            _cbv_16 = cache_load("main_runner")
            if (
                _cbv_16
                and isinstance(_cbv_16, dict)
                and _cbv_16.get("_batch_archive")
                and len(_cbv_16["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_16["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_16["_batch_archive"]
                _is_bv_16 = True
        except Exception:
            pass

    if _is_bv_16:
        _vgs_16 = {
            k: ctx.get(k)
            for k in [
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "target_options",
                "tp_option",
                "metrics_dict_full",
                "exported_within_dict",
                "consensus_df",
                "model_weights",
                "split",
                "sample_valid",
                "sample_valid_test",
                "sample_valid_train",
            ]
        }
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH -- Extended Stratified Performance: processing {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)

        for _vi_16, (_vk_16, _vd_16) in enumerate(ctx._batch_archive.items()):
            _vl_16 = _vd_16.get("target", _vk_16)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_16 + 1}/{len(ctx._batch_archive)}] Target: {_vl_16}")
            logger.info(f"{'-' * 68}")
            for _gk in [
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "metrics_dict_full",
                "exported_within_dict",
                "consensus_df",
                "model_weights",
                "split",
                "sample_valid",
                "sample_valid_test",
                "sample_valid_train",
            ]:
                if _vd_16.get(_gk) is not None:
                    ctx[_gk] = _vd_16[_gk]
            ctx["target_options"] = _vl_16
            if "timepoint" in _vd_16:
                ctx["tp_option"] = _vd_16["timepoint"]
            _run_for_current_target()
            # Archive validation reporters back into _batch_archive
            import copy

            if _vk_16 in ctx._batch_archive:
                ctx._batch_archive[_vk_16]["all_rows"] = copy.deepcopy(ctx.get("all_rows"))

        for _k, _v in _vgs_16.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH COMPLETE -- Extended Stratified Performance: {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)
    else:
        _run_for_current_target()

    # Batch fairness heatmap: Cross-Target Demographic Equity Summary
    # Rows = targets, Columns = demographic subgroups, Cells = R² or AUC.
    # Final column = max disparity (worst − best subgroup) flagged in red.
    # Works for both regression and classification.

    try:
        _is_batch_15 = "_batch_archive" in ctx and isinstance(ctx._batch_archive, dict) and len(ctx._batch_archive) > 1
        if not _is_batch_15:
            try:
                _cbv15 = cache_load("main_runner")
                if (
                    _cbv15
                    and isinstance(_cbv15, dict)
                    and _cbv15.get("_batch_archive")
                    and len(_cbv15["_batch_archive"]) > 1
                ):
                    ctx._batch_archive = _cbv15["_batch_archive"]
                    _is_batch_15 = True
            except Exception:
                pass

        if _is_batch_15:
            import numpy as np
            import pandas as pd
            from matplotlib.colors import LinearSegmentedColormap

            _TDN = TARGET_DISPLAY_NAMES if "TARGET_DISPLAY_NAMES" in ctx else {}
            _fair_rows = []

            # Read fairness data from _batch_archive[key]['all_rows'] (stored by batch wrapper above)
            _fair_source = {}
            for _fk, _fv in ctx._batch_archive.items():
                _ar = _fv.get("all_rows")
                if _ar and isinstance(_ar, list) and len(_ar) > 0:
                    _fair_source[_fk] = _ar
            if len(_fair_source) < 2:
                logger.info("  Batch fairness heatmap skipped: <2 targets with subgroup data")
                _fair_source = {}  # skip rendering

            for _fk, _fair_data in _fair_source.items():
                _tname = _fk.split("__tp")[0] if "__tp" in _fk else _fk
                _tlabel = _TDN.get(_tname, _tname)
                _is_clf_fair = any("auc_val" in r for r in _fair_data if isinstance(r, dict))
                _metric_key = "auc_val" if _is_clf_fair else "r2_val"
                _row = {"Target": _tlabel}
                _vals = []
                for _sg in _fair_data:
                    if isinstance(_sg, dict) and _metric_key in _sg:
                        _sg_name = _sg.get("Subgroup", "Unknown")
                        _row[_sg_name] = round(_sg[_metric_key], 3)
                        if _sg_name != "Overall":
                            _vals.append(_sg[_metric_key])
                if _vals and len(_vals) >= 2:
                    _row["Max Gap"] = round(max(_vals) - min(_vals), 3)
                _fair_rows.append(_row)

            if len(_fair_rows) >= 2:
                _fair_df = pd.DataFrame(_fair_rows).set_index("Target")
                # Separate Max Gap column
                _gap_col = _fair_df.pop("Max Gap") if "Max Gap" in _fair_df.columns else None

                # Sort by Overall descending
                if "Overall" in _fair_df.columns:
                    _fair_df = _fair_df.sort_values("Overall", ascending=False)
                    if _gap_col is not None:
                        _gap_col = _gap_col.loc[_fair_df.index]

                _n_r = len(_fair_df)
                _n_c = len(_fair_df.columns)
                _fig_w = max(12, _n_c * 1.1 + 4)
                _fig_h = max(5, _n_r * 0.5 + 2)

                fig_fair, ax_fair = plt.subplots(figsize=(_fig_w, _fig_h))

                _is_clf_mode = any("auc" in str(c).lower() for c in _fair_df.columns) or (
                    _fair_df.values[~np.isnan(_fair_df.values.astype(float))].mean() > 0.4
                )
                _vmin_f = 0.5 if _is_clf_mode else 0
                _vmax_f = 1.0 if _is_clf_mode else max(0.3, np.nanmax(_fair_df.values.astype(float)) + 0.02)

                _cmap_fair = LinearSegmentedColormap.from_list(
                    "fair_cmap", ["#FEE2E2", "#FECACA", "#FFFFFF", "#BAE6FD", "#3B82F6"], N=256
                )

                _im_f = ax_fair.imshow(
                    _fair_df.values.astype(float), aspect="auto", cmap=_cmap_fair, vmin=_vmin_f, vmax=_vmax_f
                )

                # Cell annotations
                for _ri in range(_n_r):
                    for _ci in range(_n_c):
                        _v = _fair_df.values[_ri, _ci]
                        if not np.isnan(float(_v)):
                            _tc = "#1B2A4A" if float(_v) > (_vmin_f + _vmax_f) / 2 else "#7F1D1D"
                            ax_fair.text(
                                _ci,
                                _ri,
                                f"{float(_v):.3f}",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color=_tc,
                                fontweight="bold",
                            )

                ax_fair.set_xticks(range(_n_c))
                ax_fair.set_xticklabels(_fair_df.columns, rotation=40, ha="right", fontsize=9)
                ax_fair.set_yticks(range(_n_r))
                ax_fair.set_yticklabels(_fair_df.index, fontsize=9)

                _metric_label = "AUC" if _is_clf_mode else "R²"
                ax_fair.set_title(
                    f"Batch Fairness -- {_metric_label} by Demographic Subgroup\n({_n_r} targets × {_n_c} subgroups)",
                    fontsize=13,
                    fontweight="bold",
                    pad=14,
                )

                _cbar_f = plt.colorbar(_im_f, ax=ax_fair, shrink=0.7, pad=0.02)
                _cbar_f.set_label(_metric_label, fontsize=10)

                # Max gap annotation on right margin
                if _gap_col is not None:
                    for _ri, (_tgt, _gap) in enumerate(_gap_col.items()):
                        if pd.notna(_gap):
                            _gc = "#DC2626" if _gap > 0.05 else "#6B7280"
                            ax_fair.text(
                                _n_c + 0.3,
                                _ri,
                                f"Δ{_gap:.3f}",
                                va="center",
                                ha="left",
                                fontsize=8,
                                fontweight="bold",
                                color=_gc,
                            )
                    ax_fair.text(_n_c + 0.3, -0.8, "Max Gap", ha="left", fontsize=8, fontweight="bold", color="#374151")

                plt.tight_layout()
                try:
                    fig_fair.savefig(
                        f"{results_dir_summary}/batch_fairness_heatmap.png",  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                        dpi=250,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    fig_fair.savefig(
                        f"{results_dir_summary}/batch_fairness_heatmap.pdf",  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    logger.info("   Saved: batch_fairness_heatmap.png / .pdf")
                except Exception:
                    pass
                plt.show()

    except Exception as _e_fair:
        logger.info(f"  Warning: Batch fairness heatmap: {_e_fair}")
