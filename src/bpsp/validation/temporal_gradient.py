"""Temporal gradient testing (the `temporal_gradient` stage, Supplement S1.16.2).

Cross-timepoint temporal transfer matrix. Trains on concurrent data at
each timepoint and tests on concurrent data at every other timepoint.
Produces a full transfer matrix and distance-dependent decay curves
showing how predictive relationships degrade with temporal separation.

The stage requires the `main` stage of the same run.
"""

import copy
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score

from bpsp.context import ctx
from bpsp.data.preprocessing import across_categories_data, create_lightweight_model, detect_task_type
from bpsp.reporting import _RS, cache_load

logger = logging.getLogger(__name__)

# Target color palette (consistent across manuscript)
_TG_COLORS = {
    "depress_D_p": "#1565C0",  # Blue
    "sbt_core_depression": "#4169E1",  # Royal Blue (matches notebook)
    "anxdisord_D_p": "#C62828",  # Red
    "sbt_core_anxiety": "#D32F2F",  # Red (lighter)
    "adhd_D_p": "#E6A817",  # Gold (darkened for visibility)
    "social_problems_D_p": "#E65100",  # Orange
    "internal_D_p": "#6A1B9A",  # Purple
    "external_D_p": "#2E7D32",  # Green
    "parent_depress_D_p": "#0D47A1",  # Dark Blue
    "parent_depressed_p": "#191970",  # Midnight Blue (ASR)
    "parent_anxdisord_D_p": "#B71C1C",  # Dark Red
    "parent_adhd_D_p": "#F57F17",  # Dark Gold
    "parent_external_D_p": "#1B5E20",  # Dark Green
    "parent_internal_D_p": "#4A148C",  # Dark Purple
    # Depression Subtypes (SBT, matches _BATCH_TARGET_COLORS in the main stage)
    "sbt_anxiety_depression": "#DA70D6",  # Orchid
    "sbt_guilt_hopelessness": "#4B0082",  # Indigo
    "sbt_fatigue_sleep": "#6A5ACD",  # Slate Blue
    "sbt_somatic_depression": "#BC8F8F",  # Rosy Brown
    "sbt_emotional_dysregulation": "#FF4500",  # Orange Red
    "sbt_social_withdrawal": "#5F9EA0",  # Cadet Blue
    "sbt_avoidance_fear": "#9370DB",  # Medium Purple
    "sbt_aggression_irritability": "#B22222",  # Firebrick
    "sbt_academic_cognitive": "#B8960B",  # Golden Amber
    "sbt_perfectionism_achievement": "#C44A80",  # Rose Magenta
}

_TG_LABELS = {
    "depress_D_p": "Depression",
    "sbt_core_depression": "Depression (Primary)",
    "anxdisord_D_p": "Anxiety",
    "sbt_core_anxiety": "Anxiety (Primary)",
    "adhd_D_p": "ADHD",
    "social_problems_D_p": "Social Problems",
    "internal_D_p": "Internalizing",
    "external_D_p": "Externalizing",
    "parent_depress_D_p": "Parent Depression",
    "parent_depressed_p": "Parent Depression",
    "parent_anxdisord_D_p": "Parent Anxiety",
    "parent_adhd_D_p": "Parent ADHD",
    "parent_external_D_p": "Parent Externalizing",
    "parent_internal_D_p": "Parent Internalizing",
    # Depression Subtypes
    "sbt_anxiety_depression": "Anxiety/Depression",
    "sbt_guilt_hopelessness": "Guilt/Hopelessness",
    "sbt_fatigue_sleep": "Fatigue/Sleep",
    "sbt_somatic_depression": "Somatic",
    "sbt_emotional_dysregulation": "Emotional Dysregulation",
    "sbt_social_withdrawal": "Social Withdrawal",
    "sbt_avoidance_fear": "Avoidance/Fear",
    "sbt_aggression_irritability": "Aggression/Irritability",
    "sbt_academic_cognitive": "Academic/Cognitive",
    "sbt_perfectionism_achievement": "Perfectionism/Achievement",
}

_DIST_LABELS = {
    0: "0\n(same TP)",
    1: "1 wave\n(~1 yr)",
    2: "2 waves\n(~2 yr)",
    3: "3 waves\n(~3 yr)",
    4: "4 waves\n(~4 yr)",
}

# CORE: Build transfer matrix for a single target


def _build_transfer_matrix(target_name, timepoints=None, group_val=None):
    """
    Train on concurrent data at each timepoint, test on every other.
    Returns (matrix_df, details_dict) or (None, None) on failure.
    """
    if timepoints is None:
        timepoints = [0, 1, 2, 3, 4]

    _group = group_val

    # 1. Gather concurrent data at each available timepoint
    tp_data = {}
    for _t in timepoints:
        try:
            _X_tr, _X_te, _y_tr, _y_te = across_categories_data(
                _t, [target_name], group=_group, scale_x=True, scale_y=False, verbose=False
            )
            _X_all = pd.concat([_X_tr, _X_te], axis=0, ignore_index=True)
            if isinstance(_y_tr, (pd.Series, pd.DataFrame)):
                _y_all = pd.concat([_y_tr, _y_te], axis=0, ignore_index=True)
            else:
                _y_all = np.concatenate([np.asarray(_y_tr).ravel(), np.asarray(_y_te).ravel()])
            _y_all = np.asarray(_y_all).ravel().astype(float)

            _mask = ~np.isnan(_y_all)
            _X_all = _X_all[_mask].reset_index(drop=True)
            _y_all = _y_all[_mask]

            if len(_y_all) < 50:
                logger.info(f"    T{_t}: SKIPPED (N={len(_y_all)} < 50)")
                continue

            tp_data[_t] = (_X_all, _y_all)
            logger.info(f"    T{_t}: N={len(_X_all):,}, features={_X_all.shape[1]}")
        except Exception as _e:
            logger.info(f"    T{_t}: SKIPPED ({_e})")

    if len(tp_data) < 2:
        logger.info("    Fewer than 2 timepoints available -- cannot build matrix.")
        return None, None

    available_tps = sorted(tp_data.keys())

    # 2. Global feature intersection (same features for all pairs)
    # Ensures diagonal (same-TP) entries don't benefit from having
    # more features than off-diagonal, isolating temporal decay.
    _common_cols = set(tp_data[available_tps[0]][0].columns)
    for _t in available_tps[1:]:
        _common_cols &= set(tp_data[_t][0].columns)
    _common_cols = sorted(_common_cols)
    logger.info(f"    Common features across all timepoints: {len(_common_cols)}")

    if len(_common_cols) < 10:
        logger.info("    Too few common features -- cannot build matrix.")
        return None, None

    # 3. Build NxN transfer matrix
    # DIAGONAL (train TP == test TP): 5-fold CV to avoid in-sample
    # inflation. Without CV, the model trains and tests on identical
    # rows, producing overfitted R² that exaggerates apparent decay.
    # OFF-DIAGONAL: train on ALL data at one TP, test on ALL data at
    # another TP. This is valid because feature values differ across
    # timepoints even for the same subjects.
    from sklearn.model_selection import KFold, StratifiedKFold

    _n_cv = 5
    matrix = pd.DataFrame(
        np.nan, index=[f"Train T{t}" for t in available_tps], columns=[f"Test T{t}" for t in available_tps]
    )
    details = {}

    _task = detect_task_type(tp_data[available_tps[0]][1])
    _is_clf = _task == "classification"

    for _t_train in available_tps:
        _X_train_full, _y_train_full = tp_data[_t_train]

        for _t_test in available_tps:
            _X_test_full, _y_test_full = tp_data[_t_test]

            if _t_train == _t_test:
                # DIAGONAL: K-fold CV (prevents in-sample inflation)
                _X_cv = _X_train_full[_common_cols].copy()
                _y_cv = _y_train_full.copy()

                if _is_clf:
                    _cv = StratifiedKFold(n_splits=_n_cv, shuffle=True, random_state=42)
                else:
                    _cv = KFold(n_splits=_n_cv, shuffle=True, random_state=42)

                _fold_scores = []
                for _tr_idx, _te_idx in _cv.split(_X_cv, _y_cv):
                    _imp = SimpleImputer(strategy="median")
                    _Xf_tr = pd.DataFrame(_imp.fit_transform(_X_cv.iloc[_tr_idx]), columns=_common_cols)
                    _Xf_te = pd.DataFrame(_imp.transform(_X_cv.iloc[_te_idx]), columns=_common_cols)
                    _m = create_lightweight_model(is_classification=_is_clf)
                    _m.fit(_Xf_tr, _y_cv[_tr_idx])
                    if _is_clf:
                        _p = _m.predict_proba(_Xf_te)[:, 1]
                        _fold_scores.append(roc_auc_score(_y_cv[_te_idx], _p))
                    else:
                        _p = _m.predict(_Xf_te)
                        _fold_scores.append(r2_score(_y_cv[_te_idx], _p))

                _score = np.mean(_fold_scores)
                _score_sd = np.std(_fold_scores, ddof=1)
                _n_tr = int(len(_X_cv) * (_n_cv - 1) / _n_cv)
                _n_te = len(_X_cv) - _n_tr
                _extra = {"cv_folds": _fold_scores, "cv_sd": _score_sd}

            else:
                # OFF-DIAGONAL: full train → full test (valid: different TPs)
                _Xtr = _X_train_full[_common_cols].copy()
                _Xte = _X_test_full[_common_cols].copy()

                _imp = SimpleImputer(strategy="median")
                _Xtr = pd.DataFrame(_imp.fit_transform(_Xtr), columns=_common_cols)
                _Xte = pd.DataFrame(_imp.transform(_Xte), columns=_common_cols)

                _model = create_lightweight_model(is_classification=_is_clf)
                _model.fit(_Xtr, _y_train_full)

                if _is_clf:
                    _pred = _model.predict_proba(_Xte)[:, 1]
                    _score = roc_auc_score(_y_test_full, _pred)
                else:
                    _pred = _model.predict(_Xte)
                    _score = r2_score(_y_test_full, _pred)

                _n_tr = len(_Xtr)
                _n_te = len(_Xte)
                _extra = {}

            matrix.loc[f"Train T{_t_train}", f"Test T{_t_test}"] = _score
            details[(_t_train, _t_test)] = {
                "score": _score,
                "n_features": len(_common_cols),
                "n_train": _n_tr,
                "n_test": _n_te,
                "distance": abs(_t_test - _t_train),
                "is_diagonal": (_t_train == _t_test),
                **_extra,
            }

    return matrix, details


def _compute_decay_summary(details, available_tps):
    """Compute mean score at each temporal distance (forward only)."""
    max_dist = max(available_tps) - min(available_tps)
    rows = []
    for d in range(0, max_dist + 1):
        scores = []
        for (t_train, t_test), info in details.items():
            if (t_test - t_train) == d:  # forward only
                scores.append(info["score"])
        if scores:
            rows.append(
                {
                    "distance": d,
                    "mean": np.mean(scores),
                    "std": np.std(scores, ddof=1) if len(scores) > 1 else 0.0,
                    "n": len(scores),
                }
            )
    return pd.DataFrame(rows)


def _compute_asymmetry(details, available_tps):
    """Compare forward vs backward transfer at each temporal distance.
    Tests whether predictive relationships are symmetric across development."""
    max_dist = max(available_tps) - min(available_tps)
    rows = []
    for d in range(1, max_dist + 1):
        fwd = [info["score"] for (t_tr, t_te), info in details.items() if (t_te - t_tr) == d]
        bwd = [info["score"] for (t_tr, t_te), info in details.items() if (t_tr - t_te) == d]
        if fwd and bwd:
            rows.append(
                {
                    "distance": d,
                    "forward_mean": np.mean(fwd),
                    "backward_mean": np.mean(bwd),
                    "delta": np.mean(fwd) - np.mean(bwd),
                    "n_forward": len(fwd),
                    "n_backward": len(bwd),
                }
            )
    return pd.DataFrame(rows)


def _compute_decay_slopes(all_results):
    """Compute linear decay slope (distance 1→max) for each target.
    Returns DataFrame with target, slope, peak R², endpoint R², total_drop_pct."""
    from scipy import stats

    rows = []
    for _tgt, _res in all_results.items():
        _decay = _res.get("decay")
        if _decay is None or len(_decay) < 2:
            continue
        _fwd = _decay[_decay["distance"] >= 1].copy()
        if len(_fwd) < 2:
            continue
        _x = _fwd["distance"].values.astype(float)
        _y = _fwd["mean"].values
        _slope, _intercept, _rval, _, _stderr = stats.linregress(_x, _y)
        _peak = _fwd["mean"].values[0]
        _end = _fwd["mean"].values[-1]
        _pct = ((_peak - _end) / _peak * 100) if _peak > 0 else 0
        rows.append(
            {
                "target": _tgt,
                "label": _TG_LABELS.get(_tgt, _tgt),
                "slope": _slope,
                "slope_se": _stderr,
                "peak_r2": _peak,
                "end_r2": _end,
                "total_drop_pct": _pct,
                "r_value": _rval,
                "color": _TG_COLORS.get(_tgt, "#666666"),
            }
        )
    return pd.DataFrame(rows)


# Reference autocorrelations (T0→T4) from TargetsAllTps.csv
# Populate with verified values to enable slope-vs-stability scatter.
# Set to None or {} to skip that panel.
_AUTOCORR_T0T4 = {
    "depress_D_p": 0.411,
    "sbt_core_depression": 0.352,
    "anxdisord_D_p": 0.471,
    "adhd_D_p": 0.587,
    "social_problems_D_p": 0.500,
    "internal_D_p": 0.536,
    "external_D_p": 0.581,
    "parent_depress_D_p": 0.584,
    "parent_anxdisord_D_p": 0.552,
    "parent_adhd_D_p": 0.626,
    "parent_external_D_p": 0.635,
    "parent_internal_D_p": 0.650,
}


def _compute_autocorr_decay(target_name, available_tps, group_val=None):
    """Compute pairwise forward autocorrelation r² from the sample DataFrame.
    Uses the same subjects/timepoints the model sees.
    Returns decay DataFrame with columns: distance, mean (r²), std, n, mean_r."""
    from scipy.stats import pearsonr

    if "sample" not in ctx:
        return None

    _samp = ctx["sample"]
    if "subject" not in _samp.columns or "time" not in _samp.columns:
        return None
    if target_name not in _samp.columns:
        return None

    # Apply group filter to match model subsetting
    _s = _samp.copy()
    if group_val == "Female":
        _s = _s[_s["sex"] == 2]
    elif group_val == "Male":
        _s = _s[_s["sex"] == 1]
    elif group_val == "high_ale":
        _s = _s[_s.get("high_ale", pd.Series(False)) == True]
    elif group_val == "low_ale":
        _s = _s[_s.get("low_ale_children_p", pd.Series(False)) == True]
    elif group_val is not None:
        logger.info(f"    Warning: group '{group_val}' not supported in autocorr filter -- using full sample")

    # Extract (subject, target) at each timepoint
    tp_vals = {}
    for _t in available_tps:
        _sub = _s[_s["time"] == _t][["subject", target_name]].dropna()
        if len(_sub) > 50:
            tp_vals[_t] = _sub.set_index("subject")[target_name]

    if len(tp_vals) < 2:
        return None

    max_dist = max(available_tps) - min(available_tps)
    rows = []
    for d in range(1, max_dist + 1):
        r_vals = []
        for t1 in available_tps:
            t2 = t1 + d
            if t1 in tp_vals and t2 in tp_vals:
                _common = tp_vals[t1].index.intersection(tp_vals[t2].index)
                if len(_common) > 50:
                    _r, _ = pearsonr(tp_vals[t1].loc[_common], tp_vals[t2].loc[_common])
                    r_vals.append(_r)
        if r_vals:
            _r_arr = np.array(r_vals)
            _r2_arr = _r_arr**2
            rows.append(
                {
                    "distance": d,
                    "mean": np.mean(_r2_arr),
                    "std": np.std(_r2_arr, ddof=1) if len(_r2_arr) > 1 else 0.0,
                    "n": len(r_vals),
                    "mean_r": np.mean(_r_arr),
                }
            )

    return pd.DataFrame(rows) if rows else None


# VISUALIZATION


def _plot_gradient_results(all_results, fig_dir="temporal_gradient_figures"):
    """
    Publication figures:
      Figure 1: Multi-target decay curves with autocorrelation overlay
      Figure 2: Per-target transfer heatmaps (shared colorbar)
      Figure 3: Directional asymmetry with N-pair annotations
      Figure 4: Decay slopes (A) + autocorrelation scatter with CI band (B)
    """
    import seaborn as sns
    from matplotlib.lines import Line2D

    os.makedirs(fig_dir, exist_ok=True)

    targets_with_data = {t: r for t, r in all_results.items() if r.get("decay") is not None and len(r["decay"]) > 1}

    if not targets_with_data:
        logger.info("  No targets with sufficient data for visualization.")
        return

    # FIGURE 1: Multi-target decay curves
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)

    # Subtle vertical separator between CV reference and transfer decay
    ax.axvline(x=0.5, color="#E0E0E0", linewidth=0.8, linestyle="-", zorder=0)
    ax.axvspan(-0.3, 0.5, alpha=0.025, color="#BDBDBD", zorder=0)

    # Zone labels: understated, professional
    ax.text(
        0.15,
        0.97,
        "CV reference",
        transform=ax.get_xaxis_transform(),
        fontsize=6.5,
        color="#333333",
        ha="center",
        va="top",
        fontfamily="sans-serif",
    )
    ax.text(
        2.5,
        0.97,
        "Transfer decay",
        transform=ax.get_xaxis_transform(),
        fontsize=6.5,
        color="#333333",
        ha="center",
        va="top",
        fontfamily="sans-serif",
    )

    # Pre-compute peak (distance 1) values
    _peak_vals = {}
    _stp_vals = {}
    for _tgt, _res in targets_with_data.items():
        _d = _res["decay"]
        _row0 = _d[_d["distance"] == 0]
        _row1 = _d[_d["distance"] == 1]
        _stp_vals[_tgt] = _row0["mean"].values[0] if len(_row0) else np.nan
        _peak_vals[_tgt] = _row1["mean"].values[0] if len(_row1) else _stp_vals[_tgt]

    _sd_label_added = False
    _n_tgts = len(targets_with_data)
    _many = _n_tgts > 6
    # Adapt line weight and markers to target count
    _lw = 1.6 if _many else 2.6
    _ms = 4 if _many else 8
    _mew = 0.6 if _many else 1.0

    for _tgt, _res in targets_with_data.items():
        _decay = _res["decay"]
        _col = _TG_COLORS.get(_tgt, "#666666")
        _lab_base = _TG_LABELS.get(_tgt, _tgt)
        _peak = _peak_vals.get(_tgt, np.nan)

        _d0 = _decay[_decay["distance"] == 0]
        _d1plus = _decay[_decay["distance"] >= 1]

        # Reference segment: skip when many targets to reduce clutter
        if not _many and len(_d0) and len(_d1plus):
            _ref_x = [_d0["distance"].values[0], _d1plus["distance"].values[0]]
            _ref_y = [_d0["mean"].values[0], _d1plus["mean"].values[0]]
            ax.plot(_ref_x, _ref_y, "--", color=_col, linewidth=1.4, alpha=0.5, zorder=2)
            ax.plot(
                _d0["distance"].values[0],
                _d0["mean"].values[0],
                "o",
                color="white",
                markersize=7,
                markeredgecolor=_col,
                markeredgewidth=1.8,
                zorder=4,
            )

        # Decay segment
        if len(_d1plus) >= 2:
            _lab = f"{_lab_base}" if _many else (f"{_lab_base} ({_peak:.3f})" if not np.isnan(_peak) else _lab_base)
            ax.plot(
                _d1plus["distance"],
                _d1plus["mean"],
                "o-",
                color=_col,
                label=_lab,
                linewidth=_lw,
                markersize=_ms,
                markeredgecolor="white",
                markeredgewidth=_mew,
                zorder=3,
            )
            # SD ribbon: skip when many targets (overlapping fills = mud)
            if not _many and _d1plus["std"].sum() > 0:
                _sd_lbl = "\u00b11 SD (across TP pairs)" if not _sd_label_added else "_nolegend_"
                ax.fill_between(
                    _d1plus["distance"],
                    _d1plus["mean"] - _d1plus["std"],
                    _d1plus["mean"] + _d1plus["std"],
                    alpha=0.10,
                    color=_col,
                    zorder=1,
                    label=_sd_lbl,
                )
                _sd_label_added = True

        # Annotate endpoint: skip when many targets to prevent stacking
        _n_tgts = len(targets_with_data)
        _max_row = _decay[_decay["distance"] == _decay["distance"].max()]
        if _n_tgts <= 6 and len(_max_row) and not np.isnan(_peak) and _peak > 0:
            _end_val = _max_row["mean"].values[0]
            _pct_drop = ((_peak - _end_val) / _peak) * 100
            if _n_tgts <= 4:
                _ann_txt = f"R\u00b2={_end_val:.3f} ({-_pct_drop:+.1f}%)"
            else:
                _ann_txt = f"{-_pct_drop:+.1f}%"
            ax.annotate(
                _ann_txt,
                xy=(_max_row["distance"].values[0], _end_val),
                textcoords="offset points",
                xytext=(12, 0),
                fontsize=7,
                fontweight="bold",
                color=_col,
                va="center",
                ha="left",
            )

    # Autocorrelation overlay: skip when many targets (too many dashed lines)
    _any_autocorr = False
    if not _many:
        for _tgt, _res in targets_with_data.items():
            _ac = _res.get("autocorr_decay")
            if _ac is not None and len(_ac) > 0:
                _col = _TG_COLORS.get(_tgt, "#666666")
                ax.plot(
                    _ac["distance"],
                    _ac["mean"],
                    "s--",
                    color=_col,
                    linewidth=1.2,
                    markersize=4,
                    markeredgecolor=_col,
                    markerfacecolor="white",
                    alpha=0.55,
                    zorder=2,
                )
                _any_autocorr = True

    # Build extra legend handles
    _extra_handles, _extra_labels = [], []
    if _any_autocorr:
        _extra_handles.append(
            Line2D(
                [0],
                [0],
                color="#666666",
                linestyle="--",
                linewidth=1.2,
                marker="s",
                markersize=4,
                markeredgecolor="#666666",
                markerfacecolor="white",
                alpha=0.55,
            )
        )
        _extra_labels.append("Outcome autocorrelation (r\u00b2)")

    ax.set_xlabel("Temporal Distance (waves forward)", fontsize=11, labelpad=8)
    ax.set_ylabel("R\u00b2", fontsize=11, labelpad=8)
    ax.set_title(
        "Temporal Transfer Gradient: Predictive Relationship Decay by Target", fontsize=12, fontweight="bold", pad=12
    )

    # Dynamic x-ticks
    _all_dists = set()
    for _res in targets_with_data.values():
        _all_dists |= set(_res["decay"]["distance"].astype(int).tolist())
    _all_dists = sorted(_all_dists)
    ax.set_xticks(_all_dists)
    ax.set_xticklabels([_DIST_LABELS.get(d, f"{d}") for d in _all_dists], fontsize=9)

    # Legend: sorted by peak performance descending
    handles, labels = ax.get_legend_handles_labels()
    # Separate model handles from SD/autocorr handles
    _model_idx = [
        i for i, l in enumerate(labels) if not l.startswith("\u00b1") and l != "Outcome autocorrelation (r\u00b2)"
    ]
    _other_idx = [i for i in range(len(labels)) if i not in _model_idx]
    _label_to_peak = {}
    for _tgt, _pv in _peak_vals.items():
        _lbase = _TG_LABELS.get(_tgt, _tgt)
        for _l in labels:
            if _l.startswith(_lbase):
                _label_to_peak[_l] = _pv
    _model_sorted = sorted(_model_idx, key=lambda i: _label_to_peak.get(labels[i], 0), reverse=True)
    _final_handles = [handles[i] for i in _model_sorted] + [handles[i] for i in _other_idx] + _extra_handles
    _final_labels = [labels[i] for i in _model_sorted] + [labels[i] for i in _other_idx] + _extra_labels
    _leg_ncol = 2 if _many else 1
    _leg_fs = 7 if _many else 8
    _leg_loc = "upper center" if _many else "lower left"
    _leg_bbox = (0.5, -0.18) if _many else None
    _leg_title = None if _many else "Target (peak R\u00b2 at 1 wave)"
    ax.legend(
        _final_handles,
        _final_labels,
        fontsize=_leg_fs,
        loc=_leg_loc,
        bbox_to_anchor=_leg_bbox,
        framealpha=0.92,
        ncol=_leg_ncol,
        edgecolor="#CCCCCC",
        borderpad=0.6,
        labelspacing=0.3,
        columnspacing=1.0,
        title=_leg_title,
        title_fontsize=7.5,
    )

    ax.tick_params(labelsize=8.5)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ax.grid(axis="x", alpha=0.06, linewidth=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/temporal_gradient_decay_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fig_dir}/temporal_gradient_decay_curves.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # FIGURE 2: Per-target heatmaps (shared colorbar)
    # Shared vmin/vmax across all targets for cross-comparison
    # Forward/backward triangle labels
    _n_targets = len(targets_with_data)
    if _n_targets == 0:
        return

    # Compute shared color range across all targets
    _all_vals = []
    for _res in targets_with_data.values():
        _m = _res["matrix"].astype(float)
        _all_vals.extend(_m.values[~np.isnan(_m.values)].tolist())
    _shared_vmin = min(_all_vals) - 0.01
    _shared_vmax = max(_all_vals) + 0.01

    _ncols = min(3, _n_targets)
    _nrows = int(np.ceil(_n_targets / _ncols))
    fig2, axes2 = plt.subplots(_nrows, _ncols, figsize=(5.0 * _ncols + 0.8, 4.2 * _nrows), dpi=300)
    if _n_targets == 1:
        axes2 = np.array([axes2])
    axes2 = axes2.flatten() if hasattr(axes2, "flatten") else [axes2]

    _heatmap_mappable = None
    for _idx, (_tgt, _res) in enumerate(targets_with_data.items()):
        _ax = axes2[_idx]
        _mat = _res["matrix"].astype(float)
        _col = _TG_COLORS.get(_tgt, "#1565C0")
        _lab = _TG_LABELS.get(_tgt, _tgt)
        _n = _mat.shape[0]

        _map_fn = getattr(_mat, "map", None) or _mat.applymap
        _annot = _map_fn(lambda x: f"{x:.3f}" if not np.isnan(x) else "")

        _hm = sns.heatmap(
            _mat,
            annot=_annot,
            fmt="",
            cmap="RdYlGn",
            ax=_ax,
            vmin=_shared_vmin,
            vmax=_shared_vmax,
            linewidths=1.0,
            linecolor="white",
            cbar=False,
            annot_kws={"fontsize": 8.5, "fontweight": "bold"},
        )
        if _heatmap_mappable is None:
            _heatmap_mappable = _hm.collections[0]

        # Highlight diagonal
        for _d in range(_n):
            _ax.add_patch(plt.Rectangle((_d, _d), 1, 1, fill=False, edgecolor="#333333", linewidth=2.0, zorder=5))

        # Forward/backward labels
        _ax.text(_n - 0.3, 0.3, "\u2192 Fwd", fontsize=6.5, color="#444444", ha="right", va="top", zorder=6)
        _ax.text(0.3, _n - 0.3, "\u2190 Bwd", fontsize=6.5, color="#444444", ha="left", va="bottom", zorder=6)

        _ax.set_title(_lab, fontsize=10.5, fontweight="bold", color=_col, pad=8)
        _ax.tick_params(labelsize=7.5, rotation=0)
        _ax.set_yticklabels(_ax.get_yticklabels(), rotation=0)

    # Hide unused axes
    for _idx in range(len(targets_with_data), len(axes2)):
        axes2[_idx].set_visible(False)

    # Shared colorbar
    if _heatmap_mappable is not None:
        _cbar_ax = fig2.add_axes([0.94, 0.15, 0.015, 0.7])
        fig2.colorbar(_heatmap_mappable, cax=_cbar_ax, label="R\u00b2")
        _cbar_ax.tick_params(labelsize=7.5)

    plt.suptitle(
        "Temporal Transfer Matrices by Target\n(diagonal = same-timepoint CV; off-diagonal = cross-timepoint transfer)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])
    fig2.savefig(f"{fig_dir}/temporal_gradient_heatmaps.png", dpi=300, bbox_inches="tight")
    fig2.savefig(f"{fig_dir}/temporal_gradient_heatmaps.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    logger.info(f"\n  Figures saved to {fig_dir}/")

    # FIGURE 3: Forward vs Backward Asymmetry
    # N-pairs annotation at each distance
    _asym_data = {}
    for _tgt, _res in targets_with_data.items():
        _det_raw = _res.get("details")
        if _det_raw and isinstance(_det_raw, dict):
            import ast as _ast_mod

            _det = {}
            for _k, _v in _det_raw.items():
                try:
                    _kt = _ast_mod.literal_eval(_k) if isinstance(_k, str) else _k
                    _det[_kt] = _v
                except Exception:
                    continue
            if _det:
                _avail = sorted(set(t for (t, _) in _det.keys()) | set(t for (_, t) in _det.keys()))
                _asym = _compute_asymmetry(_det, _avail)
                if len(_asym) > 0:
                    _asym_data[_tgt] = _asym

    if _asym_data:
        _many_asym = len(_asym_data) > 6
        fig3, ax3 = plt.subplots(figsize=(8.5, 5.5), dpi=300)

        ax3.axhline(y=0, color="#BDBDBD", linewidth=1.0, linestyle="-", zorder=0)
        ax3.axhspan(-0.015, 0.015, alpha=0.04, color="#BDBDBD", zorder=0)

        _asym_lw = 1.4 if _many_asym else 2.0
        _asym_ms = 4 if _many_asym else 7
        for _tgt, _asym in _asym_data.items():
            _col = _TG_COLORS.get(_tgt, "#666666")
            _lab = _TG_LABELS.get(_tgt, _tgt)
            ax3.plot(
                _asym["distance"],
                _asym["delta"],
                "s-",
                color=_col,
                label=_lab,
                linewidth=_asym_lw,
                markersize=_asym_ms,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )

        # N-pairs annotations (once per distance, above highest point)
        _all_d = set()
        for _a in _asym_data.values():
            _all_d |= set(_a["distance"].astype(int).tolist())
        _all_d = sorted(_all_d)
        for _d in _all_d:
            _n_at_d = None
            for _a in _asym_data.values():
                _row = _a[_a["distance"] == _d]
                if len(_row):
                    _n_at_d = int(_row["n_forward"].values[0])
                    break
            if _n_at_d is not None:
                _max_delta = max(
                    _a[_a["distance"] == _d]["delta"].values[0]
                    for _a in _asym_data.values()
                    if len(_a[_a["distance"] == _d])
                )
                ax3.annotate(
                    f"n={_n_at_d} pairs",
                    xy=(_d, _max_delta),
                    textcoords="offset points",
                    xytext=(0, 12),
                    fontsize=6.5,
                    ha="center",
                    color="#555555",
                )

        ax3.set_xlabel("Temporal Distance (waves)", fontsize=11, labelpad=8)
        ax3.set_ylabel("Asymmetry (Forward R\u00b2 minus Backward R\u00b2)", fontsize=10, labelpad=8)
        ax3.set_title("Directional Asymmetry in Temporal Transfer", fontsize=12, fontweight="bold", pad=12)

        ax3.set_xticks(_all_d)
        ax3.set_xticklabels([f"{d} wave{'s' if d > 1 else ''}\n(~{d} yr)" for d in _all_d], fontsize=8.5)

        ax3.text(
            0.98,
            0.95,
            "Positive: earlier \u2192 later prediction stronger",
            transform=ax3.transAxes,
            fontsize=7,
            ha="right",
            va="top",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#CCCCCC", linewidth=0.5),
        )
        ax3.text(
            0.98,
            0.05,
            "Negative: later \u2192 earlier prediction stronger",
            transform=ax3.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#CCCCCC", linewidth=0.5),
        )

        _leg3_ncol = 2 if _many_asym else 1
        _leg3_fs = 7 if _many_asym else 8.5
        _leg3_loc = "upper center" if _many_asym else "upper left"
        _leg3_bbox = (0.5, -0.18) if _many_asym else None
        ax3.legend(
            fontsize=_leg3_fs,
            loc=_leg3_loc,
            bbox_to_anchor=_leg3_bbox,
            framealpha=0.92,
            ncol=_leg3_ncol,
            edgecolor="#CCCCCC",
            borderpad=0.6,
            labelspacing=0.3,
        )
        ax3.tick_params(labelsize=8.5)
        ax3.grid(axis="y", alpha=0.12, linewidth=0.5)
        for sp in ["top", "right"]:
            ax3.spines[sp].set_visible(False)
        plt.tight_layout()
        fig3.savefig(f"{fig_dir}/temporal_gradient_asymmetry.png", dpi=300, bbox_inches="tight")
        fig3.savefig(f"{fig_dir}/temporal_gradient_asymmetry.pdf", dpi=300, bbox_inches="tight")
        plt.show()

    # FIGURE 4: Decay Slopes + Autocorrelation Comparison
    # Regression confidence band on 4B
    # Panel labels A, B
    # CI footnote on 4A
    # Spearman rho alongside Pearson r on 4B
    _slopes_df = _compute_decay_slopes(targets_with_data)

    if len(_slopes_df) >= 2:
        _has_autocorr = bool(_AUTOCORR_T0T4) and sum(1 for t in _slopes_df["target"] if t in _AUTOCORR_T0T4) >= 3
        _n_panels = 2 if _has_autocorr else 1
        _widths = [1.2, 1] if _has_autocorr else [1]

        fig4, axes4 = plt.subplots(
            1, _n_panels, figsize=(5.5 * _n_panels, 5.5), dpi=300, gridspec_kw={"width_ratios": _widths}
        )
        if _n_panels == 1:
            axes4 = [axes4]

        # Panel A: Decay slope bar chart
        _ax_a = axes4[0]
        # Panel label
        _ax_a.text(-0.14, 1.03, "A", transform=_ax_a.transAxes, fontsize=14, fontweight="bold", va="top")

        _sorted = _slopes_df.sort_values("slope", ascending=True)
        _n_bars = len(_sorted)
        _bar_h = min(0.6, 4.0 / max(_n_bars, 1))

        _y_pos = np.arange(_n_bars)
        _ax_a.barh(
            _y_pos,
            _sorted["slope"],
            height=_bar_h,
            color=_sorted["color"].values,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        _ax_a.errorbar(
            _sorted["slope"],
            _y_pos,
            xerr=_sorted["slope_se"] * 1.96,
            fmt="none",
            ecolor="#333333",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            zorder=4,
        )

        for _i, (_, _row) in enumerate(_sorted.iterrows()):
            # Place labels to the right of zero: always in clear space
            _ax_a.text(
                0.002,
                _i,
                f"{_row['slope']:.4f}  ({_row['total_drop_pct']:.1f}%)",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
                color=_row["color"],
            )

        _ax_a.axvline(x=0, color="#BDBDBD", linewidth=0.8, zorder=0)
        _ax_a.set_yticks(_y_pos)
        _ax_a.set_yticklabels(_sorted["label"], fontsize=9)
        # Pad right side for labels
        _xmin = min(_sorted["slope"] - _sorted["slope_se"] * 1.96) - 0.003
        _ax_a.set_xlim(left=_xmin, right=0.022)
        _ax_a.set_xlabel("Decay Slope (R\u00b2 per wave, from peak)", fontsize=10, labelpad=8)
        _ax_a.set_title("Rate of Temporal Transfer Decay", fontsize=11, fontweight="bold", pad=10)
        _ax_a.tick_params(labelsize=8.5, axis="y", length=0)
        _ax_a.grid(axis="x", alpha=0.12, linewidth=0.5)
        for sp in ["top", "right"]:
            _ax_a.spines[sp].set_visible(False)

        # CI footnote
        _ax_a.text(
            0.02,
            -0.08,
            "95% CI from linear regression on 3\u20134 distance points",
            transform=_ax_a.transAxes,
            fontsize=7,
            color="#555555",
        )

        # Panel B: Slope vs Autocorrelation scatter
        if _has_autocorr:
            from scipy import stats

            _ax_b = axes4[1]
            # Panel label
            _ax_b.text(-0.14, 1.03, "B", transform=_ax_b.transAxes, fontsize=14, fontweight="bold", va="top")

            _scatter_rows = []
            for _, _row in _slopes_df.iterrows():
                if _row["target"] in _AUTOCORR_T0T4:
                    _scatter_rows.append(
                        {
                            "target": _row["target"],
                            "label": _row["label"],
                            "slope": _row["slope"],
                            "autocorr": _AUTOCORR_T0T4[_row["target"]],
                            "color": _row["color"],
                        }
                    )
                else:
                    logger.info(f"    Skipping {_row['label']} from scatter -- no value in _AUTOCORR_T0T4")
            _sdf = pd.DataFrame(_scatter_rows)

            if len(_sdf) >= 3:
                _ax_b.set_axisbelow(True)

                # Scatter points
                for _, _row in _sdf.iterrows():
                    _ax_b.scatter(
                        _row["autocorr"],
                        _row["slope"],
                        color=_row["color"],
                        s=85,
                        zorder=4,
                        edgecolors="white",
                        linewidths=1.2,
                    )

                # Smart label placement: alternate above/below to reduce overlap
                for _li, (_, _row) in enumerate(_sdf.sort_values("autocorr").iterrows()):
                    _yo = 10 if _li % 2 == 0 else -14
                    _va = "bottom" if _yo > 0 else "top"
                    _ax_b.annotate(
                        _row["label"],
                        xy=(_row["autocorr"], _row["slope"]),
                        textcoords="offset points",
                        xytext=(0, _yo),
                        fontsize=7,
                        color=_row["color"],
                        fontweight="bold",
                        ha="center",
                        va=_va,
                    )

                # Regression line + confidence band
                _r_pears, _p_pears = stats.pearsonr(_sdf["autocorr"], _sdf["slope"])
                _sl, _int, _, _, _se = stats.linregress(_sdf["autocorr"], _sdf["slope"])
                _n_pts = len(_sdf)
                _x_fit = np.linspace(_sdf["autocorr"].min() - 0.03, _sdf["autocorr"].max() + 0.03, 100)
                _y_fit = _sl * _x_fit + _int
                _ax_b.plot(_x_fit, _y_fit, "-", color="#78909C", linewidth=1.3, alpha=0.5, zorder=2)

                # CI band around regression
                if _n_pts >= 3:
                    _x_mean = _sdf["autocorr"].mean()
                    _ss_x = np.sum((_sdf["autocorr"] - _x_mean) ** 2)
                    _s_err = _se * np.sqrt(1.0 / _n_pts + (_x_fit - _x_mean) ** 2 / _ss_x)
                    _t_val = stats.t.ppf(0.975, df=_n_pts - 2)
                    _ax_b.fill_between(
                        _x_fit,
                        _y_fit - _t_val * _s_err,
                        _y_fit + _t_val * _s_err,
                        alpha=0.08,
                        color="#78909C",
                        zorder=1,
                    )

                # Spearman rho alongside Pearson r
                _stat_text = f"r = {_r_pears:.2f}"
                if _n_pts >= 5:
                    _stat_text += f" (p = {_p_pears:.3f})"
                if _n_pts >= 4:
                    _rho, _p_rho = stats.spearmanr(_sdf["autocorr"], _sdf["slope"])
                    _stat_text += f"\n\u03c1 = {_rho:.2f}"

                _ax_b.text(
                    0.05,
                    0.05,
                    _stat_text,
                    transform=_ax_b.transAxes,
                    fontsize=8.5,
                    fontweight="bold",
                    color="#455A64",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#CFD8DC", linewidth=0.5
                    ),
                )

                _ax_b.set_xlabel("T0 to T4 Autocorrelation (r)", fontsize=10, labelpad=8)
                _ax_b.set_ylabel("Decay Slope (R\u00b2 per wave)", fontsize=10, labelpad=8)
                _ax_b.set_title("Decay Rate vs Outcome Stability", fontsize=11, fontweight="bold", pad=10)
                # Axis padding
                _ax_b.set_xlim(_sdf["autocorr"].min() - 0.05, _sdf["autocorr"].max() + 0.05)
                _ax_b.set_ylim(_sdf["slope"].min() - 0.005, _sdf["slope"].max() + 0.005)
                _ax_b.grid(alpha=0.12, linewidth=0.5)
                for sp in ["top", "right"]:
                    _ax_b.spines[sp].set_visible(False)

        plt.tight_layout()
        fig4.savefig(f"{fig_dir}/temporal_gradient_slopes.png", dpi=300, bbox_inches="tight")
        fig4.savefig(f"{fig_dir}/temporal_gradient_slopes.pdf", dpi=300, bbox_inches="tight")
        plt.show()

        # Print slope summary
        logger.info("\n  Decay Slope Summary (R\u00b2 per wave, distance 1 to max):")
        for _, _row in _slopes_df.sort_values("slope").iterrows():
            logger.info(
                f"    {_row['label']:30s}  slope = {_row['slope']:+.4f}  "
                f"(peak {_row['peak_r2']:.3f} -> {_row['end_r2']:.3f}, "
                f"{_row['total_drop_pct']:.1f}% drop)"
            )

    logger.info(f"\n  All figures saved to {fig_dir}/")


# Main body function


def _run_temporal_gradient():
    """Run temporal gradient testing for CURRENT globals."""

    if not ctx.run_temporal_gradient:
        logger.info("  Temporal Gradient Testing disabled.")
        return

    # Gate check
    _skip = False
    for _req in ["across_categories_data", "sample", "create_lightweight_model", "detect_task_type"]:
        if _req not in ctx:
            logger.info(f"  {_req} not found. Run setup first.")
            _skip = True
    if _skip:
        return

    _RS.pop("temporal_gradient", None)

    _target = ctx.target_options if "target_options" in ctx else "target"
    _group = ctx.group if "group" in ctx else None
    _timepoints = [0, 1, 2, 3, 4]

    display(
        HTML(
            '<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #00695C;'
            "background:linear-gradient(135deg,#E0F2F1,#F5F5F5);padding:14px 20px;"
            'margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
            '<span style="font-size:15px;font-weight:700;color:#00695C;">'
            "Temporal Gradient Testing</span><br>"
            '<span style="font-size:12px;color:#546E7A;">'
            "Cross-timepoint transfer matrix: train at each wave, test at every other wave</span>"
            "</div>"
        )
    )

    logger.info(f"\n  Target: {_target}")
    logger.info(f"  Timepoints: {_timepoints}")
    logger.info("")

    matrix, details = _build_transfer_matrix(_target, _timepoints, _group)

    if matrix is None:
        logger.info("  Could not build transfer matrix.")
        return

    # Print matrix
    logger.info(f"\n  Transfer Matrix ({_target}):")
    logger.info(f"  (diagonal entries are {5}-fold CV; off-diagonal are full train→test)")
    logger.info(matrix.to_string(float_format="{:.3f}".format))

    # Print per-cell details
    logger.info("\n  Cell Details:")
    for (_tt, _te), info in sorted(details.items()):
        _cv_tag = " [CV]" if info.get("is_diagonal") else ""
        _sd_tag = f" (SD={info['cv_sd']:.3f})" if "cv_sd" in info else ""
        logger.info(
            f"    Train T{_tt} -> Test T{_te}: R²={info['score']:.4f}{_sd_tag}  "
            f"N_train={info['n_train']:,}  N_test={info['n_test']:,}  "
            f"feats={info['n_features']}{_cv_tag}"
        )

    # Compute decay
    available_tps = sorted(set(t for (t, _) in details.keys()) | set(t for (_, t) in details.keys()))
    decay = _compute_decay_summary(details, available_tps)

    if len(decay) > 0:
        logger.info("\n  Forward-Only Decay Summary:")
        for _, row in decay.iterrows():
            _d = int(row["distance"])
            logger.info(f"    Distance {_d} wave(s): R² = {row['mean']:.4f} (+/- {row['std']:.4f}, n={int(row['n'])})")

    # Store
    _RS["temporal_gradient"] = {
        "target": _target,
        "matrix": matrix.to_dict(),
        "details": {str(k): v for k, v in details.items()},
        "decay": decay.to_dict() if decay is not None else None,
    }

    # Compute autocorrelation decay from sample
    _autocorr_decay = _compute_autocorr_decay(_target, available_tps, _group)
    if _autocorr_decay is not None:
        _RS["temporal_gradient"]["autocorr_decay"] = _autocorr_decay.to_dict()
        logger.info("\n  Autocorrelation Decay (r²):")
        for _, row in _autocorr_decay.iterrows():
            _d = int(row["distance"])
            logger.info(
                f"    Distance {_d} wave(s): r² = {row['mean']:.4f}  (r = {row['mean_r']:.3f}, n={int(row['n'])} pairs)"
            )

    # Single-target figure
    if len(decay) > 1:
        _col = _TG_COLORS.get(_target, "#1565C0")
        _lab = _TG_LABELS.get(_target, _target)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        # Same-TP reference line
        _stp_val = decay[decay["distance"] == 0]["mean"].values
        if len(_stp_val):
            ax.axhline(y=_stp_val[0], color=_col, linewidth=0.8, linestyle=":", alpha=0.4, zorder=1)

        ax.plot(
            decay["distance"],
            decay["mean"],
            "o-",
            color=_col,
            linewidth=2.4,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.3,
            zorder=3,
            label=_lab,
        )
        if decay["std"].sum() > 0:
            ax.fill_between(
                decay["distance"],
                decay["mean"] - decay["std"],
                decay["mean"] + decay["std"],
                alpha=0.12,
                color=_col,
                zorder=1,
            )

        # Annotate scores
        for _, row in decay.iterrows():
            ax.annotate(
                f"{row['mean']:.3f}",
                xy=(row["distance"], row["mean"]),
                textcoords="offset points",
                xytext=(0, 13),
                fontsize=9.5,
                ha="center",
                fontweight="bold",
                color=_col,
            )

        # Dynamic x-ticks
        _dists = sorted(decay["distance"].astype(int).tolist())
        ax.set_xticks(_dists)
        ax.set_xticklabels([_DIST_LABELS.get(d, f"{d}") for d in _dists], fontsize=9)
        ax.set_xlabel("Temporal Distance (waves forward)", fontsize=10, labelpad=8)
        ax.set_ylabel("R²", fontsize=10, labelpad=8)
        ax.set_title(f"Temporal Transfer Gradient: {_lab}", fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
        ax.grid(axis="y", alpha=0.12, linewidth=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()

        _fig_dir = "temporal_gradient_figures"
        os.makedirs(_fig_dir, exist_ok=True)
        fig.savefig(f"{_fig_dir}/temporal_gradient_{_target}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{_fig_dir}/temporal_gradient_{_target}.pdf", dpi=300, bbox_inches="tight")
        plt.show()


def run() -> None:
    """Run this stage (notebook cell 20)."""
    global traceback
    warnings.filterwarnings("ignore")

    # Batch dispatch
    _is_bv_tg = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_tg:
        try:
            _cbv_tg = cache_load("main_runner")
            if (
                _cbv_tg
                and isinstance(_cbv_tg, dict)
                and _cbv_tg.get("_batch_archive")
                and len(_cbv_tg["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_tg["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_tg["_batch_archive"]
                _is_bv_tg = True
        except Exception:
            pass

    if _is_bv_tg:
        _vgs_tg = {
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
                "last_fitted_model",
                "sample_valid",
                "sample_valid_test",
                "sample_valid_train",
                "results_summary",
            ]
        }
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH -- Temporal Gradient: processing {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)

        # Collect all results for combined figure
        _all_gradient_results = {}

        # Deduplicate: run once per unique target
        _seen_targets_tg = set()
        for _vi_tg, (_vk_tg, _vd_tg) in enumerate(ctx._batch_archive.items()):
            _vl_tg = _vd_tg.get("target", _vk_tg)
            if _vl_tg in _seen_targets_tg:
                logger.info(f"\n  Skipping {_vk_tg} (gradient already computed for {_vl_tg})")
                continue
            _seen_targets_tg.add(_vl_tg)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_tg + 1}/{len(ctx._batch_archive)}] Target: {_vl_tg}")
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
                "last_fitted_model",
                "results_summary",
                "sample_valid",
                "sample_valid_test",
                "sample_valid_train",
            ]:
                if _vd_tg.get(_gk) is not None:
                    ctx[_gk] = _vd_tg[_gk]
            ctx["target_options"] = _vl_tg
            if "timepoint" in _vd_tg:
                ctx["tp_option"] = _vd_tg["timepoint"]
            try:
                _run_temporal_gradient()
            except Exception as _e_tg:
                logger.info(f"  Error for {_vl_tg}: {_e_tg}")
                import traceback

                traceback.print_exc()

            # Collect for combined plot
            _rs_ref = ctx.get("_RS")
            if _rs_ref and "temporal_gradient" in _rs_ref:
                _tg_data = _rs_ref["temporal_gradient"]
                _decay_dict = _tg_data.get("decay")
                _mat_dict = _tg_data.get("matrix")
                _all_gradient_results[_vl_tg] = {
                    "decay": pd.DataFrame(_decay_dict) if _decay_dict else None,
                    "matrix": pd.DataFrame(_mat_dict) if _mat_dict else None,
                    "details": _tg_data.get("details"),
                    "autocorr_decay": pd.DataFrame(_tg_data["autocorr_decay"])
                    if _tg_data.get("autocorr_decay")
                    else None,
                }
                # Archive back
                if _vk_tg in ctx._batch_archive:
                    ctx._batch_archive[_vk_tg]["_RS_temporal_gradient"] = copy.deepcopy(_tg_data)

        # Combined multi-target figures (split child / parent)
        if _all_gradient_results:
            _n_ok = len(_all_gradient_results)
            _n_total = len(_seen_targets_tg)
            if _n_ok < _n_total:
                logger.info(
                    f"\n  Warning: Combined figure includes {_n_ok}/{_n_total} targets ({_n_total - _n_ok} failed)"
                )

            # Split into child and parent groups
            _child_results = {k: v for k, v in _all_gradient_results.items() if not k.startswith("parent_")}
            _parent_results = {k: v for k, v in _all_gradient_results.items() if k.startswith("parent_")}

            if _child_results:
                logger.info(f"\n{'=' * 68}")
                logger.info(f"  CHILD TARGETS -- Temporal Gradient ({len(_child_results)} targets)")
                logger.info(f"{'=' * 68}")
                _plot_gradient_results(_child_results, fig_dir="temporal_gradient_figures/child")

            if _parent_results:
                logger.info(f"\n{'=' * 68}")
                logger.info(f"  PARENT TARGETS -- Temporal Gradient ({len(_parent_results)} targets)")
                logger.info(f"{'=' * 68}")
                _plot_gradient_results(_parent_results, fig_dir="temporal_gradient_figures/parent")

            # Combined slopes figure (all targets) if enough for scatter
            if len(_all_gradient_results) >= 4:
                logger.info(f"\n{'=' * 68}")
                logger.info(f"  ALL TARGETS -- Combined Slope Comparison ({len(_all_gradient_results)} targets)")
                logger.info(f"{'=' * 68}")
                _plot_gradient_results(_all_gradient_results, fig_dir="temporal_gradient_figures/combined")

        # Restore globals
        for _k, _v in _vgs_tg.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH COMPLETE -- Temporal Gradient: {len(_seen_targets_tg)} unique targets")
        logger.info("=" * 72)
    else:
        _run_temporal_gradient()
