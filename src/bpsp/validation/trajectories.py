"""Cross-timepoint performance trajectories (the `trajectories` stage, Supplement S1.16.1).

Sweeps T0 to T4, running cross-validated models at each wave to produce
developmental prediction curves. Tracks how model performance evolves
as participants age through the ABCD Study (ages 9 to 13).

Each timepoint uses all available features at that wave (not a common
intersection), so R² reflects real-world predictive capacity at each
age. Feature counts are reported per timepoint for transparency.

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
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from bpsp.context import ctx
from bpsp.data.preprocessing import across_categories_data, create_lightweight_model, detect_task_type
from bpsp.reporting import _RS, cache_load

logger = logging.getLogger(__name__)

# Target colors / labels (shared with the temporal_gradient stage if available)
_TP_TRAJ_COLORS = {
    "depress_D_p": "#1565C0",
    "sbt_core_depression": "#4169E1",
    "anxdisord_D_p": "#C62828",
    "sbt_core_anxiety": "#D32F2F",
    "adhd_D_p": "#E6A817",
    "social_problems_D_p": "#E65100",
    "internal_D_p": "#6A1B9A",
    "external_D_p": "#2E7D32",
    "parent_depress_D_p": "#0D47A1",
    "parent_depressed_p": "#191970",
    "parent_anxdisord_D_p": "#B71C1C",
    "parent_adhd_D_p": "#F57F17",
    "parent_external_D_p": "#1B5E20",
    "parent_internal_D_p": "#4A148C",
    "sbt_anxiety_depression": "#DA70D6",
    "sbt_guilt_hopelessness": "#4B0082",
    "sbt_fatigue_sleep": "#6A5ACD",
    "sbt_somatic_depression": "#BC8F8F",
    "sbt_emotional_dysregulation": "#FF4500",
    "sbt_social_withdrawal": "#5F9EA0",
    "sbt_avoidance_fear": "#9370DB",
    "sbt_aggression_irritability": "#B22222",
    "sbt_academic_cognitive": "#B8960B",
    "sbt_perfectionism_achievement": "#C44A80",
}
_TP_TRAJ_LABELS = {
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


def _get_target_color(tgt):
    if "_TG_COLORS" in ctx:
        return ctx["_TG_COLORS"].get(tgt, _TP_TRAJ_COLORS.get(tgt, "#1a73e8"))
    return _TP_TRAJ_COLORS.get(tgt, "#1a73e8")


def _get_target_label(tgt):
    if "_TG_LABELS" in ctx:
        return ctx["_TG_LABELS"].get(tgt, _TP_TRAJ_LABELS.get(tgt, tgt))
    return _TP_TRAJ_LABELS.get(tgt, tgt)


def _run_tp_trajectories():
    """Run cross-timepoint performance trajectories for CURRENT globals."""

    _skip = False
    for _req in ["X_train", "X_test", "y_train", "y_test"]:
        if _req not in ctx:
            logger.info(f"  {_req} not found. Run the Main Analysis Runner first.")
            _skip = True
    if "across_categories_data" not in ctx:
        logger.info("  across_categories_data not found -- skipping TP trajectories.")
        _skip = True
    if "sample" not in ctx:
        logger.info("  sample not found. Run setup first.")
        _skip = True

    if _skip:
        return

    _RS.pop("tp_trajectory", None)

    _target = ctx.target_options if "target_options" in ctx else "target"
    _group = ctx.group if "group" in ctx else None
    _timepoints = [0, 1, 2, 3, 4]
    _n_cv = ctx.VAL_N_OUTER if "VAL_N_OUTER" in ctx else 5
    _tgt_color = _get_target_color(_target)
    _tgt_label = _get_target_label(_target)

    display(
        HTML(
            '<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #1565C0;'
            "background:linear-gradient(135deg,#E3F2FD,#F5F5F5);padding:14px 20px;"
            'margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
            '<span style="font-size:15px;font-weight:700;color:#1565C0;">'
            "Cross-Timepoint Performance Trajectories</span><br>"
            '<span style="font-size:12px;color:#546E7A;">'
            "Sweeping T0 to T4 for developmental prediction curves</span>"
            "</div>"
        )
    )

    logger.info(f"\n  Target: {_tgt_label} ({_target})")
    logger.info(f"  Group: {_group or 'All'}")
    logger.info(f"  CV: {_n_cv}-fold")
    logger.info("")

    _tp_results = []

    for _t in _timepoints:
        try:
            _X_tr, _X_te, _y_tr, _y_te = across_categories_data(
                _t, [_target], group=_group, scale_x=True, scale_y=False, verbose=False
            )

            _X_all = pd.concat([_X_tr, _X_te], axis=0).reset_index(drop=True)
            _y_all = np.concatenate([np.asarray(_y_tr).ravel(), np.asarray(_y_te).ravel()])

            _mask = ~np.isnan(_y_all.astype(float))
            _X_all = _X_all[_mask]
            _y_all = _y_all[_mask]

            if len(_y_all) < _n_cv * 2:
                logger.info(f"  T{_t}: SKIPPED (N={len(_y_all)} < minimum)")
                continue

            _task = detect_task_type(_y_all)
            _is_clf = _task == "classification"
            _metric = "AUC" if _is_clf else "R\u00b2"

            if _is_clf:
                _cv = StratifiedKFold(n_splits=_n_cv, shuffle=True, random_state=42)
            else:
                _cv = KFold(n_splits=_n_cv, shuffle=True, random_state=42)

            _fold_scores = []
            for _train_idx, _test_idx in _cv.split(_X_all, _y_all):
                _X_fold_tr = _X_all.iloc[_train_idx].copy()
                _X_fold_te = _X_all.iloc[_test_idx].copy()

                _imp = SimpleImputer(strategy="median")
                _X_fold_tr = pd.DataFrame(
                    _imp.fit_transform(_X_fold_tr), columns=_X_fold_tr.columns, index=_X_fold_tr.index
                )
                _X_fold_te = pd.DataFrame(
                    _imp.transform(_X_fold_te), columns=_X_fold_te.columns, index=_X_fold_te.index
                )

                _m = create_lightweight_model(is_classification=_is_clf)
                _m.fit(_X_fold_tr, _y_all[_train_idx])
                if _is_clf:
                    _pred = _m.predict_proba(_X_fold_te)[:, 1]
                    _fold_scores.append(roc_auc_score(_y_all[_test_idx], _pred))
                else:
                    _pred = _m.predict(_X_fold_te)
                    _fold_scores.append(r2_score(_y_all[_test_idx], _pred))

            _mean = np.mean(_fold_scores)
            _std = np.std(_fold_scores, ddof=1)
            _se = _std / np.sqrt(_n_cv)
            _ci_half = stats.t.ppf(0.975, df=_n_cv - 1) * _se

            _tp_results.append(
                {
                    "timepoint": _t,
                    "n_subjects": len(_X_all),
                    "n_features": _X_all.shape[1],
                    "mean": _mean,
                    "std": _std,
                    "ci_lo": _mean - _ci_half,
                    "ci_hi": _mean + _ci_half,
                    "metric": _metric,
                    "fold_scores": _fold_scores,
                }
            )

            logger.info(
                f"  T{_t}: {_metric} = {_mean:.4f} \u00b1 {_ci_half:.4f}  (N={len(_X_all):,}, {_X_all.shape[1]} feats)"
            )

        except Exception as _e:
            logger.info(f"  T{_t}: SKIPPED ({_e})")

    # Publication figure
    if len(_tp_results) >= 2:
        _df = pd.DataFrame(_tp_results)

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

        _x = _df["timepoint"].values
        _y = _df["mean"].values
        _lo_ci = _df["ci_lo"].values
        _hi_ci = _df["ci_hi"].values

        ax.fill_between(_x, _lo_ci, _hi_ci, alpha=0.12, color=_tgt_color, zorder=2)
        ax.plot(_x, _y, "-", color=_tgt_color, linewidth=2.0, zorder=3)
        ax.plot(_x, _y, "o", color="white", markersize=8, markeredgecolor=_tgt_color, markeredgewidth=1.8, zorder=4)

        for _i, _row in _df.iterrows():
            ax.annotate(
                f"{_row['mean']:.3f}",
                xy=(_row["timepoint"], _row["ci_hi"]),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=9.5,
                ha="center",
                va="bottom",
                fontweight="bold",
                color=_tgt_color,
            )
            ax.annotate(
                f"N={_row['n_subjects']:,}  ({_row['n_features']} feats)",
                xy=(_row["timepoint"], _row["ci_lo"]),
                textcoords="offset points",
                xytext=(0, -12),
                fontsize=7,
                ha="center",
                va="top",
                color="#555555",
            )

        if len(_x) >= 3:
            _slope, _intercept, _rval, _pval, _stderr = stats.linregress(_x, _y)
            _trend_x = np.linspace(_x.min(), _x.max(), 50)
            _trend_y = _slope * _trend_x + _intercept
            ax.plot(_trend_x, _trend_y, "--", color="#AAAAAA", linewidth=1.0, alpha=0.6, zorder=2)
            _slope_ci_lo = _slope - stats.t.ppf(0.975, df=len(_x) - 2) * _stderr
            _slope_ci_hi = _slope + stats.t.ppf(0.975, df=len(_x) - 2) * _stderr
            ax.text(
                0.98,
                0.04,
                f"Slope: {_slope:.4f} [{_slope_ci_lo:.4f}, {_slope_ci_hi:.4f}] per wave",
                transform=ax.transAxes,
                fontsize=7.5,
                ha="right",
                va="bottom",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc", linewidth=0.5),
            )

        _tp_labels = {
            0: "Baseline\n(9-10y)",
            1: "1-Year\n(10-11y)",
            2: "2-Year\n(11-12y)",
            3: "3-Year\n(12-13y)",
            4: "4-Year\n(13-14y)",
        }
        _available_tps = sorted(set(r["timepoint"] for r in _tp_results))
        ax.set_xticks(_available_tps)
        ax.set_xticklabels([_tp_labels.get(t, f"T{t}") for t in _available_tps], fontsize=8.5)
        ax.set_xlabel("ABCD Study Wave", fontsize=9.5)
        ax.set_ylabel(f"Cross-Validated {_df.iloc[0]['metric']}", fontsize=9.5)
        ax.set_title(f"Predictive Performance Across Development \u2014 {_tgt_label}", fontsize=10.5, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        plt.tight_layout()

        try:
            _fig_dir = os.path.join(os.getcwd(), "tp_trajectory_figures")
            os.makedirs(_fig_dir, exist_ok=True)
            fig.savefig(os.path.join(_fig_dir, f"tp_trajectory_{_target}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(_fig_dir, f"tp_trajectory_{_target}.pdf"), dpi=300, bbox_inches="tight")
        except Exception as _e:
            logger.info(f"Warning: {_e}")
        plt.show()

        # Quantitative summary
        if len(_df) >= 2:
            _slope_sum = np.polyfit(_df["timepoint"], _df["mean"], 1)[0]
            _mean_r2 = _df["mean"].mean()
            _range_r2 = _df["mean"].max() - _df["mean"].min()
            _cv_coeff = (_df["mean"].std() / _mean_r2 * 100) if _mean_r2 > 0 else 0
            _best = _df.loc[_df["mean"].idxmax()]
            _worst = _df.loc[_df["mean"].idxmin()]
            _feat_range = f"{_df['n_features'].min()}\u2013{_df['n_features'].max()}"

            logger.info(f"\n  Summary for {_tgt_label}:")
            logger.info(f"    Mean R\u00b2 across waves:  {_mean_r2:.4f}")
            logger.info(
                f"    Range:                 {_df['mean'].min():.4f}\u2013"
                f"{_df['mean'].max():.4f} (\u0394={_range_r2:.4f})"
            )
            logger.info(f"    CV (stability):        {_cv_coeff:.1f}%")
            logger.info(f"    Slope:                 {_slope_sum:+.4f} R\u00b2/wave")
            logger.info(f"    Peak:                  T{int(_best['timepoint'])} ({_best['mean']:.4f})")
            logger.info(f"    Nadir:                 T{int(_worst['timepoint'])} ({_worst['mean']:.4f})")
            logger.info(f"    Feature range:         {_feat_range} across waves")

        _RS["tp_trajectory"] = {
            "results": _df.to_dict(),
            "target": _target,
            "label": _tgt_label,
            "color": _tgt_color,
        }
    else:
        logger.info("\n  Fewer than 2 timepoints completed -- cannot plot trajectory.")


def _plot_combined_trajectories(all_traj, fig_dir="tp_trajectory_figures"):
    """Combined multi-target overlay after batch run."""
    if not all_traj or len(all_traj) < 2:
        return

    _many = len(all_traj) > 6
    _lw = 1.6 if _many else 2.2
    _ms = 4 if _many else 7

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)

    for _tgt, _info in all_traj.items():
        _df = _info.get("df")
        if _df is None or len(_df) < 2:
            continue
        _col = _info.get("color", "#666666")
        _lab = _info.get("label", _tgt)

        ax.plot(
            _df["timepoint"],
            _df["mean"],
            "o-",
            color=_col,
            label=_lab,
            linewidth=_lw,
            markersize=_ms,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        if not _many:
            ax.fill_between(_df["timepoint"], _df["ci_lo"], _df["ci_hi"], alpha=0.08, color=_col, zorder=1)

    _tp_labels = {
        0: "Baseline\n(9-10y)",
        1: "1-Year\n(10-11y)",
        2: "2-Year\n(11-12y)",
        3: "3-Year\n(12-13y)",
        4: "4-Year\n(13-14y)",
    }
    _all_tps = set()
    for _info in all_traj.values():
        _df = _info.get("df")
        if _df is not None:
            _all_tps |= set(_df["timepoint"].astype(int).tolist())
    _all_tps = sorted(_all_tps)
    ax.set_xticks(_all_tps)
    ax.set_xticklabels([_tp_labels.get(t, f"T{t}") for t in _all_tps], fontsize=8.5)

    ax.set_xlabel("ABCD Study Wave", fontsize=10, labelpad=8)
    ax.set_ylabel("Cross-Validated R\u00b2", fontsize=10, labelpad=8)
    ax.set_title("Concurrent Predictive Performance Across Development", fontsize=12, fontweight="bold", pad=12)

    _leg_ncol = 2 if _many else 1
    _leg_fs = 7 if _many else 8.5
    _leg_loc = "upper center" if _many else "lower left"
    _leg_bbox = (0.5, -0.15) if _many else None
    ax.legend(
        fontsize=_leg_fs,
        loc=_leg_loc,
        bbox_to_anchor=_leg_bbox,
        framealpha=0.92,
        ncol=_leg_ncol,
        edgecolor="#CCCCCC",
        borderpad=0.6,
        labelspacing=0.3,
    )

    ax.tick_params(labelsize=8.5)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "tp_trajectory_combined.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "tp_trajectory_combined.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"\n  Combined figure saved to {fig_dir}/")


def run() -> None:
    """Run this stage (notebook cell 19)."""
    global traceback
    warnings.filterwarnings("ignore")

    # Batch dispatch
    _is_bv_tp = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_tp:
        try:
            _cbv_tp = cache_load("main_runner")
            if (
                _cbv_tp
                and isinstance(_cbv_tp, dict)
                and _cbv_tp.get("_batch_archive")
                and len(_cbv_tp["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_tp["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_tp["_batch_archive"]
                _is_bv_tp = True
        except Exception:
            pass  # non-critical; logged upstream or optional visualization

    if _is_bv_tp:
        _vgs_tp = {
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
        logger.info(f"  BATCH -- TP Trajectories: processing {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)

        _all_traj_results = {}

        _seen_targets_tp = set()
        for _vi_tp, (_vk_tp, _vd_tp) in enumerate(ctx._batch_archive.items()):
            _vl_tp = _vd_tp.get("target", _vk_tp)
            if _vl_tp in _seen_targets_tp:
                logger.info(f"\n  Skipping {_vk_tp} -- trajectory already computed for {_vl_tp}")
                continue
            _seen_targets_tp.add(_vl_tp)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_tp + 1}/{len(ctx._batch_archive)}] Target: {_vl_tp}")
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
                if _vd_tp.get(_gk) is not None:
                    ctx[_gk] = _vd_tp[_gk]
            ctx["target_options"] = _vl_tp
            if "timepoint" in _vd_tp:
                ctx["tp_option"] = _vd_tp["timepoint"]
            try:
                _run_tp_trajectories()
            except Exception as _e_tp:
                logger.info(f"  Error for {_vl_tp}: {_e_tp}")
                import traceback

                traceback.print_exc()

            # Collect for combined figure
            _rs_ref = ctx.get("_RS")
            if _rs_ref and "tp_trajectory" in _rs_ref:
                _traj_data = _rs_ref["tp_trajectory"]
                _traj_df = pd.DataFrame(_traj_data.get("results", {}))
                if len(_traj_df) >= 2:
                    _all_traj_results[_vl_tp] = {
                        "df": _traj_df,
                        "label": _traj_data.get("label", _vl_tp),
                        "color": _traj_data.get("color", "#666666"),
                    }
                if _vk_tp in ctx._batch_archive:
                    ctx._batch_archive[_vk_tp]["_RS_tp_trajectory"] = copy.deepcopy(_traj_data)

        # Combined multi-target figures
        if len(_all_traj_results) >= 2:
            _child_traj = {k: v for k, v in _all_traj_results.items() if not k.startswith("parent_")}
            _parent_traj = {k: v for k, v in _all_traj_results.items() if k.startswith("parent_")}

            if len(_child_traj) >= 2:
                logger.info(f"\n{'=' * 68}")
                logger.info(f"  COMBINED -- Child Targets ({len(_child_traj)})")
                logger.info(f"{'=' * 68}")
                _plot_combined_trajectories(_child_traj, fig_dir="tp_trajectory_figures/child")

            if len(_parent_traj) >= 2:
                logger.info(f"\n{'=' * 68}")
                logger.info(f"  COMBINED -- Parent Targets ({len(_parent_traj)})")
                logger.info(f"{'=' * 68}")
                _plot_combined_trajectories(_parent_traj, fig_dir="tp_trajectory_figures/parent")

            if len(_all_traj_results) >= 4:
                logger.info(f"\n{'=' * 68}")
                logger.info(f"  COMBINED -- All Targets ({len(_all_traj_results)})")
                logger.info(f"{'=' * 68}")
                _plot_combined_trajectories(_all_traj_results, fig_dir="tp_trajectory_figures")

        for _k, _v in _vgs_tp.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH COMPLETE -- TP Trajectories: {len(_seen_targets_tp)} unique targets")
        logger.info("=" * 72)
    else:
        _run_tp_trajectories()
