"""PCA baseline and temporal hold-out (the `baselines` stage, Supplement S1.3, S1.6).

PCA baseline tests whether predictive gradients persist when replacing
original features with principal components, ruling out that
high-dimensional subjective feature sets benefit from more degrees of
freedom. Temporal hold-out trains on pooled concurrent earlier-wave data
and tests on later waves without retraining, confirming feature-outcome
relationships transfer across developmental stages.

Each analysis runs independently. The stage requires the `main` stage of
the same run.
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from bpsp.context import ctx
from bpsp.data.preprocessing import across_categories_data, bootstrap_ci, create_lightweight_model, detect_task_type
from bpsp.reporting import _RS, cache_load

logger = logging.getLogger(__name__)

# Main body function (wrapped for batch dispatch)


def _run_for_current_target():
    """Run supplementary validation analyses for current globals."""

    # Common gate check
    _COMMON_SKIP = False
    for _req in ["X_train", "X_test", "y_train", "y_test"]:
        if _req not in ctx:
            logger.info(f"Error: {_req} not found. Run the Main Analysis Runner first.")
            _COMMON_SKIP = True
    # Early check for bootstrap_ci to avoid late NameError after expensive compute
    if "bootstrap_ci" not in ctx:
        logger.info("Warning: bootstrap_ci not found. Run setup first.")
        _COMMON_SKIP = True

    if not _COMMON_SKIP:
        # Clear stale _RS sub-keys from prior batch iteration
        for _rk_clear in ["pca_baseline", "temporal_holdout"]:
            _RS.pop(_rk_clear, None)

        _target_name = ctx.target_options if "target_options" in ctx else "target"
        _tp = ctx.tp_option if "tp_option" in ctx else "?"
        _task = detect_task_type(ctx.y_train)
        _is_clf = _task == "classification"
        _metric = "AUC" if _is_clf else "R²"
        _metric_fn = roc_auc_score if _is_clf else r2_score

        logger.info("=" * 72)
        logger.info("   VALIDATION HUB")
        logger.info(f"  Target: {_target_name}  |  Timepoint: T{_tp}  |  Task: {_task}")
        logger.info("=" * 72)

        _analyses_selected = []
        if ctx.run_pca_baseline:
            _analyses_selected.append("PCA Baseline")
        if ctx.run_temporal_holdout:
            _analyses_selected.append("Temporal Hold-Out")

        logger.info(f"  Running: {', '.join(_analyses_selected) if _analyses_selected else 'None selected'}")
        logger.info("")

        # Analysis 1: PCA baseline comparison
        if ctx.run_pca_baseline:
            display(
                HTML(
                    '<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #1a73e8;'
                    "background:linear-gradient(135deg,#E3F2FD,#F5F5F5);padding:14px 20px;"
                    'margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
                    '<span style="font-size:15px;font-weight:700;color:#1a73e8;">PCA Baseline Comparison</span><br>'
                    '<span style="font-size:12px;color:#546E7A;">Supplementary: Data-Driven vs Theory-Driven Predictive Setup</span>'
                    "</div>"
                )
            )

            _target_name = ctx.target_options if "target_options" in ctx else "target"
            _tp = ctx.tp_option if "tp_option" in ctx else "?"
            _task = detect_task_type(ctx.y_train)
            _is_clf = _task == "classification"

            logger.info(f"\n  Target: {_target_name}  |  Timepoint: T{_tp}  |  Task: {_task}")
            logger.info(
                f"  Features: {ctx.X_train.shape[1]}  |  N_train: {len(ctx.X_train)}  |  N_test: {len(ctx.X_test)}"
            )

            # 1. Determine optimal PCA components via explained variance
            _max_components = min(ctx.X_train.shape[0], ctx.X_train.shape[1])
            _pca_full = PCA(n_components=_max_components, random_state=42)
            _pca_full.fit(ctx.X_train)
            _cumvar = np.cumsum(_pca_full.explained_variance_ratio_)

            # Pick n_components at 90%, 95%, 99% thresholds
            _thresholds = {0.90: None, 0.95: None, 0.99: None}
            for _t in _thresholds:
                _idx = np.searchsorted(_cumvar, _t)
                _thresholds[_t] = min(_idx + 1, _max_components)

            logger.info("\n  PCA variance explained:")
            for _t, _n in _thresholds.items():
                logger.info(f"    {_t * 100:.0f}% → {_n} components")

            # 2. Sweep over component counts
            _n_components_list = sorted(set([5, 10, 20, 50] + list(_thresholds.values()) + [_max_components]))
            _n_components_list = [n for n in _n_components_list if n <= _max_components]

            _results = []
            for _nc in _n_components_list:
                _pca = PCA(n_components=_nc, random_state=42)
                _X_tr_pca = _pca.fit_transform(ctx.X_train)
                _X_te_pca = _pca.transform(ctx.X_test)

                _model = create_lightweight_model(is_classification=_is_clf)
                _y_tr = np.asarray(ctx.y_train).ravel()
                _y_te = np.asarray(ctx.y_test).ravel()
                _model.fit(_X_tr_pca, _y_tr)

                if _is_clf:
                    _y_pred = _model.predict_proba(_X_te_pca)[:, 1]
                    _score = roc_auc_score(_y_te, _y_pred)
                    _metric_name = "AUC"
                else:
                    _y_pred = _model.predict(_X_te_pca)
                    _score = r2_score(_y_te, _y_pred)
                    _metric_name = "R²"

                _results.append(
                    {
                        "n_components": _nc,
                        "var_explained": _cumvar[min(_nc - 1, len(_cumvar) - 1)],
                        _metric_name: _score,
                    }
                )
                logger.info(
                    f"    PCA-{_nc:>3d} ({_cumvar[min(_nc - 1, len(_cumvar) - 1)] * 100:.1f}% var) → {_metric_name} = {_score:.4f}"
                )

            _results_df = pd.DataFrame(_results)

            # 3. Get the main pipeline score for comparison
            # results_summary is a LIST of dicts; also check metrics_dict_full
            _pipeline_score = None
            try:
                # Correct key extraction: metrics_dict_full['metrics'] is a list of
                # dicts keyed by model name (e.g. {'meta_model': {'r2': .., 'rmse': ..}, ...})
                if "metrics_dict_full" in ctx and isinstance(ctx.metrics_dict_full, dict):
                    for _cat_mets in ctx.metrics_dict_full.get("metrics", []):
                        if isinstance(_cat_mets, dict):
                            _mm = _cat_mets.get("meta_model", _cat_mets.get("ensemble", {}))
                            if isinstance(_mm, dict) and _mm:
                                _pipeline_score = _mm.get("r2") or _mm.get("auc")
                                if _pipeline_score is not None:
                                    break
                if _pipeline_score is None and "results_summary" in ctx and isinstance(ctx.results_summary, list):
                    for _rs in ctx.results_summary:
                        if isinstance(_rs, dict):
                            _pipeline_score = (
                                _rs.get("R2") or _rs.get("ROC-AUC") or _rs.get("test_r2") or _rs.get("test_auc")
                            )
                            if _pipeline_score is not None:
                                break
            except Exception:
                pass  # non-critical; logged upstream or optional visualization

            _cohens_f2 = None
            # 3b. Effect size: Cohen's f² for R² improvement
            if _pipeline_score is not None and not _is_clf:
                _best_pca_r2 = max((r.get("R²", r.get("AUC", 0)) for r in _results), default=0)
                _denom = max(1 - max(_pipeline_score, _best_pca_r2), 0.001)
                _cohens_f2 = abs(_pipeline_score - _best_pca_r2) / _denom
                _eff = "small" if _cohens_f2 < 0.15 else ("medium" if _cohens_f2 < 0.35 else "large")
                logger.info(f"\n  Pipeline R² = {_pipeline_score:.4f}  |  Best PCA R² = {_best_pca_r2:.4f}")
                logger.info(
                    f"  ΔR² = {_pipeline_score - _best_pca_r2:+.4f}  |  Cohen's f² = {_cohens_f2:.3f} ({_eff} effect)"
                )
            elif _pipeline_score is not None:
                logger.info(f"\n  Pipeline AUC = {_pipeline_score:.4f}")

            # 4. Publication figure
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

            # Hero: PCA performance curve with pipeline reference
            _x_vals = _results_df["n_components"]
            _y_vals = _results_df[_metric_name]

            # Shaded gap between pipeline and PCA curve
            if _pipeline_score is not None:
                ax.fill_between(_x_vals, _y_vals, _pipeline_score, alpha=0.06, color="#d93025", zorder=1)
                ax.axhline(
                    y=_pipeline_score,
                    color="#d93025",
                    linestyle="--",
                    linewidth=1.5,
                    zorder=2,
                    label=f"Theory-Driven Pipeline ({_metric_name}={_pipeline_score:.4f})",
                )

            ax.plot(
                _x_vals,
                _y_vals,
                "o-",
                color="#1a73e8",
                markersize=6,
                linewidth=2.0,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label="PCA Baseline",
                zorder=3,
            )

            # Best PCA annotation
            _best_idx = _y_vals.idxmax()
            _best_n = _results_df.loc[_best_idx, "n_components"]
            _best_score = _y_vals[_best_idx]
            ax.annotate(
                f"Best: {_best_score:.4f}\n({int(_best_n)} PCs)",
                xy=(_best_n, _best_score),
                fontsize=9,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
                textcoords="offset points",
                xytext=(18, -18),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
            )

            # Score annotations on each point
            for _, _row in _results_df.iterrows():
                if _row["n_components"] != _best_n:
                    ax.text(
                        _row["n_components"],
                        _row[_metric_name] - 0.012,
                        f"{_row[_metric_name]:.3f}",
                        ha="center",
                        fontsize=7,
                        color="#546E7A",
                        alpha=0.8,
                    )

            ax.set_xlabel("Number of PCA Components", fontsize=9.5)
            ax.set_ylabel(_metric_name, fontsize=9.5)
            ax.set_title(f"PCA Baseline Comparison -- {_target_name} (T{_tp})", fontsize=10.5, fontweight="bold")
            ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor="#cccccc")
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.15, linewidth=0.5)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)

            # Inset: Scree curve (top-left corner)
            _inset_ax = ax.inset_axes([0.06, 0.55, 0.28, 0.38])
            _pca_x = range(1, len(_cumvar) + 1)
            _inset_ax.fill_between(_pca_x, 0, _cumvar, alpha=0.1, color="#1a73e8")
            _inset_ax.plot(_pca_x, _cumvar, color="#1a73e8", linewidth=1.0)
            _threshold_colors = {0.90: "#e8710a", 0.95: "#d93025", 0.99: "#7b1fa2"}
            for _t_val, _n_comp in _thresholds.items():
                _tc = _threshold_colors.get(_t_val, "#999999")
                _inset_ax.axhline(y=_t_val, color=_tc, linestyle=":", linewidth=0.6, alpha=0.4)
                _inset_ax.scatter([_n_comp], [_t_val], color=_tc, s=12, zorder=4)
            _inset_ax.set_title("Var. explained", fontsize=7, color="#78909C", pad=2)
            _inset_ax.tick_params(labelsize=5.5)
            _inset_ax.set_ylim(0, 1.05)
            for sp in _inset_ax.spines.values():
                sp.set_linewidth(0.5)
            for sp in ["top", "right"]:
                _inset_ax.spines[sp].set_visible(False)

            plt.tight_layout()

            # Save
            try:
                _fig_dir = os.path.join(os.getcwd(), "pca_baseline_figures")
                os.makedirs(_fig_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(_fig_dir, f"pca_baseline_{_target_name}_T{_tp}.png"), dpi=300, bbox_inches="tight"
                )
                fig.savefig(
                    os.path.join(_fig_dir, f"pca_baseline_{_target_name}_T{_tp}.pdf"), dpi=300, bbox_inches="tight"
                )
                logger.info(f"\n   Saved to {_fig_dir}/")
            except Exception as _e:
                logger.info(f"\n  Warning: Could not save figure: {_e}")

            plt.show()
            # 5. Summary
            if _pipeline_score is not None:
                _delta = _pipeline_score - _best_score
                _pct = (_delta / abs(_best_score) * 100) if _best_score != 0 else float("inf")
                _pca_color = "#2E7D32" if _delta > 0 else "#C62828"
                display(
                    HTML(
                        f'<div style="font-family:sans-serif;max-width:700px;border:2px solid {_pca_color};'
                        f'border-radius:8px;overflow:hidden;margin:16px 0;">'
                        f'<div style="background:{_pca_color};color:white;padding:12px 18px;font-size:14px;font-weight:700;">'
                        f"PCA Baseline Comparison -- Summary</div>"
                        f'<div style="padding:16px 20px;line-height:1.8;font-size:13px;">'
                        f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
                        f'<div><span style="color:#78909C;font-size:11px;text-transform:uppercase;">Pipeline {_metric_name}</span>'
                        f'<br><span style="font-size:20px;font-weight:700;color:#1565C0;">{_pipeline_score:.4f}</span></div>'
                        f'<div><span style="color:#78909C;font-size:11px;text-transform:uppercase;">Best PCA {_metric_name}</span>'
                        f'<br><span style="font-size:20px;font-weight:700;">{_best_score:.4f}</span>'
                        f'<span style="font-size:12px;color:#78909C;"> ({int(_best_n)} components)</span></div>'
                        f'<div><span style="color:#78909C;font-size:11px;text-transform:uppercase;">Improvement</span>'
                        f'<br><span style="font-size:20px;font-weight:700;color:{_pca_color};">Δ{_metric_name} = {_delta:+.4f} ({_pct:+.1f}%)</span></div>'
                        f"</div></div></div>"
                    )
                )
            else:
                logger.info("\n  Warning: pipeline score not found. Run the main stage first for comparison.")

            # Store for dashboard
            _RS["pca_baseline"] = {
                "results": _results_df.to_dict(),
                "pipeline_score": _pipeline_score,
                "best_pca_score": _best_score,
                "best_n_components": int(_best_n),
                "metric": _metric_name,
                "target": _target_name,
                "timepoint": _tp,
                "cohens_f2": _cohens_f2,
                "delta": (_pipeline_score - _best_score) if _pipeline_score is not None else None,
            }

            # 6. PCA Component Loadings
            # Top features per principal component with publication-quality
            # heatmap and summary table. Uses DISPLAY_NAMES from preprocessing for
            # readable labels when available.
            _N_PCS_SHOW = 5
            _N_TOP_FEAT = 10

            _feat_names = list(ctx.X_train.columns)
            _dn = ctx.get("DISPLAY_NAMES", {})

            # Build loadings DataFrame
            _loading_rows = []
            for _pc_i in range(_N_PCS_SHOW):
                _pc_loadings = _pca_full.components_[_pc_i]
                _top_idx = np.argsort(np.abs(_pc_loadings))[::-1][:_N_TOP_FEAT]
                for _rank, _j in enumerate(_top_idx, 1):
                    _raw = _feat_names[_j]
                    _loading_rows.append(
                        {
                            "PC": _pc_i + 1,
                            "Rank": _rank,
                            "Variable": _raw,
                            "Display Name": _dn.get(_raw, _raw),
                            "Loading": _pc_loadings[_j],
                            "Abs Loading": abs(_pc_loadings[_j]),
                            "Var Explained (%)": round(_pca_full.explained_variance_ratio_[_pc_i] * 100, 2),
                        }
                    )
            _loadings_df = pd.DataFrame(_loading_rows)

            # Console summary
            logger.info(f"\n  PCA Component Loadings -- Top {_N_TOP_FEAT} features × {_N_PCS_SHOW} PCs")
            logger.info("  " + "-" * 66)
            for _pc_i in range(_N_PCS_SHOW):
                _pc_sub = _loadings_df[_loadings_df["PC"] == _pc_i + 1]
                _var_pct = _pc_sub["Var Explained (%)"].iloc[0]
                logger.info(f"\n  PC{_pc_i + 1} ({_var_pct:.1f}% variance):")
                for _, _r in _pc_sub.iterrows():
                    logger.info(f"    {_r['Display Name']:45s} {_r['Loading']:+.3f}")

            # Publication heatmap
            _heat_data = _loadings_df.pivot(index="Rank", columns="PC", values="Loading")
            _heat_labels = _loadings_df.pivot(index="Rank", columns="PC", values="Display Name")

            fig_load, ax_load = plt.subplots(figsize=(10, 7), dpi=300)

            _abs_max = max(abs(_heat_data.min().min()), abs(_heat_data.max().max()))
            _im = ax_load.imshow(
                _heat_data.values, cmap="RdBu_r", aspect="auto", vmin=-_abs_max, vmax=_abs_max, interpolation="nearest"
            )

            # Annotate cells with feature name + loading
            for _ri in range(_heat_data.shape[0]):
                for _ci in range(_heat_data.shape[1]):
                    _val = _heat_data.values[_ri, _ci]
                    _name = _heat_labels.values[_ri, _ci]
                    # Truncate long names
                    _name_short = (_name[:22] + "..") if len(str(_name)) > 24 else _name
                    _txt_color = "white" if abs(_val) > _abs_max * 0.6 else "#333333"
                    ax_load.text(
                        _ci,
                        _ri,
                        f"{_name_short}\n{_val:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=_txt_color,
                        fontweight="medium",
                        linespacing=1.4,
                    )

            # Axes
            ax_load.set_xticks(range(_N_PCS_SHOW))
            _pc_xlabels = [
                f"PC{i + 1}\n({_pca_full.explained_variance_ratio_[i] * 100:.1f}%)" for i in range(_N_PCS_SHOW)
            ]
            ax_load.set_xticklabels(_pc_xlabels, fontsize=9, fontweight="bold")
            ax_load.set_yticks(range(_N_TOP_FEAT))
            ax_load.set_yticklabels([f"{i + 1}" for i in range(_N_TOP_FEAT)], fontsize=8)
            ax_load.set_ylabel("Feature Rank (by absolute loading)", fontsize=9.5)
            ax_load.set_xlabel("Principal Component", fontsize=9.5)
            ax_load.set_title(
                f"PCA Component Loadings -- {_target_name} (T{_tp})\n"
                f"Top {_N_TOP_FEAT} features per component · {ctx.X_train.shape[1]} total features",
                fontsize=11,
                fontweight="bold",
                pad=12,
            )

            _cbar = fig_load.colorbar(_im, ax=ax_load, shrink=0.75, pad=0.03)
            _cbar.set_label("Component Loading", fontsize=9)
            _cbar.ax.tick_params(labelsize=7.5)

            for sp in ["top", "right"]:
                ax_load.spines[sp].set_visible(False)
            plt.tight_layout()

            # Save
            try:
                _fig_dir = os.path.join(os.getcwd(), "pca_baseline_figures")
                os.makedirs(_fig_dir, exist_ok=True)
                fig_load.savefig(
                    os.path.join(_fig_dir, f"pca_loadings_{_target_name}_T{_tp}.png"), dpi=300, bbox_inches="tight"
                )
                fig_load.savefig(
                    os.path.join(_fig_dir, f"pca_loadings_{_target_name}_T{_tp}.pdf"), dpi=300, bbox_inches="tight"
                )
                # Save loadings table as CSV
                _loadings_df.to_csv(os.path.join(_fig_dir, f"pca_loadings_{_target_name}_T{_tp}.csv"), index=False)
                logger.info(f"\n   Loadings saved to {_fig_dir}/")
            except Exception as _e:
                logger.info(f"\n  Warning: Could not save loadings: {_e}")

            plt.show()

            # HTML summary table
            _html_rows = []
            for _pc_i in range(_N_PCS_SHOW):
                _pc_sub = _loadings_df[_loadings_df["PC"] == _pc_i + 1].head(5)
                _var_pct = _pc_sub["Var Explained (%)"].iloc[0]
                _feat_str = " · ".join(f"{r['Display Name']} ({r['Loading']:+.3f})" for _, r in _pc_sub.iterrows())
                _html_rows.append(
                    f'<tr><td style="font-weight:700;text-align:center;padding:8px 12px;">'
                    f"PC{_pc_i + 1}</td>"
                    f'<td style="text-align:center;padding:8px 10px;">{_var_pct:.1f}%</td>'
                    f'<td style="padding:8px 12px;font-size:12px;">{_feat_str}</td></tr>'
                )
            display(
                HTML(
                    '<div style="font-family:sans-serif;max-width:850px;margin:16px 0;">'
                    '<table style="border-collapse:collapse;width:100%;border:1px solid #dee2e6;">'
                    '<thead><tr style="background:#f1f3f5;">'
                    '<th style="padding:10px 12px;border-bottom:2px solid #adb5bd;text-align:center;width:60px;">PC</th>'
                    '<th style="padding:10px 12px;border-bottom:2px solid #adb5bd;text-align:center;width:80px;">Var %</th>'
                    '<th style="padding:10px 12px;border-bottom:2px solid #adb5bd;">Top 5 Features (loading)</th>'
                    "</tr></thead><tbody>" + "".join(_html_rows) + "</tbody></table></div>"
                )
            )

            # Store loadings in _RS
            _RS["pca_baseline"]["loadings"] = _loadings_df.to_dict()

        # Analysis 2: temporal hold-out validation
        if ctx.run_temporal_holdout:
            # Temporal hold-out always uses across-category features
            # regardless of the main analysis split. This tests temporal
            # generalization of the full biopsychosocial feature set.
            # Additional dependency check for temporal analysis
            _temporal_skip = False
            for _treq in ["sample", "across_categories_data"]:
                if _treq not in ctx:
                    logger.info(f"Warning: {_treq} not found -- skipping temporal hold-out.")
                    _temporal_skip = True
            if not _temporal_skip:
                logger.info("=" * 72)
                logger.info("  TEMPORAL HOLD-OUT VALIDATION")
                logger.info("  Train on early waves (T0-T2), test on later waves (T3-T4)")
                logger.info("=" * 72)

                _target = ctx.target_options
                _tp_train = [0, 1, 2]  # Early waves
                _tp_test = [3, 4]  # Later waves
                _group = ctx.group if "group" in ctx else None

                logger.info(f"\n  Target: {_target}")
                logger.info(f"  Training timepoints: T{_tp_train}")
                logger.info(f"  Testing timepoints:  T{_tp_test}")

                _temporal_results = []

                # 1. Get same-timepoint baseline (standard approach)
                # Use the current X_train/X_test/y_train/y_test from the Main Analysis Runner as baseline
                _baseline_score = None
                _bl_pred = None
                if "X_train" in ctx and "y_train" in ctx:
                    _task = detect_task_type(ctx.y_train)
                    _is_clf = _task == "classification"
                    _metric = "AUC" if _is_clf else "R²"

                    _bl_model = create_lightweight_model(is_classification=_is_clf)
                    _y_tr = np.asarray(ctx.y_train).ravel()
                    _y_te = np.asarray(ctx.y_test).ravel()
                    _y_te_baseline = _y_te.copy()
                    _bl_model.fit(ctx.X_train, _y_tr)
                    if _is_clf:
                        _bl_pred = _bl_model.predict_proba(ctx.X_test)[:, 1]
                        _baseline_score = roc_auc_score(_y_te, _bl_pred)
                    else:
                        _bl_pred = _bl_model.predict(ctx.X_test)
                        _baseline_score = r2_score(_y_te, _bl_pred)

                    _tp_current = ctx.tp_option if "tp_option" in ctx else "?"
                    logger.info(f"\n  Standard (same-TP T{_tp_current}): {_metric} = {_baseline_score:.4f}")

                # 2. Temporal hold-out: Train T0-T2, test T3-T4
                logger.info("\n  Building temporal hold-out splits...")

                # Collect training data from early timepoints
                _train_frames_X = []
                _train_frames_y = []
                _test_frames_X = []
                _test_frames_y = []

                for _t in _tp_train + _tp_test:
                    try:
                        _X_tr, _X_te, _y_tr, _y_te = across_categories_data(
                            _t, [_target], group=_group, scale_x=True, scale_y=False, verbose=False
                        )
                        # Combine train+test from this timepoint
                        _X_all = pd.concat([_X_tr, _X_te], axis=0)
                        _y_all = (
                            pd.concat([_y_tr, _y_te], axis=0)
                            if isinstance(_y_tr, (pd.Series, pd.DataFrame))
                            else np.concatenate([_y_tr, _y_te])
                        )

                        if _t in _tp_train:
                            _train_frames_X.append(_X_all)
                            _train_frames_y.append(_y_all)
                            logger.info(f"    T{_t} → TRAIN: {len(_X_all)} subjects, {_X_all.shape[1]} features")
                        else:
                            _test_frames_X.append(_X_all)
                            _test_frames_y.append(_y_all)
                            logger.info(f"    T{_t} → TEST:  {len(_X_all)} subjects, {_X_all.shape[1]} features")
                    except Exception as _e:
                        logger.info(f"    T{_t} → SKIPPED ({_e})")

                if _train_frames_X and _test_frames_X:
                    # Align features across timepoints (take intersection)
                    _common_cols = set(_train_frames_X[0].columns)
                    for _df in _train_frames_X[1:] + _test_frames_X:
                        _common_cols &= set(_df.columns)
                    _common_cols = sorted(_common_cols)
                    logger.info(f"\n    Common features across timepoints: {len(_common_cols)}")

                    _X_temporal_train = pd.concat(
                        [df[_common_cols] for df in _train_frames_X], axis=0, ignore_index=True
                    )
                    _X_temporal_test = pd.concat([df[_common_cols] for df in _test_frames_X], axis=0, ignore_index=True)

                    # Handle y concatenation carefully
                    if isinstance(_train_frames_y[0], pd.DataFrame):
                        _y_temporal_train = pd.concat(_train_frames_y, axis=0, ignore_index=True).iloc[:, 0].values
                        _y_temporal_test = pd.concat(_test_frames_y, axis=0, ignore_index=True).iloc[:, 0].values
                    elif isinstance(_train_frames_y[0], pd.Series):
                        _y_temporal_train = pd.concat(_train_frames_y, axis=0, ignore_index=True).values
                        _y_temporal_test = pd.concat(_test_frames_y, axis=0, ignore_index=True).values
                    else:
                        _y_temporal_train = np.concatenate(_train_frames_y)
                        _y_temporal_test = np.concatenate(_test_frames_y)

                    # Drop NaN targets
                    _mask_tr = ~np.isnan(_y_temporal_train.astype(float))
                    _mask_te = ~np.isnan(_y_temporal_test.astype(float))
                    _X_temporal_train = _X_temporal_train[_mask_tr]
                    _y_temporal_train = _y_temporal_train[_mask_tr]
                    _X_temporal_test = _X_temporal_test[_mask_te]
                    _y_temporal_test = _y_temporal_test[_mask_te]

                    logger.info(f"    Temporal train: {len(_X_temporal_train)} subjects")
                    logger.info(f"    Temporal test:  {len(_X_temporal_test)} subjects")

                    # Impute NaN features
                    from sklearn.impute import SimpleImputer

                    _imp = SimpleImputer(strategy="median")
                    _imp.fit(_X_temporal_train)
                    _X_temporal_train = pd.DataFrame(_imp.transform(_X_temporal_train), columns=_common_cols)
                    _X_temporal_test = pd.DataFrame(_imp.transform(_X_temporal_test), columns=_common_cols)

                    # Fit and evaluate
                    _task = detect_task_type(_y_temporal_train)
                    _is_clf = _task == "classification"
                    _metric = "AUC" if _is_clf else "R²"

                    _temp_model = create_lightweight_model(is_classification=_is_clf)
                    _temp_model.fit(_X_temporal_train, _y_temporal_train.ravel())

                    if _is_clf:
                        _temp_pred = _temp_model.predict_proba(_X_temporal_test)[:, 1]
                        _temp_score = roc_auc_score(_y_temporal_test, _temp_pred)
                    else:
                        _temp_pred = _temp_model.predict(_X_temporal_test)
                        _temp_score = r2_score(_y_temporal_test, _temp_pred)

                    # Bootstrap CI
                    _point, _lo, _hi = bootstrap_ci(
                        _y_temporal_test.ravel(),
                        _temp_pred,
                        roc_auc_score if _is_clf else r2_score,
                        n_boot=500,
                        seed=42,
                    )

                    _temp_html = (
                        f'<div style="font-family:sans-serif;max-width:700px;border:2px solid #00695C;'
                        f'border-radius:8px;overflow:hidden;margin:16px 0;">'
                        f'<div style="background:#00695C;color:white;padding:12px 18px;font-size:14px;font-weight:700;">'
                        f"Temporal Hold-Out Results</div>"
                        f'<div style="padding:16px 20px;line-height:1.8;font-size:13px;">'
                        f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
                        f'<div><span style="color:#78909C;font-size:11px;text-transform:uppercase;">Temporal {_metric}</span>'
                        f'<br><span style="font-size:20px;font-weight:700;color:#00695C;">{_temp_score:.4f}</span>'
                        f'<span style="font-size:12px;color:#78909C;"> [{_lo:.4f}, {_hi:.4f}]</span></div>'
                    )
                    if _baseline_score is not None:
                        _delta = _baseline_score - _temp_score
                        _deg_color = "#C62828" if _delta > 0.05 else "#E65100" if _delta > 0 else "#2E7D32"
                        _temp_html += (
                            f'<div><span style="color:#78909C;font-size:11px;text-transform:uppercase;">Standard {_metric}</span>'
                            f'<br><span style="font-size:20px;font-weight:700;">{_baseline_score:.4f}</span></div>'
                            f'<div><span style="color:#78909C;font-size:11px;text-transform:uppercase;">Degradation</span>'
                            f'<br><span style="font-size:20px;font-weight:700;color:{_deg_color};">Δ = {_delta:+.4f}</span></div>'
                        )
                    _temp_html += "</div></div></div>"
                    display(HTML(_temp_html))

                    # 3. Publication figure
                    # Add bootstrap CI for baseline (same-TP)
                    _bl_point, _bl_lo, _bl_hi = (
                        bootstrap_ci(
                            _y_te_baseline, _bl_pred, roc_auc_score if _is_clf else r2_score, n_boot=500, seed=42
                        )
                        if _bl_pred is not None
                        else (_baseline_score, _baseline_score, _baseline_score)
                    )

                    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=300)

                    _bar_data = []
                    if _baseline_score is not None:
                        _bar_data.append(
                            {
                                "label": f"Same-TP (T{_tp_current})",
                                "score": _baseline_score,
                                "ci_lo": _bl_lo,
                                "ci_hi": _bl_hi,
                                "color": "#1a73e8",
                            }
                        )
                    _bar_data.append(
                        {
                            "label": "Temporal (T0-T2 → T3-T4)",
                            "score": _temp_score,
                            "ci_lo": _lo,
                            "ci_hi": _hi,
                            "color": "#d93025",
                        }
                    )

                    for _i, _bd in enumerate(_bar_data):
                        ax.barh(
                            _i,
                            _bd["score"],
                            height=0.45,
                            color=_bd["color"],
                            alpha=0.7,
                            zorder=3,
                            edgecolor="white",
                            linewidth=0.6,
                        )
                        ax.errorbar(
                            _bd["score"],
                            _i,
                            xerr=[[_bd["score"] - _bd["ci_lo"]], [_bd["ci_hi"] - _bd["score"]]],
                            fmt="o",
                            color=_bd["color"],
                            markersize=7,
                            markeredgecolor="white",
                            markeredgewidth=1.2,
                            capsize=4,
                            capthick=1.0,
                            ecolor="#333333",
                            zorder=4,
                        )
                        ax.text(
                            _bd["ci_hi"] + 0.008,
                            _i,
                            f"{_bd['score']:.4f}  [{_bd['ci_lo']:.3f}, {_bd['ci_hi']:.3f}]",
                            va="center",
                            fontsize=8.5,
                            fontweight="bold",
                            color=_bd["color"],
                        )

                    # Delta bracket
                    if _baseline_score is not None and len(_bar_data) == 2:
                        _deg = _baseline_score - _temp_score
                        _pct = (_deg / abs(_baseline_score) * 100) if _baseline_score != 0 else 0
                        _deg_color = "#2E7D32" if _deg <= 0 else "#C62828"
                        _x_brk = max(_bar_data[0]["ci_hi"], _bar_data[1]["ci_hi"]) + 0.04
                        ax.plot(
                            [_x_brk, _x_brk + 0.01, _x_brk + 0.01, _x_brk],
                            [0, 0, 1, 1],
                            color=_deg_color,
                            linewidth=1.0,
                            zorder=5,
                        )
                        ax.text(
                            _x_brk + 0.018,
                            0.5,
                            f"Δ = {-_deg:+.4f}\n({-_pct:+.1f}%)",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color=_deg_color,
                        )

                    ax.set_yticks(list(range(len(_bar_data))))
                    ax.set_yticklabels([d["label"] for d in _bar_data], fontsize=9.5, fontweight="bold")
                    ax.set_xlabel(_metric, fontsize=9)
                    ax.set_title(f"Temporal Generalization -- {_target}", fontsize=10, fontweight="bold")
                    _min_x = min(d["ci_lo"] for d in _bar_data)
                    ax.set_xlim(left=max(0, _min_x - 0.05))
                    ax.invert_yaxis()
                    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
                    for sp in ["top", "right"]:
                        ax.spines[sp].set_visible(False)
                    plt.tight_layout()

                    try:
                        _fig_dir = os.path.join(os.getcwd(), "temporal_holdout_figures")
                        os.makedirs(_fig_dir, exist_ok=True)
                        fig.savefig(
                            os.path.join(_fig_dir, f"temporal_holdout_{_target}.png"), dpi=300, bbox_inches="tight"
                        )
                        fig.savefig(
                            os.path.join(_fig_dir, f"temporal_holdout_{_target}.pdf"), dpi=300, bbox_inches="tight"
                        )
                    except Exception as _e:
                        logger.info(f"Warning: {_e}")

                    plt.show()
                    # Store for dashboard
                    _RS["temporal_holdout"] = {
                        "temporal_score": _temp_score,
                        "temporal_ci": [_lo, _hi],
                        "baseline_score": _baseline_score,
                        "metric": _metric,
                        "target": _target,
                        "n_train": len(_X_temporal_train),
                        "n_test": len(_X_temporal_test),
                        "n_common_features": len(_common_cols),
                    }

                else:
                    logger.info("\n  Error: Could not build temporal hold-out -- insufficient timepoint data.")


def run() -> None:
    """Run this stage (notebook cell 17)."""
    global copy, traceback
    warnings.filterwarnings("ignore")

    # Batch dispatch
    _is_bv_24 = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_24:
        try:
            _cbv_24 = cache_load("main_runner")
            if (
                _cbv_24
                and isinstance(_cbv_24, dict)
                and _cbv_24.get("_batch_archive")
                and len(_cbv_24["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_24["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_24["_batch_archive"]
                _is_bv_24 = True
        except Exception:
            pass  # non-critical; logged upstream or optional visualization

    if _is_bv_24:
        _vgs_24 = {
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
        logger.info(f"  BATCH -- Validation Hub: processing {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)

        for _vi_24, (_vk_24, _vd_24) in enumerate(ctx._batch_archive.items()):
            _vl_24 = _vd_24.get("target", _vk_24)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_24 + 1}/{len(ctx._batch_archive)}] Target: {_vl_24}")
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
                if _vd_24.get(_gk) is not None:
                    ctx[_gk] = _vd_24[_gk]
            ctx["target_options"] = _vl_24
            if "timepoint" in _vd_24:
                ctx["tp_option"] = _vd_24["timepoint"]
            try:
                _run_for_current_target()
            except Exception as _e24:
                logger.info(f"  Warning: Error for {_vl_24}: {_e24}")
                import traceback

                traceback.print_exc()
            # Archive results back into _batch_archive
            import copy

            _rs_ref = ctx.get("_RS")
            if _vk_24 in ctx._batch_archive and _rs_ref is not None:
                for _rk in ["pca_baseline", "temporal_holdout"]:
                    if _rk in _rs_ref:
                        ctx._batch_archive[_vk_24][f"_RS_{_rk}"] = copy.deepcopy(_rs_ref[_rk])

        for _k, _v in _vgs_24.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH COMPLETE -- Validation Hub: {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)
    else:
        _run_for_current_target()
