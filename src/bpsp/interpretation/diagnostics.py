"""Model diagnostics (the `diagnostics` stage).

Diagnostics for the most recent analysis run, each switched on or off in
the `diagnostics` settings: base-learner diversity (prediction scatter
matrices), bootstrap stability of feature importance (resampled rankings
with violin plots), learning curves (performance versus training set
size), SHAP beeswarm plots (per model and consensus), and supplementary
effect size, feature overlap and prediction diagnostics.
"""

import logging

from bpsp.context import ctx

logger = logging.getLogger(__name__)


# Shared imports
import copy
import time as _time
import traceback as _traceback
import warnings as _warnings

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

_SECTION_COLORS = {
    "diversity": "#4C72B0",
    "stability": "#DD8452",
    "learning": "#55A868",
    "shap": "#C44E52",
    "supplementary": "#8172B3",
}

# Shared utility functions


def _detect_task_type(y):
    try:
        _global_fn = ctx.get("detect_task_type", None)
        if _global_fn is not None:
            return _global_fn(y)
    except Exception:
        pass
    # Local fallback
    _y_arr = np.asarray(y).ravel()
    _unique = np.unique(_y_arr[~np.isnan(_y_arr)] if np.issubdtype(_y_arr.dtype, np.floating) else _y_arr)
    if len(_unique) <= 20 and np.all(_unique == _unique.astype(int)):
        return "classification"
    return "regression"


def _create_lightweight_model(is_classification, n_features):
    """Create a lightweight CatBoost model for bootstrap/learning curve analyses.
    Uses global factory if available, otherwise builds locally."""
    try:
        _global_fn = ctx.get("create_lightweight_model", None)
        if _global_fn is not None:
            return _global_fn(is_classification, n_features)
    except Exception:
        pass
    # Local fallback: CatBoost with conservative settings
    try:
        if is_classification:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=200,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=5,
                verbose=0,
                random_seed=42,
                allow_writing_files=False,
                thread_count=1,
            )
        else:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=200,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=5,
                verbose=0,
                random_seed=42,
                allow_writing_files=False,
                thread_count=1,
            )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        if is_classification:
            return GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        else:
            return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)


def _get_display_name(name):
    _dn = ctx.get("DISPLAY_NAMES", None)
    if _dn is not None and isinstance(_dn, dict):
        return _dn.get(name, name)
    return name


def _feature_names(X):
    """Extract feature names from DataFrame or generate generic names."""
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{i}" for i in range(X.shape[1])]


def _to_numpy(X):
    """Convert to dense numpy array."""
    if hasattr(X, "values"):
        return X.values
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _section_header(title, color="#333333"):
    """Print a styled section header."""
    display(
        HTML(
            f'<div style="background: linear-gradient(135deg, {color}22, {color}11); '
            f"border-left: 4px solid {color}; padding: 12px 16px; margin: 16px 0 12px 0; "
            f'border-radius: 0 8px 8px 0;">'
            f'<h3 style="margin:0; color: {color};"> {title}</h3></div>'
        )
    )


def _subsection_header(title):
    """Print a styled subsection header."""
    display(
        HTML(
            f'<div style="border-bottom: 2px solid #ddd; padding-bottom: 4px; '
            f'margin: 12px 0 8px 0;">'
            f'<h4 style="margin:0; color: #555;">▸ {title}</h4></div>'
        )
    )


def _safe_html_table(df, caption="", max_rows=50):
    """Display a DataFrame as a styled HTML table."""
    _style = (
        df.head(max_rows)
        .style.set_caption(caption)
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-weight", "bold"),
                        ("font-size", "13px"),
                        ("text-align", "left"),
                        ("padding", "6px 0"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#f7f7f7"),
                        ("padding", "6px 10px"),
                        ("border-bottom", "2px solid #ddd"),
                        ("font-size", "11px"),
                    ],
                },
                {"selector": "td", "props": [("padding", "5px 10px"), ("font-size", "11px")]},
                {"selector": "tr:nth-child(even)", "props": [("background-color", "#fafafa")]},
            ]
        )
    )
    display(_style)


_GATE_MESSAGES = []

# Timing infrastructure
_section_times = {}


def run() -> None:
    """Run this stage (notebook cell 10)."""
    global _CatPool, copy, lowess, shap
    # @markdown Set to `"last"` for current globals, or a target key from `_batch_archive`.

    if (
        ctx.diagnostics_target != "last"
        and "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and ctx.diagnostics_target in ctx._batch_archive
    ):
        _diag_data = ctx._batch_archive[ctx.diagnostics_target]
        import copy

        for _gk in [
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "metrics_dict_full",
            "consensus_df",
            "model_weights",
            "split",
            "exported_within_dict",
            "last_fitted_model",
            "category_results",
            "sample_valid",
            "sample_valid_test",
            "sample_valid_train",
        ]:
            if _diag_data.get(_gk) is not None:
                ctx[_gk] = _diag_data[_gk]
        ctx["target_options"] = _diag_data.get("target", ctx.diagnostics_target)
        if "timepoint" in _diag_data:
            ctx["tp_option"] = _diag_data["timepoint"]
        logger.info(f" Loaded batch target: {ctx.diagnostics_target}")
    elif ctx.diagnostics_target != "last":
        if "_batch_archive" in ctx and isinstance(ctx._batch_archive, dict):
            logger.info(f"Warning:  Target '{ctx.diagnostics_target}' not found in _batch_archive.")
            logger.info(f"   Available: {list(ctx._batch_archive.keys())}")
        else:
            logger.info("Warning:  No _batch_archive available. Using current globals.")

    _warnings.filterwarnings("ignore", category=FutureWarning)
    _warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # Shared style
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.dpi": 120,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    # Unified gate: check the run context for the main stage's results
    _SKIP = False

    for _req in ["X_train", "X_test", "y_train", "y_test"]:
        if _req not in ctx:
            _GATE_MESSAGES.append(f"Missing required variable: `{_req}`. Run the main stage first.")
            _SKIP = True

    if _SKIP:
        display(
            HTML(
                '<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; '
                'padding:16px; margin:12px 0;">'
                "<strong>Warning: Diagnostics Hub -- Skipped</strong><br>" + "<br>".join(_GATE_MESSAGES) + "</div>"
            )
        )
    _hub_start = _time.time()

    if not _SKIP:
        # Resolve shared references
        _X_train_np = _to_numpy(ctx.X_train)
        _X_test_np = _to_numpy(ctx.X_test)
        _y_train_np = np.asarray(ctx.y_train).ravel()
        _y_test_np = np.asarray(ctx.y_test).ravel()
        _feat_names = _feature_names(ctx.X_train)
        _n_features = _X_train_np.shape[1]

        _task_type = None
        if (
            "last_fitted_model" in ctx
            and ctx.last_fitted_model is not None
            and hasattr(ctx.last_fitted_model, "task_type")
        ):
            _task_type = ctx.last_fitted_model.task_type
        if _task_type is None:
            _task_type = _detect_task_type(_y_train_np)

        _is_classification = _task_type == "classification"

        display(
            HTML(
                '<div style="background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; '
                'padding: 20px 24px; border-radius: 12px; margin-bottom: 16px;">'
                '<h2 style="margin:0 0 8px 0; color: #e94560;"> Advanced Model Diagnostics Hub</h2>'
                '<p style="margin:0; opacity:0.85;">Consolidated analyses: Diversity · Stability · '
                "Learning Curves · SHAP · Supplementary</p>"
                f'<p style="margin:4px 0 0 0; opacity:0.65; font-size:12px;">'
                f"Task type: <b>{_task_type}</b> | Features: <b>{_n_features}</b> | "
                f"Train samples: <b>{_X_train_np.shape[0]}</b> | Test samples: <b>{_X_test_np.shape[0]}</b></p>"
                "</div>"
            )
        )

        # SECTION 1: Base Learner Prediction Diversity
        if ctx.run_diversity_analysis:
            _t0 = _time.time()
            _section_header("Base Learner Prediction Diversity", _SECTION_COLORS["diversity"])

            try:
                # Extract base learner predictions
                _has_model = (
                    "last_fitted_model" in ctx
                    and ctx.last_fitted_model is not None
                    and hasattr(ctx.last_fitted_model, "models")
                )
                if not _has_model:
                    display(
                        HTML(
                            '<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; '
                            'padding:12px; margin:8px 0;">'
                            "<strong>Warning: Skipped:</strong> <code>last_fitted_model</code> not found or has no "
                            "<code>.models</code> attribute.</div>"
                        )
                    )
                else:
                    _base_models = ctx.last_fitted_model.models  # dict: name -> fitted model
                    _model_names = list(_base_models.keys())
                    _n_models = len(_model_names)

                    logger.info(f"Found {_n_models} base learners: {', '.join(_model_names)}")

                    # Collect predictions on test set
                    _pred_dict = {}
                    for _mname, _mobj in _base_models.items():
                        try:
                            # Determine feature set (TabPFN may use a reduced feature set)
                            _X_input = ctx.X_test
                            if _mname.lower() == "tabpfn" or (
                                hasattr(_mobj, "selected_features_") and _mobj.selected_features_ is not None
                            ):
                                try:
                                    _sel_feats = _mobj.selected_features_
                                    if hasattr(ctx.X_test, "columns"):
                                        _X_input = ctx.X_test[_sel_feats]
                                    else:
                                        _X_input = _X_test_np[:, _sel_feats]
                                except Exception:
                                    _X_input = ctx.X_test

                            if _is_classification and hasattr(_mobj, "predict_proba"):
                                _preds = _mobj.predict_proba(_X_input)
                                if _preds.ndim == 2:
                                    _preds = _preds[:, 1] if _preds.shape[1] == 2 else _preds[:, 0]
                            else:
                                try:
                                    _preds = _mobj.predict(_X_input)
                                except Exception:
                                    # CatBoost fallback: try with Pool
                                    try:
                                        from catboost import Pool as _CatPool

                                        _pool = _CatPool(_X_input)
                                        _preds = _mobj.predict(_pool)
                                    except Exception as _pool_err:
                                        raise _pool_err

                            _pred_dict[_mname] = np.asarray(_preds, dtype=float).ravel()
                        except Exception as _e:
                            logger.info(f"  Warning: Could not get predictions from {_mname}: {_e}")

                    if len(_pred_dict) < 2:
                        display(
                            HTML(
                                '<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; '
                                'padding:12px; margin:8px 0;">'
                                "<strong>Warning: Skipped:</strong> Need at least 2 base learners with valid "
                                "predictions for diversity analysis.</div>"
                            )
                        )
                    else:
                        _pred_df = pd.DataFrame(_pred_dict)
                        _active_names = list(_pred_dict.keys())
                        _n_active = len(_active_names)

                        # 1a. Pairwise Correlation Heatmap
                        _subsection_header("Pairwise Prediction Correlations")
                        _corr_matrix = _pred_df.corr(method="pearson")

                        # Display names for axes
                        _disp_names = [_get_display_name(n) if ctx.use_display_names else n for n in _active_names]
                        _corr_display = _corr_matrix.copy()
                        _corr_display.index = _disp_names
                        _corr_display.columns = _disp_names

                        fig, ax = plt.subplots(figsize=(max(8, _n_active * 1.2), max(6, _n_active * 1.0)))
                        _mask = np.triu(np.ones_like(_corr_matrix, dtype=bool), k=1)
                        sns.heatmap(
                            _corr_display,
                            mask=_mask,
                            annot=True,
                            fmt=".3f",
                            cmap="RdYlBu_r",
                            center=0.5,
                            vmin=0,
                            vmax=1,
                            linewidths=0.5,
                            linecolor="white",
                            cbar_kws={"label": "Pearson r", "shrink": 0.8},
                            ax=ax,
                            square=True,
                        )
                        ax.set_title("Base Learner Prediction Correlation Matrix", fontweight="bold", pad=12)
                        plt.xticks(rotation=45, ha="right")
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        plt.show()

                        # Compute mean pairwise correlation (lower triangle only)
                        _tril_idx = np.tril_indices(_n_active, k=-1)
                        _corr_vals = _corr_matrix.values[_tril_idx]
                        _mean_corr = np.mean(_corr_vals)
                        _min_corr = np.min(_corr_vals)
                        _max_corr = np.max(_corr_vals)

                        logger.info("\nPairwise correlation summary:")
                        logger.info(f"  Mean:  {_mean_corr:.4f}")
                        logger.info(f"  Min:   {_min_corr:.4f}")
                        logger.info(f"  Max:   {_max_corr:.4f}")
                        logger.info(f"  Range: {_max_corr - _min_corr:.4f}")

                        # Diversity interpretation
                        if _mean_corr < 0.5:
                            _div_verdict = "HIGH diversity -- base learners capture substantially different patterns"
                        elif _mean_corr < 0.75:
                            _div_verdict = "MODERATE diversity -- reasonable variation among base learners"
                        elif _mean_corr < 0.9:
                            _div_verdict = "LOW diversity -- base learners largely agree; ensemble gains may be limited"
                        else:
                            _div_verdict = (
                                "VERY LOW diversity -- near-redundant predictions; consider replacing similar models"
                            )
                        logger.info(f"  Interpretation: {_div_verdict}")

                        # 1b. Individual Model Performance
                        _subsection_header("Individual Model Performance")
                        _perf_records = []
                        for _mname in _active_names:
                            _p = _pred_dict[_mname]
                            if _is_classification:
                                try:
                                    _auc = roc_auc_score(_y_test_np, _p)
                                except Exception:
                                    _auc = np.nan
                                _p_binary = (_p >= 0.5).astype(int)
                                _acc = accuracy_score(_y_test_np, _p_binary)
                                _perf_records.append(
                                    {
                                        "Model": _get_display_name(_mname) if ctx.use_display_names else _mname,
                                        "AUC": _auc,
                                        "Accuracy": _acc,
                                    }
                                )
                            else:
                                _r2 = r2_score(_y_test_np, _p)
                                _rmse = np.sqrt(mean_squared_error(_y_test_np, _p))
                                _mae = mean_absolute_error(_y_test_np, _p)
                                _perf_records.append(
                                    {
                                        "Model": _get_display_name(_mname) if ctx.use_display_names else _mname,
                                        "R²": _r2,
                                        "RMSE": _rmse,
                                        "MAE": _mae,
                                    }
                                )

                        _perf_df = pd.DataFrame(_perf_records)
                        _safe_html_table(_perf_df, caption="Individual Base Learner Performance on Test Set")

                        # Performance bar chart
                        _metric_col = "AUC" if _is_classification else "R²"
                        _perf_sorted = _perf_df.sort_values(_metric_col, ascending=True)

                        fig, ax = plt.subplots(figsize=(10, max(4, _n_active * 0.6)))
                        _bar_colors = plt.cm.viridis(np.linspace(0.3, 0.85, _n_active))
                        bars = ax.barh(
                            _perf_sorted["Model"],
                            _perf_sorted[_metric_col],
                            color=_bar_colors,
                            edgecolor="white",
                            linewidth=0.5,
                        )
                        for _bar in bars:
                            _w = _bar.get_width()
                            ax.text(
                                _w + 0.005,
                                _bar.get_y() + _bar.get_height() / 2,
                                f"{_w:.4f}",
                                ha="left",
                                va="center",
                                fontsize=10,
                            )
                        ax.set_xlabel(_metric_col)
                        ax.set_title(f"Base Learner {_metric_col} on Test Set", fontweight="bold")
                        ax.set_xlim(left=min(0, _perf_sorted[_metric_col].min() - 0.05))
                        plt.tight_layout()
                        plt.show()

                        # 1c. Meta-Learner Weights
                        _subsection_header("Meta-Learner Weights")
                        _meta_weights = None
                        if hasattr(ctx.last_fitted_model, "meta_model") and hasattr(
                            ctx.last_fitted_model.meta_model, "coef_"
                        ):
                            _coefs = np.asarray(ctx.last_fitted_model.meta_model.coef_).ravel()
                            # Map coefficients to model names
                            # Meta-model input order matches _base_models key order
                            _meta_model_names = list(_base_models.keys())
                            if len(_coefs) == len(_meta_model_names):
                                _meta_weights = dict(zip(_meta_model_names, _coefs))
                            elif len(_coefs) > len(_meta_model_names):
                                # May have extra columns (e.g., original features passed through)
                                _meta_weights = dict(zip(_meta_model_names, _coefs[: len(_meta_model_names)]))
                        elif "model_weights" in ctx and ctx.model_weights is not None:
                            _meta_weights = ctx.model_weights

                        if _meta_weights is not None:
                            _weight_df = pd.DataFrame(
                                [
                                    {
                                        "Model": _get_display_name(k) if ctx.use_display_names else k,
                                        "Weight": v,
                                        "Abs Weight": abs(v),
                                        "Weight %": 0.0,
                                    }
                                    for k, v in _meta_weights.items()
                                    if k in _active_names
                                ]
                            )
                            if len(_weight_df) > 0:
                                _total_abs = _weight_df["Abs Weight"].sum()
                                if _total_abs > 0:
                                    _weight_df["Weight %"] = _weight_df["Abs Weight"] / _total_abs * 100
                                _weight_df = _weight_df.sort_values("Abs Weight", ascending=True)

                                _safe_html_table(
                                    _weight_df[["Model", "Weight", "Weight %"]].round(4),
                                    caption="Meta-Learner Coefficients (Weights)",
                                )

                                fig, ax = plt.subplots(figsize=(10, max(4, len(_weight_df) * 0.5)))
                                _wcolors = ["#e74c3c" if w < 0 else "#2ecc71" for w in _weight_df["Weight"]]
                                ax.barh(
                                    _weight_df["Model"],
                                    _weight_df["Weight"],
                                    color=_wcolors,
                                    edgecolor="white",
                                    linewidth=0.5,
                                )
                                ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
                                for i, (_, row) in enumerate(_weight_df.iterrows()):
                                    _w = row["Weight"]
                                    _ha = "left" if _w >= 0 else "right"
                                    _offset = 0.005 if _w >= 0 else -0.005
                                    ax.text(_w + _offset, i, f"{_w:.4f}", ha=_ha, va="center", fontsize=10)
                                ax.set_xlabel("Meta-Learner Coefficient")
                                ax.set_title("Meta-Learner Weights by Base Model", fontweight="bold")
                                plt.tight_layout()
                                plt.show()
                        else:
                            logger.info("Meta-learner weights not available.")

                        # 1d. Scatter Matrix of Most Diverse Pairs
                        if ctx.show_scatter_matrix and _n_active >= 2:
                            _subsection_header(f"Scatter Plots -- Top {ctx.top_n_scatter_pairs} Most Diverse Pairs")

                            # Find most diverse (lowest corr) pairs
                            _pair_corrs = []
                            for i in range(len(_active_names)):
                                for j in range(i + 1, len(_active_names)):
                                    _pair_corrs.append((_active_names[i], _active_names[j], _corr_matrix.iloc[i, j]))
                            _pair_corrs.sort(key=lambda x: x[2])
                            _top_pairs = _pair_corrs[: ctx.top_n_scatter_pairs]

                            _n_pairs = len(_top_pairs)
                            fig, axes = plt.subplots(1, _n_pairs, figsize=(6 * _n_pairs, 5))
                            if _n_pairs == 1:
                                axes = [axes]

                            for _idx, (_m1, _m2, _r) in enumerate(_top_pairs):
                                ax = axes[_idx]
                                _d1 = _get_display_name(_m1) if ctx.use_display_names else _m1
                                _d2 = _get_display_name(_m2) if ctx.use_display_names else _m2
                                ax.scatter(
                                    _pred_dict[_m1],
                                    _pred_dict[_m2],
                                    alpha=0.4,
                                    s=20,
                                    edgecolors="none",
                                    c=_SECTION_COLORS["diversity"],
                                )
                                # Identity line
                                _lims = [
                                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                                ]
                                ax.plot(_lims, _lims, "k--", alpha=0.3, linewidth=1)
                                ax.set_xlim(_lims)
                                ax.set_ylim(_lims)
                                ax.set_xlabel(_d1, fontsize=10)
                                ax.set_ylabel(_d2, fontsize=10)
                                ax.set_title(f"r = {_r:.3f}", fontsize=11, fontweight="bold")
                                ax.set_aspect("equal", adjustable="box")

                            fig.suptitle("Most Diverse Base Learner Pairs (Predictions)", fontweight="bold", y=1.02)
                            plt.tight_layout()
                            plt.show()

                        # 1e. Diversity Summary Box
                        _subsection_header("Diversity Summary")
                        _ensemble_preds = None
                        if hasattr(ctx.last_fitted_model, "predict"):
                            try:
                                if _is_classification and hasattr(ctx.last_fitted_model, "predict_proba"):
                                    _ensemble_preds = ctx.last_fitted_model.predict_proba(_X_test_np)
                                    if _ensemble_preds.ndim == 2:
                                        _ensemble_preds = _ensemble_preds[:, 1]
                                else:
                                    _ensemble_preds = ctx.last_fitted_model.predict(_X_test_np)
                                _ensemble_preds = np.asarray(_ensemble_preds).ravel()
                            except Exception:
                                _ensemble_preds = None

                        _summary_rows = []
                        _summary_rows.append({"Metric": "Number of Base Learners", "Value": str(_n_active)})
                        _summary_rows.append({"Metric": "Mean Pairwise Correlation", "Value": f"{_mean_corr:.4f}"})
                        _summary_rows.append({"Metric": "Min Pairwise Correlation", "Value": f"{_min_corr:.4f}"})
                        _summary_rows.append({"Metric": "Max Pairwise Correlation", "Value": f"{_max_corr:.4f}"})
                        _summary_rows.append({"Metric": "Diversity Assessment", "Value": _div_verdict})

                        if _ensemble_preds is not None:
                            if _is_classification:
                                try:
                                    _ens_auc = roc_auc_score(_y_test_np, _ensemble_preds)
                                    _best_base = _perf_df["AUC"].max()
                                    _summary_rows.append({"Metric": "Ensemble AUC", "Value": f"{_ens_auc:.4f}"})
                                    _summary_rows.append({"Metric": "Best Base AUC", "Value": f"{_best_base:.4f}"})
                                    _summary_rows.append(
                                        {"Metric": "Ensemble Gain", "Value": f"{_ens_auc - _best_base:+.4f}"}
                                    )
                                except Exception:
                                    pass
                            else:
                                _ens_r2 = r2_score(_y_test_np, _ensemble_preds)
                                _best_base = _perf_df["R²"].max()
                                _summary_rows.append({"Metric": "Ensemble R²", "Value": f"{_ens_r2:.4f}"})
                                _summary_rows.append({"Metric": "Best Base R²", "Value": f"{_best_base:.4f}"})
                                _summary_rows.append(
                                    {"Metric": "Ensemble Gain", "Value": f"{_ens_r2 - _best_base:+.4f}"}
                                )

                        _safe_html_table(pd.DataFrame(_summary_rows), caption="Prediction Diversity Summary")

            except Exception as _e:
                logger.info(f"Error: Diversity analysis error: {_e}")
                _traceback.print_exc()

            _section_times["Diversity Analysis"] = _time.time() - _t0
            logger.info(f"\n Diversity Analysis completed in {_section_times['Diversity Analysis']:.1f}s")

        # SECTION 2: Feature Importance Stability (Bootstrap)
        if ctx.run_bootstrap_stability:
            _t0 = _time.time()
            _section_header("Feature Importance Stability (Bootstrap)", _SECTION_COLORS["stability"])

            try:
                _is_clf_local = _is_classification
                _model_template = _create_lightweight_model(_is_clf_local, _n_features)

                logger.info(f"Running {ctx.n_bootstraps} bootstrap iterations with {type(_model_template).__name__}...")
                logger.info(f"Tracking top {ctx.top_k_features} features by importance")

                _n_train = _X_train_np.shape[0]
                _importance_matrix = np.zeros((ctx.n_bootstraps, _n_features))
                _rank_matrix = np.zeros((ctx.n_bootstraps, _n_features), dtype=int)
                _oob_scores = []

                for _b in range(ctx.n_bootstraps):
                    _rng = np.random.RandomState(_b)
                    _boot_idx = _rng.choice(_n_train, size=_n_train, replace=True)
                    _oob_idx = np.setdiff1d(np.arange(_n_train), _boot_idx)

                    _X_boot = _X_train_np[_boot_idx]
                    _y_boot = _y_train_np[_boot_idx]

                    _m = copy.deepcopy(_model_template)
                    try:
                        _m.fit(_X_boot, _y_boot)
                    except Exception:
                        # Retry without deepcopy
                        _m = _create_lightweight_model(_is_clf_local, _n_features)
                        _m.fit(_X_boot, _y_boot)

                    # Extract feature importance
                    if hasattr(_m, "feature_importances_"):
                        _imp = np.asarray(_m.feature_importances_).ravel()
                    elif hasattr(_m, "get_feature_importance"):
                        _imp = np.asarray(_m.get_feature_importance()).ravel()
                    elif hasattr(_m, "coef_"):
                        _imp = np.abs(np.asarray(_m.coef_).ravel())
                    else:
                        _imp = np.zeros(_n_features)

                    _importance_matrix[_b] = _imp
                    # Ranks: rank 1 = most important
                    _rank_matrix[_b] = stats.rankdata(-_imp, method="ordinal").astype(int)

                    # OOB score
                    if len(_oob_idx) > 0:
                        try:
                            _X_oob = _X_train_np[_oob_idx]
                            _y_oob = _y_train_np[_oob_idx]
                            if _is_clf_local:
                                _oob_pred = _m.predict(_X_oob)
                                _oob_scores.append(accuracy_score(_y_oob, _oob_pred))
                            else:
                                _oob_pred = _m.predict(_X_oob)
                                _oob_scores.append(r2_score(_y_oob, _oob_pred))
                        except Exception:
                            pass

                    if (_b + 1) % max(1, ctx.n_bootstraps // 5) == 0:
                        logger.info(f"  Bootstrap {_b + 1}/{ctx.n_bootstraps} complete")

                logger.info(f"Bootstrap complete. OOB scores available: {len(_oob_scores)}")
                if _oob_scores:
                    logger.info(
                        f"  OOB {'Accuracy' if _is_clf_local else 'R²'}: "
                        f"mean={np.mean(_oob_scores):.4f}, std={np.std(_oob_scores):.4f}"
                    )

                # Compute stability metrics
                _mean_importance = _importance_matrix.mean(axis=0)
                _std_importance = _importance_matrix.std(axis=0)
                _mean_rank = _rank_matrix.mean(axis=0)
                _std_rank = _rank_matrix.std(axis=0)
                _median_rank = np.median(_rank_matrix, axis=0)

                # Stability score: lower std_rank relative to mean_rank = more stable
                _cv_rank = np.where(_mean_rank > 0, _std_rank / _mean_rank, 0)
                _stability_score = 1.0 - np.clip(_cv_rank / _cv_rank.max() if _cv_rank.max() > 0 else _cv_rank, 0, 1)

                _stability_df = pd.DataFrame(
                    {
                        "Feature": _feat_names,
                        "Mean Importance": _mean_importance,
                        "Std Importance": _std_importance,
                        "Mean Rank": _mean_rank,
                        "Std Rank": _std_rank,
                        "Median Rank": _median_rank,
                        "Stability Score": _stability_score,
                    }
                ).sort_values("Mean Importance", ascending=False)

                _top_feat_idx = _stability_df.head(ctx.top_k_features).index.tolist()
                _top_feat_names = _stability_df.head(ctx.top_k_features)["Feature"].tolist()

                # 2a. Rank-Frequency Heatmap
                _subsection_header(f"Feature Rank Distribution (Top {ctx.top_k_features})")

                # Build rank frequency matrix for top features
                _top_k_actual = min(ctx.top_k_features, _n_features)
                _top_indices = [_feat_names.index(fn) for fn in _top_feat_names[:_top_k_actual]]
                _max_rank_show = min(_n_features, 30)
                _rank_freq = np.zeros((_top_k_actual, _max_rank_show))

                for _fi, _global_idx in enumerate(_top_indices):
                    for _r in range(_max_rank_show):
                        _rank_freq[_fi, _r] = np.mean(_rank_matrix[:, _global_idx] == (_r + 1))

                _rank_freq_df = pd.DataFrame(
                    _rank_freq,
                    index=[
                        _get_display_name(fn) if ctx.use_display_names else fn for fn in _top_feat_names[:_top_k_actual]
                    ],
                    columns=[str(r + 1) for r in range(_max_rank_show)],
                )

                fig, ax = plt.subplots(figsize=(max(12, _max_rank_show * 0.5), max(6, _top_k_actual * 0.4)))
                sns.heatmap(
                    _rank_freq_df,
                    cmap="YlOrRd",
                    ax=ax,
                    linewidths=0.3,
                    cbar_kws={"label": "Frequency", "shrink": 0.8},
                    xticklabels=True,
                    yticklabels=True,
                )
                ax.set_xlabel("Rank Position")
                ax.set_ylabel("Feature")
                ax.set_title("Bootstrap Rank Distribution Heatmap", fontweight="bold")
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=9)
                plt.tight_layout()
                plt.show()

                # 2b. Stability Score Bar Chart
                _subsection_header("Feature Stability Scores")
                _top_stab = _stability_df.head(ctx.top_k_features).sort_values("Stability Score", ascending=True)
                _disp_labels = [_get_display_name(fn) if ctx.use_display_names else fn for fn in _top_stab["Feature"]]

                fig, ax = plt.subplots(figsize=(10, max(5, ctx.top_k_features * 0.35)))
                _cmap = plt.cm.RdYlGn
                ctx._norm = plt.Normalize(vmin=0, vmax=1)
                _colors = [_cmap(ctx._norm(s)) for s in _top_stab["Stability Score"]]
                bars = ax.barh(
                    _disp_labels, _top_stab["Stability Score"], color=_colors, edgecolor="white", linewidth=0.5
                )
                for _bar in bars:
                    _w = _bar.get_width()
                    ax.text(
                        _w + 0.01, _bar.get_y() + _bar.get_height() / 2, f"{_w:.3f}", ha="left", va="center", fontsize=9
                    )
                ax.set_xlabel("Stability Score (1 = perfectly stable)")
                ax.set_title(
                    f"Top {ctx.top_k_features} Feature Stability Scores ({ctx.n_bootstraps} bootstraps)",
                    fontweight="bold",
                )
                ax.set_xlim(0, 1.15)
                plt.tight_layout()
                plt.show()

                # 2c. Violin / Box Plot
                if ctx.show_violin_plot and _top_k_actual >= 2:
                    _subsection_header("Rank Distribution (Violin + Box)")
                    _violin_data = []
                    for _fn in _top_feat_names[: min(_top_k_actual, 15)]:
                        _gi = _feat_names.index(_fn)
                        _dn = _get_display_name(_fn) if ctx.use_display_names else _fn
                        for _r in _rank_matrix[:, _gi]:
                            _violin_data.append({"Feature": _dn, "Rank": _r})
                    _violin_df = pd.DataFrame(_violin_data)

                    _n_violin = _violin_df["Feature"].nunique()
                    fig, ax = plt.subplots(figsize=(max(10, _n_violin * 0.8), 6))
                    _order = [
                        _get_display_name(fn) if ctx.use_display_names else fn
                        for fn in _top_feat_names[: min(_top_k_actual, 15)]
                    ]
                    sns.violinplot(
                        data=_violin_df, x="Feature", y="Rank", order=_order, inner="box", palette="Set2", ax=ax, cut=0
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("Rank (1 = most important)")
                    ax.set_title("Feature Rank Distributions Across Bootstraps", fontweight="bold")
                    ax.invert_yaxis()
                    plt.xticks(rotation=45, ha="right", fontsize=9)
                    plt.tight_layout()
                    plt.show()

                # 2d. Summary Table
                _subsection_header("Stability Summary Table")
                _display_stab = _stability_df.head(ctx.top_k_features).copy()
                _display_stab["Feature"] = [
                    _get_display_name(fn) if ctx.use_display_names else fn for fn in _display_stab["Feature"]
                ]
                _display_stab = _display_stab.round(4)
                _safe_html_table(_display_stab, caption=f"Top {ctx.top_k_features} Features -- Bootstrap Stability")

                # 2e. Comparison with Consensus
                if "consensus_df" in ctx and ctx.consensus_df is not None:
                    _subsection_header("Comparison with Consensus Rankings")
                    try:
                        _con = ctx.consensus_df.copy()
                        _con_feat_col = None
                        for _c in ["feature", "Feature", "feature_name", "Feature_Name"]:
                            if _c in _con.columns:
                                _con_feat_col = _c
                                break
                        if _con_feat_col is not None:
                            _con_top = _con.head(ctx.top_k_features)[_con_feat_col].tolist()
                            _boot_top = _top_feat_names[: ctx.top_k_features]
                            _overlap = set(_con_top) & set(_boot_top)
                            _jaccard = (
                                len(_overlap) / len(set(_con_top) | set(_boot_top))
                                if len(set(_con_top) | set(_boot_top)) > 0
                                else 0
                            )

                            logger.info(
                                f"Top-{ctx.top_k_features} overlap with consensus: {len(_overlap)}/{ctx.top_k_features}"
                            )
                            logger.info(f"Jaccard similarity: {_jaccard:.4f}")
                            logger.info(f"Overlapping features: {sorted(_overlap)}")

                            _only_boot = set(_boot_top) - set(_con_top)
                            _only_con = set(_con_top) - set(_boot_top)
                            if _only_boot:
                                logger.info(f"In bootstrap only: {sorted(_only_boot)}")
                            if _only_con:
                                logger.info(f"In consensus only: {sorted(_only_con)}")
                    except Exception as _e:
                        logger.info(f"  Could not compare with consensus: {_e}")

            except Exception as _e:
                logger.info(f"Error: Bootstrap stability error: {_e}")
                _traceback.print_exc()

            _section_times["Bootstrap Stability"] = _time.time() - _t0
            logger.info(f"\n Bootstrap Stability completed in {_section_times['Bootstrap Stability']:.1f}s")

        # SECTION 3: Learning Curves
        if ctx.run_learning_curves:
            _t0 = _time.time()
            _section_header("Learning Curves", _SECTION_COLORS["learning"])

            try:
                _is_clf_lc = _is_classification
                _model_lc = _create_lightweight_model(_is_clf_lc, _n_features)

                _fractions = np.linspace(0.1, 1.0, ctx.n_fractions)
                _n_train_total = _X_train_np.shape[0]
                _metric_name = "AUC" if _is_clf_lc else "R²"

                logger.info(
                    f"Training {type(_model_lc).__name__} at {ctx.n_fractions} data fractions × {ctx.n_repeats} repeats"
                )
                logger.info(f"Training set size: {_n_train_total}")

                _train_scores_all = np.zeros((ctx.n_fractions, ctx.n_repeats))
                _test_scores_all = np.zeros((ctx.n_fractions, ctx.n_repeats))

                for _fi, _frac in enumerate(_fractions):
                    _n_samples = max(10, int(_frac * _n_train_total))
                    for _rep in range(ctx.n_repeats):
                        _rng = np.random.RandomState(_rep * 1000 + _fi)
                        _idx = _rng.choice(_n_train_total, size=_n_samples, replace=False)
                        _X_sub = _X_train_np[_idx]
                        _y_sub = _y_train_np[_idx]

                        _m = copy.deepcopy(_model_lc)
                        try:
                            _m.fit(_X_sub, _y_sub)
                        except Exception:
                            _m = _create_lightweight_model(_is_clf_lc, _n_features)
                            _m.fit(_X_sub, _y_sub)

                        # Train score
                        if _is_clf_lc:
                            try:
                                _tr_pred = _m.predict_proba(_X_sub)
                                if _tr_pred.ndim == 2:
                                    _tr_pred = _tr_pred[:, 1]
                                _train_scores_all[_fi, _rep] = roc_auc_score(_y_sub, _tr_pred)
                            except Exception:
                                _train_scores_all[_fi, _rep] = accuracy_score(_y_sub, _m.predict(_X_sub))

                            try:
                                _te_pred = _m.predict_proba(_X_test_np)
                                if _te_pred.ndim == 2:
                                    _te_pred = _te_pred[:, 1]
                                _test_scores_all[_fi, _rep] = roc_auc_score(_y_test_np, _te_pred)
                            except Exception:
                                _test_scores_all[_fi, _rep] = accuracy_score(_y_test_np, _m.predict(_X_test_np))
                        else:
                            _train_scores_all[_fi, _rep] = r2_score(_y_sub, _m.predict(_X_sub))
                            _test_scores_all[_fi, _rep] = r2_score(_y_test_np, _m.predict(_X_test_np))

                    if (_fi + 1) % max(1, ctx.n_fractions // 5) == 0:
                        logger.info(f"  Fraction {_fi + 1}/{ctx.n_fractions} ({_frac:.1%}, n={_n_samples}) complete")

                # Compute statistics
                _train_mean = _train_scores_all.mean(axis=1)
                _train_std = _train_scores_all.std(axis=1)
                _test_mean = _test_scores_all.mean(axis=1)
                _test_std = _test_scores_all.std(axis=1)

                # 95% CI
                _ci_mult = 1.96
                _train_ci_lo = _train_mean - _ci_mult * _train_std / np.sqrt(ctx.n_repeats)
                _train_ci_hi = _train_mean + _ci_mult * _train_std / np.sqrt(ctx.n_repeats)
                _test_ci_lo = _test_mean - _ci_mult * _test_std / np.sqrt(ctx.n_repeats)
                _test_ci_hi = _test_mean + _ci_mult * _test_std / np.sqrt(ctx.n_repeats)

                _sample_sizes = (np.array(_fractions) * _n_train_total).astype(int)
                _sample_sizes = np.maximum(_sample_sizes, 10)

                # Convergence Detection
                _test_diffs = np.diff(_test_mean)
                _converged = False
                _convergence_idx = None
                _convergence_threshold = 0.005  # 0.5% improvement

                for _ci in range(len(_test_diffs) - 2):
                    if all(abs(_test_diffs[_ci + j]) < _convergence_threshold for j in range(3)):
                        _converged = True
                        _convergence_idx = _ci
                        break

                # Elbow detection via max curvature
                _elbow_idx = None
                if len(_test_mean) >= 4:
                    _second_diff = np.diff(_test_mean, n=2)
                    if len(_second_diff) > 0:
                        _elbow_idx = np.argmax(np.abs(_second_diff)) + 1

                # 3a. Learning Curve Plot
                _subsection_header("Learning Curve with 95% CI")

                fig, ax = plt.subplots(figsize=(12, 7))

                # CI bands
                ax.fill_between(
                    _sample_sizes, _train_ci_lo, _train_ci_hi, alpha=0.15, color="#2ecc71", label="Train 95% CI"
                )
                ax.fill_between(
                    _sample_sizes, _test_ci_lo, _test_ci_hi, alpha=0.15, color="#e74c3c", label="Test 95% CI"
                )

                # Mean lines
                ax.plot(
                    _sample_sizes,
                    _train_mean,
                    "o-",
                    color="#2ecc71",
                    linewidth=2,
                    markersize=5,
                    label=f"Train {_metric_name} (mean)",
                )
                ax.plot(
                    _sample_sizes,
                    _test_mean,
                    "s-",
                    color="#e74c3c",
                    linewidth=2,
                    markersize=5,
                    label=f"Test {_metric_name} (mean)",
                )

                # Gap annotation
                _gap = _train_mean - _test_mean
                _final_gap = _gap[-1]
                ax.annotate(
                    f"Gap = {_final_gap:.4f}",
                    xy=(_sample_sizes[-1], (_train_mean[-1] + _test_mean[-1]) / 2),
                    xytext=(15, 0),
                    textcoords="offset points",
                    fontsize=10,
                    ha="left",
                    va="center",
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
                )

                # Convergence marker
                if _converged and _convergence_idx is not None:
                    ax.axvline(
                        _sample_sizes[_convergence_idx],
                        color="purple",
                        linestyle="--",
                        alpha=0.5,
                        label=f"Convergence (n≈{_sample_sizes[_convergence_idx]})",
                    )

                # Elbow marker
                if _elbow_idx is not None and _elbow_idx < len(_sample_sizes):
                    ax.axvline(
                        _sample_sizes[_elbow_idx],
                        color="orange",
                        linestyle=":",
                        alpha=0.5,
                        label=f"Elbow (n≈{_sample_sizes[_elbow_idx]})",
                    )

                ax.set_xlabel("Training Set Size", fontsize=12)
                ax.set_ylabel(_metric_name, fontsize=12)
                ax.set_title(
                    f"Learning Curve -- {type(_model_lc).__name__} ({ctx.n_repeats} repeats)",
                    fontweight="bold",
                    fontsize=14,
                )
                ax.legend(loc="lower right", frameon=True, framealpha=0.9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

                # 3b. Marginal Gains
                if ctx.show_marginal_gains and len(_test_mean) >= 3:
                    _subsection_header("Marginal Gains per Data Increment")
                    _marginal = np.diff(_test_mean)
                    _midpoints = (_sample_sizes[:-1] + _sample_sizes[1:]) / 2

                    fig, ax = plt.subplots(figsize=(10, 4))
                    _bar_colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in _marginal]
                    ax.bar(
                        _midpoints,
                        _marginal,
                        width=(_sample_sizes[1] - _sample_sizes[0]) * 0.6,
                        color=_bar_colors,
                        edgecolor="white",
                        linewidth=0.5,
                        alpha=0.8,
                    )
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.axhline(
                        _convergence_threshold,
                        color="purple",
                        linestyle="--",
                        alpha=0.4,
                        label=f"Convergence threshold ({_convergence_threshold})",
                    )
                    ax.axhline(-_convergence_threshold, color="purple", linestyle="--", alpha=0.4)
                    ax.set_xlabel("Training Set Size (midpoint)")
                    ax.set_ylabel(f"Δ{_metric_name}")
                    ax.set_title("Marginal Test Performance Gain per Data Increment", fontweight="bold")
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

                # 3c. Convergence Assessment Box
                _subsection_header("Convergence Assessment")
                _assess_rows = [
                    {"Metric": "Final Train Score", "Value": f"{_train_mean[-1]:.4f}"},
                    {"Metric": "Final Test Score", "Value": f"{_test_mean[-1]:.4f}"},
                    {"Metric": "Train-Test Gap", "Value": f"{_final_gap:.4f}"},
                    {"Metric": "Converged?", "Value": "Yes" if _converged else "No"},
                ]
                if _converged and _convergence_idx is not None:
                    _assess_rows.append(
                        {
                            "Metric": "Convergence Point",
                            "Value": f"n ≈ {_sample_sizes[_convergence_idx]} ({_fractions[_convergence_idx]:.0%} of data)",
                        }
                    )
                if _elbow_idx is not None and _elbow_idx < len(_sample_sizes):
                    _assess_rows.append(
                        {
                            "Metric": "Elbow Point",
                            "Value": f"n ≈ {_sample_sizes[_elbow_idx]} ({_fractions[_elbow_idx]:.0%} of data)",
                        }
                    )

                # Bias-variance diagnosis
                if _final_gap > 0.15:
                    _bv_diagnosis = "HIGH VARIANCE -- model overfits; more data or regularization may help"
                elif _final_gap > 0.05:
                    _bv_diagnosis = "MODERATE VARIANCE -- some overfitting present"
                elif _test_mean[-1] < 0.3:
                    _bv_diagnosis = "HIGH BIAS -- model underfits; consider more complexity or features"
                else:
                    _bv_diagnosis = "GOOD FIT -- balanced bias-variance tradeoff"
                _assess_rows.append({"Metric": "Bias-Variance Diagnosis", "Value": _bv_diagnosis})

                # Score improvement from 50% to 100% data
                if ctx.n_fractions >= 5:
                    _half_idx = ctx.n_fractions // 2
                    _improvement = _test_mean[-1] - _test_mean[_half_idx]
                    _assess_rows.append(
                        {
                            "Metric": f"Gain from {_fractions[_half_idx]:.0%} → 100% data",
                            "Value": f"{_improvement:+.4f}",
                        }
                    )

                _safe_html_table(pd.DataFrame(_assess_rows), caption="Learning Curve Summary")

            except Exception as _e:
                logger.info(f"Error: Learning curves error: {_e}")
                _traceback.print_exc()

            _section_times["Learning Curves"] = _time.time() - _t0
            logger.info(f"\n Learning Curves completed in {_section_times['Learning Curves']:.1f}s")

        # SECTION 4: Ensemble-Weighted SHAP Beeswarm
        if ctx.run_shap_beeswarm:
            _t0 = _time.time()
            _section_header("Ensemble-Weighted SHAP Beeswarm", _SECTION_COLORS["shap"])

            try:
                _shap_available = True
                try:
                    import shap
                except ImportError:
                    _shap_available = False
                    display(
                        HTML(
                            '<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; '
                            'padding:12px; margin:8px 0;">'
                            "<strong>Warning: SHAP library not available.</strong> "
                            "Install with <code>pip install shap</code>.</div>"
                        )
                    )

                _has_model_shap = (
                    "last_fitted_model" in ctx
                    and ctx.last_fitted_model is not None
                    and hasattr(ctx.last_fitted_model, "models")
                )

                if _shap_available and _has_model_shap:
                    _base_models_shap = ctx.last_fitted_model.models
                    _shap_model_names = list(_base_models_shap.keys())

                    # Get meta-learner weights for weighting SHAP values
                    _shap_weights = {}
                    if hasattr(ctx.last_fitted_model, "meta_model") and hasattr(
                        ctx.last_fitted_model.meta_model, "coef_"
                    ):
                        _coefs = np.asarray(ctx.last_fitted_model.meta_model.coef_).ravel()
                        for _i, _mn in enumerate(_shap_model_names):
                            if _i < len(_coefs):
                                _shap_weights[_mn] = _coefs[_i]
                    elif "model_weights" in ctx and ctx.model_weights is not None:
                        _shap_weights = dict(ctx.model_weights)

                    # Normalize weights to sum to 1 (absolute values)
                    if _shap_weights:
                        _total_w = sum(abs(v) for v in _shap_weights.values())
                        if _total_w > 0:
                            _shap_weights = {k: abs(v) / _total_w for k, v in _shap_weights.items()}
                    else:
                        # Equal weights fallback
                        _n_shap_models = len(_shap_model_names)
                        _shap_weights = {mn: 1.0 / _n_shap_models for mn in _shap_model_names}

                    logger.info(f"Computing SHAP values for {len(_shap_model_names)} base models...")
                    logger.info(f"Weights: {', '.join(f'{k}: {v:.3f}' for k, v in _shap_weights.items())}")

                    # Use a subsample for SHAP computation efficiency
                    _shap_max_samples = min(200, _X_test_np.shape[0])
                    _shap_idx = np.random.RandomState(42).choice(
                        _X_test_np.shape[0], size=_shap_max_samples, replace=False
                    )
                    _X_shap = _X_test_np[_shap_idx]
                    _X_shap_bg = _X_train_np[: min(100, _X_train_np.shape[0])]

                    _per_model_shap = {}
                    _excluded_models = []

                    for _mn in _shap_model_names:
                        _model_obj = _base_models_shap[_mn]

                        # Skip TabPFN (not supported by standard SHAP explainers)
                        _model_class_name = type(_model_obj).__name__
                        if "tabpfn" in _model_class_name.lower() or "TabPFN" in _model_class_name:
                            logger.info(f"  Skipping {_mn} (TabPFN not supported by SHAP)")
                            _excluded_models.append(_mn)
                            continue

                        try:
                            # Try TreeExplainer first (fast for tree-based models)
                            _is_tree = any(
                                hasattr(_model_obj, attr)
                                for attr in ["feature_importances_", "get_booster", "get_feature_importance"]
                            ) and hasattr(_model_obj, "predict")

                            _tree_types = (
                                "catboost",
                                "xgboost",
                                "lgbm",
                                "lightgbm",
                                "randomforest",
                                "gradientboosting",
                                "extratrees",
                                "decisiontree",
                            )
                            _is_tree = _is_tree or any(t in _model_class_name.lower() for t in _tree_types)

                            if _is_tree:
                                try:
                                    _explainer = shap.TreeExplainer(_model_obj)
                                    _sv = _explainer.shap_values(_X_shap)
                                    if isinstance(_sv, list):
                                        _sv = _sv[1] if len(_sv) == 2 else _sv[0]
                                    _per_model_shap[_mn] = np.asarray(_sv)
                                    logger.info(f"  {_mn}: TreeExplainer OK, shape {_per_model_shap[_mn].shape}")
                                    continue
                                except Exception as _tree_err:
                                    logger.info(
                                        f"  {_mn}: TreeExplainer failed ({_tree_err}), trying LinearExplainer..."
                                    )

                            # Try LinearExplainer
                            _is_linear = hasattr(_model_obj, "coef_")
                            if _is_linear:
                                try:
                                    _explainer = shap.LinearExplainer(_model_obj, _X_shap_bg)
                                    _sv = _explainer.shap_values(_X_shap)
                                    if isinstance(_sv, list):
                                        _sv = _sv[1] if len(_sv) == 2 else _sv[0]
                                    _per_model_shap[_mn] = np.asarray(_sv)
                                    logger.info(f"  {_mn}: LinearExplainer OK, shape {_per_model_shap[_mn].shape}")
                                    continue
                                except Exception as _lin_err:
                                    logger.info(
                                        f"  {_mn}: LinearExplainer failed ({_lin_err}), trying KernelExplainer..."
                                    )

                            # Fallback: KernelExplainer (slow but universal)
                            _predict_fn = _model_obj.predict
                            if _is_classification and hasattr(_model_obj, "predict_proba"):
                                _predict_fn = lambda x, _m=_model_obj: _m.predict_proba(x)[:, 1]

                            _bg = shap.kmeans(_X_shap_bg, min(10, _X_shap_bg.shape[0]))
                            _explainer = shap.KernelExplainer(_predict_fn, _bg)
                            _sv = _explainer.shap_values(_X_shap, nsamples=100)
                            if isinstance(_sv, list):
                                _sv = _sv[1] if len(_sv) == 2 else _sv[0]
                            _per_model_shap[_mn] = np.asarray(_sv)
                            logger.info(f"  {_mn}: KernelExplainer OK, shape {_per_model_shap[_mn].shape}")

                        except Exception as _e:
                            logger.info(f"  {_mn}: All SHAP methods failed: {_e}")
                            _excluded_models.append(_mn)

                    if len(_per_model_shap) == 0:
                        display(
                            HTML(
                                '<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; '
                                'padding:12px; margin:8px 0;">'
                                "<strong>Warning: Could not compute SHAP values for any model.</strong></div>"
                            )
                        )
                    else:
                        # Compute weighted SHAP
                        _weighted_shap = np.zeros((_shap_max_samples, _n_features))
                        _total_weight_used = 0
                        for _mn, _sv in _per_model_shap.items():
                            _w = _shap_weights.get(_mn, 0)
                            if _sv.shape == _weighted_shap.shape:
                                _weighted_shap += _w * _sv
                                _total_weight_used += _w

                        if _total_weight_used > 0:
                            _weighted_shap /= _total_weight_used

                        # Feature importance ranking by mean |SHAP|
                        _mean_abs_shap = np.mean(np.abs(_weighted_shap), axis=0)
                        _shap_order = np.argsort(-_mean_abs_shap)
                        _top_shap_idx = _shap_order[: ctx.top_n_shap]

                        # 4a. Manual Beeswarm Plot
                        _subsection_header(f"Ensemble-Weighted SHAP Beeswarm (Top {ctx.top_n_shap})")

                        _n_show = min(ctx.top_n_shap, len(_top_shap_idx))
                        _shap_display = _weighted_shap[:, _top_shap_idx[:_n_show]]
                        _feat_display = [_feat_names[i] for i in _top_shap_idx[:_n_show]]
                        _feat_display_names = [
                            _get_display_name(fn) if ctx.use_display_names else fn for fn in _feat_display
                        ]
                        _X_shap_display = _X_shap[:, _top_shap_idx[:_n_show]]

                        # Beeswarm: each feature on y-axis, SHAP value on x-axis, color by feature value
                        fig_width = 12 if not ctx.show_weight_chart else 15
                        if ctx.show_weight_chart:
                            fig, (ax_bee, ax_w) = plt.subplots(
                                1,
                                2,
                                figsize=(fig_width, max(7, _n_show * 0.35)),
                                gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05},
                            )
                        else:
                            fig, ax_bee = plt.subplots(figsize=(12, max(7, _n_show * 0.35)))

                        _cmap_beeswarm = plt.cm.RdBu_r
                        for _fi in range(_n_show):
                            _y_pos = _n_show - 1 - _fi
                            _shap_vals = _shap_display[:, _fi]
                            _feat_vals = _X_shap_display[:, _fi]

                            # Normalize feature values to [0, 1] for coloring
                            _fmin, _fmax = np.nanmin(_feat_vals), np.nanmax(_feat_vals)
                            if _fmax > _fmin:
                                _normed = (_feat_vals - _fmin) / (_fmax - _fmin)
                            else:
                                _normed = np.full_like(_feat_vals, 0.5)

                            # Add jitter to y position
                            _jitter = np.random.RandomState(42 + _fi).uniform(-0.25, 0.25, size=len(_shap_vals))
                            _y_jittered = _y_pos + _jitter

                            # Sort by SHAP value so high-value points are drawn on top
                            _sort_idx = np.argsort(np.abs(_shap_vals))
                            ax_bee.scatter(
                                _shap_vals[_sort_idx],
                                _y_jittered[_sort_idx],
                                c=_normed[_sort_idx],
                                cmap=_cmap_beeswarm,
                                s=8,
                                alpha=0.6,
                                edgecolors="none",
                                rasterized=True,
                            )

                        ax_bee.set_yticks(range(_n_show))
                        ax_bee.set_yticklabels(list(reversed(_feat_display_names)), fontsize=9)
                        ax_bee.set_xlabel("SHAP Value (ensemble-weighted)", fontsize=11)
                        ax_bee.axvline(0, color="gray", linewidth=0.8, linestyle="-")
                        ax_bee.set_title(
                            f"Ensemble-Weighted SHAP Beeswarm (Top {_n_show})", fontweight="bold", fontsize=13
                        )

                        # Colorbar
                        _sm = plt.cm.ScalarMappable(cmap=_cmap_beeswarm, norm=plt.Normalize(0, 1))
                        _sm.set_array([])
                        _cbar = plt.colorbar(_sm, ax=ax_bee, shrink=0.5, aspect=30, pad=0.02)
                        _cbar.set_label("Feature Value (normalized)", fontsize=10)
                        _cbar.set_ticks([0, 0.5, 1])
                        _cbar.set_ticklabels(["Low", "Mid", "High"])

                        # 4b. Weight sidebar
                        if ctx.show_weight_chart:
                            _shap_importance = _mean_abs_shap[_top_shap_idx[:_n_show]]
                            _shap_imp_reversed = list(reversed(_shap_importance))

                            ax_w.barh(
                                range(_n_show),
                                _shap_imp_reversed,
                                color=_SECTION_COLORS["shap"],
                                alpha=0.7,
                                edgecolor="white",
                            )
                            ax_w.set_yticks([])
                            ax_w.set_xlabel("Mean |SHAP|", fontsize=10)
                            ax_w.set_title("Importance", fontweight="bold", fontsize=10)

                        plt.tight_layout()
                        plt.show()

                        # 4c. SHAP Summary Table
                        _subsection_header("SHAP Feature Importance Summary")
                        _shap_table_rows = []
                        for _rank, _gi in enumerate(_top_shap_idx[:_n_show]):
                            _fn = _feat_names[_gi]
                            _dn = _get_display_name(_fn) if ctx.use_display_names else _fn
                            _shap_table_rows.append(
                                {
                                    "Rank": _rank + 1,
                                    "Feature": _dn,
                                    "Mean |SHAP|": _mean_abs_shap[_gi],
                                    "Max |SHAP|": np.max(np.abs(_weighted_shap[:, _gi])),
                                    "Std SHAP": np.std(_weighted_shap[:, _gi]),
                                }
                            )
                        _shap_table_df = pd.DataFrame(_shap_table_rows).round(4)
                        _safe_html_table(_shap_table_df, caption=f"Top {_n_show} Ensemble-Weighted SHAP Features")

                        # Report excluded models
                        if _excluded_models:
                            logger.info(f"\nModels excluded from SHAP: {', '.join(_excluded_models)}")
                        logger.info(f"Models included: {', '.join(_per_model_shap.keys())}")
                        logger.info(f"SHAP computed on {_shap_max_samples} test samples")

            except Exception as _e:
                logger.info(f"Error: SHAP Beeswarm error: {_e}")
                _traceback.print_exc()

            _section_times["SHAP Beeswarm"] = _time.time() - _t0
            logger.info(f"\n SHAP Beeswarm completed in {_section_times['SHAP Beeswarm']:.1f}s")

        # SECTION 5: Supplementary Quick Analyses
        if ctx.run_supplementary:
            _t0 = _time.time()
            _section_header("Supplementary Quick Analyses", _SECTION_COLORS["supplementary"])

            # 5A: Effect Size Summary
            if ctx.run_effect_sizes:
                _subsection_header("5A. Effect Size Summary")

                try:
                    # Get current target info
                    _current_target = None
                    if "tp_option" in ctx and ctx.tp_option is not None:
                        _current_target = ctx.tp_option
                    elif "target_options" in ctx and ctx.target_options is not None and len(ctx.target_options) > 0:
                        _current_target = ctx.target_options[0]

                    logger.info(f"Current target: {_current_target}")

                    # Compute effect sizes for current target
                    _effect_rows = []

                    if not _is_classification:
                        # Cohen's f² from R²
                        _model_for_pred = None
                        if "last_fitted_model" in ctx and ctx.last_fitted_model is not None:
                            _model_for_pred = ctx.last_fitted_model

                        if _model_for_pred is not None:
                            try:
                                _y_pred_eff = _model_for_pred.predict(_X_test_np).ravel()
                                _r2_eff = r2_score(_y_test_np, _y_pred_eff)
                                _f2 = _r2_eff / (1 - _r2_eff) if _r2_eff < 1.0 else float("inf")

                                # Cohen's f² benchmarks: 0.02 small, 0.15 medium, 0.35 large
                                if _f2 >= 0.35:
                                    _f2_label = "Large"
                                elif _f2 >= 0.15:
                                    _f2_label = "Medium"
                                elif _f2 >= 0.02:
                                    _f2_label = "Small"
                                else:
                                    _f2_label = "Negligible"

                                _effect_rows.append(
                                    {
                                        "Target": _current_target or "Current",
                                        "R²": round(_r2_eff, 4),
                                        "Cohen's f²": round(_f2, 4),
                                        "Effect Size": _f2_label,
                                    }
                                )
                            except Exception as _e_eff:
                                logger.info(f"  Could not compute effect size: {_e_eff}")

                        # Also compute Cohen's d between predicted and actual
                        if _model_for_pred is not None:
                            try:
                                _y_pred_d = _model_for_pred.predict(_X_test_np).ravel()
                                _pooled_std = np.sqrt((np.std(_y_test_np) ** 2 + np.std(_y_pred_d) ** 2) / 2)
                                if _pooled_std > 0:
                                    _cohens_d = (np.mean(_y_pred_d) - np.mean(_y_test_np)) / _pooled_std
                                else:
                                    _cohens_d = 0.0
                                logger.info(f"  Cohen's d (predicted vs actual means): {_cohens_d:.4f}")
                            except Exception:
                                pass

                    else:
                        # Classification: compute effect size from AUC
                        if "last_fitted_model" in ctx and ctx.last_fitted_model is not None:
                            try:
                                if hasattr(ctx.last_fitted_model, "predict_proba"):
                                    _y_prob = ctx.last_fitted_model.predict_proba(_X_test_np)
                                    if _y_prob.ndim == 2:
                                        _y_prob = _y_prob[:, 1]
                                else:
                                    _y_prob = ctx.last_fitted_model.predict(_X_test_np)
                                _auc_eff = roc_auc_score(_y_test_np, _y_prob)
                                # Convert AUC to Cohen's d (approximate: d ≈ sqrt(2) * Φ⁻¹(AUC))
                                _cohens_d_from_auc = np.sqrt(2) * stats.norm.ppf(_auc_eff)

                                if abs(_cohens_d_from_auc) >= 0.8:
                                    _d_label = "Large"
                                elif abs(_cohens_d_from_auc) >= 0.5:
                                    _d_label = "Medium"
                                elif abs(_cohens_d_from_auc) >= 0.2:
                                    _d_label = "Small"
                                else:
                                    _d_label = "Negligible"

                                _effect_rows.append(
                                    {
                                        "Target": _current_target or "Current",
                                        "AUC": round(_auc_eff, 4),
                                        "Cohen's d (from AUC)": round(_cohens_d_from_auc, 4),
                                        "Effect Size": _d_label,
                                    }
                                )
                            except Exception as _e_eff:
                                logger.info(f"  Could not compute classification effect size: {_e_eff}")

                    # Batch mode: process other targets from _batch_archive
                    if (
                        "_batch_archive" in ctx
                        and ctx._batch_archive is not None
                        and isinstance(ctx._batch_archive, dict)
                    ):
                        logger.info(f"  Processing batch archive ({len(ctx._batch_archive)} targets)...")
                        for _bt, _bdata in ctx._batch_archive.items():
                            if _bt == _current_target:
                                continue
                            try:
                                _b_metrics = None
                                if isinstance(_bdata, dict):
                                    _b_metrics = _bdata.get("metrics", _bdata.get("metrics_dict_full", None))
                                    if _b_metrics is None and "r2" in _bdata:
                                        _b_metrics = _bdata

                                if _b_metrics is not None:
                                    _b_r2 = None
                                    _b_auc = None
                                    for _key in ["r2", "R²", "test_r2", "R2", "r2_score"]:
                                        if _key in _b_metrics:
                                            _b_r2 = _b_metrics[_key]
                                            break
                                    for _key in ["auc", "AUC", "test_auc", "roc_auc"]:
                                        if _key in _b_metrics:
                                            _b_auc = _b_metrics[_key]
                                            break

                                    if _b_r2 is not None:
                                        _b_f2 = _b_r2 / (1 - _b_r2) if _b_r2 < 1.0 else float("inf")
                                        if _b_f2 >= 0.35:
                                            _b_f2_label = "Large"
                                        elif _b_f2 >= 0.15:
                                            _b_f2_label = "Medium"
                                        elif _b_f2 >= 0.02:
                                            _b_f2_label = "Small"
                                        else:
                                            _b_f2_label = "Negligible"
                                        _effect_rows.append(
                                            {
                                                "Target": _bt,
                                                "R²": round(_b_r2, 4),
                                                "Cohen's f²": round(_b_f2, 4),
                                                "Effect Size": _b_f2_label,
                                            }
                                        )
                                    elif _b_auc is not None:
                                        _b_d = np.sqrt(2) * stats.norm.ppf(max(0.501, min(0.999, _b_auc)))
                                        if abs(_b_d) >= 0.8:
                                            _b_d_label = "Large"
                                        elif abs(_b_d) >= 0.5:
                                            _b_d_label = "Medium"
                                        elif abs(_b_d) >= 0.2:
                                            _b_d_label = "Small"
                                        else:
                                            _b_d_label = "Negligible"
                                        _effect_rows.append(
                                            {
                                                "Target": _bt,
                                                "AUC": round(_b_auc, 4),
                                                "Cohen's d (from AUC)": round(_b_d, 4),
                                                "Effect Size": _b_d_label,
                                            }
                                        )
                            except Exception as _e_batch:
                                logger.info(f"  Batch target {_bt}: {_e_batch}")

                    if _effect_rows:
                        _effect_df = pd.DataFrame(_effect_rows)
                        _safe_html_table(_effect_df, caption="Effect Size Summary Across Targets")

                        # Effect size bar chart
                        _metric_col_eff = "Cohen's f²" if "Cohen's f²" in _effect_df.columns else "Cohen's d (from AUC)"
                        if _metric_col_eff in _effect_df.columns:
                            _eff_sorted = _effect_df.dropna(subset=[_metric_col_eff]).sort_values(
                                _metric_col_eff, ascending=True
                            )
                            if len(_eff_sorted) > 0:
                                fig, ax = plt.subplots(figsize=(10, max(3, len(_eff_sorted) * 0.5)))
                                _eff_colors = []
                                for _es in _eff_sorted["Effect Size"]:
                                    if _es == "Large":
                                        _eff_colors.append("#27ae60")
                                    elif _es == "Medium":
                                        _eff_colors.append("#f39c12")
                                    elif _es == "Small":
                                        _eff_colors.append("#e67e22")
                                    else:
                                        _eff_colors.append("#e74c3c")

                                ax.barh(
                                    _eff_sorted["Target"],
                                    _eff_sorted[_metric_col_eff],
                                    color=_eff_colors,
                                    edgecolor="white",
                                    linewidth=0.5,
                                )

                                # Reference lines for Cohen's f² thresholds
                                if "f²" in _metric_col_eff:
                                    for _thresh, _lbl in [(0.02, "Small"), (0.15, "Medium"), (0.35, "Large")]:
                                        ax.axvline(_thresh, color="gray", linestyle=":", alpha=0.4)
                                        ax.text(
                                            _thresh, len(_eff_sorted) - 0.5, _lbl, fontsize=8, color="gray", ha="center"
                                        )

                                ax.set_xlabel(_metric_col_eff)
                                ax.set_title("Effect Sizes Across Targets", fontweight="bold")
                                plt.tight_layout()
                                plt.show()
                    else:
                        logger.info("  No effect size data available.")

                except Exception as _e:
                    logger.info(f"Error: Effect size error: {_e}")
                    _traceback.print_exc()

            # 5B: Feature Overlap Across Targets
            if ctx.run_feature_overlap:
                _subsection_header("5B. Feature Overlap Across Targets")

                try:
                    _target_features = {}

                    # Current target features from consensus_df
                    if "consensus_df" in ctx and ctx.consensus_df is not None:
                        _con = ctx.consensus_df.copy()
                        _con_feat_col = None
                        for _c in ["feature", "Feature", "feature_name", "Feature_Name"]:
                            if _c in _con.columns:
                                _con_feat_col = _c
                                break
                        if _con_feat_col is not None:
                            _current_target_fo = None
                            if "tp_option" in ctx and ctx.tp_option is not None:
                                _current_target_fo = ctx.tp_option
                            elif (
                                "target_options" in ctx
                                and ctx.target_options is not None
                                and len(ctx.target_options) > 0
                            ):
                                _current_target_fo = ctx.target_options[0]
                            if _current_target_fo:
                                _target_features[_current_target_fo] = _con.head(ctx.top_k_overlap)[
                                    _con_feat_col
                                ].tolist()

                    # Batch targets from _batch_archive
                    if (
                        "_batch_archive" in ctx
                        and ctx._batch_archive is not None
                        and isinstance(ctx._batch_archive, dict)
                    ):
                        for _bt, _bdata in ctx._batch_archive.items():
                            if _bt in _target_features:
                                continue
                            try:
                                _b_con = None
                                if isinstance(_bdata, dict):
                                    _b_con = _bdata.get("consensus_df", None)
                                    if _b_con is None:
                                        _b_con = _bdata.get("consensus", None)
                                    if _b_con is None and "results_summary" in _bdata:
                                        _rs = _bdata["results_summary"]
                                        if isinstance(_rs, pd.DataFrame):
                                            _b_con = _rs

                                if _b_con is not None and isinstance(_b_con, pd.DataFrame):
                                    _b_feat_col = None
                                    for _c in ["feature", "Feature", "feature_name", "Feature_Name"]:
                                        if _c in _b_con.columns:
                                            _b_feat_col = _c
                                            break
                                    if _b_feat_col is not None:
                                        _target_features[_bt] = _b_con.head(ctx.top_k_overlap)[_b_feat_col].tolist()
                            except Exception:
                                pass

                    if len(_target_features) < 2:
                        logger.info(
                            f"  Only {len(_target_features)} target(s) with feature lists found. "
                            "Need ≥ 2 for overlap analysis."
                        )
                        if _target_features:
                            logger.info(f"  Available: {list(_target_features.keys())}")
                    else:
                        _overlap_targets = list(_target_features.keys())
                        _n_targets = len(_overlap_targets)
                        logger.info(
                            f"  Comparing top-{ctx.top_k_overlap} features across {_n_targets} targets: "
                            f"{', '.join(_overlap_targets)}"
                        )

                        # Compute Jaccard similarity matrix
                        _jaccard_matrix = np.zeros((_n_targets, _n_targets))
                        for _i in range(_n_targets):
                            for _j in range(_n_targets):
                                _set_i = set(_target_features[_overlap_targets[_i]])
                                _set_j = set(_target_features[_overlap_targets[_j]])
                                _union = _set_i | _set_j
                                _jaccard_matrix[_i, _j] = len(_set_i & _set_j) / len(_union) if len(_union) > 0 else 0

                        _jaccard_df = pd.DataFrame(
                            _jaccard_matrix, index=_overlap_targets, columns=_overlap_targets
                        ).round(3)

                        # Heatmap
                        fig, ax = plt.subplots(figsize=(max(6, _n_targets * 1.2), max(5, _n_targets * 1.0)))
                        sns.heatmap(
                            _jaccard_df,
                            annot=True,
                            fmt=".3f",
                            cmap="YlGnBu",
                            vmin=0,
                            vmax=1,
                            linewidths=0.5,
                            linecolor="white",
                            cbar_kws={"label": "Jaccard Similarity", "shrink": 0.8},
                            ax=ax,
                            square=True,
                        )
                        ax.set_title(
                            f"Feature Overlap (Top {ctx.top_k_overlap}) -- Jaccard Similarity", fontweight="bold"
                        )
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        plt.show()

                        _safe_html_table(_jaccard_df, caption="Pairwise Jaccard Similarity")

                        # Find universal features (in all targets)
                        _all_feat_sets = [set(_target_features[t]) for t in _overlap_targets]
                        _universal = _all_feat_sets[0]
                        for _s in _all_feat_sets[1:]:
                            _universal = _universal & _s

                        if _universal:
                            logger.info(
                                f"\n  Universal features (in all {_n_targets} targets): "
                                f"{len(_universal)}/{ctx.top_k_overlap}"
                            )
                            for _uf in sorted(_universal):
                                _dn = _get_display_name(_uf) if ctx.use_display_names else _uf
                                logger.info(f"    • {_dn}")
                        else:
                            logger.info(f"\n  No features appear in all {_n_targets} targets' top-{ctx.top_k_overlap}.")

                        # Unique features (only in one target)
                        _unique_per_target = {}
                        for _ti, _tn in enumerate(_overlap_targets):
                            _others = set()
                            for _tj in range(_n_targets):
                                if _ti != _tj:
                                    _others |= _all_feat_sets[_tj]
                            _unique_per_target[_tn] = _all_feat_sets[_ti] - _others

                        _has_unique = any(len(v) > 0 for v in _unique_per_target.values())
                        if _has_unique:
                            logger.info("\n  Target-unique features:")
                            for _tn, _ufs in _unique_per_target.items():
                                if _ufs:
                                    _dn_list = [
                                        _get_display_name(f) if ctx.use_display_names else f for f in sorted(_ufs)
                                    ]
                                    logger.info(f"    {_tn}: {', '.join(_dn_list)}")

                except Exception as _e:
                    logger.info(f"Error: Feature overlap error: {_e}")
                    _traceback.print_exc()

            # 5C: Prediction Distribution Diagnostics
            if ctx.run_prediction_diagnostics:
                _subsection_header("5C. Prediction Distribution Diagnostics")

                try:
                    _has_ensemble = (
                        "last_fitted_model" in ctx
                        and ctx.last_fitted_model is not None
                        and hasattr(ctx.last_fitted_model, "predict")
                    )

                    if not _has_ensemble:
                        logger.info("  last_fitted_model not available for prediction diagnostics.")
                    else:
                        try:
                            _y_pred_diag = ctx.last_fitted_model.predict(_X_test_np).ravel()
                        except Exception:
                            _y_pred_diag = None

                        if _y_pred_diag is not None:
                            _residuals = _y_test_np - _y_pred_diag

                            if not _is_classification:
                                # 4-panel diagnostic plot
                                fig, axes = plt.subplots(2, 2, figsize=(14, 11))

                                # Panel 1: Predicted vs Actual
                                ax = axes[0, 0]
                                ax.scatter(
                                    _y_test_np,
                                    _y_pred_diag,
                                    alpha=0.5,
                                    s=25,
                                    edgecolors="none",
                                    c=_SECTION_COLORS["supplementary"],
                                )
                                _lims = [
                                    min(_y_test_np.min(), _y_pred_diag.min()),
                                    max(_y_test_np.max(), _y_pred_diag.max()),
                                ]
                                _pad = (_lims[1] - _lims[0]) * 0.05
                                _lims = [_lims[0] - _pad, _lims[1] + _pad]
                                ax.plot(_lims, _lims, "k--", alpha=0.5, linewidth=1, label="Identity")
                                ax.set_xlim(_lims)
                                ax.set_ylim(_lims)
                                ax.set_xlabel("Actual", fontsize=11)
                                ax.set_ylabel("Predicted", fontsize=11)
                                ax.set_title("Predicted vs Actual", fontweight="bold")
                                ax.legend()

                                # Add R² annotation
                                _r2_diag = r2_score(_y_test_np, _y_pred_diag)
                                ax.text(
                                    0.05,
                                    0.95,
                                    f"R² = {_r2_diag:.4f}",
                                    transform=ax.transAxes,
                                    fontsize=11,
                                    verticalalignment="top",
                                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                                )

                                # Panel 2: Residual Distribution
                                ax = axes[0, 1]
                                ax.hist(
                                    _residuals,
                                    bins=30,
                                    color=_SECTION_COLORS["supplementary"],
                                    edgecolor="white",
                                    alpha=0.8,
                                    density=True,
                                )
                                # Overlay normal fit
                                _res_mean = np.mean(_residuals)
                                _res_std = np.std(_residuals)
                                _x_norm = np.linspace(_residuals.min(), _residuals.max(), 100)
                                _y_norm = stats.norm.pdf(_x_norm, _res_mean, _res_std)
                                ax.plot(_x_norm, _y_norm, "r-", linewidth=2, label="Normal fit")
                                ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
                                ax.set_xlabel("Residual", fontsize=11)
                                ax.set_ylabel("Density", fontsize=11)
                                ax.set_title("Residual Distribution", fontweight="bold")
                                ax.legend()

                                # Shapiro-Wilk test (subsample if too large)
                                _sw_sample = _residuals[: min(5000, len(_residuals))]
                                try:
                                    _sw_stat, _sw_p = stats.shapiro(_sw_sample)
                                    ax.text(
                                        0.05,
                                        0.95,
                                        f"Shapiro-Wilk p={_sw_p:.4f}\n{'Normal' if _sw_p > 0.05 else 'Non-normal'}",
                                        transform=ax.transAxes,
                                        fontsize=9,
                                        verticalalignment="top",
                                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                                    )
                                except Exception:
                                    pass

                                # Panel 3: Q-Q Plot
                                ax = axes[1, 0]
                                stats.probplot(_residuals, dist="norm", plot=ax)
                                ax.set_title("Q-Q Plot (Residuals)", fontweight="bold")
                                ax.get_lines()[0].set_markerfacecolor(_SECTION_COLORS["supplementary"])
                                ax.get_lines()[0].set_alpha(0.5)
                                ax.get_lines()[0].set_markersize(4)

                                # Panel 4: Residuals vs Predicted
                                ax = axes[1, 1]
                                ax.scatter(
                                    _y_pred_diag,
                                    _residuals,
                                    alpha=0.5,
                                    s=25,
                                    edgecolors="none",
                                    c=_SECTION_COLORS["supplementary"],
                                )
                                ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                                # Add LOWESS trend line
                                try:
                                    from statsmodels.nonparametric.smoothers_lowess import lowess

                                    _lowess_result = lowess(_residuals, _y_pred_diag, frac=0.3)
                                    ax.plot(
                                        _lowess_result[:, 0], _lowess_result[:, 1], "r-", linewidth=2, label="LOWESS"
                                    )
                                    ax.legend()
                                except ImportError:
                                    # Simple moving average fallback
                                    _sort_idx = np.argsort(_y_pred_diag)
                                    _window = max(5, len(_residuals) // 20)
                                    _moving_avg = pd.Series(_residuals[_sort_idx]).rolling(_window, center=True).mean()
                                    ax.plot(_y_pred_diag[_sort_idx], _moving_avg, "r-", linewidth=2, label="Trend")
                                    ax.legend()
                                except Exception:
                                    pass

                                ax.set_xlabel("Predicted", fontsize=11)
                                ax.set_ylabel("Residual", fontsize=11)
                                ax.set_title("Residuals vs Predicted", fontweight="bold")

                                fig.suptitle(
                                    "Prediction Distribution Diagnostics", fontweight="bold", fontsize=14, y=1.01
                                )
                                plt.tight_layout()
                                plt.show()

                                # Summary statistics
                                _diag_stats = [
                                    {"Statistic": "Mean Residual", "Value": f"{_res_mean:.6f}"},
                                    {"Statistic": "Std Residual", "Value": f"{_res_std:.4f}"},
                                    {"Statistic": "Skewness", "Value": f"{stats.skew(_residuals):.4f}"},
                                    {"Statistic": "Kurtosis", "Value": f"{stats.kurtosis(_residuals):.4f}"},
                                    {"Statistic": "Max |Residual|", "Value": f"{np.max(np.abs(_residuals)):.4f}"},
                                    {"Statistic": "RMSE", "Value": f"{np.sqrt(np.mean(_residuals**2)):.4f}"},
                                    {"Statistic": "MAE", "Value": f"{np.mean(np.abs(_residuals)):.4f}"},
                                ]
                                _safe_html_table(pd.DataFrame(_diag_stats), caption="Residual Diagnostics Summary")

                            else:
                                # Classification diagnostics
                                _y_pred_proba = None
                                if hasattr(ctx.last_fitted_model, "predict_proba"):
                                    try:
                                        _y_pred_proba = ctx.last_fitted_model.predict_proba(_X_test_np)
                                        if _y_pred_proba.ndim == 2:
                                            _y_pred_proba = _y_pred_proba[:, 1]
                                    except Exception:
                                        pass

                                if _y_pred_proba is not None:
                                    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

                                    # Probability distribution by class
                                    ax = axes[0]
                                    for _cls in sorted(np.unique(_y_test_np)):
                                        _mask = _y_test_np == _cls
                                        ax.hist(
                                            _y_pred_proba[_mask],
                                            bins=25,
                                            alpha=0.6,
                                            label=f"Class {int(_cls)}",
                                            density=True,
                                        )
                                    ax.set_xlabel("Predicted Probability")
                                    ax.set_ylabel("Density")
                                    ax.set_title("Prediction Distributions by Class", fontweight="bold")
                                    ax.legend()

                                    # Calibration-like scatter
                                    ax = axes[1]
                                    _n_bins_cal = 10
                                    _bin_edges = np.linspace(0, 1, _n_bins_cal + 1)
                                    _bin_true = []
                                    _bin_pred = []
                                    for _bi in range(_n_bins_cal):
                                        _mask = (_y_pred_proba >= _bin_edges[_bi]) & (
                                            _y_pred_proba < _bin_edges[_bi + 1]
                                        )
                                        if _mask.sum() > 0:
                                            _bin_true.append(_y_test_np[_mask].mean())
                                            _bin_pred.append(_y_pred_proba[_mask].mean())
                                    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
                                    ax.scatter(_bin_pred, _bin_true, s=60, zorder=5, c=_SECTION_COLORS["supplementary"])
                                    ax.plot(_bin_pred, _bin_true, alpha=0.5, c=_SECTION_COLORS["supplementary"])
                                    ax.set_xlabel("Mean Predicted Probability")
                                    ax.set_ylabel("Fraction of Positives")
                                    ax.set_title("Calibration Plot", fontweight="bold")

                                    # Confusion-like: predicted vs actual counts
                                    ax = axes[2]
                                    _y_pred_cls = (_y_pred_proba >= 0.5).astype(int)
                                    _conf_data = pd.crosstab(
                                        pd.Series(_y_test_np, name="Actual"), pd.Series(_y_pred_cls, name="Predicted")
                                    )
                                    sns.heatmap(_conf_data, annot=True, fmt="d", cmap="Blues", ax=ax)
                                    ax.set_title("Confusion Matrix", fontweight="bold")

                                    fig.suptitle("Classification Diagnostics", fontweight="bold", fontsize=14, y=1.02)
                                    plt.tight_layout()
                                    plt.show()
                                else:
                                    logger.info("  Could not obtain predicted probabilities for diagnostics.")

                except Exception as _e:
                    logger.info(f"Error: Prediction diagnostics error: {_e}")
                    _traceback.print_exc()

            # 5D: Cross-Validated R² Distribution
            _subsection_header("5D. Cross-Validated Performance Distribution")

            try:
                _cv_model = _create_lightweight_model(_is_classification, _n_features)
                _cv_metric = "roc_auc" if _is_classification else "r2"
                _cv_display = "AUC" if _is_classification else "R²"

                # Combine train + test for CV
                _X_full = np.vstack([_X_train_np, _X_test_np])
                _y_full = np.concatenate([_y_train_np, _y_test_np])

                if _is_classification:
                    _cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                else:
                    _cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)

                logger.info(f"Running 10-fold CV with {type(_cv_model).__name__}...")
                _cv_scores = cross_val_score(
                    _cv_model, _X_full, _y_full, cv=_cv_splitter, scoring=_cv_metric, n_jobs=-1
                )
                logger.info(f"  CV {_cv_display}: {_cv_scores.mean():.4f} ± {_cv_scores.std():.4f}")
                logger.info(f"  Fold scores: {', '.join(f'{s:.4f}' for s in _cv_scores)}")

                # Violin + box + strip plot
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Panel 1: Violin + box
                ax = axes[0]
                parts = ax.violinplot(_cv_scores, positions=[0], showmeans=True, showmedians=True)
                for _pc in parts["bodies"]:
                    _pc.set_facecolor(_SECTION_COLORS["supplementary"])
                    _pc.set_alpha(0.5)
                parts["cmeans"].set_color("red")
                parts["cmedians"].set_color("blue")
                ax.scatter(np.zeros_like(_cv_scores), _cv_scores, c="black", s=40, zorder=5, alpha=0.6)
                ax.set_xticks([0])
                ax.set_xticklabels(["10-Fold CV"])
                ax.set_ylabel(_cv_display)
                ax.set_title(f"CV {_cv_display} Distribution", fontweight="bold")

                # Add mean ± CI annotation
                _cv_mean = _cv_scores.mean()
                _cv_ci = 1.96 * _cv_scores.std() / np.sqrt(len(_cv_scores))
                ax.text(
                    0.3,
                    _cv_mean,
                    f"Mean = {_cv_mean:.4f}\n95% CI: ±{_cv_ci:.4f}",
                    fontsize=10,
                    va="center",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                )

                # Panel 2: Histogram
                ax = axes[1]
                ax.hist(
                    _cv_scores,
                    bins=max(5, len(_cv_scores) // 2),
                    color=_SECTION_COLORS["supplementary"],
                    edgecolor="white",
                    alpha=0.8,
                )
                ax.axvline(_cv_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {_cv_mean:.4f}")
                ax.axvline(
                    np.median(_cv_scores),
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Median: {np.median(_cv_scores):.4f}",
                )
                ax.set_xlabel(_cv_display)
                ax.set_ylabel("Count")
                ax.set_title(f"CV {_cv_display} Histogram", fontweight="bold")
                ax.legend()

                fig.suptitle(
                    f"Cross-Validated {_cv_display} Distribution (10-fold)", fontweight="bold", fontsize=14, y=1.02
                )
                plt.tight_layout()
                plt.show()

                # Compare CV score with holdout test score
                if "last_fitted_model" in ctx and ctx.last_fitted_model is not None:
                    try:
                        if _is_classification:
                            if hasattr(ctx.last_fitted_model, "predict_proba"):
                                _holdout_pred = ctx.last_fitted_model.predict_proba(_X_test_np)
                                if _holdout_pred.ndim == 2:
                                    _holdout_pred = _holdout_pred[:, 1]
                            else:
                                _holdout_pred = ctx.last_fitted_model.predict(_X_test_np)
                            _holdout_score = roc_auc_score(_y_test_np, _holdout_pred)
                        else:
                            _holdout_pred = ctx.last_fitted_model.predict(_X_test_np).ravel()
                            _holdout_score = r2_score(_y_test_np, _holdout_pred)

                        _cv_vs_holdout = [
                            {"Metric": f"CV {_cv_display} (mean)", "Value": f"{_cv_mean:.4f}"},
                            {"Metric": f"CV {_cv_display} (std)", "Value": f"{_cv_scores.std():.4f}"},
                            {"Metric": "CV 95% CI", "Value": f"[{_cv_mean - _cv_ci:.4f}, {_cv_mean + _cv_ci:.4f}]"},
                            {"Metric": f"Holdout {_cv_display}", "Value": f"{_holdout_score:.4f}"},
                            {"Metric": "Difference (CV - Holdout)", "Value": f"{_cv_mean - _holdout_score:+.4f}"},
                        ]

                        _diff = abs(_cv_mean - _holdout_score)
                        if _diff < _cv_scores.std():
                            _cv_verdict = "Holdout score is within 1 SD of CV mean -- consistent"
                        elif _diff < 2 * _cv_scores.std():
                            _cv_verdict = "Holdout score is 1-2 SD from CV mean -- minor discrepancy"
                        else:
                            _cv_verdict = (
                                "Holdout score is > 2 SD from CV mean -- potential data leakage or distribution shift"
                            )
                        _cv_vs_holdout.append({"Metric": "Assessment", "Value": _cv_verdict})

                        _safe_html_table(pd.DataFrame(_cv_vs_holdout), caption="CV vs Holdout Comparison")
                    except Exception as _e_cv:
                        logger.info(f"  Could not compare CV vs holdout: {_e_cv}")

            except Exception as _e:
                logger.info(f"Error: Cross-validation error: {_e}")
                _traceback.print_exc()

            _section_times["Supplementary"] = _time.time() - _t0
            logger.info(f"\n Supplementary Analyses completed in {_section_times['Supplementary']:.1f}s")

        # Summary footer
        _hub_elapsed = _time.time() - _hub_start

        _footer_rows = []
        for _sname, _stime in _section_times.items():
            _footer_rows.append(
                f'<tr><td style="padding:4px 12px;">{_sname}</td>'
                f'<td style="padding:4px 12px; text-align:right;">{_stime:.1f}s</td></tr>'
            )

        _skipped = []
        if not ctx.run_diversity_analysis:
            _skipped.append("Diversity Analysis")
        if not ctx.run_bootstrap_stability:
            _skipped.append("Bootstrap Stability")
        if not ctx.run_learning_curves:
            _skipped.append("Learning Curves")
        if not ctx.run_shap_beeswarm:
            _skipped.append("SHAP Beeswarm")
        if not ctx.run_supplementary:
            _skipped.append("Supplementary")

        _skip_html = ""
        if _skipped:
            _skip_html = f'<p style="margin:8px 0 0 0; opacity:0.7; font-size:12px;">Skipped: {", ".join(_skipped)}</p>'

        display(
            HTML(
                '<div style="background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; '
                'padding: 16px 20px; border-radius: 12px; margin-top: 20px;">'
                '<h3 style="margin:0 0 8px 0; color: #e94560;"> Diagnostics Hub -- Complete</h3>'
                '<table style="color: white; border-collapse: collapse;">'
                '<tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">'
                '<th style="padding:4px 12px; text-align:left;">Section</th>'
                '<th style="padding:4px 12px; text-align:right;">Time</th></tr>'
                + "".join(_footer_rows)
                + f'<tr style="border-top: 2px solid rgba(255,255,255,0.3);">'
                f'<td style="padding:6px 12px; font-weight:bold;">Total</td>'
                f'<td style="padding:6px 12px; text-align:right; font-weight:bold;">{_hub_elapsed:.1f}s</td></tr>'
                "</table>" + _skip_html + "</div>"
            )
        )
