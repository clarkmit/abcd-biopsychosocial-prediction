"""Family- and site-grouped cross-validation (the `grouped_cv` stage, Supplement S1.5, S1.18.4).

Family-grouped CV assigns siblings to the same fold via rel_family_id,
testing whether within-family clustering (shared genetics, environment,
rater) inflates standard CV performance. Site-grouped CV holds out
entire ABCD acquisition sites to assess site-level confounding across
the 21 sites. Both use GroupKFold. Supports family-only, site-only,
or combined modes.
"""

import logging

from IPython.display import display

from bpsp.context import ctx
from bpsp.data.preprocessing import detect_task_type
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
Group-Aware Cross-Validation for Confound Assessment
------------------------------------------------------
Tests whether performance is inflated by familial clustering (twin/
sibling pairs) or site effects within standard CV folds. Family-
grouped CV assigns siblings to the same fold; site-grouped CV holds
out entire ABCD sites. Both procedures isolate the contribution of
familial- and site-level confounds to apparent model performance.

Variants
  Family-grouped CV : siblings assigned to the same fold; assesses
                       leakage from familial relatedness.
  Site-grouped CV   : all participants from one ABCD site held out
                       together; assesses site-level generalization.

Outputs
  Performance comparison table (standard CV vs each grouped variant);
  a meaningful drop in grouped CV performance indicates confound sensitivity.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

# GroupedCVCheck Class


class GroupedCVCheck:
    """
    Test model robustness to confounds via group-aware cross-validation.

    Grouping strategies:
    1. Standard CV: Random stratified splits (baseline)
    2. Family CV: Siblings always in same fold (tests family confound)
    3. Site CV: Site members always in same fold (tests site confound)

    For each grouping, computes metrics and compares to baseline.
    Large performance drop = potential confound bias.
    """

    def __init__(self, X, y, task_type="auto", n_outer=5, random_state=42):
        """X, y: training data. task_type defaults to auto-detection.
        n_outer : int
            Number of CV folds
        random_state : int
            Random seed
        """
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Ensure y is always 1-dimensional (batch archive may produce (N,1) arrays)
        if isinstance(y, pd.Series):
            self.y = y
        elif isinstance(y, pd.DataFrame):
            self.y = y.iloc[:, 0]
        else:
            _y = np.asarray(y)
            if _y.ndim > 1:
                _y = _y.ravel()
            self.y = pd.Series(_y)
        self.n_outer = n_outer
        self.random_state = random_state

        if task_type == "auto":
            self.task_type = detect_task_type(self.y.values)
        else:
            self.task_type = task_type

        self.is_classification = self.task_type == "classification"
        self.is_regression = self.task_type == "regression"

        self.results = {
            "standard": {},  # {metric: [fold_values]}
            "family_grouped": {},
            "site_grouped": {},
        }

        logger.info(f"Task type: {self.task_type.upper()}")

    def run_group_cv(self, groups, group_type="family", model_factory=None, base_model=None):
        """
        Run group-aware CV.

        Parameters:
        groups : pd.Series or np.array
            Group IDs (family_id, site_id, etc.) aligned with X
        group_type : str
            'family', 'site', or other label for reporting
        model_factory : callable, optional
            Function returning fresh model
        base_model : estimator, optional
            Pre-trained model from pipeline
        """
        groups = pd.Series(groups).reset_index(drop=True)

        if len(groups) != len(self.X):
            raise ValueError(f"groups length {len(groups)} != X length {len(self.X)}")

        logger.info(f"\n Running robustness test: {group_type.upper()}-AWARE CV")
        logger.info(f"   Groups: {len(groups.unique())} unique {group_type} IDs")
        logger.info(
            f"   Samples per group: mean={groups.value_counts().mean():.1f}, "
            f"min={groups.value_counts().min()}, max={groups.value_counts().max()}"
        )

        use_base_model = base_model is not None

        if self.is_classification:
            return self._robustness_cv_classification(groups, group_type, use_base_model, base_model, model_factory)
        else:
            return self._robustness_cv_regression(groups, group_type, use_base_model, base_model, model_factory)

    def _robustness_cv_classification(self, groups, group_type, use_base_model, base_model, model_factory):
        """Robustness test for classification."""
        from sklearn.metrics import roc_curve

        group_results = {"fold": [], "auc": [], "accuracy": [], "sensitivity": [], "specificity": [], "ppv": []}

        # Use StratifiedGroupKFold to preserve class balance within groups
        n_unique_groups = len(groups.unique())
        n_splits = min(self.n_outer, n_unique_groups)

        if n_splits < 2:
            logger.info(f"   Warning: Not enough groups for CV (need >= 2, have {n_unique_groups})")
            return None

        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            cv_iterator = sgkf.split(self.X, self.y, groups=groups)
        except Exception as e:
            logger.info(f"[WARNING] {e}")
            # Fallback if StratifiedGroupKFold not available
            logger.info("   Falling back to GroupKFold (no stratification)")
            gkf = GroupKFold(n_splits=n_splits)
            cv_iterator = gkf.split(self.X, self.y, groups=groups)
        for fold_num, (train_idx, test_idx) in enumerate(cv_iterator):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]
            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            if len(thresholds) > 1:
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5
            y_pred = (y_proba >= optimal_threshold).astype(int)
            _cm = confusion_matrix(y_test, y_pred)
            if _cm.shape == (2, 2):
                tn, fp, fn, tp = _cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            group_results["fold"].append(fold_num)
            group_results["auc"].append(roc_auc_score(y_test, y_proba))
            group_results["accuracy"].append(accuracy_score(y_test, y_pred))
            group_results["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            group_results["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            group_results["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)

            logger.info(f"   Fold {fold_num + 1}: AUC={group_results['auc'][-1]:.4f}")

        self.results[f"{group_type}_grouped"] = group_results
        return group_results

    def _robustness_cv_regression(self, groups, group_type, use_base_model, base_model, model_factory):
        """Robustness test for regression."""
        group_results = {"fold": [], "r2": [], "rmse": [], "mae": []}

        # Use GroupKFold
        n_unique_groups = len(groups.unique())
        n_splits = min(self.n_outer, n_unique_groups)

        if n_splits < 2:
            logger.info(f"   Warning: Not enough groups for CV (need >= 2, have {n_unique_groups})")
            return None

        gkf = GroupKFold(n_splits=n_splits)
        for fold_num, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y, groups=groups)):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]
            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            group_results["fold"].append(fold_num)
            group_results["r2"].append(r2_score(y_test, y_pred))
            group_results["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            group_results["mae"].append(mean_absolute_error(y_test, y_pred))

            logger.info(f"   Fold {fold_num + 1}: R²={group_results['r2'][-1]:.4f}")

        self.results[f"{group_type}_grouped"] = group_results
        return group_results

    def run_standard_cv(self, model_factory=None, base_model=None):
        """Run standard CV (baseline for comparison)."""
        logger.info("\n Running baseline: STANDARD CV")

        use_base_model = base_model is not None

        if self.is_classification:
            return self._standard_cv_classification(use_base_model, base_model, model_factory)
        else:
            return self._standard_cv_regression(use_base_model, base_model, model_factory)

    def _standard_cv_classification(self, use_base_model, base_model, model_factory):
        """Standard CV for classification (baseline)."""
        from sklearn.metrics import roc_curve

        standard_results = {"fold": [], "auc": [], "accuracy": [], "sensitivity": [], "specificity": [], "ppv": []}

        skf = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]

            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            if len(thresholds) > 1:
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5
            y_pred = (y_proba >= optimal_threshold).astype(int)

            _cm = confusion_matrix(y_test, y_pred)
            if _cm.shape == (2, 2):
                tn, fp, fn, tp = _cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            standard_results["fold"].append(fold_num)
            standard_results["auc"].append(roc_auc_score(y_test, y_proba))
            standard_results["accuracy"].append(accuracy_score(y_test, y_pred))
            standard_results["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            standard_results["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            standard_results["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)

            logger.info(f"   Fold {fold_num + 1}: AUC={standard_results['auc'][-1]:.4f}")

        self.results["standard"] = standard_results
        return standard_results

    def _standard_cv_regression(self, use_base_model, base_model, model_factory):
        """Standard CV for regression (baseline)."""
        standard_results = {"fold": [], "r2": [], "rmse": [], "mae": []}

        kf = KFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]

            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            standard_results["fold"].append(fold_num)
            standard_results["r2"].append(r2_score(y_test, y_pred))
            standard_results["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            standard_results["mae"].append(mean_absolute_error(y_test, y_pred))

            logger.info(f"   Fold {fold_num + 1}: R²={standard_results['r2'][-1]:.4f}")

        self.results["standard"] = standard_results
        return standard_results

    def generate_report(self, title="ROBUSTNESS ANALYSIS"):
        """Generate robustness comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info(title)
        logger.info("=" * 80)
        logger.info(f"\nTask Type: {self.task_type.upper()}")
        logger.info(f"CV folds: {self.n_outer}\n")

        # Determine metrics
        if self.is_classification:
            metrics = ["auc", "accuracy", "sensitivity", "specificity"]
        else:
            metrics = ["r2", "rmse", "mae"]

        # Build comparison table
        comparison_rows = []

        for cv_type in ["standard", "family_grouped", "site_grouped"]:
            if cv_type not in self.results or not self.results[cv_type]:
                continue

            row = {"CV Strategy": cv_type.replace("_", " ").title()}

            for metric in metrics:
                if metric in self.results[cv_type]:
                    values = self.results[cv_type][metric]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1)
                        row[f"{metric}"] = f"{mean_val:.3f} ± {std_val:.3f}"

            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)

        logger.info("ROBUSTNESS COMPARISON")
        logger.info("-" * 80)
        logger.info(comparison_df.to_string(index=False))

        # Interpretation
        if "standard" in self.results and "family_grouped" in self.results:
            logger.info("\n INTERPRETATION:")
            logger.info("-" * 80)

            if self.is_classification:
                standard_auc = np.mean(self.results["standard"]["auc"])
                family_auc = np.mean(self.results["family_grouped"]["auc"])
                drop = (standard_auc - family_auc) / standard_auc * 100

                logger.info(f"Family confound: AUC drop = {drop:.1f}%")
                if drop < 5:
                    logger.info("   Low family confound (robust to family clustering)")
                elif drop < 15:
                    logger.info("   Warning: Moderate family confound (consider in interpretation)")
                else:
                    logger.info("   Warning: High family confound (significant bias detected)")
            else:
                standard_r2 = np.mean(self.results["standard"]["r2"])
                family_r2 = np.mean(self.results["family_grouped"]["r2"])
                drop = (standard_r2 - family_r2) / (standard_r2 + 1e-10) * 100

                logger.info(f"Family confound: R² drop = {drop:.1f}%")

        logger.info("\n" + "=" * 80)
        logger.info("[grouped_cv] COMPLETE")
        logger.info("=" * 80)


# Create Validation Model


def create_validation_model_p3(is_classification=True):
    """
    Creates the same ensemble model used in main pipeline.
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

    # TabPFN is opt-in via model_configs

    return model


# Cache system: robustness

# Batch-aware wrapper: Site Robustness
# When _batch_archive holds multiple targets with archived state, this
# wrapper iterates over each, restoring globals before each pass.
# In single-target mode the validation runs exactly once (no change).


def _run_for_current_target():
    """Run this stage's validation for the current target."""
    # Validation stage gate (reads the main stage settings)
    _skip_grouped = not ctx.get("run_validation_cells", True)
    _VAL_N_OUTER = ctx.n_outer_folds
    _run_family = ctx.robustness_mode in ("Family Only", "Both")
    _run_site = ctx.robustness_mode in ("Site Only", "Both")

    # Dependency guard
    if "make_run_key" not in ctx:
        logger.info("\nWarning: setup has not been run in this session.")
        logger.info("   Run setup and the main stage first.")
        _RUN_14 = False
    elif _skip_grouped:
        logger.info("grouped_cv skipped: main.run_validation_cells is False")
        _RUN_14 = False
    else:
        _CTAG_14 = "robustness"
        _CKEY_14 = make_run_key({"cell": _CTAG_14})
        _CACHED_14 = cache_load(_CTAG_14, _CKEY_14)
        if _CACHED_14:
            logger.info(f"  {_CTAG_14} loaded from cache -- skipping execution.")
            _RUN_14 = False
        else:
            _RS["section"] = "robustness"
            _done_folds_14 = fold_load_all(_CKEY_14, _CTAG_14)
            _RUN_14 = True

    if _RUN_14:
        # EXECUTION: Run robustness reporter

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

            _has_demo = "sample_valid" in ctx and ctx.sample_valid is not None

            if _is_within_mode and _has_demo:
                # Within-categories mode: Per-category site/family robustness

                _TOP_K = min(5, len(_top_cats_ordered))
                _cats_to_run = _top_cats_ordered[:_TOP_K]

                logger.info("\n" + "=" * 80)
                logger.info(f" Running [grouped_cv] -- Site/Family Robustness (Within-Categories, Top {_TOP_K})")
                logger.info("=" * 80)

                task_type = detect_task_type(list(ctx.exported_within_dict.values())[0][2])
                is_classification = task_type == "classification"
                _metric_key = "auc" if is_classification else "r2"
                _metric_name = "AUC" if is_classification else "R²"
                logger.info(f"   Task type: {task_type}")

                has_family = "rel_family_id" in ctx.sample_valid.columns
                has_site = "L_site_id" in ctx.sample_valid.columns
                logger.info(f"   Available confounds: family={has_family}, site={has_site}")

                results_dir_p3 = os.path.join(os.getcwd(), "robustness_results")
                os.makedirs(results_dir_p3, exist_ok=True)

                _robustness_summary = []

                for _cat_name in _cats_to_run:
                    if _cat_name not in ctx.exported_within_dict:
                        continue
                    _Xtr, _Xte, _ytr, _yte = ctx.exported_within_dict[_cat_name]
                    _X_full = (
                        pd.concat([_Xtr, _Xte], axis=0, ignore_index=True)
                        if isinstance(_Xtr, pd.DataFrame)
                        else pd.DataFrame(np.vstack([_Xtr, _Xte]))
                    )
                    _y_full = (
                        pd.concat(
                            [pd.Series(np.asarray(_ytr).ravel()), pd.Series(np.asarray(_yte).ravel())],
                            axis=0,
                            ignore_index=True,
                        )
                        if isinstance(_ytr, (pd.Series, pd.DataFrame, np.ndarray))
                        else pd.Series(np.concatenate([np.asarray(_ytr).ravel(), np.asarray(_yte).ravel()]))
                    )

                    logger.info(f"\n  [{_cat_name}] ({_Xtr.shape[1]} features)")

                    _reporter = GroupedCVCheck(_X_full.copy(), _y_full.copy(), task_type="auto", n_outer=_VAL_N_OUTER)

                    try:
                        _reporter.run_standard_cv(
                            model_factory=lambda: create_validation_model_p3(is_classification=is_classification)
                        )
                        _std_score = np.mean(_reporter.results.get("standard", {}).get(_metric_key, [np.nan]))
                    except Exception as _e:
                        logger.info(f"    Warning: Standard CV failed: {_e}")
                        _std_score = np.nan

                    _fam_score = np.nan
                    if has_family and _run_family:
                        try:
                            _fam_ids = ctx.sample_valid["rel_family_id"].reset_index(drop=True).reindex(_X_full.index)
                            _fam_ids = _fam_ids.fillna(-1).astype(int).astype(str)  # uniform type for GroupKFold
                            _reporter.run_group_cv(
                                groups=_fam_ids,
                                group_type="family",
                                model_factory=lambda: create_validation_model_p3(is_classification=is_classification),
                            )
                            _fam_score = np.mean(_reporter.results.get("family_grouped", {}).get(_metric_key, [np.nan]))
                        except Exception as _e:
                            logger.info(f"    Warning: Family CV failed: {_e}")

                    ctx._site_score = np.nan
                    if has_site and _run_site:
                        try:
                            _site_ids = ctx.sample_valid["L_site_id"].reset_index(drop=True).reindex(_X_full.index)
                            _reporter.run_group_cv(
                                groups=_site_ids,
                                group_type="site",
                                model_factory=lambda: create_validation_model_p3(is_classification=is_classification),
                            )
                            ctx._site_score = np.mean(
                                _reporter.results.get("site_grouped", {}).get(_metric_key, [np.nan])
                            )
                        except Exception as _e:
                            logger.info(f"    Warning: Site CV failed: {_e}")

                    _row = {
                        "Category": _cat_name,
                        f"Standard {_metric_name}": _std_score,
                    }
                    if has_family and _run_family:
                        _row[f"Family {_metric_name}"] = _fam_score
                        _row["Δ Family"] = _fam_score - _std_score if not np.isnan(_fam_score) else np.nan
                    if has_site and _run_site:
                        _row[f"Site {_metric_name}"] = ctx._site_score
                        _row["Δ Site"] = ctx._site_score - _std_score if not np.isnan(ctx._site_score) else np.nan
                    _robustness_summary.append(_row)

                    logger.info(
                        f"    Standard: {_std_score:.4f}  |  Family: {_fam_score:.4f}  |  Site: {ctx._site_score:.4f}"
                    )

                    # Free reporter between categories
                    try:
                        del _reporter
                    except NameError:
                        pass
                    _pipeline_cleanup(label=f"P3 {_cat_name}", verbose=False)

                if _robustness_summary:
                    logger.info("\n" + "=" * 80)
                    logger.info("[grouped_cv] Site/Family Robustness Summary")
                    logger.info("=" * 80)

                    _rob_df = pd.DataFrame(_robustness_summary)
                    _num_cols = [c for c in _rob_df.columns if c != "Category"]
                    _delta_cols = [c for c in _num_cols if c.startswith("Δ")]

                    _rob_styler = (
                        _rob_df.style.format({c: "{:.4f}" for c in _num_cols})
                        .set_caption(f"[grouped_cv] Robustness -- {ctx.target_options} (T{ctx.tp_option})")
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
                                        ("font-size", "11px"),
                                        ("font-weight", "600"),
                                        ("padding", "6px 10px"),
                                    ],
                                },
                                {"selector": "td", "props": [("padding", "5px 10px"), ("font-size", "12px")]},
                            ]
                        )
                        .hide(axis="index")
                    )
                    for _c in _num_cols:
                        if not _c.startswith("Δ"):
                            _rob_styler = _rob_styler.background_gradient(
                                cmap="RdYlGn", subset=[_c], vmin=0, vmax=max(0, _rob_df[_c].max())
                            )
                    if _delta_cols:
                        _rob_styler = _rob_styler.background_gradient(
                            cmap="RdYlBu", subset=_delta_cols, vmin=-0.05, vmax=0.05
                        )
                    display(_rob_styler)
                    _rob_df.to_csv(f"{results_dir_p3}/robustness_within_categories.csv", index=False)
                    logger.info(f"\nSaved: {results_dir_p3}/robustness_within_categories.csv")

            elif "X_train" in ctx and "y_train" in ctx and _has_demo:
                # Across-categories mode: Original single-pool robustness

                logger.info("\n Running [grouped_cv] with pipeline variables")

                X_full = (
                    pd.concat([ctx.X_train, ctx.X_test], axis=0, ignore_index=True)
                    if "X_test" in ctx
                    else ctx.X_train.reset_index(drop=True)
                )
                y_full = (
                    pd.concat([ctx.y_train, ctx.y_test], axis=0, ignore_index=True)
                    if "y_test" in ctx
                    else ctx.y_train.reset_index(drop=True)
                )

                ctx.robust_reporter = GroupedCVCheck(
                    X_full.copy(), y_full.copy(), task_type="auto", n_outer=_VAL_N_OUTER
                )

                has_family = "rel_family_id" in ctx.sample_valid.columns
                has_site = "L_site_id" in ctx.sample_valid.columns
                task_type = detect_task_type(ctx.y_train)
                is_classification = task_type == "classification"
                logger.info(f"   Available confounds: family={has_family}, site={has_site}")
                logger.info(f"   Task type: {task_type}")

                ctx.robust_reporter.run_standard_cv(
                    model_factory=lambda: create_validation_model_p3(is_classification=is_classification)
                )

                if has_family and _run_family:
                    try:
                        family_ids = ctx.sample_valid["rel_family_id"].reset_index(drop=True).iloc[: len(X_full)]
                        family_ids = family_ids.fillna(-1).astype(int).astype(str)  # uniform type for GroupKFold
                        assert len(family_ids) == len(X_full), (
                            f"family_ids len {len(family_ids)} != X_full len {len(X_full)}"
                        )
                        ctx.robust_reporter.run_group_cv(
                            groups=family_ids,
                            group_type="family",
                            model_factory=lambda: create_validation_model_p3(is_classification=is_classification),
                        )
                    except Exception as e:
                        logger.info(f"   Warning: Error in family CV: {e}")

                if has_site and _run_site:
                    try:
                        site_ids = ctx.sample_valid["L_site_id"].reset_index(drop=True)
                        site_ids = site_ids.fillna("_unknown_site")  # NaN-safe for GroupKFold
                        assert len(site_ids) == len(X_full)
                        ctx.robust_reporter.run_group_cv(
                            groups=site_ids,
                            group_type="site",
                            model_factory=lambda: create_validation_model_p3(is_classification=is_classification),
                        )
                    except Exception as e:
                        logger.info(f"   Warning: Error in site CV: {e}")

                ctx.robust_reporter.generate_report(
                    title="[grouped_cv] Robustness: Family/Site-Aware CV - StackingEnsemble"
                )

                # Visualization
                if ctx.robust_reporter.results:
                    if is_classification:
                        metrics_to_plot = ["auc", "accuracy", "sensitivity"]
                        metric_labels = ["AUC", "Accuracy", "Sensitivity"]
                    else:
                        metrics_to_plot = ["r2", "rmse", "mae"]
                        metric_labels = ["R²", "RMSE", "MAE"]

                    strategy_labels = {
                        "standard": "Standard CV",
                        "family_grouped": "Family-Grouped CV",
                        "site_grouped": "Site-Grouped CV",
                    }
                    strategy_colors = {"standard": "#2196F3", "family_grouped": "#FF9800", "site_grouped": "#4CAF50"}
                    cv_strategies = [
                        s
                        for s in ["standard", "family_grouped", "site_grouped"]
                        if s in ctx.robust_reporter.results and ctx.robust_reporter.results[s]
                    ]

                    if len(cv_strategies) >= 2:
                        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 6))
                        if len(metrics_to_plot) == 1:
                            axes = [axes]
                        for ax_idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
                            x_pos = np.arange(len(cv_strategies))
                            means = [np.mean(ctx.robust_reporter.results[s].get(metric, [0])) for s in cv_strategies]
                            stds = [
                                np.std(ctx.robust_reporter.results[s].get(metric, [0]), ddof=1) for s in cv_strategies
                            ]
                            colors_list = [strategy_colors[s] for s in cv_strategies]
                            bars = axes[ax_idx].bar(
                                x_pos,
                                means,
                                0.6,
                                yerr=stds,
                                capsize=5,
                                color=colors_list,
                                edgecolor="black",
                                linewidth=0.5,
                                alpha=0.85,
                            )
                            for bar, m, sd in zip(bars, means, stds):
                                axes[ax_idx].text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    bar.get_height() + sd + 0.005,
                                    f"{m:.3f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=10,
                                    fontweight="bold",
                                )
                            axes[ax_idx].set_xticks(x_pos)
                            axes[ax_idx].set_xticklabels(
                                [strategy_labels[s] for s in cv_strategies], rotation=15, ha="right"
                            )
                            axes[ax_idx].set_ylabel(label, fontsize=12)
                            axes[ax_idx].set_title(f"{label} by CV Strategy", fontsize=12, fontweight="bold")
                            axes[ax_idx].spines["top"].set_visible(False)
                            axes[ax_idx].spines["right"].set_visible(False)

                        plt.suptitle("[grouped_cv] Standard vs Group-Aware CV", fontsize=14, fontweight="bold")
                        plt.tight_layout()
                        results_dir_p3 = os.path.join(os.getcwd(), "robustness_results")
                        os.makedirs(results_dir_p3, exist_ok=True)
                        plt.savefig(f"{results_dir_p3}/robustness_comparison.png", dpi=300, bbox_inches="tight")
                        plt.show()

            else:
                if not _has_demo:
                    logger.info("\nError: [grouped_cv] sample_valid not found")
                else:
                    logger.info("\nError: Required variables not found")

        except Exception as e:
            logger.info(f"\nError in [grouped_cv]: {e}")
            import traceback

            traceback.print_exc()

        logger.info("\n[grouped_cv] complete.")

        # [Cache + report save]
        _RS["section"] = "uncategorized"
        try:
            _pipeline_cleanup(label="P3 complete")

            cache_save(
                {"reporter": locals().get("reporter") or locals().get("robustness_reporter")},
                cell_tag=_CTAG_14,
                key=_CKEY_14,
                label=f"{ctx.target_options}_tp{ctx.tp_option}_[grouped_cv] Site Robustness",
            )
            report_save_state(make_run_key())
        except Exception as _ce_14:
            logger.info(f"[Cache] Could not save robustness: {_ce_14}")


def run() -> None:
    """Run this stage (notebook cell 14)."""

    # Batch dispatch
    _is_bv_17 = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_17:
        try:
            _cbv_17 = cache_load("main_runner")
            if (
                _cbv_17
                and isinstance(_cbv_17, dict)
                and _cbv_17.get("_batch_archive")
                and len(_cbv_17["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_17["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_17["_batch_archive"]
                _is_bv_17 = True
        except Exception:
            pass

    if _is_bv_17:
        _vgs_17 = {
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
        logger.info(f"  BATCH -- Site Robustness: processing {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)

        for _vi_17, (_vk_17, _vd_17) in enumerate(ctx._batch_archive.items()):
            _vl_17 = _vd_17.get("target", _vk_17)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_17 + 1}/{len(ctx._batch_archive)}] Target: {_vl_17}")
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
                if _vd_17.get(_gk) is not None:
                    ctx[_gk] = _vd_17[_gk]
            ctx["target_options"] = _vl_17
            if "timepoint" in _vd_17:
                ctx["tp_option"] = _vd_17["timepoint"]
            _run_for_current_target()
            # Archive validation reporters back into _batch_archive
            if _vk_17 in ctx._batch_archive:
                ctx._batch_archive[_vk_17]["robust_reporter"] = ctx.get("robust_reporter")
                ctx._batch_archive[_vk_17]["_site_score"] = ctx.get("_site_score")

        for _k, _v in _vgs_17.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH COMPLETE -- Site Robustness: {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)
    else:
        _run_for_current_target()
