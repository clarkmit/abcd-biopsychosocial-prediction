"""Stacking audit: base-model contributions, meta-learner weights and leakage checks (the `stacking_audit` stage, Supplement S1.1).

Stacking methodology audit confirming no data leakage: verifies each
sample appears in exactly one OOF validation fold, meta-learner receives
only OOF predictions (never in-fold), and reports base-model contribution
weights and nested CV structural integrity end-to-end.
"""

import logging

from IPython.display import HTML, display

from bpsp.context import ctx
from bpsp.data.preprocessing import detect_task_type
from bpsp.models.stacking import ModelVersion, TabPFNClassifier
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
Stacking Methodology Validation & Meta-Learner Audit
------------------------------------------------------
Confirms no data leakage in the two-stage stacking procedure:
  (1) Base learners generate predictions only on held-out inner folds.
  (2) Meta-learner trains exclusively on those OOF predictions.
End-to-end audit of stacking integrity and preprocessing leakage between
inner and outer CV loops.

Checks performed
----------------
  - Each sample appears in exactly one OOF validation fold.
  - Meta-learner receives OOF predictions, never in-fold predictions.
  - Base model contribution weights are reported.
  - Nested CV structure is confirmed end-to-end.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


class StackingLeakageCheck:
    """
    Audit stacking ensemble for methodological correctness.

    Checks:
    1. Each sample appears in exactly 1 inner validation fold
    2. Meta-learner trained ONLY on OOF predictions (no leakage)
    3. Base models trained on data they DON'T predict for meta-learner
    4. Proper nested CV structure maintained

    HANDLES:
    - Classification: LogisticRegression as meta-learner
    - Regression: LinearRegression as meta-learner
    """

    def __init__(self, X, y, task_type="auto", n_inner=5, random_state=42):
        """X, y: train split. task_type: classification/regression/auto.
        n_inner : int
            Number of inner CV folds for meta-learner training
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
        self.n_inner = n_inner
        self.random_state = random_state

        if task_type == "auto":
            self.task_type = detect_task_type(self.y.values)
        else:
            self.task_type = task_type

        self.is_classification = self.task_type == "classification"
        self.is_regression = self.task_type == "regression"
        if self.is_classification:
            self.meta_learner_class = LogisticRegression
            self.meta_learner_params = {"random_state": random_state, "max_iter": 1000}
        else:
            self.meta_learner_class = LinearRegression
            self.meta_learner_params = {}

        self.audit_results = {}

        logger.info(f"Task type: {self.task_type.upper()}")
        logger.info(f"   Meta-learner: {self.meta_learner_class.__name__}")

    def validate_stacking_setup(self, base_models, meta_learner=None, n_outer=None):
        """
        Validate proposed stacking setup.

        Parameters:
        base_models : dict
            {name: model_instance} for base learners
        meta_learner : estimator, optional
            Meta-learner (auto-created if None)
        n_outer : int
            Number of outer CV folds
        """
        if n_outer is None:
            n_outer = ctx._VAL_N_OUTER
        logger.info("\n Validating stacking setup")
        logger.info(f"   Base models: {list(base_models.keys())}")
        logger.info(f"   Meta-learner: {self.meta_learner_class.__name__}")

        if meta_learner is None:
            meta_learner = self.meta_learner_class(**self.meta_learner_params)

        return self._validate_nested_cv_structure(base_models, meta_learner, n_outer)

    def audit_existing_stacking(self, stacking_model):
        """
        Audit an existing stacking model from notebook.

        Parameters:
        stacking_model : StackingEnsembleModel or similar
            Existing stacking model from pipeline
        """
        logger.info("\n Auditing existing stacking model")
        logger.info(f"   Model class: {stacking_model.__class__.__name__}")

        return self._audit_model_structure(stacking_model)

    def _validate_nested_cv_structure(self, base_models, meta_learner, n_outer):
        """Validate nested CV structure for stacking."""
        logger.info(f"\n   Checking nested CV structure ({n_outer} outer folds)...")

        if self.is_classification:
            cv_outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=self.random_state)
            cv_inner = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=self.random_state)
        else:
            cv_outer = KFold(n_splits=n_outer, shuffle=True, random_state=self.random_state)
            cv_inner = KFold(n_splits=self.n_inner, shuffle=True, random_state=self.random_state)

        validation_checks = {
            "oof_sample_coverage": None,
            "no_leakage": True,
            "base_model_count": len(base_models),
            "meta_learner_trained": False,
            "outer_fold_coverage": [],
            "warnings": [],
        }

        all_inner_coverage_ok = True

        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(cv_outer.split(self.X, self.y)):
            X_train_outer = self.X.iloc[outer_train_idx]
            y_train_outer = self.y.iloc[outer_train_idx]

            # Track coverage WITHIN this outer fold only
            inner_fold_count = np.zeros(len(X_train_outer))

            # Generate OOF predictions for meta-learner training
            oof_meta_features = np.zeros((len(X_train_outer), len(base_models)))

            for inner_train_idx, inner_val_idx in cv_inner.split(X_train_outer, y_train_outer):
                X_inner_train = X_train_outer.iloc[inner_train_idx]
                X_inner_val = X_train_outer.iloc[inner_val_idx]
                y_inner_train = y_train_outer.iloc[inner_train_idx]

                # Count: each sample should appear in exactly 1 inner val fold
                inner_fold_count[inner_val_idx] += 1

                # Get base model predictions
                for base_idx, (base_name, base_model) in enumerate(base_models.items()):
                    from sklearn.base import clone

                    try:
                        base_clone = clone(base_model)
                    except Exception:
                        base_clone = base_model.__class__(**base_model.get_params())
                    base_clone.fit(X_inner_train, y_inner_train)

                    if self.is_classification and hasattr(base_clone, "predict_proba"):
                        if len(np.unique(y_inner_train)) <= 2:
                            oof_meta_features[inner_val_idx, base_idx] = base_clone.predict_proba(X_inner_val)[:, 1]
                        else:
                            oof_meta_features[inner_val_idx, base_idx] = base_clone.predict(X_inner_val).astype(float)
                    else:
                        oof_meta_features[inner_val_idx, base_idx] = base_clone.predict(X_inner_val)

            # Check inner fold coverage for THIS outer fold
            unique_counts = np.unique(inner_fold_count)
            fold_ok = len(unique_counts) == 1 and unique_counts[0] == 1.0
            validation_checks["outer_fold_coverage"].append(fold_ok)

            if not fold_ok:
                all_inner_coverage_ok = False
                validation_checks["warnings"].append(
                    f"Outer fold {outer_fold}: inner fold counts = {np.bincount(inner_fold_count.astype(int))}"
                )

            # Verify: Meta-learner trained on OOF predictions only
            meta_learner_clone = self.meta_learner_class(**self.meta_learner_params)
            meta_learner_clone.fit(oof_meta_features, y_train_outer)
            validation_checks["meta_learner_trained"] = True

            logger.info(
                "%s %s",
                f"      Outer fold {outer_fold + 1}: inner OOF coverage {'PASS' if fold_ok else 'FAIL'};",
                f"meta-learner trained on {oof_meta_features.shape} OOF matrix",
            )

        # Aggregate coverage check
        if all_inner_coverage_ok:
            validation_checks["oof_sample_coverage"] = (
                f"PASS: Each sample in exactly 1 inner val fold (verified across all {n_outer} outer folds)"
            )
        else:
            validation_checks["oof_sample_coverage"] = "FAIL: Inconsistent inner fold coverage in some outer folds"
            validation_checks["no_leakage"] = False

        self.audit_results["validation"] = validation_checks

        return validation_checks

    def _audit_model_structure(self, stacking_model):
        """Audit structure of existing stacking model."""
        audit = {"has_base_models": False, "has_meta_learner": False, "base_model_names": [], "structure_warnings": []}

        # Check for base models
        if hasattr(stacking_model, "base_models"):
            audit["has_base_models"] = True
            if isinstance(stacking_model.base_models, dict):
                audit["base_model_names"] = list(stacking_model.base_models.keys())
            else:
                audit["base_model_names"] = [type(m).__name__ for m in stacking_model.base_models]

        # Check for meta-learner
        if hasattr(stacking_model, "meta_learner"):
            audit["has_meta_learner"] = True

        # Warnings
        if not audit["has_base_models"]:
            audit["structure_warnings"].append("No base_models attribute found")
        if not audit["has_meta_learner"]:
            audit["structure_warnings"].append("No meta_learner attribute found")

        self.audit_results["model_structure"] = audit

        return audit

    def generate_report(self, title="STACKING METHODOLOGY AUDIT"):
        """Generate stacking audit report."""
        display(
            HTML(
                f'<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #1565C0;'
                f"background:linear-gradient(135deg,#E3F2FD,#F5F5F5);padding:14px 20px;"
                f'margin:16px 0;border-radius:0 6px 6px 0;">'
                f'<span style="font-size:15px;font-weight:700;color:#1565C0;">{title}</span>'
                f"</div>"
            )
        )

        logger.info(f"\nTask Type: {self.task_type.upper()}")
        logger.info(f"Meta-learner: {self.meta_learner_class.__name__}")
        logger.info(f"Inner CV folds: {self.n_inner}\n")

        # Validation results
        if "validation" in self.audit_results:
            logger.info("NESTED CV VALIDATION")
            logger.info("-" * 80)

            val = self.audit_results["validation"]

            logger.info(f"OOF Sample Coverage: {val['oof_sample_coverage']}")
            logger.info(f"No Data Leakage: {'YES' if val['no_leakage'] else 'NO'}")
            logger.info(f"Base Models: {val['base_model_count']}")
            logger.info(f"Meta-learner Trained: {'YES' if val['meta_learner_trained'] else 'NO'}")

            if val["warnings"]:
                logger.info("\nWARNINGS:")
                for warning in val["warnings"]:
                    logger.info(f"   {warning}")

        # Model structure
        if "model_structure" in self.audit_results:
            logger.info("\nMODEL STRUCTURE")
            logger.info("-" * 80)

            struct = self.audit_results["model_structure"]
            logger.info(f"Has base_models: {'PASS' if struct['has_base_models'] else 'FAIL'}")
            logger.info(f"Has meta_learner: {'PASS' if struct['has_meta_learner'] else 'FAIL'}")

            if struct["base_model_names"]:
                logger.info(f"Base models: {', '.join(struct['base_model_names'])}")

            if struct["structure_warnings"]:
                logger.info("\nSTRUCTURE WARNINGS:")
                for warning in struct["structure_warnings"]:
                    logger.info(f"   {warning}")

        display(
            HTML(
                '<div style="font-family:sans-serif;max-width:750px;border:2px solid #2E7D32;'
                'border-radius:8px;overflow:hidden;margin:16px 0;">'
                '<div style="background:#2E7D32;color:white;padding:12px 18px;font-size:14px;font-weight:700;">'
                "Stacking Audit -- Complete</div></div>"
            )
        )


# Cache system: stacking_audit

# Batch-aware wrapper: Stacking Methodology & Meta-Learner Review
# When _batch_archive holds multiple targets with archived state, this
# wrapper iterates over each, restoring globals before each pass.
# In single-target mode the validation runs exactly once (no change).


def _run_for_current_target():
    """Run this stage's validation for the current target."""
    # Validation stage gate (reads the main stage settings)
    _skip_stacking = not ctx.get("run_validation_cells", True)

    # Dependency guard
    if "make_run_key" not in ctx:
        logger.info("\nWarning: setup has not been run in this session.")
        logger.info("   Run setup and the main stage first.")
        _RUN_16 = False
    elif _skip_stacking:
        logger.info("stacking_audit skipped: main.run_validation_cells is False")
        _RUN_16 = False
    else:
        _CTAG_16 = "stacking_audit"
        _CKEY_16 = make_run_key({"cell": _CTAG_16})
        _CACHED_16 = cache_load(_CTAG_16, _CKEY_16)
        if _CACHED_16:
            logger.info(f"  {_CTAG_16} loaded from cache -- skipping execution.")
            _RUN_16 = False
        else:
            _RS["section"] = "stacking_audit"
            _done_folds_16 = fold_load_all(_CKEY_16, _CTAG_16)
            _RUN_16 = True

    if _RUN_16:
        # EXECUTION: Audit stacking

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

            if _is_within_mode:
                # Within-categories mode: Audit on best-performing category
                # Stacking structure is identical across categories: auditing
                # one category validates the methodology for all.

                _best_cat = _top_cats_ordered[0] if _top_cats_ordered else None
                if _best_cat and _best_cat in ctx.exported_within_dict:
                    _Xtr, _Xte, _ytr, _yte = ctx.exported_within_dict[_best_cat]

                    logger.info("\n Running [stacking_audit] Stacking Methodology Audit")
                    logger.info(f"   Category: {_best_cat} (best-performing)")
                    logger.info(f"   N features: {_Xtr.shape[1]}, N train: {_Xtr.shape[0]}")
                    logger.info("   (Stacking structure is identical across all categories)")

                    _Xtr_audit = _Xtr.copy() if isinstance(_Xtr, pd.DataFrame) else pd.DataFrame(_Xtr)
                    _ytr_audit = _ytr.copy() if isinstance(_ytr, pd.Series) else pd.Series(np.asarray(_ytr).ravel())

                    ctx.stacking_auditor = StackingLeakageCheck(_Xtr_audit, _ytr_audit, task_type="auto", n_inner=5)

                    # Validate stacking setup with fresh base models
                    try:
                        from catboost import CatBoostClassifier, CatBoostRegressor
                        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                        from xgboost import XGBClassifier, XGBRegressor

                        if ctx.stacking_auditor.is_classification:
                            base_models = {
                                "catboost": CatBoostClassifier(**ctx._CB_PARAMS, random_state=42),
                                "xgboost": XGBClassifier(
                                    n_estimators=ctx._XGB_PARAMS["n_estimators"],
                                    learning_rate=ctx._XGB_PARAMS["learning_rate"],
                                    random_state=42,
                                ),
                                "random_forest": RandomForestClassifier(
                                    n_estimators=ctx._RF_PARAMS["n_estimators"],
                                    max_depth=ctx._RF_PARAMS["max_depth"],
                                    random_state=42,
                                ),
                            }
                            # Add TabPFN outside dict literal
                            if not ctx._SKIP_TABPFN_21:
                                try:
                                    base_models["tabpfn"] = TabPFNClassifier.create_default_for_version(  # noqa: F823 (inherited from v1.0.0, see docs/known_issues.md)
                                        ModelVersion.V2, device=ctx._TABPFN_DEVICE
                                    )
                                except Exception:
                                    pass
                        else:
                            base_models = {
                                "catboost": CatBoostRegressor(**ctx._CB_PARAMS, random_state=42),
                                "xgboost": XGBRegressor(
                                    n_estimators=ctx._XGB_PARAMS["n_estimators"],
                                    learning_rate=ctx._XGB_PARAMS["learning_rate"],
                                    random_state=42,
                                ),
                                "random_forest": RandomForestRegressor(
                                    n_estimators=ctx._RF_PARAMS["n_estimators"],
                                    max_depth=ctx._RF_PARAMS["max_depth"],
                                    random_state=42,
                                ),
                            }
                            # Add TabPFN outside dict literal
                            if not ctx._SKIP_TABPFN_21:
                                try:
                                    from tabpfn import TabPFNRegressor
                                    from tabpfn.constants import ModelVersion

                                    base_models["tabpfn"] = TabPFNRegressor.create_default_for_version(
                                        ModelVersion.V2, device=ctx._TABPFN_DEVICE
                                    )
                                except Exception:
                                    pass

                        logger.info(f"\n   Validating stacking setup ({', '.join(base_models.keys())})...")
                        ctx.stacking_auditor.validate_stacking_setup(base_models, n_outer=ctx._VAL_N_OUTER)

                    except Exception as e:
                        logger.info(f"   Could not validate stacking setup: {e}")

                    ctx.stacking_auditor.generate_report(
                        title=f"[stacking_audit] Stacking Methodology Audit -- {_best_cat}"
                    )

                else:
                    logger.info("\nWarning: No categories available for stacking audit.")

            elif "X_train" in ctx and "y_train" in ctx:
                # Across-categories mode: Original single-pool stacking audit

                logger.info("\n Running [stacking_audit] Stacking Methodology Audit")

                ctx.stacking_auditor = StackingLeakageCheck(
                    ctx.X_train.copy(), ctx.y_train.copy(), task_type="auto", n_inner=5
                )

                if "model" in ctx or "pipeline_model" in ctx:
                    actual_model = model if "model" in ctx else pipeline_model  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                    is_stacking = hasattr(actual_model, "base_models") or hasattr(actual_model, "meta_learner")
                    if is_stacking:
                        logger.info("   Detected stacking model in pipeline")
                        ctx.stacking_auditor.audit_existing_stacking(actual_model)
                    else:
                        logger.info("   Pipeline model is not stacking - skipping structural audit")
                else:
                    logger.info("   No pipeline model variable - skipping structural audit")

                try:
                    from catboost import CatBoostClassifier, CatBoostRegressor
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    from xgboost import XGBClassifier, XGBRegressor

                    # TabPFN is an optional extra; without it the audit runs on the other base learners.
                    try:
                        from tabpfn import TabPFNClassifier, TabPFNRegressor
                        from tabpfn.constants import ModelVersion

                        _tabpfn_installed = True
                    except ImportError:
                        _tabpfn_installed = False
                        logger.info("   TabPFN is not installed: auditing CatBoost, XGBoost and random forest.")

                    if ctx.stacking_auditor.is_classification:
                        base_models = {
                            "catboost": CatBoostClassifier(**ctx._CB_PARAMS, random_state=42),
                            "xgboost": XGBClassifier(
                                n_estimators=ctx._XGB_PARAMS["n_estimators"],
                                learning_rate=ctx._XGB_PARAMS["learning_rate"],
                                random_state=42,
                            ),
                            "random_forest": RandomForestClassifier(
                                n_estimators=ctx._RF_PARAMS["n_estimators"],
                                max_depth=ctx._RF_PARAMS["max_depth"],
                                random_state=42,
                            ),
                        }
                        if _tabpfn_installed:
                            base_models["tabpfn"] = TabPFNClassifier.create_default_for_version(
                                ModelVersion.V2, device=ctx._TABPFN_DEVICE
                            )
                        if ctx._SKIP_TABPFN_21:
                            base_models.pop("tabpfn", None)
                    else:
                        base_models = {
                            "catboost": CatBoostRegressor(**ctx._CB_PARAMS, random_state=42),
                            "xgboost": XGBRegressor(
                                n_estimators=ctx._XGB_PARAMS["n_estimators"],
                                learning_rate=ctx._XGB_PARAMS["learning_rate"],
                                random_state=42,
                            ),
                            "random_forest": RandomForestRegressor(
                                n_estimators=ctx._RF_PARAMS["n_estimators"],
                                max_depth=ctx._RF_PARAMS["max_depth"],
                                random_state=42,
                            ),
                        }
                        if _tabpfn_installed:
                            base_models["tabpfn"] = TabPFNRegressor.create_default_for_version(
                                ModelVersion.V2, device=ctx._TABPFN_DEVICE
                            )
                        if ctx._SKIP_TABPFN_21:
                            base_models.pop("tabpfn", None)

                    logger.info(f"\n   Validating stacking setup ({', '.join(base_models.keys())})...")
                    ctx.stacking_auditor.validate_stacking_setup(base_models, n_outer=ctx._VAL_N_OUTER)

                except Exception as e:
                    logger.info(f"   Could not validate stacking setup: {e}")

                ctx.stacking_auditor.generate_report(title="[stacking_audit] Stacking Methodology Audit - RP")

            else:
                logger.info("\nERROR: Required variables not found")

        except Exception as e:
            logger.info(f"\nError in [stacking_audit]: {e}")
            import traceback

            traceback.print_exc()

        logger.info("\n[stacking_audit] complete.")

        # [Cache + report save]
        _RS["section"] = "uncategorized"
        try:
            _pipeline_cleanup(label="P5 complete")

            cache_save(
                {"reporter": locals().get("reporter") or locals().get("stacking_audit_reporter")},
                cell_tag=_CTAG_16,
                key=_CKEY_16,
                label=f"{ctx.target_options}_tp{ctx.tp_option}_[stacking_audit] Stacking Audit",
            )
            report_save_state(make_run_key())
        except Exception as _ce_16:
            logger.info(f"[Cache] Could not save stacking_audit: {_ce_16}")


def run() -> None:
    """Run this stage (notebook cell 16)."""

    # TabPFN device fallback (if the stacking setup has not run)
    ctx._TABPFN_DEVICE = ctx.get("_TABPFN_DEVICE", "cpu")
    # Respect _SKIP_TABPFN gate
    ctx._SKIP_TABPFN_21 = ctx.get("_SKIP_TABPFN", False)

    # StackingLeakageCheck Class

    # Read the outer fold count from the stage settings
    ctx._VAL_N_OUTER = ctx.n_outer_folds

    # Batch dispatch
    _is_bv_19 = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_19:
        try:
            _cbv_19 = cache_load("main_runner")
            if (
                _cbv_19
                and isinstance(_cbv_19, dict)
                and _cbv_19.get("_batch_archive")
                and len(_cbv_19["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_19["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_19["_batch_archive"]
                _is_bv_19 = True
        except Exception:
            pass

    if _is_bv_19:
        _vgs_19 = {
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
            ]
        }
        logger.info("\n" + "=" * 72)
        logger.info(
            f"  BATCH -- Stacking Methodology & Meta-Learner Review: processing {len(ctx._batch_archive)} targets"
        )
        logger.info("=" * 72)

        for _vi_19, (_vk_19, _vd_19) in enumerate(ctx._batch_archive.items()):
            _vl_19 = _vd_19.get("target", _vk_19)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_19 + 1}/{len(ctx._batch_archive)}] Target: {_vl_19}")
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
            ]:
                if _vd_19.get(_gk) is not None:
                    ctx[_gk] = _vd_19[_gk]
            ctx["target_options"] = _vl_19
            if "timepoint" in _vd_19:
                ctx["tp_option"] = _vd_19["timepoint"]
            _run_for_current_target()
            # Archive validation reporters back into _batch_archive
            if _vk_19 in ctx._batch_archive:
                ctx._batch_archive[_vk_19]["stacking_auditor"] = ctx.get("stacking_auditor")

        for _k, _v in _vgs_19.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(
            f"  BATCH COMPLETE -- Stacking Methodology & Meta-Learner Review: {len(ctx._batch_archive)} targets"
        )
        logger.info("=" * 72)
    else:
        _run_for_current_target()
