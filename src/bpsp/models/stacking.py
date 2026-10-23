"""Stacking ensemble architecture and model hyperparameters (CatBoost, XGBoost, random forest, TabPFN, meta-learner).

Two-stage stacking ensemble. Stage 1: base learners (CatBoost, XGBoost,
random forest, TabPFN) generate out-of-fold predictions via inner CV.
Stage 2: a linear (regression) or logistic (classification) meta-learner
trains on those held-out predictions. Tree-based models capture nonlinear
interactions across mixed variable types; TabPFN approximates Bayesian
posteriors for uncertainty quantification. TabPFN is skipped automatically
in across-category runs, because feature counts there typically exceed
TabPFN's recommended d <= 500 ceiling. ElasticNet is retained for
comparison only and is not part of the reported ensemble.

TabPFN (Hollmann et al., 2025, Nature) is a transformer-based prior-data
fitted network that approximates Bayesian posterior predictive
distributions over tabular data, providing built-in uncertainty
quantification without manual hyperparameter tuning. The stacking
procedure is checked for data leakage by the `stacking_audit` stage.

The model hyperparameters are set once, in PIPELINE_MODEL_CONFIGS. The
main stage and the nested_cv, stratified, grouped_cv, permutation and
stacking_audit stages read them from there; the lighter CatBoost models of
the other validation stages are set in create_lightweight_model. CatBoost's
best_params can be replaced per run by an Optuna search when
hyperparameter_tuning is on; the values here are the defaults used for the
article. The TabPFN device (GPU when available) is detected once and
reused.
"""

import logging

from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold

from bpsp.context import ctx

logger = logging.getLogger(__name__)

try:
    import torch as _torch
except ImportError:  # optional: `pip install .[tabpfn]` for TabPFN / GPU
    _torch = None


# Target Display Name Resolution
def _resolve_target_name(raw_name):
    """Resolve raw target variable name to publication-ready display name.
    Uses TARGET_DISPLAY_NAMES from preprocessing if available; falls back to raw name.
    Called at runtime (after this module has run)."""
    _tdn = ctx.get("TARGET_DISPLAY_NAMES")
    return _tdn.get(raw_name, raw_name) if isinstance(_tdn, dict) else raw_name


# Fixed defaults
ensemble_fig_width = 8
ensemble_fig_height = 6
ensemble_dpi = 500


def _resolve_ensemble_palette():
    """Resolve the effective palette/color for ensemble barplots.
    If a named palette is chosen (not 'None'), use it.
    Otherwise, build a dark-to-light gradient from the single color pick,
    so the top bar (rank 1) is the full color and the bottom bar fades
    to a pale tint: matching manuscript figure aesthetics.
    """
    if ctx.ensemble_barplot_palette != "None":
        return ctx.ensemble_barplot_palette
    # Gradient mode: dark → light tint of the target color
    import matplotlib.colors as mcolors
    import numpy as np

    n = ctx.ENSEMBLE_VIZ_CONFIG.get("top_n", 10)
    try:
        base_rgb = np.array(mcolors.to_rgb(ctx.ensemble_barplot_color))
    except ValueError:
        base_rgb = np.array(mcolors.to_rgb("steelblue"))
    white = np.array([1.0, 1.0, 1.0])
    # Blend from 100% base (rank 1) → 30% base + 70% white (rank n)
    fracs = np.linspace(0.0, 0.92, n)
    return [mcolors.to_hex(base_rgb * (1 - f) + white * f) for f in fracs]


import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier, XGBRegressor

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    from catboost import CatBoostClassifier, CatBoostRegressor

from typing import Any, Dict, List, Union

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Helper functions


def adjusted_r2_score(y_true, y_pred, n_features):
    """Adjusted R², 1 - (1 - R²)(n - 1)/(n - p - 1), for n rows and p predictors.

    Undefined when n <= p + 1 (the formula changes sign and can exceed 1), so NaN is returned.
    """
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)
    if n_samples - n_features - 1 <= 0:
        return np.nan
    adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adjusted_r2


def preprocess_boolean_target(y):
    """Coerce boolean, string-boolean, or two-valued numeric targets to {0, 1}.
    Required for learners that do not accept bool dtype labels.
    """
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    elif isinstance(y, pd.Series):
        y = y.copy()
    else:
        y = pd.Series(y)

    # Handle boolean values
    if y.dtype == bool:
        return y.astype(int)

    # Handle string 'True'/'False' values
    if y.dtype == object:
        try:
            return y.map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)
        except Exception as e:
            logger.info(f"[WARNING] {e}")
            pass

    # Handle numeric values that should be boolean
    unique_vals = np.unique(y)
    if len(unique_vals) == 2:
        # Map the smaller value to 0 and larger value to 1
        val_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        return y.map(val_map).astype(int)

    return y


# Utility wrapper used when the target dtype is bool rather than int.
def modify_ensemble_for_boolean(model, X_train, y_train):
    """Inspect the target variable and force classification mode on an
    StackingEnsembleModel instance if the target is binary-coded boolean.
    """
    # Process the target variable
    y_processed = preprocess_boolean_target(y_train)

    # Check if we should force classification mode
    unique_vals = np.unique(y_processed)
    if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
        model.task_type = "classification"

        # Reinitialize models for classification
        if hasattr(model, "_init_classification_models"):
            model._init_classification_models()

    return model, y_processed


def _safe_clone(estimator):
    """Clone that handles CatBoost's numpy-typed class_weights."""
    try:
        return clone(estimator)
    except RuntimeError:
        params = {}
        for k, v in estimator.get_params().items():
            if isinstance(v, dict):
                # Convert numpy keys/values to native Python types
                params[k] = {
                    int(dk) if hasattr(dk, "item") else dk: float(dv) if hasattr(dv, "item") else dv
                    for dk, dv in v.items()
                }
            else:
                params[k] = v
        return estimator.__class__(**params)


class StackingEnsembleModel(BaseEstimator):
    # Trains base learners in parallel; uses their OOF predictions as inputs
    # to a linear meta-learner.
    def __init__(
        self,
        task_type: str = "regression",
        model_configs: Dict[str, Dict[str, Any]] = None,
        meta_model=None,
        cv: int = 5,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.model_configs = model_configs or {}
        self.cv = cv
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = meta_model
        self.class_weights = None
        self.tabpfn_feature_reduction = None

        # Initialize base models
        if self.task_type == "regression":
            self._init_regression_models()
            if self.meta_model is None:
                self.meta_model = LinearRegression()
        else:
            self._init_classification_models()
            if self.meta_model is None:
                # For classification, use LogisticRegression with random_state
                self.meta_model = LogisticRegression(random_state=self.random_state)

    def _init_regression_models(self):
        """Instantiate base regression learners.  TabPFN is opt-in: include
        'tabpfn' as a key in model_configs to enable it.
        """
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.svm import SVR

        default_configs = {
            "catboost": {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3,
                "border_count": 128,
                "thread_count": -1,
                "verbose": False,
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "bootstrap": True,
            },
            "xgboost": {
                "n_estimators": 500,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        }

        # TabPFN: opt-in only (include 'tabpfn' in model_configs to enable)
        if "tabpfn" in self.model_configs:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion

            tabpfn_cfg = self.model_configs["tabpfn"]
            self.base_models["tabpfn"] = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2, device=tabpfn_cfg.get("device", ctx._TABPFN_DEVICE)
            )

        # Merge default configs with user-provided configs
        for model_name in default_configs:
            config = {**default_configs[model_name], **self.model_configs.get(model_name, {})}
            config["random_state"] = self.random_state

            # Remove random_state from config if model doesn't support it
            if model_name == "svm":
                config.pop("random_state", None)

            if model_name == "catboost":
                self.base_models["catboost"] = CatBoostRegressor(**config)
            elif model_name == "random_forest":
                self.base_models["random_forest"] = RandomForestRegressor(**config)
            elif model_name == "xgboost":
                self.base_models["xgboost"] = XGBRegressor(**config)
            elif model_name == "svm":
                self.base_models["svm"] = Pipeline([("scaler", RobustScaler()), ("svr", SVR(**config))])
            elif model_name == "neural_net":
                self.base_models["neural_net"] = Pipeline([("scaler", RobustScaler()), ("mlp", MLPRegressor(**config))])

    def _init_classification_models(self):
        """Instantiate base classification learners including optional TabPFN."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.svm import SVC

        default_configs = {
            "catboost": {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 6,
                "verbose": False,
                "class_weights": self.class_weights,
            },
            "random_forest": {"n_estimators": 200, "max_depth": 10, "class_weight": self.class_weights},
            "xgboost": {
                "n_estimators": 500,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": None,
            },
        }

        # TabPFN: opt-in only (include 'tabpfn' in model_configs to enable)
        if "tabpfn" in self.model_configs:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion

            tabpfn_cfg = self.model_configs["tabpfn"]
            self.base_models["tabpfn"] = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, device=tabpfn_cfg.get("device", ctx._TABPFN_DEVICE)
            )
        """
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'probability': True,
                'class_weight': 'balanced'
            },"""

        # Merge default configs with user-provided configs
        for model_name in default_configs:
            config = {**default_configs[model_name], **self.model_configs.get(model_name, {})}
            config["random_state"] = self.random_state

            # Remove random_state from config if model doesn't support it
            if model_name == "svm":
                config.pop("random_state", None)

            if model_name == "catboost":
                self.base_models["catboost"] = CatBoostClassifier(**config)
            elif model_name == "random_forest":
                self.base_models["random_forest"] = RandomForestClassifier(**config)
            elif model_name == "xgboost":
                self.base_models["xgboost"] = XGBClassifier(**config)
            elif model_name == "svm":
                self.base_models["svm"] = Pipeline([("scaler", RobustScaler()), ("svc", SVC(**config))])
            elif model_name == "neural_net":
                self.base_models["neural_net"] = Pipeline(
                    [("scaler", RobustScaler()), ("mlp", MLPClassifier(**config))]
                )

    # Set class weights for imbalanced classification targets
    def set_class_weights(self, y):
        """Set class weights for all base models and meta model"""
        if self.task_type == "regression":
            return

        # Convert y to numpy array if it's a pandas Series/DataFrame
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()

        classes = np.unique(y)
        classes.sort()
        logger.info("\nClass distribution in training data:")
        for c in classes:
            logger.info(f"Class {c}: {np.sum(y == c)} samples")

        weights = compute_class_weight("balanced", classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))
        logger.info("\nComputed class weights:")
        for c, w in self.class_weights.items():
            logger.info(f"Class {c}: weight {w:.4f}")

        # Update and verify each model's class weights
        for name, model in self.base_models.items():
            if name == "catboost":
                logger.info(f"\nSetting class weights for {name}")
                model.set_params(class_weights=self.class_weights)
                # Verify weights were set
                params = model.get_params()
                logger.info(f"CatBoost class weights after setting: {params.get('class_weights')}")

            elif name == "random_forest":
                logger.info(f"\nSetting class weights for {name}")
                model.set_params(class_weight=self.class_weights)
                # Verify weights were set
                params = model.get_params()
                logger.info(f"RandomForest class weights after setting: {params.get('class_weight')}")

            elif name == "xgboost":
                if len(classes) == 2:
                    logger.info(f"\nSetting scale_pos_weight for {name}")
                    neg_count = np.sum(y == classes[0])
                    pos_count = np.sum(y == classes[1])
                    scale_weight = float(np.sqrt(neg_count / pos_count))
                    logger.info(
                        f"scale_pos_weight (sqrt): {scale_weight:.4f}  (full ratio would be {neg_count / pos_count:.4f})"
                    )
                    model.set_params(scale_pos_weight=scale_weight)
                    # Verify weight was set
                    params = model.get_params()
                    logger.info(f"XGBoost scale_pos_weight after setting: {params.get('scale_pos_weight')}")

        # Meta-learner intentionally left WITHOUT class weights.
        # Base models already produce class-weight-adjusted OOF predictions;
        # weighting the meta-learner again would double-correct for imbalance.
        logger.info("\nMeta model (LogisticRegression): no class weights (avoids double-weighting)")

    def _get_meta_features(self, X, y=None, is_train=True):
        """Generate out-of-fold meta-features for stacking ensemble.
        TabPFN: CatBoost-based feature selection (>100), fresh instance per fold,
        feature alignment on prediction.
        """
        if hasattr(X, "get_features"):
            X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names())
        else:
            X_features = X

        if is_train:
            if self.task_type == "classification" and y is not None:
                self.n_classes_ = len(np.unique(y))
            else:
                self.n_classes_ = 2
            meta_features = np.zeros((X_features.shape[0], len(self.base_models)))
            if self.task_type == "classification":
                from sklearn.model_selection import StratifiedKFold

                kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

            # Feature selection for TabPFN (>100 features)
            tabpfn_features = None
            if X_features.shape[1] > 100 and "tabpfn" in self.base_models and "catboost" in self.base_models:
                logger.info("\nSelecting top features for TabPFN using CatBoost importance...")
                if self.task_type == "regression":
                    temp_cb = CatBoostRegressor(
                        iterations=100, learning_rate=0.1, verbose=False, random_state=self.random_state
                    )
                else:
                    temp_cb = CatBoostClassifier(
                        iterations=100, learning_rate=0.1, verbose=False, random_state=self.random_state
                    )
                temp_cb.fit(X_features, y)
                importance = pd.Series(temp_cb.feature_importances_, index=X_features.columns)
                selected_features = importance.nlargest(100).index
                tabpfn_features = X_features[selected_features]
                self.tabpfn_feature_reduction = selected_features
                logger.info(f"Selected {len(selected_features)} top features for TabPFN")

            for i, (name, model) in enumerate(self.base_models.items()):
                logger.info(f"\nGenerating meta-features for {name}...")
                temp_meta_features = np.zeros(X_features.shape[0])

                if name == "tabpfn":
                    X_for_tabpfn = tabpfn_features if tabpfn_features is not None else X_features
                    for train_idx, val_idx in kf.split(X_for_tabpfn, y):
                        X_train_fold = X_for_tabpfn.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X_for_tabpfn.iloc[val_idx]
                        from tabpfn.constants import ModelVersion

                        if self.task_type == "regression":
                            from tabpfn import TabPFNRegressor

                            fold_model = TabPFNRegressor.create_default_for_version(
                                ModelVersion.V2, device=ctx._TABPFN_DEVICE
                            )
                        else:
                            from tabpfn import TabPFNClassifier

                            fold_model = TabPFNClassifier.create_default_for_version(
                                ModelVersion.V2, device=ctx._TABPFN_DEVICE
                            )
                        try:
                            fold_model.fit(X_train_fold, y_train_fold)
                            if self.task_type == "classification":
                                fold_preds = fold_model.predict_proba(X_val_fold)[:, 1]
                            else:
                                fold_preds = fold_model.predict(X_val_fold)
                        except Exception as e:
                            logger.info(f"TabPFN fold error: {e}, using fallback")
                            fold_preds = np.full(len(X_val_fold), np.mean(y_train_fold))
                        temp_meta_features[val_idx] = fold_preds
                    if tabpfn_features is not None:
                        model.selected_features_ = selected_features

                else:
                    for train_idx, val_idx in kf.split(X_features, y):
                        X_train_fold = X_features.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X_features.iloc[val_idx]
                        model_clone = _safe_clone(model)
                        if self.task_type == "classification":
                            if name == "catboost":
                                model_clone.set_params(class_weights=self.class_weights)
                            elif name == "random_forest":
                                model_clone.set_params(class_weight=self.class_weights)
                            elif name == "xgboost" and len(np.unique(y)) == 2:
                                _y_spw = np.asarray(y).ravel()
                                _cls = np.unique(_y_spw)
                                model_clone.set_params(
                                    scale_pos_weight=float(
                                        np.sqrt(np.sum(_y_spw == _cls[0]) / np.sum(_y_spw == _cls[1]))
                                    )
                                )
                        model_clone.fit(X_train_fold, y_train_fold)
                        # Use predict_proba for classification (consistent scale with TabPFN)
                        if self.task_type == "classification" and hasattr(model_clone, "predict_proba"):
                            if getattr(self, "n_classes_", 2) <= 2:
                                fold_preds = model_clone.predict_proba(X_val_fold)[:, 1]
                            else:
                                fold_preds = model_clone.predict(X_val_fold).astype(float)
                        else:
                            fold_preds = model_clone.predict(X_val_fold)
                        temp_meta_features[val_idx] = fold_preds

                meta_features[:, i] = temp_meta_features

                # Fit on full data
                if name == "tabpfn":
                    X_for_tabpfn = tabpfn_features if tabpfn_features is not None else X_features
                    from tabpfn.constants import ModelVersion

                    if self.task_type == "regression":
                        from tabpfn import TabPFNRegressor

                        self.base_models[name] = TabPFNRegressor.create_default_for_version(
                            ModelVersion.V2, device=ctx._TABPFN_DEVICE
                        )
                    else:
                        from tabpfn import TabPFNClassifier

                        self.base_models[name] = TabPFNClassifier.create_default_for_version(
                            ModelVersion.V2, device=ctx._TABPFN_DEVICE
                        )
                    _sw_full = None
                    if self.task_type == "classification" and self.class_weights is not None:
                        _sw_full = np.array([float(self.class_weights.get(yi, 1.0)) for yi in y])
                    try:
                        try:
                            self.base_models[name].fit(X_for_tabpfn, y, sample_weight=_sw_full)
                        except TypeError:
                            self.base_models[name].fit(X_for_tabpfn, y)
                    except Exception as _tabpfn_err:
                        logger.info(
                            f"[WARNING] TabPFN full-data fit failed: {_tabpfn_err}. {'GPU OOM -- subsampling to ≤9999' if ctx._TABPFN_DEVICE == 'cuda' else 'CPU limit -- subsampling to ≤999'}."
                        )
                        _max_n = min(9999 if ctx._TABPFN_DEVICE == "cuda" else 999, len(X_for_tabpfn))
                        if ctx._TABPFN_DEVICE == "cuda":
                            import torch

                            torch.cuda.empty_cache()
                        _sub_idx = np.random.choice(len(X_for_tabpfn), _max_n, replace=False)
                        try:
                            self.base_models[name].fit(X_for_tabpfn.iloc[_sub_idx], y.iloc[_sub_idx])
                        except Exception as _e2:
                            logger.info(
                                f"[WARNING] TabPFN subsampled fit also failed: {_e2}. Predictions may be degraded."
                            )
                    if tabpfn_features is not None:
                        self.base_models[name].selected_features_ = selected_features  # transfer to new instance
                else:
                    model.fit(X_features, y)

        else:
            meta_features = []
            for name, model in self.base_models.items():
                if name == "tabpfn":
                    if hasattr(model, "selected_features_"):
                        X_for_tabpfn = X_features[model.selected_features_]
                    else:
                        X_for_tabpfn = X_features
                    if hasattr(model, "feature_names_in_"):
                        X_for_tabpfn = X_for_tabpfn.reindex(columns=list(model.feature_names_in_), fill_value=0)
                    try:
                        if self.task_type == "classification":
                            if getattr(self, "n_classes_", 2) <= 2:
                                preds = model.predict_proba(X_for_tabpfn)[:, 1]
                            else:
                                preds = model.predict(X_for_tabpfn).astype(float)
                        else:
                            preds = model.predict(X_for_tabpfn)
                    except Exception as e:
                        logger.info(f"TabPFN prediction error: {e}, using fallback")
                        fallback = 0.5 if self.task_type == "classification" else 0.0
                        preds = np.full(X_features.shape[0], fallback)
                else:
                    if self.task_type == "classification" and hasattr(model, "predict_proba"):
                        if getattr(self, "n_classes_", 2) <= 2:
                            preds = model.predict_proba(X_features)[:, 1]
                        else:
                            preds = model.predict(X_features).astype(float)
                    else:
                        preds = model.predict(X_features)
                meta_features.append(preds)

            meta_features = np.column_stack(meta_features)

        return meta_features

    def explain_meta_model(self, X_train, y_train):

        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        import shap
        from matplotlib.colors import LinearSegmentedColormap

        # Generate meta-features (base model predictions)
        meta_features = self._get_meta_features(X_train, y_train, is_train=False)

        # Get base model names for labels
        base_names = list(self.base_models.keys())

        logger.info(f"Creating network visualization for meta-model: {type(self.meta_model).__name__}")

        # Calculate SHAP values
        try:
            background = shap.kmeans(meta_features, 10)
            explainer = shap.KernelExplainer(self.meta_model.predict, background)
            shap_values = explainer.shap_values(meta_features)

            # Convert to numpy array if needed
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for multi-class

            # Calculate feature importance (1st order effects)
            feature_importance = np.abs(shap_values).mean(axis=0)

            # Calculate pairwise interactions (2nd order effects)
            interactions = np.zeros((len(base_names), len(base_names)))

            # This is a simple approximation for interactions
            # Using shap.TreeExplainer.shap_interaction_values would be more accurate
            # but doesn't work for non-tree models
            for i in range(len(base_names)):
                for j in range(i + 1, len(base_names)):
                    # Calculate interaction strength through permutation analysis
                    interaction_strength = shap.utils.approximate_interactions(i, j, shap_values, meta_features)
                    interactions[i, j] = interaction_strength
                    interactions[j, i] = interaction_strength

            # Create network graph
            G = nx.Graph()

            # Add nodes (features)
            for i, name in enumerate(base_names):
                G.add_node(name, size=feature_importance[i])

            # Add edges (interactions)
            for i in range(len(base_names)):
                for j in range(i + 1, len(base_names)):
                    if abs(interactions[i, j]) > 0.01:  # Threshold to reduce noise
                        G.add_edge(
                            base_names[i],
                            base_names[j],
                            weight=abs(interactions[i, j]),
                            color="red" if interactions[i, j] > 0 else "blue",
                        )

            # Custom colormap similar to ShapIQ
            colors = [(0.0, "darkblue"), (0.5, "white"), (1.0, "darkred")]
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

            # Create plot
            plt.figure(figsize=(12, 10))

            # Compute position using spring layout
            pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), seed=42)

            # Draw nodes with size based on importance
            node_sizes = [G.nodes[n]["size"] * 5000 + 100 for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightgray", alpha=0.8)

            # Draw edges with width and color based on interaction strength
            for u, v, data in G.edges(data=True):
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    width=data["weight"] * 5,
                    edge_color=data["color"],
                    alpha=min(0.8, data["weight"]),
                )

            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

            plt.title("Meta-model Feature Interaction Network", fontsize=16)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            # Create interaction matrix visualization
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(interactions, dtype=bool))  # Mask for upper triangle
            import seaborn as sns

            sns.heatmap(
                interactions,
                mask=mask,
                cmap=custom_cmap,
                center=0,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                xticklabels=base_names,
                yticklabels=base_names,
            )
            plt.title("Meta-model Pairwise Interaction Matrix", fontsize=16)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.info(f"Error creating network visualization: {str(e)}")
            import traceback

            traceback.print_exc()

        return

    def fit(self, X, y, categorical_features=None):
        """Fit the stacking ensemble model"""
        # Convert Pool to features if needed
        if hasattr(X, "get_features"):
            X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names())
        else:
            X_features = X

        # Set class weights before generating meta-features
        if self.task_type == "classification":
            self.set_class_weights(y)

        # Generate meta-features for training
        meta_features = self._get_meta_features(X_features, y, is_train=True)

        # Fit the meta-model on the meta-features
        self.meta_model.fit(meta_features, y)

        return self

    def predict(self, X):
        # Convert Pool to features if needed
        if hasattr(X, "get_features"):
            X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names())
        else:
            X_features = X

        # Generate meta-features for test data
        meta_features = self._get_meta_features(X_features, is_train=False)

        # Make final predictions using the meta-model
        predictions = self.meta_model.predict(meta_features)

        if self.task_type == "classification":
            return np.round(predictions).astype(int)
        return predictions

    def predict_proba(self, X):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")

        # Convert Pool to features if needed
        if hasattr(X, "get_features"):
            X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names())
        else:
            X_features = X

        # Generate meta-features for test data
        meta_features = self._get_meta_features(X_features, is_train=False)

        # Make probability predictions using the meta-model
        return self.meta_model.predict_proba(meta_features)

    def get_consensus_features(self, X_train, top_k=40):
        """Get consensus important features from tree-based models"""
        all_important_features = set()

        # Get feature importance from tree-based models
        for name, model in self.base_models.items():
            if name in ["catboost", "random_forest", "xgboost"]:
                if hasattr(model, "feature_importances_"):
                    # Get feature importance
                    importances = pd.Series(model.feature_importances_, index=X_train.columns)
                    # Get top k features from this model
                    top_features = importances.nlargest(top_k).index
                    all_important_features.update(top_features)

        return list(all_important_features)

    def get_tabpfn_uncertainty(self, X):
        """Get uncertainty estimates from TabPFN"""
        if "tabpfn" not in self.base_models:
            return None

        try:
            # Subset features if TabPFN trained on CatBoost-selected subset
            if hasattr(self.base_models["tabpfn"], "selected_features_"):
                X = X[self.base_models["tabpfn"].selected_features_]
            if hasattr(self.base_models["tabpfn"], "feature_names_in_"):
                X = X.reindex(columns=list(self.base_models["tabpfn"].feature_names_in_), fill_value=0)

            if self.task_type == "classification":
                # Get probability predictions
                probas = self.base_models["tabpfn"].predict_proba(X)

                # Calculate confidence scores
                confidence = np.max(probas, axis=1)
                mean_confidence = np.mean(confidence)
                std_confidence = np.std(confidence)

                # Calculate entropy (uncertainty measure)
                entropy = -np.sum(probas * np.log2(probas + 1e-10), axis=1)
                mean_entropy = np.mean(entropy)

                uncertainty_stats = {
                    "mean_confidence": mean_confidence,
                    "std_confidence": std_confidence,
                    "confidence_interval": (
                        mean_confidence - 1.96 * std_confidence,
                        mean_confidence + 1.96 * std_confidence,
                    ),
                    "mean_entropy": mean_entropy,
                    "min_confidence": np.min(confidence),
                    "max_confidence": np.max(confidence),
                    "uncertain_samples": np.sum(confidence < 0.7),  # samples with <70% confidence
                    "high_confidence_samples": np.sum(confidence > 0.9),  # samples with >90% confidence
                }

            else:  # regression
                # For regression, we can use the variation in predictions
                predictions = self.base_models["tabpfn"].predict(X)
                mean_pred = np.mean(predictions)
                std_pred = np.std(predictions)

                uncertainty_stats = {
                    "mean_prediction": mean_pred,
                    "std_prediction": std_pred,
                    "prediction_interval": (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred),
                }

            return uncertainty_stats

        except Exception as e:
            logger.info(f"Error calculating TabPFN uncertainty: {str(e)}")
            return None

    # TabPFN SHAP values
    def get_tabpfn_shap_values(self, model, X_test, feature_names=None, n_samples=50):
        """
        Get SHAP values specifically for TabPFN model with proper handling.
        """
        n_samples = min(n_samples, len(X_test))
        X_test_subset = X_test.iloc[:n_samples] if isinstance(X_test, pd.DataFrame) else X_test[:n_samples]

        if feature_names is None:
            feature_names = (
                X_test.columns if isinstance(X_test, pd.DataFrame) else [f"feature_{i}" for i in range(X_test.shape[1])]
            )

        def predict_function_for_shap(X):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            return model.predict(X)

        background = np.ones((1, X_test_subset.shape[1])) * float("nan")
        explainer = shap.Explainer(predict_function_for_shap, background)

        return explainer(X_test_subset)

    def plot_tabpfn_shap(self, shap_values, feature_names=None):
        """
        Create SHAP visualizations for the TabPFN model.
        """
        if len(shap_values.shape) == 3:
            logger.info("Computing SHAP values for the first class (index 0).")
            shap_values = shap_values[:, :, 0]

        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_values=shap_values, show=False)
        plt.title("Aggregate Feature Importances (TabPFN)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values=shap_values, show=False)
        plt.title("Feature Importance Distribution (TabPFN)")
        plt.tight_layout()
        plt.show()

        most_important = np.abs(shap_values.values).mean(0).argsort()[-1]
        logger.info(f'\nAnalyzing interactions for most important feature: "{feature_names[most_important]}"')

        if len(shap_values) > 1:
            inds = shap.utils.potential_interactions(shap_values[:, most_important], shap_values)

            for i in range(min(3, len(inds))):
                plt.figure(figsize=(10, 6))
                shap.plots.scatter(shap_values[:, most_important], color=shap_values[:, inds[i]], show=False)
                plt.title(f"Interaction: {feature_names[most_important]} vs {feature_names[inds[i]]}")
                plt.tight_layout()
                plt.show()

    def explain_with_shap(self, X_train, X_test):
        """SHAP analysis with special handling for TabPFN."""
        if hasattr(X_train, "get_features"):
            X_train = pd.DataFrame(X_train.get_features(), columns=X_train.get_feature_names())
        if hasattr(X_test, "get_features"):
            X_test = pd.DataFrame(X_test.get_features(), columns=X_test.get_feature_names())

        shap_values_dict = {}

        for name, model in self.base_models.items():
            logger.info(f"\nGenerating SHAP values for base model: {name}...")

            try:
                if name == "tabpfn":
                    logger.info("\nPerforming specialized TabPFN SHAP analysis...")
                    shap_values = self.get_tabpfn_shap_values(
                        model, X_test, feature_names=X_test.columns, n_samples=min(50, len(X_test))
                    )

                    shap_values_dict[name] = shap_values
                    self.plot_tabpfn_shap(shap_values, feature_names=X_test.columns)

                elif name in ["catboost", "random_forest", "xgboost"]:
                    # Tree-based base learners: use TreeExplainer.
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    shap_values_dict[name] = shap_values

                    plt.figure(figsize=(20, 16))
                    if isinstance(shap_values, list):
                        shap.summary_plot(
                            shap_values[1] if len(shap_values) > 1 else shap_values,
                            X_test,
                            plot_type="violin",
                            show=False,
                            max_display=ctx.ENSEMBLE_VIZ_CONFIG["top_n"],
                            title=f"SHAP Values - {name}",
                        )
                    else:
                        shap.summary_plot(
                            shap_values,
                            X_test,
                            plot_type="violin",
                            show=False,
                            max_display=ctx.ENSEMBLE_VIZ_CONFIG["top_n"],
                            title=f"SHAP Values - {name}",
                        )

                    plt.title(f"SHAP Feature Importance and Impact - {name}", fontsize=24)
                    plt.xlabel("SHAP value (impact on model output)", fontsize=18)
                    plt.ylabel("Features", fontsize=18)
                    plt.tick_params(axis="both", which="major", labelsize=14)

                    cbar = plt.gcf().axes[-1]
                    cbar.set_ylabel("Feature value", fontsize=18, labelpad=20)
                    cbar.tick_params(labelsize=14)

                    plt.tight_layout()
                    plt.show()

            except Exception as e:
                logger.info(f"Could not generate SHAP values for {name}: {str(e)}")
                logger.info(f"Error type: {type(e)}")
                logger.info(f"Error details: {str(e)}")

        return shap_values_dict

    def get_feature_importance(self, X_train, y_train):
        """Feature importance for each base learner."""
        importance_dict = {}

        # TabPFN: uncertainty statistics here; its permutation importance is computed at the end
        if "tabpfn" in self.base_models:
            logger.info("\nCalculating TabPFN feature importance...")
            try:
                # Get uncertainty estimates
                uncertainty_stats = self.get_tabpfn_uncertainty(X_train)
                if uncertainty_stats:
                    logger.info("\nTabPFN Uncertainty Statistics:")
                    for metric, value in uncertainty_stats.items():
                        if isinstance(value, tuple):
                            logger.info(f"{metric}: ({value[0]:.4f}, {value[1]:.4f})")
                        else:
                            logger.info(f"{metric}: {value:.4f}")

            except Exception as e:
                logger.info(f"Could not calculate TabPFN importance: {str(e)}")

        # First get importance from tree-based models
        for name, model in self.base_models.items():
            if name in ["catboost", "random_forest", "xgboost"]:
                if hasattr(model, "feature_importances_"):
                    importance_dict[name] = model.feature_importances_
            elif name == "elasticnet":
                if hasattr(model, "coef_"):
                    importance_dict[name] = np.abs(model.coef_)

        # Get consensus features for SVM
        if "svm" in self.base_models:
            try:
                from sklearn.inspection import permutation_importance

                # Get important features from tree models
                important_features = self.get_consensus_features(X_train)
                logger.info(f"Number of consensus features for SVM: {len(important_features)}")

                # Before subsetting
                logger.info("%s %s", "NaN values in X_train:", X_train.isnull().sum().sum())

                # After subsetting
                X_subset = X_train[important_features]
                logger.info("%s %s", "NaN values in X_subset:", X_subset.isnull().sum().sum())

                # Check column continuity
                logger.info("%s %s", "X_subset columns:", X_subset.columns)
                logger.info("%s %s", "Important features:", important_features)

                # Handle NaN values before scaling if they exist
                if X_subset.isnull().any().any():  # Check if any NaN exists
                    logger.info(f"Found NaN values in X_train subset crunch: {X_subset.isnull().sum().sum()}")
                    imputer = SimpleImputer(strategy="median")
                    X_subset = pd.DataFrame(
                        imputer.fit_transform(X_subset), columns=X_subset.columns, index=X_subset.index
                    )
                    logger.info(f"Imputed {X_subset.isnull().sum().sum()} NaN values")
                else:
                    logger.info("No NaN values found for X_train subset")
                # Get the model and scaler from pipeline
                model = self.base_models["svm"]
                if hasattr(model, "named_steps"):
                    scaler = model.named_steps["scaler"]
                    X_scaled = scaler.transform(X_train_subset)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                    actual_model = model.named_steps["svr" if self.task_type == "regression" else "svc"]

                    # Check if linear kernel for faster computation
                    if hasattr(actual_model, "kernel") and actual_model.kernel == "linear":
                        logger.info("Using linear kernel coefficients for SVM importance")
                        # Create full-size importance array (all zeros except important features)
                        full_importance = np.zeros(X_train.shape[1])
                        for idx, feat in enumerate(important_features):
                            full_importance[X_train.columns.get_loc(feat)] = np.abs(actual_model.coef_[0][idx])
                        importance_dict["svm"] = full_importance
                    else:
                        logger.info("Using permutation importance for RBF kernel SVM")
                        # Calculate permutation importance only for important features
                        result = permutation_importance(
                            actual_model, X_scaled, y_train, n_repeats=10, random_state=self.random_state, n_jobs=-1
                        )

                        # Map back to full feature space
                        full_importance = np.zeros(X_train.shape[1])
                        for idx, feat in enumerate(important_features):
                            full_importance[X_train.columns.get_loc(feat)] = result.importances_mean[idx]
                        importance_dict["svm"] = full_importance

            except Exception as e:
                logger.info(f"Could not calculate importance for SVM: {str(e)}")

        # Calculate permutation importance once for TabPFN
        if "tabpfn" in self.base_models:
            logger.info("\nCalculating TabPFN feature importance using permutation importance...")
            try:
                # Subset features to match what TabPFN was trained on
                X_for_perm = X_train
                if hasattr(self.base_models["tabpfn"], "selected_features_"):
                    X_for_perm = X_train[self.base_models["tabpfn"].selected_features_]
                if hasattr(self.base_models["tabpfn"], "feature_names_in_"):
                    X_for_perm = X_for_perm.reindex(
                        columns=list(self.base_models["tabpfn"].feature_names_in_), fill_value=0
                    )
                result = permutation_importance(
                    self.base_models["tabpfn"], X_for_perm, y_train, n_repeats=10, random_state=42, n_jobs=-1
                )
                importance_dict["tabpfn"] = result.importances_mean
                logger.info("Successfully calculated TabPFN importance")
            except Exception as e:
                logger.info(f"Could not calculate TabPFN importance: {str(e)}")

        return importance_dict

    def evaluate_models(self, X_test, y_test, target_options=None, tp=None, category=None):
        """Evaluate all models including TabPFN"""
        metrics_dict = {}

        # Evaluate each base model
        for name, model in self.base_models.items():
            # TabPFN may use a feature subset (>100 cols → CatBoost selection)
            if name == "tabpfn":
                X_eval = X_test
                if hasattr(model, "selected_features_"):
                    X_eval = X_test[model.selected_features_]
                if hasattr(model, "feature_names_in_"):
                    X_eval = X_eval.reindex(columns=list(model.feature_names_in_), fill_value=0)
                y_pred = model.predict(X_eval)
                if self.task_type == "classification":
                    y_proba = model.predict_proba(X_eval)
            else:
                y_pred = model.predict(X_test)

            if self.task_type == "regression":
                n_features = X_test.shape[1]  # Default to all features

                metrics = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                    "adj_r2": adjusted_r2_score(y_test, y_pred, n_features),  # Add adjusted R²
                }
            else:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                }

                if len(np.unique(y_test)) == 2:
                    for avg in ["macro", "weighted"]:
                        metrics[f"precision_{avg}"] = precision_score(y_test, y_pred, average=avg, zero_division=0)
                        metrics[f"recall_{avg}"] = recall_score(y_test, y_pred, average=avg, zero_division=0)
                        metrics[f"f1_{avg}"] = f1_score(y_test, y_pred, average=avg, zero_division=0)

                    # Per-class metrics (fixes NaN for base models)
                    for cls_label in np.unique(y_test):
                        _cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        _k = cls_label if cls_label in _cr else str(cls_label)
                        if _k in _cr:
                            metrics[f"precision_class_{cls_label}"] = _cr[_k]["precision"]
                            metrics[f"recall_class_{cls_label}"] = _cr[_k]["recall"]
                            metrics[f"f1_class_{cls_label}"] = _cr[_k]["f1-score"]

                    # AUC + AUC-PR for binary classification
                    _proba_base = None
                    if name == "tabpfn" and self.task_type == "classification":
                        _proba_base = y_proba[:, 1]
                    elif hasattr(model, "predict_proba"):
                        try:
                            if name == "tabpfn":
                                _proba_base = model.predict_proba(X_eval)[:, 1]
                            else:
                                _proba_base = model.predict_proba(X_test)[:, 1]
                        except Exception as e:
                            logger.info(f"Could not get predict_proba for {name}: {e}")
                    if _proba_base is not None:
                        try:
                            metrics["auc"] = roc_auc_score(y_test, _proba_base)
                            _pr, _rc, _ = precision_recall_curve(y_test, _proba_base)
                            metrics["auc_pr"] = sk_auc(_rc, _pr)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                        except Exception as e:
                            logger.info(f"Could not calculate AUC for {name}: {e}")

        metrics_dict[name] = metrics

        # Evaluate meta-model
        y_pred_meta = self.predict(X_test)
        if self.task_type == "regression":
            metrics_dict["meta_model"] = {
                "mse": mean_squared_error(y_test, y_pred_meta),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_meta)),
                "mae": mean_absolute_error(y_test, y_pred_meta),
                "r2": r2_score(y_test, y_pred_meta),
                "adj_r2": adjusted_r2_score(y_test, y_pred_meta, X_test.shape[1]),  # Add adjusted R²
            }
        else:
            metrics_dict["meta_model"] = {
                "accuracy": accuracy_score(y_test, y_pred_meta),
                "precision": precision_score(y_test, y_pred_meta, average="macro", zero_division=0),
                "recall": recall_score(y_test, y_pred_meta, average="macro", zero_division=0),
                "f1": f1_score(y_test, y_pred_meta, average="macro", zero_division=0),
            }

            if len(np.unique(y_test)) == 2:
                # Add per-class metrics for meta-model
                for avg in ["macro", "weighted"]:
                    metrics_dict["meta_model"][f"precision_{avg}"] = precision_score(
                        y_test, y_pred_meta, average=avg, zero_division=0
                    )
                    metrics_dict["meta_model"][f"recall_{avg}"] = recall_score(
                        y_test, y_pred_meta, average=avg, zero_division=0
                    )
                    metrics_dict["meta_model"][f"f1_{avg}"] = f1_score(
                        y_test, y_pred_meta, average=avg, zero_division=0
                    )

                # Add per-class metrics
                class_report = classification_report(y_test, y_pred_meta, output_dict=True)
                for class_label in np.unique(y_test):
                    _key = class_label if class_label in class_report else str(class_label)
                    if _key in class_report:
                        metrics_dict["meta_model"][f"precision_class_{class_label}"] = class_report[_key]["precision"]
                        metrics_dict["meta_model"][f"recall_class_{class_label}"] = class_report[_key]["recall"]
                        metrics_dict["meta_model"][f"f1_class_{class_label}"] = class_report[_key]["f1-score"]

                try:
                    proba = self.predict_proba(X_test)[:, 1]
                    metrics_dict["meta_model"]["auc"] = roc_auc_score(y_test, proba)
                except Exception as e:
                    logger.info("Could not calculate AUC for meta-model")

        # Create comparison visualizations with more detailed metrics
        self._plot_model_comparisons(metrics_dict, target_options, tp, category)

        return metrics_dict

    # TabPFN evaluation branch
    def evaluate_models(self, X_test, y_test, target_options=None, tp=None, category=None):
        """Evaluate all models including TabPFN"""
        from sklearn.metrics import auc as sk_auc
        from sklearn.metrics import precision_recall_curve

        metrics_dict = {}

        # Evaluate each base model
        for name, model in self.base_models.items():
            # TabPFN may use a feature subset (>100 cols → CatBoost selection)
            if name == "tabpfn":
                X_eval = X_test
                if hasattr(model, "selected_features_"):
                    X_eval = X_test[model.selected_features_]
                if hasattr(model, "feature_names_in_"):
                    X_eval = X_eval.reindex(columns=list(model.feature_names_in_), fill_value=0)
                y_pred = model.predict(X_eval)
                if self.task_type == "classification":
                    y_proba = model.predict_proba(X_eval)
            else:
                y_pred = model.predict(X_test)
                logger.info(f"Model: {name}, successfully fit to y_pred predict")

            if self.task_type == "regression":
                n_features = X_test.shape[1]  # Default to all features

                metrics = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                    "adj_r2": adjusted_r2_score(y_test, y_pred, n_features),  # Add adjusted R²
                }
            else:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                }

                if len(np.unique(y_test)) == 2:
                    for avg in ["macro", "weighted"]:
                        metrics[f"precision_{avg}"] = precision_score(y_test, y_pred, average=avg, zero_division=0)
                        metrics[f"recall_{avg}"] = recall_score(y_test, y_pred, average=avg, zero_division=0)
                        metrics[f"f1_{avg}"] = f1_score(y_test, y_pred, average=avg, zero_division=0)

                    # Per-class metrics (fixes NaN for base models)
                    for cls_label in np.unique(y_test):
                        _cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        _k = cls_label if cls_label in _cr else str(cls_label)
                        if _k in _cr:
                            metrics[f"precision_class_{cls_label}"] = _cr[_k]["precision"]
                            metrics[f"recall_class_{cls_label}"] = _cr[_k]["recall"]
                            metrics[f"f1_class_{cls_label}"] = _cr[_k]["f1-score"]

                    # AUC + AUC-PR for binary classification
                    _proba_base = None
                    if name == "tabpfn" and self.task_type == "classification":
                        _proba_base = y_proba[:, 1]
                    elif hasattr(model, "predict_proba"):
                        try:
                            if name == "tabpfn":
                                _proba_base = model.predict_proba(X_eval)[:, 1]
                            else:
                                _proba_base = model.predict_proba(X_test)[:, 1]
                        except Exception as e:
                            logger.info(f"Could not get predict_proba for {name}: {e}")
                    if _proba_base is not None:
                        try:
                            metrics["auc"] = roc_auc_score(y_test, _proba_base)
                            _pr, _rc, _ = precision_recall_curve(y_test, _proba_base)
                            metrics["auc_pr"] = sk_auc(_rc, _pr)
                        except Exception as e:
                            logger.info(f"Could not calculate AUC for {name}: {e}")
                else:
                    try:
                        if hasattr(model, "predict_proba"):
                            proba_mc = model.predict_proba(X_test)
                            metrics["auc_ovr"] = roc_auc_score(y_test, proba_mc, multi_class="ovr", average="macro")
                    except Exception as e:
                        logger.info(f"Could not calculate multi-class AUC for {name}: {e}")

            metrics_dict[name] = metrics

        # Evaluate meta-model
        y_pred_meta = self.predict(X_test)
        if self.task_type == "regression":
            metrics_dict["meta_model"] = {
                "mse": mean_squared_error(y_test, y_pred_meta),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_meta)),
                "mae": mean_absolute_error(y_test, y_pred_meta),
                "r2": r2_score(y_test, y_pred_meta),
                "adj_r2": adjusted_r2_score(y_test, y_pred_meta, X_test.shape[1]),  # Add adjusted R²
            }
        else:
            metrics_dict["meta_model"] = {
                "accuracy": accuracy_score(y_test, y_pred_meta),
                "precision": precision_score(y_test, y_pred_meta, average="macro", zero_division=0),
                "recall": recall_score(y_test, y_pred_meta, average="macro", zero_division=0),
                "f1": f1_score(y_test, y_pred_meta, average="macro", zero_division=0),
            }

            if len(np.unique(y_test)) == 2:
                # Add per-class metrics for meta-model
                for avg in ["macro", "weighted"]:
                    metrics_dict["meta_model"][f"precision_{avg}"] = precision_score(
                        y_test, y_pred_meta, average=avg, zero_division=0
                    )
                    metrics_dict["meta_model"][f"recall_{avg}"] = recall_score(
                        y_test, y_pred_meta, average=avg, zero_division=0
                    )
                    metrics_dict["meta_model"][f"f1_{avg}"] = f1_score(
                        y_test, y_pred_meta, average=avg, zero_division=0
                    )

                # Add per-class metrics
                class_report = classification_report(y_test, y_pred_meta, output_dict=True)
                for class_label in np.unique(y_test):
                    _key = class_label if class_label in class_report else str(class_label)
                    if _key in class_report:
                        metrics_dict["meta_model"][f"precision_class_{class_label}"] = class_report[_key]["precision"]
                        metrics_dict["meta_model"][f"recall_class_{class_label}"] = class_report[_key]["recall"]
                        metrics_dict["meta_model"][f"f1_class_{class_label}"] = class_report[_key]["f1-score"]

                try:
                    logger.info("Fallback method to do AUC for meta model...")
                    # First check if predict_proba exists and works properly
                    if hasattr(self, "predict_proba"):
                        proba = self.predict_proba(X_test)
                        if proba is not None and proba.shape[1] >= 2:
                            # Calculate ROC-AUC
                            metrics_dict["meta_model"]["auc"] = roc_auc_score(y_test, proba[:, 1])

                            # Calculate PR-AUC (Precision-Recall AUC)
                            from sklearn.metrics import auc, precision_recall_curve

                            precision, recall, _ = precision_recall_curve(y_test, proba[:, 1])
                            metrics_dict["meta_model"]["auc_pr"] = auc(recall, precision)
                        else:
                            # Fallback: try to get probabilities directly from meta_model
                            meta_features = self._get_meta_features(X_test, is_train=False)
                            if hasattr(self.meta_model, "predict_proba"):
                                meta_proba = self.meta_model.predict_proba(meta_features)

                                # Calculate ROC-AUC
                                metrics_dict["meta_model"]["auc"] = roc_auc_score(y_test, meta_proba[:, 1])

                                # Calculate PR-AUC
                                from sklearn.metrics import precision_recall_curve

                                precision, recall, _ = precision_recall_curve(y_test, meta_proba[:, 1])
                                metrics_dict["meta_model"]["auc_pr"] = auc(recall, precision)
                    else:
                        logger.info("predict_proba not available for meta-model")
                except Exception as e:
                    logger.info(f"Could not calculate AUC metrics for meta-model: {str(e)}")

                # Threshold optimization
                # Find optimal threshold from precision-recall curve
                try:
                    _meta_proba = self.predict_proba(X_test)[:, 1]
                    _pr_curve, _rc_curve, _thresholds = precision_recall_curve(y_test, _meta_proba)
                    # F1 at each threshold
                    _f1_curve = 2 * (_pr_curve[:-1] * _rc_curve[:-1]) / (_pr_curve[:-1] + _rc_curve[:-1] + 1e-10)
                    _best_idx = np.argmax(_f1_curve)
                    _opt_thresh = float(_thresholds[_best_idx])
                    _y_pred_opt = (_meta_proba >= _opt_thresh).astype(int)

                    metrics_dict["meta_model"]["optimal_threshold"] = _opt_thresh
                    metrics_dict["meta_model"]["recall_class_1_at_opt"] = recall_score(
                        y_test, _y_pred_opt, pos_label=1, zero_division=0
                    )
                    metrics_dict["meta_model"]["precision_class_1_at_opt"] = precision_score(
                        y_test, _y_pred_opt, pos_label=1, zero_division=0
                    )
                    metrics_dict["meta_model"]["f1_class_1_at_opt"] = f1_score(
                        y_test, _y_pred_opt, pos_label=1, zero_division=0
                    )
                    metrics_dict["meta_model"]["accuracy_at_opt"] = accuracy_score(y_test, _y_pred_opt)

                    # Store arrays for plotting (attach to self for _plot_model_comparisons)
                    self._threshold_results = {
                        "proba": _meta_proba,
                        "pr_curve": _pr_curve,
                        "rc_curve": _rc_curve,
                        "thresholds": _thresholds,
                        "f1_curve": _f1_curve,
                        "opt_threshold": _opt_thresh,
                        "opt_idx": _best_idx,
                        "y_pred_default": y_pred_meta,
                        "y_pred_opt": _y_pred_opt,
                        "y_test": y_test,
                    }
                    logger.info(
                        f"\nOptimal threshold: {_opt_thresh:.4f}  "
                        f"(Recall@opt: {metrics_dict['meta_model']['recall_class_1_at_opt']:.3f}, "
                        f"Precision@opt: {metrics_dict['meta_model']['precision_class_1_at_opt']:.3f}, "
                        f"F1@opt: {metrics_dict['meta_model']['f1_class_1_at_opt']:.3f})"
                    )
                except Exception as e:
                    logger.info(f"Could not compute optimal threshold: {e}")
                    self._threshold_results = None

                if "auc" not in metrics_dict.get("meta_model", {}):
                    try:
                        logger.info("Calculating AUC metrics directly from predictions... (Alternative method #2)")
                        # Since we're having dimensionality issues with the meta-model,
                        # Direct approach using the predicted classes

                        # For binary classification, convert predictions to probability-like scores
                        if len(np.unique(y_test)) == 2:
                            # Method 1: Use decision function if available (for LogisticRegression)
                            if hasattr(self.meta_model, "decision_function"):
                                try:
                                    # Generate meta-features properly
                                    meta_features = self._get_meta_features(X_test, is_train=False)

                                    # If there's a dimension mismatch, use simpler approach
                                    if meta_features.shape[1] != self.meta_model.coef_.shape[1]:
                                        raise ValueError("Dimension mismatch")

                                    decision_scores = self.meta_model.decision_function(meta_features)
                                    metrics_dict["meta_model"]["auc"] = roc_auc_score(y_test, decision_scores)

                                    # Calculate PR-AUC
                                    from sklearn.metrics import auc, precision_recall_curve

                                    precision, recall, _ = precision_recall_curve(y_test, decision_scores)
                                    metrics_dict["meta_model"]["auc_pr"] = auc(recall, precision)
                                    logger.info("Successfully calculated AUC using decision_function")
                                except Exception as e:
                                    logger.info(f"[WARNING] {e}")
                                    # Method 2: Use a simpler approach - binary prediction as score
                                    # This isn't ideal but can serve as a fallback
                                    metrics_dict["meta_model"]["auc"] = roc_auc_score(y_test, y_pred_meta)

                                    # PR-AUC
                                    precision, recall, _ = precision_recall_curve(y_test, y_pred_meta)
                                    metrics_dict["meta_model"]["auc_pr"] = auc(recall, precision)
                                    logger.info("Used binary predictions as scores for AUC calculation")
                            else:
                                # For other models without decision_function
                                metrics_dict["meta_model"]["auc"] = roc_auc_score(y_test, y_pred_meta)

                                # PR-AUC
                                precision, recall, _ = precision_recall_curve(y_test, y_pred_meta)
                                metrics_dict["meta_model"]["auc_pr"] = auc(recall, precision)
                                logger.info("Used binary predictions as scores for AUC calculation")
                    except Exception as e:
                        logger.info(f"Could not calculate AUC metrics for meta-model: {str(e)}")
                        # Don't add the metrics if calculation fails

        # Create comparison visualizations with more detailed metrics
        self._plot_model_comparisons(metrics_dict, target_options, tp, category)

        return metrics_dict

    def _plot_model_comparisons(self, metrics_dict, target_options=None, tp=None, category=None):
        """Model comparison figures and tables (with per-class detail for classification)."""
        from IPython.display import HTML, display
        from sklearn.metrics import confusion_matrix as _cm_func

        _subtitle = f"{_resolve_target_name(target_options) if target_options else ''} | TP {tp or ''}" + (
            f" | {category}" if category else ""
        )
        is_clf = self.task_type == "classification"
        models = list(metrics_dict.keys())

        # 1a. Key metrics bar chart (r² / Adj-r² excluded for regression)
        if is_clf:
            key_metrics = ["accuracy", "precision", "recall", "f1", "auc", "auc_pr"]
            r2_metrics = []
        else:
            key_metrics = ["mse", "rmse", "mae"]
            r2_metrics = ["r2", "adj_r2"]

        comp = []
        for m in models:
            for k in key_metrics:
                if (
                    k in metrics_dict[m]
                    and metrics_dict[m][k] is not None
                    and not (isinstance(metrics_dict[m][k], float) and np.isnan(metrics_dict[m][k]))
                ):
                    comp.append({"Model": m, "Metric": k.upper().replace("_", "-"), "Value": metrics_dict[m][k]})

        # Regression: MSE/RMSE/MAE + R²/Adj-R² side by side
        if not is_clf and r2_metrics:
            comp_r2 = []
            for m in models:
                for k in r2_metrics:
                    v = metrics_dict[m].get(k)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        comp_r2.append({"Model": m, "Metric": k.upper().replace("_", "-"), "Value": v})

            if comp and comp_r2:
                cdf = pd.DataFrame(comp)
                cdf_r2 = pd.DataFrame(comp_r2)
                fig, (ax_err, ax_r2) = plt.subplots(1, 2, figsize=(max(14, len(models) * 3.5), 5))

                sns.barplot(data=cdf, x="Metric", y="Value", hue="Model", ax=ax_err, palette="Set2")
                ax_err.set_title("MSE / RMSE / MAE", fontsize=13, fontweight="bold")
                ax_err.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

                sns.barplot(data=cdf_r2, x="Metric", y="Value", hue="Model", ax=ax_r2, palette="Set2")
                ax_r2.set_title("R\u00b2 / Adjusted R\u00b2", fontsize=13, fontweight="bold")
                ax_r2.axhline(0, color="#90A4AE", lw=0.8, linestyle="--")
                ax_r2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

                fig.suptitle(f"Model Comparison -- {_subtitle}", fontsize=14, fontweight="bold", y=1.02)
                plt.tight_layout()
                plt.show()

            elif comp:
                # Fallback: only error metrics available
                cdf = pd.DataFrame(comp)
                fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
                sns.barplot(data=cdf, x="Metric", y="Value", hue="Model", ax=ax, palette="Set2")
                ax.set_title(f"Model Comparison -- {_subtitle}", fontsize=14, fontweight="bold")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
                plt.tight_layout()
                plt.show()

        # Classification: single figure
        elif comp:
            cdf = pd.DataFrame(comp)
            fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
            sns.barplot(data=cdf, x="Metric", y="Value", hue="Model", ax=ax, palette="Set2")
            ax.set_title(f"Model Comparison -- {_subtitle}", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1.05)
            import matplotlib.ticker as mticker

            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
            plt.tight_layout()
            plt.show()

        # Overview table
        overview_cols = ["accuracy", "auc", "auc_pr", "precision", "recall", "f1"]
        if is_clf:
            overview = pd.DataFrame({m: {k: metrics_dict[m].get(k) for k in overview_cols} for m in models}).T
            overview.columns = ["Accuracy", "AUC-ROC", "AUC-PR", "Precision(macro)", "Recall(macro)", "F1(macro)"]
            logger.info("\n" + "=" * 70)
            logger.info("  OVERVIEW -- Macro-averaged Metrics")
            logger.info("=" * 70)
            try:
                display(
                    overview.style.format("{:.4f}", na_rep="--")
                    .background_gradient(cmap="YlGn", subset=["AUC-ROC", "AUC-PR"], vmin=0, vmax=1)
                    .highlight_max(axis=0, color="#c6efce")
                )
            except Exception:
                display(overview)

        # Per-class table (clinical focus)
        if is_clf and len([k for k in metrics_dict[models[0]] if "class_1" in k]) > 0:
            class_cols = [
                "precision_class_0",
                "recall_class_0",
                "f1_class_0",
                "precision_class_1",
                "recall_class_1",
                "f1_class_1",
            ]
            class_data = {}
            for m in models:
                class_data[m] = {c: metrics_dict[m].get(c) for c in class_cols}
            class_df = pd.DataFrame(class_data).T
            class_df.columns = ["Prec(0)", "Recall(0)", "F1(0)", "Prec(1)", "Recall(1)", "F1(1)"]
            logger.info("\n" + "=" * 70)
            logger.info("  PER-CLASS PERFORMANCE  (Class 1 = target/positive)")
            logger.info("=" * 70)
            try:
                display(
                    class_df.style.format("{:.4f}", na_rep="--").background_gradient(
                        cmap="YlOrRd", subset=["Recall(1)", "F1(1)"], vmin=0, vmax=1
                    )
                )
            except Exception:
                display(class_df)

        # Threshold optimization results
        tr = getattr(self, "_threshold_results", None)
        if tr is not None and is_clf:
            logger.info("\n" + "=" * 70)
            logger.info("  THRESHOLD OPTIMIZATION  (Meta-model)")
            logger.info("=" * 70)
            _mm = metrics_dict.get("meta_model", {})
            thresh_compare = pd.DataFrame(
                {
                    "Default (0.50)": {
                        "Threshold": 0.50,
                        "Accuracy": _mm.get("accuracy"),
                        "Precision(1)": _mm.get("precision_class_1"),
                        "Recall(1)": _mm.get("recall_class_1"),
                        "F1(1)": _mm.get("f1_class_1"),
                    },
                    f"Optimal ({tr['opt_threshold']:.3f})": {
                        "Threshold": tr["opt_threshold"],
                        "Accuracy": _mm.get("accuracy_at_opt"),
                        "Precision(1)": _mm.get("precision_class_1_at_opt"),
                        "Recall(1)": _mm.get("recall_class_1_at_opt"),
                        "F1(1)": _mm.get("f1_class_1_at_opt"),
                    },
                }
            ).T
            try:
                display(
                    thresh_compare.style.format("{:.4f}", na_rep="--").highlight_max(
                        axis=0, subset=["Recall(1)", "F1(1)"], color="#c6efce"
                    )
                )
            except Exception:
                display(thresh_compare)

            # PR curve + confusion matrices
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # PR Curve
            ax = axes[0]
            ax.plot(tr["rc_curve"], tr["pr_curve"], color="#2196F3", lw=2, label=f"AUC-PR = {_mm.get('auc_pr', 0):.3f}")
            ax.scatter(
                tr["rc_curve"][tr["opt_idx"]],
                tr["pr_curve"][tr["opt_idx"]],
                color="red",
                s=120,
                zorder=5,
                label=f"Optimal (t={tr['opt_threshold']:.3f})",
            )
            base_rate = np.mean(tr["y_test"])
            ax.axhline(y=base_rate, color="gray", ls="--", alpha=0.5, label=f"Base rate ({base_rate:.3f})")
            ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
            ax.set_ylabel("Precision (PPV)", fontsize=11)
            ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)

            # Confusion Matrix: Default threshold
            ax = axes[1]
            cm_def = _cm_func(tr["y_test"], tr["y_pred_default"])
            sns.heatmap(
                cm_def,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
            ax.set_title("Confusion Matrix\n(threshold = 0.50)", fontsize=13, fontweight="bold")

            # Confusion Matrix: Optimal threshold
            ax = axes[2]
            cm_opt = _cm_func(tr["y_test"], tr["y_pred_opt"])
            sns.heatmap(
                cm_opt,
                annot=True,
                fmt="d",
                cmap="Greens",
                ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
            ax.set_title(f"Confusion Matrix\n(threshold = {tr['opt_threshold']:.3f})", fontsize=13, fontweight="bold")

            plt.suptitle(f"Classification Analysis -- {_subtitle}", fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.show()

        elif is_clf:
            # Fallback: just show basic table
            logger.info("\nDetailed Metrics:")
            metrics_table = pd.DataFrame([metrics_dict[m] for m in models], index=models)
            display(metrics_table)

        if not is_clf:
            logger.info("\nDetailed Metrics:")
            metrics_table = pd.DataFrame([metrics_dict[m] for m in models], index=models)
            display(metrics_table)

    def plot_feature_importance(self, X_train, y_train, target_options, tp, category=None):
        """Plot feature importance for each base model and meta-model coefficients"""
        feature_importances = self.get_feature_importance(X_train, y_train)

        # Plot feature importance for each base model
        for model_name, importance in feature_importances.items():
            # Plot ElasticNet coefficients
            if model_name == "elasticnet":
                coefficients = pd.Series(importance, index=X_train.columns)
                # Apply display names if available
                if isinstance(ctx.get("DISPLAY_NAMES"), dict) and len(ctx["DISPLAY_NAMES"]) > 0:
                    coefficients.index = coefficients.index.map(lambda x: ctx["DISPLAY_NAMES"].get(x, x))
                top_20_coefs = coefficients.sort_values(key=abs, ascending=False).head(ctx.ENSEMBLE_VIZ_CONFIG["top_n"])

                plt.figure(figsize=(8, 8))
                sns.barplot(
                    x=top_20_coefs.values, y=top_20_coefs.index, palette=ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"]()
                )
                title = f"Top {ctx.ENSEMBLE_VIZ_CONFIG['top_n']} ElasticNet Coefficients for {'Regression' if self.task_type == 'regression' else 'Classification'}"
                title += f" ({_resolve_target_name(target_options)}, Timepoint {tp}"
                title += f", Category {category})" if category else ")"
                plt.title(title, fontsize=16)
                plt.xlabel("Coefficient Value", fontsize=12)
                plt.ylabel("Features", fontsize=12)
                for i, v in enumerate(top_20_coefs.values):
                    plt.text(v, i, f" {v:.4f}", va="center", fontsize=10)
                plt.tight_layout()
                plt.show()

            elif model_name == "tabpfn":
                # Plot TabPFN importance from permutation importance
                feature_importance = pd.Series(importance, index=X_train.columns)
                top_20_features = feature_importance.sort_values(ascending=False).head(ctx.ENSEMBLE_VIZ_CONFIG["top_n"])

                plt.figure(figsize=(12, 8))
                sns.barplot(
                    x=top_20_features.values, y=top_20_features.index, palette=ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"]()
                )
                title = f"Top {ctx.ENSEMBLE_VIZ_CONFIG['top_n']} Important Features - TabPFN (Permutation Importance)"
                title += f" ({_resolve_target_name(target_options)}, Timepoint {tp}"
                title += f", Category {category})" if category else ")"
                plt.title(title, fontsize=16)
                plt.xlabel("Permutation Importance Score", fontsize=12)
                plt.ylabel("Features", fontsize=12)
                for i, v in enumerate(top_20_features.values):
                    plt.text(v, i, f" {v:.4f}", va="center", fontsize=10)
                plt.tight_layout()
                plt.show()
            else:
                # Plot tree-based model
                feature_importance = pd.Series(importance, index=X_train.columns)
                # Apply display names if available
                if isinstance(ctx.get("DISPLAY_NAMES"), dict) and len(ctx["DISPLAY_NAMES"]) > 0:
                    feature_importance.index = feature_importance.index.map(lambda x: ctx["DISPLAY_NAMES"].get(x, x))
                top_20_features = feature_importance.sort_values(ascending=False).head(ctx.ENSEMBLE_VIZ_CONFIG["top_n"])

                plt.figure(figsize=(12, 8))
                sns.barplot(
                    x=top_20_features.values, y=top_20_features.index, palette=ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"]()
                )
                title = f"Top {ctx.ENSEMBLE_VIZ_CONFIG['top_n']} Important Features - {model_name}"
                title += f" ({_resolve_target_name(target_options)}, Timepoint {tp}"
                title += f", Category {category})" if category else ")"
                plt.title(title, fontsize=16)
                plt.xlabel("Feature Importance", fontsize=12)
                plt.ylabel("Features", fontsize=12)
                plt.tight_layout()
                plt.show()

        # Plot meta-model coefficients if it's a linear model
        if hasattr(self.meta_model, "coef_"):
            meta_coef = self.meta_model.coef_
            if meta_coef.ndim > 1:  # For multi-class
                meta_coef = np.mean(np.abs(meta_coef), axis=0)

            try:
                # Create feature names for meta-model (base model names)
                meta_feature_names = list(self.base_models.keys())
                meta_coefficients = pd.Series(meta_coef, index=meta_feature_names)

                plt.figure(figsize=(8, 6))
                sns.barplot(
                    x=meta_coefficients.values,
                    y=meta_coefficients.index,
                    palette=ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"](),
                )
                title = "Meta-Model Coefficients (Model Importance in Ensemble)"
                title += f" ({_resolve_target_name(target_options)}, Timepoint {tp}"
                title += f", Category {category})" if category else ")"
                plt.title(title, fontsize=16)
                plt.xlabel("Coefficient Value", fontsize=12)
                plt.ylabel("Base Model", fontsize=12)
                for i, v in enumerate(meta_coefficients.values):
                    plt.text(v, i, f" {v:.4f}", va="center", fontsize=10)
                plt.tight_layout()
                plt.show()

            except:  # the meta-learner also takes pairwise products and their mean as inputs
                logger.info("Plotting meta-learner coefficients (base learners, pairwise products and mean).")
                # Create feature names for meta-model including interactions
                base_names = list(self.base_models.keys())

                # Create names for interaction terms
                meta_feature_names = base_names.copy()
                for i, name1 in enumerate(base_names):
                    for j, name2 in enumerate(base_names[i + 1 :], i + 1):
                        meta_feature_names.append(f"{name1} × {name2}")
                meta_feature_names.append("Weighted Average")

                # lengths match
                if len(meta_coef) > len(meta_feature_names):
                    meta_feature_names.extend([f"Feature {i}" for i in range(len(meta_feature_names), len(meta_coef))])
                meta_feature_names = meta_feature_names[: len(meta_coef)]

                meta_coefficients = pd.Series(meta_coef, index=meta_feature_names)

                # Sort by absolute value
                meta_coefficients = meta_coefficients.reindex(
                    meta_coefficients.abs().sort_values(ascending=False).index
                )

                plt.figure(figsize=(8, 6))
                sns.barplot(
                    x=meta_coefficients.values,
                    y=meta_coefficients.index,
                    palette=ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"](),
                )
                title = "Meta-Model Model Importance"
                title += f" ({_resolve_target_name(target_options)}, Timepoint {tp}"
                title += f", Category {category})" if category else ")"
                plt.title(title, fontsize=16)
                plt.xlabel("Coefficient Value", fontsize=12)
                plt.ylabel("Features", fontsize=12)
                for i, v in enumerate(meta_coefficients.values):
                    plt.text(v, i, f" {v:.4f}", va="center", fontsize=10)
                plt.tight_layout()
                plt.show()

    def explain_with_shap(self, X_train, X_test):
        """SHAP analysis with special handling for TabPFN."""
        if hasattr(X_train, "get_features"):
            X_train = pd.DataFrame(X_train.get_features(), columns=X_train.get_feature_names())
        if hasattr(X_test, "get_features"):
            X_test = pd.DataFrame(X_test.get_features(), columns=X_test.get_feature_names())

        shap_values_dict = {}

        for name, model in self.base_models.items():
            logger.info(f"\nGenerating SHAP values for base model: {name}...")

            try:
                if name == "tabpfn":
                    try:
                        # Use specialized TabPFN SHAP analysis
                        logger.info("\nPerforming specialized TabPFN SHAP analysis...")
                    except Exception as e:
                        logger.info(f"Exception: {e}, alternating..")
                        logger.info("\nPerforming specialized TabPFN SHAP analysis...")
                        # Prepare data for TabPFN SHAP
                        n_samples = min(50, len(X_test))
                        X_test_subset = (
                            X_test.iloc[:n_samples] if isinstance(X_test, pd.DataFrame) else X_test[:n_samples]
                        )
                        feature_names = X_test.columns

                        # Define prediction function
                        def predict_function_for_shap(X):
                            if hasattr(model, "predict_proba"):
                                return model.predict_proba(X)
                            return model.predict(X)

                        # Create explainer with proper background
                        background = np.ones((1, X_test_subset.shape[1])) * float("nan")
                        explainer = shap.Explainer(predict_function_for_shap, background)

                        # Calculate SHAP values
                        shap_values = explainer(X_test_subset)
                        shap_values_dict[name] = shap_values

                        # Plot TabPFN SHAP visualizations
                        if len(shap_values.shape) == 3:
                            logger.info("Computing SHAP values for the first class (index 0).")
                            shap_values = shap_values[:, :, 0]

                        # 1. Aggregate Feature Importances
                        plt.figure(figsize=(12, 8))
                        shap.plots.bar(shap_values=shap_values, show=False)
                        plt.title("Aggregate Feature Importances (TabPFN)")
                        plt.tight_layout()
                        plt.show()

                        # 2. Feature Importance Distribution
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(shap_values=shap_values, show=False)
                        plt.title("Feature Importance Distribution (TabPFN)")
                        plt.tight_layout()
                        plt.show()

                        # 3. Feature Interactions
                        most_important = np.abs(shap_values.values).mean(0).argsort()[-1]
                        logger.info(
                            f'\nAnalyzing interactions for most important feature: "{feature_names[most_important]}"'
                        )

                        if len(shap_values) > 1:
                            inds = shap.utils.potential_interactions(shap_values[:, most_important], shap_values)

                            for i in range(min(3, len(inds))):
                                plt.figure(figsize=(10, 6))
                                shap.plots.scatter(
                                    shap_values[:, most_important], color=shap_values[:, inds[i]], show=False
                                )
                                plt.title(f"Interaction: {feature_names[most_important]} vs {feature_names[inds[i]]}")
                                plt.tight_layout()
                                plt.show()

                        # Print top 50 SHAP values for TabPFN
                        logger.info(f"\nTop 50 Feature SHAP Values for {name}:")
                        logger.info("-" * 50)
                        mean_shap_values = np.abs(shap_values.values).mean(0)
                        feature_importance = pd.DataFrame({"Feature": feature_names, "SHAP Value": mean_shap_values})
                        feature_importance = feature_importance.sort_values("SHAP Value", ascending=False)
                        for idx, row in feature_importance.head(50).iterrows():
                            logger.info(f"{row['Feature']}: {row['SHAP Value']:.6f}")
                            logger.info("-" * 50)

                elif name in ["catboost", "random_forest", "xgboost"]:
                    # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    shap_values_dict[name] = shap_values

                    plt.figure(figsize=(20, 16))
                    if isinstance(shap_values, list):
                        shap.summary_plot(
                            shap_values[1] if len(shap_values) > 1 else shap_values,
                            X_test,
                            plot_type="violin",
                            show=False,
                            max_display=ctx.ENSEMBLE_VIZ_CONFIG["top_n"],
                            title=f"SHAP Values - {name}",
                        )
                    else:
                        shap.summary_plot(
                            shap_values,
                            X_test,
                            plot_type="violin",
                            show=False,
                            max_display=ctx.ENSEMBLE_VIZ_CONFIG["top_n"],
                            title=f"SHAP Values - {name}",
                        )

                    plt.title(f"SHAP Feature Importance and Impact - {name}", fontsize=24)
                    plt.xlabel("SHAP value (impact on model output)", fontsize=18)
                    plt.ylabel("Features", fontsize=18)
                    plt.tick_params(axis="both", which="major", labelsize=14)

                    cbar = plt.gcf().axes[-1]
                    cbar.set_ylabel("Feature value", fontsize=18, labelpad=20)
                    cbar.tick_params(labelsize=14)

                    plt.tight_layout()
                    plt.show()

                elif name in ["elasticnet"]:
                    # Linear model
                    explainer = shap.LinearExplainer(model, X_train)
                    shap_values = explainer.shap_values(X_test)
                    shap_values_dict[name] = shap_values

            except Exception as e:
                logger.info(f"Could not generate SHAP values for {name}: {str(e)}")

        return shap_values_dict


def create_stacking_ensemble(
    task_type: str = "regression",
    model_configs: Dict[str, Dict[str, Any]] = None,
    meta_model=None,
    cv: int = 5,
    random_state: int = 42,
) -> StackingEnsembleModel:
    """Factory function to create a stacking ensemble model"""
    return StackingEnsembleModel(
        task_type=task_type, model_configs=model_configs, meta_model=meta_model, cv=cv, random_state=random_state
    )


def create_meta_features_parallel(X, y, model, cv_splits, random_state=42):
    """Generate meta-features in parallel using joblib"""

    def _fit_predict_fold(train_idx, val_idx, X, y, model):
        if isinstance(X, pd.DataFrame):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
        else:
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]

        model_clone = _safe_clone(model)
        model_clone.fit(X_train_fold, y_train_fold)
        return val_idx, model_clone.predict(X_val_fold)

    # Create CV splits
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    splits = list(kf.split(X, y))

    # Generate predictions in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_fit_predict_fold)(train_idx, val_idx, X, y, model) for train_idx, val_idx in splits
    )

    # Combine results
    meta_features = np.zeros(len(X))
    for val_idx, preds in results:
        meta_features[val_idx] = preds

    return meta_features


def enhance_stacking_ensemble_model():
    """
    Install the final fit, predict and predict_proba methods on StackingEnsembleModel
    (meta-learner inputs: out-of-fold base-learner predictions, their pairwise products
    and their mean), plus consensus importance, tree-model feature selection and the
    TabPFN uncertainty plot. Returns the class.
    """
    # Store original methods as backup
    original_methods = {
        "_get_meta_features": StackingEnsembleModel._get_meta_features
        if hasattr(StackingEnsembleModel, "_get_meta_features")
        else None,
        "fit": StackingEnsembleModel.fit,
        "predict": StackingEnsembleModel.predict,
    }

    def _enhanced_get_meta_features(self, X, y=None, is_train=True):
        """Out-of-fold base-learner predictions; TabPFN gets the top 100 CatBoost features when there are more."""
        if hasattr(X, "get_features"):
            X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names())
        else:
            X_features = X

        if is_train:
            if self.task_type == "classification" and y is not None:
                self.n_classes_ = len(np.unique(y))
            else:
                self.n_classes_ = 2
            meta_features = np.zeros((X_features.shape[0], len(self.base_models)))
            if self.task_type == "classification":
                from sklearn.model_selection import StratifiedKFold

                kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

            # If number of features exceeds TabPFN limit, prepare selected features using CatBoost
            tabpfn_features = None
            if X_features.shape[1] > 100 and "tabpfn" in self.base_models and "catboost" in self.base_models:
                logger.info("\nSelecting top features for TabPFN (selecting top 100 for optimal speed/performance)...")
                # Fit CatBoost on full data to get feature importance
                if self.task_type == "regression":
                    temp_catboost = CatBoostRegressor(
                        iterations=100, learning_rate=0.1, verbose=False, random_state=self.random_state
                    )
                else:
                    temp_catboost = CatBoostClassifier(
                        iterations=100, learning_rate=0.1, verbose=False, random_state=self.random_state
                    )

                temp_catboost.fit(X_features, y)
                importance = pd.Series(temp_catboost.feature_importances_, index=X_features.columns)
                selected_features = importance.nlargest(100).index
                tabpfn_features = X_features[selected_features]
                self.tabpfn_feature_reduction = selected_features
                logger.info(f"Selected {len(selected_features)} top features for TabPFN")

            for i, (name, model) in enumerate(self.base_models.items()):
                logger.info(f"\nGenerating meta-features for {name}...")
                temp_meta_features = np.zeros(X_features.shape[0])

                if name == "tabpfn":
                    # Use selected features for TabPFN if available
                    X_for_tabpfn = tabpfn_features if tabpfn_features is not None else X_features

                    for train_idx, val_idx in kf.split(X_for_tabpfn, y):
                        X_train_fold = X_for_tabpfn.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X_for_tabpfn.iloc[val_idx]

                        if self.task_type == "regression":
                            model_clone = TabPFNRegressor.create_default_for_version(
                                ModelVersion.V2, device=ctx._TABPFN_DEVICE
                            )
                        else:
                            model_clone = TabPFNClassifier.create_default_for_version(
                                ModelVersion.V2, device=ctx._TABPFN_DEVICE
                            )

                        try:
                            _sw = None
                            if self.task_type == "classification" and self.class_weights is not None:
                                _sw = np.array([float(self.class_weights.get(yi, 1.0)) for yi in y_train_fold])
                            try:
                                model_clone.fit(X_train_fold, y_train_fold, sample_weight=_sw)
                            except TypeError:
                                model_clone.fit(X_train_fold, y_train_fold)
                            if self.task_type == "classification":
                                if getattr(self, "n_classes_", 2) <= 2:
                                    fold_preds = model_clone.predict_proba(X_val_fold)[:, 1]
                                else:
                                    fold_preds = model_clone.predict(X_val_fold).astype(float)
                            else:
                                fold_preds = model_clone.predict(X_val_fold)
                        except Exception as e:
                            logger.info(f"TabPFN error: {e}, using fallback predictions")
                            fold_preds = np.full(len(X_val_fold), np.mean(y_train_fold))

                        temp_meta_features[val_idx] = fold_preds

                    # Store selected features for prediction time
                    if tabpfn_features is not None:
                        model.selected_features_ = selected_features

                else:
                    # Other models use all features
                    for train_idx, val_idx in kf.split(X_features, y):
                        X_train_fold = X_features.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X_features.iloc[val_idx]

                        model_clone = _safe_clone(model)
                        if self.task_type == "classification":
                            if name == "catboost":
                                model_clone.set_params(class_weights=self.class_weights)
                            elif name == "random_forest":
                                model_clone.set_params(class_weight=self.class_weights)
                            elif name == "xgboost" and len(np.unique(y)) == 2:
                                _y_spw = np.asarray(y).ravel()
                                _cls = np.unique(_y_spw)
                                model_clone.set_params(
                                    scale_pos_weight=float(
                                        np.sqrt(np.sum(_y_spw == _cls[0]) / np.sum(_y_spw == _cls[1]))
                                    )
                                )

                        try:
                            model_clone.fit(X_train_fold, y_train_fold)
                        except Exception as e:
                            logger.info(f"[WARNING] {e}")
                            model_clone.fit(X_train_fold, y_train_fold.iloc[:, 0])

                        # Use predict_proba for classification (consistent scale with TabPFN)
                        if self.task_type == "classification" and hasattr(model_clone, "predict_proba"):
                            if getattr(self, "n_classes_", 2) <= 2:
                                fold_preds = model_clone.predict_proba(X_val_fold)[:, 1]
                            else:
                                fold_preds = model_clone.predict(X_val_fold).astype(float)
                        else:
                            fold_preds = model_clone.predict(X_val_fold)
                        temp_meta_features[val_idx] = fold_preds

                meta_features[:, i] = temp_meta_features

                # Fit on full data
                if name == "tabpfn":
                    # Use selected features for final TabPFN model
                    X_for_tabpfn = tabpfn_features if tabpfn_features is not None else X_features
                    if self.task_type == "regression":
                        self.base_models[name] = TabPFNRegressor.create_default_for_version(
                            ModelVersion.V2, device=ctx._TABPFN_DEVICE
                        )
                    else:
                        self.base_models[name] = TabPFNClassifier.create_default_for_version(
                            ModelVersion.V2, device=ctx._TABPFN_DEVICE
                        )
                    _sw_full = None
                    if self.task_type == "classification" and self.class_weights is not None:
                        _sw_full = np.array([float(self.class_weights.get(yi, 1.0)) for yi in y])
                    try:
                        try:
                            self.base_models[name].fit(X_for_tabpfn, y, sample_weight=_sw_full)
                        except TypeError:
                            self.base_models[name].fit(X_for_tabpfn, y)
                    except Exception as _tabpfn_err:
                        logger.info(
                            f"[WARNING] TabPFN full-data fit failed: {_tabpfn_err}. {'GPU OOM -- subsampling to ≤9999' if ctx._TABPFN_DEVICE == 'cuda' else 'CPU limit -- subsampling to ≤999'}."
                        )
                        if ctx._TABPFN_DEVICE == "cuda":
                            import torch

                            torch.cuda.empty_cache()
                        _max_n = min(9999 if ctx._TABPFN_DEVICE == "cuda" else 999, len(X_for_tabpfn))
                        _sub_idx = np.random.choice(len(X_for_tabpfn), _max_n, replace=False)
                        try:
                            self.base_models[name].fit(X_for_tabpfn.iloc[_sub_idx], y.iloc[_sub_idx])
                        except Exception as _e2:
                            logger.info(
                                f"[WARNING] TabPFN subsampled fit also failed: {_e2}. Predictions may be degraded."
                            )
                    if tabpfn_features is not None:
                        self.base_models[name].selected_features_ = selected_features  # transfer to new instance
                else:
                    try:
                        model.fit(X_features, y)
                    except Exception as e:
                        logger.info(f"[WARNING] {e}")
                        model.fit(X_features, y.iloc[:, 0])

        else:
            # Held-out predictions
            meta_features = []
            for name, model in self.base_models.items():
                if name == "tabpfn":
                    # Use selected features for prediction if available
                    if hasattr(model, "selected_features_"):
                        X_for_tabpfn = X_features[model.selected_features_]
                    else:
                        X_for_tabpfn = X_features
                    # Align columns to match what TabPFN saw during fit
                    if hasattr(model, "feature_names_in_"):
                        X_for_tabpfn = X_for_tabpfn.reindex(columns=list(model.feature_names_in_), fill_value=0)

                    try:
                        if self.task_type == "classification":
                            if getattr(self, "n_classes_", 2) <= 2:
                                preds = model.predict_proba(X_for_tabpfn)[:, 1]
                            else:
                                preds = model.predict(X_for_tabpfn).astype(float)
                        else:
                            preds = model.predict(X_for_tabpfn)
                    except Exception as e:
                        logger.info(f"TabPFN prediction error: {e}, using fallback")
                        _fb = 0.5 if self.task_type == "classification" else 0.0
                        preds = np.full(X_features.shape[0], _fb)
                else:
                    if self.task_type == "classification" and hasattr(model, "predict_proba"):
                        if getattr(self, "n_classes_", 2) <= 2:
                            preds = model.predict_proba(X_features)[:, 1]
                        else:
                            preds = model.predict(X_features).astype(float)
                    else:
                        preds = model.predict(X_features)
                meta_features.append(preds)

            meta_features = np.column_stack(meta_features)

        return meta_features

    def _add_meta_interactions(self, meta_features):
        """Add interaction features between base model predictions"""
        if meta_features is None:
            raise ValueError("meta_features cannot be None")

        n_models = meta_features.shape[1]
        interactions = []

        # Add pairwise interactions
        for i in range(n_models):
            for j in range(i + 1, n_models):
                interactions.append(meta_features[:, i] * meta_features[:, j])

        # Add weighted averages
        weighted_avg = np.average(meta_features, axis=1, weights=np.ones(n_models) / n_models)
        interactions.append(weighted_avg)

        # Combine original features with interactions
        meta_features_enhanced = np.column_stack([meta_features] + interactions)

        return meta_features_enhanced

    def _enhanced_fit(self, X, y, categorical_features=None):
        """Fit the base learners, then the meta-learner on their out-of-fold predictions, pairwise products and mean."""
        if hasattr(X, "get_features"):
            X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names())
        else:
            X_features = X

        # Validate input data
        if isinstance(X_features, pd.DataFrame):
            has_null = X_features.isnull().any().any()
        else:
            has_null = np.isnan(X_features).any()

        if has_null:
            warnings.warn("Input contains null values. Handling with median imputation.")
            imputer = SimpleImputer(strategy="median")
            if isinstance(X_features, pd.DataFrame):
                X_features = pd.DataFrame(
                    imputer.fit_transform(X_features), columns=X_features.columns, index=X_features.index
                )
            else:
                X_features = imputer.fit_transform(X_features)

        # Set class weights for classification
        if self.task_type == "classification":
            self.set_class_weights(y)

        # Generate meta-features
        meta_features = self._get_meta_features(X_features, y, is_train=True)
        meta_features_with_interactions = self._add_meta_interactions(meta_features)

        # Fit meta-model with cross-validation
        if isinstance(self.meta_model, (LogisticRegression, ElasticNet)):
            param_grid = {
                "C": np.logspace(-4, 4, 20) if isinstance(self.meta_model, LogisticRegression) else None,
                "alpha": np.logspace(-4, 4, 20) if isinstance(self.meta_model, ElasticNet) else None,
                "l1_ratio": np.linspace(0, 1, 11) if isinstance(self.meta_model, ElasticNet) else None,
            }
            param_grid = {k: v for k, v in param_grid.items() if v is not None}

            cv = RandomizedSearchCV(self.meta_model, param_grid, n_iter=20, cv=5, random_state=self.random_state)
            cv.fit(meta_features_with_interactions, y)
            self.meta_model = cv.best_estimator_
        else:
            self.meta_model.fit(meta_features_with_interactions, y)

        return self

    def _enhanced_predict(self, X):
        """Predict from the base learners' predictions, their pairwise products and mean."""
        if hasattr(X, "get_features"):
            X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names())
        else:
            X_features = X

        # Generate meta-features
        meta_features = self._get_meta_features(X_features, is_train=False)
        meta_features_with_interactions = self._add_meta_interactions(meta_features)

        # Get predictions from meta-model
        predictions = self.meta_model.predict(meta_features_with_interactions)

        if self.task_type == "classification":
            return np.round(predictions).astype(int)
        return predictions

    def get_tree_model_features(self, X_train, top_k=50):
        """Get important features from tree-based models"""
        all_important_features = set()

        # First get importance from tree-based models only
        for name, model in self.base_models.items():
            if name in ["catboost", "random_forest", "xgboost"]:
                if hasattr(model, "feature_importances_"):
                    # Get feature importance
                    importances = pd.Series(model.feature_importances_, index=X_train.columns)
                    # Get top k features
                    top_features = importances.nlargest(top_k).index
                    all_important_features.update(top_features)

        return list(all_important_features)

    def plot_tabpfn_uncertainty(self, X_train, y_train, X_test=None):
        """Plot uncertainty visualization for TabPFN predictions"""
        if "tabpfn" not in self.base_models:
            return None

        try:
            # Generate test points if not provided
            if X_test is None:
                X_test = np.linspace(np.min(X_train), np.max(X_train), 200).reshape(-1, 1)

            # Get TabPFN predictions with quantiles
            model = self.base_models["tabpfn"]
            preds = model.predict(X_test, output_type="full")

            fig = plt.figure(figsize=(12, 6))

            # Plot original data
            ax = fig.add_subplot(121)
            ax.scatter(X_train, y_train, label="Training data", color="red", s=10)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_xlim(min(X_train) - 1, max(X_train) + 1)
            ax.set_ylim(np.min(y_train) - 1, np.max(y_train) + 1)
            ax.set_title("Training Data")
            ax.legend()

            # Plot uncertainty
            ax = fig.add_subplot(122)
            all_quantiles = preds["quantiles"]
            y = np.array(all_quantiles)

            # Calculate the maximum and minimum values
            y_max = np.max(y, axis=0)
            y_min = np.min(y, axis=0)

            # Calculate quantile bin widths
            quantile_bin_widths = np.diff(y, axis=0)

            # Normalize bin widths
            per_x_normalized_bin_widths = quantile_bin_widths / (y_max - y_min)

            # Plotting rectangles for uncertainty
            num_bins, num_data_points = per_x_normalized_bin_widths.shape
            rect_width = (X_test[1] - X_test[0]).squeeze()

            for i in range(num_data_points):
                for j in range(num_bins):
                    rect = plt.Rectangle(
                        xy=(X_test[i][0] - rect_width / 2, y[j, i]),
                        width=rect_width,
                        height=quantile_bin_widths[j, i],
                        facecolor=plt.cm.viridis(per_x_normalized_bin_widths[j, i] * 5),
                        edgecolor="none",
                    )
                    ax.add_patch(rect)

            ax.scatter(X_train, y_train, label="Training data", color="red", s=10)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("TabPFN Prediction Uncertainty")
            plt.tight_layout()
            plt.show()

            return fig

        except Exception as e:
            logger.info(f"Error plotting TabPFN uncertainty: {str(e)}")
            return None

    def analyze_feature_consensus(self, X_train, y_train, target_options, tp, category=None, top_n=None):
        """Weighted consensus feature importance across the base learners.

        Each learner's feature importances are scaled to their maximum and multiplied by the
        learner's weight, the absolute meta-learner coefficient on its out-of-fold prediction
        as a share of the total over learners; the products are summed over learners.
        """
        if top_n is None:
            top_n = ctx.ENSEMBLE_VIZ_CONFIG["top_n"]
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from sklearn.inspection import permutation_importance

        # Get meta-model coefficients for weights
        if hasattr(self.meta_model, "coef_"):
            meta_coef = self.meta_model.coef_
            if meta_coef.ndim > 1:
                meta_coef = np.mean(np.abs(meta_coef), axis=0)

            base_names = list(self.base_models.keys())
            model_weights = {}
            for i, name in enumerate(base_names):
                if i < len(meta_coef):
                    model_weights[name] = abs(meta_coef[i])
                else:
                    model_weights[name] = 0
        else:
            model_weights = {name: 1 / len(self.base_models) for name in self.base_models.keys()}

        weight_sum = sum(model_weights.values())
        model_weights = {k: v / weight_sum for k, v in model_weights.items()}

        # Get feature importance from each model
        feature_ranks = {}
        feature_importances = {}
        all_features = set()

        # First get important features from tree models
        important_features = self.get_consensus_features(X_train)
        logger.info(f"Number of consensus features from tree models: {len(important_features)}")

        # Store original column order
        original_columns = X_train.columns.tolist()

        for name, model in self.base_models.items():
            try:
                if name in ["svm", "neural_net", "tabpfn"]:
                    logger.info(f"Calculating permutation importance for {name} on consensus features...")
                    if name == "tabpfn":
                        from sklearn.inspection import permutation_importance

                        try:
                            logger.info("\nGenerating TabPFN uncertainty visualization...")
                            uncertainty_fig = self.plot_tabpfn_uncertainty(X_train, y_train)
                            if uncertainty_fig:
                                logger.info("TabPFN uncertainty visualization generated successfully")
                        except Exception as e:
                            logger.info(f"Warning: {e}")
                            continue

                    if hasattr(model, "named_steps"):
                        # Preserve feature order for SVM
                        X_subset = X_train[important_features].reindex(columns=original_columns)

                        # Before subsetting
                        logger.info("%s %s", "NaN values in X_train:", X_train.isnull().sum().sum())

                        # After subsetting
                        logger.info("%s %s", "NaN values in X_subset:", X_subset.isnull().sum().sum())

                        # Check column continuity
                        logger.info("%s %s", "X_subset columns:", X_subset.columns)
                        logger.info("%s %s", "Important features:", important_features)

                        # Handle NaN values before scaling if they exist
                        if X_subset.isnull().any().any():  # Check if any NaN exists
                            logger.info(f"Found NaN values in X_train subset crunch: {X_subset.isnull().sum().sum()}")
                            imputer = SimpleImputer(strategy="median")
                            X_subset = pd.DataFrame(
                                imputer.fit_transform(X_subset), columns=X_subset.columns, index=X_subset.index
                            )
                            logger.info(f"Imputed {X_subset.isnull().sum().sum()} NaN values")
                        else:
                            logger.info("No NaN values found for X_train subset")

                        scaler = model.named_steps["scaler"]
                        X_scaled = scaler.transform(X_subset)
                        actual_model = (
                            model.named_steps["svr" if self.task_type == "regression" else "svc"]
                            if name == "svm"
                            else model.named_steps["mlp"]
                        )

                        if name == "svm" and hasattr(actual_model, "kernel") and actual_model.kernel == "linear":
                            logger.info("Using linear kernel coefficients for SVM importance")
                            full_importance = np.zeros(len(original_columns))
                            for idx, feat in enumerate(important_features):
                                full_importance[original_columns.index(feat)] = np.abs(actual_model.coef_[0][idx])
                            importances = full_importance
                        else:
                            from sklearn.inspection import permutation_importance

                            result = permutation_importance(
                                actual_model, X_scaled, y_train, n_repeats=10, random_state=42, n_jobs=-1
                            )

                            full_importance = np.zeros(len(original_columns))
                            for idx, feat in enumerate(important_features):
                                full_importance[original_columns.index(feat)] = result.importances_mean[idx]
                            importances = full_importance
                    logger.info(f"Successfully calculated importance for {name}")

                elif hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_)
                else:
                    continue

                # Create Series with feature names
                model_importance = pd.Series(importances, index=original_columns)
                feature_importances[name] = model_importance

                # Create rank Series (1 for most important)
                ranks = model_importance.rank(ascending=False)
                feature_ranks[name] = ranks

                all_features.update(model_importance.index)

            except Exception as e:
                logger.info(f"Could not calculate importance for {name}: {str(e)}")
                logger.info(f"Model type: {type(model)}")
                if hasattr(model, "named_steps"):
                    logger.info(f"Pipeline steps: {model.named_steps.keys()}")

        # Calculate consensus scores
        consensus_scores = {}
        for feature in all_features:
            weighted_score = 0
            feature_details = {"scores": [], "ranks": [], "weighted_scores": []}

            for model_name, weight in model_weights.items():
                if model_name in feature_importances:
                    importance = feature_importances[model_name].get(feature, 0)
                    rank = feature_ranks[model_name].get(feature, len(all_features))

                    norm_importance = importance / feature_importances[model_name].max()
                    weighted_importance = norm_importance * weight

                    feature_details["scores"].append(norm_importance)
                    feature_details["ranks"].append(rank)
                    feature_details["weighted_scores"].append(weighted_importance)
                    weighted_score += weighted_importance

            consensus_scores[feature] = {
                "weighted_score": weighted_score,
                "avg_rank": np.mean(feature_details["ranks"]),
                "rank_std": np.std(feature_details["ranks"]),
                "score_std": np.std(feature_details["scores"]),
                "details": feature_details,
            }

        # Convert to DataFrame and sort
        consensus_df = pd.DataFrame.from_dict(consensus_scores, orient="index")

        # Delta-Level Redundancy Filter
        # If a variable and its delta (T2−T0) both survive consensus,
        # keep only the higher-scored one to prevent construct inflation.
        # Runs BEFORE display-name mapping (index still has delta_ prefix).
        _raw_idx = consensus_df.index.tolist()
        _delta_drop = set()
        for _feat in _raw_idx:
            if _feat.startswith("delta_"):
                _base = _feat.replace("delta_", "", 1)
                if _base in _raw_idx and _base not in _delta_drop:
                    _s_delta = consensus_df.loc[_feat, "weighted_score"]
                    _s_base = consensus_df.loc[_base, "weighted_score"]
                    _loser = _base if _s_delta >= _s_base else _feat
                    _delta_drop.add(_loser)
                    _winner = _base if _loser != _base else _feat
                    logger.info(
                        f"  [DELTA-REDUNDANCY] Dropping '{_loser}' "
                        f"(score={consensus_df.loc[_loser, 'weighted_score']:.4f}) "
                        f"-- kept '{_winner}' "
                        f"(score={consensus_df.loc[_winner, 'weighted_score']:.4f})"
                    )
        if _delta_drop:
            consensus_df = consensus_df.drop(index=list(_delta_drop))
            logger.info(f"  [DELTA-REDUNDANCY] Removed {len(_delta_drop)} redundant level/delta feature(s)")

        # Apply display names if available
        _DN = ctx.get("DISPLAY_NAMES")
        if isinstance(_DN, dict) and len(_DN) > 0:
            consensus_df.index = consensus_df.index.map(lambda x: _DN.get(x, x))
            # Deduplicate display-name collisions (keep highest weighted_score)
            if consensus_df.index.duplicated().any():
                _dup_names = consensus_df.index[consensus_df.index.duplicated(keep=False)].unique().tolist()
                logger.info(f"  Warning: DISPLAY_NAMES collision on: {_dup_names} -- keeping highest-scored entry")
                consensus_df = consensus_df[~consensus_df.index.duplicated(keep="first")]

        top_features = consensus_df.sort_values("weighted_score", ascending=False).head(top_n)

        # Print publication-ready text and create visualizations
        logger.info("\nFeature Importance Analysis for Publication")
        logger.info("=" * 50)
        logger.info("\nBased on our stacking ensemble model analysis")
        logger.info(f"Time point: {tp}")
        if category:
            logger.info(f"Category: {category}")
        logger.info("\nModel Contributions to Meta-learner:")
        for model, weight in model_weights.items():
            logger.info(f"- {model}: {weight:.2%}")

        logger.info("\nTop Features by Weighted Consensus:")
        for i, (feature, row) in enumerate(top_features.iterrows(), 1):
            consistency = "High" if row["score_std"] < 0.1 else "Moderate" if row["score_std"] < 0.2 else "Variable"
            logger.info(f"\n{i}. {feature}")
            logger.info(f"   Weighted Importance: {row['weighted_score']:.4f}")
            logger.info(f"   Average Rank: {row['avg_rank']:.1f}")
            logger.info(f"   Consistency: {consistency}")

        # Visualizations: use ENSEMBLE_VIZ_CONFIG (built from the setup ensemble_* settings)
        _viz2 = ctx.ENSEMBLE_VIZ_CONFIG
        _palette2 = _viz2["palette_fn"]()
        # Normalize: palette_fn may return a string name → convert to color list
        if isinstance(_palette2, str):
            import seaborn as _sns_pal

            _palette2 = list(_sns_pal.color_palette(_palette2, n_colors=_viz2["top_n"]))
        _scores2 = top_features["weighted_score"].values
        _labels2 = top_features.index.tolist()
        _consist2 = ["H" if s < 0.1 else "M" if s < 0.2 else "V" for s in top_features["score_std"].values]

        fig2, ax2 = plt.subplots(figsize=(_viz2["fig_width"], _viz2["fig_height"]))
        bars2 = ax2.barh(range(len(_labels2)), _scores2, color=_palette2[: len(_labels2)])
        ax2.set_yticks(range(len(_labels2)))
        ax2.set_yticklabels(_labels2, fontsize=_viz2["ylabel_fs"])
        ax2.invert_yaxis()
        ax2.set_xlabel("Weighted Ensemble Consensus Score", fontsize=9)
        ax2.set_title(
            f"{_resolve_target_name(target_options)}, Timepoint {tp}" + (f" -- {category}" if category else ""),
            fontsize=14,
            fontweight="bold",
        )

        # Score + consistency annotations at bar ends
        _x_max2 = max(_scores2) * 1.22
        for idx, (bar, sc, con) in enumerate(zip(bars2, _scores2, _consist2)):
            ax2.text(sc + _x_max2 * 0.015, idx, f"{sc:.3f}", va="center", ha="left", fontsize=8, color="#333333")
        ax2.set_xlim(0, _x_max2)

        ax2.set_axisbelow(True)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        if _viz2["save"]:
            _fn2 = f"{_viz2['save_prefix']}_consensus_plot"
            plt.savefig(f"{_fn2}.png", dpi=_viz2["dpi"], bbox_inches="tight")
        plt.show()

        # heatmap with all models
        # Rename feature_importances index to match display names
        if isinstance(_DN, dict) and len(_DN) > 0:
            feature_importances = {
                model: imp.rename(index=lambda x: _DN.get(x, x)).pipe(lambda s: s[~s.index.duplicated(keep="first")])
                for model, imp in feature_importances.items()
            }

        feature_comparison = pd.DataFrame(
            {
                model: importances.reindex(top_features.index, fill_value=0)
                for model, importances in feature_importances.items()
            }
        )
        feature_comparison = feature_comparison.div(feature_comparison.max())

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            feature_comparison,
            cmap=_viz2["heatmap_cmap"],
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Normalized Importance"},
            center=0.5,
        )
        plt.title("Feature Importance Comparison Across All Models")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        return consensus_df, model_weights, feature_comparison

    # predict_proba (must match _enhanced_fit dimensionality)
    def _enhanced_predict_proba(self, X):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification")
        X_features = pd.DataFrame(X.get_features(), columns=X.get_feature_names()) if hasattr(X, "get_features") else X
        meta = self._get_meta_features(X_features, is_train=False)
        meta = self._add_meta_interactions(meta)
        return self.meta_model.predict_proba(meta)

    # Attach the methods to the class
    StackingEnsembleModel._get_meta_features = _enhanced_get_meta_features
    StackingEnsembleModel._add_meta_interactions = _add_meta_interactions
    StackingEnsembleModel.fit = _enhanced_fit
    StackingEnsembleModel.predict = _enhanced_predict
    StackingEnsembleModel.predict_proba = _enhanced_predict_proba

    # Keep references to the attached methods
    StackingEnsembleModel._enhanced_methods = {
        "_get_meta_features": _enhanced_get_meta_features,
        "_add_meta_interactions": _add_meta_interactions,
        "fit": _enhanced_fit,
        "predict": _enhanced_predict,
        "predict_proba": _enhanced_predict_proba,
    }

    # Store original methods for potential rollback
    StackingEnsembleModel._original_methods = original_methods

    StackingEnsembleModel.analyze_feature_consensus = analyze_feature_consensus

    StackingEnsembleModel.get_tree_model_features = get_tree_model_features
    StackingEnsembleModel.plot_tabpfn_uncertainty = plot_tabpfn_uncertainty

    return StackingEnsembleModel


# Attach the methods (meta-learner interactions, consensus, tree features, uncertainty)
enhance_stacking_ensemble_model()
logger.info("[Ensemble] Meta-learner interactions and consensus methods attached to StackingEnsembleModel")

import warnings
from collections import Counter, defaultdict
from itertools import combinations

import catboost
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from mapie.classification import SplitConformalClassifier
from mapie.regression import SplitConformalRegressor
from optuna_integration import CatBoostPruningCallback
from pygam import GAM, s
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from smogn import smoter
from tqdm import tqdm

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion  # For TabPFN v2 (no auth required)

    TABPFN_AVAILABLE = True
except ImportError:  # optional dependency: `pip install .[tabpfn]`
    TabPFNClassifier = TabPFNRegressor = ModelVersion = None
    TABPFN_AVAILABLE = False
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ignore DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Set seaborn defaults for consistent figure styling.
sns.set(style="whitegrid")
sns.color_palette("cubehelix", as_cmap=True)


def run() -> None:
    """Run this stage (notebook cell 5)."""
    ctx._TABPFN_DEVICE = "cuda" if (_torch is not None and _torch.cuda.is_available()) else "cpu"
    if ctx._TABPFN_DEVICE == "cuda":
        _gpu_name = _torch.cuda.get_device_name(0)
        _gpu_mem = round(getattr(_torch.cuda.get_device_properties(0), "total_memory", 0) / 1024**3, 1)
        logger.info(f"[TabPFN] GPU detected: {_gpu_name} ({_gpu_mem} GB) -- using CUDA")
    else:
        logger.info("[TabPFN] No GPU detected -- using CPU (≤1000 samples only)")

    ctx.PIPELINE_MODEL_CONFIGS = {
        "catboost": {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "border_count": 128,
            "thread_count": -1,
            "verbose": False,
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 10,
        },
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6,
        },
        "tabpfn": {
            "device": ctx._TABPFN_DEVICE,
            "version": "v2",
        },
    }

    # Convenience accessors for code that instantiates models directly
    # (e.g. cross-category refit, stacking auditor base_models dicts).
    ctx._RF_PARAMS = ctx.PIPELINE_MODEL_CONFIGS["random_forest"]
    ctx._XGB_PARAMS = ctx.PIPELINE_MODEL_CONFIGS["xgboost"]
    ctx._CB_PARAMS = ctx.PIPELINE_MODEL_CONFIGS["catboost"]

    ctx.ENSEMBLE_VIZ_CONFIG = {
        "palette_fn": _resolve_ensemble_palette,
        "heatmap_cmap": ctx.ensemble_heatmap_cmap,
        "top_n": ctx.ensemble_top_n,
        "fig_width": ensemble_fig_width,
        "fig_height": ensemble_fig_height,
        "ylabel_fs": ctx.ensemble_ylabel_fontsize,
        "dpi": ensemble_dpi,
        "save": ctx.ensemble_save_plots,
        "save_prefix": ctx.ensemble_save_prefix,
    }
    logger.info(
        f"[Ensemble Viz] palette={ctx.ensemble_barplot_palette}, color={ctx.ensemble_barplot_color}, "
        f"heatmap={ctx.ensemble_heatmap_cmap}, top_n={ctx.ensemble_top_n}"
    )

    ctx.category_results = defaultdict(dict)
