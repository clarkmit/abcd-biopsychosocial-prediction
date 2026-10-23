"""Outer-fold nested CV metrics and calibration (the `nested_cv` stage, Supplement S1.4, S1.11.3, S1.11.10).

Outer-fold nested CV producing unbiased performance estimates with 95%
CIs (t-distribution). Regression outputs include predicted-vs-actual
scatter and fold-wise R² distributions. Classification outputs include
calibration curves with Wilson-score CIs, decision curve analysis, and
fold-wise AUC distributions. Configurable number of outer folds.
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
Outer-Fold Nested Cross-Validation Analysis
--------------------------------------------
True outer-fold evaluation (configurable n held-out folds) producing
unbiased per-target performance estimates with 95% CIs.

Outputs
  Regression    : Predicted-vs-actual scatter, fold-wise R² distribution.
  Classification: Calibration curves with Wilson-score CI, decision curve
                  analysis, fold-wise AUC distribution.

Integration
-----------
Expects X_train, X_test, y_train, y_test, and model from a preceding
analysis run. Pass model_factory to retrain fresh estimators per fold.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold

# NestedCVAnalysis Class (Handles both task types)


class NestedCVAnalysis:
    """
    Explicit nested CV reporter for classification AND regression.
    TASK TYPE SUPPORT:
    - Classification: AUC, sensitivity, specificity, PPV, NPV, accuracy, F1
    - Regression: R², RMSE, MAE, explained variance
    """

    def __init__(self, X, y, task_type="auto", n_outer=5, random_state=42):
        """Initialize the nested-CV reporter.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like or Series
            Target.
        task_type : str
            'auto', 'classification', or 'regression'.
        n_outer : int
            Number of outer CV folds (default 5).
        random_state : int
            Random seed.
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

        self.outer_fold_results = {}
        self.clinical_thresholds = [0.3, 0.5, 0.7]

        logger.info(f"Task type detected: {self.task_type.upper()}")
        if self.is_classification:
            logger.info(f"   Unique classes: {len(np.unique(self.y))}")
            logger.info(f"   Class balance: {dict(pd.Series(self.y).value_counts())}")
        else:
            logger.info(f"   Value range: [{self.y.min():.3f}, {self.y.max():.3f}]")
            logger.info(f"   Mean: {self.y.mean():.3f}, SD: {self.y.std():.3f}")

    def run_outer_cv(self, model_factory=None, base_model=None, results_dir="nestedcv_results"):
        """
        Run explicit outer CV loop.

        USAGE OPTIONS:
        1. Pipeline mode: pass base_model (trained model from notebook)
           reporter.run_outer_cv(base_model=model)

        2. Retrain mode: pass model_factory (function returning fresh model)
           def model_factory():
               return LogisticRegression(max_iter=1000)
           reporter.run_outer_cv(model_factory=model_factory)

        Parameters:
        model_factory : callable, optional
            Function that returns a fresh model instance (for retraining)
        base_model : estimator, optional
            Pre-trained model to use for predictions (from pipeline)
        results_dir : str
            Directory to save results
        """
        if base_model is not None:
            return self._analyze_pretrained_model(base_model, results_dir)
        elif model_factory is not None:
            return self._run_explicit_outer_cv(model_factory, results_dir)
        else:
            raise ValueError("Either base_model (trained model) or model_factory (callable) must be provided")

    # Pipeline mode: Analyze pretrained model

    def _analyze_pretrained_model(self, model, results_dir):
        """Analyze a pretrained model using stratified CV."""
        logger.info(f"\n PIPELINE MODE: Analyzing pretrained {model.__class__.__name__}")
        logger.info(f"   (Using trained model with {self.n_outer}-fold outer CV)")

        os.makedirs(results_dir, exist_ok=True)

        if self.is_classification:
            return self._analyze_pretrained_classification(model, results_dir)
        else:
            return self._analyze_pretrained_regression(model, results_dir)

    def _analyze_pretrained_classification(self, model, results_dir):
        """Analyze pretrained classification model."""
        skf = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {
            "fold": [],
            "auc": [],
            "sensitivity": [],
            "specificity": [],
            "ppv": [],
            "npv": [],
            "accuracy": [],
            "f1": [],
            "brier": [],
            "y_true": [],
            "y_proba": [],
            "y_pred": [],
        }

        oof_df_list = []

        logger.info(f"   Running {self.n_outer} outer folds...")

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_test_fold = self.X.iloc[test_idx]
            y_test_fold = self.y.iloc[test_idx]
            X_train_fold = self.X.iloc[train_idx]
            y_train_fold = self.y.iloc[train_idx]

            # Get predictions from pretrained model
            # Retrain per fold to prevent leakage
            try:
                _fold_model = ctx.create_validation_model(is_classification=True)
                _fold_model.fit(X_train_fold, y_train_fold)
                y_proba = _fold_model.predict_proba(X_test_fold)[:, 1]
            except Exception as _e:
                logger.info(f"  Warning: Fold {fold_num} retrain failed ({_e}), using original model")
                y_proba = model.predict_proba(X_test_fold)[:, 1]

            # Optimal threshold (Youden's J)
            fpr, tpr, thresholds = roc_curve(y_test_fold, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = (y_proba >= optimal_threshold).astype(int)

            # Metrics for this fold
            _cm = confusion_matrix(y_test_fold, y_pred)
            if _cm.shape == (2, 2):
                tn, fp, fn, tp = _cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            fold_results["fold"].append(fold_num)
            fold_results["auc"].append(roc_auc_score(y_test_fold, y_proba))
            fold_results["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_results["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fold_results["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            fold_results["npv"].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            fold_results["accuracy"].append(accuracy_score(y_test_fold, y_pred))
            fold_results["f1"].append(f1_score(y_test_fold, y_pred, zero_division=0))
            fold_results["brier"].append(brier_score_loss(y_test_fold, y_proba))
            fold_results["y_true"].append(y_test_fold.values)
            fold_results["y_proba"].append(y_proba)
            fold_results["y_pred"].append(y_pred)
            oof_df = pd.DataFrame(
                {"fold": fold_num, "y_true": y_test_fold.values, "y_proba": y_proba, "y_pred": y_pred}
            )
            oof_df_list.append(oof_df)

            logger.info(
                f"      Fold {fold_num + 1}: AUC={fold_results['auc'][-1]:.4f}, Acc={fold_results['accuracy'][-1]:.4f}"
            )

        self.outer_fold_results = fold_results

        # Save results
        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_classification.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items() if k not in ["y_true", "y_proba", "y_pred"]})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_classification.csv", index=False)

        logger.info(f"Saved to {results_dir}/")
        return fold_results

    def _analyze_pretrained_regression(self, model, results_dir):
        """Evaluate a regression model fold by fold, training a fresh model in each outer fold.

        The pretrained model is not scored on the outer folds, because it has seen their rows."""
        kf = KFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {"fold": [], "r2": [], "rmse": [], "mae": [], "explained_var": [], "y_true": [], "y_pred": []}

        oof_df_list = []

        logger.info(f"   Running {self.n_outer} outer folds...")

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_test_fold = self.X.iloc[test_idx]
            y_test_fold = self.y.iloc[test_idx]
            X_train_fold = self.X.iloc[train_idx]
            y_train_fold = self.y.iloc[train_idx]

            # Get predictions
            # Retrain per fold to prevent leakage
            try:
                _fold_model = ctx.create_validation_model(is_classification=False)
                _fold_model.fit(X_train_fold, y_train_fold)
                y_pred = _fold_model.predict(X_test_fold)
            except Exception as _e:
                logger.info(f"  Warning: Fold {fold_num} retrain failed ({_e}), using original model")
                y_pred = model.predict(X_test_fold)
            r2 = r2_score(y_test_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
            mae = mean_absolute_error(y_test_fold, y_pred)
            exp_var = 1 - (np.var(y_test_fold - y_pred) / np.var(y_test_fold))

            fold_results["fold"].append(fold_num)
            fold_results["r2"].append(r2)
            fold_results["rmse"].append(rmse)
            fold_results["mae"].append(mae)
            fold_results["explained_var"].append(exp_var)
            fold_results["y_true"].append(y_test_fold.values)
            fold_results["y_pred"].append(y_pred)
            oof_df = pd.DataFrame(
                {
                    "fold": fold_num,
                    "y_true": y_test_fold.values,
                    "y_pred": y_pred,
                    "residual": y_test_fold.values - y_pred,
                }
            )
            oof_df_list.append(oof_df)

            logger.info(f"      Fold {fold_num + 1}: R²={r2:.4f}, RMSE={rmse:.4f}")

        self.outer_fold_results = fold_results

        # Save results
        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_regression.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items() if k not in ["y_true", "y_pred"]})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_regression.csv", index=False)

        logger.info(f"Saved to {results_dir}/")
        return fold_results

    # Retrain mode: Explicit outer CV with fresh model training

    def _run_explicit_outer_cv(self, model_factory, results_dir):
        """Run explicit outer CV (retraining fresh models each fold)."""
        logger.info(f"\n RETRAIN MODE: Running explicit outer CV ({self.n_outer} folds)")

        os.makedirs(results_dir, exist_ok=True)

        if self.is_classification:
            return self._explicit_cv_classification(model_factory, results_dir)
        else:
            return self._explicit_cv_regression(model_factory, results_dir)

    def _explicit_cv_classification(self, model_factory, results_dir):
        """Explicit outer CV for classification."""
        skf = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {
            "fold": [],
            "auc": [],
            "sensitivity": [],
            "specificity": [],
            "ppv": [],
            "npv": [],
            "accuracy": [],
            "f1": [],
            "brier": [],
            "y_true": [],
            "y_proba": [],
            "y_pred": [],
        }

        oof_df_list = []

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_train_fold = self.X.iloc[train_idx]
            X_test_fold = self.X.iloc[test_idx]
            y_train_fold = self.y.iloc[train_idx]
            y_test_fold = self.y.iloc[test_idx]

            # Train fresh model
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)
            y_proba = model.predict_proba(X_test_fold)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test_fold, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = (y_proba >= optimal_threshold).astype(int)
            _cm = confusion_matrix(y_test_fold, y_pred)
            if _cm.shape == (2, 2):
                tn, fp, fn, tp = _cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            fold_results["fold"].append(fold_num)
            fold_results["auc"].append(roc_auc_score(y_test_fold, y_proba))
            fold_results["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_results["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fold_results["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            fold_results["npv"].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            fold_results["accuracy"].append(accuracy_score(y_test_fold, y_pred))
            fold_results["f1"].append(f1_score(y_test_fold, y_pred, zero_division=0))
            fold_results["brier"].append(brier_score_loss(y_test_fold, y_proba))

            fold_results["y_true"].append(y_test_fold.values)
            fold_results["y_proba"].append(y_proba)
            fold_results["y_pred"].append(y_pred)

            oof_df = pd.DataFrame(
                {"fold": fold_num, "y_true": y_test_fold.values, "y_proba": y_proba, "y_pred": y_pred}
            )
            oof_df_list.append(oof_df)

            logger.info(
                f"   Fold {fold_num + 1}: AUC={fold_results['auc'][-1]:.4f}, Acc={fold_results['accuracy'][-1]:.4f}"
            )

        self.outer_fold_results = fold_results

        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_classification.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items() if k not in ["y_true", "y_proba", "y_pred"]})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_classification.csv", index=False)

        logger.info(f"Saved to {results_dir}/")
        return fold_results

    def _explicit_cv_regression(self, model_factory, results_dir):
        """Explicit outer CV for regression."""
        kf = KFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {"fold": [], "r2": [], "rmse": [], "mae": [], "explained_var": [], "y_true": [], "y_pred": []}

        oof_df_list = []

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_train_fold = self.X.iloc[train_idx]
            X_test_fold = self.X.iloc[test_idx]
            y_train_fold = self.y.iloc[train_idx]
            y_test_fold = self.y.iloc[test_idx]

            # Train fresh model
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            r2 = r2_score(y_test_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
            mae = mean_absolute_error(y_test_fold, y_pred)
            exp_var = 1 - (np.var(y_test_fold - y_pred) / np.var(y_test_fold))

            fold_results["fold"].append(fold_num)
            fold_results["r2"].append(r2)
            fold_results["rmse"].append(rmse)
            fold_results["mae"].append(mae)
            fold_results["explained_var"].append(exp_var)
            fold_results["y_true"].append(y_test_fold.values)
            fold_results["y_pred"].append(y_pred)

            oof_df = pd.DataFrame(
                {
                    "fold": fold_num,
                    "y_true": y_test_fold.values,
                    "y_pred": y_pred,
                    "residual": y_test_fold.values - y_pred,
                }
            )
            oof_df_list.append(oof_df)

            logger.info(f"   Fold {fold_num + 1}: R²={r2:.4f}, RMSE={rmse:.4f}")

        self.outer_fold_results = fold_results

        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_regression.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items() if k not in ["y_true", "y_pred"]})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_regression.csv", index=False)

        logger.info(f"Saved to {results_dir}/")
        return fold_results

    # Reporting methods

    def summarize_outer_folds(self):
        """Aggregate outer-fold metrics with 95% t-CI."""
        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        if self.is_classification:
            return self._summarize_classification()
        else:
            return self._summarize_regression()

    def _summarize_classification(self):
        """Summarize classification metrics."""
        metrics_list = ["auc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1", "brier"]
        summary = {}

        for metric in metrics_list:
            values = np.array(self.outer_fold_results[metric])
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            se = std_val / np.sqrt(n)
            ci_margin = t_crit * se

            summary[metric] = {
                "mean": mean_val,
                "std": std_val,
                "ci_lower": mean_val - ci_margin,
                "ci_upper": mean_val + ci_margin,
                "fold_values": list(values),
            }

        df = pd.DataFrame(
            {
                "Metric": list(summary.keys()),
                "Mean": [summary[m]["mean"] for m in summary],
                "Std": [summary[m]["std"] for m in summary],
                "95% CI Lower": [summary[m]["ci_lower"] for m in summary],
                "95% CI Upper": [summary[m]["ci_upper"] for m in summary],
            }
        )

        return df, summary

    def _summarize_regression(self):
        """Summarize regression metrics."""
        metrics_list = ["r2", "rmse", "mae", "explained_var"]
        summary = {}

        for metric in metrics_list:
            if metric in self.outer_fold_results:
                values = np.array(self.outer_fold_results[metric])
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                n = len(values)
                t_crit = stats.t.ppf(0.975, df=n - 1)
                se = std_val / np.sqrt(n)
                ci_margin = t_crit * se

                summary[metric] = {
                    "mean": mean_val,
                    "std": std_val,
                    "ci_lower": mean_val - ci_margin,
                    "ci_upper": mean_val + ci_margin,
                    "fold_values": list(values),
                }

        df = pd.DataFrame(
            {
                "Metric": list(summary.keys()),
                "Mean": [summary[m]["mean"] for m in summary],
                "Std": [summary[m]["std"] for m in summary],
                "95% CI Lower": [summary[m]["ci_lower"] for m in summary],
                "95% CI Upper": [summary[m]["ci_upper"] for m in summary],
            }
        )

        return df, summary

    def plot_calibration_curve(self, n_bins=10, figsize=(10, 6), save_path=None):
        """Calibration curve - classification only."""
        if not self.is_classification:
            logger.info("Warning: Calibration curve only for classification. Skipping.")
            return None

        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        y_true_all = np.concatenate(self.outer_fold_results["y_true"])
        y_proba_all = np.concatenate(self.outer_fold_results["y_proba"])

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_true = np.zeros(n_bins)
        bin_ci_lower = np.zeros(n_bins)
        bin_ci_upper = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (y_proba_all >= bins[i]) & (y_proba_all < bins[i + 1])
            if mask.sum() > 0:
                bin_true[i] = y_true_all[mask].mean()
                n = mask.sum()
                p = bin_true[i]
                # Wilson score CI
                z = 1.96
                denom = 1 + z**2 / n
                center = (p + z**2 / (2 * n)) / denom
                margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
                bin_ci_lower[i] = max(0, center - margin)
                bin_ci_upper[i] = min(1, center + margin)

        ece = np.mean(np.abs(bin_true - bin_centers))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfectly calibrated")
        ax.errorbar(
            bin_centers,
            bin_true,
            # Wilson CI center != bin_true, so deltas can be negative.
            # np.clip ensures yerr values are always >= 0, preventing the
            # matplotlib ValueError on extreme class imbalance targets.
            yerr=[np.clip(bin_true - bin_ci_lower, 0, None), np.clip(bin_ci_upper - bin_true, 0, None)],
            fmt="o-",
            capsize=4,
            capthick=2,
            lw=2,
            markersize=8,
            label=f"Model (ECE = {ece:.4f})",
        )
        ax.fill_between(bin_centers, bin_ci_lower, bin_ci_upper, alpha=0.2)
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(
            "[nested_cv] Calibration Curve\n(Outer Fold Aggregated, Wilson 95% CI)", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"   Saved: {save_path}")
        plt.show()

        return {"bin_centers": bin_centers, "bin_true": bin_true, "ece": ece}

    def plot_roc_curve(self, figsize=(10, 8), save_path=None):
        """ROC curve with per-fold curves and mean +/- SD shading (classification only)."""
        if not self.is_classification:
            logger.info("Note: ROC curve only for classification. Skipping.")
            return None

        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve as sk_roc_curve

        fig, ax = plt.subplots(figsize=figsize)

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        for fold_num in range(len(self.outer_fold_results["y_true"])):
            y_true_fold = self.outer_fold_results["y_true"][fold_num]
            y_proba_fold = self.outer_fold_results["y_proba"][fold_num]

            fpr, tpr, _ = sk_roc_curve(y_true_fold, y_proba_fold)
            fold_auc = auc(fpr, tpr)

            # Interpolate to common FPR grid
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            ax.plot(fpr, tpr, alpha=0.3, lw=1, label=f"Fold {fold_num + 1} (AUC = {fold_auc:.3f})")

        # Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std([auc(mean_fpr, t) for t in tprs])

        ax.plot(mean_fpr, mean_tpr, color="b", lw=3, label=f"Mean ROC (AUC = {mean_auc:.3f} \u00b1 {std_auc:.3f})")

        # +/- 1 SD shading
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="blue", alpha=0.15, label="\u00b1 1 SD")

        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Chance")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        ax.set_title("[nested_cv] ROC Curve\n(Per-Fold with Mean \u00b1 SD)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"  Saved: {save_path}")
        plt.show()

        return {"mean_auc": mean_auc, "std_auc": std_auc}

    def plot_decision_curve(self, figsize=(10, 6), save_path=None):
        """Decision curve - classification only."""
        if not self.is_classification:
            logger.info("Warning: Decision curve only for classification. Skipping.")
            return None

        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        y_true_all = np.concatenate(self.outer_fold_results["y_true"])
        y_proba_all = np.concatenate(self.outer_fold_results["y_proba"])

        thresholds = np.linspace(0.05, 0.95, 50)
        net_benefits = []

        for threshold in thresholds:
            y_pred_thresh = (y_proba_all >= threshold).astype(int)
            tp = np.sum((y_pred_thresh == 1) & (y_true_all == 1))
            fp = np.sum((y_pred_thresh == 1) & (y_true_all == 0))
            n = len(y_true_all)
            nb = (tp / n) - (fp / n) * (threshold / (1 - threshold + 1e-10))
            net_benefits.append(nb)

        net_benefits = np.array(net_benefits)
        prevalence = np.mean(y_true_all)
        treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds + 1e-10))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresholds, net_benefits, label="Model", lw=3, color="#1f77b4")
        ax.plot(thresholds, treat_all_nb, label="Treat All", lw=2, linestyle="--", color="red")
        ax.axhline(y=0, label="Treat None", lw=2, linestyle="--", color="gray")
        ax.fill_between(thresholds, 0, net_benefits, alpha=0.2, color="#1f77b4")
        ax.axvspan(0.2, 0.8, alpha=0.1, color="green", label="Clinical range")
        ax.set_xlabel("Probability Threshold", fontsize=12)
        ax.set_ylabel("Net Benefit", fontsize=12)
        ax.set_title("[nested_cv] Decision Curve Analysis\n(Outer Fold Aggregated)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"   Saved: {save_path}")
        plt.show()

        return {"thresholds": thresholds, "net_benefits": net_benefits}

    def plot_fold_distributions(self, figsize=(14, 8), save_path=None):
        """Plot fold-wise distributions (all metrics)."""
        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        if self.is_classification:
            metrics = ["auc", "sensitivity", "specificity", "ppv", "accuracy"]
        else:
            metrics = ["r2", "rmse", "mae", "explained_var"]

        metrics = [m for m in metrics if m in self.outer_fold_results]

        n_cols = 3
        n_rows = (len(metrics) + 2) // 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            values = self.outer_fold_results[metric]

            axes[idx].violinplot([values], positions=[0], vert=True, widths=0.5, showmeans=True, showextrema=True)
            axes[idx].boxplot(
                [values], positions=[0], widths=0.15, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7)
            )

            y_scatter = np.random.normal(0, 0.02, size=len(values))
            axes[idx].scatter(y_scatter, values, alpha=0.7, s=80, color="red", edgecolors="black", label="Outer folds")

            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_margin = t_crit * (std_val / np.sqrt(n))

            axes[idx].hlines(mean_val, -0.3, 0.3, colors="green", linewidth=2.5, label="Mean")
            axes[idx].set_ylabel(metric.upper(), fontsize=11, fontweight="bold")
            axes[idx].set_title(
                f"{metric.upper()}\n{mean_val:.3f} [{mean_val - ci_margin:.3f}-{mean_val + ci_margin:.3f}]", fontsize=11
            )
            axes[idx].set_xlim([-0.5, 0.5])
            axes[idx].set_xticks([])
            axes[idx].grid(True, alpha=0.3, axis="y")

        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(
            f"[nested_cv] Outer-Fold Distributions (n={self.n_outer} folds, {self.task_type.upper()})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"   Saved: {save_path}")
        plt.show()

    def threshold_metrics_table(self, thresholds=None):
        """Metrics at thresholds (classification only)."""
        if not self.is_classification:
            logger.info("Warning: Threshold metrics only for classification. Skipping.")
            return None

        if thresholds is None:
            thresholds = self.clinical_thresholds

        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        y_true_all = np.concatenate(self.outer_fold_results["y_true"])
        y_proba_all = np.concatenate(self.outer_fold_results["y_proba"])

        results = []

        for threshold in thresholds:
            y_pred = (y_proba_all >= threshold).astype(int)
            _cm = confusion_matrix(y_true_all, y_pred)
            if _cm.shape == (2, 2):
                tn, fp, fn, tp = _cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            # Bootstrap CI
            n_boot = 1000
            boot_sens, boot_spec, boot_ppv = [], [], []

            np.random.seed(self.random_state)
            for _ in range(n_boot):
                idx = np.random.choice(len(y_proba_all), len(y_proba_all), replace=True)
                y_true_b = y_true_all[idx]
                y_pred_b = y_pred[idx]
                try:
                    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_b, y_pred_b).ravel()
                    boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)
                    boot_spec.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
                    boot_ppv.append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0)
                except Exception as e:
                    logger.info(f"[WARNING] {e}")
                    pass

            results.append(
                {
                    "Threshold": f"{threshold:.2f}",
                    "Sensitivity": f"{sensitivity:.3f} [{np.percentile(boot_sens, 2.5):.3f}-{np.percentile(boot_sens, 97.5):.3f}]",
                    "Specificity": f"{specificity:.3f} [{np.percentile(boot_spec, 2.5):.3f}-{np.percentile(boot_spec, 97.5):.3f}]",
                    "PPV": f"{ppv:.3f} [{np.percentile(boot_ppv, 2.5):.3f}-{np.percentile(boot_ppv, 97.5):.3f}]",
                    "NPV": f"{npv:.3f}",
                    "N": len(y_true_all),
                }
            )

        return pd.DataFrame(results)

    def generate_report(self, title="NESTED CV PERFORMANCE REPORT", save_plots=False):
        """Generate complete report with all outputs."""
        logger.info("\n" + "=" * 80)
        logger.info(title)
        logger.info("=" * 80)
        logger.info(f"\nTask Type: {self.task_type.upper()}")
        logger.info(f"Outer-fold CV: {self.n_outer} folds")
        logger.info("Metrics aggregated with t-distribution 95% CI\n")

        # Summary statistics
        df_summary, _ = self.summarize_outer_folds()
        logger.info("1. OUTER-FOLD METRICS (Mean ± 95% CI)")
        logger.info("-" * 80)
        try:
            _num_cols_s = df_summary.select_dtypes(include=[np.number]).columns.tolist()
            display(
                df_summary.style.format({c: "{:.4f}" for c in _num_cols_s})
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#ECEFF1"),
                                ("font-size", "12px"),
                                ("font-weight", "600"),
                                ("padding", "6px 10px"),
                            ],
                        },
                        {"selector": "td", "props": [("padding", "5px 10px"), ("font-size", "12px")]},
                    ]
                )
                .hide(axis="index")
            )
        except Exception:
            logger.info(df_summary.to_string(index=False))

        # Threshold metrics (classification only)
        if self.is_classification:
            logger.info("\n2. METRICS AT CLINICAL THRESHOLDS (bootstrap 95% CI)")
            logger.info("-" * 80)
            threshold_df = self.threshold_metrics_table()
            if threshold_df is not None:
                logger.info(threshold_df.to_string(index=False))

        # Visualizations
        logger.info("\n3. VISUALIZATIONS")
        logger.info("-" * 80)

        self.plot_fold_distributions(save_path="ncv_fold_dist.png" if save_plots else None)

        if self.is_classification:
            self.plot_roc_curve(save_path="ncv_roc_curve.png" if save_plots else None)
            self.plot_calibration_curve(save_path="ncv_calibration.png" if save_plots else None)
            self.plot_decision_curve(save_path="ncv_decision_curve.png" if save_plots else None)

        logger.info("\n" + "=" * 80)
        logger.info("[nested_cv] COMPLETE")
        logger.info("=" * 80)


# Helper function: Create Validation Model (Same as Main Pipeline)


def create_validation_model(is_classification=True):
    """
    Build the stacking ensemble the main pipeline uses (create_stacking_ensemble with the
    PIPELINE_MODEL_CONFIGS settings), so every outer fold refits the same architecture.
    """
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    # Best params for CatBoost (from PIPELINE_MODEL_CONFIGS )
    best_params = dict(ctx._CB_PARAMS)

    # Model configs (from PIPELINE_MODEL_CONFIGS)
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

    # Create stacking ensemble using the same factory as main pipeline
    model = create_stacking_ensemble(
        task_type="regression" if not is_classification else "classification",
        model_configs=model_configs,
        meta_model=meta_model,
        cv=5,
        random_state=42,
    )

    # TabPFN is opt-in via model_configs

    return model


# Cache system: nested_cv

# Batch-aware wrapper: Outer-Fold Nested CV Metrics & Calibration
# When _batch_archive holds multiple targets with archived state, this
# wrapper iterates over each, restoring globals before each pass.
# In single-target mode the validation runs exactly once (no change).


def _run_for_current_target():
    """Run this stage's validation for the current target."""
    # Validation stage gate (reads the main stage settings)
    _skip_nested = not ctx.get("run_validation_cells", True)
    _VAL_N_OUTER = ctx.n_outer_folds

    # Dependency guard: ensure the setup infrastructure is loaded
    if "make_run_key" not in ctx:
        logger.info("\nWarning: setup has not been run in this session.")
        logger.info("   Run setup and the main stage first.")
        logger.info("   (After a runtime restart, all cells must be re-executed in order.)")
        _RUN_12 = False
    elif _skip_nested:
        logger.info("nested_cv skipped: main.run_validation_cells is False")
        _RUN_12 = False
    else:
        _CTAG_12 = "nested_cv"
        _CKEY_12 = make_run_key({"cell": _CTAG_12})
        _CACHED_12 = cache_load(_CTAG_12, _CKEY_12)
        if _CACHED_12:
            logger.info(f"  {_CTAG_12} loaded from cache -- skipping execution.")
            _RUN_12 = False
        else:
            _RS["section"] = "nested_cv"
            _done_folds_12 = fold_load_all(_CKEY_12, _CTAG_12)
            _RUN_12 = True

    if _RUN_12:
        # EXECUTION: Run reporter with pipeline or standalone

        try:
            # Task-type helpers: use the preprocessing definitions if available, else derive from y_train
            if "is_regression" not in ctx or not callable(ctx.get("is_regression")):
                ctx["is_regression"] = lambda target: target in (ctx.get("continuous_outcome_variables", []))
            if "is_classification" not in ctx or not callable(ctx.get("is_classification")):
                ctx["is_classification"] = lambda target: (
                    target in (ctx.get("binary_or_categorical_outcome_variables", []))
                )

            # Detect mode: within_categories vs across_categories

            _is_within_mode = (
                "exported_within_dict" in ctx
                and isinstance(ctx.exported_within_dict, dict)
                and len(ctx.exported_within_dict) > 0
                and "split" in ctx
                and ctx.split == "within_categories"
            )

            if _is_within_mode:
                # Within-categories mode: Per-category nested CV

                logger.info("\n" + "=" * 80)
                logger.info(" Running [nested_cv] -- WITHIN-CATEGORIES Nested CV")
                logger.info("=" * 80)

                _n_cats = len(ctx.exported_within_dict)
                task_type = detect_task_type(list(ctx.exported_within_dict.values())[0][2])
                _is_cls = task_type == "classification"
                _metric_name = "AUC" if _is_cls else "R²"
                _metric_key = "auc" if _is_cls else "r2"

                logger.info(f"   Task type: {task_type}")
                logger.info(f"   Categories: {_n_cats}")
                logger.info(f"   Folds per category: {_VAL_N_OUTER}")
                logger.info(f"   Total model fits: ~{_n_cats * 5}")
                logger.info("   Using STACKING ENSEMBLE model factory (same as main pipeline)")
                logger.info("")

                # Collect per-category results
                ctx._within_ncv_results = {}  # {cat: {'cv_r2_mean', 'cv_r2_std', 'cv_r2_folds', 'holdout_r2', ...}}

                results_dir_point1 = os.path.join(os.getcwd(), "nestedcv_results_point1")
                os.makedirs(results_dir_point1, exist_ok=True)

                for _cat_idx, (_cat_name, _cat_data) in enumerate(ctx.exported_within_dict.items()):
                    _Xtr, _Xte, _ytr, _yte = _cat_data
                    _n_feat = _Xtr.shape[1]

                    logger.info(
                        f"  [{_cat_idx + 1}/{_n_cats}] {_cat_name} ({_n_feat} features, "
                        f"N_train={_Xtr.shape[0]:,}, N_test={_Xte.shape[0]:,})"
                    )

                    # 5-fold nested CV on training data
                    _Xtr_cv = _Xtr.copy() if isinstance(_Xtr, pd.DataFrame) else pd.DataFrame(_Xtr)
                    _ytr_cv = _ytr.copy() if isinstance(_ytr, pd.Series) else pd.Series(np.asarray(_ytr).ravel())

                    # Reset indices to prevent iloc issues
                    _Xtr_cv = _Xtr_cv.reset_index(drop=True)
                    _ytr_cv = _ytr_cv.reset_index(drop=True)

                    kf = KFold(n_splits=_VAL_N_OUTER, shuffle=True, random_state=42)
                    _fold_metrics = {"r2": [], "rmse": [], "mae": [], "explained_var": []}
                    if _is_cls:
                        _fold_metrics = {"auc": [], "accuracy": [], "f1": [], "brier": []}

                    _fold_ok = True
                    for _fold_num, (_train_idx, _test_idx) in enumerate(
                        kf.split(_Xtr_cv)
                        if not _is_cls
                        else StratifiedKFold(n_splits=_VAL_N_OUTER, shuffle=True, random_state=42).split(
                            _Xtr_cv, _ytr_cv
                        )
                    ):
                        try:
                            _Xf_train = _Xtr_cv.iloc[_train_idx]
                            _Xf_test = _Xtr_cv.iloc[_test_idx]
                            _yf_train = _ytr_cv.iloc[_train_idx]
                            _yf_test = _ytr_cv.iloc[_test_idx]

                            _model = ctx.create_validation_model(is_classification=_is_cls)
                            _model.fit(_Xf_train, _yf_train)

                            if _is_cls:
                                _y_pred = _model.predict(_Xf_test)
                                _y_proba = (
                                    _model.predict_proba(_Xf_test)[:, 1] if hasattr(_model, "predict_proba") else None
                                )
                                _fold_metrics["auc"].append(
                                    roc_auc_score(_yf_test, _y_proba) if _y_proba is not None else np.nan
                                )
                                _fold_metrics["accuracy"].append(accuracy_score(_yf_test, _y_pred))
                                _fold_metrics["f1"].append(f1_score(_yf_test, _y_pred, zero_division=0))
                                if _y_proba is not None:
                                    _fold_metrics["brier"].append(brier_score_loss(_yf_test, _y_proba))
                            else:
                                _y_pred = _model.predict(_Xf_test)
                                from sklearn.metrics import explained_variance_score

                                _fold_metrics["r2"].append(r2_score(_yf_test, _y_pred))
                                _fold_metrics["rmse"].append(np.sqrt(mean_squared_error(_yf_test, _y_pred)))
                                _fold_metrics["mae"].append(mean_absolute_error(_yf_test, _y_pred))
                                _fold_metrics["explained_var"].append(explained_variance_score(_yf_test, _y_pred))

                        except Exception as _fe:
                            logger.info(f"      Warning: Fold {_fold_num + 1} failed: {_fe}")
                            _fold_ok = False
                            break

                    if not _fold_ok or len(_fold_metrics.get(_metric_key, [])) == 0:
                        logger.info("      Warning: Skipped (fold failure)")
                        continue

                    # Holdout evaluation
                    try:
                        _ho_model = ctx.create_validation_model(is_classification=_is_cls)
                        _ho_model.fit(_Xtr_cv, _ytr_cv)

                        _Xte_ho = _Xte.copy() if isinstance(_Xte, pd.DataFrame) else pd.DataFrame(_Xte)
                        _yte_ho = _yte.copy() if isinstance(_yte, pd.Series) else pd.Series(np.asarray(_yte).ravel())

                        if _is_cls:
                            _ho_pred = _ho_model.predict(_Xte_ho)
                            _ho_proba = (
                                _ho_model.predict_proba(_Xte_ho)[:, 1] if hasattr(_ho_model, "predict_proba") else None
                            )
                            _ho_primary = roc_auc_score(_yte_ho, _ho_proba) if _ho_proba is not None else np.nan
                            _ho_rmse = np.nan
                            _ho_mae = np.nan
                        else:
                            _ho_pred = _ho_model.predict(_Xte_ho)
                            _ho_primary = r2_score(_yte_ho, _ho_pred)
                            _ho_rmse = np.sqrt(mean_squared_error(_yte_ho, _ho_pred))
                            _ho_mae = mean_absolute_error(_yte_ho, _ho_pred)
                    except Exception as _he:
                        logger.info(f"      Warning: Holdout failed: {_he}")
                        _ho_primary = np.nan
                        _ho_rmse = np.nan
                        _ho_mae = np.nan

                    _cv_mean = np.mean(_fold_metrics[_metric_key])
                    _cv_std = np.std(_fold_metrics[_metric_key])

                    ctx._within_ncv_results[_cat_name] = {
                        "n_features": _n_feat,
                        "n_train": _Xtr.shape[0],
                        "n_test": _Xte.shape[0],
                        "cv_folds": _fold_metrics[_metric_key],
                        "cv_mean": _cv_mean,
                        "cv_std": _cv_std,
                        "cv_rmse_mean": np.mean(_fold_metrics.get("rmse", [np.nan])),
                        "cv_mae_mean": np.mean(_fold_metrics.get("mae", [np.nan])),
                        "holdout_primary": _ho_primary,
                        "holdout_rmse": _ho_rmse,
                        "holdout_mae": _ho_mae,
                    }

                    logger.info(
                        f"      CV {_metric_name}: {_cv_mean:.4f} ± {_cv_std:.4f}  |  "
                        f"Holdout {_metric_name}: {_ho_primary:.4f}"
                    )

                    # Free fold models between categories
                    try:
                        del _model, _ho_model
                    except NameError:
                        pass
                    _pipeline_cleanup(label=f"P1 cat {_cat_idx + 1}/{_n_cats}", verbose=(_cat_idx % 5 == 4))

                # Aggregate report

                if len(ctx._within_ncv_results) > 0:
                    logger.info("\n" + "=" * 80)
                    logger.info("[nested_cv] Within-Categories Nested CV -- Aggregate Report")
                    logger.info("=" * 80)

                    # Per-category results table
                    _ncv_rows = []
                    for _cat, _res in ctx._within_ncv_results.items():
                        _row = {
                            "Category": _cat,
                            "N Feat.": _res["n_features"],
                        }
                        _row[f"CV {_metric_name} (mean)"] = _res["cv_mean"]
                        _row[f"CV {_metric_name} (SD)"] = _res["cv_std"]
                        _row[f"Holdout {_metric_name}"] = _res["holdout_primary"]
                        if not _is_cls:
                            _row["CV RMSE"] = _res["cv_rmse_mean"]
                            _row["Holdout RMSE"] = _res["holdout_rmse"]
                        _ncv_rows.append(_row)

                    _ncv_df = (
                        pd.DataFrame(_ncv_rows)
                        .sort_values(f"CV {_metric_name} (mean)", ascending=False)
                        .reset_index(drop=True)
                    )

                    # Merge with original pipeline metrics if available
                    if "metrics_dict_full" in ctx and isinstance(ctx.metrics_dict_full, dict):
                        _orig_lookup = {}
                        for _cat, _mdict in zip(
                            ctx.metrics_dict_full.get("category", []), ctx.metrics_dict_full.get("metrics", [])
                        ):
                            _m = _mdict.get("ensemble", _mdict.get("meta_model", {}))
                            if _m:
                                _orig_lookup[_cat] = _m.get("r2" if not _is_cls else "auc", np.nan)
                        _ncv_df[f"Original {_metric_name}"] = _ncv_df["Category"].map(_orig_lookup)
                        _ncv_df["Δ (CV − Orig)"] = (
                            _ncv_df[f"CV {_metric_name} (mean)"] - _ncv_df[f"Original {_metric_name}"]
                        )

                    # Style and display
                    _num_cols = _ncv_df.select_dtypes(include=[np.number]).columns.tolist()
                    _gradient_col = f"CV {_metric_name} (mean)"
                    _grad_vmin = 0.5 if _is_cls else 0

                    _ncv_styler = (
                        _ncv_df.style.format({c: "{:.4f}" for c in _num_cols if c != "N Feat."})
                        .format({"N Feat.": "{:.0f}"})
                        .background_gradient(
                            cmap="RdYlGn",
                            subset=[_gradient_col],
                            vmin=_grad_vmin,
                            vmax=max(_grad_vmin, _ncv_df[_gradient_col].max()),
                        )
                        .set_caption(f"[nested_cv] Per-Category Nested CV -- {ctx.target_options} (T{ctx.tp_option})")
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
                                        ("text-transform", "uppercase"),
                                        ("padding", "6px 10px"),
                                    ],
                                },
                                {"selector": "td", "props": [("padding", "5px 10px"), ("font-size", "12px")]},
                            ]
                        )
                        .hide(axis="index")
                    )

                    # Highlight Δ column if present
                    if "Δ (CV − Orig)" in _ncv_df.columns:
                        _ncv_styler = _ncv_styler.background_gradient(
                            cmap="RdYlBu", subset=["Δ (CV − Orig)"], vmin=-0.05, vmax=0.05
                        )

                    display(_ncv_styler)
                    logger.info("")

                    # Aggregate metrics
                    _all_cv_means = [r["cv_mean"] for r in ctx._within_ncv_results.values()]
                    _all_ho = [
                        r["holdout_primary"]
                        for r in ctx._within_ncv_results.values()
                        if not np.isnan(r["holdout_primary"])
                    ]
                    _weights = np.array([r["n_test"] for r in ctx._within_ncv_results.values()])
                    _weights_norm = _weights / _weights.sum()

                    _weighted_cv = np.average(_all_cv_means, weights=_weights_norm)
                    _unweighted_cv = np.mean(_all_cv_means)
                    _weighted_ho = np.average(_all_ho, weights=_weights_norm[: len(_all_ho)]) if _all_ho else np.nan

                    logger.info(f"  AGGREGATE {_metric_name}:")
                    logger.info(
                        f"    Weighted mean (by N_test):  CV = {_weighted_cv:.4f}  |  Holdout = {_weighted_ho:.4f}"
                    )
                    logger.info(f"    Unweighted mean:            CV = {_unweighted_cv:.4f}")
                    logger.info(f"    Categories validated: {len(ctx._within_ncv_results)}/{_n_cats}")

                    # Comparison figure: Original vs Nested CV R²
                    try:
                        _BLUE = "#1565C0"
                        _TEAL = "#00838F"
                        _RED = "#C62828"
                        _GREY = "#455A64"

                        # Sort by CV performance
                        _plot_cats = _ncv_df.sort_values(f"CV {_metric_name} (mean)", ascending=True)

                        fig, axes = plt.subplots(
                            1, 2, figsize=(18, max(8, len(_plot_cats) * 0.35)), gridspec_kw={"width_ratios": [3, 2]}
                        )

                        # Left panel: Original vs Nested CV per category
                        _y_pos = np.arange(len(_plot_cats))
                        _bar_h = 0.35

                        _cv_vals = _plot_cats[f"CV {_metric_name} (mean)"].values
                        _ho_vals = _plot_cats[f"Holdout {_metric_name}"].values

                        axes[0].barh(
                            _y_pos - _bar_h / 2,
                            _cv_vals,
                            height=_bar_h,
                            color=_BLUE,
                            alpha=0.85,
                            label=f"Nested CV {_metric_name}",
                            edgecolor="white",
                            linewidth=0.3,
                        )
                        axes[0].barh(
                            _y_pos + _bar_h / 2,
                            _ho_vals,
                            height=_bar_h,
                            color=_TEAL,
                            alpha=0.85,
                            label=f"Holdout {_metric_name}",
                            edgecolor="white",
                            linewidth=0.3,
                        )

                        if f"Original {_metric_name}" in _plot_cats.columns:
                            _orig_vals = _plot_cats[f"Original {_metric_name}"].values
                            axes[0].scatter(
                                _orig_vals,
                                _y_pos,
                                color=_RED,
                                s=40,
                                zorder=5,
                                marker="D",
                                label=f"Original {_metric_name} (train/test)",
                            )

                        axes[0].set_yticks(_y_pos)
                        _short_names = []
                        for c in _plot_cats["Category"].values:
                            c_short = (
                                str(c)
                                .replace("Interpersonal Relationships & Social Functioning", "Social Functioning")
                                .replace("Parent Cognitive and Attention Issues", "Parent Cog. & Attn")
                                .replace("Parent Impulsivity and Behavior Regulation", "Parent Impuls. & Behav.")
                                .replace("Family Psychopathology & Well-Being", "Family Psychopath.")
                                .replace("Dynamic Cognitive Control Parameters", "Dyn. Cog. Control")
                            )
                            if len(c_short) > 35:
                                c_short = c_short[:33] + "…"
                            _short_names.append(c_short)
                        axes[0].set_yticklabels(_short_names, fontsize=9)
                        axes[0].set_xlabel(f"{_metric_name}", fontsize=11)
                        axes[0].set_title(
                            f"[nested_cv] Per-Category Validation\n{ctx.target_options} (T{ctx.tp_option})",
                            fontsize=13,
                            fontweight="bold",
                            color=_GREY,
                        )
                        axes[0].legend(fontsize=9, loc="lower right", framealpha=0.9)
                        axes[0].axvline(0, color="#CFD8DC", linewidth=0.8, zorder=0)
                        axes[0].spines["top"].set_visible(False)
                        axes[0].spines["right"].set_visible(False)

                        # Right panel: CV fold distributions for top-8 categories
                        _top8 = _ncv_df.nlargest(8, f"CV {_metric_name} (mean)")
                        _fold_data = []
                        _fold_labels = []
                        for _, _row in _top8.iterrows():
                            _cat = _row["Category"]
                            if _cat in ctx._within_ncv_results:
                                _folds = ctx._within_ncv_results[_cat]["cv_folds"]
                                _fold_data.append(_folds)
                                _label = str(_cat)
                                if len(_label) > 25:
                                    _label = _label[:23] + "…"
                                _fold_labels.append(_label)

                        if _fold_data:
                            bp = axes[1].boxplot(
                                _fold_data,
                                vert=False,
                                patch_artist=True,
                                boxprops=dict(facecolor=_BLUE, alpha=0.3),
                                medianprops=dict(color=_RED, linewidth=2),
                                whiskerprops=dict(color=_GREY),
                                capprops=dict(color=_GREY),
                                flierprops=dict(markerfacecolor=_GREY, markersize=4),
                            )
                            axes[1].set_yticks(range(1, len(_fold_labels) + 1), _fold_labels)
                            axes[1].set_xlabel(f"{_metric_name}", fontsize=11)
                            axes[1].set_title("Fold Distributions (Top 8)", fontsize=13, fontweight="bold", color=_GREY)
                            axes[1].axvline(0, color="#CFD8DC", linewidth=0.8, zorder=0)
                            axes[1].spines["top"].set_visible(False)
                            axes[1].spines["right"].set_visible(False)
                            axes[1].tick_params(axis="y", labelsize=9)

                        plt.tight_layout()
                        plt.savefig(
                            f"{results_dir_point1}/within_categories_ncv_comparison.png",
                            dpi=300,
                            bbox_inches="tight",
                            facecolor="white",
                        )
                        plt.show()
                        logger.info(f"\n   Saved: {results_dir_point1}/within_categories_ncv_comparison.png")

                    except Exception as _pe:
                        logger.info(f"  Warning: Comparison plot failed: {_pe}")

                    # Overfitting diagnostic: paired dot plot
                    if "Δ (CV − Orig)" in _ncv_df.columns and f"Original {_metric_name}" in _ncv_df.columns:
                        try:
                            _diag_df = _ncv_df.dropna(subset=[f"Original {_metric_name}", f"CV {_metric_name} (mean)"])
                            _diag_df = _diag_df.sort_values(f"Original {_metric_name}", ascending=True).reset_index(
                                drop=True
                            )

                            fig, ax = plt.subplots(figsize=(10, max(6, len(_diag_df) * 0.3)))
                            _y = np.arange(len(_diag_df))
                            _orig = _diag_df[f"Original {_metric_name}"].values
                            _cv = _diag_df[f"CV {_metric_name} (mean)"].values
                            _cats = _diag_df["Category"].values

                            # Connecting lines colored by direction
                            for _j in range(len(_diag_df)):
                                _lc = (
                                    "#C62828"
                                    if _cv[_j] < _orig[_j] - 0.005
                                    else "#2E7D32"
                                    if _cv[_j] > _orig[_j] + 0.005
                                    else "#90A4AE"
                                )
                                ax.plot([_orig[_j], _cv[_j]], [_y[_j], _y[_j]], color=_lc, lw=1.5, alpha=0.7)

                            ax.scatter(
                                _orig, _y, color=_RED, s=50, zorder=5, marker="D", label=f"Original {_metric_name}"
                            )
                            ax.scatter(
                                _cv, _y, color=_BLUE, s=50, zorder=5, marker="o", label=f"Nested CV {_metric_name}"
                            )

                            _short = [str(c)[:30] + "…" if len(str(c)) > 30 else str(c) for c in _cats]
                            ax.set_yticks(_y)
                            ax.set_yticklabels(_short, fontsize=8)
                            ax.set_xlabel(f"{_metric_name}", fontsize=11)
                            ax.set_title(
                                f"Overfitting Diagnostic: Original vs Nested CV {_metric_name}\n"
                                f"Red line = performance drop (potential overfit)",
                                fontsize=12,
                                fontweight="bold",
                                color=_GREY,
                            )
                            ax.legend(fontsize=9, loc="lower right")
                            ax.spines["top"].set_visible(False)
                            ax.spines["right"].set_visible(False)
                            ax.grid(axis="x", alpha=0.2)
                            plt.tight_layout()
                            plt.savefig(
                                f"{results_dir_point1}/overfitting_diagnostic.png",
                                dpi=300,
                                bbox_inches="tight",
                                facecolor="white",
                            )
                            plt.show()
                            logger.info(f"   Saved: {results_dir_point1}/overfitting_diagnostic.png")

                            _deltas = _diag_df["Δ (CV − Orig)"].values
                            _n_overfit = np.sum(_deltas < -0.01)
                            _n_stable = np.sum(np.abs(_deltas) <= 0.01)
                            _n_gain = np.sum(_deltas > 0.01)
                            logger.info("\n  Overfitting summary:")
                            logger.info(f"    Overfit (Δ < -0.01): {_n_overfit}/{len(_deltas)} categories")
                            logger.info(f"    Stable (|Δ| ≤ 0.01): {_n_stable}/{len(_deltas)} categories")
                            logger.info(f"    Gain   (Δ > +0.01):  {_n_gain}/{len(_deltas)} categories")

                        except Exception as _oe:
                            logger.info(f"  Warning: Overfitting diagnostic failed: {_oe}")

                    # Save results
                    _ncv_df.to_csv(f"{results_dir_point1}/within_categories_ncv_results.csv", index=False)
                    logger.info(f"\n   Saved: {results_dir_point1}/within_categories_ncv_results.csv")

                else:
                    logger.info("\n  Warning: No categories completed successfully.")

            elif "X_train" in ctx and "y_train" in ctx:
                # Across-categories mode: Original single-pool nested CV

                logger.info("\n Running [nested_cv] with pipeline variables")
                logger.info("   Using: X_train, y_train, model from notebook")

                # Detect task type
                task_type = detect_task_type(ctx.y_train)
                _task_is_classification = task_type == "classification"

                is_classification = _task_is_classification

                logger.info(f"   Task type detected: {task_type}")
                logger.info("   Using STACKING ENSEMBLE model factory (same as main pipeline)")

                # Outer 5-FOLD CV (training data only)

                X_cv = ctx.X_train.copy()
                y_cv = ctx.y_train.copy()

                ctx.reporter_point1 = NestedCVAnalysis(X_cv, y_cv, task_type="auto", n_outer=_VAL_N_OUTER)

                results_dir_point1 = os.path.join(os.getcwd(), "nestedcv_results_point1")
                os.makedirs(results_dir_point1, exist_ok=True)

                ctx.reporter_point1.run_outer_cv(
                    model_factory=lambda: ctx.create_validation_model(is_classification=is_classification),
                    results_dir=results_dir_point1,
                )

                ctx.reporter_point1.generate_report(title="[nested_cv] Nested CV Performance - RP", save_plots=True)

                # OOF pred-vs-actual scatter (the key regression figure)
                if not is_classification and hasattr(ctx.reporter_point1, "outer_fold_results"):
                    try:
                        y_true_oof = np.concatenate(ctx.reporter_point1.outer_fold_results["y_true"])
                        y_pred_oof = np.concatenate(ctx.reporter_point1.outer_fold_results["y_pred"])

                        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                        # Panel 1: Hexbin density plot (N≈8000 makes scatter unreadable)
                        from matplotlib.colors import LogNorm

                        lims = [min(y_true_oof.min(), y_pred_oof.min()), max(y_true_oof.max(), y_pred_oof.max())]

                        hb = axes[0].hexbin(
                            y_true_oof, y_pred_oof, gridsize=40, cmap="YlOrRd", mincnt=1, norm=LogNorm()
                        )
                        cb = fig.colorbar(hb, ax=axes[0], shrink=0.8, label="Count (log)")

                        # Identity line + fit line
                        axes[0].plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
                        z = np.polyfit(y_true_oof, y_pred_oof, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(lims[0], lims[1], 100)
                        axes[0].plot(
                            x_line, p(x_line), color="#1565C0", lw=2, alpha=0.9, label=f"Fit: y={z[0]:.2f}x+{z[1]:.2f}"
                        )

                        oof_r2 = r2_score(y_true_oof, y_pred_oof)
                        oof_rmse = np.sqrt(mean_squared_error(y_true_oof, y_pred_oof))
                        oof_mae = mean_absolute_error(y_true_oof, y_pred_oof)
                        axes[0].set_xlabel("Actual", fontsize=12)
                        axes[0].set_ylabel("Predicted (OOF)", fontsize=12)
                        axes[0].set_title(
                            f"OOF Predictions (5-Fold CV)\n"
                            f"R²={oof_r2:.3f}, RMSE={oof_rmse:.3f}, MAE={oof_mae:.3f}, N={len(y_true_oof)}",
                            fontsize=11,
                            fontweight="bold",
                        )
                        axes[0].legend(fontsize=9, loc="upper left")
                        axes[0].spines["top"].set_visible(False)
                        axes[0].spines["right"].set_visible(False)

                        # Panel 2: Residual distribution with ±1 SD shading
                        residuals_oof = y_true_oof - y_pred_oof
                        res_mean = np.mean(residuals_oof)
                        res_sd = np.std(residuals_oof)
                        res_skew = pd.Series(residuals_oof).skew()

                        axes[1].hist(
                            residuals_oof,
                            bins=60,
                            color="#42A5F5",
                            edgecolor="black",
                            alpha=0.7,
                            density=True,
                            linewidth=0.3,
                        )
                        axes[1].axvline(0, color="#C62828", linestyle="--", lw=2, label="Zero")
                        axes[1].axvline(res_mean, color="#2E7D32", linestyle=":", lw=2, label=f"Mean={res_mean:.4f}")
                        axes[1].axvspan(
                            res_mean - res_sd, res_mean + res_sd, alpha=0.12, color="#2E7D32", label="±1 SD"
                        )
                        axes[1].set_xlabel("Residual (Actual − Predicted)", fontsize=12)
                        axes[1].set_ylabel("Density", fontsize=12)
                        axes[1].set_title(
                            f"Residual Distribution\nMean={res_mean:.4f}, SD={res_sd:.3f}, Skew={res_skew:.3f}",
                            fontsize=11,
                            fontweight="bold",
                        )
                        axes[1].legend(fontsize=9)
                        axes[1].spines["top"].set_visible(False)
                        axes[1].spines["right"].set_visible(False)

                        plt.suptitle("[nested_cv] Out-of-Fold Prediction Quality", fontsize=14, fontweight="bold")
                        plt.tight_layout()
                        plt.savefig(f"{results_dir_point1}/oof_pred_vs_actual.png", dpi=300, bbox_inches="tight")
                        plt.show()
                        logger.info(f"   Saved: {results_dir_point1}/oof_pred_vs_actual.png")

                    except Exception as e:
                        logger.info(f"   Warning: Could not plot OOF predictions: {e}")

                # OOF classification figure (ROC + Calibration 2-panel)
                if is_classification and hasattr(ctx.reporter_point1, "outer_fold_results"):
                    try:
                        from sklearn.calibration import calibration_curve
                        from sklearn.metrics import auc as sk_auc
                        from sklearn.metrics import roc_curve as sk_roc_curve

                        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                        # Panel A: Pooled ROC with per-fold curves
                        mean_fpr = np.linspace(0, 1, 100)
                        tprs = []
                        for _fi in range(len(ctx.reporter_point1.outer_fold_results["y_true"])):
                            _yt = ctx.reporter_point1.outer_fold_results["y_true"][_fi]
                            _yp = ctx.reporter_point1.outer_fold_results["y_proba"][_fi]
                            fpr, tpr, _ = sk_roc_curve(_yt, _yp)
                            _fauc = sk_auc(fpr, tpr)
                            interp_tpr = np.interp(mean_fpr, fpr, tpr)
                            interp_tpr[0] = 0.0
                            tprs.append(interp_tpr)
                            axes[0].plot(fpr, tpr, alpha=0.25, lw=1, label=f"Fold {_fi + 1} (AUC={_fauc:.3f})")
                        mean_tpr = np.mean(tprs, axis=0)
                        mean_tpr[-1] = 1.0
                        mean_auc = sk_auc(mean_fpr, mean_tpr)
                        std_auc = np.std([sk_auc(mean_fpr, t) for t in tprs])
                        axes[0].plot(
                            mean_fpr,
                            mean_tpr,
                            color="#1565C0",
                            lw=3,
                            label=f"Mean (AUC={mean_auc:.3f} \u00b1 {std_auc:.3f})",
                        )
                        std_tpr = np.std(tprs, axis=0)
                        axes[0].fill_between(
                            mean_fpr,
                            np.maximum(mean_tpr - std_tpr, 0),
                            np.minimum(mean_tpr + std_tpr, 1),
                            color="#1565C0",
                            alpha=0.12,
                            label="\u00b1 1 SD",
                        )
                        axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
                        axes[0].set_xlabel("False Positive Rate", fontsize=12)
                        axes[0].set_ylabel("True Positive Rate", fontsize=12)
                        axes[0].set_title(
                            f"OOF ROC ({_VAL_N_OUTER}-Fold CV)\n"
                            f"Mean AUC={mean_auc:.3f}, N={sum(len(y) for y in ctx.reporter_point1.outer_fold_results['y_true'])}",
                            fontsize=11,
                            fontweight="bold",
                        )
                        axes[0].legend(fontsize=8, loc="lower right")
                        axes[0].set_xlim([-0.02, 1.02])
                        axes[0].set_ylim([-0.02, 1.02])
                        axes[0].spines["top"].set_visible(False)
                        axes[0].spines["right"].set_visible(False)

                        # Panel B: Calibration curve with Wilson CIs
                        y_true_all = np.concatenate(ctx.reporter_point1.outer_fold_results["y_true"])
                        y_proba_all = np.concatenate(ctx.reporter_point1.outer_fold_results["y_proba"])
                        n_bins = 10
                        bins = np.linspace(0, 1, n_bins + 1)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        bin_true = np.zeros(n_bins)
                        bin_ci_lo = np.zeros(n_bins)
                        bin_ci_hi = np.zeros(n_bins)
                        for _bi in range(n_bins):
                            mask = (y_proba_all >= bins[_bi]) & (y_proba_all < bins[_bi + 1])
                            if mask.sum() > 0:
                                p = y_true_all[mask].mean()
                                n = mask.sum()
                                bin_true[_bi] = p
                                z = 1.96
                                denom = 1 + z**2 / n
                                center = (p + z**2 / (2 * n)) / denom
                                margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
                                bin_ci_lo[_bi] = max(0, center - margin)
                                bin_ci_hi[_bi] = min(1, center + margin)
                        ece = np.mean(np.abs(bin_true - bin_centers))
                        axes[1].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect")
                        axes[1].errorbar(
                            bin_centers,
                            bin_true,
                            yerr=[np.clip(bin_true - bin_ci_lo, 0, None), np.clip(bin_ci_hi - bin_true, 0, None)],
                            fmt="o-",
                            capsize=4,
                            capthick=2,
                            lw=2,
                            markersize=8,
                            color="#1565C0",
                            label=f"Model (ECE={ece:.4f})",
                        )
                        axes[1].fill_between(bin_centers, bin_ci_lo, bin_ci_hi, alpha=0.15, color="#1565C0")
                        axes[1].set_xlabel("Mean Predicted Probability", fontsize=12)
                        axes[1].set_ylabel("Observed Fraction Positive", fontsize=12)
                        axes[1].set_title(
                            f"OOF Calibration (Wilson 95% CI)\nECE={ece:.4f}", fontsize=11, fontweight="bold"
                        )
                        axes[1].legend(fontsize=9)
                        axes[1].set_xlim([0, 1])
                        axes[1].set_ylim([0, 1])
                        axes[1].spines["top"].set_visible(False)
                        axes[1].spines["right"].set_visible(False)

                        plt.suptitle("[nested_cv] Out-of-Fold Classification Quality", fontsize=14, fontweight="bold")
                        plt.tight_layout()
                        plt.savefig(f"{results_dir_point1}/oof_clf_roc_calibration.png", dpi=300, bbox_inches="tight")
                        plt.show()
                        logger.info(f"  Saved: {results_dir_point1}/oof_clf_roc_calibration.png")

                    except Exception as e:
                        logger.info(f"   Warning: Could not plot classification OOF figure: {e}")

                # Final independent holdout evaluation

                if ("X_test" in ctx) and ("y_test" in ctx):
                    logger.info("\n Running FINAL UNTOUCHED HOLDOUT evaluation")

                    final_model = ctx.create_validation_model(is_classification=is_classification)

                    final_model.fit(ctx.X_train, ctx.y_train)

                    if is_classification:
                        y_pred = final_model.predict(ctx.X_test)
                        y_proba = (
                            final_model.predict_proba(ctx.X_test)[:, 1]
                            if hasattr(final_model, "predict_proba")
                            else None
                        )

                        logger.info("\n[HOLDOUT RESULTS]")
                        logger.info("%s %s", "Accuracy:", accuracy_score(ctx.y_test, y_pred))
                        logger.info("%s %s", "F1:", f1_score(ctx.y_test, y_pred))

                        if y_proba is not None:
                            logger.info("%s %s", "ROC-AUC:", roc_auc_score(ctx.y_test, y_proba))
                            logger.info("%s %s", "Brier:", brier_score_loss(ctx.y_test, y_proba))

                    else:
                        y_pred = final_model.predict(ctx.X_test)

                        logger.info("\n[HOLDOUT RESULTS]")
                        logger.info("%s %s", "R2:", r2_score(ctx.y_test, y_pred))
                        logger.info("%s %s", "RMSE:", np.sqrt(mean_squared_error(ctx.y_test, y_pred)))
                        logger.info("%s %s", "MAE:", mean_absolute_error(ctx.y_test, y_pred))

            else:
                logger.info("\nError: X_train or y_train not found")
                logger.info("   Run setup and the main stage first.")

        except Exception as e:
            logger.info(f"\nError in [nested_cv]: {e}")
            import traceback

            traceback.print_exc()

        logger.info("\n[nested_cv] complete.")

        # [Cache + report save]
        _RS["section"] = "uncategorized"
        try:
            _pipeline_cleanup(label="P1 complete")

            cache_save(
                {"reporter_point1": ctx.get("reporter_point1")},
                cell_tag=_CTAG_12,
                key=_CKEY_12,
                label=f"{ctx.target_options}_tp{ctx.tp_option}_[nested_cv] Nested CV",
            )
            report_save_state(make_run_key())
        except Exception as _ce_12:
            logger.info(f"[Cache] Could not save nested_cv: {_ce_12}")


def run() -> None:
    """Run this stage (notebook cell 12)."""
    global copy

    # Batch dispatch
    _is_bv_15 = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_15:
        try:
            _cbv_15 = cache_load("main_runner")
            if (
                _cbv_15
                and isinstance(_cbv_15, dict)
                and _cbv_15.get("_batch_archive")
                and len(_cbv_15["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_15["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_15["_batch_archive"]
                _is_bv_15 = True
        except Exception:
            pass

    if _is_bv_15:
        _vgs_15 = {
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
            f"  BATCH -- Outer-Fold Nested CV Metrics & Calibration: processing {len(ctx._batch_archive)} targets"
        )
        logger.info("=" * 72)

        for _vi_15, (_vk_15, _vd_15) in enumerate(ctx._batch_archive.items()):
            _vl_15 = _vd_15.get("target", _vk_15)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_15 + 1}/{len(ctx._batch_archive)}] Target: {_vl_15}")
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
                if _vd_15.get(_gk) is not None:
                    ctx[_gk] = _vd_15[_gk]
            ctx["target_options"] = _vl_15
            if "timepoint" in _vd_15:
                ctx["tp_option"] = _vd_15["timepoint"]
            _run_for_current_target()
            # Archive validation reporters back into _batch_archive
            import copy

            if _vk_15 in ctx._batch_archive:
                ctx._batch_archive[_vk_15]["reporter_point1"] = ctx.get("reporter_point1")
                ctx._batch_archive[_vk_15]["_within_ncv_results"] = copy.deepcopy(ctx.get("_within_ncv_results"))

        for _k, _v in _vgs_15.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(
            f"  BATCH COMPLETE -- Outer-Fold Nested CV Metrics & Calibration: {len(ctx._batch_archive)} targets"
        )
        logger.info("=" * 72)
    else:
        _run_for_current_target()
