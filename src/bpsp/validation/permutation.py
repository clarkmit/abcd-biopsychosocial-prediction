"""Permutation testing and null distributions (the `permutation` stage, Supplement S1.4.4, S1.11.5).

Empirical null distribution via label permutation with the Phipson-Smyth
correction (p = [r+1]/[n+1], preventing p = 0 from finite permutations).
A single CatBoost model (the main pipeline's settings with 250 iterations)
is scored by k-fold CV on the training and test rows together, first on the
true labels and then on each permutation of them; the p-value is upper-tailed.
"""

import logging

from bpsp.context import ctx
from bpsp.data.preprocessing import detect_task_type
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
Permutation-Based Model Significance Testing
---------------------------------------------
Empirical null distribution via label permutation. Refits a single CatBoost
model on shuffled outcome labels to estimate chance-level performance, then
computes an upper-tailed p-value against the same model's observed metric.

Procedure
---------
  1. Record observed performance metric (AUC or R²).
  2. Permute outcome labels n_permutations times and refit the model.
  3. Compute p = P(null ≥ observed).
  4. Plot null histogram with observed value marked.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

# Style constants --
_P4_BLUE = "#1565C0"
_P4_RED = "#C62828"
_P4_GREEN = "#2E7D32"
_P4_GREY = "#455A64"
_P4_LTGREY = "#ECEFF1"
_P4_SLATE = "#1F2D3D"
_P4_ORANGE = "#E65100"
_P4_TEAL = "#00838F"


def _section_hdr(title, subtitle="", color=_P4_BLUE):
    sub_html = f'<br><span style="font-size:12px;color:#B0BEC5;">{subtitle}</span>' if subtitle else ""
    display(
        HTML(
            f"<div style=\"font-family:'Helvetica Neue',Arial,sans-serif;max-width:800px;"
            f"border-left:4px solid {color};background:linear-gradient(135deg,{color}11,{color}06);"
            f'padding:14px 20px;margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
            f'<span style="font-size:15px;font-weight:700;color:{color};">{title}</span>'
            f"{sub_html}</div>"
        )
    )


def _kpi_panel(metrics_dict, title="", color=_P4_BLUE):
    html = (
        f'<div style="font-family:sans-serif;max-width:750px;border:2px solid {color};'
        f'border-radius:8px;overflow:hidden;margin:12px 0;">'
        f'<div style="background:{color};color:white;padding:12px 18px;'
        f'font-size:14px;font-weight:700;">{title}</div>'
        f'<div style="padding:16px 20px;display:flex;gap:24px;flex-wrap:wrap;">'
    )
    for label, value, sub in metrics_dict:
        html += (
            f'<div style="flex:1 1 140px;">'
            f'<span style="color:#78909C;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:.4px;">{label}</span><br>'
            f'<span style="font-size:20px;font-weight:700;color:{_P4_SLATE};">{value}</span>'
        )
        if sub:
            html += f'<br><span style="font-size:11px;color:#78909C;">{sub}</span>'
        html += "</div>"
    html += "</div></div>"
    display(HTML(html))


def _fmt_table(df, caption="", gradient_col=None, highlight_col=None):
    """Display a DataFrame with consistent notebook styling."""
    _num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    styler = (
        df.style.format({c: "{:.4f}" for c in _num_cols})
        .set_caption(caption)
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "14px"),
                        ("font-weight", "700"),
                        ("color", _P4_GREY),
                        ("padding-bottom", "8px"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", _P4_LTGREY),
                        ("color", _P4_GREY),
                        ("font-size", "11px"),
                        ("font-weight", "600"),
                        ("text-transform", "uppercase"),
                        ("letter-spacing", ".4px"),
                        ("padding", "6px 10px"),
                    ],
                },
                {"selector": "td", "props": [("padding", "5px 10px"), ("font-size", "12px")]},
                {"selector": "tr:hover td", "props": [("background-color", "#F5F5F5")]},
            ]
        )
        .hide(axis="index")
    )
    if gradient_col and gradient_col in df.columns and df[gradient_col].nunique(dropna=True) > 1:
        styler = styler.background_gradient(
            cmap="RdYlGn", subset=[gradient_col], vmin=0, vmax=max(0, df[gradient_col].max())
        )
    if highlight_col and highlight_col in df.columns:
        styler = styler.map(
            lambda v: "color: #2E7D32; font-weight: bold" if v == "Yes" else "color: #C62828; font-weight: bold",
            subset=[highlight_col],
        )
    display(styler)


def get_scoring_metric(task_type):
    """Get appropriate scoring metric for task type."""
    if task_type == "classification":
        return "roc_auc"
    else:
        return "r2"


# PermutationTester Class


class PermutationTester:
    """
    Test model significance via label permutation.

    MOTIVATION (concern about "predictive pleiotropy"):
    - Many psychological variables covary (higher-order internalizing and
      externalizing spectra, shared rater effects)
    - Strong predictive performance might reflect covariance, not true prediction
    - Solution: Show observed performance >> null distribution

    METHODOLOGY:
    1. Observe: Train model on true y, measure performance
    2. Permute: Randomly shuffle y labels
    3. Null: Train model on permuted y, measure performance
    4. Repeat: Generate null distribution from 100+ permutations
    5. P-value: proportion of null scores ≥ observed, upper-tailed, with the
       Phipson & Smyth correction
    """

    def __init__(self, X, y, task_type="auto", n_permutations=100, cv_folds=5, random_state=42):
        """X: features, y: outcomes. task_type inferred if 'auto'.
        n_permutations : int
            Number of permutations for null distribution (default 100, use 1000 for final)
        cv_folds : int
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
        self.n_permutations = n_permutations
        self.cv_folds = cv_folds
        self.random_state = random_state

        if task_type == "auto":
            self.task_type = detect_task_type(self.y.values)
        else:
            self.task_type = task_type

        self.is_classification = self.task_type == "classification"
        self.is_regression = self.task_type == "regression"
        self.scoring_metric = get_scoring_metric(self.task_type)

        self.observed_score = None
        self.null_scores = []
        self.p_value = None

        logger.info(f"Task type: {self.task_type.upper()}")
        logger.info(f"   Scoring metric: {self.scoring_metric.upper()}")
        logger.info(f"   Permutations: {n_permutations}")

    def run_permutation_test(self, model_factory=None, base_model=None, results_dir="permutation_results"):
        """
        Run permutation significance test.

        Parameters:
        model_factory : callable, optional
            Function returning fresh model (for retraining each fold)
        base_model : estimator, optional
            Pre-trained model from pipeline
        results_dir : str
            Directory to save results
        """
        os.makedirs(results_dir, exist_ok=True)

        if model_factory is None and base_model is None:
            raise ValueError("Either model_factory or base_model must be provided")

        logger.info(f"\n Running permutation test ({self.n_permutations} permutations)")

        # Step 1: Compute observed score on true labels
        logger.info("\n   Step 1: Computing observed score (true labels)...")
        self.observed_score = self._compute_cv_score(self.y, model_factory)
        logger.info(f"           Observed {self.scoring_metric.upper()}: {self.observed_score:.4f}")

        # Step 2: Generate null distribution via label permutation
        logger.info(f"\n   Step 2: Generating null distribution ({self.n_permutations} permutations)...")
        rng = np.random.default_rng(self.random_state)

        for perm_num in range(self.n_permutations):
            # Permute labels
            y_perm = rng.permutation(self.y.values)

            # Compute score on permuted labels
            perm_score = self._compute_cv_score(y_perm, model_factory)
            self.null_scores.append(perm_score)

            if (perm_num + 1) % 20 == 0:
                logger.info(f"           Progress: {perm_num + 1}/{self.n_permutations}")

        self.null_scores = np.array(self.null_scores)

        # Step 3: Compute p-value
        self.p_value = self._compute_p_value()

        # Step 4: Save results
        self._save_results(results_dir)

        return {"observed_score": self.observed_score, "null_scores": self.null_scores, "p_value": self.p_value}

    def _compute_cv_score(self, y, model_factory):
        """Compute CV score (mean across folds)."""
        fold_scores = []

        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in cv.split(self.X, y):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
            y_test = y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx]

            # y is pandas-compatible (stacking ensemble uses .iloc internally)
            if isinstance(y_train, np.ndarray):
                y_train = pd.Series(y_train, index=X_train.index)
            if isinstance(y_test, np.ndarray):
                y_test = pd.Series(y_test, index=X_test.index)

            # Always retrain model (required for valid permutation null)
            model = model_factory()
            model.fit(X_train, y_train)

            # Score
            if self.is_classification:
                y_proba = model.predict_proba(X_test)
                classes = np.unique(y_test)
                if len(classes) == 2:
                    score = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    score = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            else:
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

            fold_scores.append(score)

        return np.mean(fold_scores)

    def _compute_p_value(self):
        """
        Compute upper-tailed p-value (Phipson & Smyth 2010).

        P-value = (# null scores ≥ observed + 1) / (# permutations + 1)
        The "+1" prevents p=0 and accounts for the observed score as one permutation
        (Phipson & Smyth, 2010).
        """
        num_extreme = np.sum(self.null_scores >= self.observed_score)
        p_value = (num_extreme + 1) / (len(self.null_scores) + 1)

        return p_value

    def _save_results(self, results_dir):
        """Save permutation results to CSV."""
        results_df = pd.DataFrame({"permutation": range(len(self.null_scores)), "score": self.null_scores})

        results_df.to_csv(f"{results_dir}/permutation_null_distribution.csv", index=False)

        summary_df = pd.DataFrame(
            {
                "Metric": ["Observed Score", "Null Mean", "Null SD", "P-value", "Significant (α=0.05)"],
                "Value": [
                    f"{self.observed_score:.4f}",
                    f"{np.mean(self.null_scores):.4f}",
                    f"{np.std(self.null_scores):.4f}",
                    f"{self.p_value:.4f}",
                    "Yes (significant)" if self.p_value < 0.05 else "No (n.s.)",
                ],
            }
        )

        summary_df.to_csv(f"{results_dir}/permutation_test_summary.csv", index=False)

        logger.info(f"\nSaved results to {results_dir}/")

    def plot_null_distribution(self, figsize=(12, 6), save_path=None):
        """
        Publication-quality null distribution plot with KDE overlay,
        rejection zone shading, and effect-size annotation.
        """
        if self.observed_score is None:
            raise ValueError("Run run_permutation_test() first.")

        _metric_upper = self.scoring_metric.upper()

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("#FAFBFC")

        # Histogram of null distribution
        _counts, _bins, _patches = ax.hist(
            self.null_scores,
            bins=min(50, max(20, len(self.null_scores) // 15)),
            alpha=0.65,
            color=_P4_BLUE,
            edgecolor="white",
            linewidth=0.6,
            label=f"Null distribution (n={len(self.null_scores)})",
            zorder=2,
        )

        # KDE overlay for distributional shape
        try:
            sns.kdeplot(self.null_scores, ax=ax, color=_P4_GREY, linewidth=1.8, alpha=0.7, zorder=3)
        except Exception:
            pass  # graceful fallback if too few unique values

        # Rejection zone: shade area ≥ observed
        _xlim = ax.get_xlim()
        ax.axvspan(
            self.observed_score,
            _xlim[1],
            alpha=0.07,
            color=_P4_RED,
            zorder=1,
            label=f"Rejection zone (p={self.p_value:.4f})",
        )
        ax.set_xlim(_xlim)  # restore in case axvspan shifted it

        # Null mean line
        null_mean = np.mean(self.null_scores)
        null_sd = np.std(self.null_scores)
        ax.axvline(
            null_mean,
            color=_P4_GREEN,
            linestyle=":",
            linewidth=2.0,
            alpha=0.8,
            label=f"Null mean = {null_mean:.4f}",
            zorder=4,
        )

        # Observed score line
        ax.axvline(
            self.observed_score,
            color=_P4_RED,
            linestyle="--",
            linewidth=2.8,
            alpha=0.9,
            label=f"Observed {_metric_upper} = {self.observed_score:.4f}",
            zorder=5,
        )

        # Effect-size annotation box
        _d = (self.observed_score - null_mean) / (null_sd + 1e-10)
        _sig_label = "Significant" if self.p_value < 0.05 else "Not significant"
        _sig_color = _P4_GREEN if self.p_value < 0.05 else _P4_ORANGE

        _annot_text = (
            f"Observed: {self.observed_score:.4f}\n"
            f"Null mean: {null_mean:.4f} ± {null_sd:.4f}\n"
            f"Effect: {_d:.2f} SD above null\n"
            f"p = {self.p_value:.4f}  ({_sig_label})"
        )
        _bbox_props = dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor=_sig_color,
            linewidth=1.5,
            alpha=0.95,
        )
        ax.annotate(
            _annot_text,
            xy=(self.observed_score, ax.get_ylim()[1] * 0.85),
            xytext=(0.97, 0.95),
            textcoords="axes fraction",
            fontsize=10,
            fontweight="500",
            color=_P4_SLATE,
            ha="right",
            va="top",
            bbox=_bbox_props,
            arrowprops=dict(arrowstyle="->", color=_sig_color, lw=1.5),
            zorder=6,
        )

        # Axis labels and title
        _target_lbl = ctx.get("target_options", "Target")
        _tp_lbl = ctx.get("tp_option", "")
        _tp_str = f" (T{_tp_lbl})" if _tp_lbl else ""

        ax.set_xlabel(f"{_metric_upper} Score", fontsize=12, fontweight="600", color=_P4_GREY)
        ax.set_ylabel("Frequency", fontsize=12, fontweight="600", color=_P4_GREY)
        ax.set_title(
            f"Permutation Significance Test -- {_target_lbl}{_tp_str}\n"
            f"{len(self.null_scores)} permutations  ·  "
            f"p = {self.p_value:.4f}  ·  {_sig_label}",
            fontsize=14,
            fontweight="bold",
            color=_P4_SLATE,
            pad=16,
            linespacing=1.5,
        )

        # Legend
        ax.legend(
            fontsize=10,
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            edgecolor="#CFD8DC",
            fancybox=True,
        )

        # Grid and spine cleanup
        ax.grid(True, alpha=0.2, axis="y", linestyle=":", color="#B0BEC5")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CFD8DC")
        ax.spines["bottom"].set_color("#CFD8DC")
        ax.tick_params(labelsize=10, colors=_P4_GREY)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
            logger.info(f"   Saved: {save_path}")
        plt.show()

    def generate_report(self, title="Permutation Test: Model Significance"):
        """Generate styled HTML report for permutation test results."""
        if self.observed_score is None:
            raise ValueError("Run run_permutation_test() first.")

        _metric_upper = self.scoring_metric.upper()
        _target_lbl = ctx.get("target_options", "Target")
        _tp_lbl = ctx.get("tp_option", "")

        # Section header
        _section_hdr(
            title,
            subtitle=f"{_target_lbl}  ·  T{_tp_lbl}  ·  {self.n_permutations} permutations",
            color=_P4_BLUE,
        )

        # KPI panel
        null_mean = np.mean(self.null_scores)
        null_sd = np.std(self.null_scores)
        _d = (self.observed_score - null_mean) / (null_sd + 1e-10)
        _sig = self.p_value < 0.05
        _sig_label = "Significant" if _sig else "Not significant"

        _kpi_panel(
            [
                (f"Observed {_metric_upper}", f"{self.observed_score:.4f}", None),
                (f"Null Mean {_metric_upper}", f"{null_mean:.4f}", f"± {null_sd:.4f}"),
                ("p-value", f"{self.p_value:.4f}", _sig_label),
                ("Effect Size", f"{_d:.2f} SD", "above null mean"),
            ],
            title=title,
            color=_P4_GREEN if _sig else _P4_ORANGE,
        )

        # Results table
        results_table = pd.DataFrame(
            {
                "Statistic": [
                    "Observed Score",
                    "Null Mean",
                    "Null SD",
                    "Null Min",
                    "Null Max",
                    "P-value (upper-tailed)",
                    "Significant (α=0.05)",
                ],
                "Value": [
                    self.observed_score,
                    null_mean,
                    null_sd,
                    float(np.min(self.null_scores)),
                    float(np.max(self.null_scores)),
                    self.p_value,
                    np.nan,  # handled by highlight
                ],
                "Interpretation": [
                    f"Best {_metric_upper} from true labels",
                    f"Expected {_metric_upper} under H₀",
                    "Spread of null distribution",
                    "Worst null permutation",
                    "Best null permutation",
                    "Phipson & Smyth (2010)",
                    "YES" if _sig else "NO",
                ],
            }
        )
        # Format numeric values nicely but leave Interpretation as-is
        _num_cols_rt = ["Value"]
        _rt_styler = (
            results_table.style.format({c: "{:.4f}" for c in _num_cols_rt}, na_rep="--")
            .set_caption(f"Permutation Test Summary -- {_target_lbl} (T{_tp_lbl})")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "14px"),
                            ("font-weight", "700"),
                            ("color", _P4_GREY),
                            ("padding-bottom", "8px"),
                        ],
                    },
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", _P4_LTGREY),
                            ("color", _P4_GREY),
                            ("font-size", "11px"),
                            ("font-weight", "600"),
                            ("text-transform", "uppercase"),
                            ("letter-spacing", ".4px"),
                            ("padding", "6px 10px"),
                        ],
                    },
                    {"selector": "td", "props": [("padding", "5px 10px"), ("font-size", "12px")]},
                    {"selector": "tr:hover td", "props": [("background-color", "#F5F5F5")]},
                ]
            )
            .hide(axis="index")
        )
        display(_rt_styler)

        # Interpretation panel
        if _sig:
            _interp_html = (
                f'<div style="font-family:sans-serif;max-width:750px;border-left:4px solid {_P4_GREEN};'
                f'background:#E8F5E9;padding:14px 20px;margin:12px 0;border-radius:0 6px 6px 0;">'
                f'<span style="font-size:13px;font-weight:600;color:{_P4_GREEN};">Model is statistically significant</span><br>'
                f'<span style="font-size:12px;color:#37474F;line-height:1.7;">'
                f"The observed {_metric_upper} of {self.observed_score:.4f} is {_d:.2f} standard deviations "
                f"above the null mean ({null_mean:.4f}). Only {int(np.sum(self.null_scores >= self.observed_score))} "
                f"of {len(self.null_scores)} permutations matched or exceeded this value (p = {self.p_value:.4f})."
                f"</span></div>"
            )
        else:
            _interp_html = (
                f'<div style="font-family:sans-serif;max-width:750px;border-left:4px solid {_P4_ORANGE};'
                f'background:#FFF3E0;padding:14px 20px;margin:12px 0;border-radius:0 6px 6px 0;">'
                f'<span style="font-size:13px;font-weight:600;color:{_P4_ORANGE};">Model is not statistically significant</span><br>'
                f'<span style="font-size:12px;color:#37474F;line-height:1.7;">'
                f"The observed {_metric_upper} of {self.observed_score:.4f} could plausibly arise "
                f"from the null distribution (p = {self.p_value:.4f})."
                f"</span></div>"
            )
        display(HTML(_interp_html))

        # Null distribution figure
        self.plot_null_distribution()

    def plot_within_categories_forest(
        self, perm_results_list, metric_name="R²", target_lbl="", tp_lbl="", save_path=None
    ):
        """
        Forest plot of observed scores vs null means across categories.
        Called after within-categories permutation testing.

        Parameters:
        perm_results_list : list of dict
            Each dict has 'Category', 'Observed {metric}', 'Null Mean {metric}', 'p-value'
        """
        if not perm_results_list:
            return

        n_cats = len(perm_results_list)
        obs_key = f"Observed {metric_name}"
        null_key = f"Null Mean {metric_name}"

        cats = [r["Category"] for r in perm_results_list]
        obs_scores = [r.get(obs_key, np.nan) for r in perm_results_list]
        null_means = [r.get(null_key, np.nan) for r in perm_results_list]
        pvals = [r.get("p-value", np.nan) for r in perm_results_list]

        # Sort by observed score descending
        _order = np.argsort(obs_scores)[::-1]
        cats = [cats[i] for i in _order]
        obs_scores = [obs_scores[i] for i in _order]
        null_means = [null_means[i] for i in _order]
        pvals = [pvals[i] for i in _order]

        fig_height = max(4, n_cats * 0.7 + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height), facecolor="white")
        ax.set_facecolor("#FAFBFC")

        y_pos = np.arange(n_cats)

        # Null distribution violins (background context)
        null_dists = [r.get("null_scores", []) for r in perm_results_list]
        null_dists = [null_dists[i] for i in _order]
        _has_dists = any(len(d) > 2 for d in null_dists)
        if _has_dists:
            for i in range(n_cats):
                if len(null_dists[i]) > 2:
                    from scipy.stats import gaussian_kde

                    try:
                        _nd = np.array(null_dists[i])
                        _kde = gaussian_kde(_nd, bw_method=0.3)
                        _xgrid = np.linspace(_nd.min() - 0.01, _nd.max() + 0.01, 80)
                        _density = _kde(_xgrid)
                        _density = _density / _density.max() * 0.35  # scale to row height
                        ax.fill_between(
                            _xgrid, y_pos[i] - _density, y_pos[i] + _density, alpha=0.15, color="#90A4AE", zorder=1
                        )
                    except Exception:
                        pass  # skip kde if it fails

        # Null mean markers (gray diamonds)
        ax.scatter(null_means, y_pos, color=_P4_GREY, marker="D", s=60, zorder=3, label="Null mean", alpha=0.7)

        # Observed score markers (colored circles)
        _colors = [_P4_GREEN if p < 0.05 else _P4_ORANGE for p in pvals]
        ax.scatter(obs_scores, y_pos, color=_colors, marker="o", s=100, zorder=4, edgecolors="white", linewidths=1.2)

        # Connecting lines (null → observed)
        for i in range(n_cats):
            ax.plot(
                [null_means[i], obs_scores[i]],
                [y_pos[i], y_pos[i]],
                color=_colors[i],
                linewidth=1.5,
                alpha=0.5,
                zorder=2,
            )

        # P-value annotations
        for i in range(n_cats):
            _x_annot = max(obs_scores[i], null_means[i]) + 0.005
            _p_str = f"p={pvals[i]:.4f}" if not np.isnan(pvals[i]) else "p=N/A"
            _fw = "bold" if pvals[i] < 0.05 else "normal"
            ax.text(_x_annot, y_pos[i], _p_str, va="center", ha="left", fontsize=9.5, fontweight=_fw, color=_colors[i])

        # Labels and formatting
        # Abbreviate long category names
        _cat_labels = [c[:35] + "…" if len(c) > 35 else c for c in cats]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(_cat_labels, fontsize=10, color=_P4_SLATE)
        ax.set_xlabel(f"{metric_name} Score", fontsize=12, fontweight="600", color=_P4_GREY)
        ax.set_title(
            f"Permutation Test -- Forest Plot\n{target_lbl}  ·  T{tp_lbl}  ·  {n_cats} categories",
            fontsize=14,
            fontweight="bold",
            color=_P4_SLATE,
            pad=16,
            linespacing=1.5,
        )

        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CFD8DC")
        ax.spines["bottom"].set_color("#CFD8DC")
        ax.grid(True, axis="x", alpha=0.2, linestyle=":", color="#B0BEC5")
        ax.tick_params(labelsize=10, colors=_P4_GREY)

        # Legend
        _leg_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=_P4_GREEN,
                markeredgecolor="white",
                markersize=10,
                label="Observed (p<0.05)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=_P4_ORANGE,
                markeredgecolor="white",
                markersize=10,
                label="Observed (p≥0.05)",
            ),
            plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=_P4_GREY, markersize=8, label="Null mean"),
            mpatches.Patch(facecolor="#90A4AE", alpha=0.25, label="Null distribution"),
        ]
        ax.legend(
            handles=_leg_handles, fontsize=10, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CFD8DC"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
            logger.info(f"   Saved: {save_path}")
        plt.show()


# Create Validation Model


def create_validation_model_p4(is_classification=True):
    """
    Single CatBoost model with the main pipeline's fixed hyperparameters and
    250 iterations (1,000 in the main pipeline).

    The permutation test uses it for both the observed score and every
    permuted fit, so the test asks whether this model predicts the target
    better than chance. Refitting the full stacking ensemble 1,000 times per
    target was not computationally feasible on Colab.

    See: Ojala & Garriga (2010), JMLR; Phipson & Smyth (2010) for
    p-value computation.
    """
    from catboost import CatBoostClassifier, CatBoostRegressor

    # Use main pipeline params but reduce iterations for permutation speed.
    # All other hyperparameters (depth, lr, l2) are preserved exactly.
    params = {**ctx._CB_PARAMS, "iterations": 250, "verbose": 0}
    if is_classification:
        return CatBoostClassifier(**params)
    else:
        return CatBoostRegressor(**params)


# Cache system: permutation

# Batch-aware wrapper: Permutation Testing & Null Distributions
# When _batch_archive holds multiple targets with archived state, this
# wrapper iterates over each, restoring globals before each pass.
# In single-target mode the validation runs exactly once (no change).


def _run_for_current_target():
    """Run this stage's validation for the current target."""
    # Validation stage gate (reads the main stage settings)
    _skip_perm = not ctx.get("run_validation_cells", True)
    _VAL_N_PERMUTATIONS = ctx.n_permutations
    _VAL_N_OUTER = ctx.n_outer_folds

    # Dependency guard
    if "make_run_key" not in ctx:
        logger.info("\nWarning: setup has not been run in this session.")
        logger.info("   Run setup and the main stage first.")
        _RUN_15 = False
    elif _skip_perm:
        logger.info("permutation skipped: main.run_validation_cells is False")
        _RUN_15 = False
    else:
        _CTAG_15 = "permutation"
        _CKEY_15 = make_run_key({"cell": _CTAG_15})
        _CACHED_15 = cache_load(_CTAG_15, _CKEY_15)
        if _CACHED_15:
            logger.info(f"  {_CTAG_15} loaded from cache -- skipping execution.")
            _RUN_15 = False
        else:
            _RS["section"] = "permutation"
            _done_folds_15 = fold_load_all(_CKEY_15, _CTAG_15)
            _RUN_15 = True  # moved inside else block

    if _RUN_15:
        # EXECUTION: Run permutation test

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
                # Within-categories mode: Permutation test per top-K categories

                # Permutation testing is expensive: limit to top 5 + bottom 3
                _TOP_K = min(5, len(_top_cats_ordered))
                _BOT_K = min(3, max(0, len(_top_cats_ordered) - _TOP_K))
                _cats_to_run = _top_cats_ordered[:_TOP_K]
                if _BOT_K > 0:
                    _cats_to_run += _top_cats_ordered[-_BOT_K:]

                _section_hdr(
                    "Permutation Testing (Within-Categories)",
                    subtitle=f"Testing {len(_cats_to_run)} categories (top {_TOP_K} + bottom {_BOT_K})  ·  "
                    f"{_VAL_N_PERMUTATIONS} permutations per category",
                    color=_P4_BLUE,
                )

                task_type = detect_task_type(list(ctx.exported_within_dict.values())[0][2])
                is_classification = task_type == "classification"
                _metric_name = "AUC" if is_classification else "R²"

                ctx._perm_results = []

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

                    try:
                        _perm_reporter = PermutationTester(
                            _X_full.copy(),
                            _y_full.copy(),
                            task_type="auto",
                            n_permutations=_VAL_N_PERMUTATIONS,
                            cv_folds=_VAL_N_OUTER,
                        )
                        _result = _perm_reporter.run_permutation_test(
                            model_factory=lambda: create_validation_model_p4(is_classification=is_classification),
                            results_dir="permutation_results",
                        )

                        _obs = _result.get("observed_score", np.nan) if _result else np.nan
                        _p = _result.get("p_value", np.nan) if _result else np.nan
                        _null_mean = np.mean(_result.get("null_scores", [np.nan])) if _result else np.nan

                        ctx._perm_results.append(
                            {
                                "Category": _cat_name,
                                f"Observed {_metric_name}": _obs,
                                f"Null Mean {_metric_name}": _null_mean,
                                "p-value": _p,
                                "Significant (p<0.05)": "Yes" if _p < 0.05 else "No",
                                "null_scores": list(_result.get("null_scores", [])) if _result else [],
                            }
                        )
                        logger.info(f"    Observed: {_obs:.4f}  |  Null mean: {_null_mean:.4f}  |  p = {_p:.4f}")

                    except Exception as _e:
                        logger.info(f"    Warning: Failed: {_e}")
                    finally:
                        try:
                            del _perm_reporter
                        except NameError:
                            pass
                        _pipeline_cleanup(label=f"P4 {_cat_name}", verbose=False)

                if ctx._perm_results:
                    _section_hdr(
                        "Permutation Testing -- Summary",
                        subtitle=f"{ctx.target_options} (T{ctx.tp_option})  ·  {len(ctx._perm_results)} categories tested",
                        color=_P4_GREEN if all(r["p-value"] < 0.05 for r in ctx._perm_results) else _P4_ORANGE,
                    )

                    ctx._perm_df = pd.DataFrame(ctx._perm_results)
                    _fmt_table(
                        ctx._perm_df,
                        caption=f"Permutation Tests -- {ctx.target_options} (T{ctx.tp_option})",
                        gradient_col=f"Observed {_metric_name}",
                        highlight_col="Significant (p<0.05)",
                    )

                    _n_sig = sum(1 for r in ctx._perm_results if r["p-value"] < 0.05)

                    # Summary KPI panel
                    _kpi_panel(
                        [
                            ("Categories Tested", str(len(ctx._perm_results)), None),
                            (
                                "Significant (p<0.05)",
                                f"{_n_sig}/{len(ctx._perm_results)}",
                                f"{_n_sig / len(ctx._perm_results) * 100:.0f}%",
                            ),
                            ("Permutations Each", str(_VAL_N_PERMUTATIONS), None),
                        ],
                        title="Within-Categories Permutation Summary",
                        color=_P4_GREEN if _n_sig == len(ctx._perm_results) else _P4_ORANGE,
                    )

                    # Forest plot
                    try:
                        PermutationTester.__new__(PermutationTester).plot_within_categories_forest(
                            ctx._perm_results,
                            metric_name=_metric_name,
                            target_lbl=ctx.target_options,
                            tp_lbl=str(ctx.tp_option),
                            save_path="permutation_results/within_categories_forest.png",
                        )
                    except Exception as _fe:
                        logger.info(f"  Warning: Forest plot error: {_fe}")

                    os.makedirs("permutation_results", exist_ok=True)
                    ctx._perm_df.to_csv("permutation_results/within_categories_perm.csv", index=False)

            elif "X_train" in ctx and "y_train" in ctx:
                # Across-categories mode: Original single-pool permutation test

                _section_hdr(
                    "Permutation Testing (Across-Categories)",
                    subtitle=f"{_VAL_N_PERMUTATIONS} permutations  ·  {_VAL_N_OUTER}-fold CV",
                    color=_P4_BLUE,
                )

                task_type = detect_task_type(ctx.y_train)
                is_classification = task_type == "classification"
                logger.info(f"   Task type: {task_type}")

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

                ctx.perm_reporter = PermutationTester(
                    X_full.copy(),
                    y_full.copy(),
                    task_type="auto",
                    n_permutations=_VAL_N_PERMUTATIONS,
                    cv_folds=_VAL_N_OUTER,
                )

                result = ctx.perm_reporter.run_permutation_test(
                    model_factory=lambda: create_validation_model_p4(is_classification=is_classification),
                    results_dir="permutation_results",
                )
                ctx.perm_reporter.generate_report(title="Permutation Significance Test -- CatBoost")

            else:
                logger.info("\nError: X_train or y_train not found")

        except Exception as e:
            logger.info(f"\nError in [permutation]: {e}")
            import traceback

            traceback.print_exc()

        # Completion banner
        display(
            HTML(
                f'<div style="font-family:sans-serif;max-width:750px;border:2px solid {_P4_GREEN};'
                f'border-radius:8px;overflow:hidden;margin:16px 0;">'
                f'<div style="background:{_P4_GREEN};color:white;padding:10px 18px;'
                f'font-size:13px;font-weight:700;">Permutation Testing -- Complete</div></div>'
            )
        )

        # [Cache + report save]
        _RS["section"] = "uncategorized"
        try:
            _pipeline_cleanup(label="P4 complete")

            cache_save(
                {"reporter": locals().get("reporter") or locals().get("permutation_reporter")},
                cell_tag=_CTAG_15,
                key=_CKEY_15,
                label=f"{ctx.target_options}_tp{ctx.tp_option}_[permutation] Permutation Testing",
            )
            report_save_state(make_run_key())
        except Exception as _ce_15:
            logger.info(f"[Cache] Could not save permutation: {_ce_15}")


def run() -> None:
    """Run this stage (notebook cell 15)."""
    global copy

    # Batch dispatch
    _is_bv_18 = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_18:
        try:
            _cbv_18 = cache_load("main_runner")
            if (
                _cbv_18
                and isinstance(_cbv_18, dict)
                and _cbv_18.get("_batch_archive")
                and len(_cbv_18["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_18["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_18["_batch_archive"]
                _is_bv_18 = True
        except Exception:
            pass

    if _is_bv_18:
        _vgs_18 = {
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
            f"  BATCH -- Permutation Testing & Null Distributions: processing {len(ctx._batch_archive)} targets"
        )
        logger.info("=" * 72)

        for _vi_18, (_vk_18, _vd_18) in enumerate(ctx._batch_archive.items()):
            _vl_18 = _vd_18.get("target", _vk_18)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_18 + 1}/{len(ctx._batch_archive)}] Target: {_vl_18}")
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
                if _vd_18.get(_gk) is not None:
                    ctx[_gk] = _vd_18[_gk]
            ctx["target_options"] = _vl_18
            if "timepoint" in _vd_18:
                ctx["tp_option"] = _vd_18["timepoint"]
            _run_for_current_target()
            # Archive validation reporters back into _batch_archive
            import copy

            if _vk_18 in ctx._batch_archive:
                ctx._batch_archive[_vk_18]["perm_reporter"] = ctx.get("perm_reporter")
                ctx._batch_archive[_vk_18]["_perm_results"] = copy.deepcopy(ctx.get("_perm_results"))
                ctx._batch_archive[_vk_18]["_perm_df"] = copy.deepcopy(ctx.get("_perm_df"))

        for _k, _v in _vgs_18.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH COMPLETE -- Permutation Testing & Null Distributions: {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)
    else:
        _run_for_current_target()
