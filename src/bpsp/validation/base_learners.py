"""Base learners versus the stacked ensemble on held-out data, for all archived models (the `base_learners` stage, Supplement S5.3).

Held-out R² (regression) or AUC (classification) of every base learner and
of the stacked ensemble, for every archived model (within- and
across-category). Nothing is refit or re-scored: evaluate_models() already
scored each base learner and the meta-learner on the held-out test set for
every category it ran and stored them in metrics_dict_full; this stage
collects them from _batch_archive (or from the current run in
single-target mode).
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from bpsp.context import ctx

logger = logging.getLogger(__name__)

_BLP_NAMES = {"catboost": "CatBoost", "xgboost": "XGBoost", "random_forest": "Random Forest", "tabpfn": "TabPFN"}


def _blp_entries(selection):
    arch = ctx.get("_batch_archive")
    if isinstance(arch, dict) and len(arch) > 0:
        entries = {k: v for k, v in arch.items() if isinstance(v, dict)}
    elif isinstance(ctx.get("metrics_dict_full"), dict):
        g = ctx
        entries = {
            f"{g.get('target_options')}__tp{g.get('tp_option')}": dict(
                metrics_dict_full=g["metrics_dict_full"],
                split=g.get("split"),
                target=g.get("target_options"),
                timepoint=g.get("tp_option"),
            )
        }
    else:
        entries = {}
    if selection and selection.strip().lower() != "all":
        wanted = [s.strip() for s in selection.split(",") if s.strip()]
        entries = {k: v for k, v in entries.items() if k in wanted or str(v.get("target", k)) in wanted}
    return entries


_BLP_INK, _BLP_INK2, _BLP_MUTED, _BLP_GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
_BLP_COLORS = {"CatBoost": "#2a78d6", "XGBoost": "#eb6834", "Random Forest": "#1baf7a", "TabPFN": "#6250d6"}


def _blp_target_label(tgt):
    names = ctx.get("TARGET_DISPLAY_NAMES")
    return names.get(tgt, tgt) if isinstance(names, dict) else tgt


def _blp_plot(df, learners, out_dir):
    """Dot plot per model (base learners as colored dots, ensemble as a black diamond); for many
    models (within-category runs), a scatter of the ensemble against the best single learner."""
    from matplotlib.lines import Line2D

    for met in df["Metric"].unique():
        g = df[df["Metric"] == met].copy()
        g["label"] = [
            f"{_blp_target_label(t)}, T{tp}" + ("" if c == "All features" else f"  |  {c}")
            for t, tp, c in zip(g["Target"], g["TP"], g["Category"])
        ]
        if len(g) <= 30:
            g = g.iloc[::-1]
            fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(g) + 1.6))
            yy = np.arange(len(g))
            for i, (_, r) in enumerate(g.iterrows()):
                vals = [r[l] for l in learners if pd.notna(r.get(l))] + [r["Stacked ensemble"]]
                ax.plot([min(vals), max(vals)], [i, i], color=_BLP_GRID, lw=2.2, zorder=1, solid_capstyle="round")
                for l in learners:
                    if pd.notna(r.get(l)):
                        ax.plot(
                            r[l], i, "o", ms=6.5, color=_BLP_COLORS.get(l, _BLP_MUTED), mec="white", mew=1, zorder=2
                        )
                ax.plot(r["Stacked ensemble"], i, "D", ms=6.5, color=_BLP_INK, mec="white", mew=1, zorder=3)
            ax.set_yticks(yy)
            ax.set_yticklabels(g["label"], fontsize=8.5, color=_BLP_INK2)
            handles = [
                Line2D([], [], ls="", marker="o", ms=6.5, color=_BLP_COLORS.get(l, _BLP_MUTED), mec="white", label=l)
                for l in learners
            ] + [Line2D([], [], ls="", marker="D", ms=6.5, color=_BLP_INK, mec="white", label="Stacked ensemble")]
            ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(0, 1.0),
                ncol=len(handles),
                frameon=False,
                fontsize=8,
                labelcolor=_BLP_INK2,
                handletextpad=0.3,
                columnspacing=1.2,
            )
            ax.set_xlabel(f"Held-out {met}", fontsize=9, color=_BLP_INK2)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
            ax.grid(axis="x", color=_BLP_GRID, lw=0.8)
        else:
            fig, ax = plt.subplots(figsize=(5.6, 5.2))
            cols = {"within_categories": "#2a78d6", "across_categories": "#eb6834"}
            for sp, gg in g.groupby("Split"):
                ax.plot(
                    gg["Best base value"],
                    gg["Stacked ensemble"],
                    "o",
                    ms=5,
                    alpha=0.75,
                    color=cols.get(sp, _BLP_MUTED),
                    mec="white",
                    mew=0.6,
                    label=str(sp).replace("_", " "),
                )
            lo = min(g["Best base value"].min(), g["Stacked ensemble"].min())
            hi = max(g["Best base value"].max(), g["Stacked ensemble"].max())
            ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=_BLP_MUTED, lw=1)
            ax.text(
                hi,
                hi,
                "equal ",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=_BLP_MUTED,
                rotation=45,
                rotation_mode="anchor",
                transform_rotates_text=True,
            )
            ax.set_xlabel(f"Best single base learner, held-out {met}", fontsize=9, color=_BLP_INK2)
            ax.set_ylabel(f"Stacked ensemble, held-out {met}", fontsize=9, color=_BLP_INK2)
            ax.legend(frameon=False, fontsize=8, labelcolor=_BLP_INK2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(False)
            ax.grid(color=_BLP_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_BLP_MUTED)
        ax.tick_params(colors=_BLP_INK2, labelsize=8)
        ax.set_title(
            f"Base learners and stacked ensemble, held-out {met}",
            loc="left",
            fontsize=11,
            fontweight="bold",
            color=_BLP_INK,
            pad=26 if len(g) <= 30 else 10,
        )
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"base_learners_vs_ensemble_{'auc' if met == 'AUC' else 'r2'}.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.show()


def _run_base_learner_table():
    out_dir = os.path.join(os.getcwd(), "base_learner_performance")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for key, e in _blp_entries(ctx.blp_targets).items():
        mdf = e.get("metrics_dict_full")
        if not isinstance(mdf, dict) or not mdf.get("metrics"):
            logger.info(f"  [{key}] skipped: no metrics_dict_full archived")
            continue
        tp = e.get("timepoint", str(key).split("__tp")[-1] if "__tp" in str(key) else ctx.get("tp_option", "?"))
        cats = list(mdf.get("category", [])) or ["All"] * len(mdf["metrics"])
        for cat, md in zip(cats, mdf["metrics"]):
            if not isinstance(md, dict):
                continue
            meta = md.get("meta_model") or md.get("ensemble") or {}
            mk = "r2" if "r2" in meta else ("auc" if "auc" in meta else None)
            if mk is None:
                continue
            base = {
                _BLP_NAMES.get(n, n): float(v[mk])
                for n, v in md.items()
                if n not in ("meta_model", "ensemble") and isinstance(v, dict) and mk in v
            }
            if not base:
                continue
            best = max(base, key=base.get)
            rows.append(
                dict(
                    Target=e.get("target", key),
                    TP=tp,
                    Split=e.get("split", "?"),
                    Category=cat if cat not in (None, "") else "All features",
                    Metric="R²" if mk == "r2" else "AUC",
                    **base,
                    **{
                        "Stacked ensemble": float(meta[mk]),
                        "Best base learner": best,
                        "Best base value": base[best],
                        "Ensemble gain": float(meta[mk]) - base[best],
                    },
                )
            )
    if not rows:
        logger.info("No archived base-learner metrics found. Run the Main Analysis Runner for the targets first.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)
    learners = [c for c in ["CatBoost", "XGBoost", "Random Forest", "TabPFN"] if c in df.columns]
    learners += [
        c
        for c in df.columns
        if c not in learners
        and c
        not in [
            "Target",
            "TP",
            "Split",
            "Category",
            "Metric",
            "Stacked ensemble",
            "Best base learner",
            "Best base value",
            "Ensemble gain",
        ]
    ]
    df = df[
        ["Target", "TP", "Split", "Category", "Metric"]
        + learners
        + ["Stacked ensemble", "Best base learner", "Best base value", "Ensemble gain"]
    ]
    df.to_csv(os.path.join(out_dir, "base_learner_vs_ensemble_all_models.csv"), index=False)

    within_lbl = f"Ensemble within {ctx.blp_gain_tolerance} or better (%)"
    summ = []
    for (tgt, tp, sp, met), g in df.groupby(["Target", "TP", "Split", "Metric"], sort=False):
        r = dict(Target=tgt, TP=tp, Split=sp, Metric=met, Models=len(g))
        for l in learners:
            r[f"Mean {l}"] = g[l].mean()
        r["Mean ensemble"] = g["Stacked ensemble"].mean()
        r["Ensemble ≥ best base (%)"] = 100 * (g["Ensemble gain"] >= 0).mean()
        r[within_lbl] = 100 * (g["Ensemble gain"] >= -ctx.blp_gain_tolerance).mean()
        r["Median gain"] = g["Ensemble gain"].median()
        r["Most often best base"] = g["Best base learner"].mode().iat[0]
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out_dir, "base_learner_vs_ensemble_by_target.csv"), index=False)
    num = [c for c in summ.columns if c.startswith("Mean") or c == "Median gain"]
    logger.info(
        f"\n{'=' * 78}\n  Held-out performance: each base learner vs the stacked ensemble (per target)\n{'=' * 78}"
    )
    display(
        summ.style.format(
            {
                **{c: "{:.3f}" for c in num},
                "Ensemble ≥ best base (%)": "{:.0f}",
                within_lbl: "{:.0f}",
                "Models": "{:.0f}",
            }
        ).hide(axis="index")
    )

    logger.info(f"\n{'=' * 78}\n  Across all {len(df)} archived models\n{'=' * 78}")
    for met, g in df.groupby("Metric"):
        logger.info(
            f"  {met}: ensemble ≥ best base learner in {100 * (g['Ensemble gain'] >= 0).mean():.0f}% of models; "
            f"within {ctx.blp_gain_tolerance} or better in {100 * (g['Ensemble gain'] >= -ctx.blp_gain_tolerance).mean():.0f}%; "
            f"median gain {g['Ensemble gain'].median():+.4f} "
            f"(IQR {g['Ensemble gain'].quantile(0.25):+.4f} to {g['Ensemble gain'].quantile(0.75):+.4f})"
        )
        logger.info(
            "       best single learner: "
            + ", ".join(f"{k} {v}" for k, v in g["Best base learner"].value_counts().items())
        )

    if ctx.blp_make_plots:
        _blp_plot(df, learners, out_dir)
    logger.info(f"\nSaved to {out_dir}/")
    return df, summ


def run() -> None:
    """Run this stage (notebook cell 23)."""

    ctx.base_learner_all_models_df, ctx.base_learner_by_target_df = _run_base_learner_table()
