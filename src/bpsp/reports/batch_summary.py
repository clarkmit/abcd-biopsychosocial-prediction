"""Batch summary for classification runs (the `batch_summary` stage).

Summary of the most recent classification batch, read from _batch_archive:
formatted AUC, F1, precision and recall tables with colormapped cells
across all targets in the batch, for multi-target comparisons such as
depression worsening thresholds and severity percentiles.
"""

import logging

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from bpsp.context import ctx
from bpsp.reporting import cache_load

logger = logging.getLogger(__name__)

# CONFIG
_SAVE_FIG = True
_FIG_DPI = 300

_BG = "#F8F7F3"  # exact Sankey background (_DEP_WORSENING_BG)
_INK = "#0F172A"
_INK_MID = "#475569"
_INK_LT = "#94A3B8"
_BORDER = "#E8E6E0"  # warm-tinted border
_ROW_ALT = "#EDECEA"  # slightly darker for zebra rows

_M_CFG = {
    "accuracy": ("Accuracy", 0.50, 1.00, LinearSegmentedColormap.from_list("a", ["#FEF2F2", "#166534"])),
    "f1": ("F1", 0.50, 1.00, LinearSegmentedColormap.from_list("f", ["#FFFBEB", "#1E3A8A"])),
    "precision": ("Precision", 0.50, 1.00, LinearSegmentedColormap.from_list("p", ["#FAF5FF", "#581C87"])),
    "recall": ("Recall", 0.50, 1.00, LinearSegmentedColormap.from_list("r", ["#ECFDF5", "#92400E"])),
    "rmse": ("RMSE", None, None, LinearSegmentedColormap.from_list("rm", ["#166534", "#991B1B"])),
    "mae": ("MAE", None, None, LinearSegmentedColormap.from_list("ma", ["#166634", "#991B1B"])),
}

_TIERS = [
    (0.00, 0.65, "#FEE2E2"),
    (0.65, 0.75, "#FEF9C3"),
    (0.75, 0.85, "#DCFCE7"),
    (0.85, 1.01, "#DBEAFE"),
]

_MC = {
    "catboost": "#0072B2",
    "xgboost": "#D55E00",
    "random_forest": "#009E73",
    "tabpfn": "#CC79A7",
    "elastic_net": "#E69F00",
    "meta_model": _INK,
    "ensemble": _INK,
}


def run() -> None:
    """Run this stage (notebook cell 7)."""
    global os
    _RESULTS_DIR = ctx.get("RESULTS_DIR", "results")

    # GUARDS
    _has_batch = "_batch_archive" in ctx and isinstance(ctx._batch_archive, dict) and len(ctx._batch_archive) > 1
    if not _has_batch:
        try:
            _cached = cache_load("main_runner")
            if _cached and isinstance(_cached, dict):
                if _cached.get("_batch_archive") and len(_cached["_batch_archive"]) > 1:
                    ctx._batch_archive = _cached["_batch_archive"]
                    _has_batch = True
        except Exception:
            pass

    if not _has_batch:
        logger.info("Warning:  Batch Metric Dashboard requires batch mode results.")
    else:
        _TDN = ctx.get("TARGET_DISPLAY_NAMES", {})

        # Data extraction
        _rows = []
        for _tkey, _tdata in ctx._batch_archive.items():
            _mdf = _tdata.get("metrics_dict_full", {})
            _tr = _tdata.get("target", _tkey.split("__tp")[0] if "__tp" in _tkey else _tkey)
            _disp = _TDN.get(_tr, _TDN.get(_tkey, _tkey))

            _yt = _tdata.get("y_train")
            _clf = False
            if _yt is not None:
                try:
                    _clf = len(np.unique(_yt)) <= 10
                except Exception:
                    pass
            if not _clf:
                try:
                    _clf = not ctx.get("is_regression", lambda t: True)(_tr)
                except Exception:
                    pass

            _best, _mscores = {}, {}
            if _mdf.get("category"):
                for _cat, _mets in zip(_mdf["category"], _mdf["metrics"]):
                    if isinstance(_mets, dict):
                        for _mn, _mv in _mets.items():
                            if isinstance(_mv, dict):
                                _pk0 = "auc" if _clf else "r2"
                                _sc = _mv.get(_pk0)
                                if _sc is not None:
                                    _mscores.setdefault(_mn, []).append(_sc)
                        _mm = _mets.get("meta_model", _mets.get("ensemble", {}))
                        if isinstance(_mm, dict):
                            _pk0 = "auc" if _clf else "r2"
                            _sc = _mm.get(_pk0)
                            if _sc is not None and _sc > _best.get(_pk0, -999):
                                _best = dict(_mm)

            _np2, _nt2 = None, None
            if _yt is not None and _clf:
                try:
                    _ya = np.array(_yt)
                    _nt2 = len(_ya)
                    _np2 = int(np.sum(_ya == 1))
                except Exception:
                    pass

            _bscores = {k: float(np.mean(v)) for k, v in _mscores.items() if k not in ("meta_model", "ensemble") and v}

            _rows.append(
                {
                    "key": _tkey,
                    "display": _disp,
                    "is_clf": _clf,
                    "auc": _best.get("auc"),
                    "accuracy": _best.get("accuracy"),
                    "f1": _best.get("f1", _best.get("f1_weighted")),
                    "precision": _best.get("precision"),
                    "recall": _best.get("recall"),
                    "r2": _best.get("r2"),
                    "rmse": _best.get("rmse"),
                    "mae": _best.get("mae"),
                    "n_pos": _np2,
                    "n_total": _nt2,
                    "base_scores": _bscores,
                }
            )

        _df = pd.DataFrame(_rows)
        _all_clf = _df["is_clf"].all()
        _all_reg = (~_df["is_clf"]).all()

        _pk = "auc" if _all_clf else ("r2" if _all_reg else "auc")
        _pkl = "AUC" if _pk == "auc" else "R²"
        _df = _df.sort_values(_pk, ascending=True, na_position="first").reset_index(drop=True)
        _n = len(_df)

        _sec_keys = (
            ["accuracy", "f1", "precision", "recall"]
            if _all_clf
            else ["rmse", "mae"]
            if _all_reg
            else ["accuracy", "f1", "rmse", "mae"]
        )
        _n_sec = len(_sec_keys)

        # FIGURE: column headers only, no figure-level text at all
        _rh = 0.42
        _fh = _n * _rh + 0.90
        _fw = 13.5

        fig = plt.figure(figsize=(_fw, _fh), facecolor=_BG)

        # widened panel 0 from 3.0 → 3.4 to accommodate long target names
        _wr = [3.4, 3.4, _n_sec * 1.15, 1.20]
        gs = gridspec.GridSpec(1, 4, width_ratios=_wr, wspace=0.018, left=0.010, right=0.988, top=0.920, bottom=0.110)

        _ylim = (-0.5, _n - 0.5)

        def _zebra(ax):
            for i in range(_n):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, facecolor=_ROW_ALT, alpha=0.55, zorder=0)

        def _spine(ax, bottom=True):
            for s in ax.spines.values():
                s.set_visible(False)
            if bottom:
                ax.spines["bottom"].set_visible(True)
                ax.spines["bottom"].set_color(_BORDER)
                ax.spines["bottom"].set_linewidth(0.4)

        # PANEL 0: Target labels
        ax_n = fig.add_subplot(gs[0, 0], facecolor=_BG)
        _zebra(ax_n)

        for i, row in _df.iterrows():
            ax_n.text(
                0.02,
                i,
                row["display"],
                va="center",
                ha="left",
                fontsize=11,
                fontweight="600",
                color=_INK,
                transform=ax_n.get_yaxis_transform(),
                clip_on=False,
            )

        ax_n.set_xlim(0, 1)
        ax_n.set_ylim(*_ylim)
        ax_n.set_yticks([])
        ax_n.set_xticks([])
        ax_n.set_title("Target", fontsize=10, fontweight="600", color=_INK, pad=5, loc="left")
        _spine(ax_n, bottom=False)

        # PANEL 1: AUC bar track
        ax_a = fig.add_subplot(gs[0, 1], facecolor=_BG)
        _zebra(ax_a)

        _vals = _df[_pk].fillna(0).astype(float).values
        _dmin = float(_df[_pk].dropna().min())
        _dmax = float(_df[_pk].dropna().max())
        _ref = 0.5 if _pk == "auc" else 0.0
        _xlo = max(_ref, np.floor((_dmin - 0.03) * 20) / 20)
        _xhi = min(1.00, np.ceil((_dmax + 0.02) * 20) / 20)

        for _t0, _t1, _tc in _TIERS:
            _a = max(_t0, _xlo)
            _b = min(_t1, _xhi + 0.04)
            if _b > _a:
                ax_a.axvspan(_a, _b, facecolor=_tc, alpha=0.35, zorder=0)

        if _ref >= _xlo:
            ax_a.axvline(_ref, color=_INK_LT, linewidth=0.65, linestyle="--", alpha=0.45, zorder=1)

        for i in range(_n):
            row = _df.iloc[i]
            v = float(_vals[i])
            bs = row["base_scores"] if isinstance(row["base_scores"], dict) else {}

            ax_a.add_patch(
                FancyBboxPatch(
                    (_xlo, i - 0.155),
                    _xhi - _xlo,
                    0.31,
                    boxstyle="round,pad=0.005",
                    facecolor="#E2E8F0",
                    edgecolor="none",
                    zorder=1,
                )
            )

            _bw = max(0.0, v - _ref)
            if _bw > 0:
                ax_a.add_patch(
                    FancyBboxPatch(
                        (_ref, i - 0.155),
                        _bw,
                        0.31,
                        boxstyle="round,pad=0.005",
                        facecolor="#334155",
                        edgecolor="none",
                        alpha=0.80,
                        zorder=2,
                    )
                )

            ax_a.text(
                v + 0.003,
                i,
                f"{v:.3f}",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="600",
                color=_INK,
                path_effects=[pe.withStroke(linewidth=2.0, foreground=_BG)],
                zorder=5,
            )

            if bs:
                _sv = list(bs.values())
                ax_a.plot(
                    [min(_sv), max(_sv)],
                    [i - 0.30, i - 0.30],
                    color=_INK_LT,
                    linewidth=0.8,
                    alpha=0.4,
                    zorder=3,
                    solid_capstyle="round",
                )
                for mn, ms in bs.items():
                    ax_a.scatter(
                        ms, i - 0.30, color=_MC.get(mn, _INK_LT), s=16, zorder=4, edgecolors=_BG, linewidth=0.4
                    )

        ax_a.set_xlim(_xlo - 0.005, _xhi + 0.070)
        ax_a.set_ylim(*_ylim)
        ax_a.set_yticks([])
        ax_a.tick_params(axis="x", labelsize=8, colors=_INK_MID, pad=2)
        ax_a.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax_a.set_title(f"Meta-model {_pkl}", fontsize=10, fontweight="600", color=_INK, pad=5, loc="left")
        _spine(ax_a, bottom=True)
        ax_a.grid(axis="x", alpha=0.12, color=_BORDER, linewidth=0.35, zorder=0)

        _all_mn = set()
        for _r in _rows:
            if isinstance(_r.get("base_scores"), dict):
                _all_mn.update(_r["base_scores"].keys())
        _mlh = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_MC.get(m, "#888"),
                markersize=4.5,
                label=m.replace("_", " ").title(),
            )
            for m in _MC
            if m in _all_mn
        ]
        if _mlh:
            _leg = ax_a.legend(
                handles=_mlh,
                fontsize=7,
                loc="lower right",
                framealpha=0.95,
                edgecolor=_BORDER,
                fancybox=False,
                ncol=1,
                handletextpad=0.3,
                borderpad=0.4,
                labelspacing=0.18,
            )
            _leg.set_title("Base learners", prop={"size": 7})

        # PANEL 2: Secondary Metrics (column headers only, no panel title)
        ax_m = fig.add_subplot(gs[0, 2], facecolor=_BG)
        _zebra(ax_m)

        for _jd in range(1, _n_sec):
            ax_m.axvline(_jd - 0.5, color=_BORDER, linewidth=0.35, alpha=0.65, zorder=1)

        _mat = np.full((_n, _n_sec), np.nan)
        for _i, _row in _df.iterrows():
            for _j, _mk in enumerate(_sec_keys):
                _v = _row.get(_mk)
                if _v is not None:
                    _mat[_i, _j] = float(_v)

        for _j, _mk in enumerate(_sec_keys):
            _lbl, _vmin0, _vmax0, _cm0 = _M_CFG[_mk]
            _col = _mat[:, _j]
            _valid = _col[~np.isnan(_col)]
            if not len(_valid):
                continue

            _vn = _vmin0 if _vmin0 is not None else float(_valid.min())
            _vx = _vmax0 if _vmax0 is not None else float(_valid.max())
            if _vn >= _vx:
                _vx = _vn + 0.01
            ctx._norm = Normalize(vmin=_vn, vmax=_vx)

            for _i in range(_n):
                _v = _mat[_i, _j]
                if np.isnan(_v):
                    continue
                _fc = _cm0(ctx._norm(_v))

                ax_m.add_patch(
                    FancyBboxPatch(
                        (_j - 0.43, _i - 0.34),
                        0.86,
                        0.68,
                        boxstyle="round,pad=0.030",
                        facecolor=_fc,
                        edgecolor=_BG,
                        linewidth=1.2,
                        zorder=2,
                    )
                )

                _lum = 0.299 * _fc[0] + 0.587 * _fc[1] + 0.114 * _fc[2]
                _tc = "white" if _lum < 0.46 else _INK
                ax_m.text(
                    _j, _i, f"{_v:.3f}", ha="center", va="center", fontsize=9.5, fontweight="600", color=_tc, zorder=3
                )

        ax_m.set_xlim(-0.58, _n_sec - 0.42)
        ax_m.set_ylim(*_ylim)
        ax_m.set_yticks([])
        ax_m.set_xticks(range(_n_sec))
        # Metric names as column headers only: no panel title above them
        ax_m.set_xticklabels([_M_CFG[k][0] for k in _sec_keys], fontsize=10, fontweight="600", color=_INK)
        ax_m.xaxis.set_ticks_position("top")
        ax_m.xaxis.set_tick_params(length=0, pad=5)
        ax_m.set_title("", pad=0)  # explicitly empty
        for _sp in ax_m.spines.values():
            _sp.set_visible(False)

        # PANEL 3: Positive class %
        ax_p = fig.add_subplot(gs[0, 3], facecolor=_BG)
        _zebra(ax_p)

        if _all_clf:
            for _i, _row in _df.iterrows():
                _np3 = _row.get("n_pos")
                _nt3 = _row.get("n_total")
                if _np3 is not None and _nt3 and _nt3 > 0:
                    _pct = _np3 / _nt3
                    _vis = max(_pct, 0.022)

                    ax_p.add_patch(
                        FancyBboxPatch(
                            (0.0, _i - 0.145),
                            1.0,
                            0.29,
                            boxstyle="round,pad=0.004",
                            facecolor="#E2E8F0",
                            edgecolor="none",
                            zorder=1,
                        )
                    )
                    ax_p.add_patch(
                        FancyBboxPatch(
                            (1.0 - _vis, _i - 0.145),
                            _vis,
                            0.29,
                            boxstyle="round,pad=0.004",
                            facecolor="#2563EB",
                            edgecolor="none",
                            alpha=0.78,
                            zorder=2,
                        )
                    )

                    _lx = 1.0 - _vis / 2 if _pct > 0.12 else 1.0 - _vis - 0.04
                    _ha = "center" if _pct > 0.12 else "right"
                    _tc = "white" if _pct > 0.12 else _INK_MID
                    ax_p.text(
                        _lx, _i, f"{_pct:.1%}", ha=_ha, va="center", fontsize=8.5, fontweight="600", color=_tc, zorder=3
                    )

        ax_p.set_xlim(0, 1)
        ax_p.set_ylim(*_ylim)
        ax_p.set_yticks([])
        ax_p.set_xticks([0, 0.5, 1.0])
        ax_p.set_xticklabels(["0%", "50%", "100%"], fontsize=7.5, color=_INK_MID)
        ax_p.set_title("Pos. class %", fontsize=10, fontweight="600", color=_INK, pad=5)
        _spine(ax_p, bottom=True)

        # Save & show
        try:
            import os

            os.makedirs(_RESULTS_DIR, exist_ok=True)
            _out = f"{_RESULTS_DIR}/batch_metric_dashboard.png"
            fig.savefig(_out, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_BG)
            logger.info(f"  Saved: {_out}")
        except Exception as _e:
            logger.info(f"  Warning:  Save error: {_e}")

        plt.show()
        logger.info(f"\n   Batch Metric Dashboard -- {_n} targets rendered.")
