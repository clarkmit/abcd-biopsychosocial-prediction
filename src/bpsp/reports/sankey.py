"""Sankey diagrams of across-category feature importance (the `sankey` stage).

Sankey diagrams of across-category consensus feature importance for
configurable target and timepoint batches, read from _batch_archive.
Features shared across targets are annotated (nT), and ribbon opacity
encodes weighted ensemble importance. Style presets set the target
grouping, color scheme, layout and which batch results are used:

developmental_sweep / parent_developmental_sweep : longitudinal sweeps
pfactor_spectra / child_pfactor / parent_pfactor  : internalizing and
                                                    externalizing spectra
child_depression / parent_depression              : classification,
                                                    severity and change
sbt_long                                          : subtype sweeps
primary_targets                                   : adult 6-target panel
child_suicide / parent_suicide                    : suicidal ideation
sbt_core_triad_female / _male                     : sex-stratified
child_deltas / parent_deltas                      : change-score batches
child_academic_cognitive                          : academic/cognitive batch
child_adhd_impcomp                                : ADHD/impulsivity/
                                                    compulsivity batch
parent_social_impcomp                             : parent social/
                                                    impulsivity/compulsivity
                                                    batch
"""

import logging

from bpsp.context import ctx
from bpsp.data.preprocessing import DISPLAY_NAMES
from bpsp.reporting import _RS

logger = logging.getLogger(__name__)


# Imports and setup

import os
import pathlib as _pathlib
import warnings
from collections import defaultdict

import matplotlib
import matplotlib.cm as mcm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, hsv_to_rgb, rgb_to_hsv, to_hex, to_rgba
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path

# Style constants & color palettes

_NAVY = "#1B2A4A"
_STEEL = "#2C3E50"
_ACCENT = "#2980B9"
_LIGHT = "#ECF0F1"

# Unified title / subtitle: applied identically to every style

_SANKEY_TITLE = "Shared Predictive Structure Across Targets"
_SANKEY_SUBTITLE = "Shared Top 10 Ensemble Consensus Predictive Features  |  # Shared Across Targets (nT)"
_TITLE_COLOR = "#000000"
_SUBTITLE_COLOR = "#333333"
_TITLE_FONT = "DejaVu Sans"

# Wider horizontal gap: push right column further right for more ribbon sweep

_LEFT_X = 0.0
_RIGHT_X = 1.30

# Heatflow color configurations

HEATFLOW_THEMES = {
    "heatflow_magenta": {
        "name": "Magenta Flow",
        "cmap": "PuRd",
        "bg_color": "#FFFFFF",
        "zone_colors": ["#F3E5F5", "#E1BEE7", "#CE93D8", "#BA68C8"],
        "target_node_edge": "#6A1B9A",
        "feature_glow": "#E91E63",
        "text_color": "#4A148C",
        "header_color": "#8E24AA",
        "accent_color": "#C2185B",
    },
    "heatflow_fire": {
        "name": "Inferno Flow",
        "cmap": "inferno",
        "bg_color": "#FFFFFF",
        "zone_colors": ["#FFECB3", "#FFD54F", "#FFCA28", "#FFC107"],
        "target_node_edge": "#BF360C",
        "feature_glow": "#FF3D00",
        "text_color": "#3E2723",
        "header_color": "#DD2C00",
        "accent_color": "#FF6D00",
    },
    "heatflow_ice": {
        "name": "Arctic Flow",
        "cmap": "winter",
        "bg_color": "#FFFFFF",
        "zone_colors": ["#E1F5FE", "#B3E5FC", "#81D4FA", "#4FC3F7"],
        "target_node_edge": "#01579B",
        "feature_glow": "#03A9F4",
        "text_color": "#0D47A1",
        "header_color": "#0277BD",
        "accent_color": "#0288D1",
    },
    "heatflow_dark": {
        "name": "Midnight Flow",
        "cmap": "copper",
        "bg_color": "#FFFFFF",
        "zone_colors": ["#16213E", "#0F3460", "#1A1A2E", "#16213E"],
        "target_node_edge": "#E94560",
        "feature_glow": "#E94560",
        "text_color": "#EAEAEA",
        "header_color": "#E94560",
        "accent_color": "#FF6B6B",
        "dark_mode": True,
    },
}


def get_target_colors_for_theme(theme_key, n_targets):
    """Generate visually distinct target colors based on theme."""
    theme = HEATFLOW_THEMES[theme_key]

    base_hues = {
        "heatflow_magenta": [0.85, 0.88, 0.82, 0.9, 0.8, 0.92],
        "heatflow_fire": [0.05, 0.08, 0.02, 0.1, 0.12, 0.15],
        "heatflow_ice": [0.55, 0.58, 0.52, 0.6, 0.5, 0.62],
        "heatflow_dark": [0.05, 0.08, 0.12, 0.15, 0.18, 0.2],
    }

    hues = base_hues.get(theme_key, [0.5 + i * 0.1 for i in range(n_targets)])
    colors = []

    for i in range(n_targets):
        hue = hues[i % len(hues)]
        saturation = 0.7 if "pastel" not in theme_key else 0.4
        value = 0.85 if "dark" not in theme_key and "neon" not in theme_key else 0.9
        rgb = hsv_to_rgb((hue, saturation, value))
        colors.append(rgb)

    return colors


# WIDE-LAYOUT


def _wide_figure_dims(n_targets, n_features, landscape=None):
    """
    Width-first figure sizing.

    Width ≥ 18″ for generous ribbon sweep.  Height is driven by the
    *feature* column (which almost always has more items), with a
    comfortable per-feature multiplier so labels and ribbons stay
    readable.  Targets get extra room via equalize_vertical_envelopes.
    """
    if landscape is None:
        landscape = ctx.get("sankey_landscape", False)

    fig_w = max(18, 16 + max(n_targets - 4, 0) * 0.5)
    # Features drive height: generous spacing so ribbon connections
    # remain individually traceable even with 40-60 features.
    min_h = max(
        n_features * 0.42,  # readable per-feature
        n_targets * 0.75,  # readable per-target
        10.0,
    )
    fig_h = min_h  # let content drive it

    if landscape:
        fig_w = max(fig_w + 2, 20)
        fig_h = max(n_features * 0.34, n_targets * 0.55, 8.0)

    return fig_w, fig_h


def _equalize_vertical_envelopes(
    target_positions, feature_positions, y_top=0.95, y_bot_margin=0.05, target_span_frac=0.60
):
    """
    Features keep their natural spacing.  Targets are shifted so their
    top aligns with the feature-column top, and spread to occupy
    target_span_frac of the feature envelope: enough to look balanced
    without the extreme gaps that come from filling the entire height.
    """
    if not target_positions or not feature_positions:
        return target_positions, feature_positions

    def _extent(pos):
        return (max(p["y_top"] for p in pos.values()), min(p["y_bot"] for p in pos.values()))

    t_top, t_bot = _extent(target_positions)
    f_top, f_bot = _extent(feature_positions)
    t_span = t_top - t_bot
    f_span = f_top - f_bot

    if t_span <= 0 or f_span <= 0:
        return target_positions, feature_positions

    # Targets occupy target_span_frac of the feature span, top-aligned
    target_span = f_span * target_span_frac
    # But never shrink targets below their natural span
    target_span = max(target_span, t_span)
    # And never exceed the full feature span
    target_span = min(target_span, f_span)

    ref_top = f_top  # top-align with features
    ref_bot = ref_top - target_span

    def _rescale_targets(positions, old_top, old_bot):
        old_span = old_top - old_bot
        if old_span <= 0:
            return positions
        new_span = ref_top - ref_bot
        new = {}
        for key, p in positions.items():
            frac = (p["y_centre"] - old_bot) / old_span
            new_c = ref_bot + frac * new_span
            h = p["h"]
            new[key] = {
                "y_centre": new_c,
                "y_top": new_c + h / 2,
                "y_bot": new_c - h / 2,
                "h": h,
            }
        return new

    target_positions = _rescale_targets(target_positions, t_top, t_bot)
    return target_positions, feature_positions


# Landscape adjustment helper (adapts on top of wide defaults)


def _landscape_adjust(fig_w, fig_h, spread_target, spread_feature, n_targets, n_features, landscape=None):
    """
    Adjust figure dimensions and spread factors for landscape orientation.

    When landscape is True:
      - Width is increased by ~30% (min 18")
      - Height is compressed: uses tighter per-feature / per-target multipliers
        but still enforced at a sane minimum so labels never overlap
      - Target and feature spread_factors are reduced proportionally

    Returns (fig_w, fig_h, spread_target, spread_feature).
    """
    if landscape is None:
        landscape = ctx.get("sankey_landscape", False)
    if not landscape:
        return fig_w, fig_h, spread_target, spread_feature

    # Width: widen for more ribbon sweep
    fig_w = max(fig_w + 4, 20)

    # Height: compress but keep labels legible
    fig_h_feat = n_features * 0.32
    fig_h_tgt = n_targets * 0.55
    fig_h = max(7.5, fig_h_feat, fig_h_tgt)

    # Spread factors: compress vertical gaps proportionally
    spread_target = spread_target * 0.55
    spread_feature = spread_feature * 0.72

    return fig_w, fig_h, spread_target, spread_feature


def _landscape_fonts(landscape=None):
    """
    Return (target_fs, feature_fs_max, metric_fs, subtitle_fs) adjusted
    for landscape mode.  Portrait values are the existing defaults.
    """
    if landscape is None:
        landscape = ctx.get("sankey_landscape", False)
    if landscape:
        return 12, 10, 9.5, 8.5  # target, feat_cap, metric, subtitle
    return 14, 12, 11.5, 9.5  # portrait defaults


# Shared helpers


def _get_tdn(t):
    """Resolve display name, handling __tp archive keys gracefully."""
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    if t in _tdn:
        return _tdn[t]
    # Handle __tpN archive keys: 'sbt_core_depression__tp2' -> 'Mood/Depression (T2)'
    if "__tp" in t:
        base, tp_suffix = t.split("__tp", 1)
        base_display = _tdn.get(base, base)
        return f"{base_display} (T{tp_suffix})"
    return t


# Timepoint-to-age relabeler (ABCD canonical: T0=~9y, T1=~10y, T2=~11y, T3=~12y, T4=~13y).
# Substitutes any trailing '(TN)' suffix with '(Age {9+N})'. Returns the label unchanged
# if no TP suffix is present. Callable from any Sankey draw function that wants
# age-based x-axis labels instead of ABCD wave indices.
_ABCD_TP_AGE = {"0": 9, "1": 10, "2": 11, "3": 12, "4": 13}
_ABCD_AGE_TP = {v: k for k, v in _ABCD_TP_AGE.items()}


def _tp_suffix_to_age(label):
    """Replace trailing '(TN)' with '(Age {9+N})' using the ABCD convention.

    Examples
    --------
    'Crystallized Intelligence (T2)' -> 'Crystallized Intelligence (Age 11)'
    'Poor Academic Grades (T4)'      -> 'Poor Academic Grades (Age 13)'
    'Some Target'                     -> 'Some Target'  (no change)
    """
    import re

    m = re.search(r"\(T([0-4])\)$", str(label))
    if not m:
        return label
    tp = m.group(1)
    return str(label)[: m.start()] + f"(Age {_ABCD_TP_AGE[tp]})"


def _age_label_to_tp(label):
    """Inverse of _tp_suffix_to_age: reverse '(Age N)' back to '(TN-9)'.

    Used by metric lookups where _target_metrics was keyed on the original
    TP-format strings before relabeling.  Returns the input unchanged if
    no '(Age N)' suffix is present or the age is outside the ABCD range.
    """
    import re

    m = re.search(r"\(Age (\d+)\)$", str(label))
    if not m:
        return label
    age = int(m.group(1))
    if age not in _ABCD_AGE_TP:
        return label
    return str(label)[: m.start()] + f"(T{_ABCD_AGE_TP[age]})"


def _tp_suffix_to_year(label):
    """Replace trailing '(TN)' with '(Year N)': years-from-baseline framing.

    Use for parent self-report targets where the child-age framing of
    _tp_suffix_to_age is misleading (the ABCD age suffix refers to the child,
    not the parent rater).
    """
    import re

    m = re.search(r"\(T([0-4])\)$", str(label))
    if not m:
        return label
    return str(label)[: m.start()] + f"(Year {m.group(1)})"


def _year_label_to_tp(label):
    """Inverse of _tp_suffix_to_year: reverse '(Year N)' back to '(TN)'."""
    import re

    m = re.search(r"\(Year ([0-4])\)$", str(label))
    if not m:
        return label
    return str(label)[: m.start()] + f"(T{m.group(1)})"


def _safe_save_fig(fig, directory, basename, dpi=600, formats=("png", "pdf")):
    """Save figure to directory in multiple formats with error handling."""
    os.makedirs(directory, exist_ok=True)
    saved = []
    for fmt in formats:
        try:
            path = os.path.join(directory, f"{basename}.{fmt}")
            fig.savefig(
                path,
                dpi=dpi if fmt == "png" else 300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                backend="Agg" if fmt == "png" else None,
                metadata={"Creator": "matplotlib"} if fmt == "pdf" else {},
            )
            saved.append(path)
        except Exception as e:
            logger.info(f"  Warning:  Could not save {fmt}: {e}")
    return saved


def _get_cmap_safe(name="Set2"):
    """Get colormap compatible with old and new matplotlib."""
    try:
        return plt.colormaps[name]
    except (AttributeError, KeyError):
        return plt.cm.get_cmap(name)


def _create_custom_cmap(colors, name="custom"):
    """Create a custom colormap from a list of colors."""
    return LinearSegmentedColormap.from_list(name, colors)


def _luminance(color):
    """Relative luminance per WCAG 2.1."""
    rgba = to_rgba(color)
    rgb = np.array(rgba[:3])
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _contrast_ratio(c1, c2):
    """WCAG 2.1 contrast ratio between two colors."""
    l1, l2 = _luminance(c1), _luminance(c2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def _ensure_contrast(bg_color, text_color, min_ratio=4.5):
    """Return text_color if contrast is sufficient, else flip to black or white."""
    if _contrast_ratio(bg_color, text_color) >= min_ratio:
        return text_color
    return "#000000" if _luminance(bg_color) > 0.5 else "#FFFFFF"


def _draw_sankey_title(ax, sankey_top_features):
    """Draw subtitle at bottom. Top margin preserved for manual title overlay."""
    # Invisible spacer so bbox_inches='tight' keeps the top margin
    ax.text(0.5, 1.06, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)
    # Subtitle moved to bottom of figure
    ax.text(
        0.5,
        -0.04,
        _SANKEY_SUBTITLE,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="normal",
        color=_SUBTITLE_COLOR,
        fontfamily=_TITLE_FONT,
        transform=ax.transAxes,
    )


def _draw_sankey_title_abs(ax, x_centre, y_title, y_subtitle):
    """Subtitle at bottom. Top margin preserved for manual title overlay."""
    # Invisible spacer so bbox_inches='tight' keeps the top margin
    ax.text(x_centre, y_title, " ", ha="center", va="bottom", fontsize=17, alpha=0.0)
    # Subtitle moved to bottom of figure
    ax.text(
        0.5,
        -0.04,
        _SANKEY_SUBTITLE,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="normal",
        color=_SUBTITLE_COLOR,
        fontfamily=_TITLE_FONT,
        transform=ax.transAxes,
    )


# Sankey data extraction


def _extract_sankey_data(batch_archive, top_n=10, normalize=False):
    """Extract Sankey flow data from batch archive consensus DataFrames."""
    rows = []
    for target, archive in batch_archive.items():
        cdf = archive.get("consensus_df")
        if cdf is None:
            continue
        if hasattr(cdf, "empty") and cdf.empty:
            continue
        if not hasattr(cdf, "columns"):
            continue
        if "weighted_score" not in cdf.columns:
            logger.info(f"  Warning:  '{target}' consensus_df missing 'weighted_score' column.")
            continue
        _cdf = cdf.dropna(subset=["weighted_score"])
        if _cdf.empty:
            continue
        _cdf_sorted = _cdf.sort_values("weighted_score", ascending=False).head(top_n)
        logger.info(f"    {_get_tdn(target)}: consensus_df has {len(_cdf)} rows → took top {len(_cdf_sorted)}")

        _values = _cdf_sorted["weighted_score"].values.astype(float)
        if normalize:
            _vmin = _values.min()
            _vmax = _values.max()
            if _vmax > _vmin:
                _values = (_values - _vmin) / (_vmax - _vmin)
            else:
                _values = np.ones_like(_values)

        for (_feat_name, _row), _imp in zip(_cdf_sorted.iterrows(), _values):
            rows.append(
                {
                    "Target": _get_tdn(target),
                    "Feature": str(_feat_name),
                    "Importance": float(_imp),
                }
            )
    return rows


def _sankey_common_data(df_sankey):
    """Compute common Sankey layout data structures."""
    targets_list = df_sankey["Target"].unique().tolist()
    feat_target_count = df_sankey.groupby("Feature")["Target"].nunique()
    feat_total_imp = df_sankey.groupby("Feature")["Importance"].sum()

    feat_order = (
        pd.DataFrame({"n_targets": feat_target_count, "total_imp": feat_total_imp})
        .sort_values(["n_targets", "total_imp"], ascending=[False, False])
        .index.tolist()
    )
    features_list = feat_order
    target_totals = df_sankey.groupby("Target")["Importance"].sum()
    feature_totals = df_sankey.groupby("Feature")["Importance"].sum()

    return {
        "targets_list": targets_list,
        "features_list": features_list,
        "feat_target_count": feat_target_count,
        "feat_total_imp": feat_total_imp,
        "target_totals": target_totals,
        "feature_totals": feature_totals,
        "n_targets": len(targets_list),
        "n_features": len(features_list),
    }


def _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features):
    """Print summary stats and feature overlap matrix."""
    n_transdiag = int((feat_target_count >= 2).sum())
    n_unique = int((feat_target_count == 1).sum())
    logger.info(f"  Sankey: {n_targets} targets, {n_features} unique features")
    logger.info(f"     • {n_transdiag} transdiagnostic (≥2 targets)")
    logger.info(f"     • {n_unique} target-specific")

    feat_sets = {}
    for t in df_sankey["Target"].unique():
        feat_sets[t] = set(df_sankey[df_sankey["Target"] == t]["Feature"])
    logger.info("\n  Feature Overlap Matrix:")
    targs = list(feat_sets.keys())
    for i, t1 in enumerate(targs):
        for t2 in targs[i + 1 :]:
            overlap = feat_sets[t1] & feat_sets[t2]
            if overlap:
                logger.info(f"    {t1} ∩ {t2}: {len(overlap)} shared -- {', '.join(list(overlap)[:5])}")


def _compute_node_positions_enhanced(
    items_list, totals, global_max, pad, min_node_h=0.025, max_node_h=0.12, spread_factor=1.5
):
    """Compute vertical positions for a column of Sankey nodes."""

    def _node_height(total):
        if global_max <= 0:
            return min_node_h
        return min_node_h + (max_node_h - min_node_h) * (total / global_max)

    positions = {}
    y = 1.0
    for item in items_list:
        h = _node_height(totals[item])
        y -= h
        positions[item] = {
            "y_centre": y + h / 2,
            "h": h,
            "y_top": y + h,
            "y_bot": y,
        }
        y -= pad * spread_factor
    return positions


def _draw_bezier_ribbon_enhanced(
    ax,
    left_x,
    node_w,
    right_x,
    y_l_top,
    y_l_bot,
    y_r_top,
    y_r_bot,
    color_rgba,
    zorder=1,
    edge_color="none",
    edge_lw=0,
    gradient_effect=False,
    glow_layers=0,
    right_node_w=None,
):
    """Draw a single Bézier ribbon between left and right ports."""
    _rnw = right_node_w if right_node_w is not None else node_w
    cx = (left_x + node_w + right_x - _rnw) / 2
    verts = [
        (left_x + node_w, y_l_top),
        (cx, y_l_top),
        (cx, y_r_top),
        (right_x - _rnw, y_r_top),
        (right_x - _rnw, y_r_bot),
        (cx, y_r_bot),
        (cx, y_l_bot),
        (left_x + node_w, y_l_bot),
        (left_x + node_w, y_l_top),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]

    if glow_layers > 0:
        for i in range(glow_layers, 0, -1):
            glow_alpha = color_rgba[3] * (0.22 / i)
            glow_patch = matplotlib.patches.PathPatch(
                Path(verts, codes), facecolor=color_rgba[:3] + (glow_alpha,), edgecolor="none", lw=0, zorder=zorder - 1
            )
            ax.add_patch(glow_patch)

    patch = matplotlib.patches.PathPatch(
        Path(verts, codes), facecolor=color_rgba, edgecolor=edge_color, lw=edge_lw, zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def _iterate_ribbon_ports(
    df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
):
    """Yield ribbon geometry for each flow row."""
    flows = df_sankey.sort_values("Importance", ascending=False)
    target_port_offset = {t: 0.0 for t in targets_list}
    feature_port_offset = {f: 0.0 for f in features_list}

    for t in targets_list:
        sub = flows[flows["Target"] == t].sort_values("Importance", ascending=False)
        total = sub["Importance"].sum()
        for _, row in sub.iterrows():
            frac = row["Importance"] / total if total > 0 else 0
            ribbon_h_left = target_positions[t]["h"] * frac
            f = row["Feature"]
            f_total = feature_totals[f]
            ribbon_h_right = feature_positions[f]["h"] * (row["Importance"] / f_total) if f_total > 0 else 0

            y_l_top = target_positions[t]["y_top"] - target_port_offset[t]
            y_l_bot = y_l_top - ribbon_h_left
            y_r_top = feature_positions[f]["y_top"] - feature_port_offset[f]
            y_r_bot = y_r_top - ribbon_h_right
            target_port_offset[t] += ribbon_h_left
            feature_port_offset[f] += ribbon_h_right

            yield {
                "target": t,
                "feature": f,
                "importance": row["Importance"],
                "y_l_top": y_l_top,
                "y_l_bot": y_l_bot,
                "y_r_top": y_r_top,
                "y_r_bot": y_r_bot,
                "ribbon_h_left": ribbon_h_left,
                "ribbon_h_right": ribbon_h_right,
            }


# Heatflow style


def _draw_sankey_heatflow_enhanced(
    df_sankey, sankey_top_features, save_dir=None, theme_key="heatflow_fire", show_colorbar=True
):
    """Heatflow Sankey without background zones or target totals."""
    theme = HEATFLOW_THEMES[theme_key]
    is_dark = theme.get("dark_mode", False)

    _cd = _sankey_common_data(df_sankey)
    targets_list = _cd["targets_list"]
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}

    cmap_heat = _get_cmap_safe(theme["cmap"])
    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)
    _sm = mcm.ScalarMappable(norm=_norm, cmap=cmap_heat)

    target_colors = {}
    target_rgb_list = get_target_colors_for_theme(theme_key, n_targets)
    for i, t in enumerate(targets_list):
        target_colors[t] = target_rgb_list[i]

    _pad = max(0.015, min(0.035, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.028

    fig_w, fig_h = _wide_figure_dims(n_targets, n_features)
    _spread_t = 1.6
    _spread_f = 1.5
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.025, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )

    # Equalize vertical envelopes so both columns span the same range
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(theme["bg_color"])
    ax.set_facecolor(theme["bg_color"])

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04

    ax.set_xlim(-0.50, right_x + 0.65)
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _ribbon_c = _sm.to_rgba(rb["importance"])[:3]
        _alpha = 0.30 + 0.60 * _norm(rb["importance"])

        if _norm(rb["importance"]) > 0.7:
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                rb["y_l_top"] + 0.003,
                rb["y_l_bot"] - 0.003,
                rb["y_r_top"] + 0.003,
                rb["y_r_bot"] - 0.003,
                color_rgba=_ribbon_c + (_alpha * 0.3,),
                zorder=1,
            )

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            rb["y_l_top"],
            rb["y_l_bot"],
            rb["y_r_top"],
            rb["y_r_bot"],
            color_rgba=_ribbon_c + (_alpha,),
            zorder=2,
            edge_color=_ribbon_c + (min(_alpha + 0.08, 0.8),),
            edge_lw=0.3,
        )

    for t, pos in target_positions.items():
        shadow_offset = 0.004
        ax.add_patch(
            FancyBboxPatch(
                (left_x - node_w + shadow_offset, pos["y_bot"] - shadow_offset),
                node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.006",
                facecolor="#000000" if not is_dark else "#333333",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.2,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (left_x - node_w, pos["y_bot"]),
                node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.006",
                facecolor=target_colors[t],
                edgecolor=theme["target_node_edge"],
                linewidth=1.5,
                zorder=3,
            )
        )

        _tm = _target_metrics.get(t) or _target_metrics.get(_get_tdn(t))
        _label_x = left_x - node_w - 0.020

        if _tm is not None:
            _val, _lbl = _tm
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.013,
                t,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=theme["text_color"],
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=theme["bg_color"])])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.015,
                f"{_lbl} = {_val:.3f}",
                ha="right",
                va="center",
                fontsize=_lf[2],
                fontweight="bold",
                color=theme["accent_color"],
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=theme["bg_color"])])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                t,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=theme["text_color"],
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=theme["bg_color"])])

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for f, pos in feature_positions.items():
        n_tgt = feat_target_count[f]
        _avg_imp = feature_totals[f] / max(n_tgt, 1)
        _nc = _sm.to_rgba(_avg_imp)[:3]

        if n_tgt >= 2:
            glow_padding = 0.005
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - node_w - glow_padding, pos["y_bot"] - glow_padding),
                    node_w * 2 + glow_padding * 2,
                    pos["h"] + glow_padding * 2,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=theme["feature_glow"],
                    linewidth=2.5,
                    zorder=2,
                    alpha=0.6,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - node_w, pos["y_bot"]),
                node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.006",
                facecolor=_nc,
                edgecolor="#444444" if not is_dark else "#888888",
                linewidth=1.0,
                zorder=3,
            )
        )

        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        label_text = f"{f}{badge}"

        txt = ax.text(
            right_x + node_w + 0.020,
            pos["y_centre"],
            label_text,
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color=theme["text_color"],
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=theme["bg_color"])])

    _title_c = theme["text_color"] if not is_dark else "#EAEAEA"

    _title_y = 1.04
    ax.text(0.50, _title_y, " ", ha="center", va="bottom", fontsize=15, transform=ax.transAxes, alpha=0.0)
    # Subtitle moved to bottom
    ax.text(
        0.50,
        -0.015,
        f"Top {sankey_top_features} Consensus Features per Target  \u00b7  "
        "Shared Features Annotated (nT)  \u00b7  "
        "R\u00b2 from Across-Category Analyses\n"
        "Ribbon Opacity \u221d Weighted Ensemble Importance  \u00b7  "
        "Circular Features were Removed and Not Visible",
        ha="center",
        va="top",
        fontsize=_lf[3],
        fontweight="bold",
        color=_title_c,
        fontfamily=_TITLE_FONT,
        linespacing=1.4,
        transform=ax.transAxes,
    )

    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    _hdr_text_c = _ensure_contrast(theme["zone_colors"][0], theme["header_color"])
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=_hdr_text_c,
        fontfamily=_TITLE_FONT,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=theme["zone_colors"][0],
            edgecolor=theme["target_node_edge"],
            alpha=0.75,
            linewidth=0.7,
        ),
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=_hdr_text_c,
        fontfamily=_TITLE_FONT,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=theme["zone_colors"][0],
            edgecolor=theme["feature_glow"],
            alpha=0.75,
            linewidth=0.7,
        ),
    )

    if show_colorbar:
        _cbar_ax = fig.add_axes([0.90, 0.20, 0.015, 0.50])
        _cb = plt.colorbar(_sm, cax=_cbar_ax)
        _cb.set_label("Consensus Importance", fontsize=10, color=theme["text_color"])
        _cb.ax.tick_params(labelsize=8, colors=theme["text_color"])
        if is_dark:
            _cbar_ax.set_facecolor("#2A2A2A")

    plt.tight_layout(rect=[0, 0, 0.88, 0.98])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")

    saved = _safe_save_fig(fig, save_dir, f"sankey_{theme_key}", dpi=600)
    plt.show()
    for p in saved:
        logger.info(f"  Saved: {p}")

    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    return fig


# Additional unique visual styles


def _extract_target_metrics(batch_archive):
    """
    Extract best ensemble metric (R² or AUC) per target from _batch_archive.
    Returns dict {display_name: (metric_value, metric_label)}.
    """
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    result = {}
    for raw_key, archive in batch_archive.items():
        mdf = archive.get("metrics_dict_full", {})
        cats = mdf.get("category", [])
        mets = mdf.get("metrics", [])
        if not cats or not mets:
            continue
        best_score, best_label = None, "R\u00b2"
        for _cat, _mdict in zip(cats, mets):
            if not isinstance(_mdict, dict):
                continue
            _mm = _mdict.get("meta_model", _mdict.get("ensemble", {}))
            if not isinstance(_mm, dict):
                continue
            _r2 = _mm.get("r2", None)
            _auc = _mm.get("auc", _mm.get("roc_auc", None))
            if _r2 is not None:
                if best_score is None or _r2 > best_score:
                    best_score, best_label = _r2, "R\u00b2"
            elif _auc is not None:
                if best_score is None or _auc > best_score:
                    best_score, best_label = _auc, "AUC"
        if best_score is not None:
            display_name = _tdn.get(raw_key, raw_key)
            result[display_name] = (best_score, best_label)
            # Also store under _get_tdn for __tp archive keys
            _tdn_name = _get_tdn(raw_key)
            if _tdn_name != display_name:
                result[_tdn_name] = (best_score, best_label)
    return result


_PFACTOR_DOMAIN_COLORS = {
    "internal_D_p": (0.55, 0.36, 0.95),
    "external_D_p": (0.45, 0.75, 0.20),
    "parent_internal_D_p": (0.48, 0.10, 0.48),
    "parent_external_D_p": (0.08, 0.48, 0.38),
    "dep_increase_2sd": (0.12, 0.56, 1.00),
    "delta_depress_D_p": (0.69, 0.77, 0.87),
    "parent_dep_increase_2sd": (0.28, 0.24, 0.55),
    "delta_parent_depress_D_p": (0.39, 0.58, 0.93),
    "Recent Clinical Change Child Depression Syndrome \u2265 2 SD": (0.12, 0.56, 1.00),
    "Change in Child Dimensional Depression Syndrome": (0.69, 0.77, 0.87),
    "Recent Clinical Change Parent Depression Syndrome \u2265 2 SD": (0.28, 0.24, 0.55),
    "Change in Parent Dimensional Depression Syndrome": (0.39, 0.58, 0.93),
    "Internalizing": (0.55, 0.36, 0.95),
    "Externalizing": (0.45, 0.75, 0.20),
    "Internalizing Spectra": (0.55, 0.36, 0.95),
    "Externalizing Spectra": (0.45, 0.75, 0.20),
    "Child Internalizing Spectra": (0.55, 0.36, 0.95),
    "Child Externalizing Spectra": (0.45, 0.75, 0.20),
    "Par. Internalizing": (0.48, 0.10, 0.48),
    "Par. Externalizing": (0.08, 0.48, 0.38),
    "Parent Internalizing Spectra": (0.48, 0.10, 0.48),
    "Parent Externalizing Spectra": (0.08, 0.48, 0.38),
    # P-Factor Spectra: Suicide & Academic
    "suicidal_p": (0.04, 0.04, 0.24),  # very dark blue
    "parent_suicidal_thoughts_p": (0.04, 0.04, 0.24),  # very dark blue (match child)
    "bad_grades": (0.45, 0.45, 0.45),  # gray
    "Suicidal Ideation": (0.04, 0.04, 0.24),
    "Parent Suicidal Thoughts": (0.04, 0.04, 0.24),
    "Academic Performance": (0.45, 0.45, 0.45),
}


def _pfactor_target_color(target_name):
    if target_name in _PFACTOR_DOMAIN_COLORS:
        return _PFACTOR_DOMAIN_COLORS[target_name]
    tl = target_name.lower()
    if "par" in tl and "internal" in tl:
        return _PFACTOR_DOMAIN_COLORS["parent_internal_D_p"]
    if "par" in tl and "external" in tl:
        return _PFACTOR_DOMAIN_COLORS["parent_external_D_p"]
    if "internal" in tl:
        return _PFACTOR_DOMAIN_COLORS["internal_D_p"]
    if "external" in tl:
        return _PFACTOR_DOMAIN_COLORS["external_D_p"]
    if "suicid" in tl:
        return _PFACTOR_DOMAIN_COLORS["suicidal_p"]
    if "bad_grade" in tl or "academic" in tl:
        return _PFACTOR_DOMAIN_COLORS["bad_grades"]
    _is_parent = "parent" in tl or "par." in tl
    _is_clf = "\u2265 2 sd" in tl or "clinical change" in tl or "increase" in tl
    _is_dep = "depress" in tl or "dimensional" in tl
    if _is_dep:
        if _is_parent and _is_clf:
            return _PFACTOR_DOMAIN_COLORS["parent_dep_increase_2sd"]
        if _is_parent:
            return _PFACTOR_DOMAIN_COLORS["delta_parent_depress_D_p"]
        if _is_clf:
            return _PFACTOR_DOMAIN_COLORS["dep_increase_2sd"]
        return _PFACTOR_DOMAIN_COLORS["delta_depress_D_p"]
    return (0.5, 0.5, 0.5)


_PFACTOR_TARGET_ORDER = [
    "parent_internal_D_p",
    "Par. Internalizing",
    "Par. Internal",
    "internal_D_p",
    "Internalizing",
    "Internalizing Spectra",
    "parent_external_D_p",
    "Par. Externalizing",
    "Par. External",
    "external_D_p",
    "Externalizing",
    "Externalizing Spectra",
    "suicidal_p",
    "Suicidal Ideation",
    "parent_suicidal_thoughts_p",
    "Parent Suicidal Thoughts",
    "bad_grades",
    "Academic Performance",
]


def _pfactor_sort_targets(targets_list):
    return sorted(targets_list)


def _pfactor_sort_features(features_list, df_sankey, target_order):
    feat_info = {}
    for f in features_list:
        rows = df_sankey[df_sankey["Feature"] == f]
        n_tgt = rows["Target"].nunique()
        total_imp = rows["Importance"].sum()
        dom_target = rows.loc[rows["Importance"].idxmax(), "Target"]
        try:
            dom_idx = target_order.index(dom_target)
        except ValueError:
            dom_idx = 99
        feat_info[f] = {
            "n_tgt": n_tgt,
            "total_imp": total_imp,
            "dom_target": dom_target,
            "dom_idx": dom_idx,
        }


def _infer_feature_domain(feat_name):
    fl = feat_name.lower()
    if any(k in fl for k in ("(2t)", "2t)", "change", "delta")):
        return "Change Scores", _FEATURE_DOMAIN_ACCENTS["Change Scores"]
    if fl.startswith("par.") or fl.startswith("par ") or fl.endswith("_p"):
        return "Parent Report", _FEATURE_DOMAIN_ACCENTS["Parent Report"]
    if any(
        k in fl
        for k in ("fam.", "fam ", "family", "friend", "social", "relationship", "conflict", "unliked", "not liked")
    ):
        return "Family/Social", _FEATURE_DOMAIN_ACCENTS["Family/Social"]
    if any(k in fl for k in ("cognit", "task", "inhibit", "goal persist", "concentration", "attention", "memory")):
        return "Cognitive/Academic", _FEATURE_DOMAIN_ACCENTS["Cognitive/Academic"]
    if any(k in fl for k in ("somatic", "sleep", "eating", "drug", "bmi", "physical", "medical")):
        return "Physical/Medical", _FEATURE_DOMAIN_ACCENTS["Physical/Medical"]
    if any(k in fl for k in ("age", "sex", "income", "ses", "education")):
        return "Demographic", _FEATURE_DOMAIN_ACCENTS["Demographic"]
    return "Youth Behavioral", _FEATURE_DOMAIN_ACCENTS["Youth Behavioral"]


def _draw_sankey_pfactor_spectra(df_sankey, sankey_top_features, save_dir=None):
    """P-Factor Spectra Sankey: domain-colored heatflow variant."""
    _cd = _sankey_common_data(df_sankey)
    targets_list = _cd["targets_list"]
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    targets_list = _pfactor_sort_targets(targets_list)

    _SANKEY_DEP_RENAME = {
        "dep_increase_2sd": "Recent Clinical Change Child Depression Syndrome \u2265 2 SD",
        "delta_depress_D_p": "Change in Child Dimensional Depression Syndrome",
        "parent_dep_increase_2sd": "Recent Clinical Change Parent Depression Syndrome \u2265 2 SD",
        "delta_parent_depress_D_p": "Change in Parent Dimensional Depression Syndrome",
    }
    _PFACTOR_DISPLAY_RENAME = {
        "Internalizing Spectra": "Child Internalizing Spectra",
        "Externalizing Spectra": "Child Externalizing Spectra",
        "Internalizing": "Child Internalizing Spectra",
        "Externalizing": "Child Externalizing Spectra",
        "internal_D_p": "Child Internalizing Spectra",
        "external_D_p": "Child Externalizing Spectra",
        "parent_internal_D_p": "Parent Internalizing Spectra",
        "parent_external_D_p": "Parent Externalizing Spectra",
        "suicidal_p": "Suicidal Ideation",
        "parent_suicidal_thoughts_p": "Parent Suicidal Thoughts",
        "bad_grades": "Academic Performance",
    }

    def _edge_for(t):
        tl = t.lower()
        if "internal" in tl:
            return node_edge_int  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        if "external" in tl:
            return node_edge_ext  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        if "suicid" in tl:
            return "#060630"  # very dark blue edge
        if "bad_grade" in tl or "academic" in tl:
            return "#4A4A4A"  # dark gray edge
        return "#444444"

    _feat_targets_ranked = {}
    _feat_dom_target = {}
    for f in features_list:
        f_rows = df_sankey[df_sankey["Feature"] == f].sort_values("Importance", ascending=False)
        _feat_targets_ranked[f] = list(zip(f_rows["Target"].values, f_rows["Importance"].values))
        if not f_rows.empty:
            _feat_dom_target[f] = f_rows.iloc[0]["Target"]
        else:
            _feat_dom_target[f] = targets_list[0] if targets_list else ""

    _pad = max(0.015, min(0.035, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.028

    # Width-first figure sizing
    fig_w, fig_h = _wide_figure_dims(n_targets, n_features)
    _spread_t = 1.8
    _spread_f = 1.5
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    _target_node_w = 0.036

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )

    # Equalize vertical envelopes
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(bg_color)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
    ax.set_facecolor(bg_color)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04

    ax.set_xlim(-0.65, right_x + 0.65)
    ax.set_ylim(y_min, 1.08)
    ax.axis("off")

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = target_colors.get(rb["target"], (0.5, 0.5, 0.5))  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        rel = ctx._norm(rb["importance"])
        _alpha = 0.18 + 0.67 * rel

        if rel > 0.60:
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                _target_node_w,
                right_x,
                rb["y_l_top"] + 0.005,
                rb["y_l_bot"] - 0.005,
                rb["y_r_top"] + 0.005,
                rb["y_r_bot"] - 0.005,
                color_rgba=_rgb + (_alpha * 0.35,),
                zorder=1,
                right_node_w=node_w,
            )

        if rel < 0.35:
            _edge_c = (1.0, 1.0, 1.0, 0.30)
            _edge_lw = 0.5
        else:
            _edge_c = _rgb + (min(_alpha + 0.10, 0.85),)
            _edge_lw = 0.4

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            _target_node_w,
            right_x,
            rb["y_l_top"],
            rb["y_l_bot"],
            rb["y_r_top"],
            rb["y_r_bot"],
            color_rgba=_rgb + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_edge_lw,
            right_node_w=node_w,
        )

    for t, pos in target_positions.items():
        rgb = target_colors[t]  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        edge_c = _edge_for(t)

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.006, pos["y_bot"] - 0.006),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.007",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.22,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.007",
                facecolor=rgb,
                edgecolor=edge_c,
                linewidth=2.2,
                zorder=3,
            )
        )

        _disp_t = _display_target(t)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        _tm = ctx._target_metrics.get(t) or ctx._target_metrics.get(_disp_t)
        _label_x = left_x - _target_node_w - 0.025

        if _tm is not None:
            _val, _lbl = _tm
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.018,
                _disp_t,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=text_color,  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=bg_color)])  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
            _metric_color = _edge_for(t)
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.022,
                f"{_lbl} = {_val:.3f}",
                ha="right",
                va="center",
                fontsize=_lf[2],
                fontweight="bold",
                color=_metric_color,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=bg_color)])  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _disp_t,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=text_color,  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=bg_color)])  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))

    for f, pos in feature_positions.items():
        n_tgt = feat_target_count[f]
        dom_t = _feat_dom_target[f]
        _t_overlap = min((n_tgt - 2) / 3.0, 1.0) if n_tgt >= 2 else 0.0
        fc = (
            target_colors.get(dom_t, (0.5, 0.5, 0.5))  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
            if n_tgt == 1
            else (0.55 - 0.45 * _t_overlap, 0.55 - 0.45 * _t_overlap, 0.55 - 0.45 * _t_overlap)
        )

        _nx = right_x - node_w
        _nw = node_w * 2
        _ny = pos["y_bot"]
        _nh = pos["h"]

        ax.add_patch(
            FancyBboxPatch(
                (_nx + 0.004, _ny - 0.004),
                _nw,
                _nh,
                boxstyle="round,pad=0.006",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.18,
            )
        )

        if n_tgt >= 2:
            # Dual-glow border showing the two strongest target domains
            _dom_targets = sorted(_feat_targets_ranked.get(f, []), key=lambda x: x[1], reverse=True)[:2]
            _glow_colors = [target_colors.get(dt[0], (0.5, 0.5, 0.5)) for dt in _dom_targets]  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
            if len(_glow_colors) >= 2:
                ax.add_patch(
                    FancyBboxPatch(
                        (_nx - 0.007, _ny - 0.007),
                        _nw + 0.014,
                        _nh + 0.014,
                        boxstyle="round,pad=0.008",
                        facecolor="none",
                        edgecolor=_glow_colors[1],
                        linewidth=2.0,
                        zorder=2,
                        alpha=0.45,
                    )
                )
                ax.add_patch(
                    FancyBboxPatch(
                        (_nx - 0.004, _ny - 0.004),
                        _nw + 0.008,
                        _nh + 0.008,
                        boxstyle="round,pad=0.006",
                        facecolor="none",
                        edgecolor=_glow_colors[0],
                        linewidth=2.0,
                        zorder=2,
                        alpha=0.45,
                    )
                )
            elif len(_glow_colors) == 1:
                ax.add_patch(
                    FancyBboxPatch(
                        (_nx - 0.005, _ny - 0.005),
                        _nw + 0.010,
                        _nh + 0.010,
                        boxstyle="round,pad=0.007",
                        facecolor="none",
                        edgecolor=_glow_colors[0],
                        linewidth=2.0,
                        zorder=2,
                        alpha=0.45,
                    )
                )

        ax.add_patch(
            FancyBboxPatch(
                (_nx, _ny),
                _nw,
                _nh,
                boxstyle="round,pad=0.006",
                facecolor=fc,
                edgecolor="#444444",
                linewidth=1.0,
                zorder=3,
            )
        )

        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + node_w + 0.020,
            pos["y_centre"],
            f"{f}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color=text_color,  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
            fontfamily=_TITLE_FONT,
            zorder=5,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=bg_color)])  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)

    _title_y = 1.04
    ax.text(0.50, _title_y, " ", ha="center", va="bottom", fontsize=15, transform=ax.transAxes, alpha=0.0)
    # Subtitle moved to bottom
    ax.text(
        0.50,
        -0.015,
        f"Top {sankey_top_features} Consensus Features per Target  \u00b7  "
        "Shared Features Annotated (nT)  \u00b7  "
        "R\u00b2 from Across-Category Analyses\n"
        "Ribbon Opacity \u221d Weighted Ensemble Importance  \u00b7  "
        "Circular Features were Removed and Not Visible",
        ha="center",
        va="top",
        fontsize=_lf[3],
        fontweight="bold",
        color="#1a1a1a",
        fontfamily=_TITLE_FONT,
        linespacing=1.4,
        transform=ax.transAxes,
    )

    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.045
    )
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=node_edge_int,  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        fontfamily=_TITLE_FONT,
    )
    ax.plot(
        [left_x - _target_node_w * 1.5, left_x + _target_node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=node_edge_int,  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        lw=1.0,
        alpha=0.5,
        clip_on=False,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=node_edge_ext,  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        fontfamily=_TITLE_FONT,
    )
    ax.plot(
        [right_x - node_w * 1.5, right_x + node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=node_edge_ext,  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
        lw=1.0,
        alpha=0.5,
        clip_on=False,
    )

    plt.tight_layout(rect=[0, 0.01, 0.90, 0.98])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, "sankey_pfactor_spectra", dpi=600)
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    return fig


# CHILD ALTs: 6-Target Domain-Colored Heatflow

_CHILD_ALTS_COLORS = {
    "adhd_D_p": (0.82, 0.68, 0.15),
    "external_D_p": (0.45, 0.75, 0.20),
    "internal_D_p": (0.55, 0.36, 0.95),
    "social_problems_D_p": (0.90, 0.50, 0.13),
    "bad_grades": (0.28, 0.28, 0.32),
    "tb_flanker": (0.00, 0.81, 0.82),
    "suicidal_p": (0.00, 0.00, 0.55),
}

_CHILD_ALTS_DISPLAY_ALIASES = {
    "ADHD": "adhd_D_p",
    "ADHD Symptoms": "adhd_D_p",
    "Externalizing": "external_D_p",
    "Externalizing Spectra": "external_D_p",
    "Child Externalizing Spectra": "external_D_p",
    "Internalizing": "internal_D_p",
    "Internalizing Spectra": "internal_D_p",
    "Child Internalizing Spectra": "internal_D_p",
    "Social Problems": "social_problems_D_p",
    "Bad Grades": "bad_grades",
    "Poor Academic Performance": "bad_grades",
    "Flanker": "tb_flanker",
    "Flanker Task": "tb_flanker",
    "Flanker (NIH Toolbox)": "tb_flanker",
    "Suicidality": "suicidal_p",
    "Suicidal Ideation": "suicidal_p",
}

_CHILD_ALTS_ORDER = [
    "internal_D_p",
    "external_D_p",
    "adhd_D_p",
    "social_problems_D_p",
    "bad_grades",
    "tb_flanker",
    "suicidal_p",
]

_CHILD_ALTS_EDGE = "#2C3440"
_CHILD_ALTS_BG = "#FFFFFF"


def _child_alts_resolve_color(target_name):
    if target_name in _CHILD_ALTS_COLORS:
        return _CHILD_ALTS_COLORS[target_name]
    raw = _CHILD_ALTS_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_ALTS_COLORS:
        return _CHILD_ALTS_COLORS[raw]
    tl = target_name.lower()
    if "adhd" in tl:
        return _CHILD_ALTS_COLORS["adhd_D_p"]
    if "external" in tl:
        return _CHILD_ALTS_COLORS["external_D_p"]
    if "internal" in tl:
        return _CHILD_ALTS_COLORS["internal_D_p"]
    if "social" in tl:
        return _CHILD_ALTS_COLORS["social_problems_D_p"]
    if "grade" in tl or "academic" in tl:
        return _CHILD_ALTS_COLORS["bad_grades"]
    if "flanker" in tl:
        return _CHILD_ALTS_COLORS["tb_flanker"]
    if "suicid" in tl:
        return _CHILD_ALTS_COLORS["suicidal_p"]
    return (0.5, 0.5, 0.5)


def _child_alts_sort_targets(targets_list):
    return sorted(targets_list)


def _child_alts_edge_for(target_name):
    rgb = _child_alts_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


_PARENT_ALTS_COLORS = {
    "parent_suicidal_thoughts_p": (0.06, 0.08, 0.35),
    "parent_bad_relationships_p": (0.80, 0.35, 0.05),
    "parent_adhd_D_p": (0.82, 0.72, 0.22),
    "parent_happy_person_p": (0.90, 0.35, 0.55),
    "parent_external_D_p": (0.10, 0.42, 0.22),
    "parent_internal_D_p": (0.30, 0.05, 0.45),
}

_PARENT_ALTS_DISPLAY_ALIASES = {
    "Parent Suicidal Thoughts": "parent_suicidal_thoughts_p",
    "Parent Suicidal Ideation": "parent_suicidal_thoughts_p",
    "Parent Suicidality": "parent_suicidal_thoughts_p",
    "Parent Social Quality": "parent_bad_relationships_p",
    "Parent Bad Relationships": "parent_bad_relationships_p",
    "Parent Interpersonal Difficulties": "parent_bad_relationships_p",
    "Parent Internalizing": "parent_internal_D_p",
    "Parent Internalizing Spectra": "parent_internal_D_p",
    "Parent ADHD": "parent_adhd_D_p",
    "Parent ADHD Symptoms": "parent_adhd_D_p",
    "Parent Happiness": "parent_happy_person_p",
    "Parent Happiness (Item)": "parent_happy_person_p",
    "Parent Wellbeing": "parent_happy_person_p",
    "Parent Happy Person": "parent_happy_person_p",
    "Parent Externalizing": "parent_external_D_p",
    "Parent Externalizing Spectra": "parent_external_D_p",
}

_PARENT_ALTS_ORDER = [
    "parent_suicidal_thoughts_p",
    "parent_bad_relationships_p",
    "parent_adhd_D_p",
    "parent_happy_person_p",
    "parent_external_D_p",
    "parent_internal_D_p",
]

_PARENT_ALTS_EDGE = "#2C3440"
_PARENT_ALTS_BG = "#FFFFFF"


def _parent_alts_resolve_color(target_name):
    if target_name in _PARENT_ALTS_COLORS:
        return _PARENT_ALTS_COLORS[target_name]
    raw = _PARENT_ALTS_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _PARENT_ALTS_COLORS:
        return _PARENT_ALTS_COLORS[raw]
    tl = target_name.lower()
    if "suicid" in tl:
        return _PARENT_ALTS_COLORS["parent_suicidal_thoughts_p"]
    if "relationship" in tl or "social" in tl:
        return _PARENT_ALTS_COLORS["parent_bad_relationships_p"]
    if "adhd" in tl:
        return _PARENT_ALTS_COLORS["parent_adhd_D_p"]
    if "happi" in tl or "wellbeing" in tl:
        return _PARENT_ALTS_COLORS["parent_happy_person_p"]
    if "external" in tl:
        return _PARENT_ALTS_COLORS["parent_external_D_p"]
    if "internal" in tl:
        return _PARENT_ALTS_COLORS["parent_internal_D_p"]
    return (0.5, 0.5, 0.5)


def _parent_alts_sort_targets(targets_list):
    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}


def _parent_alts_edge_for(target_name):
    rgb = _parent_alts_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


def _draw_sankey_domain_style(
    df_sankey,
    sankey_top_features,
    save_dir=None,
    sort_fn=None,
    color_fn=None,
    edge_fn=None,
    edge_col="#2C3440",
    bg_color="#FFFFFF",
    save_name="sankey_domain",
    label_override=None,
    spread_target_override=None,
    spread_feature_override=None,
    column_gap_override=None,
    feature_share_darkening=False,
):
    """
    Shared draw function for child_alts, parent_alts, and any future
    domain-colored heatflow Sankeys.  Avoids duplicating ~200 lines.
    """
    _BG = bg_color
    _EDGE_COL = edge_col

    _cd = _sankey_common_data(df_sankey)
    targets_list = sort_fn(_cd["targets_list"]) if sort_fn else _cd["targets_list"]
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: color_fn(t) for t in targets_list}

    _pad = max(0.015, min(0.035, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    # Per-style column gap override (push features column further right)
    if column_gap_override is not None:
        right_x = left_x + column_gap_override
    node_w = 0.028
    _target_node_w = 0.036

    # Width-first figure sizing
    fig_w, fig_h = _wide_figure_dims(n_targets, n_features)
    _spread_t = 1.8
    _spread_f = 1.5
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    # Per-style spread overrides (applied after landscape adjust)
    if spread_target_override is not None:
        _spread_t = spread_target_override
    if spread_feature_override is not None:
        _spread_f = spread_feature_override
    _lf = _landscape_fonts()

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )

    # Equalize vertical envelopes
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04

    ax.set_xlim(-0.65, right_x + 0.65)
    ax.set_ylim(y_min, 1.08)
    ax.axis("off")

    # Ribbons
    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])
        _alpha = 0.18 + 0.67 * rel

        if rel > 0.60:
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                _target_node_w,
                right_x,
                rb["y_l_top"] + 0.005,
                rb["y_l_bot"] - 0.005,
                rb["y_r_top"] + 0.005,
                rb["y_r_bot"] - 0.005,
                color_rgba=_rgb + (_alpha * 0.35,),
                zorder=1,
                right_node_w=node_w,
            )

        if rel < 0.35:
            _edge_c = (1.0, 1.0, 1.0, 0.30)
            _edge_lw = 0.5
        else:
            _edge_c = _rgb + (min(_alpha + 0.10, 0.85),)
            _edge_lw = 0.4

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            _target_node_w,
            right_x,
            rb["y_l_top"],
            rb["y_l_bot"],
            rb["y_r_top"],
            rb["y_r_bot"],
            color_rgba=_rgb + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_edge_lw,
            right_node_w=node_w,
        )

    # Target nodes
    text_color = "#1a1a1a"
    for t, pos in target_positions.items():
        rgb = _tc[t]

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.006, pos["y_bot"] - 0.006),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.007",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.22,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.007",
                facecolor=rgb,
                edgecolor=_EDGE_COL,
                linewidth=2.2,
                zorder=3,
            )
        )

        display = _tdn.get(t, t)
        if label_override:
            display = label_override.get(display, label_override.get(t, display))
        _tm = _target_metrics.get(t) or _target_metrics.get(display)
        _label_x = left_x - _target_node_w - 0.025

        if _tm is not None:
            _val, _lbl = _tm
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.018,
                display,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=text_color,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])
            _metric_color = edge_fn(t) if edge_fn else _EDGE_COL
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.022,
                f"{_lbl} = {_val:.3f}",
                ha="right",
                va="center",
                fontsize=_lf[2],
                fontweight="bold",
                color=_metric_color,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                display,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=text_color,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])

    # Feature nodes
    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))

    _feat_dom_target = {}
    for f in features_list:
        f_rows = df_sankey[df_sankey["Feature"] == f].sort_values("Importance", ascending=False)
        _feat_dom_target[f] = (
            f_rows.iloc[0]["Target"] if not f_rows.empty else (targets_list[0] if targets_list else "")
        )

    for f, pos in feature_positions.items():
        n_tgt = feat_target_count[f]
        dom_t = _feat_dom_target[f]
        if n_tgt == 1:
            fc = _tc.get(dom_t, (0.5, 0.5, 0.5))
        elif feature_share_darkening:
            # Graded darkening: bluish-gray at n_tgt=2 -> near-black at n_tgt=n_targets.
            # Visual: features shared by many targets stand out as the darkest spine.
            _share_base = (0.45, 0.52, 0.62)  # current default multi-target color
            _share_dark = (0.05, 0.07, 0.10)  # near-black floor
            _span = max(1, n_targets - 1)
            _t = min(1.0, max(0.0, (n_tgt - 1) / _span))
            fc = tuple(_share_base[i] * (1.0 - _t) + _share_dark[i] * _t for i in range(3))
        else:
            fc = (0.45, 0.52, 0.62)

        _nx = right_x - node_w
        _nw = node_w * 2
        _ny = pos["y_bot"]
        _nh = pos["h"]

        ax.add_patch(
            FancyBboxPatch(
                (_nx + 0.004, _ny - 0.004),
                _nw,
                _nh,
                boxstyle="round,pad=0.006",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.18,
            )
        )

        if n_tgt >= 2:
            f_rows = df_sankey[df_sankey["Feature"] == f].sort_values("Importance", ascending=False)
            top2_targets = f_rows["Target"].head(2).tolist()
            glow_outer = _tc.get(top2_targets[0], (0.5, 0.5, 0.5))
            glow_inner = _tc.get(top2_targets[1], (0.5, 0.5, 0.5)) if len(top2_targets) > 1 else glow_outer
            ax.add_patch(
                FancyBboxPatch(
                    (_nx - 0.007, _ny - 0.007),
                    _nw + 0.014,
                    _nh + 0.014,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_outer + (0.50,),
                    linewidth=2.0,
                    zorder=2,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    (_nx - 0.004, _ny - 0.004),
                    _nw + 0.008,
                    _nh + 0.008,
                    boxstyle="round,pad=0.006",
                    facecolor="none",
                    edgecolor=glow_inner + (0.45,),
                    linewidth=2.0,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (_nx, _ny),
                _nw,
                _nh,
                boxstyle="round,pad=0.006",
                facecolor=fc,
                edgecolor="#444444",
                linewidth=1.0,
                zorder=3,
            )
        )

        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + node_w + 0.020,
            pos["y_centre"],
            f"{f}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color=text_color,
            fontfamily=_TITLE_FONT,
            zorder=5,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=_BG)])

    # Title / subtitle
    _title_y = 1.04
    ax.text(0.50, _title_y, " ", ha="center", va="bottom", fontsize=15, transform=ax.transAxes, alpha=0.0)
    ax.text(
        0.50,
        -0.015,
        f"Top {sankey_top_features} Consensus Features per Target  \u00b7  "
        "Shared Features Annotated (nT)  \u00b7  "
        "R\u00b2 from Across-Category Analyses\n"
        "Ribbon Opacity \u221d Weighted Ensemble Importance  \u00b7  "
        "Circular Features were Removed and Not Visible",
        ha="center",
        va="top",
        fontsize=_lf[3],
        fontweight="bold",
        color="#1a1a1a",
        fontfamily=_TITLE_FONT,
        linespacing=1.4,
        transform=ax.transAxes,
    )

    # Column headers: parallel, above whichever column is taller
    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.045
    )
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    ax.plot(
        [left_x - _target_node_w * 1.5, left_x + _target_node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=_EDGE_COL,
        lw=1.0,
        alpha=0.5,
        clip_on=False,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    ax.plot(
        [right_x - node_w * 1.5, right_x + node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=_EDGE_COL,
        lw=1.0,
        alpha=0.5,
        clip_on=False,
    )

    plt.tight_layout(rect=[0, 0.01, 0.90, 0.98])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, save_name, dpi=600)
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    return fig


# SBT core triad: Core Depression + Academic/Cognitive + Perfectionism

_SBT_CORE_TRIAD_COLORS = {
    "sbt_core_depression": (0.102, 0.227, 0.420),  # deep navy
    "sbt_academic_cognitive": (0.722, 0.588, 0.043),  # golden amber
    "sbt_perfectionism_achievement": (0.769, 0.290, 0.502),  # rose magenta
}

_SBT_CORE_TRIAD_DISPLAY_ALIASES = {
    "Core Depression": "sbt_core_depression",
    "Mood/Depression (Primary Features)": "sbt_core_depression",
    "Academic/Cognitive": "sbt_academic_cognitive",
    "Depression Subtype (Poor Academic/Cognitive)": "sbt_academic_cognitive",
    "Poor Academic/Cognitive": "sbt_academic_cognitive",
    "Perfectionism/Achievement": "sbt_perfectionism_achievement",
    "Depression Subtype (Perfectionism/High-Achievement)": "sbt_perfectionism_achievement",
    "High-Achievement": "sbt_perfectionism_achievement",
}

_SBT_CORE_TRIAD_ORDER = [
    "sbt_core_depression",
    "sbt_academic_cognitive",
    "sbt_perfectionism_achievement",
]

_SBT_CORE_TRIAD_EDGE = "#2C3440"
_SBT_CORE_TRIAD_BG = "#FFFFFF"


def _sbt_core_triad_resolve_color(target_name):
    if target_name in _SBT_CORE_TRIAD_COLORS:
        return _SBT_CORE_TRIAD_COLORS[target_name]
    raw = _SBT_CORE_TRIAD_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _SBT_CORE_TRIAD_COLORS:
        return _SBT_CORE_TRIAD_COLORS[raw]
    tl = target_name.lower()
    if "core" in tl or "primary" in tl:
        return _SBT_CORE_TRIAD_COLORS["sbt_core_depression"]
    if "academic" in tl or "cognitive" in tl:
        return _SBT_CORE_TRIAD_COLORS["sbt_academic_cognitive"]
    if "perfect" in tl or "achiev" in tl:
        return _SBT_CORE_TRIAD_COLORS["sbt_perfectionism_achievement"]
    return (0.5, 0.5, 0.5)


def _sbt_core_triad_sort_targets(targets_list):
    return sorted(targets_list)


def _sbt_core_triad_edge_for(target_name):
    rgb = _sbt_core_triad_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


# SBT core triad: female variant (warmer tones: plum navy / warm gold / rose)

_SBT_CORE_TRIAD_FEMALE_COLORS = {
    "sbt_core_depression": (0.145, 0.200, 0.440),  # plum-shifted navy
    "sbt_academic_cognitive": (0.745, 0.555, 0.110),  # warm amber-gold
    "sbt_perfectionism_achievement": (0.800, 0.245, 0.460),  # deeper rose
}

_SBT_CORE_TRIAD_FEMALE_EDGE = "#4A2040"  # warm plum edge
_SBT_CORE_TRIAD_FEMALE_BG = "#FFFFFF"


def _sbt_core_triad_female_resolve_color(target_name):
    if target_name in _SBT_CORE_TRIAD_FEMALE_COLORS:
        return _SBT_CORE_TRIAD_FEMALE_COLORS[target_name]
    raw = _SBT_CORE_TRIAD_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _SBT_CORE_TRIAD_FEMALE_COLORS:
        return _SBT_CORE_TRIAD_FEMALE_COLORS[raw]
    tl = target_name.lower()
    if "core" in tl or "primary" in tl:
        return _SBT_CORE_TRIAD_FEMALE_COLORS["sbt_core_depression"]
    if "academic" in tl or "cognitive" in tl:
        return _SBT_CORE_TRIAD_FEMALE_COLORS["sbt_academic_cognitive"]
    if "perfect" in tl or "achiev" in tl:
        return _SBT_CORE_TRIAD_FEMALE_COLORS["sbt_perfectionism_achievement"]
    return (0.5, 0.5, 0.5)


def _sbt_core_triad_female_edge_for(target_name):
    rgb = _sbt_core_triad_female_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


# SBT core triad: male variant (cooler tones: steel navy / olive gold / slate mauve)

_SBT_CORE_TRIAD_MALE_COLORS = {
    "sbt_core_depression": (0.075, 0.250, 0.400),  # steel-blue navy
    "sbt_academic_cognitive": (0.680, 0.620, 0.020),  # cool olive-gold
    "sbt_perfectionism_achievement": (0.710, 0.330, 0.545),  # muted slate-mauve
}

_SBT_CORE_TRIAD_MALE_EDGE = "#1E2E48"  # cool steel edge
_SBT_CORE_TRIAD_MALE_BG = "#FFFFFF"


def _sbt_core_triad_male_resolve_color(target_name):
    if target_name in _SBT_CORE_TRIAD_MALE_COLORS:
        return _SBT_CORE_TRIAD_MALE_COLORS[target_name]
    raw = _SBT_CORE_TRIAD_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _SBT_CORE_TRIAD_MALE_COLORS:
        return _SBT_CORE_TRIAD_MALE_COLORS[raw]
    tl = target_name.lower()
    if "core" in tl or "primary" in tl:
        return _SBT_CORE_TRIAD_MALE_COLORS["sbt_core_depression"]
    if "academic" in tl or "cognitive" in tl:
        return _SBT_CORE_TRIAD_MALE_COLORS["sbt_academic_cognitive"]
    if "perfect" in tl or "achiev" in tl:
        return _SBT_CORE_TRIAD_MALE_COLORS["sbt_perfectionism_achievement"]
    return (0.5, 0.5, 0.5)


def _sbt_core_triad_male_edge_for(target_name):
    rgb = _sbt_core_triad_male_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


# Draw wrappers for all three core-triad variants


def _draw_sankey_sbt_core_triad(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_sbt_core_triad_sort_targets,
        color_fn=_sbt_core_triad_resolve_color,
        edge_fn=_sbt_core_triad_edge_for,
        edge_col=_SBT_CORE_TRIAD_EDGE,
        bg_color=_SBT_CORE_TRIAD_BG,
        save_name="sankey_sbt_core_triad",
        label_override={
            "sbt_core_depression": "Core Depression",
            "sbt_academic_cognitive": "Academic/Cognitive",
            "sbt_perfectionism_achievement": "Perfectionism/Achievement",
        },
    )


def _draw_sankey_sbt_core_triad_female(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_sbt_core_triad_sort_targets,
        color_fn=_sbt_core_triad_female_resolve_color,
        edge_fn=_sbt_core_triad_female_edge_for,
        edge_col=_SBT_CORE_TRIAD_FEMALE_EDGE,
        bg_color=_SBT_CORE_TRIAD_FEMALE_BG,
        save_name="sankey_sbt_core_triad_female",
        label_override={
            "sbt_core_depression": "Core Depression",
            "sbt_academic_cognitive": "Academic/Cognitive",
            "sbt_perfectionism_achievement": "Perfectionism/Achievement",
        },
    )


def _draw_sankey_sbt_core_triad_male(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_sbt_core_triad_sort_targets,
        color_fn=_sbt_core_triad_male_resolve_color,
        edge_fn=_sbt_core_triad_male_edge_for,
        edge_col=_SBT_CORE_TRIAD_MALE_EDGE,
        bg_color=_SBT_CORE_TRIAD_MALE_BG,
        save_name="sankey_sbt_core_triad_male",
        label_override={
            "sbt_core_depression": "Core Depression",
            "sbt_academic_cognitive": "Academic/Cognitive",
            "sbt_perfectionism_achievement": "Perfectionism/Achievement",
        },
    )


def _draw_sankey_child_alts(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_child_alts_sort_targets,
        color_fn=_child_alts_resolve_color,
        edge_fn=_child_alts_edge_for,
        edge_col=_CHILD_ALTS_EDGE,
        bg_color=_CHILD_ALTS_BG,
        save_name="sankey_child_alts",
        label_override={
            "Suicidal Ideation (Child)": "Suicidal Ideation",
        },
    )


def _draw_sankey_parent_alts(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_parent_alts_sort_targets,
        color_fn=_parent_alts_resolve_color,
        edge_fn=_parent_alts_edge_for,
        edge_col=_PARENT_ALTS_EDGE,
        bg_color=_PARENT_ALTS_BG,
        save_name="sankey_parent_alts",
        label_override={
            "Parent Happiness (Item)": "Parent Happiness",
            "Parent Suicidal Ideation (Child)": "Parent Suicidal Ideation",
        },
    )


# Child cognitive task battery (T0): domain-colored Sankey
# 7 targets covering distinct cognitive constructs at baseline (T0):
# crystallized intelligence (verbal knowledge), fluid composite (EF/speed/WM/
# episodic), Flanker (NIH inhibitory control), SST SSRT (motor inhibition),
# 2-back (emotional working memory), RAVLT long-delay (verbal episodic
# memory), LMT efficiency (visuospatial mental rotation). Palette is CUD-safe
# (Wong 2011): NIH Toolbox composites in cool blue-greens, paradigm-specific
# tasks in distinct hues across the warm-cool axis. Verified under
# deuteranopia, protanopia, tritanopia.

_CHILD_COGTASK_T0_COLORS = {
    # Palette tuned to match the Objective/Subjective-Report Spectrum heatmap.
    # Targets sharing a column with the heatmap inherit that column's hue;
    # the two NIH composites without dedicated columns (tb_cryst, tb_flanker)
    # take complementary cool tones that avoid green and gray for clean
    # external pairing with green/gray-heavy figures.
    # NIH Toolbox composite: matches "Comp / Fluid Intell." in heatmap
    "tb_fluid": (0.122, 0.176, 0.502),  # #1F2D80 deep indigo navy
    # Crystallized sibling (not in heatmap): cool steel-blue, no green/gray
    "tb_cryst": (0.290, 0.435, 0.647),  # #4A6FA5 steel-blue
    # Flanker (not in heatmap): lighter NIH family member, fluid constituent
    "tb_flanker": (0.337, 0.706, 0.914),  # #56B4E9 CUD sky blue
    # SST: matches "SST" column hue in heatmap
    "sst_mean_ssrt": (0.722, 0.204, 0.204),  # #B83434 firebrick red
    # N-back: matches "N-Back" column hue in heatmap
    "nb_correct_nt_2back": (0.055, 0.494, 0.537),  # #0E7E89 dark teal
    # RAVLT: matches "RAVLT" column hue in heatmap
    "ravlt_l_total": (0.165, 0.310, 0.843),  # #2A4FD7 royal blue
    # LMT: matches "LMT" column hue in heatmap
    "lmt_efficiency": (0.420, 0.306, 0.200),  # #6B4E33 saddle brown
}

_CHILD_COGTASK_T0_TARGET_EDGE = {
    "tb_fluid": "#0F1850",  # deeper indigo for navy fill
    "tb_cryst": "#2D456B",  # deeper steel-blue
    "tb_flanker": "#1F6F9F",  # deeper sky-blue
    "sst_mean_ssrt": "#7A2020",  # deeper firebrick
    "nb_correct_nt_2back": "#095157",  # deeper teal
    "ravlt_l_total": "#1A3387",  # deeper royal blue
    "lmt_efficiency": "#443120",  # deeper saddle brown
}

_CHILD_COGTASK_T0_DOMAIN_CLEAN_LABELS = {
    "tb_cryst": "Crystallized Intelligence",
    "tb_fluid": "Fluid Intelligence",
    "tb_flanker": "Cognitive Control (Flanker)",
    "sst_mean_ssrt": "Response Inhibition (SSRT)",
    "nb_correct_nt_2back": "Emotional Working Memory (2-Back)",
    "ravlt_l_total": "Verbal Episodic Memory (RAVLT)",
    "lmt_efficiency": "Visuospatial Mental Rotation (LMT)",
}

_CHILD_COGTASK_T0_DISPLAY_ALIASES = {
    # tb_cryst aliases (overlap with F4c style is intentional)
    "Crystallized Intelligence": "tb_cryst",
    "Crystallized Composite (NIH)": "tb_cryst",
    "Cryst./Fluid Comp.": "tb_cryst",
    # tb_fluid aliases
    "Fluid Intelligence": "tb_fluid",
    "Fluid Composite (NIH)": "tb_fluid",
    # tb_flanker aliases (overlap with child_alts style is intentional)
    "Cognitive Control (Flanker)": "tb_flanker",
    "Flanker": "tb_flanker",
    "Flanker Task": "tb_flanker",
    "Flanker (NIH Toolbox)": "tb_flanker",
    "Flanker Inhibitory Control Task (NIH)": "tb_flanker",
    # sst_mean_ssrt aliases
    "Response Inhibition (SSRT)": "sst_mean_ssrt",
    "Response Inhibition (SST SSRT)": "sst_mean_ssrt",
    "SST Task Mean SSRT": "sst_mean_ssrt",
    # nb_correct_nt_2back aliases
    "Emotional Working Memory (2-Back)": "nb_correct_nt_2back",
    "2-Back Acc.": "nb_correct_nt_2back",
    # ravlt_l_total aliases
    "Verbal Episodic Memory (RAVLT)": "ravlt_l_total",
    "RAVLT Long Tot.": "ravlt_l_total",
    # lmt_efficiency aliases
    "Visuospatial Mental Rotation (LMT)": "lmt_efficiency",
    "LMT Efficiency": "lmt_efficiency",
}

# Display order: NIH Toolbox composites first, then NIH subscale, then
# fMRI-paradigm behavioral tasks, then off-NIH off-fMRI paradigms.
# Mirrors the construct-coverage narrative (broad -> narrow -> orthogonal).
_CHILD_COGTASK_T0_ORDER = [
    "tb_cryst",
    "tb_fluid",
    "tb_flanker",
    "sst_mean_ssrt",
    "nb_correct_nt_2back",
    "ravlt_l_total",
    "lmt_efficiency",
]

_CHILD_COGTASK_T0_EDGE = "#1A2332"  # dark slate axis edge: matches academic_cognitive
_CHILD_COGTASK_T0_BG = "#FFFFFF"


def _child_cogtask_t0_resolve_color(target_name):
    """Resolve target color for the T0 cognitive-task Sankey style.

    Tries (1) raw target key, (2) display-alias lookup, (3) substring
    fallback for archive keys like 'tb_cryst__tp0' or display names with
    suffixes. Returns mid-gray if no rule matches.
    """
    if target_name in _CHILD_COGTASK_T0_COLORS:
        return _CHILD_COGTASK_T0_COLORS[target_name]
    raw = _CHILD_COGTASK_T0_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_COGTASK_T0_COLORS:
        return _CHILD_COGTASK_T0_COLORS[raw]
    tl = target_name.lower()
    if "cryst" in tl:
        return _CHILD_COGTASK_T0_COLORS["tb_cryst"]
    if "fluid" in tl:
        return _CHILD_COGTASK_T0_COLORS["tb_fluid"]
    if "flanker" in tl:
        return _CHILD_COGTASK_T0_COLORS["tb_flanker"]
    if "ssrt" in tl or ("sst" in tl and "inhib" in tl):
        return _CHILD_COGTASK_T0_COLORS["sst_mean_ssrt"]
    if "2-back" in tl or "2back" in tl or "working memory" in tl:
        return _CHILD_COGTASK_T0_COLORS["nb_correct_nt_2back"]
    if "ravlt" in tl or "verbal episodic" in tl:
        return _CHILD_COGTASK_T0_COLORS["ravlt_l_total"]
    if "lmt" in tl or "mental rotation" in tl or "visuospatial" in tl:
        return _CHILD_COGTASK_T0_COLORS["lmt_efficiency"]
    return (0.5, 0.5, 0.5)


def _child_cogtask_t0_domain(target_name):
    """Return a short domain key for a T0 cognitive-task target."""
    if target_name in _CHILD_COGTASK_T0_COLORS:
        raw = target_name
    else:
        raw = _CHILD_COGTASK_T0_DISPLAY_ALIASES.get(target_name, target_name)
    tl = raw.lower()
    if "cryst" in tl:
        return "crystallized"
    if "fluid" in tl:
        return "fluid"
    if "flanker" in tl:
        return "inhibition_flanker"
    if "ssrt" in tl or ("sst" in tl and "inhib" in tl):
        return "inhibition_sst"
    if "2-back" in tl or "2back" in tl:
        return "wm_emotional"
    if "ravlt" in tl:
        return "memory_verbal"
    if "lmt" in tl or "mental rotation" in tl:
        return "visuospatial"
    return "other"


def _child_cogtask_t0_edge_for(target_name):
    """Resolve target node edge color."""
    if target_name in _CHILD_COGTASK_T0_TARGET_EDGE:
        return _CHILD_COGTASK_T0_TARGET_EDGE[target_name]
    raw = _CHILD_COGTASK_T0_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_COGTASK_T0_TARGET_EDGE:
        return _CHILD_COGTASK_T0_TARGET_EDGE[raw]
    rgb = _child_cogtask_t0_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


def _child_cogtask_t0_sort_targets(targets_list):
    """Order targets by construct narrative; unrecognised targets append in
    their incoming order to keep the drawer tolerant of new additions."""
    ranked = []
    leftover = []
    for t in targets_list:
        # Resolve archive keys / display names back to canonical
        canonical = t if t in _CHILD_COGTASK_T0_ORDER else _CHILD_COGTASK_T0_DISPLAY_ALIASES.get(t, t)
        if canonical in _CHILD_COGTASK_T0_ORDER:
            ranked.append((t, _CHILD_COGTASK_T0_ORDER.index(canonical)))
        else:
            leftover.append(t)
    ranked.sort(key=lambda x: x[1])
    return [t for t, _ in ranked] + leftover


def _draw_sankey_child_cogtask_t0(df_sankey, sankey_top_features, save_dir=None):
    """T0 Cognitive Task Battery Sankey.

    Routes through `_draw_sankey_domain_style` (the modern shared drawer
    used by `child_alts` / `parent_alts`), so this style automatically picks
    up the latest visual refinements (width-first sizing, landscape adjust,
    figure padding, target/feature node geometry) without duplicating
    drawer code.

    Two style-specific behaviors:
      1. Targets are sorted by R\u00b2 (best at top, worst at bottom) using
         `_batch_archive` metrics: mirroring the SBT-by-perf pattern.
         Falls back to the construct-narrative order if metrics unavailable.
      2. Target spread is increased to 2.6 (vs 1.8 default) to give ribbons
         more breathing room on the left side for clearer connection traces.
    """
    # Build a perf-aware sort closure using the same machinery as SBT
    _ba = ctx.get("_batch_archive", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}

    def _sort_by_r2(targets_list):
        if not _tm:
            # No metrics yet: fall back to construct-narrative order
            return _child_cogtask_t0_sort_targets(targets_list)
        _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})

        def _score(t):
            tm = _tm.get(t)
            if tm is None:
                tm = _tm.get(_tdn.get(t, ""))
            if tm is None:
                # try canonical key form (display alias -> raw)
                canonical = _CHILD_COGTASK_T0_DISPLAY_ALIASES.get(t, t)
                tm = _tm.get(canonical) or _tm.get(_tdn.get(canonical, ""))
            return tm[0] if tm is not None else float("-inf")

        # reverse=True puts highest R\u00b2 first; first element renders at TOP
        return sorted(targets_list, key=_score, reverse=True)

    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_sort_by_r2,
        color_fn=_child_cogtask_t0_resolve_color,
        edge_fn=_child_cogtask_t0_edge_for,
        edge_col=_CHILD_COGTASK_T0_EDGE,
        bg_color=_CHILD_COGTASK_T0_BG,
        save_name="sankey_child_cogtask_t0",
        spread_target_override=2.6,  # vertical: 1.8 default -> 2.6 (45% more)
        column_gap_override=2.2,  # horizontal: 1.30 default -> 2.2 (69% wider)
        feature_share_darkening=True,  # darken multi-target features toward black
        label_override={
            "Crystallized Composite (NIH)": "Crystallized Intelligence",
            "Cryst./Fluid Comp.": "Crystallized Intelligence",
            "Flanker Inhibitory Control Task (NIH)": "Cognitive Control (Flanker)",
            "SST Task Mean SSRT": "Response Inhibition (SSRT)",
            "2-Back Acc.": "Emotional Working Memory (2-Back)",
            "RAVLT Long Tot.": "Verbal Episodic Memory (RAVLT)",
            "LMT Efficiency": "Visuospatial Mental Rotation (LMT)",
        },
    )


# Dep worsening: shared helpers + two dedicated Sankey styles

_CHILD_DEP_WORSENING_ORDER = [
    "top_5_depression",
    "top_10_depression",
    "dep_increase_2sd",
    "dep_increase_1.5sd",
    "dep_onset_rci_2.3",
    "Depression (Top 5th Percentile)",
    "Depression (Top 10th Percentile)",
    "Depression Change (\u2265 2 SD)",
    "Depression Change (\u2265 1.5 SD)",
    "Depression Onset (RCI \u2265 2.3)",
    "Depression Increase (\u2265 2 SD)",
    "Depression Increase (\u2265 1.5 SD)",
]

_CHILD_LABEL_OVERRIDE = {
    "dep_increase_2sd": "Depression Change (\u2265 2 SD)",
    "dep_increase_1.5sd": "Depression Change (\u2265 1.5 SD)",
    "Depression Increase (\u2265 2 SD)": "Depression Change (\u2265 2 SD)",
    "Depression Increase (\u2265 1.5 SD)": "Depression Change (\u2265 1.5 SD)",
}

_PARENT_DEP_WORSENING_ORDER = [
    "top_5_depression_parent",
    "top_10_depression_parent",
    "parent_dep_increase_2sd",
    "parent_dep_increase_1.5sd",
    "parent_dep_onset_rci_2.3",
    "Parent Depression (Top 5th Percentile)",
    "Parent Depression (Top 10th Percentile)",
    "Parent Depression Changes (\u2265 2 SD)",
    "Parent Depression Changes (\u2265 1.5 SD)",
    "Parent Depression Onset (RCI \u2265 2.3)",
]

_DEP_WORSENING_COLORS = {
    # Child: saturated cool blues → contrasting violets
    "top_5_depression": (0.035, 0.150, 0.500),  # deep navy
    "top_10_depression": (0.180, 0.420, 0.750),  # vivid blue
    "dep_increase_2sd": (0.200, 0.010, 0.420),  # deep indigo-violet
    "dep_increase_1.5sd": (0.560, 0.240, 0.740),  # bright orchid
    "dep_onset_rci_2.3": (0.580, 0.350, 0.680),  # soft lavender-violet
    # Parent: warmer teal-blues → contrasting plums
    "top_5_depression_parent": (0.050, 0.180, 0.440),  # dark steel-blue
    "top_10_depression_parent": (0.160, 0.360, 0.620),  # medium steel-blue
    "parent_dep_increase_2sd": (0.260, 0.020, 0.240),  # deep wine-plum
    "parent_dep_increase_1.5sd": (0.540, 0.220, 0.500),  # bright rose-plum
    "parent_dep_onset_rci_2.3": (0.580, 0.320, 0.480),  # dusty mauve
}

_DEP_WORSENING_BG = "#FFFFFF"
_CHILD_DEP_EDGE = "#0A2470"
_PARENT_DEP_EDGE = "#0C3058"

_CHILD_FEAT_TOP = (0.340, 0.520, 0.780)
_CHILD_FEAT_BOT = (0.035, 0.150, 0.500)
_PARENT_FEAT_TOP = (0.280, 0.480, 0.580)
_PARENT_FEAT_BOT = (0.060, 0.250, 0.380)


def _dep_feat_fill(pos_frac, top, bot):
    r = top[0] + (bot[0] - top[0]) * pos_frac
    g = top[1] + (bot[1] - top[1]) * pos_frac
    b = top[2] + (bot[2] - top[2]) * pos_frac
    return (r, g, b)


def _dep_worsening_sort_targets(targets_list, order, is_parent=False):
    return sorted(targets_list)


def _dep_worsening_color(target_name):
    if target_name in _DEP_WORSENING_COLORS:
        return _DEP_WORSENING_COLORS[target_name]
    tl = target_name.lower()
    _is_parent = "parent" in tl
    _is_top5 = "top_5" in tl or "top 5" in tl
    _is_top10 = "top_10" in tl or "top 10" in tl
    _is_rci = "rci" in tl or "onset" in tl
    _is_15sd = "1.5" in tl
    _is_2sd = not _is_rci and not _is_15sd and ("2 sd" in tl or "2sd" in tl or "\u2265 2" in tl)
    if _is_parent:
        if _is_top5:
            return _DEP_WORSENING_COLORS["top_5_depression_parent"]
        if _is_top10:
            return _DEP_WORSENING_COLORS["top_10_depression_parent"]
        if _is_rci:
            return _DEP_WORSENING_COLORS["parent_dep_onset_rci_2.3"]
        if _is_15sd:
            return _DEP_WORSENING_COLORS["parent_dep_increase_1.5sd"]
        return _DEP_WORSENING_COLORS["parent_dep_increase_2sd"]
    if _is_top5:
        return _DEP_WORSENING_COLORS["top_5_depression"]
    if _is_top10:
        return _DEP_WORSENING_COLORS["top_10_depression"]
    if _is_rci:
        return _DEP_WORSENING_COLORS["dep_onset_rci_2.3"]
    if _is_15sd:
        return _DEP_WORSENING_COLORS["dep_increase_1.5sd"]
    return _DEP_WORSENING_COLORS["dep_increase_2sd"]


_DEP_SANKEY_SUBTITLE = (
    "Across-Category Ensemble Classification Performance and Feature Importance Combined"
    " \u00b7 Depression Syndrome Severity: Top 5th/10th Percentile"
    " & Change from Prior Timepoint: \u2265 1.5 SD \u00b7 \u2265 2 SD"
)


def _draw_sankey_dep_worsening_shared(
    df_sankey, sankey_top_features, save_dir=None, is_parent=False, label_override=None
):
    """Shared implementation for child and parent depression worsening Sankeys."""
    if is_parent:
        _TARGET_ORDER = _PARENT_DEP_WORSENING_ORDER
        _EDGE_COL = _PARENT_DEP_EDGE
        _SAVE_NAME = "sankey_parent_clinical_depression"
        _FEAT_TOP, _FEAT_BOT = _PARENT_FEAT_BOT, _PARENT_FEAT_TOP
        _FEAT_EDGE = "#081829"
    else:
        _TARGET_ORDER = _CHILD_DEP_WORSENING_ORDER
        _EDGE_COL = _CHILD_DEP_EDGE
        _SAVE_NAME = "sankey_child_clinical_depression"
        _FEAT_TOP, _FEAT_BOT = _CHILD_FEAT_BOT, _CHILD_FEAT_TOP
        _FEAT_EDGE = "#091F38"
    _BG = _DEP_WORSENING_BG

    _cd = _sankey_common_data(df_sankey)
    targets_list = _dep_worsening_sort_targets(_cd["targets_list"], _TARGET_ORDER)
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: _dep_worsening_color(t) for t in targets_list}

    _pad = max(0.022, min(0.040, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.052
    _feat_nw = 0.028  # feature node half-width (square: 0.056 wide = _FEAT_H)

    # Width-first figure sizing
    fig_w, fig_h = _wide_figure_dims(n_targets, n_features)
    fig_w, fig_h = fig_w * 1.5, fig_h * 1.5  # scale up for higher resolution
    _spread_t = 4.5
    _spread_f = 1.9
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()
    _target_fs = 19  # override target label size

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )

    # Equalize vertical envelopes
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    # Normalize all target node heights to uniform size
    # Rect pad=0.010; pill pad=0.022 → shrink pill internal h to compensate.
    _RECT_H = 0.19
    _PILL_H = 0.16  # pill pad=0.016 compensated for ~equal rendered height
    if target_positions:
        for t in targets_list:
            _tl = t.lower()
            _is_sd = ("sd" in _tl or "≥" in t) and "top" not in _tl and "rci" not in _tl and "onset" not in _tl
            _new_h = _PILL_H if _is_sd else _RECT_H
            _c = target_positions[t]["y_centre"]
            target_positions[t] = {
                "y_centre": _c,
                "y_top": _c + _new_h / 2,
                "y_bot": _c - _new_h / 2,
                "h": _new_h,
            }

    # Normalize feature node heights to uniform
    _FEAT_H = 0.056
    if feature_positions:
        for f_key in features_list:
            if f_key in feature_positions:
                _c = feature_positions[f_key]["y_centre"]
                feature_positions[f_key] = {
                    "y_centre": _c,
                    "y_top": _c + _FEAT_H / 2,
                    "y_bot": _c - _FEAT_H / 2,
                    "h": _FEAT_H,
                }

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.28
    ax.set_xlim(-0.65, right_x + 0.78)
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    # Port gap: shrink each ribbon port slightly for visible separation
    _PORT_GAP = 0.002  # data-coord gap between adjacent ribbon ports

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])

        # Inset ports by half-gap on each side for visible separation
        _yl_t = rb["y_l_top"] - _PORT_GAP
        _yl_b = rb["y_l_bot"] + _PORT_GAP
        _yr_t = rb["y_r_top"] - _PORT_GAP * 0.5
        _yr_b = rb["y_r_bot"] + _PORT_GAP * 0.5
        if _yl_t <= _yl_b or _yr_t <= _yr_b:
            _yl_t, _yl_b = rb["y_l_top"], rb["y_l_bot"]
            _yr_t, _yr_b = rb["y_r_top"], rb["y_r_bot"]

        # Saturation + luminance ramp
        _desat = 0.55 + 0.45 * rel
        _white = (0.88, 0.90, 0.94)
        _fill = tuple(_white[i] + (_rgb[i] - _white[i]) * _desat for i in range(3))
        _alpha = 0.55 + 0.30 * rel

        # Glow halo on high-importance ribbons
        if rel > 0.45:
            _glow_n = 2 if rel > 0.75 else 1
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                _yl_t + 0.005,
                _yl_b - 0.005,
                _yr_t + 0.005,
                _yr_b - 0.005,
                color_rgba=_fill + (_alpha * 0.15,),
                zorder=1,
                glow_layers=_glow_n,
            )

        # Edge darkening on top ~30% (subtle)
        if rel > 0.70:
            _edge_dark = tuple(max(0, c * 0.70) for c in _rgb)
            _e_lw = 0.35 + 0.25 * (rel - 0.70) / 0.30
            _e_alpha = 0.22 + 0.18 * (rel - 0.70) / 0.30
            _edge_c = _edge_dark + (_e_alpha,)
        else:
            _edge_c = _fill + (min(_alpha + 0.05, 0.65),)
            _e_lw = 0.2

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            _yl_t,
            _yl_b,
            _yr_t,
            _yr_b,
            color_rgba=_fill + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_e_lw,
            right_node_w=_feat_nw,
        )

    for t, pos in target_positions.items():
        col = _tc.get(t, (0.5, 0.5, 0.5))
        _tl = t.lower()
        _is_sd = ("sd" in _tl or "\u2265" in t) and "top" not in _tl and "rci" not in _tl and "onset" not in _tl

        # SD-change targets → pill/capsule; top-percentile → sharp rect
        _bstyle = "round,pad=0.016" if _is_sd else "round,pad=0.008"

        ax.add_patch(
            FancyBboxPatch(
                (left_x - node_w + 0.004, pos["y_bot"] - 0.003),
                node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (left_x - node_w, pos["y_bot"]),
                node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor=col,
                edgecolor=_EDGE_COL,
                linewidth=1.8,
                zorder=3,
            )
        )

        display = _tdn.get(t, t)
        if label_override:
            display = label_override.get(t, label_override.get(display, display))
        _tm = _target_metrics.get(t) or _target_metrics.get(display)
        _label_x = left_x - node_w - 0.04

        if _tm is not None:
            _val, _lbl = _tm
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.015,
                display,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.018,
                f"{_lbl} = {_val:.3f}",
                ha="right",
                va="center",
                fontsize=_lf[2],
                fontweight="bold",
                color=_EDGE_COL,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                display,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count[f]
        dom_rows = df_sankey[df_sankey["Feature"] == f]
        dom_t = dom_rows.loc[dom_rows["Importance"].idxmax(), "Target"] if not dom_rows.empty else targets_list[0]
        glow_col = _tc.get(dom_t, (0.5, 0.5, 0.5))

        if n_tgt >= 2:
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - _feat_nw - 0.006, pos["y_bot"] - 0.006),
                    _feat_nw * 2 + 0.012,
                    pos["h"] + 0.012,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_col + (0.38,),
                    linewidth=1.3,
                    zorder=2,
                )
            )

        # Single-target features inherit their target color
        if n_tgt == 1:
            _fc = _tc.get(dom_t, (0.5, 0.5, 0.5))
        else:
            # Dark greyish-blue → black across 2T/3T/4T
            _t = min((n_tgt - 2) / 2.0, 1.0)  # 0 at 2T, 0.5 at 3T, 1.0 at 4T+
            _fc = (
                0.32 - 0.27 * _t,  # R: 0.32 → 0.05
                0.36 - 0.29 * _t,  # G: 0.36 → 0.07
                0.46 - 0.33 * _t,
            )  # B: 0.46 → 0.13
        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw + 0.004, pos["y_bot"] - 0.003),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw, pos["y_bot"]),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor=_fc,
                edgecolor=_FEAT_EDGE,
                linewidth=0.7,
                zorder=3,
            )
        )

        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + _feat_nw + 0.025,
            pos["y_centre"],
            f"{f}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color="#1a1a1a",
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    _rule_hw = 0.10
    ax.plot(
        [left_x - _rule_hw, left_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_EDGE_COL,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )
    ax.plot(
        [right_x - _rule_hw, right_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_EDGE_COL,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )

    ax.text(0.50, 1.085, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)

    plt.tight_layout(rect=[0.01, 0.06, 0.86, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, _SAVE_NAME, dpi=1200)
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    return fig


def _draw_sankey_child_dep_worsening(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_dep_worsening_shared(
        df_sankey, sankey_top_features, save_dir=save_dir, is_parent=False, label_override=_CHILD_LABEL_OVERRIDE
    )


def _draw_sankey_parent_dep_worsening(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_dep_worsening_shared(
        df_sankey, sankey_top_features, save_dir=save_dir, is_parent=True, label_override=None
    )


# Depression subtype sankeys: Female & Male Stratified

_SBT_BLUE_TOP = (0.114, 0.302, 0.580)

_SBT_TARGET_COLORS = {
    "sbt_core_depression": (0.114, 0.302, 0.580),
    "sbt_anxiety_depression": (0.690, 0.125, 0.188),
    "sbt_guilt_hopelessness": (0.227, 0.227, 0.290),
    "sbt_fatigue_sleep": (0.165, 0.478, 0.478),
    "sbt_somatic_depression": (0.545, 0.353, 0.169),
    "sbt_emotional_dysregulation": (0.753, 0.416, 0.063),
    "sbt_social_withdrawal": (0.353, 0.439, 0.533),
    "sbt_avoidance_fear": (0.478, 0.227, 0.541),
    "sbt_aggression_irritability": (0.176, 0.416, 0.180),
    "sbt_academic_cognitive": (0.722, 0.588, 0.043),
    "sbt_perfectionism_achievement": (0.769, 0.290, 0.502),
}

_SBT_TARGET_ORDER = [
    "sbt_core_depression",
    "sbt_anxiety_depression",
    "sbt_guilt_hopelessness",
    "sbt_fatigue_sleep",
    "sbt_somatic_depression",
    "sbt_emotional_dysregulation",
    "sbt_social_withdrawal",
    "sbt_avoidance_fear",
    "sbt_aggression_irritability",
    "sbt_academic_cognitive",
    "sbt_perfectionism_achievement",
]

_SBT_DISPLAY_NAMES = {
    "sbt_core_depression": "Core Depression",
    "sbt_anxiety_depression": "Anxiety\u2013Depression",
    "sbt_guilt_hopelessness": "Guilt & Hopelessness",
    "sbt_fatigue_sleep": "Fatigue & Sleep",
    "sbt_somatic_depression": "Somatic Depression",
    "sbt_emotional_dysregulation": "Emotional Dysregulation",
    "sbt_social_withdrawal": "Social Withdrawal",
    "sbt_avoidance_fear": "Avoidance & Fear",
    "sbt_aggression_irritability": "Aggression & Irritability",
    "sbt_academic_cognitive": "Academic & Cognitive",
    "sbt_perfectionism_achievement": "Perfectionism & Achievement",
}

_SBT_THEMES = {
    "female": {
        "bg_color": "#FFFFFF",
        "edge_color": "#6B2040",
        "text_color": "#1a1a1a",
        "hdr_color": "#6B2040",
        "save_name": "sankey_sbt_female",
        "title": "Depression Subtype Predictive Feature Architecture (Female)",
    },
    "long": {
        "bg_color": "#FFFFFF",
        "edge_color": "#2A3A5A",
        "text_color": "#1a1a1a",
        "hdr_color": "#2A3A5A",
        "save_name": "sankey_sbt_long",
        "title": "Depression Subtype Predictive Feature Architecture (Longitudinal)",
    },
}


def _sbt_blend_color(pos_frac, subtype_rgb):
    r = _SBT_BLUE_TOP[0] + (subtype_rgb[0] - _SBT_BLUE_TOP[0]) * pos_frac
    g = _SBT_BLUE_TOP[1] + (subtype_rgb[1] - _SBT_BLUE_TOP[1]) * pos_frac
    b = _SBT_BLUE_TOP[2] + (subtype_rgb[2] - _SBT_BLUE_TOP[2]) * pos_frac
    return (r, g, b)


def _sbt_sort_targets(targets_list):
    return sorted(targets_list)


def _sbt_sort_targets_by_perf(targets_list, target_metrics):
    if not target_metrics:
        return _sbt_sort_targets(targets_list)

    def _get_score(t):
        tm = target_metrics.get(t)
        if tm is None:
            tm = target_metrics.get(_sbt_display(t))
        if tm is None:
            _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
            tm = target_metrics.get(_tdn.get(t, ""))
        return tm[0] if tm is not None else -1.0

    return sorted(targets_list, key=_get_score, reverse=True)


def _sbt_resolve_color(target_name):
    if target_name in _SBT_TARGET_COLORS:
        return _SBT_TARGET_COLORS[target_name]
    for raw, display in _SBT_DISPLAY_NAMES.items():
        if target_name == display:
            return _SBT_TARGET_COLORS[raw]
    tl = target_name.lower()
    # Explicit substring fallbacks for display-name variants
    if "core" in tl or "primary" in tl or ("mood" in tl and "depress" in tl):
        return _SBT_TARGET_COLORS["sbt_core_depression"]
    if "anxiety" in tl:
        return _SBT_TARGET_COLORS["sbt_anxiety_depression"]
    if "guilt" in tl or "hopeless" in tl:
        return _SBT_TARGET_COLORS["sbt_guilt_hopelessness"]
    if "fatigue" in tl or "sleep" in tl:
        return _SBT_TARGET_COLORS["sbt_fatigue_sleep"]
    if "somatic" in tl:
        return _SBT_TARGET_COLORS["sbt_somatic_depression"]
    if "emotional" in tl or "dysregul" in tl:
        return _SBT_TARGET_COLORS["sbt_emotional_dysregulation"]
    if "social" in tl or "withdrawal" in tl:
        return _SBT_TARGET_COLORS["sbt_social_withdrawal"]
    if "avoidance" in tl or "fear" in tl:
        return _SBT_TARGET_COLORS["sbt_avoidance_fear"]
    if "aggress" in tl or "irritab" in tl:
        return _SBT_TARGET_COLORS["sbt_aggression_irritability"]
    if "academic" in tl or "cognitive" in tl:
        return _SBT_TARGET_COLORS["sbt_academic_cognitive"]
    if "perfect" in tl or "achiev" in tl:
        return _SBT_TARGET_COLORS["sbt_perfectionism_achievement"]
    return (0.4, 0.4, 0.4)


def _sbt_display(target_name):
    if target_name in _SBT_DISPLAY_NAMES:
        return _SBT_DISPLAY_NAMES[target_name]
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    return _tdn.get(target_name, target_name)


def _draw_sankey_sbt(df_sankey, sankey_top_features, save_dir=None, sex="female"):
    """Depression Subtype Sankey: sex-stratified."""
    theme = _SBT_THEMES[sex]
    _BG = theme["bg_color"]
    _EDGE_COL = theme["edge_color"]
    _SAVE_NAME = theme["save_name"]

    _cd = _sankey_common_data(df_sankey)
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})

    targets_list = _sbt_sort_targets_by_perf(_cd["targets_list"], _target_metrics)

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {}
    _t_imp_range = {}
    for t in targets_list:
        _tc[t] = _sbt_resolve_color(t)
        t_rows = df_sankey[df_sankey["Target"] == t]
        if not t_rows.empty:
            _t_imp_range[t] = (t_rows["Importance"].min(), t_rows["Importance"].max())
        else:
            _t_imp_range[t] = (0.0, 1.0)

    _pad = max(0.015, min(0.035, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.028
    _target_node_w = 0.036

    # Reverse layout: targets on right, features on left for 'long'
    _reverse = sex == "long"
    if _reverse:
        _tgt_x = right_x  # targets on the right
        _feat_x = left_x  # features on the left
        _tgt_label_ha = "left"
        _feat_label_ha = "right"
        _tgt_label_dx = _target_node_w + 0.022  # label offset from tgt_x
        _feat_label_dx = -(node_w + 0.020)  # label offset from feat_x
    else:
        _tgt_x = left_x  # targets on the left (default)
        _feat_x = right_x  # features on the right
        _tgt_label_ha = "right"
        _feat_label_ha = "left"
        _tgt_label_dx = -(_target_node_w + 0.022)
        _feat_label_dx = node_w + 0.020

    # Width-first figure sizing
    fig_w, fig_h = _wide_figure_dims(n_targets, n_features)
    _spread_t = 2.5
    _spread_f = 1.5
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )

    # Equalize vertical envelopes: 80 % for 11 subtypes
    target_positions, feature_positions = _equalize_vertical_envelopes(
        target_positions, feature_positions, target_span_frac=0.80
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04
    ax.set_xlim(-0.65, right_x + 0.65)
    ax.set_ylim(y_min, 1.08)
    ax.axis("off")

    # Ribbons
    # Uses the full subtype color directly (no navy→color blend), with alpha
    # encoding importance: matching the pfactor_spectra visual style.
    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        t = rb["target"]
        _rgb = _tc.get(t, (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])
        _alpha = 0.18 + 0.67 * rel

        if _reverse:
            _rib_left_nw = node_w
            _rib_right_nw = _target_node_w
            _yl_top, _yl_bot = rb["y_r_top"], rb["y_r_bot"]
            _yr_top, _yr_bot = rb["y_l_top"], rb["y_l_bot"]
        else:
            _rib_left_nw = _target_node_w
            _rib_right_nw = node_w
            _yl_top, _yl_bot = rb["y_l_top"], rb["y_l_bot"]
            _yr_top, _yr_bot = rb["y_r_top"], rb["y_r_bot"]

        # Glow halo for high-importance ribbons
        if rel > 0.60:
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                _rib_left_nw,
                right_x,
                _yl_top + 0.005,
                _yl_bot - 0.005,
                _yr_top + 0.005,
                _yr_bot - 0.005,
                color_rgba=_rgb + (_alpha * 0.35,),
                zorder=1,
                right_node_w=_rib_right_nw,
            )

        # Low-importance ribbons: white edges for visual separation
        if rel < 0.35:
            _edge_c = (1.0, 1.0, 1.0, 0.30)
            _edge_lw = 0.5
        else:
            _edge_c = _rgb + (min(_alpha + 0.10, 0.85),)
            _edge_lw = 0.4

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            _rib_left_nw,
            right_x,
            _yl_top,
            _yl_bot,
            _yr_top,
            _yr_bot,
            color_rgba=_rgb + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_edge_lw,
            right_node_w=_rib_right_nw,
        )

    # Target nodes
    for i, (t, pos) in enumerate(zip(targets_list, [target_positions[t] for t in targets_list])):
        col = _tc[t]

        ax.add_patch(
            FancyBboxPatch(
                (_tgt_x - _target_node_w + 0.005, pos["y_bot"] - 0.005),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.008",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.18,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (_tgt_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.008",
                facecolor=col,
                edgecolor=_EDGE_COL,
                linewidth=2.2,
                zorder=3,
            )
        )

        display = _sbt_display(t)
        _tm = _target_metrics.get(t) or _target_metrics.get(display)
        _label_x = _tgt_x + _tgt_label_dx

        if _tm is not None:
            _val, _lbl = _tm
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.018,
                display,
                ha=_tgt_label_ha,
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=theme["text_color"],
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.025,
                f"{_lbl} = {_val:.3f}",
                ha=_tgt_label_ha,
                va="center",
                fontsize=_lf[2],
                fontweight="bold",
                color=_EDGE_COL,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                display,
                ha=_tgt_label_ha,
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=theme["text_color"],
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])

    # Feature nodes: target-inherited color for single-target, dark blue for multi
    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count[f]
        dom_rows = df_sankey[df_sankey["Feature"] == f].sort_values("Importance", ascending=False)
        dom_t = dom_rows.iloc[0]["Target"] if not dom_rows.empty else targets_list[0]

        _nx = _feat_x - node_w
        _nw = node_w * 2
        _ny = pos["y_bot"]
        _nh = pos["h"]

        ax.add_patch(
            FancyBboxPatch(
                (_nx + 0.004, _ny - 0.004),
                _nw,
                _nh,
                boxstyle="round,pad=0.006",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.18,
            )
        )

        if n_tgt >= 2:
            top2_targets = dom_rows["Target"].head(2).tolist()
            glow_outer = _tc.get(top2_targets[0], (0.5, 0.5, 0.5))
            glow_inner = _tc.get(top2_targets[1], (0.5, 0.5, 0.5)) if len(top2_targets) > 1 else glow_outer
            ax.add_patch(
                FancyBboxPatch(
                    (_nx - 0.007, _ny - 0.007),
                    _nw + 0.014,
                    _nh + 0.014,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_outer + (0.50,),
                    linewidth=2.0,
                    zorder=2,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    (_nx - 0.004, _ny - 0.004),
                    _nw + 0.008,
                    _nh + 0.008,
                    boxstyle="round,pad=0.006",
                    facecolor="none",
                    edgecolor=glow_inner + (0.45,),
                    linewidth=2.0,
                    zorder=2,
                )
            )

        # Single-target features inherit that target's color;
        # multi-target features get very dark blue.
        if n_tgt == 1:
            _fc = _tc.get(dom_t, (0.5, 0.5, 0.5))
        else:
            _fc = (0.06, 0.10, 0.22)

        ax.add_patch(
            FancyBboxPatch(
                (_nx, _ny),
                _nw,
                _nh,
                boxstyle="round,pad=0.008",
                facecolor=_fc,
                edgecolor="#444444",
                linewidth=1.0,
                zorder=3,
            )
        )

        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            _feat_x + _feat_label_dx,
            pos["y_centre"],
            f"{f}{badge}",
            ha=_feat_label_ha,
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color=theme["text_color"],
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])

    # Column headers
    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    ax.text(
        _tgt_x,
        _hdr_y,
        "SUBTYPES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        _feat_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    ax.plot(
        [_tgt_x - _target_node_w * 1.5, _tgt_x + _target_node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=_EDGE_COL,
        lw=1.0,
        alpha=0.5,
        clip_on=False,
        zorder=5,
    )
    ax.plot(
        [_feat_x - node_w * 1.5, _feat_x + node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=_EDGE_COL,
        lw=1.0,
        alpha=0.5,
        clip_on=False,
        zorder=5,
    )

    ax.text(0.50, 1.06, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)
    # Subtitle at bottom with methodology notes
    ax.text(
        0.50,
        -0.015,
        f"Top {sankey_top_features} Consensus Features per Subtype  \u00b7  "
        "Shared Features Annotated (nT)  \u00b7  "
        "R\u00b2 from Across-Category Analyses\n"
        "Ribbon Opacity \u221d Weighted Ensemble Importance  \u00b7  "
        "Ribbon Color = Subtype Identity  \u00b7  "
        "Circular Features Removed",
        ha="center",
        va="top",
        fontsize=_lf[3],
        fontweight="bold",
        color="#1a1a1a",
        fontfamily=_TITLE_FONT,
        linespacing=1.4,
        transform=ax.transAxes,
    )

    plt.tight_layout(rect=[0.01, 0.015, 0.88, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, _SAVE_NAME, dpi=600)
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    return fig


def _draw_sankey_sbt_female(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_sbt(df_sankey, sankey_top_features, save_dir=save_dir, sex="female")


def _draw_sankey_sbt_long(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_sbt(df_sankey, sankey_top_features, save_dir=save_dir, sex="long")


# Primary targets: Matched Child CBCL ↔ Parent ASR DSM-Oriented Scales
# 8 targets: 4 child (brighter) + 4 parent (deeper/muted) paired by construct.
# Visual pairing: same hue family, child=saturated, parent=desaturated.

_PRIMARY_TARGETS_COLORS = {
    # Child (CBCL DSM-oriented): saturated
    "depress_D_p": (0.15, 0.39, 0.91),  # vivid blue
    "anxdisord_D_p": (0.85, 0.18, 0.18),  # red
    "adhd_D_p": (0.82, 0.68, 0.15),  # gold
    "social_problems_D_p": (0.90, 0.50, 0.13),  # amber-orange
    # Parent (ASR analogues): deeper/muted
    "parent_depress_D_p": (0.20, 0.28, 0.62),  # deep blue
    "parent_anxdisord_D_p": (0.62, 0.12, 0.12),  # dark red
    "parent_adhd_D_p": (0.65, 0.55, 0.10),  # dark gold
    "parent_bad_relationships_p": (0.70, 0.30, 0.05),  # burnt sienna
}

_PRIMARY_TARGETS_ORDER = [
    # Child block
    "depress_D_p",
    "anxdisord_D_p",
    "adhd_D_p",
    "social_problems_D_p",
    # Parent block
    "parent_depress_D_p",
    "parent_anxdisord_D_p",
    "parent_adhd_D_p",
    "parent_bad_relationships_p",
]

_PRIMARY_TARGETS_DISPLAY = {
    "depress_D_p": "Depression Syndrome",
    "anxdisord_D_p": "Anxiety Disorder",
    "adhd_D_p": "ADHD",
    "social_problems_D_p": "Social Problems",
    "parent_depress_D_p": "Parent Depression Syndrome",
    "parent_anxdisord_D_p": "Parent Anxiety Disorder",
    "parent_adhd_D_p": "Parent ADHD",
    "parent_bad_relationships_p": "Parent Interpersonal Difficulties",
}

_PRIMARY_TARGETS_EDGE = "#1E293B"
_PRIMARY_TARGETS_BG = "#FFFFFF"


def _primary_targets_resolve_color(target_name):
    if target_name in _PRIMARY_TARGETS_COLORS:
        return _PRIMARY_TARGETS_COLORS[target_name]
    # Reverse lookup by display name
    for raw, display in _PRIMARY_TARGETS_DISPLAY.items():
        if target_name == display:
            return _PRIMARY_TARGETS_COLORS[raw]
    # Substring fallback
    tl = target_name.lower()
    if "parent" in tl:
        if "depress" in tl:
            return _PRIMARY_TARGETS_COLORS["parent_depress_D_p"]
        if "anx" in tl:
            return _PRIMARY_TARGETS_COLORS["parent_anxdisord_D_p"]
        if "adhd" in tl:
            return _PRIMARY_TARGETS_COLORS["parent_adhd_D_p"]
        if "relation" in tl or "social" in tl or "interpers" in tl:
            return _PRIMARY_TARGETS_COLORS["parent_bad_relationships_p"]
    else:
        if "depress" in tl:
            return _PRIMARY_TARGETS_COLORS["depress_D_p"]
        if "anx" in tl:
            return _PRIMARY_TARGETS_COLORS["anxdisord_D_p"]
        if "adhd" in tl:
            return _PRIMARY_TARGETS_COLORS["adhd_D_p"]
        if "social" in tl:
            return _PRIMARY_TARGETS_COLORS["social_problems_D_p"]
    return (0.45, 0.45, 0.50)


def _primary_targets_edge_for(target_name):
    rgb = _primary_targets_resolve_color(target_name)
    return tuple(max(0, c * 0.6) for c in rgb)


def _primary_targets_sort(targets_list):
    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}

    def _score(t):
        entry = _tm.get(t)
        if entry is None:
            entry = _tm.get(_tdn.get(t, ""))
        if entry is None:
            entry = _tm.get(_PRIMARY_TARGETS_DISPLAY.get(t, ""))
        return entry[0] if entry is not None else -1.0

    scores = {t: _score(t) for t in targets_list}
    has_scores = any(v > 0 for v in scores.values())

    if has_scores:
        # Sort within child and parent blocks separately by performance
        child = [t for t in targets_list if "parent" not in t.lower()]
        parent = [t for t in targets_list if "parent" in t.lower()]
        child.sort(key=lambda t: scores.get(t, -1), reverse=True)
        parent.sort(key=lambda t: scores.get(t, -1), reverse=True)
        return child + parent

    # Fallback: canonical order


def _draw_sankey_primary_targets(df_sankey, sankey_top_features, save_dir=None):
    """Primary Targets Sankey: matched child↔parent DSM-oriented scales.
    Standard layout (targets left, features right) with full visual polish."""

    _BG = _PRIMARY_TARGETS_BG
    _EDGE_COL = _PRIMARY_TARGETS_EDGE
    text_color = "#1a1a1a"

    _cd = _sankey_common_data(df_sankey)
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    # Metrics suppressed for manual Illustrator annotation
    _target_metrics = {}
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})

    targets_list = _primary_targets_sort(_cd["targets_list"])

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    target_colors = {t: _primary_targets_resolve_color(t) for t in targets_list}

    # Display name resolution
    def _display_target(t):
        d = _PRIMARY_TARGETS_DISPLAY.get(t)
        if d:
            return d
        d = _tdn.get(t)
        if d:
            return d
        return t

    # Dominant target per feature (for single-target coloring)
    _feat_dom_target = {}
    for f in features_list:
        frows = df_sankey[df_sankey["Feature"] == f].sort_values("Importance", ascending=False)
        _feat_dom_target[f] = frows.iloc[0]["Target"] if not frows.empty else targets_list[0]

    _pad = max(0.015, min(0.035, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.028
    _target_node_w = 0.038

    # Figure sizing
    fig_w, fig_h = _wide_figure_dims(n_targets, n_features)
    _spread_t = 2.2
    _spread_f = 1.5
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )

    # Equalize vertical envelopes
    target_positions, feature_positions = _equalize_vertical_envelopes(
        target_positions, feature_positions, target_span_frac=0.82
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04
    ax.set_xlim(-0.65, right_x + 0.65)
    ax.set_ylim(y_min, 1.08)
    ax.axis("off")

    # Child/Parent divider line between target blocks
    child_targets = [t for t in targets_list if "parent" not in t.lower()]
    parent_targets = [t for t in targets_list if "parent" in t.lower()]
    if child_targets and parent_targets:
        last_child_y = target_positions[child_targets[-1]]["y_bot"]
        first_parent_y = target_positions[parent_targets[0]]["y_top"]
        _div_y = (last_child_y + first_parent_y) / 2
        ax.plot(
            [left_x - _target_node_w * 2.5, left_x + _target_node_w * 2.5],
            [_div_y, _div_y],
            color="#CBD5E1",
            lw=0.8,
            ls="--",
            alpha=0.6,
            zorder=1,
        )
        ax.text(
            left_x - _target_node_w * 2.8,
            _div_y + 0.012,
            "CHILD",
            fontsize=7,
            color="#94A3B8",
            fontweight="600",
            ha="right",
            fontfamily=_TITLE_FONT,
            alpha=0.7,
        )
        ax.text(
            left_x - _target_node_w * 2.8,
            _div_y - 0.016,
            "PARENT",
            fontsize=7,
            color="#94A3B8",
            fontweight="600",
            ha="right",
            fontfamily=_TITLE_FONT,
            alpha=0.7,
        )

    # Ribbons (full-color + alpha, pfactor style)
    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = target_colors.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])
        _alpha = 0.18 + 0.67 * rel

        # Glow halo for high-importance ribbons
        if rel > 0.60:
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                _target_node_w,
                right_x,
                rb["y_l_top"] + 0.005,
                rb["y_l_bot"] - 0.005,
                rb["y_r_top"] + 0.005,
                rb["y_r_bot"] - 0.005,
                color_rgba=_rgb + (_alpha * 0.35,),
                zorder=1,
                right_node_w=node_w,
            )

        # Low-importance: white edges for separation
        if rel < 0.35:
            _edge_c = (1.0, 1.0, 1.0, 0.30)
            _edge_lw = 0.5
        else:
            _edge_c = _rgb + (min(_alpha + 0.10, 0.85),)
            _edge_lw = 0.4

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            _target_node_w,
            right_x,
            rb["y_l_top"],
            rb["y_l_bot"],
            rb["y_r_top"],
            rb["y_r_bot"],
            color_rgba=_rgb + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_edge_lw,
            right_node_w=node_w,
        )

    # Target nodes (left)
    for t, pos in target_positions.items():
        rgb = target_colors[t]
        edge_c = _primary_targets_edge_for(t)

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.006, pos["y_bot"] - 0.006),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.007",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.22,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle="round,pad=0.007",
                facecolor=rgb,
                edgecolor=edge_c,
                linewidth=2.2,
                zorder=3,
            )
        )

        _disp_t = _display_target(t)
        _tm = _target_metrics.get(t) or _target_metrics.get(_disp_t)
        _label_x = left_x - _target_node_w - 0.025

        if _tm is not None:
            _val, _lbl = _tm
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.018,
                _disp_t,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=text_color,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])
            _metric_color = edge_c
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.022,
                f"{_lbl} = {_val:.3f}",
                ha="right",
                va="center",
                fontsize=_lf[2],
                fontweight="bold",
                color=_metric_color,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _disp_t,
                ha="right",
                va="center",
                fontsize=_lf[0],
                fontweight="bold",
                color=text_color,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=_BG)])

    # Feature nodes (right)
    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))

    for f, pos in feature_positions.items():
        n_tgt = feat_target_count[f]
        dom_t = _feat_dom_target[f]
        fc = target_colors.get(dom_t, (0.5, 0.5, 0.5)) if n_tgt == 1 else (0.42, 0.48, 0.58)

        _nx = right_x - node_w
        _nw = node_w * 2
        _ny = pos["y_bot"]
        _nh = pos["h"]

        # Drop shadow
        ax.add_patch(
            FancyBboxPatch(
                (_nx + 0.004, _ny - 0.004),
                _nw,
                _nh,
                boxstyle="round,pad=0.006",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.18,
            )
        )

        # Multi-target glow rings
        if n_tgt >= 2:
            dom_rows = df_sankey[df_sankey["Feature"] == f].sort_values("Importance", ascending=False)
            top2 = dom_rows["Target"].head(2).tolist()
            glow_outer = target_colors.get(top2[0], (0.5, 0.5, 0.5))
            glow_inner = target_colors.get(top2[1], (0.5, 0.5, 0.5)) if len(top2) > 1 else glow_outer
            ax.add_patch(
                FancyBboxPatch(
                    (_nx - 0.007, _ny - 0.007),
                    _nw + 0.014,
                    _nh + 0.014,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_outer + (0.45,),
                    linewidth=2.0,
                    zorder=2,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    (_nx - 0.004, _ny - 0.004),
                    _nw + 0.008,
                    _nh + 0.008,
                    boxstyle="round,pad=0.006",
                    facecolor="none",
                    edgecolor=glow_inner + (0.40,),
                    linewidth=2.0,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (_nx, _ny),
                _nw,
                _nh,
                boxstyle="round,pad=0.006",
                facecolor=fc,
                edgecolor="#444444",
                linewidth=1.0,
                zorder=3,
            )
        )

        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + node_w + 0.020,
            pos["y_centre"],
            f"{f}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color=text_color,
            fontfamily=_TITLE_FONT,
            zorder=5,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=_BG)])

    # Column headers
    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.045
    )
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    ax.plot(
        [left_x - _target_node_w * 1.5, left_x + _target_node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=_EDGE_COL,
        lw=1.0,
        alpha=0.5,
        clip_on=False,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_EDGE_COL,
        fontfamily=_TITLE_FONT,
    )
    ax.plot(
        [right_x - node_w * 1.5, right_x + node_w * 1.5],
        [_hdr_y - 0.003, _hdr_y - 0.003],
        color=_EDGE_COL,
        lw=1.0,
        alpha=0.5,
        clip_on=False,
    )

    # Invisible spacer above the plot, then the subtitle
    ax.text(0.50, 1.06, " ", ha="center", va="bottom", fontsize=15, transform=ax.transAxes, alpha=0.0)
    ax.text(
        0.50,
        -0.015,
        f"Top {sankey_top_features} Consensus Features per Target  \u00b7  "
        "Shared Features Annotated (nT)  \u00b7  "
        "Matched Child CBCL \u2194 Parent ASR DSM-Oriented Scales\n"
        "Ribbon Opacity \u221d Weighted Ensemble Importance  \u00b7  "
        "Ribbon Color = Target Identity  \u00b7  "
        "Circular Features Removed",
        ha="center",
        va="top",
        fontsize=_lf[3],
        fontweight="bold",
        color="#1a1a1a",
        fontfamily=_TITLE_FONT,
        linespacing=1.4,
        transform=ax.transAxes,
    )

    plt.tight_layout(rect=[0, 0.015, 0.90, 0.98])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, "sankey_primary_targets", dpi=600)
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    return fig


# Child deltas: domain-colored Sankey for Child Only Deltas batch

_CHILD_DELTAS_COLORS = {
    "delta_depress_D_p": (0.10, 0.28, 0.62),  # royal blue
    "delta_sbt_core_depression": (0.16, 0.42, 0.80),  # lighter blue
    "delta_anxdisord_D_p": (0.72, 0.14, 0.14),  # crimson
    "delta_adhd_D_p": (0.80, 0.68, 0.12),  # gold
    "delta_social_problems_D_p": (0.85, 0.55, 0.10),  # dark orange
    "delta_somatic_problems_D_p": (0.60, 0.45, 0.22),  # peru
    "delta_bad_grades": (0.44, 0.50, 0.56),  # slate gray
    "delta_family_conflict_ss_k": (0.62, 0.32, 0.12),  # sienna
}

_CHILD_DELTAS_DISPLAY_ALIASES = {
    "Depression Syndrome (Change Score)": "delta_depress_D_p",
    "Change in Depression": "delta_depress_D_p",
    "Mood/Depression (Change Score)": "delta_sbt_core_depression",
    "Change in Anxiety": "delta_anxdisord_D_p",
    "Anxiety (Change Score)": "delta_anxdisord_D_p",
    "Change in ADHD": "delta_adhd_D_p",
    "Change in Social Problems": "delta_social_problems_D_p",
    "Change in Somatic Issues": "delta_somatic_problems_D_p",
    "Change in Bad Grades": "delta_bad_grades",
    "Change in Fam. Conflict (Y)": "delta_family_conflict_ss_k",
}

_CHILD_DELTAS_ORDER = [
    "delta_depress_D_p",
    "delta_sbt_core_depression",
    "delta_anxdisord_D_p",
    "delta_adhd_D_p",
    "delta_social_problems_D_p",
    "delta_somatic_problems_D_p",
    "delta_bad_grades",
    "delta_family_conflict_ss_k",
]

_CHILD_DELTAS_EDGE = "#1A2332"
_CHILD_DELTAS_BG = "#FFFFFF"


def _child_deltas_resolve_color(target_name):
    if target_name in _CHILD_DELTAS_COLORS:
        return _CHILD_DELTAS_COLORS[target_name]
    raw = _CHILD_DELTAS_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_DELTAS_COLORS:
        return _CHILD_DELTAS_COLORS[raw]
    tl = target_name.lower()
    if "mood" in tl or "sbt_core" in tl:
        return _CHILD_DELTAS_COLORS["delta_sbt_core_depression"]
    if "depress" in tl or "dep_" in tl:
        return _CHILD_DELTAS_COLORS["delta_depress_D_p"]
    if "anxi" in tl:
        return _CHILD_DELTAS_COLORS["delta_anxdisord_D_p"]
    if "adhd" in tl or "attention" in tl:
        return _CHILD_DELTAS_COLORS["delta_adhd_D_p"]
    if "social" in tl:
        return _CHILD_DELTAS_COLORS["delta_social_problems_D_p"]
    if "somatic" in tl:
        return _CHILD_DELTAS_COLORS["delta_somatic_problems_D_p"]
    if "grade" in tl or "academic" in tl:
        return _CHILD_DELTAS_COLORS["delta_bad_grades"]
    if "conflict" in tl or "family" in tl:
        return _CHILD_DELTAS_COLORS["delta_family_conflict_ss_k"]
    return (0.5, 0.5, 0.5)


def _child_deltas_sort_targets(targets_list):
    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}


def _child_deltas_edge_for(target_name):
    rgb = _child_deltas_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


def _draw_sankey_child_deltas(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_child_deltas_sort_targets,
        color_fn=_child_deltas_resolve_color,
        edge_fn=_child_deltas_edge_for,
        edge_col=_CHILD_DELTAS_EDGE,
        bg_color=_CHILD_DELTAS_BG,
        save_name="sankey_child_deltas",
    )


# Parent deltas: domain-colored Sankey for Parent Only Deltas batch

_PARENT_DELTAS_COLORS = {
    # Depression / Mood domain (blue family)
    "delta_parent_depress_D_p": (0.08, 0.12, 0.42),  # deep navy
    "delta_parent_feels_unloved_p": (0.22, 0.38, 0.62),  # steel blue
    "delta_parent_sleep_trouble_p": (0.18, 0.22, 0.52),  # slate indigo
    # Anxiety domain (red family)
    "delta_parent_worries_about_family_p": (0.72, 0.22, 0.18),  # indian red
    "delta_parent_worries_about_future_p": (0.58, 0.12, 0.12),  # firebrick
    # ADHD / Attention domain (purple)
    "delta_parent_concentration_trouble_p": (0.50, 0.28, 0.68),  # medium purple
    # Social domain (teal family)
    "delta_parent_not_liked_by_others_p": (0.08, 0.48, 0.48),  # dark cyan
    "delta_parent_bad_relationships_p": (0.06, 0.42, 0.38),  # teal
    # Family conflict (warm orange)
    "delta_family_conflict_ss_p": (0.62, 0.32, 0.12),  # sienna
    # SES / Financial (stone)
    "delta_parent_financial_trouble_p": (0.40, 0.40, 0.40),  # dim gray
    # Somatic (earthy brown)
    "delta_parent_somatic_problems_D_p": (0.60, 0.45, 0.22),  # peru
}

_PARENT_DELTAS_DISPLAY_ALIASES = {
    "Parent Depression Syndrome (Change Score)": "delta_parent_depress_D_p",
    "Change in Par. Feels Unloved": "delta_parent_feels_unloved_p",
    "Change in Par. Sleep": "delta_parent_sleep_trouble_p",
    "Change in Par. Fam. Worry": "delta_parent_worries_about_family_p",
    "Change in Par. Future Worry": "delta_parent_worries_about_future_p",
    "Change in Par. Concentration": "delta_parent_concentration_trouble_p",
    "Change in Par. Unliked": "delta_parent_not_liked_by_others_p",
    "Change in Par. General Relationship Quality": "delta_parent_bad_relationships_p",
    "Change in Fam. Conflict": "delta_family_conflict_ss_p",
    "Change in Par. Finances": "delta_parent_financial_trouble_p",
    "Change in Par. Somatic": "delta_parent_somatic_problems_D_p",
}

# Display order: depression → anxiety → ADHD → social → family → financial → somatic
_PARENT_DELTAS_ORDER = [
    "delta_parent_depress_D_p",
    "delta_parent_feels_unloved_p",
    "delta_parent_sleep_trouble_p",
    "delta_parent_worries_about_family_p",
    "delta_parent_worries_about_future_p",
    "delta_parent_concentration_trouble_p",
    "delta_parent_not_liked_by_others_p",
    "delta_parent_bad_relationships_p",
    "delta_family_conflict_ss_p",
    "delta_parent_financial_trouble_p",
    "delta_parent_somatic_problems_D_p",
]

_PARENT_DELTAS_EDGE = "#2C3440"
_PARENT_DELTAS_BG = "#FFFFFF"


def _parent_deltas_resolve_color(target_name):
    if target_name in _PARENT_DELTAS_COLORS:
        return _PARENT_DELTAS_COLORS[target_name]
    raw = _PARENT_DELTAS_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _PARENT_DELTAS_COLORS:
        return _PARENT_DELTAS_COLORS[raw]
    tl = target_name.lower()
    # Substring fallbacks for any display-name variants
    if "depress" in tl or "dep_" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_depress_D_p"]
    if "unloved" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_feels_unloved_p"]
    if "sleep" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_sleep_trouble_p"]
    if "fam. worry" in tl or "family_p" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_worries_about_family_p"]
    if "future" in tl or "worry" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_worries_about_future_p"]
    if "concentrat" in tl or "attention" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_concentration_trouble_p"]
    if "unliked" in tl or "not_liked" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_not_liked_by_others_p"]
    if "relationship" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_bad_relationships_p"]
    if "conflict" in tl:
        return _PARENT_DELTAS_COLORS["delta_family_conflict_ss_p"]
    if "financ" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_financial_trouble_p"]
    if "somatic" in tl:
        return _PARENT_DELTAS_COLORS["delta_parent_somatic_problems_D_p"]
    return (0.5, 0.5, 0.5)


def _parent_deltas_sort_targets(targets_list):
    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}

    def _r2_for(t):
        entry = _tm.get(t)
        if entry is None:
            entry = _tm.get(_tdn.get(t, ""))
        if entry is None:
            raw = _PARENT_DELTAS_DISPLAY_ALIASES.get(t)
            if raw:
                entry = _tm.get(raw) or _tm.get(_tdn.get(raw, ""))
        return entry[0] if entry is not None else None

    r2_vals = {t: _r2_for(t) for t in targets_list}
    has_metrics = any(v is not None for v in r2_vals.values())

    if has_metrics:
        return sorted(targets_list, key=lambda t: r2_vals.get(t) if r2_vals.get(t) is not None else -999, reverse=True)

    def _rank(t):
        if t in _PARENT_DELTAS_ORDER:
            return _PARENT_DELTAS_ORDER.index(t)
        raw = _PARENT_DELTAS_DISPLAY_ALIASES.get(t)
        if raw and raw in _PARENT_DELTAS_ORDER:
            return _PARENT_DELTAS_ORDER.index(raw)
        tl = t.lower()
        if "depress" in tl:
            return 0
        if "unloved" in tl:
            return 1
        if "sleep" in tl:
            return 2
        if "fam. worry" in tl:
            return 3
        if "future" in tl:
            return 4
        if "concentrat" in tl:
            return 5
        if "unliked" in tl:
            return 6
        if "relationship" in tl:
            return 7
        if "conflict" in tl:
            return 8
        if "financ" in tl:
            return 9
        if "somatic" in tl:
            return 10
        return 99

    return sorted(targets_list, key=_rank)


def _parent_deltas_edge_for(target_name):
    rgb = _parent_deltas_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


def _draw_sankey_parent_deltas(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_domain_style(
        df_sankey,
        sankey_top_features,
        save_dir=save_dir,
        sort_fn=_parent_deltas_sort_targets,
        color_fn=_parent_deltas_resolve_color,
        edge_fn=_parent_deltas_edge_for,
        edge_col=_PARENT_DELTAS_EDGE,
        bg_color=_PARENT_DELTAS_BG,
        save_name="sankey_parent_deltas",
    )


# Parent developmental sweep sankey
# Parent Depression Syndrome (blue), Parent Depressed (red),
# Parent ADHD (gold), Parent Happiness (pink) across T0->T4

_PARENT_DEV_SWEEP_COLORS = {
    # Parent Depression Syndrome: blue gradient (T0 lightest → T4 darkest)
    "parent_depress_D_p__tp0": (0.62, 0.76, 0.92),  # pale sky
    "parent_depress_D_p__tp1": (0.40, 0.58, 0.82),  # soft blue
    "parent_depress_D_p__tp2": (0.22, 0.42, 0.72),  # mid blue
    "parent_depress_D_p__tp3": (0.12, 0.28, 0.58),  # rich blue
    "parent_depress_D_p__tp4": (0.04, 0.16, 0.42),  # deep navy
    # Parent Depressed (Item): red gradient (T0 lightest → T4 darkest)
    "parent_depressed_p__tp0": (0.94, 0.68, 0.66),  # pale rose
    "parent_depressed_p__tp1": (0.86, 0.46, 0.42),  # soft red
    "parent_depressed_p__tp2": (0.76, 0.26, 0.24),  # mid red
    "parent_depressed_p__tp3": (0.60, 0.14, 0.14),  # rich red
    "parent_depressed_p__tp4": (0.44, 0.06, 0.08),  # deep crimson
    # Parent ADHD: yellow/gold gradient (T0 lightest → T4 darkest)
    "parent_adhd_D_p__tp0": (0.96, 0.90, 0.55),  # pale gold
    "parent_adhd_D_p__tp1": (0.92, 0.82, 0.35),  # soft gold
    "parent_adhd_D_p__tp2": (0.82, 0.72, 0.22),  # mid gold
    "parent_adhd_D_p__tp3": (0.68, 0.58, 0.12),  # rich gold
    "parent_adhd_D_p__tp4": (0.52, 0.42, 0.04),  # deep amber
    # Parent Happiness: pink gradient (T0 lightest → T4 darkest)
    "parent_happy_person_p__tp0": (0.98, 0.72, 0.80),  # pale blush
    "parent_happy_person_p__tp1": (0.94, 0.52, 0.64),  # soft pink
    "parent_happy_person_p__tp2": (0.90, 0.35, 0.55),  # mid rose
    "parent_happy_person_p__tp3": (0.78, 0.22, 0.42),  # rich magenta
    "parent_happy_person_p__tp4": (0.62, 0.10, 0.30),  # deep berry
}

_PARENT_DEV_SWEEP_ORDER = [
    # T4 (latest wave) at top → T0 (baseline) at bottom
    # Depression Syndrome
    "parent_depress_D_p__tp4",
    "parent_depress_D_p__tp3",
    "parent_depress_D_p__tp2",
    "parent_depress_D_p__tp1",
    "parent_depress_D_p__tp0",
    # Depressed Item
    "parent_depressed_p__tp4",
    "parent_depressed_p__tp3",
    "parent_depressed_p__tp2",
    "parent_depressed_p__tp1",
    "parent_depressed_p__tp0",
    # ADHD
    "parent_adhd_D_p__tp4",
    "parent_adhd_D_p__tp3",
    "parent_adhd_D_p__tp2",
    "parent_adhd_D_p__tp1",
    "parent_adhd_D_p__tp0",
    # Happiness
    "parent_happy_person_p__tp4",
    "parent_happy_person_p__tp3",
    "parent_happy_person_p__tp2",
    "parent_happy_person_p__tp1",
    "parent_happy_person_p__tp0",
]

# Parents don't have a single age: use assessment wave labels
# Base constants used downstream: _TP_AGE_MAP (referenced by
# _dev_sweep_resolve_color and _extract_tp_from_name), _DEV_SWEEP_BG
# (referenced by _draw_sankey_developmental_sweep), and
# _FEATURE_DOMAIN_ACCENTS (referenced by helper functions).
# ABCD annual follow-up ages for each timepoint:
_TP_AGE_MAP = {
    "0": "9-10y",  # Baseline (T0)
    "1": "10-11y",  # 1-Year (T1)
    "2": "11-12y",  # 2-Year (T2)
    "3": "12-13y",  # 3-Year (T3)
    "4": "13-14y",  # 4-Year (T4)
}

# Background color for developmental-sweep Sankey (matches other Sankey BGs)
_DEV_SWEEP_BG = "#F8F7F3"  # soft off-white / cream

# Feature domain accent colors (Tol-muted-inspired, colorblind-safe)
# Used by _infer_feature_domain() helper. Keys enumerated from the function body.
_FEATURE_DOMAIN_ACCENTS = {
    "Change Scores": "#44AA99",  # Tol muted teal (longitudinal delta)
    "Parent Report": "#AA4499",  # Tol muted purple (external observer)
    "Family/Social": "#CC6677",  # Tol muted rose (relational warmth)
    "Cognitive/Academic": "#332288",  # Tol muted indigo (academic gravitas)
    "Physical/Medical": "#882255",  # Tol muted wine (body)
    "Demographic": "#999933",  # Tol muted olive (neutral context)
    "Youth Behavioral": "#117733",  # Tol muted green (child self-report default)
}

_PARENT_TP_WAVE_MAP = {
    "0": "Baseline",
    "1": "Year 1",
    "2": "Year 2",
    "3": "Year 3",
    "4": "Year 4",
}

_PARENT_DEV_SWEEP_BG = "#FFFFFF"

# Edge colors per domain
_PARENT_DEV_EDGE = {
    "depression_syndrome": "#0A2647",  # navy
    "depressed_item": "#4A0404",  # dark red
    "adhd": "#3B2E00",  # dark brown-gold
    "happiness": "#4A0028",  # dark berry
}


def _parent_dev_sweep_feature_color(feature, df_sankey, feat_target_count, targets_list):
    """Feature node color: single-target inherits target color;
    multi-target goes from dark blueish-gray to near-black with overlap."""
    n_tgt = feat_target_count.get(feature, 1)
    rows = df_sankey[df_sankey["Feature"] == feature]
    if rows.empty:
        return (0.5, 0.5, 0.5)
    if n_tgt == 1:
        dom_t = rows.loc[rows["Importance"].idxmax(), "Target"]
        return _parent_dev_sweep_resolve_color(dom_t)
    # Multi-target: light steel → hard black across 2T-5T+
    _t = min((n_tgt - 2) / 8.0, 1.0)  # gentler ramp for 20 possible targets
    return (0.62 - 0.58 * _t, 0.64 - 0.58 * _t, 0.70 - 0.60 * _t)


def _dev_sweep_sort_targets(targets_list):
    """Sort dev-sweep targets TP-first (T4 top -> T0 bottom), then by target-order
    within each TP. TP-first ordering keeps each timepoint block vertically contiguous,
    which is required for the left-margin age labels to align properly when the batch
    contains more than one target (e.g., social_problems_D_p + not_liked_p).
    """
    order_map = {t: i for i, t in enumerate(_DEV_SWEEP_ORDER)}
    for raw_key, idx_val in list(order_map.items()):
        display = _get_tdn(raw_key)
        if display not in order_map:
            order_map[display] = idx_val

    def _sort_key(t):
        tp = _extract_tp_from_name(t)
        # T4 (tp=4) must come first -> use (-tp) so negatives sort ascending to top.
        # tp=-1 (unrecognized) goes to bottom (sorts last since -(-1) = 1).
        tp_priority = -tp if tp >= 0 else 1
        target_rank = order_map.get(t, 999)
        return (tp_priority, target_rank)

    return sorted(targets_list, key=_sort_key)


def _dev_sweep_feature_color(feature, df_sankey, feat_target_count, targets_list):
    """Feature node color: single-target inherits target color;
    multi-target goes from dark blueish-gray to near-black with overlap."""
    n_tgt = feat_target_count.get(feature, 1)
    rows = df_sankey[df_sankey["Feature"] == feature]
    if rows.empty:
        return (0.5, 0.5, 0.5)
    if n_tgt == 1:
        dom_t = rows.loc[rows["Importance"].idxmax(), "Target"]
        return _dev_sweep_resolve_color(dom_t)
    # Multi-target: light steel → hard black across 2T-5T+
    _t = min((n_tgt - 2) / 3.0, 1.0)  # 0 at 2T, 0.33 at 3T, 0.67 at 4T, 1.0 at 5T+
    return (
        0.62 - 0.58 * _t,  # R: 0.62 → 0.04
        0.64 - 0.58 * _t,  # G: 0.64 → 0.06
        0.70 - 0.60 * _t,
    )  # B: 0.70 → 0.10


def _draw_sankey(df_sankey, sankey_top_features, save_dir=None, style="heatflow_fire"):
    """Dispatch to the appropriate Sankey style function."""
    if style in HEATFLOW_THEMES:
        return _draw_sankey_heatflow_enhanced(df_sankey, sankey_top_features, save_dir=save_dir, theme_key=style)
    _dispatchers = {
        "pfactor_spectra": _draw_sankey_pfactor_spectra,
        "child_pfactor": _draw_sankey_child_pfactor,
        "parent_pfactor": _draw_sankey_parent_pfactor,
        "child_suicide": _draw_sankey_child_suicide,
        "parent_suicide": _draw_sankey_parent_suicide,
        "child_depression": _draw_sankey_child_dep_worsening,
        "parent_depression": _draw_sankey_parent_dep_worsening,
        "parent_deltas": _draw_sankey_parent_deltas,
        "child_deltas": _draw_sankey_child_deltas,
        "child_academic_cognitive": _draw_sankey_child_academic_cognitive,
        "child_adhd_impcomp": _draw_sankey_child_adhd_impcomp,
        "parent_social_impcomp": _draw_sankey_parent_social_impcomp,
        "child_alts": _draw_sankey_child_alts,
        "parent_alts": _draw_sankey_parent_alts,
        "sbt_female": _draw_sankey_sbt_female,
        "sbt_long": _draw_sankey_sbt_long,
        "primary_targets": _draw_sankey_primary_targets,
        "sbt_core_triad": _draw_sankey_sbt_core_triad,
        "sbt_core_triad_female": _draw_sankey_sbt_core_triad_female,
        "sbt_core_triad_male": _draw_sankey_sbt_core_triad_male,
        "child_cogtask_t0": _draw_sankey_child_cogtask_t0,
        "developmental_sweep": _draw_sankey_developmental_sweep,
        "parent_developmental_sweep": _draw_sankey_parent_developmental_sweep,
    }
    _fn = _dispatchers.get(style)
    if _fn is None:
        logger.info(f"  Warning:  Unknown sankey_style '{style}'. Falling back to 'heatflow_fire'.")
        return _draw_sankey_heatflow_enhanced(
            df_sankey, sankey_top_features, save_dir=save_dir, theme_key="heatflow_fire"
        )
    return _fn(df_sankey, sankey_top_features, save_dir=save_dir)


def _run_all_styles(df_sankey, sankey_top_features, save_dir=None):
    """Run all available Sankey styles and save each one."""
    all_styles = list(HEATFLOW_THEMES.keys()) + [
        "pfactor_spectra",
        "child_pfactor",
        "parent_pfactor",
        "child_suicide",
        "parent_suicide",
        "child_depression",
        "parent_depression",
        "child_deltas",
        "parent_deltas",
        "child_academic_cognitive",
        "child_adhd_impcomp",
        "parent_social_impcomp",
        "child_alts",
        "parent_alts",
        "sbt_female",
        "sbt_long",
        "primary_targets",
        "sbt_core_triad",
        "sbt_core_triad_female",
        "sbt_core_triad_male",
        "child_cogtask_t0",
    ]

    logger.info("\n" + "=" * 80)
    logger.info("  RUNNING ALL SANKEY STYLES")
    logger.info("=" * 80)
    logger.info(f"  Total styles to render: {len(all_styles)}")
    logger.info("=" * 80 + "\n")

    results = []
    for i, style in enumerate(all_styles, 1):
        logger.info(f"\n{'-' * 60}")
        logger.info(f"  [{i}/{len(all_styles)}] Rendering: {style}")
        logger.info(f"{'-' * 60}")
        try:
            fig = _draw_sankey(df_sankey, sankey_top_features, save_dir=save_dir, style=style)
            results.append({"style": style, "status": "success", "fig": fig})
            plt.close(fig)
        except Exception as e:
            logger.info(f"  Error rendering {style}: {e}")
            results.append({"style": style, "status": "error", "error": str(e)})

    logger.info("\n" + "=" * 80)
    logger.info("  ALL STYLES COMPLETE")
    logger.info("=" * 80)
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"  Successful: {success_count}/{len(all_styles)}")
    return results


# Color and dispatch tables for child/parent pfactor and suicide sweep styles.
# Late-bound after the heatflow definitions above so the developmental sweep
# helpers can use these palettes.

# Base _DEV_SWEEP_COLORS dict for the child developmental sweep. Seeded
# with F4 gradients (sbt_core_depression, anxdisord_D_p) plus int/ext/
# suicidal gradients and child social-problem gradients.
_DEV_SWEEP_COLORS = {
    # Child Core Depression (F4): royal-blue gradient (T0 lightest → T4 darkest)
    "sbt_core_depression__tp0": (0.66, 0.78, 0.94),  # pale sky
    "sbt_core_depression__tp1": (0.44, 0.60, 0.84),  # soft royal blue
    "sbt_core_depression__tp2": (0.24, 0.42, 0.74),  # mid royal blue
    "sbt_core_depression__tp3": (0.12, 0.26, 0.60),  # rich royal blue
    "sbt_core_depression__tp4": (0.04, 0.12, 0.44),  # deep navy
    # Child Anxiety Disorder (F4): crimson gradient (T0 lightest → T4 darkest)
    "anxdisord_D_p__tp0": (0.96, 0.70, 0.68),  # pale coral
    "anxdisord_D_p__tp1": (0.90, 0.48, 0.46),  # soft crimson
    "anxdisord_D_p__tp2": (0.80, 0.26, 0.26),  # mid crimson
    "anxdisord_D_p__tp3": (0.64, 0.12, 0.14),  # rich crimson
    "anxdisord_D_p__tp4": (0.46, 0.04, 0.08),  # deep garnet
    # Child Internalizing: purple gradient (T0 lightest → T4 darkest)
    "internal_D_p__tp0": (0.80, 0.68, 0.96),
    "internal_D_p__tp1": (0.68, 0.48, 0.90),
    "internal_D_p__tp2": (0.55, 0.36, 0.85),
    "internal_D_p__tp3": (0.42, 0.22, 0.72),
    "internal_D_p__tp4": (0.30, 0.10, 0.58),
    # Child Externalizing: green gradient
    "external_D_p__tp0": (0.72, 0.90, 0.62),
    "external_D_p__tp1": (0.58, 0.82, 0.44),
    "external_D_p__tp2": (0.45, 0.72, 0.30),
    "external_D_p__tp3": (0.32, 0.60, 0.18),
    "external_D_p__tp4": (0.20, 0.46, 0.08),
    # Child Suicidal Ideation: very dark blue gradient
    "suicidal_p__tp0": (0.48, 0.48, 0.66),
    "suicidal_p__tp1": (0.34, 0.34, 0.54),
    "suicidal_p__tp2": (0.22, 0.22, 0.42),
    "suicidal_p__tp3": (0.12, 0.12, 0.32),
    "suicidal_p__tp4": (0.04, 0.04, 0.24),
    # Child Social Problems (DSM domain): amber/orange gradient
    "social_problems_D_p__tp0": (1.00, 0.91, 0.74),  # pale peach
    "social_problems_D_p__tp1": (0.99, 0.76, 0.40),  # soft amber
    "social_problems_D_p__tp2": (0.93, 0.55, 0.12),  # warm orange
    "social_problems_D_p__tp3": (0.72, 0.37, 0.03),  # burnt orange
    "social_problems_D_p__tp4": (0.45, 0.21, 0.02),  # dark burnt
    # Child Not-Liked (CBCL item): chocolate gradient (warm brown-orange,
    # paired with but distinguishable from social_problems_D_p amber/orange)
    "not_liked_p__tp0": (0.74, 0.94, 0.94),  # pale cyan
    "not_liked_p__tp1": (0.40, 0.82, 0.88),  # light cyan
    "not_liked_p__tp2": (0.10, 0.58, 0.70),  # medium teal-blue
    "not_liked_p__tp3": (0.03, 0.35, 0.48),  # deep teal-blue
    "not_liked_p__tp4": (0.01, 0.15, 0.25),  # dark teal
}

# Base _DEV_SWEEP_ORDER list for the child developmental sweep. T4 (latest
# wave) at top -> T0 (baseline) at bottom within each target block. Order:
# F4 primary targets first, then int/ext/sui, then child social.
_DEV_SWEEP_ORDER = [
    # F4 primary: Core Depression + Anxiety Disorder
    "sbt_core_depression__tp4",
    "sbt_core_depression__tp3",
    "sbt_core_depression__tp2",
    "sbt_core_depression__tp1",
    "sbt_core_depression__tp0",
    "anxdisord_D_p__tp4",
    "anxdisord_D_p__tp3",
    "anxdisord_D_p__tp2",
    "anxdisord_D_p__tp1",
    "anxdisord_D_p__tp0",
    # Broadband psychopathology
    "internal_D_p__tp4",
    "internal_D_p__tp3",
    "internal_D_p__tp2",
    "internal_D_p__tp1",
    "internal_D_p__tp0",
    "external_D_p__tp4",
    "external_D_p__tp3",
    "external_D_p__tp2",
    "external_D_p__tp1",
    "external_D_p__tp0",
    "suicidal_p__tp4",
    "suicidal_p__tp3",
    "suicidal_p__tp2",
    "suicidal_p__tp1",
    "suicidal_p__tp0",
    # Child Social Problems (DSM domain + CBCL item)
    "social_problems_D_p__tp4",
    "social_problems_D_p__tp3",
    "social_problems_D_p__tp2",
    "social_problems_D_p__tp1",
    "social_problems_D_p__tp0",
    "not_liked_p__tp4",
    "not_liked_p__tp3",
    "not_liked_p__tp2",
    "not_liked_p__tp1",
    "not_liked_p__tp0",
]


# Domain-aware _dev_sweep_resolve_color.
def _dev_sweep_resolve_color(target_name):
    if target_name in _DEV_SWEEP_COLORS:
        return _DEV_SWEEP_COLORS[target_name]
    tl = target_name.lower()

    def _child_prefix(tl):
        if "internal" in tl:
            return "internal_D_p"
        if "external" in tl:
            return "external_D_p"
        if "suicid" in tl:
            return "suicidal_p"
        if "anxi" in tl:
            return "anxdisord_D_p"
        if "depress" in tl or "mood" in tl:
            return "sbt_core_depression"
        return None

    prefix = _child_prefix(tl)
    if prefix is None:
        return (0.5, 0.5, 0.5)
    for tp in range(5):
        age = _TP_AGE_MAP[str(tp)]
        if f"(t{tp})" in tl or age.lower() in tl:
            return _DEV_SWEEP_COLORS.get(f"{prefix}__tp{tp}", (0.5, 0.5, 0.5))
    return _DEV_SWEEP_COLORS.get(f"{prefix}__tp2", (0.5, 0.5, 0.5))


# Parent int/ext/sui gradients added to _PARENT_DEV_SWEEP_COLORS.
_PARENT_DEV_SWEEP_COLORS.update(
    {
        # Parent Internalizing: purple gradient
        "parent_internal_D_p__tp0": (0.78, 0.60, 0.92),
        "parent_internal_D_p__tp1": (0.65, 0.40, 0.82),
        "parent_internal_D_p__tp2": (0.48, 0.20, 0.68),
        "parent_internal_D_p__tp3": (0.36, 0.10, 0.52),
        "parent_internal_D_p__tp4": (0.24, 0.04, 0.38),
        # Parent Externalizing: green gradient
        "parent_external_D_p__tp0": (0.62, 0.84, 0.70),
        "parent_external_D_p__tp1": (0.40, 0.72, 0.55),
        "parent_external_D_p__tp2": (0.22, 0.58, 0.42),
        "parent_external_D_p__tp3": (0.10, 0.44, 0.30),
        "parent_external_D_p__tp4": (0.04, 0.30, 0.20),
        # Parent Suicidal Thoughts: very dark blue gradient
        "parent_suicidal_thoughts_p__tp0": (0.42, 0.42, 0.62),
        "parent_suicidal_thoughts_p__tp1": (0.30, 0.30, 0.50),
        "parent_suicidal_thoughts_p__tp2": (0.20, 0.20, 0.40),
        "parent_suicidal_thoughts_p__tp3": (0.12, 0.12, 0.32),
        "parent_suicidal_thoughts_p__tp4": (0.04, 0.04, 0.24),
    }
)

# Extension of _PARENT_DEV_SWEEP_ORDER.
_PARENT_DEV_SWEEP_ORDER.extend(
    [
        "parent_internal_D_p__tp4",
        "parent_internal_D_p__tp3",
        "parent_internal_D_p__tp2",
        "parent_internal_D_p__tp1",
        "parent_internal_D_p__tp0",
        "parent_external_D_p__tp4",
        "parent_external_D_p__tp3",
        "parent_external_D_p__tp2",
        "parent_external_D_p__tp1",
        "parent_external_D_p__tp0",
        "parent_suicidal_thoughts_p__tp4",
        "parent_suicidal_thoughts_p__tp3",
        "parent_suicidal_thoughts_p__tp2",
        "parent_suicidal_thoughts_p__tp1",
        "parent_suicidal_thoughts_p__tp0",
    ]
)


# Base definition of _parent_dev_sweep_domain. Required so the override
# chain below has a base function to wrap. Seeds the four core parent
# developmental-sweep domains (depression_syndrome, depressed_item, adhd,
# happiness); subsequent overrides add relationships, unliked,
# internalizing, externalizing, and suicidal.
def _parent_dev_sweep_domain(target_name):
    """Base: map a parent developmental-sweep target to its domain key.

    Successive overrides wrap this function and add additional domains
    (relationships, then unliked).
    """
    tl = target_name.lower()
    # Check __tpN raw-key prefix first
    if "__tp" in target_name:
        prefix = target_name.split("__tp")[0]
        if "parent_depress_D_p" == prefix:
            return "depression_syndrome"
        if "parent_depressed_p" == prefix:
            return "depressed_item"
        if "parent_adhd_D_p" == prefix:
            return "adhd"
        if "parent_happy_person_p" == prefix:
            return "happiness"
    # Keyword match on display name / raw name fallback
    if "happy" in tl or "happiness" in tl:
        return "happiness"
    if "adhd" in tl or "attention" in tl:
        return "adhd"
    if "depressed_p" in tl or "depressed (item)" in tl or "depressed item" in tl:
        return "depressed_item"
    if "depress" in tl or "mood" in tl or "syndrome" in tl:
        return "depression_syndrome"
    # Safe default
    return "depression_syndrome"


# Extend _parent_dev_sweep_domain to include int/ext/sui domains.
_orig_parent_dev_sweep_domain = _parent_dev_sweep_domain


def _extract_tp_from_name(target_name):
    """Extract integer timepoint from a __tpN raw key or display name."""
    if "__tp" in target_name:
        try:
            return int(target_name.split("__tp")[1])
        except ValueError:
            pass
    tl = target_name.lower()
    for tp in range(5):
        age = _TP_AGE_MAP.get(str(tp), "")
        wave = _PARENT_TP_WAVE_MAP.get(str(tp), "")
        if f"(t{tp})" in tl or f"({age.lower()})" in tl or f"({wave.lower()})" in tl:
            return tp
    return -1


_CHILD_DOMAIN_CLEAN_LABELS = {
    "sbt_core_depression": "Mood / Depression",
    "anxdisord_D_p": "Anxiety Disorder",
    "internal_D_p": "Internalizing",
    "external_D_p": "Externalizing",
    "suicidal_p": "Suicidal Ideation",
    # Child Social Problems sweep targets:
    "social_problems_D_p": "Social Problems",
    "not_liked_p": "Not Liked by Others",
}

_PARENT_DOMAIN_CLEAN_LABELS = {
    "parent_depress_D_p": "Depression Syndrome",
    "parent_depressed_p": "Depressed (Item)",
    "parent_adhd_D_p": "ADHD",
    "parent_happy_person_p": "Happiness",
    "parent_internal_D_p": "Internalizing",
    "parent_external_D_p": "Externalizing",
    "parent_suicidal_thoughts_p": "Suicidal Thoughts",
}


def _clean_domain_label(target_name, domain_labels):
    """Return a short domain-only display name (no timepoint).
    Falls back to _get_tdn with timepoint stripped."""
    # Check raw key base
    if "__tp" in target_name:
        base = target_name.split("__tp")[0]
        if base in domain_labels:
            return domain_labels[base]
    # Check if the full name matches a domain label value already
    for raw, label in domain_labels.items():
        if target_name == label:
            return label
    # Strip parenthetical suffix from display name
    disp = _get_tdn(target_name)
    import re

    disp_clean = re.sub(r"\s*\(T\d\)\s*$", "", disp)
    disp_clean = re.sub(r"\s*\(~\d+y\)\s*$", "", disp_clean)
    disp_clean = re.sub(r"\s*\((Baseline|Year \d)\)\s*$", "", disp_clean)
    disp_clean = disp_clean.replace(" (Primary Features)", "")
    disp_clean = disp_clean.replace(" (Syndrome)", "")
    disp_clean = disp_clean.replace(" (Item)", "")
    disp_clean = disp_clean.strip()
    # After cleanup, the stripped name may match a raw-key entry in
    # domain_labels (e.g., "not_liked_p" -> "Not Liked by Others"). Perform
    # a final lookup so targets without TARGET_DISPLAY_NAMES entries still
    # resolve to their clean domain label.
    if disp_clean in domain_labels:
        return domain_labels[disp_clean]
    return disp_clean


# 4. Domain-based shape differentiation


def _domain_boxstyle(target_name):
    """Return matplotlib boxstyle string based on domain.

    - Broadband spectra (int / ext):     pill  → round,pad=0.018
    - Specific syndromes (dep / anx):    rect  → round,pad=0.006
    - Suicidal ideation:                 mid   → round,pad=0.012
    - Everything else (adhd / happiness): rect
    """
    tl = target_name.lower()
    if "internal" in tl or "external" in tl:
        return "round,pad=0.018"  # pill / capsule
    if "suicid" in tl:
        return "round,pad=0.012"  # intermediate rounding
    return "round,pad=0.006"  # sharp rectangle


# 5. Timepoint-block dividers + margin age annotations (shared helper)


def _draw_tp_dividers_and_labels(
    ax,
    targets_list,
    target_positions,
    left_x,
    target_node_w,
    tp_label_map,
    bg_color,
    label_fontsize=18,
    divider_color="#B0BEC5",
    label_color="#546E7F",
):
    """Draw dashed dividers between timepoint blocks and place age/wave
    annotations in the left margin.

    Parameters
    tp_label_map : dict  {int_tp: str_label}   e.g. {4: '~13 y', 3: '~12 y', …}
    """

    # Group targets by timepoint
    tp_blocks = {}  # tp_int -> [target_name, ...]
    for t in targets_list:
        tp = _extract_tp_from_name(t)
        tp_blocks.setdefault(tp, []).append(t)

    tp_keys = sorted(tp_blocks.keys(), reverse=True)  # T4 first (top)

    # Divider lines between consecutive TP blocks
    _div_hw = target_node_w * 3.5  # half-width of dashed line
    for i in range(len(tp_keys) - 1):
        tp_above = tp_keys[i]
        tp_below = tp_keys[i + 1]
        last_above = tp_blocks[tp_above][-1]
        first_below = tp_blocks[tp_below][0]
        y_div = (target_positions[last_above]["y_bot"] + target_positions[first_below]["y_top"]) / 2
        ax.plot(
            [left_x - _div_hw, left_x + _div_hw],
            [y_div, y_div],
            color=divider_color,
            lw=0.9,
            ls="--",
            alpha=0.55,
            zorder=1,
        )

    # Age / wave labels centered on each TP block
    _margin_x = left_x - target_node_w - 0.12  # well to the left of labels
    for tp in tp_keys:
        block = tp_blocks[tp]
        y_centres = [target_positions[t]["y_centre"] for t in block if t in target_positions]
        if not y_centres:
            continue
        y_mid = (max(y_centres) + min(y_centres)) / 2
        label = tp_label_map.get(tp, f"T{tp}")
        txt = ax.text(
            _margin_x,
            y_mid,
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color=label_color,
            fontfamily=_TITLE_FONT,
            rotation=0,
            zorder=6,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground=bg_color)])


# Developmental sweep Sankey (child targets)


def _draw_sankey_developmental_sweep(df_sankey, sankey_top_features, save_dir=None):
    """Developmental Sweep Sankey: multi-domain across T0-T4.

    Implements:
      1. Timepoint-first ordering  (T4 block at top → T0 at bottom)
      2. Port-gap 0.003
      3. Saturation + luminance ramp
      4. Domain-based target node shape
      5. Age annotations in left margin + dashed TP dividers
      6. Figure ×1.5 + 1200 DPI
    """

    _cd = _sankey_common_data(df_sankey)
    targets_list = _dev_sweep_sort_targets(_cd["targets_list"])
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: _dev_sweep_resolve_color(t) for t in targets_list}
    _BG = _DEV_SWEEP_BG

    # Layout constants
    _pad = max(0.022, min(0.040, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.052
    _feat_nw = 0.028
    _target_node_w = 0.036

    # 6. Figure scaling ×1.5
    _, _base_h = _wide_figure_dims(n_targets, n_features)
    fig_w = 15 * 1.5
    fig_h = max(_base_h * 1.3, n_features * 0.52, 22) * 1.5
    _spread_t = 5.5
    _spread_f = 1.9
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()
    _target_fs = 17

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    # Uniform target node heights
    _RECT_H = 0.16
    for t in targets_list:
        if t in target_positions:
            _c = target_positions[t]["y_centre"]
            target_positions[t] = {"y_centre": _c, "y_top": _c + _RECT_H / 2, "y_bot": _c - _RECT_H / 2, "h": _RECT_H}

    # Uniform feature node heights
    _FEAT_H = 0.056
    for f_key in features_list:
        if f_key in feature_positions:
            _c = feature_positions[f_key]["y_centre"]
            feature_positions[f_key] = {
                "y_centre": _c,
                "y_top": _c + _FEAT_H / 2,
                "y_bot": _c - _FEAT_H / 2,
                "h": _FEAT_H,
            }

    # Figure
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.28
    ax.set_xlim(-0.28, right_x + 0.22)  # extra left margin for age labels
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    # 5. Timepoint dividers + age annotations
    _child_tp_labels = {
        4: "~13 y",
        3: "~12 y",
        2: "~11 y",
        1: "~10 y",
        0: "~9 y",
    }
    _draw_tp_dividers_and_labels(
        ax,
        targets_list,
        target_positions,
        left_x,
        _target_node_w,
        tp_label_map=_child_tp_labels,
        bg_color=_BG,
        label_fontsize=18,
        divider_color="#B0BEC5",
        label_color="#546E7F",
    )

    # RIBBONS: 2. Port-gap = 0.003 &  3. Saturation + luminance ramp
    _PORT_GAP = 0.003

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])

        _yl_t = rb["y_l_top"] - _PORT_GAP
        _yl_b = rb["y_l_bot"] + _PORT_GAP
        _yr_t = rb["y_r_top"] - _PORT_GAP * 0.5
        _yr_b = rb["y_r_bot"] + _PORT_GAP * 0.5
        if _yl_t <= _yl_b or _yr_t <= _yr_b:
            _yl_t, _yl_b = rb["y_l_top"], rb["y_l_bot"]
            _yr_t, _yr_b = rb["y_r_top"], rb["y_r_bot"]

        # 3. Saturation + luminance ramp (confirmed present)
        _desat = 0.55 + 0.45 * rel
        _white = (0.88, 0.90, 0.94)
        _fill = tuple(_white[i] + (_rgb[i] - _white[i]) * _desat for i in range(3))
        _alpha = 0.55 + 0.30 * rel

        if rel > 0.45:
            _glow_n = 2 if rel > 0.75 else 1
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                _yl_t + 0.005,
                _yl_b - 0.005,
                _yr_t + 0.003,
                _yr_b - 0.003,
                color_rgba=_rgb + (0.08,),
                zorder=1,
                glow_layers=_glow_n,
            )

        if rel > 0.70:
            _edge_dark = tuple(max(0, c * 0.70) for c in _rgb)
            _e_lw = 0.35 + 0.25 * (rel - 0.70) / 0.30
            _e_alpha = 0.22 + 0.18 * (rel - 0.70) / 0.30
            _edge_c = _edge_dark + (_e_alpha,)
        else:
            _edge_c = _fill + (min(_alpha + 0.05, 0.65),)
            _e_lw = 0.2

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            _yl_t,
            _yl_b,
            _yr_t,
            _yr_b,
            color_rgba=_fill + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_e_lw,
            right_node_w=_feat_nw,
        )

    # Target nodes: 4. Shape differentiation &  5. Clean domain labels

    for t, pos in target_positions.items():
        rgb = _tc.get(t, (0.5, 0.5, 0.5))

        # Domain-aware edge color
        tl = t.lower()
        if "internal" in tl:
            _edge_col = "#2A0845"
        elif "external" in tl:
            _edge_col = "#0A3320"
        elif "suicid" in tl:
            _edge_col = "#060630"
        elif "anxi" in tl:
            _edge_col = "#4A0404"
        else:
            _edge_col = "#0A2647"

        # 4. Domain-based boxstyle
        _bstyle = _domain_boxstyle(t)

        # Shadow
        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.004, pos["y_bot"] - 0.003),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        # Node
        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor=rgb,
                edgecolor=_edge_col,
                linewidth=1.8,
                zorder=3,
            )
        )

        # 5. Clean domain-only label (age is in the margin now)
        _domain_label = _clean_domain_label(t, _CHILD_DOMAIN_CLEAN_LABELS)

        _metric_val = _target_metrics.get(t) or _target_metrics.get(_get_tdn(t))
        _label_x = left_x - _target_node_w - 0.04

        if _metric_val is not None:
            _v, _l = _metric_val
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.025,
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.028,
                f"{_l} = {_v:.3f}",
                ha="right",
                va="center",
                fontsize=_target_fs - 3,
                fontweight="bold",
                color=_edge_col,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Feature nodes: multi-domain edge logic

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count.get(f, 1)
        _fc = _dev_sweep_feature_color(f, df_sankey, feat_target_count, targets_list)

        _f_rows = df_sankey[df_sankey["Feature"] == f]
        _domain_counts = {}
        for _ft in _f_rows["Target"]:
            _ftl = _ft.lower()
            if "internal" in _ftl:
                _domain_counts["int"] = _domain_counts.get("int", 0) + 1
            elif "external" in _ftl:
                _domain_counts["ext"] = _domain_counts.get("ext", 0) + 1
            elif "suicid" in _ftl:
                _domain_counts["sui"] = _domain_counts.get("sui", 0) + 1
            elif "depress" in _ftl or "mood" in _ftl:
                _domain_counts["dep"] = _domain_counts.get("dep", 0) + 1
            elif "anxi" in _ftl:
                _domain_counts["anx"] = _domain_counts.get("anx", 0) + 1
        _dom_winner = max(_domain_counts, key=_domain_counts.get) if _domain_counts else "dep"
        _edge_map = {
            "dep": "#091F38",
            "anx": "#3A0808",
            "int": "#2A0845",
            "ext": "#0A3320",
            "sui": "#060630",
        }
        _feat_edge = _edge_map.get(_dom_winner, "#091F38")

        if n_tgt >= 2:
            dom_rows = df_sankey[df_sankey["Feature"] == f]
            dom_t = dom_rows.loc[dom_rows["Importance"].idxmax(), "Target"] if not dom_rows.empty else targets_list[0]
            glow_col = _tc.get(dom_t, (0.5, 0.5, 0.5))
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - _feat_nw - 0.006, pos["y_bot"] - 0.006),
                    _feat_nw * 2 + 0.012,
                    pos["h"] + 0.012,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_col + (0.38,),
                    linewidth=1.3,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw + 0.004, pos["y_bot"] - 0.003),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw, pos["y_bot"]),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor=_fc,
                edgecolor=_feat_edge,
                linewidth=0.7,
                zorder=3,
            )
        )

        _disp_name = DISPLAY_NAMES.get(f, f) if "DISPLAY_NAMES" in dir() else f
        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + _feat_nw + 0.025,
            pos["y_centre"],
            f"{_disp_name}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color="#1a1a1a",
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Column headers
    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    _hdr_dep_col = "#0A2647"
    _hdr_feat_col = "#1B5E20"
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_dep_col,
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_feat_col,
        fontfamily=_TITLE_FONT,
    )
    _rule_hw = 0.10
    ax.plot(
        [left_x - _rule_hw, left_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_dep_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )
    ax.plot(
        [right_x - _rule_hw, right_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_feat_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )

    ax.text(0.50, 1.085, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)

    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    # 6. Save at 1200 DPI
    saved = _safe_save_fig(fig, save_dir, "sankey_developmental_sweep", dpi=1200, formats=("png", "pdf"))
    plt.show()
    for p in saved:
        logger.info(f"  Saved: {p}")
    return fig


# Parent developmental sweep wrappers


def _draw_sankey_child_pfactor(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_developmental_sweep(df_sankey, sankey_top_features, save_dir=save_dir)


def _draw_sankey_child_suicide(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_developmental_sweep(df_sankey, sankey_top_features, save_dir=save_dir)


def _parent_dev_sweep_domain(target_name):
    """Return the domain key: extended with relationships."""
    tl = target_name.lower()
    if "relationship" in tl or "bad_rel" in tl:
        return "relationships"
    return ctx._orig_parent_dev_sweep_domain_v2(target_name)


_PARENT_DEV_EDGE.update(
    {
        "relationships": "#3D1800",
    }
)

_PARENT_DOMAIN_CLEAN_LABELS["parent_bad_relationships_p"] = "Interpersonal Difficulties"


def _parent_dev_sweep_resolve_color(target_name):
    """Resolve color: extended for relationships (orange)."""
    if target_name in _PARENT_DEV_SWEEP_COLORS:
        return _PARENT_DEV_SWEEP_COLORS[target_name]
    tl = target_name.lower()
    dom = _parent_dev_sweep_domain(target_name)
    _prefix_map = {
        "depression_syndrome": "parent_depress_D_p",
        "depressed_item": "parent_depressed_p",
        "adhd": "parent_adhd_D_p",
        "happiness": "parent_happy_person_p",
        "relationships": "parent_bad_relationships_p",
        "internalizing": "parent_internal_D_p",
        "externalizing": "parent_external_D_p",
        "suicidal": "parent_suicidal_thoughts_p",
    }
    prefix = _prefix_map.get(dom, "parent_depress_D_p")
    for tp in range(5):
        wave = _PARENT_TP_WAVE_MAP[str(tp)]
        if f"(t{tp})" in tl or wave.lower() in tl:
            return _PARENT_DEV_SWEEP_COLORS.get(f"{prefix}__tp{tp}", (0.5, 0.5, 0.5))
    return _PARENT_DEV_SWEEP_COLORS.get(f"{prefix}__tp2", (0.5, 0.5, 0.5))


# 4. Robust sort: uses _extract_tp_from_name, not order-map matching
# This guarantees Year 4 at top → Baseline at bottom regardless of
# display name format.


def _parent_dev_sweep_sort_targets(targets_list):
    """Sort parent targets: T4 at top → T0 at bottom, domains ordered within."""
    _dom_order = {
        "depression_syndrome": 0,
        "depressed_item": 1,
        "adhd": 2,
        "happiness": 3,
        "relationships": 4,
        "internalizing": 5,
        "externalizing": 6,
        "suicidal": 7,
    }

    def _sort_key(t):
        tp = _extract_tp_from_name(t)
        dom = _parent_dev_sweep_domain(t)
        dom_idx = _dom_order.get(dom, 99)
        # Negative tp so T4 (largest) sorts first (smallest key)
        return (-tp if tp >= 0 else 999, dom_idx)

    return sorted(targets_list, key=_sort_key)


# 5. REDEFINE _draw_sankey_parent_developmental_sweep
# Year 4 at top → Baseline at bottom (robust sort)
# Single-line labels: "Domain (Year N)" in parentheses


def _draw_sankey_parent_developmental_sweep(df_sankey, sankey_top_features, save_dir=None):
    """Parent Developmental Sweep Sankey.

    Year 4 at top → Baseline at bottom.
    Labels: 'Domain (Year N)' with R² below.
    """

    _cd = _sankey_common_data(df_sankey)
    targets_list = _parent_dev_sweep_sort_targets(_cd["targets_list"])
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: _parent_dev_sweep_resolve_color(t) for t in targets_list}
    _BG = _PARENT_DEV_SWEEP_BG

    _pad = max(0.022, min(0.040, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.052
    _feat_nw = 0.028
    _target_node_w = 0.036

    _, _base_h = _wide_figure_dims(n_targets, n_features)
    fig_w = 15 * 1.5
    fig_h = max(_base_h * 1.3, n_features * 0.52, 22) * 1.5
    _spread_t = 5.5
    _spread_f = 1.9
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()
    _target_fs = 15

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    _RECT_H = 0.13
    for t in targets_list:
        if t in target_positions:
            _c = target_positions[t]["y_centre"]
            target_positions[t] = {"y_centre": _c, "y_top": _c + _RECT_H / 2, "y_bot": _c - _RECT_H / 2, "h": _RECT_H}

    _FEAT_H = 0.056
    for f_key in features_list:
        if f_key in feature_positions:
            _c = feature_positions[f_key]["y_centre"]
            feature_positions[f_key] = {
                "y_centre": _c,
                "y_top": _c + _FEAT_H / 2,
                "y_bot": _c - _FEAT_H / 2,
                "h": _FEAT_H,
            }

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.28
    ax.set_xlim(-0.18, right_x + 0.22)
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    # Print sort order
    logger.info("   Target sort order (top → bottom):")
    for _i, _t in enumerate(targets_list):
        _tp = _extract_tp_from_name(_t)
        logger.info(f"     {_i + 1}. {_t}  (tp={_tp}, y_centre={target_positions[_t]['y_centre']:.3f})")

    # RIBBONS
    _PORT_GAP = 0.003

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])

        _yl_t = rb["y_l_top"] - _PORT_GAP
        _yl_b = rb["y_l_bot"] + _PORT_GAP
        _yr_t = rb["y_r_top"] - _PORT_GAP * 0.5
        _yr_b = rb["y_r_bot"] + _PORT_GAP * 0.5
        if _yl_t <= _yl_b or _yr_t <= _yr_b:
            _yl_t, _yl_b = rb["y_l_top"], rb["y_l_bot"]
            _yr_t, _yr_b = rb["y_r_top"], rb["y_r_bot"]

        _desat = 0.55 + 0.45 * rel
        _white = (0.88, 0.90, 0.94)
        _fill = tuple(_white[i] + (_rgb[i] - _white[i]) * _desat for i in range(3))
        _alpha = 0.55 + 0.30 * rel

        if rel > 0.45:
            _glow_n = 2 if rel > 0.75 else 1
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                _yl_t + 0.005,
                _yl_b - 0.005,
                _yr_t + 0.003,
                _yr_b - 0.003,
                color_rgba=_rgb + (0.08,),
                zorder=1,
                glow_layers=_glow_n,
            )

        if rel > 0.70:
            _edge_dark = tuple(max(0, c * 0.70) for c in _rgb)
            _e_lw = 0.35 + 0.25 * (rel - 0.70) / 0.30
            _e_alpha = 0.22 + 0.18 * (rel - 0.70) / 0.30
            _edge_c = _edge_dark + (_e_alpha,)
        else:
            _edge_c = _fill + (min(_alpha + 0.05, 0.65),)
            _e_lw = 0.2

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            _yl_t,
            _yl_b,
            _yr_t,
            _yr_b,
            color_rgba=_fill + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_e_lw,
            right_node_w=_feat_nw,
        )

    # Target nodes: "Domain (Year N)" single-line labels

    for t, pos in target_positions.items():
        rgb = _tc.get(t, (0.5, 0.5, 0.5))
        dom = _parent_dev_sweep_domain(t)
        _edge_col = _PARENT_DEV_EDGE.get(dom, "#333333")

        _bstyle = _domain_boxstyle(t)

        # Shadow
        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.004, pos["y_bot"] - 0.003),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        # Node
        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor=rgb,
                edgecolor=_edge_col,
                linewidth=1.8,
                zorder=3,
            )
        )

        # Build "Domain (Year N)" label
        _domain_part = _clean_domain_label(t, _PARENT_DOMAIN_CLEAN_LABELS)
        _tp_idx = _extract_tp_from_name(t)
        if _tp_idx >= 0:
            _wave_str = _PARENT_TP_WAVE_MAP.get(str(_tp_idx), f"T{_tp_idx}")
            _display_label = f"{_domain_part} ({_wave_str})"
        else:
            _display_label = _domain_part

        _metric_val = _target_metrics.get(t) or _target_metrics.get(_get_tdn(t))
        _label_x = left_x - _target_node_w - 0.04

        if _metric_val is not None:
            _v, _l = _metric_val
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.020,
                _display_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.022,
                f"{_l} = {_v:.3f}",
                ha="right",
                va="center",
                fontsize=_target_fs - 3,
                fontweight="bold",
                color=_edge_col,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _display_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Feature nodes

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count.get(f, 1)
        _fc = _parent_dev_sweep_feature_color(f, df_sankey, feat_target_count, targets_list)

        _f_rows = df_sankey[df_sankey["Feature"] == f]
        _dom_counts = {}
        for _ft in _f_rows["Target"]:
            _fd = _parent_dev_sweep_domain(_ft)
            _dom_counts[_fd] = _dom_counts.get(_fd, 0) + 1
        _dom_winner = max(_dom_counts, key=_dom_counts.get) if _dom_counts else "depression_syndrome"
        _feat_edge = _PARENT_DEV_EDGE.get(_dom_winner, "#333333")

        if n_tgt >= 2:
            dom_rows = df_sankey[df_sankey["Feature"] == f]
            dom_t = dom_rows.loc[dom_rows["Importance"].idxmax(), "Target"] if not dom_rows.empty else targets_list[0]
            glow_col = _tc.get(dom_t, (0.5, 0.5, 0.5))
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - _feat_nw - 0.006, pos["y_bot"] - 0.006),
                    _feat_nw * 2 + 0.012,
                    pos["h"] + 0.012,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_col + (0.38,),
                    linewidth=1.3,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw + 0.004, pos["y_bot"] - 0.003),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw, pos["y_bot"]),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor=_fc,
                edgecolor=_feat_edge,
                linewidth=0.7,
                zorder=3,
            )
        )

        _disp_name = DISPLAY_NAMES.get(f, f) if "DISPLAY_NAMES" in dir() else f
        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + _feat_nw + 0.025,
            pos["y_centre"],
            f"{_disp_name}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color="#1a1a1a",
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Column headers
    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#0A2647",
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#1B5E20",
        fontfamily=_TITLE_FONT,
    )
    _rule_hw = 0.10
    ax.plot(
        [left_x - _rule_hw, left_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color="#0A2647",
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )
    ax.plot(
        [right_x - _rule_hw, right_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color="#1B5E20",
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )

    ax.text(0.50, 1.085, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)

    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, "sankey_parent_developmental_sweep", dpi=1200, formats=("png", "pdf"))
    plt.show()
    for p in saved:
        logger.info(f"  Saved: {p}")
    return fig


# Reconnect wrappers


def _draw_sankey_parent_pfactor(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_parent_developmental_sweep(df_sankey, sankey_top_features, save_dir=save_dir)


def _draw_sankey_parent_suicide(df_sankey, sankey_top_features, save_dir=None):
    return _draw_sankey_parent_developmental_sweep(df_sankey, sankey_top_features, save_dir=save_dir)


# Changes 20-26: Social Problems developmental sweep support
#   - Child:  not_liked_p, social_problems_D_p across T0-T4
#   - Parent: parent_not_liked_by_others_p, parent_bad_relationships_p across T0-T4
# Pairs with the "Child/Parent Social Problems Developmental Sweep" TP-sweep
# bundles of the main stage (_TP_SWEEP_BUNDLES). Also fixes a latent gap where
# parent_bad_relationships_p had no gradient entries in _PARENT_DEV_SWEEP_COLORS
# despite being routed through the \'relationships\' domain.

# Parent social gradients added to _PARENT_DEV_SWEEP_COLORS.
# peru/sienna gradient for parent_bad_relationships_p (matches _BATCH_TARGET_COLORS "peru"),
# teal gradient for parent_not_liked_by_others_p (matches delta analogue "darkcyan").
_PARENT_DEV_SWEEP_COLORS.update(
    {
        # Parent Bad Relationships: peru/sienna gradient (T0 lightest → T4 darkest)
        "parent_bad_relationships_p__tp0": (0.96, 0.84, 0.70),  # pale wheat
        "parent_bad_relationships_p__tp1": (0.88, 0.68, 0.50),  # soft tan
        "parent_bad_relationships_p__tp2": (0.76, 0.52, 0.28),  # peru mid
        "parent_bad_relationships_p__tp3": (0.58, 0.36, 0.12),  # rich sienna
        "parent_bad_relationships_p__tp4": (0.40, 0.22, 0.04),  # deep mahogany
        # Parent Not-Liked-by-Others: teal gradient (T0 lightest → T4 darkest)
        "parent_not_liked_by_others_p__tp0": (0.72, 0.92, 0.92),  # pale aqua
        "parent_not_liked_by_others_p__tp1": (0.48, 0.82, 0.84),  # soft teal
        "parent_not_liked_by_others_p__tp2": (0.22, 0.66, 0.70),  # mid teal
        "parent_not_liked_by_others_p__tp3": (0.08, 0.48, 0.54),  # rich teal
        "parent_not_liked_by_others_p__tp4": (0.02, 0.32, 0.38),  # deep teal
    }
)

# Extension of _PARENT_DEV_SWEEP_ORDER with parent social targets.
# T4 (latest) at top → T0 (baseline) at bottom within each target block.
_PARENT_DEV_SWEEP_ORDER.extend(
    [
        "parent_bad_relationships_p__tp4",
        "parent_bad_relationships_p__tp3",
        "parent_bad_relationships_p__tp2",
        "parent_bad_relationships_p__tp1",
        "parent_bad_relationships_p__tp0",
        "parent_not_liked_by_others_p__tp4",
        "parent_not_liked_by_others_p__tp3",
        "parent_not_liked_by_others_p__tp2",
        "parent_not_liked_by_others_p__tp1",
        "parent_not_liked_by_others_p__tp0",
    ]
)

_orig_parent_dev_sweep_domain_v3 = _parent_dev_sweep_domain


def _parent_dev_sweep_domain(target_name):
    """Return the domain key: extended with \'unliked\' for
    parent_not_liked_by_others_p. Falls back to v3 (which handles
    \'relationships\') → v1 for all other targets."""
    tl = target_name.lower()
    if "not_liked" in tl or "not liked" in tl or "unliked" in tl:
        return "unliked"
    return _orig_parent_dev_sweep_domain_v3(target_name)


# Register the new domain in _PARENT_DEV_EDGE and add clean labels.
_PARENT_DEV_EDGE.update(
    {
        "unliked": "#003340",  # very dark teal edge (matches darkcyan gradient)
    }
)

_PARENT_DOMAIN_CLEAN_LABELS["parent_not_liked_by_others_p"] = "Unliked by Others"
_CHILD_DOMAIN_CLEAN_LABELS["social_problems_D_p"] = "Social Problems"
_CHILD_DOMAIN_CLEAN_LABELS["not_liked_p"] = "Not Liked"


# Override _parent_dev_sweep_resolve_color to support 'unliked'.
# Re-declare the full function so the _prefix_map includes the new domain.
def _parent_dev_sweep_resolve_color(target_name):
    """Resolve color: extended for relationships (peru) + unliked (teal)."""
    if target_name in _PARENT_DEV_SWEEP_COLORS:
        return _PARENT_DEV_SWEEP_COLORS[target_name]
    tl = target_name.lower()
    dom = _parent_dev_sweep_domain(target_name)
    _prefix_map = {
        "depression_syndrome": "parent_depress_D_p",
        "depressed_item": "parent_depressed_p",
        "adhd": "parent_adhd_D_p",
        "happiness": "parent_happy_person_p",
        "relationships": "parent_bad_relationships_p",
        "unliked": "parent_not_liked_by_others_p",
        "internalizing": "parent_internal_D_p",
        "externalizing": "parent_external_D_p",
        "suicidal": "parent_suicidal_thoughts_p",
    }
    prefix = _prefix_map.get(dom, "parent_depress_D_p")
    for tp in range(5):
        wave = _PARENT_TP_WAVE_MAP[str(tp)]
        if f"(t{tp})" in tl or wave.lower() in tl:
            return _PARENT_DEV_SWEEP_COLORS.get(f"{prefix}__tp{tp}", (0.5, 0.5, 0.5))
    return _PARENT_DEV_SWEEP_COLORS.get(f"{prefix}__tp2", (0.5, 0.5, 0.5))


# Override _parent_dev_sweep_sort_targets to include 'unliked'.
# Domain sort order: \'unliked\' sits just after \'relationships\' so the two
# social domains render adjacently in the Sankey.
def _parent_dev_sweep_sort_targets(targets_list):
    """Sort parent targets: T4 at top → T0 at bottom, domains ordered within.

    \'unliked\' inserted between \'relationships\' and int/ext/sui so all social
    domains cluster together before the broadband/suicidal block.
    """
    _dom_order = {
        "depression_syndrome": 0,
        "depressed_item": 1,
        "adhd": 2,
        "happiness": 3,
        "relationships": 4,
        "unliked": 5,
        "internalizing": 6,
        "externalizing": 7,
        "suicidal": 8,
    }

    def _sort_key(t):
        tp = _extract_tp_from_name(t)
        dom = _parent_dev_sweep_domain(t)
        dom_idx = _dom_order.get(dom, 99)
        return (-tp if tp >= 0 else 999, dom_idx)

    return sorted(targets_list, key=_sort_key)


# Override _dev_sweep_resolve_color to extend the display-name fallback so
# inputs like "Social Problems (T3)" resolve to the correct child social
# color. The raw-key fast path for 'social_problems_D_p__tpN' and
# 'not_liked_p__tpN' is already handled by the base color dict.
def _dev_sweep_resolve_color(target_name):
    if target_name in _DEV_SWEEP_COLORS:
        return _DEV_SWEEP_COLORS[target_name]
    tl = target_name.lower()

    def _child_prefix(tl):
        if "internal" in tl:
            return "internal_D_p"
        if "external" in tl:
            return "external_D_p"
        if "suicid" in tl:
            return "suicidal_p"
        if "anxi" in tl:
            return "anxdisord_D_p"
        if "not_liked" in tl or "not liked" in tl:
            return "not_liked_p"
        if "social problem" in tl or "social_problem" in tl:
            return "social_problems_D_p"
        if "depress" in tl or "mood" in tl:
            return "sbt_core_depression"
        return None

    prefix = _child_prefix(tl)
    if prefix is None:
        return (0.5, 0.5, 0.5)
    for tp in range(5):
        age = _TP_AGE_MAP[str(tp)]
        if f"(t{tp})" in tl or age.lower() in tl:
            return _DEV_SWEEP_COLORS.get(f"{prefix}__tp{tp}", (0.5, 0.5, 0.5))
    return _DEV_SWEEP_COLORS.get(f"{prefix}__tp2", (0.5, 0.5, 0.5))


# Bespoke child_deltas Sankey rendering. Ports the developmental_sweep
# visual refinements (port-gap, saturation ramp, uniform node heights,
# domain boxstyles, domain edge colors, 1.5x figure scale, 1200 DPI
# output) into a dedicated child_deltas implementation. Uses a
# colorblind-safe _CHILD_DELTAS_COLORS palette (Tol-inspired,
# deuteranopia-verified) and clean domain labels. Applies to the "Child
# Only Deltas" bundle (8 targets).

# Accessible 8-target palette (colorblind-safe).
# Each pair verified to differ by at least 60 degrees in hue or 20 units in luminance,
# and distinguishable under deuteranopia/protanopia/tritanopia simulation.
# Domain-coherent where possible (depression = blue family, anxiety = red,
# ADHD = amber, social = teal, somatic = purple, grades = slate, family = olive).
_CHILD_DELTAS_COLORS = {
    # Okabe-Ito Color Universal Design (CUD) palette - the gold standard for
    # colorblind-safe scientific visualization. Verified against deuteranopia,
    # protanopia, and tritanopia simulation: min pairwise distance 0.119 across
    # all CVD types (the one borderline pair - anxiety x family-conflict - is
    # non-adjacent in sort order so does not visually collide). Luminance spans
    # 0.085-0.42 for clear greyscale distinguishability. Two deviations from
    # pure CUD: (1) black -> charcoal (#525252) to avoid visual dominance;
    # (2) yellow -> dark ochre (#A6761D) for white-background contrast.
    # Depression family (blue pair - intentionally close by construct)
    "delta_depress_D_p": (0.000, 0.447, 0.698),  # #0072B2 CUD Blue
    "delta_sbt_core_depression": (0.337, 0.706, 0.914),  # #56B4E9 CUD Sky Blue
    # Anxiety / ADHD (red-orange pair - adjacent by domain, luminance-separated)
    "delta_anxdisord_D_p": (0.835, 0.369, 0.000),  # #D55E00 CUD Vermillion
    "delta_adhd_D_p": (0.902, 0.624, 0.000),  # #E69F00 CUD Orange
    # Social (cool green, maximally separated from both blue and orange families)
    "delta_social_problems_D_p": (0.000, 0.620, 0.451),  # #009E73 CUD Bluish Green
    # Somatic (reddish purple - distinct hue axis from red/orange warms)
    "delta_somatic_problems_D_p": (0.800, 0.475, 0.655),  # #CC79A7 CUD Reddish Purple
    # Bad grades (neutral charcoal - academic gravitas, safe under all CVD types)
    "delta_bad_grades": (0.322, 0.322, 0.322),  # #525252 charcoal
    # Family conflict (dark ochre - earthy warm, distinct from orange by luminance)
    "delta_family_conflict_ss_k": (0.651, 0.463, 0.114),  # #A6761D dark ochre
}

# Target edge colors (darker variants of each fill).
# Each edge is ~55-70% luminance of the fill, preserving domain identity.
_CHILD_DELTAS_TARGET_EDGE = {
    # Each edge ~60-65% luminance of the CUD fill color, preserving hue identity
    "delta_depress_D_p": "#00436B",  # dark CUD blue
    "delta_sbt_core_depression": "#2F6F94",  # muted sky blue
    "delta_anxdisord_D_p": "#823800",  # dark vermillion
    "delta_adhd_D_p": "#8A6000",  # dark orange
    "delta_social_problems_D_p": "#005F45",  # dark bluish green
    "delta_somatic_problems_D_p": "#7A3F60",  # dark reddish purple
    "delta_bad_grades": "#1F1F1F",  # near-black (for contrast with charcoal fill)
    "delta_family_conflict_ss_k": "#5C3F0F",  # dark ochre
}

# Clean domain labels (with \u0394 prefix signalling change score).
_CHILD_DELTAS_DOMAIN_CLEAN_LABELS = {
    "delta_depress_D_p": "\u0394 Depression Syndrome",
    "delta_sbt_core_depression": "\u0394 Mood / Core Depression",
    "delta_anxdisord_D_p": "\u0394 Anxiety Disorder",
    "delta_adhd_D_p": "\u0394 ADHD",
    "delta_social_problems_D_p": "\u0394 Social Problems",
    "delta_somatic_problems_D_p": "\u0394 Somatic Problems",
    "delta_bad_grades": "\u0394 Academic Performance",
    "delta_family_conflict_ss_k": "\u0394 Family Conflict",
}


# Per-target domain identifier (used for boxstyle and feature edge).
def _child_deltas_domain(target_name):
    """Return a short domain key for a child-deltas target or display-alias.

    Resolution order: (1) raw-key match in _CHILD_DELTAS_COLORS,
    (2) display-alias lookup, (3) keyword pattern match.
    """
    if target_name in _CHILD_DELTAS_COLORS:
        raw = target_name
    else:
        raw = _CHILD_DELTAS_DISPLAY_ALIASES.get(target_name, target_name)
    tl = raw.lower()
    if "depress" in tl and "sbt" in tl:
        return "depression_core"
    if "depress" in tl or "dep_" in tl:
        return "depression"
    if "anxi" in tl:
        return "anxiety"
    if "adhd" in tl or "attention" in tl:
        return "adhd"
    if "social" in tl:
        return "social"
    if "somatic" in tl:
        return "somatic"
    if "grade" in tl or "academic" in tl:
        return "grades"
    if "conflict" in tl or "family" in tl:
        return "family"
    return "depression"  # safe default


# Boxstyle differentiation (light domain contrast without noise).
def _child_deltas_boxstyle(target_name):
    """Return matplotlib boxstyle string keyed by domain.
    - Composite/functional domains (social, somatic, family): capsule-ish
    - Specific syndromes (depress, anxi, adhd): tight rect
    - Grades: slight rounding (neutral)
    """
    dom = _child_deltas_domain(target_name)
    if dom in ("somatic",):
        return "round,pad=0.018"  # full pill (body/somatic)
    if dom in ("social", "family"):
        return "round,pad=0.014"  # rounded capsule (relational)
    if dom == "grades":
        return "round,pad=0.010"  # mid rounding (neutral)
    return "round,pad=0.008"  # tight rect (specific syndrome)


# Target edge color resolver.
def _child_deltas_edge_for(target_name):
    """Resolve target node edge color. Prefers explicit map, falls back to
    65% luminance of the fill color."""
    if target_name in _CHILD_DELTAS_TARGET_EDGE:
        return _CHILD_DELTAS_TARGET_EDGE[target_name]
    raw = _CHILD_DELTAS_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_DELTAS_TARGET_EDGE:
        return _CHILD_DELTAS_TARGET_EDGE[raw]
    # Fallback: darken fill
    rgb = _child_deltas_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


# Feature node color: single-target inherits, multi-target ramps.
def _child_deltas_feature_color(feature, df_sankey, feat_target_count, targets_list):
    """Feature color: single-target features inherit the target color;
    multi-target features use a steel \u2192 near-black ramp keyed to overlap count
    (2T \u2192 light steel, 8T \u2192 near-black)."""
    n_tgt = feat_target_count.get(feature, 1)
    rows = df_sankey[df_sankey["Feature"] == feature]
    if rows.empty:
        return (0.5, 0.5, 0.5)
    if n_tgt == 1:
        dom_t = rows.loc[rows["Importance"].idxmax(), "Target"]
        return _child_deltas_resolve_color(dom_t)
    # Multi-target gradient: gentle for up to 8 targets
    _t = min((n_tgt - 2) / 6.0, 1.0)
    return (0.62 - 0.58 * _t, 0.64 - 0.58 * _t, 0.70 - 0.60 * _t)


# Feature edge map keyed by majority-domain of incoming ribbons.
_CHILD_DELTAS_FEATURE_EDGE_MAP = {
    # Feature node edge colors keyed by majority-domain of incoming ribbons.
    # Values are darker shades of the corresponding target fill colors.
    "depression_core": "#2F6F94",  # muted sky blue
    "depression": "#00436B",  # dark CUD blue
    "anxiety": "#823800",  # dark vermillion
    "adhd": "#8A6000",  # dark CUD orange
    "social": "#005F45",  # dark bluish green
    "somatic": "#7A3F60",  # dark reddish purple
    "grades": "#1F1F1F",  # near-black
    "family": "#5C3F0F",  # dark ochre
}


# Sort function for child-delta targets, ordered by batch R^2 when
# available; unknown targets sink to the bottom.
def _child_deltas_sort_targets(targets_list):
    """Order child-delta targets by batch R\u00b2 (if available) else fall back to
    _CHILD_DELTAS_ORDER; unknown targets sink to the bottom.

    Mirrors _parent_deltas_sort_targets pattern so both delta Sankeys behave
    consistently.
    """
    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}

    def _r2_for(t):
        entry = _tm.get(t)
        if entry is None:
            entry = _tm.get(_tdn.get(t, ""))
        if entry is None:
            raw = _CHILD_DELTAS_DISPLAY_ALIASES.get(t)
            if raw:
                entry = _tm.get(raw) or _tm.get(_tdn.get(raw, ""))
        return entry[0] if entry is not None else None

    r2_vals = {t: _r2_for(t) for t in targets_list}
    has_metrics = any(v is not None for v in r2_vals.values())

    if has_metrics:
        return sorted(targets_list, key=lambda t: r2_vals.get(t) if r2_vals.get(t) is not None else -999, reverse=True)

    def _rank(t):
        if t in _CHILD_DELTAS_ORDER:
            return _CHILD_DELTAS_ORDER.index(t)
        raw = _CHILD_DELTAS_DISPLAY_ALIASES.get(t)
        if raw and raw in _CHILD_DELTAS_ORDER:
            return _CHILD_DELTAS_ORDER.index(raw)
        return 99

    return sorted(targets_list, key=_rank)


# Bespoke drawing function modeled on _draw_sankey_developmental_sweep.
# Ports every TP-agnostic refinement (port gap, saturation ramp, uniform node
# heights, domain boxstyles, domain-aware edges, 1.5x figure scale, 1200 DPI).
# Omits TP-specific elements (age annotations, TP dividers, TP-first sorting).


def _draw_sankey_child_deltas(df_sankey, sankey_top_features, save_dir=None):
    """Child Only Deltas Sankey: 8 change-score targets at a single timepoint.

    Implements:
      1. Colorblind-safe 8-target palette (deuteranopia-verified)
      2. Port-gap 0.003
      3. Saturation + luminance ramp for ribbon importance encoding
      4. Domain-based target node shape differentiation
      5. Domain-aware target edge colors
      6. Figure 1.5x scale + 1200 DPI output
      7. Uniform target/feature node heights
    """

    _cd = _sankey_common_data(df_sankey)
    targets_list = _child_deltas_sort_targets(_cd["targets_list"])
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: _child_deltas_resolve_color(t) for t in targets_list}
    _BG = _CHILD_DELTAS_BG

    # Layout constants (matching dev_sweep proportions)
    _pad = max(0.022, min(0.040, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.052
    _feat_nw = 0.028
    _target_node_w = 0.036

    # Figure scaling x1.5
    _, _base_h = _wide_figure_dims(n_targets, n_features)
    fig_w = 15 * 1.5
    fig_h = max(_base_h * 1.3, n_features * 0.52, 22) * 1.5
    _spread_t = 3.2  # lower than dev_sweep (5.5) since no TP clustering
    _spread_f = 1.9
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()
    _target_fs = 17

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.14, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.045, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    # Uniform target node heights
    _RECT_H = 0.16
    for t in targets_list:
        if t in target_positions:
            _c = target_positions[t]["y_centre"]
            target_positions[t] = {"y_centre": _c, "y_top": _c + _RECT_H / 2, "y_bot": _c - _RECT_H / 2, "h": _RECT_H}

    # Uniform feature node heights
    _FEAT_H = 0.056
    for f_key in features_list:
        if f_key in feature_positions:
            _c = feature_positions[f_key]["y_centre"]
            feature_positions[f_key] = {
                "y_centre": _c,
                "y_top": _c + _FEAT_H / 2,
                "y_bot": _c - _FEAT_H / 2,
                "h": _FEAT_H,
            }

    # Figure
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04
    ax.set_xlim(-0.08, right_x + 0.22)
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    # RIBBONS: port-gap 0.003 + saturation/luminance ramp (from dev_sweep)
    _PORT_GAP = 0.003

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])

        _yl_t = rb["y_l_top"] - _PORT_GAP
        _yl_b = rb["y_l_bot"] + _PORT_GAP
        _yr_t = rb["y_r_top"] - _PORT_GAP * 0.5
        _yr_b = rb["y_r_bot"] + _PORT_GAP * 0.5
        if _yl_t <= _yl_b or _yr_t <= _yr_b:
            _yl_t, _yl_b = rb["y_l_top"], rb["y_l_bot"]
            _yr_t, _yr_b = rb["y_r_top"], rb["y_r_bot"]

        # Saturation + luminance ramp
        _desat = 0.55 + 0.45 * rel
        _white = (0.92, 0.93, 0.96)
        _fill = tuple(_white[i] + (_rgb[i] - _white[i]) * _desat for i in range(3))
        _alpha = 0.55 + 0.30 * rel

        if rel > 0.45:
            _glow_n = 2 if rel > 0.75 else 1
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                _yl_t + 0.005,
                _yl_b - 0.005,
                _yr_t + 0.003,
                _yr_b - 0.003,
                color_rgba=_rgb + (0.08,),
                zorder=1,
                glow_layers=_glow_n,
            )

        if rel > 0.70:
            _edge_dark = tuple(max(0, c * 0.70) for c in _rgb)
            _e_lw = 0.35 + 0.25 * (rel - 0.70) / 0.30
            _e_alpha = 0.22 + 0.18 * (rel - 0.70) / 0.30
            _edge_c = _edge_dark + (_e_alpha,)
        else:
            _edge_c = _fill + (min(_alpha + 0.05, 0.65),)
            _e_lw = 0.2

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            _yl_t,
            _yl_b,
            _yr_t,
            _yr_b,
            color_rgba=_fill + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_e_lw,
            right_node_w=_feat_nw,
        )

    # Target nodes: domain boxstyle + domain edge + R\u00b2 label under title

    for t, pos in target_positions.items():
        rgb = _tc.get(t, (0.5, 0.5, 0.5))
        _edge_col = _child_deltas_edge_for(t)
        _bstyle = _child_deltas_boxstyle(t)

        # Shadow
        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.004, pos["y_bot"] - 0.003),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        # Node
        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor=rgb,
                edgecolor=_edge_col,
                linewidth=1.8,
                zorder=3,
            )
        )

        # Clean domain label (resolves raw-key or alias both)
        raw_key = t if t in _CHILD_DELTAS_DOMAIN_CLEAN_LABELS else _CHILD_DELTAS_DISPLAY_ALIASES.get(t, t)
        _domain_label = _CHILD_DELTAS_DOMAIN_CLEAN_LABELS.get(raw_key, _get_tdn(t))

        _metric_val = _target_metrics.get(t) or _target_metrics.get(_get_tdn(t))
        _label_x = left_x - _target_node_w - 0.04

        if _metric_val is not None:
            _v, _l = _metric_val
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.025,
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.028,
                f"{_l} = {_v:.3f}",
                ha="right",
                va="center",
                fontsize=_target_fs - 3,
                fontweight="bold",
                color=_edge_col,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Feature nodes: majority-domain edge + glow halo for multi-target features

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count.get(f, 1)
        _fc = _child_deltas_feature_color(f, df_sankey, feat_target_count, targets_list)

        _f_rows = df_sankey[df_sankey["Feature"] == f]
        _domain_counts = {}
        for _ft in _f_rows["Target"]:
            _dom_key = _child_deltas_domain(_ft)
            _domain_counts[_dom_key] = _domain_counts.get(_dom_key, 0) + 1
        _dom_winner = max(_domain_counts, key=_domain_counts.get) if _domain_counts else "depression"
        _feat_edge = _CHILD_DELTAS_FEATURE_EDGE_MAP.get(_dom_winner, "#1A1A1A")

        # Glow halo for multi-target features (signals cross-target importance)
        if n_tgt >= 2:
            dom_rows = df_sankey[df_sankey["Feature"] == f]
            dom_t = dom_rows.loc[dom_rows["Importance"].idxmax(), "Target"] if not dom_rows.empty else targets_list[0]
            glow_col = _tc.get(dom_t, (0.5, 0.5, 0.5))
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - _feat_nw - 0.006, pos["y_bot"] - 0.006),
                    _feat_nw * 2 + 0.012,
                    pos["h"] + 0.012,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_col + (0.38,),
                    linewidth=1.3,
                    zorder=2,
                )
            )

        # Shadow
        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw + 0.004, pos["y_bot"] - 0.003),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        # Node
        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw, pos["y_bot"]),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor=_fc,
                edgecolor=_feat_edge,
                linewidth=0.7,
                zorder=3,
            )
        )

        _disp_name = DISPLAY_NAMES.get(f, f) if "DISPLAY_NAMES" in dir() else f
        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + _feat_nw + 0.025,
            pos["y_centre"],
            f"{_disp_name}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color="#1a1a1a",
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Column headers
    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    _hdr_tgt_col = "#0A1F4D"
    _hdr_feat_col = "#1E293B"
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS (CHANGE SCORES)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_tgt_col,
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_feat_col,
        fontfamily=_TITLE_FONT,
    )
    _rule_hw = 0.10
    ax.plot(
        [left_x - _rule_hw, left_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_tgt_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )
    ax.plot(
        [right_x - _rule_hw, right_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_feat_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )

    ax.text(0.50, 1.085, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)

    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, "sankey_child_deltas", dpi=1200, formats=("png", "pdf"))
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    return fig


# Child academic / cognitive: domain-colored Sankey for Academic/Cognitive batch
# 3 targets: bad_grades (concurrent), delta_bad_grades (change score), tb_cryst
# (NIH Toolbox crystallized intelligence composite). Palette separates the
# subjective (grade-based) vs objective (test-based) axes and the concurrent
# vs change-score axis using luminance + hue variation within a cool/earth
# family. Verified under deuteranopia, protanopia, tritanopia.

_CHILD_ACADEMIC_COGNITIVE_COLORS = {
    # Subjective academic axis: slate charcoal + lighter variant for delta.
    # Objective cognitive axis: teal (distinct hue from slate; CUD-safe).
    "bad_grades": (0.322, 0.322, 0.322),  # #525252 charcoal: concurrent subjective academic
    "delta_bad_grades": (0.549, 0.549, 0.549),  # #8C8C8C lighter slate: delta subjective academic
    "tb_cryst": (0.000, 0.620, 0.451),  # #009E73 CUD bluish-green: objective cognitive
}

_CHILD_ACADEMIC_COGNITIVE_TARGET_EDGE = {
    "bad_grades": "#1F1F1F",  # near-black (for contrast with charcoal fill)
    "delta_bad_grades": "#3A3A3A",  # darker slate
    "tb_cryst": "#005F45",  # dark bluish-green
}

_CHILD_ACADEMIC_COGNITIVE_DOMAIN_CLEAN_LABELS = {
    "bad_grades": "Academic Performance",
    "delta_bad_grades": "\u0394 Academic Performance",
    "tb_cryst": "Crystallized Intelligence",
}

_CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES = {
    "Poor Academic Grades": "bad_grades",
    "Academic Performance": "bad_grades",
    "Change in Bad Grades": "delta_bad_grades",
    "\u0394 Academic Performance": "delta_bad_grades",
    "Crystallized Intelligence": "tb_cryst",
    "Crystallized Composite (NIH)": "tb_cryst",
    "Cryst./Fluid Comp.": "tb_cryst",
}

_CHILD_ACADEMIC_COGNITIVE_ORDER = [
    "bad_grades",
    "delta_bad_grades",
    "tb_cryst",
]

_CHILD_ACADEMIC_COGNITIVE_EDGE = "#1A2332"
_CHILD_ACADEMIC_COGNITIVE_BG = "#FFFFFF"


def _child_academic_cognitive_resolve_color(target_name):
    if target_name in _CHILD_ACADEMIC_COGNITIVE_COLORS:
        return _CHILD_ACADEMIC_COGNITIVE_COLORS[target_name]
    raw = _CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_ACADEMIC_COGNITIVE_COLORS:
        return _CHILD_ACADEMIC_COGNITIVE_COLORS[raw]
    tl = target_name.lower()
    if "cryst" in tl:
        return _CHILD_ACADEMIC_COGNITIVE_COLORS["tb_cryst"]
    if "delta" in tl and ("grade" in tl or "academic" in tl):
        return _CHILD_ACADEMIC_COGNITIVE_COLORS["delta_bad_grades"]
    if "grade" in tl or "academic" in tl:
        return _CHILD_ACADEMIC_COGNITIVE_COLORS["bad_grades"]
    return (0.5, 0.5, 0.5)


def _child_academic_cognitive_domain(target_name):
    """Return a short domain key for an academic/cognitive target."""
    if target_name in _CHILD_ACADEMIC_COGNITIVE_COLORS:
        raw = target_name
    else:
        raw = _CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES.get(target_name, target_name)
    tl = raw.lower()
    if "cryst" in tl:
        return "crystallized"
    if "delta" in tl and ("grade" in tl or "academic" in tl):
        return "grades_delta"
    if "grade" in tl or "academic" in tl:
        return "grades"
    return "grades"


def _child_academic_cognitive_boxstyle(target_name):
    """Boxstyle keyed by domain.
    - crystallized (objective test-based): full pill
    - grades (concurrent subjective): tight rect
    - grades_delta (change score): rounded capsule
    """
    dom = _child_academic_cognitive_domain(target_name)
    if dom == "crystallized":
        return "round,pad=0.018"
    if dom == "grades_delta":
        return "round,pad=0.014"
    return "round,pad=0.008"


def _child_academic_cognitive_edge_for(target_name):
    """Resolve target node edge color."""
    if target_name in _CHILD_ACADEMIC_COGNITIVE_TARGET_EDGE:
        return _CHILD_ACADEMIC_COGNITIVE_TARGET_EDGE[target_name]
    raw = _CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_ACADEMIC_COGNITIVE_TARGET_EDGE:
        return _CHILD_ACADEMIC_COGNITIVE_TARGET_EDGE[raw]
    rgb = _child_academic_cognitive_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


def _child_academic_cognitive_feature_color(feature, df_sankey, feat_target_count, targets_list):
    """Feature node color. Single-target -> inherit target color.
    Multi-target -> steel-to-dark ramp keyed to overlap count (2T->3T range)."""
    n_tgt = feat_target_count.get(feature, 1)
    rows = df_sankey[df_sankey["Feature"] == feature]
    if rows.empty:
        return (0.5, 0.5, 0.5)
    if n_tgt == 1:
        dom_t = rows.loc[rows["Importance"].idxmax(), "Target"]
        return _child_academic_cognitive_resolve_color(dom_t)
    # Multi-target gradient: compressed range for 2-3 targets
    _t = min((n_tgt - 2) / 1.0, 1.0)  # 0 at 2T, 1.0 at 3T
    return (0.50 - 0.42 * _t, 0.52 - 0.42 * _t, 0.58 - 0.44 * _t)


_CHILD_ACADEMIC_COGNITIVE_FEATURE_EDGE_MAP = {
    "grades": "#1F1F1F",  # near-black
    "grades_delta": "#3A3A3A",  # darker slate
    "crystallized": "#005F45",  # dark bluish-green
}


def _child_academic_cognitive_sort_targets(targets_list):
    """Order targets by age DESCENDING (highest age at top, lowest at bottom),
    with R\u00b2 descending as secondary sort for ties within the same age.

    Falls back to pure R\u00b2-descending sort if no age suffixes are present
    (e.g., single-TP batch where targets never pass through _tp_suffix_to_age).
    Final fallback is _CHILD_ACADEMIC_COGNITIVE_ORDER.
    """
    import re

    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}

    def _age_of(t):
        """Parse trailing '(Age N)' -> int; return -1 if not found."""
        m = re.search(r"\(Age (\d+)\)$", str(t))
        return int(m.group(1)) if m else -1

    def _r2_for(t):
        # 4-tier lookup chain mirroring the metric-render code:
        # direct, TDN, reverse-age->TP, TDN-of-TP-form.
        tp_form = _age_label_to_tp(t)
        for key in (t, _get_tdn(t), tp_form, _get_tdn(tp_form)):
            entry = _tm.get(key)
            if entry is not None:
                return entry[0]
        raw = _CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES.get(t) or _CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES.get(tp_form)
        if raw:
            entry = _tm.get(raw) or _tm.get(_tdn.get(raw, ""))
            if entry is not None:
                return entry[0]
        return None

    # Pre-compute to avoid redundant work in sort key
    ages = {t: _age_of(t) for t in targets_list}
    r2s = {t: _r2_for(t) for t in targets_list}
    any_age = any(v >= 0 for v in ages.values())

    if any_age:
        # Primary: age DESC (highest age -> top of figure).
        # Secondary: R\u00b2 DESC (within same age, higher-performing target on top).
        return sorted(
            targets_list,
            key=lambda t: (-ages[t], -(r2s[t] if r2s[t] is not None else -999)),
        )

    # Fallback path: no age suffixes (single-TP batch). Preserve original
    # R\u00b2-descending behavior so non-TP-sweep runs still look sensible. ---
    has_metrics = any(v is not None for v in r2s.values())
    if has_metrics:
        return sorted(targets_list, key=lambda t: r2s.get(t) if r2s.get(t) is not None else -999, reverse=True)

    def _rank(t):
        if t in _CHILD_ACADEMIC_COGNITIVE_ORDER:
            return _CHILD_ACADEMIC_COGNITIVE_ORDER.index(t)
        raw = _CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES.get(t)
        if raw and raw in _CHILD_ACADEMIC_COGNITIVE_ORDER:
            return _CHILD_ACADEMIC_COGNITIVE_ORDER.index(raw)
        return 99

    return sorted(targets_list, key=_rank)


def _draw_sankey_child_academic_cognitive(df_sankey, sankey_top_features, save_dir=None):
    """Child Academic/Cognitive Sankey: 3 targets (bad_grades, delta_bad_grades, tb_cryst).

    Mirrors _draw_sankey_child_deltas implementation with:
      1. 3-target palette (slate concurrent / lighter slate delta / teal objective)
      2. Port-gap 0.003
      3. Saturation + luminance ramp for ribbon importance encoding
      4. Domain-based target node shape differentiation
      5. Domain-aware target edge colors
      6. Figure 1.5x scale + 1200 DPI output
      7. Uniform target/feature node heights
    """

    # Relabel any trailing '(TN)' -> '(Age N+9)' using the ABCD canonical mapping.
    # Applied BEFORE _sankey_common_data so target_totals / positions key on age labels.
    df_sankey = df_sankey.copy()
    df_sankey["Target"] = df_sankey["Target"].apply(_tp_suffix_to_age)

    _cd = _sankey_common_data(df_sankey)
    targets_list = _child_academic_cognitive_sort_targets(_cd["targets_list"])
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: _child_academic_cognitive_resolve_color(t) for t in targets_list}
    _BG = _CHILD_ACADEMIC_COGNITIVE_BG

    # Layout constants
    _pad = max(0.022, min(0.040, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.052
    _feat_nw = 0.028
    _target_node_w = 0.036

    # Figure scaling x1.5
    _, _base_h = _wide_figure_dims(n_targets, n_features)
    fig_w = 15 * 1.5
    fig_h = max(_base_h * 1.3, n_features * 0.52, 22) * 1.5
    _spread_t = 2.2  # tighter for 3 targets
    _spread_f = 1.9
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()
    _target_fs = 17

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.18, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.050, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    # Uniform target node heights (slightly taller for 3-target layout)
    _RECT_H = 0.20
    for t in targets_list:
        if t in target_positions:
            _c = target_positions[t]["y_centre"]
            target_positions[t] = {"y_centre": _c, "y_top": _c + _RECT_H / 2, "y_bot": _c - _RECT_H / 2, "h": _RECT_H}

    # Uniform feature node heights
    _FEAT_H = 0.056
    for f_key in features_list:
        if f_key in feature_positions:
            _c = feature_positions[f_key]["y_centre"]
            feature_positions[f_key] = {
                "y_centre": _c,
                "y_top": _c + _FEAT_H / 2,
                "y_bot": _c - _FEAT_H / 2,
                "h": _FEAT_H,
            }

    # Figure
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04
    ax.set_xlim(-0.08, right_x + 0.22)
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    # RIBBONS: port-gap 0.003 + saturation/luminance ramp
    _PORT_GAP = 0.003

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])

        _yl_t = rb["y_l_top"] - _PORT_GAP
        _yl_b = rb["y_l_bot"] + _PORT_GAP
        _yr_t = rb["y_r_top"] - _PORT_GAP * 0.5
        _yr_b = rb["y_r_bot"] + _PORT_GAP * 0.5
        if _yl_t <= _yl_b or _yr_t <= _yr_b:
            _yl_t, _yl_b = rb["y_l_top"], rb["y_l_bot"]
            _yr_t, _yr_b = rb["y_r_top"], rb["y_r_bot"]

        _desat = 0.55 + 0.45 * rel
        _white = (0.92, 0.93, 0.96)
        _fill = tuple(_white[i] + (_rgb[i] - _white[i]) * _desat for i in range(3))
        _alpha = 0.55 + 0.30 * rel

        if rel > 0.45:
            _glow_n = 2 if rel > 0.75 else 1
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                _yl_t + 0.005,
                _yl_b - 0.005,
                _yr_t + 0.003,
                _yr_b - 0.003,
                color_rgba=_rgb + (0.08,),
                zorder=1,
                glow_layers=_glow_n,
            )

        if rel > 0.70:
            _edge_dark = tuple(max(0, c * 0.70) for c in _rgb)
            _e_lw = 0.35 + 0.25 * (rel - 0.70) / 0.30
            _e_alpha = 0.22 + 0.18 * (rel - 0.70) / 0.30
            _edge_c = _edge_dark + (_e_alpha,)
        else:
            _edge_c = _fill + (min(_alpha + 0.05, 0.65),)
            _e_lw = 0.2

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            _yl_t,
            _yl_b,
            _yr_t,
            _yr_b,
            color_rgba=_fill + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_e_lw,
            right_node_w=_feat_nw,
        )

    # Target nodes: domain boxstyle + domain edge + R^2 label under title
    for t, pos in target_positions.items():
        rgb = _tc.get(t, (0.5, 0.5, 0.5))
        _edge_col = _child_academic_cognitive_edge_for(t)
        _bstyle = _child_academic_cognitive_boxstyle(t)

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.004, pos["y_bot"] - 0.003),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor=rgb,
                edgecolor=_edge_col,
                linewidth=1.8,
                zorder=3,
            )
        )

        raw_key = (
            t
            if t in _CHILD_ACADEMIC_COGNITIVE_DOMAIN_CLEAN_LABELS
            else _CHILD_ACADEMIC_COGNITIVE_DISPLAY_ALIASES.get(t, t)
        )
        _domain_label = _CHILD_ACADEMIC_COGNITIVE_DOMAIN_CLEAN_LABELS.get(raw_key, _get_tdn(t))

        _tp_form = _age_label_to_tp(t)  # reverse-map for metric-dict lookup
        _metric_val = (
            _target_metrics.get(t)
            or _target_metrics.get(_get_tdn(t))
            or _target_metrics.get(_tp_form)
            or _target_metrics.get(_get_tdn(_tp_form))
        )
        _label_x = left_x - _target_node_w - 0.04

        if _metric_val is not None:
            _v, _l = _metric_val
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.025,
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.028,
                f"{_l} = {_v:.3f}",
                ha="right",
                va="center",
                fontsize=_target_fs - 3,
                fontweight="bold",
                color=_edge_col,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Feature nodes
    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count.get(f, 1)
        _fc = _child_academic_cognitive_feature_color(f, df_sankey, feat_target_count, targets_list)

        _f_rows = df_sankey[df_sankey["Feature"] == f]
        _domain_counts = {}
        for _ft in _f_rows["Target"]:
            _dom_key = _child_academic_cognitive_domain(_ft)
            _domain_counts[_dom_key] = _domain_counts.get(_dom_key, 0) + 1
        _dom_winner = max(_domain_counts, key=_domain_counts.get) if _domain_counts else "grades"
        _feat_edge = _CHILD_ACADEMIC_COGNITIVE_FEATURE_EDGE_MAP.get(_dom_winner, "#1A1A1A")

        if n_tgt >= 2:
            dom_rows = df_sankey[df_sankey["Feature"] == f]
            dom_t = dom_rows.loc[dom_rows["Importance"].idxmax(), "Target"] if not dom_rows.empty else targets_list[0]
            glow_col = _tc.get(dom_t, (0.5, 0.5, 0.5))
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - _feat_nw - 0.006, pos["y_bot"] - 0.006),
                    _feat_nw * 2 + 0.012,
                    pos["h"] + 0.012,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_col + (0.38,),
                    linewidth=1.3,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw + 0.004, pos["y_bot"] - 0.003),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw, pos["y_bot"]),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor=_fc,
                edgecolor=_feat_edge,
                linewidth=0.7,
                zorder=3,
            )
        )

        _disp_name = DISPLAY_NAMES.get(f, f) if "DISPLAY_NAMES" in dir() else f
        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + _feat_nw + 0.025,
            pos["y_centre"],
            f"{_disp_name}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color="#1a1a1a",
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    # Column headers
    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    _hdr_tgt_col = "#0A1F4D"
    _hdr_feat_col = "#1E293B"
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS (ACADEMIC / COGNITIVE)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_tgt_col,
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_feat_col,
        fontfamily=_TITLE_FONT,
    )
    _rule_hw = 0.10
    ax.plot(
        [left_x - _rule_hw, left_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_tgt_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )
    ax.plot(
        [right_x - _rule_hw, right_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_feat_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )

    ax.text(0.50, 1.085, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)

    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, "sankey_child_academic_cognitive", dpi=1200, formats=("png", "pdf"))
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    return fig


# CHILD ADHD / impulsivity / compulsivity: domain-colored Sankey
# 3 targets: adhd_D_p (CBCL DSM-oriented ADHD composite), impulsive_p (CBCL
# q41 single item), compulsions_p (CBCL q66 single item). Palette separates
# the three dyscontrol dimensions: ADHD = amber (broad), impulsivity = red-
# orange (stop-failure), compulsivity = violet (repetitive-action).
# Verified CUD-safe under deuteranopia, protanopia, tritanopia.

_CHILD_ADHD_IMPCOMP_COLORS = {
    # Palette per user spec: ADHD=yellow, Impulsivity=chartreuse (yellow-green),
    # Compulsivity=turquoise. Saturation/luminance tuned for white-background
    # readability and acceptable CVD performance.  Hue separations: yellow->
    # yellow-green ~35 deg, yellow-green->turquoise ~95 deg: the latter is
    # well-distinguished under all CVD simulations; the former is tighter in
    # deuteranopia (yellow and yellow-green may desaturate toward similar
    # ochre) but saturation + boxstyle differentiation preserve identity.
    "adhd_D_p": (0.922, 0.706, 0.063),  # #EBB410 gold-yellow: broad ADHD composite
    "impulsive_p": (0.557, 0.788, 0.275),  # #8EC946 chartreuse  -- narrow impulsivity item
    "compulsions_p": (0.157, 0.761, 0.824),  # #28C2D2 turquoise   -- narrow compulsivity item
}

_CHILD_ADHD_IMPCOMP_TARGET_EDGE = {
    # Edge colors at ~55-60% luminance of each fill, preserving hue identity.
    "adhd_D_p": "#8A6900",  # dark goldenrod    (yellow edge)
    "impulsive_p": "#4E7120",  # dark olive-green  (chartreuse edge)
    "compulsions_p": "#0D6675",  # dark teal         (turquoise edge)
}

_CHILD_ADHD_IMPCOMP_DOMAIN_CLEAN_LABELS = {
    "adhd_D_p": "ADHD Syndrome",
    "impulsive_p": "Impulsivity",
    "compulsions_p": "Compulsivity",
}

_CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES = {
    "ADHD Syndrome": "adhd_D_p",
    "ADHD": "adhd_D_p",
    "Impulsive Behavior": "impulsive_p",
    "Impulsivity": "impulsive_p",
    "Compulsive Behavior": "compulsions_p",
    "Compulsivity": "compulsions_p",
}

_CHILD_ADHD_IMPCOMP_ORDER = [
    "adhd_D_p",
    "impulsive_p",
    "compulsions_p",
]

_CHILD_ADHD_IMPCOMP_EDGE = "#1A2332"
_CHILD_ADHD_IMPCOMP_BG = "#FFFFFF"


def _child_adhd_impcomp_resolve_color(target_name):
    if target_name in _CHILD_ADHD_IMPCOMP_COLORS:
        return _CHILD_ADHD_IMPCOMP_COLORS[target_name]
    raw = _CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_ADHD_IMPCOMP_COLORS:
        return _CHILD_ADHD_IMPCOMP_COLORS[raw]
    tl = target_name.lower()
    if "compuls" in tl:
        return _CHILD_ADHD_IMPCOMP_COLORS["compulsions_p"]
    if "impuls" in tl:
        return _CHILD_ADHD_IMPCOMP_COLORS["impulsive_p"]
    if "adhd" in tl or "attention" in tl:
        return _CHILD_ADHD_IMPCOMP_COLORS["adhd_D_p"]
    return (0.5, 0.5, 0.5)


def _child_adhd_impcomp_domain(target_name):
    """Return a short domain key for an ADHD/impulsivity/compulsivity target."""
    if target_name in _CHILD_ADHD_IMPCOMP_COLORS:
        raw = target_name
    else:
        raw = _CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES.get(target_name, target_name)
    tl = raw.lower()
    if "compuls" in tl:
        return "compulsivity"
    if "impuls" in tl:
        return "impulsivity"
    if "adhd" in tl or "attention" in tl:
        return "adhd"
    return "adhd"


def _child_adhd_impcomp_boxstyle(target_name):
    """Boxstyle keyed by domain.
    - adhd (broad syndrome composite): tight rect
    - impulsivity (narrow stop-failure item): rounded capsule
    - compulsivity (narrow repetitive-action item): full pill
    """
    dom = _child_adhd_impcomp_domain(target_name)
    if dom == "compulsivity":
        return "round,pad=0.018"
    if dom == "impulsivity":
        return "round,pad=0.014"
    return "round,pad=0.008"


def _child_adhd_impcomp_edge_for(target_name):
    """Resolve target node edge color."""
    if target_name in _CHILD_ADHD_IMPCOMP_TARGET_EDGE:
        return _CHILD_ADHD_IMPCOMP_TARGET_EDGE[target_name]
    raw = _CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _CHILD_ADHD_IMPCOMP_TARGET_EDGE:
        return _CHILD_ADHD_IMPCOMP_TARGET_EDGE[raw]
    rgb = _child_adhd_impcomp_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


def _child_adhd_impcomp_feature_color(feature, df_sankey, feat_target_count, targets_list):
    """Feature node color. Single-target -> inherit. Multi-target -> steel ramp."""
    n_tgt = feat_target_count.get(feature, 1)
    rows = df_sankey[df_sankey["Feature"] == feature]
    if rows.empty:
        return (0.5, 0.5, 0.5)
    if n_tgt == 1:
        dom_t = rows.loc[rows["Importance"].idxmax(), "Target"]
        return _child_adhd_impcomp_resolve_color(dom_t)
    _t = min((n_tgt - 2) / 1.0, 1.0)
    return (0.50 - 0.42 * _t, 0.52 - 0.42 * _t, 0.58 - 0.44 * _t)


_CHILD_ADHD_IMPCOMP_FEATURE_EDGE_MAP = {
    # Keyed by majority-domain of each feature's incoming ribbons.
    "adhd": "#8A6900",  # dark goldenrod
    "impulsivity": "#4E7120",  # dark olive-green
    "compulsivity": "#0D6675",  # dark teal
}


def _child_adhd_impcomp_sort_targets(targets_list):
    """Order targets by age DESCENDING (highest age at top, lowest at bottom),
    with R\u00b2 descending as secondary sort for ties within the same age.

    Falls back to pure R\u00b2-descending sort if no age suffixes are present.
    Final fallback is _CHILD_ADHD_IMPCOMP_ORDER.
    """
    import re

    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}

    def _age_of(t):
        m = re.search(r"\(Age (\d+)\)$", str(t))
        return int(m.group(1)) if m else -1

    def _r2_for(t):
        tp_form = _age_label_to_tp(t)
        for key in (t, _get_tdn(t), tp_form, _get_tdn(tp_form)):
            entry = _tm.get(key)
            if entry is not None:
                return entry[0]
        raw = _CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES.get(t) or _CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES.get(tp_form)
        if raw:
            entry = _tm.get(raw) or _tm.get(_tdn.get(raw, ""))
            if entry is not None:
                return entry[0]
        return None

    ages = {t: _age_of(t) for t in targets_list}
    r2s = {t: _r2_for(t) for t in targets_list}
    any_age = any(v >= 0 for v in ages.values())

    if any_age:
        return sorted(
            targets_list,
            key=lambda t: (-ages[t], -(r2s[t] if r2s[t] is not None else -999)),
        )

    has_metrics = any(v is not None for v in r2s.values())
    if has_metrics:
        return sorted(targets_list, key=lambda t: r2s.get(t) if r2s.get(t) is not None else -999, reverse=True)

    def _rank(t):
        if t in _CHILD_ADHD_IMPCOMP_ORDER:
            return _CHILD_ADHD_IMPCOMP_ORDER.index(t)
        raw = _CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES.get(t)
        if raw and raw in _CHILD_ADHD_IMPCOMP_ORDER:
            return _CHILD_ADHD_IMPCOMP_ORDER.index(raw)
        return 99

    return sorted(targets_list, key=_rank)


def _draw_sankey_child_adhd_impcomp(df_sankey, sankey_top_features, save_dir=None):
    """Child ADHD/Impulsivity/Compulsivity Sankey: 3 dyscontrol targets.

    Mirrors _draw_sankey_child_deltas implementation with:
      1. 3-target CUD-safe palette (orange / vermillion / violet)
      2. Port-gap 0.003
      3. Saturation + luminance ramp for ribbon importance encoding
      4. Domain-based target node shape differentiation
      5. Domain-aware target edge colors
      6. Figure 1.5x scale + 1200 DPI output
      7. Uniform target/feature node heights
    """

    # Relabel any trailing '(TN)' -> '(Age N+9)' using the ABCD canonical mapping.
    df_sankey = df_sankey.copy()
    df_sankey["Target"] = df_sankey["Target"].apply(_tp_suffix_to_age)

    _cd = _sankey_common_data(df_sankey)
    targets_list = _child_adhd_impcomp_sort_targets(_cd["targets_list"])
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: _child_adhd_impcomp_resolve_color(t) for t in targets_list}
    _BG = _CHILD_ADHD_IMPCOMP_BG

    _pad = max(0.022, min(0.040, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.052
    _feat_nw = 0.028
    _target_node_w = 0.036

    _, _base_h = _wide_figure_dims(n_targets, n_features)
    fig_w = 15 * 1.5
    fig_h = max(_base_h * 1.3, n_features * 0.52, 22) * 1.5
    _spread_t = 2.2
    _spread_f = 1.9
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()
    _target_fs = 17

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.18, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.050, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    _RECT_H = 0.20
    for t in targets_list:
        if t in target_positions:
            _c = target_positions[t]["y_centre"]
            target_positions[t] = {"y_centre": _c, "y_top": _c + _RECT_H / 2, "y_bot": _c - _RECT_H / 2, "h": _RECT_H}

    _FEAT_H = 0.056
    for f_key in features_list:
        if f_key in feature_positions:
            _c = feature_positions[f_key]["y_centre"]
            feature_positions[f_key] = {
                "y_centre": _c,
                "y_top": _c + _FEAT_H / 2,
                "y_bot": _c - _FEAT_H / 2,
                "h": _FEAT_H,
            }

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04
    ax.set_xlim(-0.08, right_x + 0.22)
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    _PORT_GAP = 0.003

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])

        _yl_t = rb["y_l_top"] - _PORT_GAP
        _yl_b = rb["y_l_bot"] + _PORT_GAP
        _yr_t = rb["y_r_top"] - _PORT_GAP * 0.5
        _yr_b = rb["y_r_bot"] + _PORT_GAP * 0.5
        if _yl_t <= _yl_b or _yr_t <= _yr_b:
            _yl_t, _yl_b = rb["y_l_top"], rb["y_l_bot"]
            _yr_t, _yr_b = rb["y_r_top"], rb["y_r_bot"]

        _desat = 0.55 + 0.45 * rel
        _white = (0.92, 0.93, 0.96)
        _fill = tuple(_white[i] + (_rgb[i] - _white[i]) * _desat for i in range(3))
        _alpha = 0.55 + 0.30 * rel

        if rel > 0.45:
            _glow_n = 2 if rel > 0.75 else 1
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                _yl_t + 0.005,
                _yl_b - 0.005,
                _yr_t + 0.003,
                _yr_b - 0.003,
                color_rgba=_rgb + (0.08,),
                zorder=1,
                glow_layers=_glow_n,
            )

        if rel > 0.70:
            _edge_dark = tuple(max(0, c * 0.70) for c in _rgb)
            _e_lw = 0.35 + 0.25 * (rel - 0.70) / 0.30
            _e_alpha = 0.22 + 0.18 * (rel - 0.70) / 0.30
            _edge_c = _edge_dark + (_e_alpha,)
        else:
            _edge_c = _fill + (min(_alpha + 0.05, 0.65),)
            _e_lw = 0.2

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            _yl_t,
            _yl_b,
            _yr_t,
            _yr_b,
            color_rgba=_fill + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_e_lw,
            right_node_w=_feat_nw,
        )

    for t, pos in target_positions.items():
        rgb = _tc.get(t, (0.5, 0.5, 0.5))
        _edge_col = _child_adhd_impcomp_edge_for(t)
        _bstyle = _child_adhd_impcomp_boxstyle(t)

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.004, pos["y_bot"] - 0.003),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor=rgb,
                edgecolor=_edge_col,
                linewidth=1.8,
                zorder=3,
            )
        )

        raw_key = t if t in _CHILD_ADHD_IMPCOMP_DOMAIN_CLEAN_LABELS else _CHILD_ADHD_IMPCOMP_DISPLAY_ALIASES.get(t, t)
        _domain_label = _CHILD_ADHD_IMPCOMP_DOMAIN_CLEAN_LABELS.get(raw_key, _get_tdn(t))

        _tp_form = _age_label_to_tp(t)  # reverse-map for metric-dict lookup
        _metric_val = (
            _target_metrics.get(t)
            or _target_metrics.get(_get_tdn(t))
            or _target_metrics.get(_tp_form)
            or _target_metrics.get(_get_tdn(_tp_form))
        )
        _label_x = left_x - _target_node_w - 0.04

        if _metric_val is not None:
            _v, _l = _metric_val
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.025,
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.028,
                f"{_l} = {_v:.3f}",
                ha="right",
                va="center",
                fontsize=_target_fs - 3,
                fontweight="bold",
                color=_edge_col,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count.get(f, 1)
        _fc = _child_adhd_impcomp_feature_color(f, df_sankey, feat_target_count, targets_list)

        _f_rows = df_sankey[df_sankey["Feature"] == f]
        _domain_counts = {}
        for _ft in _f_rows["Target"]:
            _dom_key = _child_adhd_impcomp_domain(_ft)
            _domain_counts[_dom_key] = _domain_counts.get(_dom_key, 0) + 1
        _dom_winner = max(_domain_counts, key=_domain_counts.get) if _domain_counts else "adhd"
        _feat_edge = _CHILD_ADHD_IMPCOMP_FEATURE_EDGE_MAP.get(_dom_winner, "#1A1A1A")

        if n_tgt >= 2:
            dom_rows = df_sankey[df_sankey["Feature"] == f]
            dom_t = dom_rows.loc[dom_rows["Importance"].idxmax(), "Target"] if not dom_rows.empty else targets_list[0]
            glow_col = _tc.get(dom_t, (0.5, 0.5, 0.5))
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - _feat_nw - 0.006, pos["y_bot"] - 0.006),
                    _feat_nw * 2 + 0.012,
                    pos["h"] + 0.012,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_col + (0.38,),
                    linewidth=1.3,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw + 0.004, pos["y_bot"] - 0.003),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw, pos["y_bot"]),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor=_fc,
                edgecolor=_feat_edge,
                linewidth=0.7,
                zorder=3,
            )
        )

        _disp_name = DISPLAY_NAMES.get(f, f) if "DISPLAY_NAMES" in dir() else f
        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + _feat_nw + 0.025,
            pos["y_centre"],
            f"{_disp_name}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color="#1a1a1a",
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    _hdr_tgt_col = "#0A1F4D"
    _hdr_feat_col = "#1E293B"
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS (ADHD / IMPULSIVITY / COMPULSIVITY)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_tgt_col,
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_feat_col,
        fontfamily=_TITLE_FONT,
    )
    _rule_hw = 0.10
    ax.plot(
        [left_x - _rule_hw, left_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_tgt_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )
    ax.plot(
        [right_x - _rule_hw, right_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_feat_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )

    ax.text(0.50, 1.085, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)

    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, "sankey_child_adhd_impcomp", dpi=1200, formats=("png", "pdf"))
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    return fig


# Parent social/impulsivity/compulsivity: Sankey palette and helpers
# Parent ASR analog of the child ADHD/Impulsivity/Compulsivity batch.
# 5 regression targets spanning three latent dimensions:
#   - Avoidant Personality (DSM composite)        -- broad social-withdrawal composite
#   - Not Liked by Others, Impulsivity            -- narrow social/externalizing items
#   - Repeats Acts (Compulsions), Obsessive       -- narrow OCD items
#     Thoughts (Obsessions)
# Palette: blue-violet (Avoidant) / coral (Not Liked) / chartreuse (Impulsive,
# mirrors child) / turquoise (Repeats Acts, mirrors child compulsions) /
# violet (Obsessive Thoughts). Verified CUD-safe under deuteranopia,
# protanopia, tritanopia: max ambiguous pair (chartreuse vs turquoise) is
# disambiguated by hue separation (~95 deg) and saturation contrast.
# Boxstyle tiering: composite (tight rect) / social+impulsivity item
# (rounded capsule) / OCD item (full pill): preserves identity under CVD.

_PARENT_SOCIAL_IMPCOMP_COLORS = {
    "parent_avoidant_person_D_p": (
        0.294,
        0.420,
        0.745,
    ),  # #4B6BBE steel-blue/violet: broad Avoidant Personality composite (internalizing/social)
    "parent_not_liked_by_others_p": (
        0.878,
        0.443,
        0.424,
    ),  # #E0716C coral             -- narrow peer-dislike item (social)
    "parent_impulsive_p": (
        0.557,
        0.788,
        0.275,
    ),  # #8EC946 chartreuse        -- narrow impulsivity item (mirrors child impulsive_p)
    "parent_repeats_acts_p": (
        0.157,
        0.761,
        0.824,
    ),  # #28C2D2 turquoise         -- narrow compulsivity item (mirrors child compulsions_p)
    "parent_obsessive_thoughts_p": (0.557, 0.365, 0.749),  # #8E5DBF violet            -- narrow obsessions item (OCD)
}

_PARENT_SOCIAL_IMPCOMP_TARGET_EDGE = {
    # Edge colors at ~55-60% luminance of each fill, preserving hue identity.
    "parent_avoidant_person_D_p": "#1F2D5C",  # dark blue-violet
    "parent_not_liked_by_others_p": "#7F2522",  # dark crimson
    "parent_impulsive_p": "#4E7120",  # dark olive-green
    "parent_repeats_acts_p": "#0D6675",  # dark teal
    "parent_obsessive_thoughts_p": "#3D2659",  # dark violet
}

_PARENT_SOCIAL_IMPCOMP_DOMAIN_CLEAN_LABELS = {
    "parent_avoidant_person_D_p": "Avoidant Personality",
    "parent_not_liked_by_others_p": "Not Liked by Others",
    "parent_impulsive_p": "Impulsivity",
    "parent_repeats_acts_p": "Compulsivity",
    "parent_obsessive_thoughts_p": "Obsessions",
}

_PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES = {
    "Avoidant Personality": "parent_avoidant_person_D_p",
    "Avoidant Personality (Parent)": "parent_avoidant_person_D_p",
    "Parent Avoidant Personality": "parent_avoidant_person_D_p",
    "Not Liked by Others": "parent_not_liked_by_others_p",
    "Not Liked by Others (Parent)": "parent_not_liked_by_others_p",
    "Parent Not Liked by Others": "parent_not_liked_by_others_p",
    "Impulsivity (Parent)": "parent_impulsive_p",
    "Parent Impulsivity": "parent_impulsive_p",
    "Impulsivity": "parent_impulsive_p",
    "Compulsivity (Parent)": "parent_repeats_acts_p",
    "Parent Compulsivity": "parent_repeats_acts_p",
    "Compulsivity": "parent_repeats_acts_p",
    "Repeats Acts": "parent_repeats_acts_p",
    "Obsessions (Parent)": "parent_obsessive_thoughts_p",
    "Parent Obsessions": "parent_obsessive_thoughts_p",
    "Obsessions": "parent_obsessive_thoughts_p",
    "Obsessive Thoughts": "parent_obsessive_thoughts_p",
}

# Canonical sort order: composite -> social item -> externalizing item ->
# OCD compulsion item -> OCD obsession item. Mirrors clinical progression
# from broad internalizing/social to narrow OCD specifics.
_PARENT_SOCIAL_IMPCOMP_ORDER = [
    "parent_avoidant_person_D_p",
    "parent_not_liked_by_others_p",
    "parent_impulsive_p",
    "parent_repeats_acts_p",
    "parent_obsessive_thoughts_p",
]

_PARENT_SOCIAL_IMPCOMP_EDGE = "#1A2332"
_PARENT_SOCIAL_IMPCOMP_BG = "#FFFFFF"


def _parent_social_impcomp_resolve_color(target_name):
    if target_name in _PARENT_SOCIAL_IMPCOMP_COLORS:
        return _PARENT_SOCIAL_IMPCOMP_COLORS[target_name]
    raw = _PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _PARENT_SOCIAL_IMPCOMP_COLORS:
        return _PARENT_SOCIAL_IMPCOMP_COLORS[raw]
    tl = target_name.lower()
    if "avoidant" in tl:
        return _PARENT_SOCIAL_IMPCOMP_COLORS["parent_avoidant_person_D_p"]
    if "not liked" in tl or "not_liked" in tl:
        return _PARENT_SOCIAL_IMPCOMP_COLORS["parent_not_liked_by_others_p"]
    if "compuls" in tl or "repeats" in tl:
        return _PARENT_SOCIAL_IMPCOMP_COLORS["parent_repeats_acts_p"]
    if "obsess" in tl:
        return _PARENT_SOCIAL_IMPCOMP_COLORS["parent_obsessive_thoughts_p"]
    if "impuls" in tl:
        return _PARENT_SOCIAL_IMPCOMP_COLORS["parent_impulsive_p"]
    return (0.5, 0.5, 0.5)


def _parent_social_impcomp_domain(target_name):
    """Return a short domain key for a parent social/impulsivity/compulsivity target."""
    if target_name in _PARENT_SOCIAL_IMPCOMP_COLORS:
        raw = target_name
    else:
        raw = _PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES.get(target_name, target_name)
    tl = raw.lower()
    if "avoidant" in tl:
        return "avoidant"
    if "not_liked" in tl or "not liked" in tl:
        return "social"
    if "compuls" in tl or "repeats" in tl:
        return "compulsivity"
    if "obsess" in tl:
        return "obsession"
    if "impuls" in tl:
        return "impulsivity"
    return "avoidant"


def _parent_social_impcomp_boxstyle(target_name):
    """Boxstyle keyed by domain.
    - avoidant (broad DSM composite): tight rect
    - social / impulsivity (narrow externalising-side items): rounded capsule
    - compulsivity / obsession (narrow OCD items): full pill
    """
    dom = _parent_social_impcomp_domain(target_name)
    if dom in ("compulsivity", "obsession"):
        return "round,pad=0.018"
    if dom in ("social", "impulsivity"):
        return "round,pad=0.014"
    return "round,pad=0.008"


def _parent_social_impcomp_edge_for(target_name):
    """Resolve target node edge color."""
    if target_name in _PARENT_SOCIAL_IMPCOMP_TARGET_EDGE:
        return _PARENT_SOCIAL_IMPCOMP_TARGET_EDGE[target_name]
    raw = _PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES.get(target_name)
    if raw and raw in _PARENT_SOCIAL_IMPCOMP_TARGET_EDGE:
        return _PARENT_SOCIAL_IMPCOMP_TARGET_EDGE[raw]
    rgb = _parent_social_impcomp_resolve_color(target_name)
    return tuple(max(0, c * 0.65) for c in rgb)


def _parent_social_impcomp_feature_color(feature, df_sankey, feat_target_count, targets_list):
    """Feature node color. Single-target -> inherit. Multi-target -> steel ramp."""
    n_tgt = feat_target_count.get(feature, 1)
    rows = df_sankey[df_sankey["Feature"] == feature]
    if rows.empty:
        return (0.5, 0.5, 0.5)
    if n_tgt == 1:
        dom_t = rows.loc[rows["Importance"].idxmax(), "Target"]
        return _parent_social_impcomp_resolve_color(dom_t)
    _t = min((n_tgt - 2) / 3.0, 1.0)
    return (0.50 - 0.42 * _t, 0.52 - 0.42 * _t, 0.58 - 0.44 * _t)


_PARENT_SOCIAL_IMPCOMP_FEATURE_EDGE_MAP = {
    # Keyed by majority-domain of each feature's incoming ribbons.
    "avoidant": "#1F2D5C",  # dark blue-violet
    "social": "#7F2522",  # dark crimson
    "impulsivity": "#4E7120",  # dark olive-green
    "compulsivity": "#0D6675",  # dark teal
    "obsession": "#3D2659",  # dark violet
}


def _parent_social_impcomp_sort_targets(targets_list):
    """Order targets by age DESCENDING (highest age at top, lowest at bottom),
    with R\u00b2 descending as secondary sort for ties within the same age.

    Falls back to pure R\u00b2-descending sort if no age suffixes are present.
    Final fallback is _PARENT_SOCIAL_IMPCOMP_ORDER.
    """
    import re

    _ba = ctx.get("_batch_archive", {})
    _tdn = ctx.get("TARGET_DISPLAY_NAMES", {})
    _tm = _extract_target_metrics(_ba) if _ba else {}

    def _age_of(t):
        # Parse "(Year N)" suffix (parent self-report, years-from-baseline).
        m = re.search(r"\(Year (\d+)\)$", str(t))
        return int(m.group(1)) if m else -1

    def _r2_for(t):
        tp_form = _year_label_to_tp(t)
        for key in (t, _get_tdn(t), tp_form, _get_tdn(tp_form)):
            entry = _tm.get(key)
            if entry is not None:
                return entry[0]
        raw = _PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES.get(t) or _PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES.get(tp_form)
        if raw:
            entry = _tm.get(raw) or _tm.get(_tdn.get(raw, ""))
            if entry is not None:
                return entry[0]
        return None

    ages = {t: _age_of(t) for t in targets_list}
    r2s = {t: _r2_for(t) for t in targets_list}
    any_age = any(v >= 0 for v in ages.values())

    if any_age:
        return sorted(
            targets_list,
            key=lambda t: (-ages[t], -(r2s[t] if r2s[t] is not None else -999)),
        )

    has_metrics = any(v is not None for v in r2s.values())
    if has_metrics:
        return sorted(targets_list, key=lambda t: r2s.get(t) if r2s.get(t) is not None else -999, reverse=True)

    def _rank(t):
        if t in _PARENT_SOCIAL_IMPCOMP_ORDER:
            return _PARENT_SOCIAL_IMPCOMP_ORDER.index(t)
        raw = _PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES.get(t)
        if raw and raw in _PARENT_SOCIAL_IMPCOMP_ORDER:
            return _PARENT_SOCIAL_IMPCOMP_ORDER.index(raw)
        return 99

    return sorted(targets_list, key=_rank)


def _draw_sankey_parent_social_impcomp(df_sankey, sankey_top_features, save_dir=None):
    """Parent Social/Impulsivity/Compulsivity Sankey: 5 ASR-rated targets.

    ASR analog of the child ADHD/Impulsivity/Compulsivity Sankey. Mirrors
    _draw_sankey_child_adhd_impcomp implementation with:
      1. 5-target CUD-safe palette (blue-violet / coral / chartreuse /
         turquoise / violet)
      2. Port-gap 0.003
      3. Saturation + luminance ramp for ribbon importance encoding
      4. 3-tier domain-based target node shape (composite / social-impulsivity
         item / OCD item)
      5. Domain-aware target edge colors
      6. Figure 1.5x scale + 1200 DPI output
      7. Uniform target/feature node heights
    """

    # Relabel any trailing '(TN)' -> '(Year N)': years-from-baseline framing.
    # Parent targets are self-reports; the ABCD child-age mapping used by
    # _tp_suffix_to_age is misleading here (ages refer to the child rather
    # than the parent rater). Year-from-baseline preserves longitudinal
    # ordering without imposing a child-age frame.
    df_sankey = df_sankey.copy()
    df_sankey["Target"] = df_sankey["Target"].apply(_tp_suffix_to_year)

    _cd = _sankey_common_data(df_sankey)
    targets_list = _parent_social_impcomp_sort_targets(_cd["targets_list"])
    features_list = _cd["features_list"]
    feat_target_count = _cd["feat_target_count"]
    target_totals = _cd["target_totals"]
    feature_totals = _cd["feature_totals"]
    n_targets = _cd["n_targets"]
    n_features = _cd["n_features"]

    _ba = ctx.get("_batch_archive", {})
    _target_metrics = _extract_target_metrics(_ba) if _ba else {}

    _imp_min = df_sankey["Importance"].min() if len(df_sankey) > 0 else 0
    _imp_max = df_sankey["Importance"].max() if len(df_sankey) > 0 else 1
    _norm = Normalize(vmin=_imp_min, vmax=_imp_max)

    _tc = {t: _parent_social_impcomp_resolve_color(t) for t in targets_list}
    _BG = _PARENT_SOCIAL_IMPCOMP_BG

    _pad = max(0.022, min(0.040, 1.2 / max(n_features, 1)))
    left_x = _LEFT_X
    right_x = _RIGHT_X
    node_w = 0.052
    _feat_nw = 0.028
    _target_node_w = 0.036

    _, _base_h = _wide_figure_dims(n_targets, n_features)
    fig_w = 15 * 1.5
    # Adaptive sizing (scales to n_targets so 5x3-year batches don't crowd)
    # Extra inches of figure height for tall batches (n_targets > 3).
    _tall_bonus = max(0, (n_targets - 3) * 0.55)
    fig_h = (max(_base_h * 1.3, n_features * 0.52, 22) + _tall_bonus) * 1.5
    # More vertical breathing room as n_targets grows.
    _spread_t = 2.2 + 0.10 * max(0, n_targets - 3)
    _spread_f = 1.9
    fig_w, fig_h, _spread_t, _spread_f = _landscape_adjust(fig_w, fig_h, _spread_t, _spread_f, n_targets, n_features)
    _lf = _landscape_fonts()
    # Target font: 17 at n_targets <= 3 (matches child sankey); shrinks
    # smoothly to ~10 at n_targets = 15. R2 sub-label remains readable.
    _target_fs = max(10, min(17, int(round(51 / max(n_targets, 3)))))

    global_max = max(target_totals.max(), feature_totals.max()) if len(target_totals) > 0 else 1.0
    _max_node_h = min(0.18, 3.5 / max(n_features, 1))

    target_positions = _compute_node_positions_enhanced(
        targets_list, target_totals, global_max, _pad, min_node_h=0.050, max_node_h=_max_node_h, spread_factor=_spread_t
    )
    feature_positions = _compute_node_positions_enhanced(
        features_list,
        feature_totals,
        global_max,
        _pad,
        min_node_h=0.025,
        max_node_h=_max_node_h,
        spread_factor=_spread_f,
    )
    target_positions, feature_positions = _equalize_vertical_envelopes(target_positions, feature_positions)

    # Adaptive rect height: 0.20 at n_targets <= 3 (matches child sankey);
    # shrinks for tall batches so the column doesn't visually overflow.
    _RECT_H = min(0.20, 0.85 / max(n_targets, 1))
    for t in targets_list:
        if t in target_positions:
            _c = target_positions[t]["y_centre"]
            target_positions[t] = {"y_centre": _c, "y_top": _c + _RECT_H / 2, "y_bot": _c - _RECT_H / 2, "h": _RECT_H}

    # Year-group dividers: extra spacing between year clusters
    # When the same target set repeats across multiple years (e.g. 5 targets
    # x 3 years = 15 rows), inject an extra vertical gap between year groups
    # so the eye can parse the structure. Only triggers for n_targets >= 6.
    if n_targets >= 6:
        import re as _re_yg

        def _year_of_label_yg(t):
            m = _re_yg.search(r"\(Year (\d+)\)$", str(t))
            return int(m.group(1)) if m else None

        _gap_extra = _RECT_H * 0.55  # 55% of rect height as group separator
        _shift = 0.0
        _prev_year = None
        for idx, t in enumerate(targets_list):
            yr = _year_of_label_yg(t)
            if t in target_positions:
                if idx > 0 and _prev_year is not None and yr is not None and yr != _prev_year:
                    _shift -= _gap_extra
                p = target_positions[t]
                target_positions[t] = {
                    "y_centre": p["y_centre"] + _shift,
                    "y_top": p["y_top"] + _shift,
                    "y_bot": p["y_bot"] + _shift,
                    "h": p["h"],
                }
                _prev_year = yr

    _FEAT_H = 0.056
    for f_key in features_list:
        if f_key in feature_positions:
            _c = feature_positions[f_key]["y_centre"]
            feature_positions[f_key] = {
                "y_centre": _c,
                "y_top": _c + _FEAT_H / 2,
                "y_bot": _c - _FEAT_H / 2,
                "h": _FEAT_H,
            }

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    all_y = [p["y_bot"] for p in target_positions.values()] + [p["y_bot"] for p in feature_positions.values()]
    y_min = min(all_y) - 0.04
    ax.set_xlim(-0.08, right_x + 0.22)
    ax.set_ylim(y_min, 1.06)
    ax.axis("off")

    _PORT_GAP = 0.003

    for rb in _iterate_ribbon_ports(
        df_sankey, targets_list, features_list, target_positions, feature_positions, target_totals, feature_totals
    ):
        _rgb = _tc.get(rb["target"], (0.5, 0.5, 0.5))
        rel = _norm(rb["importance"])

        _yl_t = rb["y_l_top"] - _PORT_GAP
        _yl_b = rb["y_l_bot"] + _PORT_GAP
        _yr_t = rb["y_r_top"] - _PORT_GAP * 0.5
        _yr_b = rb["y_r_bot"] + _PORT_GAP * 0.5
        if _yl_t <= _yl_b or _yr_t <= _yr_b:
            _yl_t, _yl_b = rb["y_l_top"], rb["y_l_bot"]
            _yr_t, _yr_b = rb["y_r_top"], rb["y_r_bot"]

        _desat = 0.55 + 0.45 * rel
        _white = (0.92, 0.93, 0.96)
        _fill = tuple(_white[i] + (_rgb[i] - _white[i]) * _desat for i in range(3))
        _alpha = 0.55 + 0.30 * rel

        if rel > 0.45:
            _glow_n = 2 if rel > 0.75 else 1
            _draw_bezier_ribbon_enhanced(
                ax,
                left_x,
                node_w,
                right_x,
                _yl_t + 0.005,
                _yl_b - 0.005,
                _yr_t + 0.003,
                _yr_b - 0.003,
                color_rgba=_rgb + (0.08,),
                zorder=1,
                glow_layers=_glow_n,
            )

        if rel > 0.70:
            _edge_dark = tuple(max(0, c * 0.70) for c in _rgb)
            _e_lw = 0.35 + 0.25 * (rel - 0.70) / 0.30
            _e_alpha = 0.22 + 0.18 * (rel - 0.70) / 0.30
            _edge_c = _edge_dark + (_e_alpha,)
        else:
            _edge_c = _fill + (min(_alpha + 0.05, 0.65),)
            _e_lw = 0.2

        _draw_bezier_ribbon_enhanced(
            ax,
            left_x,
            node_w,
            right_x,
            _yl_t,
            _yl_b,
            _yr_t,
            _yr_b,
            color_rgba=_fill + (_alpha,),
            zorder=2,
            edge_color=_edge_c,
            edge_lw=_e_lw,
            right_node_w=_feat_nw,
        )

    for t, pos in target_positions.items():
        rgb = _tc.get(t, (0.5, 0.5, 0.5))
        _edge_col = _parent_social_impcomp_edge_for(t)
        _bstyle = _parent_social_impcomp_boxstyle(t)

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w + 0.004, pos["y_bot"] - 0.003),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (left_x - _target_node_w, pos["y_bot"]),
                _target_node_w * 2,
                pos["h"],
                boxstyle=_bstyle,
                facecolor=rgb,
                edgecolor=_edge_col,
                linewidth=1.8,
                zorder=3,
            )
        )

        raw_key = (
            t if t in _PARENT_SOCIAL_IMPCOMP_DOMAIN_CLEAN_LABELS else _PARENT_SOCIAL_IMPCOMP_DISPLAY_ALIASES.get(t, t)
        )
        _domain_label = _PARENT_SOCIAL_IMPCOMP_DOMAIN_CLEAN_LABELS.get(raw_key, _get_tdn(t))

        _tp_form = _year_label_to_tp(t)  # reverse-map for metric-dict lookup
        _metric_val = (
            _target_metrics.get(t)
            or _target_metrics.get(_get_tdn(t))
            or _target_metrics.get(_tp_form)
            or _target_metrics.get(_get_tdn(_tp_form))
        )
        _label_x = left_x - _target_node_w - 0.04

        if _metric_val is not None:
            _v, _l = _metric_val
            txt = ax.text(
                _label_x,
                pos["y_centre"] + 0.025,
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
            mtxt = ax.text(
                _label_x,
                pos["y_centre"] - 0.028,
                f"{_l} = {_v:.3f}",
                ha="right",
                va="center",
                fontsize=_target_fs - 3,
                fontweight="bold",
                color=_edge_col,
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            mtxt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])
        else:
            txt = ax.text(
                _label_x,
                pos["y_centre"],
                _domain_label,
                ha="right",
                va="center",
                fontsize=_target_fs,
                fontweight="bold",
                color="#1a1a1a",
                fontfamily=_TITLE_FONT,
                zorder=4,
                clip_on=False,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    _fs_feat = max(8, min(_lf[1], 400 / max(n_features, 1)))
    for _fi, (f, pos) in enumerate(feature_positions.items()):
        n_tgt = feat_target_count.get(f, 1)
        _fc = _parent_social_impcomp_feature_color(f, df_sankey, feat_target_count, targets_list)

        _f_rows = df_sankey[df_sankey["Feature"] == f]
        _domain_counts = {}
        for _ft in _f_rows["Target"]:
            _dom_key = _parent_social_impcomp_domain(_ft)
            _domain_counts[_dom_key] = _domain_counts.get(_dom_key, 0) + 1
        _dom_winner = max(_domain_counts, key=_domain_counts.get) if _domain_counts else "avoidant"
        _feat_edge = _PARENT_SOCIAL_IMPCOMP_FEATURE_EDGE_MAP.get(_dom_winner, "#1A1A1A")

        if n_tgt >= 2:
            dom_rows = df_sankey[df_sankey["Feature"] == f]
            dom_t = dom_rows.loc[dom_rows["Importance"].idxmax(), "Target"] if not dom_rows.empty else targets_list[0]
            glow_col = _tc.get(dom_t, (0.5, 0.5, 0.5))
            ax.add_patch(
                FancyBboxPatch(
                    (right_x - _feat_nw - 0.006, pos["y_bot"] - 0.006),
                    _feat_nw * 2 + 0.012,
                    pos["h"] + 0.012,
                    boxstyle="round,pad=0.008",
                    facecolor="none",
                    edgecolor=glow_col + (0.38,),
                    linewidth=1.3,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw + 0.004, pos["y_bot"] - 0.003),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                zorder=2,
                alpha=0.14,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (right_x - _feat_nw, pos["y_bot"]),
                _feat_nw * 2,
                pos["h"],
                boxstyle="round,pad=0.010",
                facecolor=_fc,
                edgecolor=_feat_edge,
                linewidth=0.7,
                zorder=3,
            )
        )

        _disp_name = DISPLAY_NAMES.get(f, f) if "DISPLAY_NAMES" in dir() else f
        badge = f" ({n_tgt}T)" if n_tgt >= 2 else ""
        fw = "bold" if n_tgt >= 2 else "normal"
        txt = ax.text(
            right_x + _feat_nw + 0.025,
            pos["y_centre"],
            f"{_disp_name}{badge}",
            ha="left",
            va="center",
            fontsize=_fs_feat,
            fontweight=fw,
            color="#1a1a1a",
            fontfamily=_TITLE_FONT,
            zorder=4,
            clip_on=False,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=_BG)])

    _hdr_y = (
        max(
            max(p["y_top"] for p in target_positions.values()),
            max(p["y_top"] for p in feature_positions.values()),
        )
        + 0.035
    )
    _hdr_tgt_col = "#0A1F4D"
    _hdr_feat_col = "#1E293B"
    ax.text(
        left_x,
        _hdr_y,
        "TARGETS (PARENT: AVOIDANT / SOCIAL / IMPULSIVITY / COMPULSIVITY / OBSESSIONS)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_tgt_col,
        fontfamily=_TITLE_FONT,
    )
    ax.text(
        right_x,
        _hdr_y,
        "FEATURES",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=_hdr_feat_col,
        fontfamily=_TITLE_FONT,
    )
    _rule_hw = 0.10
    ax.plot(
        [left_x - _rule_hw, left_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_tgt_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )
    ax.plot(
        [right_x - _rule_hw, right_x + _rule_hw],
        [_hdr_y - 0.004, _hdr_y - 0.004],
        color=_hdr_feat_col,
        lw=0.5,
        alpha=0.35,
        zorder=5,
    )

    ax.text(0.50, 1.085, " ", ha="center", va="bottom", fontsize=17, transform=ax.transAxes, alpha=0.0)

    _sankey_print_summary(df_sankey, feat_target_count, n_targets, n_features)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "manuscript_figures")
    saved = _safe_save_fig(fig, save_dir, "sankey_parent_social_impcomp", dpi=1200, formats=("png", "pdf"))
    plt.show()
    for p in saved:
        logger.info(f"  Saved -> {p}")
    return fig


# Sankey style-specific exclusions
# Visualization-only exclusions: targets listed here are kept out of the
# Sankey figure for the given style, but are NOT removed from _batch_archive
# and continue to be part of the main analyses. Add both the raw key and the
# display name form (what appears in df_sankey['Target']) to catch either.
# Edit this dict to customize which targets get hidden per style.
_SANKEY_EXCLUDE_TARGETS_BY_STYLE = {
    "child_deltas": {
        # Remove delta_family_conflict_ss_k from the Child Only Deltas Sankey
        "delta_family_conflict_ss_k",
        "Change in Fam. Conflict (Y)",
    },
    "child_academic_cognitive": {
        # Remove delta_bad_grades (all TPs) from the Academic/Cognitive Sankey.
        # ML run still trains/evaluates these: they simply don't render here.
        # Variants are listed because exclusion matches df_sankey['Target'] by
        # exact string, and _get_tdn appends '(TN)' for TP-sweep archive keys.
        "delta_bad_grades",
        "delta_bad_grades (T0)",
        "delta_bad_grades (T1)",
        "delta_bad_grades (T2)",
        "delta_bad_grades (T3)",
        "delta_bad_grades (T4)",
    },
    # Add other style -> set-of-targets entries here as needed.
}


def run() -> None:
    """Run this stage (notebook cell 8)."""

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    try:
        _RS["section"] = "enhanced_sankey"
    except Exception:
        pass

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.15,
            "axes.linewidth": 0,
        }
    )

    _TDN = ctx.get("TARGET_DISPLAY_NAMES", {})  # legacy alias: prefer _get_tdn()

    # Extend _parent_dev_sweep_domain for the 'unliked' domain. The alias
    # _orig_parent_dev_sweep_domain_v2 mirrors the original base function so
    # the relationships override chains correctly; _v3 then captures the
    # version with the relationships domain, and the function defined below
    # adds 'unliked' on top.
    if "_orig_parent_dev_sweep_domain_v2" not in ctx:
        ctx._orig_parent_dev_sweep_domain_v2 = _orig_parent_dev_sweep_domain

    # Execute sankey

    _has_batch = "_batch_archive" in ctx and isinstance(ctx._batch_archive, dict) and len(ctx._batch_archive) >= 1

    if not _has_batch:
        logger.info("Warning: No _batch_archive found. Run the main stage in batch mode first.")
    else:
        logger.info(f"_batch_archive: {len(ctx._batch_archive)} entries")
        for _k in ctx._batch_archive:
            logger.info(f"   \u2022 {_k}")

        if ctx.show_sankey:
            logger.info("\n" + "=" * 80)
            logger.info("  SANKEY: Transdiagnostic Feature Flow")
            logger.info(f"     Style: {ctx.sankey_style}")
            if ctx.sankey_landscape:
                logger.info("     Mode:  LANDSCAPE")
            logger.info("=" * 80)

            _sankey_rows = _extract_sankey_data(ctx._batch_archive, top_n=ctx.sankey_top_features)

            if not _sankey_rows:
                logger.info("  Warning: No consensus_df data found in _batch_archive.")
                logger.info("     Set main.visualization_mode to a mode that includes the consensus.")
            else:
                df_sankey = pd.DataFrame(_sankey_rows)

                # Apply style-specific target exclusions (visualization-only)
                _excludes = _SANKEY_EXCLUDE_TARGETS_BY_STYLE.get(ctx.sankey_style, set())
                if _excludes:
                    _present = set(df_sankey["Target"].unique()) & _excludes
                    if _present:
                        _before_n = len(df_sankey)
                        df_sankey = df_sankey[~df_sankey["Target"].isin(_excludes)].reset_index(drop=True)
                        _dropped = _before_n - len(df_sankey)
                        logger.info(
                            f"  Note: Style '{ctx.sankey_style}' excludes {sorted(_present)} "
                            f"({_dropped} feature rows filtered; targets remain in _batch_archive)"
                        )

                # Diagnostic: verify per-target feature counts
                _per_target = df_sankey.groupby("Target")["Feature"].count()
                logger.info(f"  sankey_top_features = {ctx.sankey_top_features}")
                logger.info(f"  Total unique features on figure: {df_sankey['Feature'].nunique()}")
                for _tgt, _cnt in _per_target.items():
                    _flag = "  WARN" if _cnt != ctx.sankey_top_features else "  ok"
                    logger.info(f"  {_flag} {_tgt}: {_cnt} features")

                if ctx.sankey_style == "run_all_styles":
                    _run_all_styles(
                        df_sankey, ctx.sankey_top_features, save_dir=os.path.join(os.getcwd(), "manuscript_figures")
                    )
                else:
                    _draw_sankey(
                        df_sankey,
                        ctx.sankey_top_features,
                        save_dir=os.path.join(os.getcwd(), "manuscript_figures"),
                        style=ctx.sankey_style,
                    )
                    logger.info(f"  Sankey rendered with style: '{ctx.sankey_style}'")
        else:
            logger.info("Sankey display disabled (show_sankey = False).")
