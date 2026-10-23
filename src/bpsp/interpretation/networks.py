"""Networks of the top predictive features (the `networks` stage).

EBICglasso partial correlation networks, Pearson correlation matrices and
hierarchical clustering dendrograms of the top ML-predictive features for
the targets of the current run. A Fiedler eigenvector layout positions
nodes to reflect network community structure. Optional bootstrap CIs (edge
stability) and case-dropping centrality stability, both off by default
because they are slow.
"""

import logging

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, PathPatch, Wedge
from matplotlib.path import Path
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from bpsp.context import ctx
from bpsp.data.preprocessing import DISPLAY_NAMES, TARGET_DISPLAY_NAMES
from bpsp.reporting import _RS, cache_load

logger = logging.getLogger(__name__)
# graphical_lasso is called directly so the penalty can be chosen by EBIC (EBICglasso)
import warnings

import networkx as _nx_ver
import sklearn
from IPython.display import HTML, display
from sklearn.covariance import graphical_lasso
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS

# 0. Style constants

_CLR_SLATE = "#1F2D3D"
_CLR_CHARCOAL = "#34495E"
_CLR_STEEL = "#546E7A"
_CLR_SILVER = "#BDC3C7"
_CLR_CLOUD = "#EEF2F7"

_TXT_SECONDARY = "#4A5568"
_TXT_MUTED = "#607D8B"
_TXT_HEADER = "#546E7A"
_ARROW_CLR = "#78909C"

_EDGE_POS = "#0072B2"
# Okabe-Ito vermillion
_EDGE_NEG = "#D55E00"

# module-level dash constant: was (0,(5,3)) in draw, (0,(4,2)) in legend
_DASH_NEG = (0, (5, 3))

# border colors swapped for directional consistency with edges
# Positive EI → blue family (matches _EDGE_POS blue = positive partial r)
# Negative EI → warm/orange family
_BORDER_POS = "#2E86AB"
_BORDER_NEG = "#E67E22"
_BORDER_ZERO = "#BDC3C7"

_TARGET_CLR = "#C0392B"
_RING_UNFILLED = "#E0E0E0"

# _EDGE_MAXIMUM sets the edge-width scale of magnitude-based rendering
_EDGE_MAXIMUM = 1.0

_SHADOW_ALPHA = 0.09
_SHADOW_SCALE = 1.04
_CURVE_STRENGTH = 0.12  # kept as fallback; _draw_fancy_edge uses adaptive version

# Network panel background
_NET_BG = "#F8F7F3"
_GLOW_SPREAD_1 = 5.5
_GLOW_SPREAD_2 = 2.8
_GLOW_ALPHA_1 = 0.09
_GLOW_ALPHA_2 = 0.22
# reduced from 0.20 for better edge contrast on warm background
_EDGE_TINT_F = 0.12

# magnitude-based edge width and opacity range
# Wider range than the old 2.0-2.6 rank-based band: Epskamp/qgraph default ≈ 0.5-5pt
_LW_MIN = 0.6
_LW_MAX = 4.0
_ALPHA_MIN = 0.12
_ALPHA_MAX = 0.85

# EBICglasso hyperparameter (Epskamp, Borsboom & Fried 2018)
# γ = 1.0 is the conservative alternative to the 0.5 default; produces sparser
# networks where retained edges are more robust. Well-cited in Foygel & Drton
# (2010) and routinely used by Fried's group for robustness checks.
_EBIC_GAMMA = 1.0
_EBIC_N_ALPHA = 100  # lambda grid size for main analysis
_EBIC_N_ALPHA_BOOT = 50  # reduced grid for bootstrap (speed)

# Visualization-only display threshold (qgraph `minimum` equivalent).
# Edges with |partial r| below this are omitted from the figure for visual
# clarity; the full EBICglasso adjacency matrix is preserved in the pcor_df
# and available in supplement. At n ≈ 1000, |pcor| < 0.10 has very modest
# practical effect size.
_DISPLAY_THRESH = 0.07

_DOMAIN_COLORS = {
    "Youth Behavioral": "#4A90D9",
    "Parent Report": "#E67E22",
    "Family/Social": "#27AE60",
    "Cognitive/Academic": "#8E44AD",
    "Neuroimaging": "#E74C3C",
    "Physical/Medical": "#16A085",
    "Demographic/Other": "#7F8C8D",
    "Change Scores": "#2C3E50",
    "Sleep/Screen": "#F39C12",
    "SES/Environment": "#1ABC9C",
}
_DOMAIN_FALLBACK = "#7F8C8D"

_CAT_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
    "#44AA99",
    "#882255",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#88CCEE",
    "#661100",
    "#6699CC",
    "#BBBBBB",
    "#275D8E",
    "#AE5F3C",
]

_CORR_CMAP = LinearSegmentedColormap.from_list(
    "bwr_accessible",
    ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"],
    N=256,
)

# 0B. Helper functions

# EBICglasso: graphical lasso with the penalty selected by EBIC


def _ebic_glasso(X_scaled, gamma=_EBIC_GAMMA, n_alpha=_EBIC_N_ALPHA, alpha_min_ratio=0.01, max_iter=1000):
    """
    EBICglasso: select GLASSO regularization parameter by minimizing the
    Extended Bayesian Information Criterion (Foygel & Drton 2010).

    This is the standard estimation method in psychological network analysis
    (Epskamp & Fried 2018; Burger et al. 2022).

    Parameters
    X_scaled : ndarray (n, p), standardized data
    gamma    : float, EBIC hyperparameter (0 = BIC, 0.5 = default, 1 = conservative)
    n_alpha  : int, number of lambda values in the grid
    alpha_min_ratio : float, ratio of min to max lambda
    max_iter : int, GLASSO iterations per lambda

    Returns
    best_prec  : ndarray (p, p) or None
    best_cov   : ndarray (p, p) or None
    best_alpha : float or None
    gamma      : float, the γ used (for reporting)
    """
    n, p = X_scaled.shape
    S = np.cov(X_scaled, rowvar=False, bias=False)  # sample covariance

    # Build lambda grid: from max (fully sparse) to min (dense)
    alpha_max = np.max(np.abs(S - np.diag(np.diag(S))))
    if alpha_max < 1e-10:
        alpha_max = 1.0
    alpha_min = alpha_max * alpha_min_ratio
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alpha)

    best_ebic = np.inf
    best_prec = None
    best_cov = None
    best_alpha = None

    for alpha in alphas:
        try:
            cov_, prec_ = graphical_lasso(S, alpha=alpha, max_iter=max_iter, mode="cd")
            # Log-likelihood (Gaussian)
            sign, logdet = np.linalg.slogdet(prec_)
            if sign <= 0:
                continue
            ll = logdet - np.trace(S @ prec_)

            # Number of non-zero off-diagonal unique edges
            upper = np.abs(prec_[np.triu_indices(p, k=1)])
            E = int(np.sum(upper > 1e-10))

            # EBIC (Foygel & Drton 2010)
            ebic = -n * ll + E * np.log(n) + 4.0 * gamma * E * np.log(p)

            if ebic < best_ebic:
                best_ebic = ebic
                best_prec = prec_
                best_cov = cov_
                best_alpha = alpha
        except Exception:
            continue

    return best_prec, best_cov, best_alpha, gamma


def _draw_bezier_edge(ax, p1, p2, color, lw, alpha, linestyle="-", edge_idx=0):
    """Original flat-color Bézier edge: kept for bootstrap CI figure."""
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max(np.sqrt(dx**2 + dy**2), 1e-6)
    sign = 1 if edge_idx % 2 == 0 else -1
    curve_offset = _CURVE_STRENGTH * length * sign
    ctrl_x = mx - (dy / length) * curve_offset
    ctrl_y = my + (dx / length) * curve_offset
    verts = [p1, (ctrl_x, ctrl_y), p2]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    if linestyle == "-":
        patch = PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round", zorder=1)
    else:
        patch = PathPatch(
            path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round", linestyle=linestyle, zorder=1
        )
    ax.add_patch(patch)
    return (ctrl_x, ctrl_y)


def _adaptive_curve_strength(p1, p2, data_span=12.0, min_cs=0.06, max_cs=0.22):
    """
    Returns a curve_strength value that scales with chord length relative to
    the overall layout data span.
      Short edges → low curvature  (clean, unobtrusive, avoids tight loops)
      Long edges  → higher curvature (visually separates arc from chord,
                                      reduces overlap with other edges)
    """
    dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    norm = np.clip(dist / (data_span + 1e-9), 0.0, 1.0)
    return min_cs + (max_cs - min_cs) * norm


def _draw_fancy_edge(
    ax,
    p1,
    p2,
    color1,
    color2,
    base_color,
    lw,
    base_alpha,
    linestyle="-",
    edge_idx=0,
    is_strong=False,
    data_span=12.0,
    curve_sign=None,
):
    """
    Tinted + selective-glow Bézier edge for the warm off-white network panel.
    Uses _adaptive_curve_strength() for per-edge curvature.

    curve_sign overrides edge_idx-based alternation when provided.
    For circle layouts, caller passes geometry-based sign so edges always curve
    outward from the layout center.
    """
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max(np.sqrt(dx**2 + dy**2), 1e-6)
    if curve_sign is not None:
        sign = curve_sign
    else:
        sign = 1 if edge_idx % 2 == 0 else -1

    cs_val = _adaptive_curve_strength(p1, p2, data_span=data_span)
    ctrl_x = mx - (dy / length) * cs_val * length * sign
    ctrl_y = my + (dx / length) * cs_val * length * sign

    c_base = np.array(matplotlib.colors.to_rgba(base_color)[:3])
    c_n1 = np.array(matplotlib.colors.to_rgba(color1)[:3])
    c_n2 = np.array(matplotlib.colors.to_rgba(color2)[:3])
    c_node = (c_n1 + c_n2) / 2.0
    c_tint = np.clip((1.0 - _EDGE_TINT_F) * c_base + _EDGE_TINT_F * c_node, 0.0, 1.0)

    def _bpath():
        return Path([(p1[0], p1[1]), (ctrl_x, ctrl_y), (p2[0], p2[1])], [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    if is_strong:
        _glow_layers = [
            (7.0, 0.04),
            (5.0, 0.07),
            (3.5, 0.11),
            (2.2, 0.16),
            (1.4, 0.22),
        ]
        for _glw_mult, _galp in _glow_layers:
            ax.add_patch(
                PathPatch(
                    _bpath(),
                    facecolor="none",
                    edgecolor=tuple(c_tint),
                    lw=lw * _glw_mult,
                    alpha=_galp,
                    capstyle="round",
                    zorder=1,
                )
            )

    ax.add_patch(
        PathPatch(
            _bpath(),
            facecolor="none",
            edgecolor=tuple(c_tint),
            lw=lw,
            alpha=base_alpha,
            capstyle="round",
            linestyle=linestyle,
            zorder=2,
        )
    )

    return (ctrl_x, ctrl_y)


# bootstrap networks are estimated with _ebic_glasso
def _bootstrap_edge_weights(X_raw, feat_names, n_boot=1000, seed=42, ci_level=0.95, gamma=_EBIC_GAMMA):
    rng = np.random.RandomState(seed)
    n, p = X_raw.shape
    edge_samples = np.full((n_boot, p, p), np.nan)
    n_converged = 0
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Xb = StandardScaler().fit_transform(X_raw[idx])
        try:
            prec, _, _, _ = _ebic_glasso(Xb, gamma=gamma, n_alpha=_EBIC_N_ALPHA_BOOT, max_iter=1000)
            if prec is None:
                continue
            d = np.sqrt(np.diag(prec))
            pcor = -prec / np.outer(d, d)
            np.fill_diagonal(pcor, 1.0)
            edge_samples[b] = pcor
            n_converged += 1
        except Exception:
            pass  # non-critical; logged upstream or optional visualization
    alpha = 1.0 - ci_level
    ci_lo = np.nanpercentile(edge_samples, 100 * alpha / 2, axis=0)
    ci_hi = np.nanpercentile(edge_samples, 100 * (1 - alpha / 2), axis=0)
    median = np.nanmedian(edge_samples, axis=0)
    # inclusion = proportion of bootstrap samples where
    # EBICglasso retains the edge (i.e. |pcor| > 1e-6 after regularization).
    # With EBICglasso (γ=0.5), zero edges are truly zero from regularization,
    # so this threshold captures the meaningful sparsity pattern.
    inclusion = np.nanmean(np.abs(edge_samples) > 1e-6, axis=0)
    return {
        "edge_samples": edge_samples,
        "ci_lower": pd.DataFrame(ci_lo, index=feat_names, columns=feat_names),
        "ci_upper": pd.DataFrame(ci_hi, index=feat_names, columns=feat_names),
        "inclusion_prob": pd.DataFrame(inclusion, index=feat_names, columns=feat_names),
        "median": pd.DataFrame(median, index=feat_names, columns=feat_names),
        "n_converged": n_converged,
    }


# centrality stability is estimated with _ebic_glasso
# centrality funcs use thresholded graph
def _centrality_stability(
    X_raw, feat_names, centrality_func, n_boot=1000, seed=42, cor_threshold=0.7, gamma=_EBIC_GAMMA
):
    from scipy.stats import spearmanr

    rng = np.random.RandomState(seed)
    n, p = X_raw.shape
    try:
        X_full_sc = StandardScaler().fit_transform(X_raw)
        prec, _, _, _ = _ebic_glasso(X_full_sc, gamma=gamma, n_alpha=_EBIC_N_ALPHA, max_iter=1000)
        if prec is None:
            return {"CS": np.nan, "proportions": [], "avg_cors": [], "sd_cors": []}
        d = np.sqrt(np.diag(prec))
        pcor = -prec / np.outer(d, d)
        np.fill_diagonal(pcor, 1.0)
        pcor_full = pd.DataFrame(pcor, index=feat_names, columns=feat_names)
    except Exception:
        return {"CS": np.nan, "proportions": [], "avg_cors": [], "sd_cors": []}
    cent_full = centrality_func(pcor_full, feat_names)
    cent_full_arr = np.array([cent_full[f] for f in feat_names])
    proportions = np.arange(0.10, 0.80, 0.05)
    avg_cors = []
    sd_cors = []
    _bpp = min(n_boot, 200)
    for prop in proportions:
        subset_n = max(int(n * (1 - prop)), p + 2)
        if subset_n <= p + 1:
            avg_cors.append(np.nan)
            sd_cors.append(np.nan)
            continue
        cors = []
        for _ in range(_bpp):
            idx = rng.choice(n, size=subset_n, replace=False)
            Xsub = StandardScaler().fit_transform(X_raw[idx])
            try:
                prec_s, _, _, _ = _ebic_glasso(Xsub, gamma=gamma, n_alpha=_EBIC_N_ALPHA_BOOT, max_iter=1000)
                if prec_s is None:
                    continue
                d_s = np.sqrt(np.diag(prec_s))
                pcor_s = -prec_s / np.outer(d_s, d_s)
                np.fill_diagonal(pcor_s, 1.0)
                pcor_sub = pd.DataFrame(pcor_s, index=feat_names, columns=feat_names)
                cent_sub = centrality_func(pcor_sub, feat_names)
                cent_sub_arr = np.array([cent_sub[f] for f in feat_names])
                r, _ = spearmanr(cent_full_arr, cent_sub_arr)
                cors.append(r)
            except Exception:
                pass  # non-critical; logged upstream or optional visualization
        avg_cors.append(np.mean(cors) if cors else np.nan)
        sd_cors.append(np.std(cors) if len(cors) > 1 else np.nan)
    cs_val = 0.0
    for prop, avg_r in zip(proportions, avg_cors):
        if not np.isnan(avg_r) and avg_r >= cor_threshold:
            cs_val = prop
    return {"CS": cs_val, "proportions": list(proportions), "avg_cors": avg_cors, "sd_cors": sd_cors}


# EI and Strength computed from thresholded graph G
# (Robinaugh et al. 2016 definition: sum of edge weights in estimated network)


def _compute_EI_from_graph(G, feat_names):
    """Expected Influence from thresholded network G (Robinaugh et al. 2016).
    Sums signed edge weights of retained (non-zero) edges only."""
    import networkx as nx

    return {f: float(sum(G[f][nb]["weight"] for nb in G.neighbors(f))) if f in G else 0.0 for f in feat_names}


def _compute_EI_thresholded(pcor_df, feat_names, thresh=1e-6):
    """Build a temporary thresholded graph from pcor_df, then compute EI.
    Used by centrality stability where the full graph object is not available."""
    import networkx as nx

    G_tmp = nx.Graph()
    for f in feat_names:
        G_tmp.add_node(f)
    n = len(feat_names)
    for i in range(n):
        for j in range(i + 1, n):
            pc = pcor_df.iloc[i, j]
            if abs(pc) > thresh:
                G_tmp.add_edge(feat_names[i], feat_names[j], weight=pc)
    return _compute_EI_from_graph(G_tmp, feat_names)


def _compute_strength_from_graph(G, feat_names):
    """Node strength from thresholded network G (sum of |edge weights|)."""
    import networkx as nx

    return {f: float(sum(abs(G[f][nb]["weight"]) for nb in G.neighbors(f))) if f in G else 0.0 for f in feat_names}


def _compute_strength_thresholded(pcor_df, feat_names, thresh=1e-6):
    """Build a temporary thresholded graph from pcor_df, then compute strength."""
    import networkx as nx

    G_tmp = nx.Graph()
    for f in feat_names:
        G_tmp.add_node(f)
    n = len(feat_names)
    for i in range(n):
        for j in range(i + 1, n):
            pc = pcor_df.iloc[i, j]
            if abs(pc) > thresh:
                G_tmp.add_edge(feat_names[i], feat_names[j], weight=pc)
    return _compute_strength_from_graph(G_tmp, feat_names)


def _fiedler_circle_layout(G, scale=6.5):
    """
    Place nodes on a circle ordered by the Fiedler vector (second eigenvector
    of the normalized graph Laplacian).  Structurally similar nodes end up
    adjacent, reducing edge crossings and exposing community structure.

    Falls back to plain circular layout if the graph has < 3 nodes or if
    the eigensolver fails.

    A lightweight 2-opt pass is then run to further reduce edge crossings.
    """
    import networkx as nx
    import numpy as np

    nodes = list(G.nodes())
    n = len(nodes)
    if n < 3:
        return nx.circular_layout(G, scale=scale)

    try:
        idx = {nd: i for i, nd in enumerate(nodes)}
        A = np.zeros((n, n))
        for u, v, d in G.edges(data=True):
            w = abs(d.get("weight", 1.0))
            A[idx[u], idx[v]] = w
            A[idx[v], idx[u]] = w

        deg = A.sum(axis=1)
        deg_safe = np.where(deg > 0, deg, 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
        L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        eigvals, eigvecs = np.linalg.eigh(L_norm)
        fiedler = eigvecs[:, 1]

        order = np.argsort(fiedler)
        ordered_nodes = [nodes[i] for i in order]

        def _count_crossings(seq):
            pos_map = {nd: i for i, nd in enumerate(seq)}
            edges = list(G.edges())
            crossings = 0
            for k in range(len(edges)):
                a, b = pos_map[edges[k][0]], pos_map[edges[k][1]]
                if a > b:
                    a, b = b, a
                for l in range(k + 1, len(edges)):
                    c, d = pos_map[edges[l][0]], pos_map[edges[l][1]]
                    if c > d:
                        c, d = d, c
                    if a < c < b < d or c < a < d < b:
                        crossings += 1
            return crossings

        improved = True
        best_seq = ordered_nodes[:]
        best_cross = _count_crossings(best_seq)
        iterations = 0
        while improved and iterations < 60:
            improved = False
            iterations += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = best_seq[:i] + best_seq[i : j + 1][::-1] + best_seq[j + 1 :]
                    c = _count_crossings(candidate)
                    if c < best_cross:
                        best_seq, best_cross = candidate, c
                        improved = True

        ordered_nodes = best_seq

        ei_vals = nx.get_node_attributes(G, "ei") if nx.get_node_attributes(G, "ei") else {}
        if ei_vals:
            top_ei_node = max(ei_vals, key=lambda nd: ei_vals[nd])
        else:
            top_ei_node = best_seq[0]
        if top_ei_node in best_seq:
            rot = best_seq.index(top_ei_node)
            best_seq = best_seq[rot:] + best_seq[:rot]
        ordered_nodes = best_seq

        angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
        pos = {nd: (scale * np.cos(angles[i]), scale * np.sin(angles[i])) for i, nd in enumerate(ordered_nodes)}
        return pos

    except Exception:
        return nx.circular_layout(G, scale=scale)


# precision-implied R² (Haslbeck & Waldorp 2018 canonical method)
# Uses **sample** covariance (not regularised) when prec/cov provided.
def _compute_predictability(feat_list, prec=None, cov=None, cross_df=None, pcor_df=None, thresh=0.05):
    """
    Node predictability (R²).

    If prec and cov matrices are provided, uses precision-implied coefficients:
        R²_j = 1 − 1 / (Θ_jj · Σ_jj)
    (Haslbeck & Waldorp 2018, as implemented in mgm::predict / EGAnet).

    cov should be the **sample** covariance of the standardized
    data, not the regularised covariance from GLASSO, to match the Haslbeck &
    Waldorp derivation.

    Otherwise, falls back to OLS regression on network neighbors.
    """
    r2 = {}
    if prec is not None and cov is not None:
        for i, f in enumerate(feat_list):
            r2[f] = max(0.0, 1.0 - 1.0 / (prec[i, i] * cov[i, i] + 1e-12))
        return r2
    # Fallback: OLS (used by consensus network where no single precision matrix exists)
    if cross_df is None or pcor_df is None:
        return {f: 0.0 for f in feat_list}
    for f in feat_list:
        neighbors = [o for o in feat_list if o != f and abs(pcor_df.loc[f, o]) > thresh]
        if len(neighbors) == 0:
            r2[f] = 0.0
            continue
        subset = cross_df[neighbors + [f]].dropna()
        if len(subset) < 10:
            r2[f] = 0.0
            continue
        reg = LinearRegression()
        reg.fit(subset[neighbors].values, subset[f].values)
        r2[f] = max(0.0, reg.score(subset[neighbors].values, subset[f].values))
    return r2


def _network_stats(G):
    nn = G.number_of_nodes()
    ne = G.number_of_edges()
    mx = nn * (nn - 1) / 2
    density = ne / mx if mx > 0 else 0.0
    if ne > 0:
        ws = [abs(G.edges[e]["weight"]) for e in G.edges()]
        return {
            "n_nodes": nn,
            "n_edges": ne,
            "density": density,
            "avg_abs_weight": np.mean(ws),
            "median_abs_weight": np.median(ws),
            "max_abs_weight": np.max(ws),
        }
    return {
        "n_nodes": nn,
        "n_edges": ne,
        "density": density,
        "avg_abs_weight": 0,
        "median_abs_weight": 0,
        "max_abs_weight": 0,
    }


# MDS stress fallback threshold lowered 0.15 → 0.10
# (Kruskal: ≤0.10 "fair", ≤0.05 "good"; 0.15 was too lax)
def _mds_layout(G, pcor_df, feat_list, scale=5.0, seed=42):
    feat_in_g = [nd for nd in feat_list if nd in G.nodes()]
    sub = pcor_df.loc[feat_in_g, feat_in_g].values
    dist = 1 - np.abs(sub)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, normalized_stress="auto", max_iter=500)
    coords = mds.fit_transform(dist)
    stress = mds.stress_
    if stress > 0.10:
        logger.info(f"  Warning:  MDS stress = {stress:.4f} > 0.10 -- falling back to Fruchterman-Reingold")
        import networkx as nx

        pos = nx.spring_layout(G, weight="aw", k=2.5 / len(G) ** 0.5, iterations=200, seed=seed)
        pa = np.array([list(pos[nd]) for nd in feat_in_g]) * scale
        pos = {feat_in_g[i]: (pa[i, 0], pa[i, 1]) for i in range(len(feat_in_g))}
        for nd in G.nodes():
            if nd not in pos:
                pos[nd] = nx.spring_layout(G, weight="aw", seed=seed)[nd]
                pos[nd] = (pos[nd][0] * scale, pos[nd][1] * scale)
        return pos, f"Fruchterman-Reingold (MDS stress too high: {stress:.3f})", stress
    coords = coords / (np.abs(coords).max() + 1e-9) * scale
    pos = {feat_in_g[i]: (coords[i, 0], coords[i, 1]) for i in range(len(feat_in_g))}
    for nd in G.nodes():
        if nd not in pos:
            import networkx as nx

            extra = nx.spring_layout(G, weight="aw", seed=seed)
            pos[nd] = (extra[nd][0] * scale, extra[nd][1] * scale)
    return pos, f"MDS (stress-1 = {stress:.4f})", stress


def _draw_predictability_node(ax, x, y, radius, r2, node_color, border_color, border_width):
    """
    Predictability-ring node for warm off-white background.
    Drop shadow provides depth. Ring fill = R² in darkened node color.

    Inner circle radius 0.07→0.30 so donut hole is visible.
    Node z-orders bumped (shadow→3, bg→4, fill→5, inner→6) so
      edges (zorder 1-2) never bleed over nodes.
    Shadow offset is radius-relative, not hardcoded data coords.
    """
    _shadow_off = radius * 0.25
    shadow = Circle(
        (x + _shadow_off, y - _shadow_off),
        radius * _SHADOW_SCALE,
        facecolor="#000000",
        alpha=_SHADOW_ALPHA,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(shadow)
    bg = Circle((x, y), radius, facecolor=_RING_UNFILLED, edgecolor=border_color, linewidth=border_width, zorder=4)
    ax.add_patch(bg)
    if r2 > 0.01:
        _nc_rgb = np.array(matplotlib.colors.to_rgb(node_color))
        _nc_dark = tuple(np.clip(_nc_rgb * 0.65, 0, 1))
        theta_start = 90
        theta_end = 90 - (r2 * 360)
        fill = Wedge((x, y), radius, theta_end, theta_start, facecolor=_nc_dark, edgecolor="none", alpha=1.0, zorder=5)
        ax.add_patch(fill)
    inner = Circle((x, y), radius * 0.30, facecolor="white", edgecolor="none", alpha=1.0, zorder=6)
    ax.add_patch(inner)
    # tick mark at 12 o'clock: R² reference line
    # (Haslbeck & Waldorp 2018 convention; makes ring fill readable)
    _tick_inner = radius * 0.28
    _tick_outer = radius * 1.02
    ax.plot(
        [x, x],
        [y + _tick_inner, y + _tick_outer],
        color=border_color,
        lw=1.5,
        solid_capstyle="round",
        alpha=0.7,
        zorder=6,
        clip_on=False,
    )


def _get_domain_colors(feat_list, cat_of_feat, is_within):
    _f2c = None
    try:
        if "FEATURE_TO_CATEGORY" in ctx and isinstance(FEATURE_TO_CATEGORY, dict):  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
            _f2c = FEATURE_TO_CATEGORY  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
    except Exception:
        pass  # non-critical; logged upstream or optional visualization
    if _f2c and len(_f2c) > 0:
        domains_used = {}
        colors = {}
        for f in feat_list:
            domain = _f2c.get(f, "Demographic/Other")
            domains_used[domain] = True
            colors[f] = _DOMAIN_COLORS.get(domain, _DOMAIN_FALLBACK)
        handles = [
            mpatches.Patch(fc=_DOMAIN_COLORS.get(d, _DOMAIN_FALLBACK), ec="none", alpha=0.85, label=d)
            for d in sorted(domains_used.keys())
        ]
        return colors, handles
    if is_within:
        unique_cats = list(dict.fromkeys(cat_of_feat[f] for f in feat_list))
        cat_cmap = {c: _CAT_COLORS[i % len(_CAT_COLORS)] for i, c in enumerate(unique_cats)}
        colors = {f: cat_cmap[cat_of_feat[f]] for f in feat_list}
        handles = [mpatches.Patch(fc=cat_cmap[c], ec="none", alpha=0.85, label=c[:25]) for c in unique_cats]
        return colors, handles
    colors = {f: _CAT_COLORS[i % len(_CAT_COLORS)] for i, f in enumerate(feat_list)}
    return colors, None


def _save_fig(fig, basename, facecolor="white"):
    fig.savefig(f"{basename}.png", dpi=300, bbox_inches="tight", facecolor=facecolor)
    fig.savefig(f"{basename}.pdf", bbox_inches="tight", facecolor=facecolor)
    logger.info(f"  Saved: {basename}.png  +  {basename}.pdf")


# caption reports the EBIC γ, following the Burger et al. reporting checklist
# edge selection described as EBICglasso regularization, not arbitrary threshold
# n/p ratio reported
def _print_network_caption(
    analysis_title, n_subj, n_feat, gamma_str, layout_lbl, nstats, pred_r2, target_lbl=None, nstats_full=None
):
    avg_r2 = np.mean(list(pred_r2.values())) if pred_r2 else 0
    _np_ratio = n_subj / max(n_feat, 1)
    cap = (
        f"\n  +- FIGURE CAPTION (per Burger et al. 2022 reporting standards) ---\n"
        f"  | {analysis_title}\n"
        f"  | Partial correlation network of top-{n_feat} ML ensemble features"
    )
    if target_lbl:
        cap += f"\n  | for {target_lbl}"
    cap += (
        f".\n"
        f"  | Edges: regularised partial correlations (EBICglasso, {gamma_str}).\n"
        f"  | Edge selection: edges shrunk to zero by EBICglasso are excluded.\n"
        f"  | Display threshold: |partial r| < {_DISPLAY_THRESH} omitted for visual\n"
        f"  |   clarity; full EBICglasso adjacency available in supplement.\n"
    )
    if nstats_full:
        cap += (
            f"  | Full model: {nstats_full['n_edges']} edges "
            f"(density = {nstats_full['density']:.3f});  "
            f"displayed: {nstats['n_edges']} edges.\n"
        )
    cap += (
        f"  | Edge colour: blue = positive, vermillion = negative (Okabe-Ito).\n"
        f"  | Negative edges rendered as dashed lines for grayscale accessibility.\n"
        f"  | Edge width/opacity ∝ |partial r|, fixed maximum = {_EDGE_MAXIMUM}\n"
        f"  | (enables cross-figure comparison).\n"
        f"  | Node size ∝ ensemble consensus importance.\n"
        f"  | Colored fill within each node ring = predictability\n"
        f"  | (R²: precision-implied variance explained; avg = {avg_r2:.2f}).\n"
        f"  | R² computed via Haslbeck & Waldorp (2018) precision-implied method\n"
        f"  | using sample covariance of standardised data.\n"
        f"  | Border colour: blue = positive EI, orange = negative EI.\n"
        f"  | EI computed from full EBICglasso model (Robinaugh et al. 2016).\n"
        f"  | Layout: {layout_lbl}.\n"
        f"  | Network (displayed): {nstats['n_nodes']} nodes, {nstats['n_edges']} edges, "
        f"density = {nstats['density']:.3f},\n"
        f"  |   avg |weight| = {nstats['avg_abs_weight']:.4f}.\n"
        f"  | N = {n_subj:,}  ·  n/p = {_np_ratio:.0f}.\n"
        f"  | NOTE: Network constructed from top ML-selected features;\n"
        f"  |   conditional dependencies reflect inter-feature structure among\n"
        f"  |   top predictors, not a theory-driven psychometric network.\n"
        f"  | Cross-sectional structure; not interpretable as within-person\n"
        f"  | causal dynamics (Borsboom et al. 2021).\n"
        f"  +---"
    )
    logger.info(cap)


# Sidebar renderer: shared by per-target and consensus network figures


def _render_centrality_sidebar(ax_ei, ei_sorted, label_of_feat, feat_colors, biv_r, imp_of_feat=None):
    """
    Chip-table sidebar: two metric columns (EI, r(target)).

    Each metric cell = rounded chip with proportional fill + bold centered number.
      • EI, r(target): diverging: blue fill (+), red fill (−), fills from center
    Feature column: small colored dot (node color) + rank number + name.
    Rows sorted by |EI| descending at call site.
    """
    _tr = ax_ei.transAxes
    ax_ei.axis("off")
    _n_rows = len(ei_sorted)

    _COL_HDR_BG = "#E4EBF5"
    _COL_HDR_CLR = "#263238"
    _SEP_CLR = "#BED0E8"
    _ROW_GRAD_TOP = np.array([0.82, 0.90, 0.97])
    _ROW_GRAD_BOT = np.array([0.97, 0.99, 1.00])
    _FEAT_CLR = "#1A2533"
    _RANK_CLR = "#546E7A"
    _FOOT_CLR = "#78909C"
    _CHIP_TRK = "#DDE6F0"
    _CHIP_POS = "#1565C0"
    _CHIP_NEG = "#C62828"
    _NUM_POS = "#0A0A0A"
    _NUM_NEG = "#0A0A0A"
    _NUM_ZERO = "#444444"

    _col_h = 0.038
    _usable = 1.0 - _col_h - 0.004
    _row_h = min(_usable / max(_n_rows, 1), 0.074)
    _chip_h_f = 0.68

    def _row_cy(i):
        return 1.0 - _col_h - _row_h * (i + 0.5)

    _ei_vals = [v for _, v in ei_sorted]
    _ei_max = max(abs(v) for v in _ei_vals) if _ei_vals else 1.0
    _ei_max = _ei_max or 1.0

    # when biv_r is None (consensus: multi-target), skip r(target) chips
    _show_biv_r = biv_r is not None

    if _show_biv_r:
        _ry_vals = [biv_r.get(f, 0.0) for f, _ in ei_sorted]
        _ry_abs = [abs(v) for v in _ry_vals]
        _ry_max = max(_ry_abs) if _ry_abs else 1.0
        _ry_max = _ry_max or 1.0
    else:
        _ry_max = 1.0

    _dot_x = 0.022
    _dot_r = 0.009
    _rank_x = 0.048
    _name_x0 = 0.068
    _name_x1 = 0.480
    _ei_cl, _ei_cr = 0.492, 0.720
    _ry_cl, _ry_cr = 0.734, 0.988

    _col_bot = 1.0 - _col_h
    _col_cy = 1.0 - _col_h * 0.50
    ax_ei.add_patch(
        mpatches.Rectangle((0.0, _col_bot), 1.0, _col_h, fc=_COL_HDR_BG, ec="none", transform=_tr, zorder=1)
    )
    ax_ei.plot([0.0, 1.0], [_col_bot, _col_bot], color="#8FAEC8", lw=1.4, transform=_tr, zorder=2, clip_on=False)

    _hkw = dict(
        va="center", transform=_tr, zorder=3, family="sans-serif", fontsize=7.8, fontweight="800", color=_COL_HDR_CLR
    )
    ax_ei.text(_name_x0, _col_cy, "FEATURE", ha="left", **_hkw)
    ax_ei.text((_ei_cl + _ei_cr) / 2, _col_cy, "EI (|EI|↓)", ha="center", **_hkw)
    _ry_hdr = "r(target)" if _show_biv_r else "(multi-target)"
    ax_ei.text((_ry_cl + _ry_cr) / 2, _col_cy, _ry_hdr, ha="center", **_hkw)

    for _ri, (_f, _eiv) in enumerate(ei_sorted):
        _yc = _row_cy(_ri)
        _rbot = _yc - _row_h * 0.50
        _nc = feat_colors.get(_f, "#888")
        _lbl = label_of_feat.get(_f, _f)
        _ryv = biv_r.get(_f, 0.0) if _show_biv_r else None
        _ch = _chip_h_f * _row_h

        _t = _ri / max(_n_rows - 1, 1)
        _row_rgb = (1 - _t) * _ROW_GRAD_TOP + _t * _ROW_GRAD_BOT
        ax_ei.add_patch(
            mpatches.Rectangle((0.0, _rbot), 1.0, _row_h, fc=tuple(_row_rgb), ec="none", transform=_tr, zorder=0)
        )

        ax_ei.plot([0.0, 1.0], [_rbot, _rbot], color=_SEP_CLR, lw=0.4, transform=_tr, zorder=1, clip_on=False)

        ax_ei.add_patch(
            Circle(
                (_dot_x, _yc),
                _dot_r,
                facecolor=_nc,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.90,
                transform=_tr,
                zorder=3,
            )
        )

        ax_ei.text(
            _rank_x,
            _yc,
            str(_ri + 1),
            ha="center",
            va="center",
            transform=_tr,
            fontsize=6.8,
            fontweight="700",
            color=_RANK_CLR,
            zorder=4,
            family="sans-serif",
        )

        import textwrap as _tw

        if len(_lbl) <= 22:
            _name_lines = [_lbl]
            _name_fs = 8.8
        else:
            _wrapped = _tw.wrap(_lbl, width=22)[:2]
            _name_lines = _wrapped
            _name_fs = 7.8 if max(len(l) for l in _wrapped) > 18 else 8.2
        _n_lines = len(_name_lines)
        _line_gap = _row_h * 0.44
        _name_y_start = _yc + (_line_gap * (_n_lines - 1) / 2.0)
        for _li, _line in enumerate(_name_lines):
            ax_ei.text(
                _name_x0,
                _name_y_start - _li * _line_gap,
                _line,
                fontsize=_name_fs,
                color=_FEAT_CLR,
                va="center",
                fontweight="600",
                transform=_tr,
                zorder=3,
                family="sans-serif",
                clip_on=True,
            )

        def _chip(x0, x1, value, vmax, diverging, fill_clr, num_clr):
            _cw = x1 - x0
            _cx = (x0 + x1) / 2.0
            _cbot = _yc - _ch * 0.50
            ax_ei.add_patch(
                mpatches.FancyBboxPatch(
                    (x0, _cbot),
                    _cw,
                    _ch,
                    boxstyle="round,pad=0.005",
                    fc=_CHIP_TRK,
                    ec="#B0C4D8",
                    lw=0.5,
                    transform=_tr,
                    zorder=2,
                )
            )
            _frac = min(abs(value) / vmax, 1.0) if vmax > 0 else 0.0
            if _frac > 0.02:
                _pad_v = _ch * 0.12
                if diverging:
                    _hw = _cw / 2.0
                    _fw = _frac * _hw
                    _fx0 = _cx if value >= 0 else (_cx - _fw)
                    ax_ei.add_patch(
                        mpatches.FancyBboxPatch(
                            (_fx0, _cbot + _pad_v),
                            _fw,
                            _ch - 2 * _pad_v,
                            boxstyle="round,pad=0.003",
                            fc=fill_clr,
                            ec="none",
                            alpha=0.78,
                            transform=_tr,
                            zorder=3,
                        )
                    )
                    ax_ei.plot(
                        [_cx, _cx],
                        [_cbot + _pad_v, _cbot + _ch - _pad_v],
                        color="#8FAEC8",
                        lw=0.7,
                        transform=_tr,
                        zorder=4,
                    )
                else:
                    _fw = _frac * _cw
                    ax_ei.add_patch(
                        mpatches.FancyBboxPatch(
                            (x0, _cbot + _ch * 0.12),
                            _fw,
                            _ch * 0.76,
                            boxstyle="round,pad=0.003",
                            fc=fill_clr,
                            ec="none",
                            alpha=0.78,
                            transform=_tr,
                            zorder=3,
                        )
                    )
            _sgn = "+" if (diverging and value > 0.005) else ""
            ax_ei.text(
                _cx,
                _yc,
                f"{_sgn}{value:.2f}",
                ha="center",
                va="center",
                transform=_tr,
                fontsize=8.5,
                fontweight="800",
                color=num_clr,
                zorder=5,
                family="sans-serif",
            )

        _ei_nc = _NUM_POS if _eiv > 0.005 else (_NUM_NEG if _eiv < -0.005 else _NUM_ZERO)
        _chip(_ei_cl, _ei_cr, _eiv, _ei_max, True, _CHIP_POS if _eiv >= 0 else _CHIP_NEG, _ei_nc)

        if _show_biv_r and _ryv is not None:
            _ry_nc = _NUM_POS if _ryv > 0.005 else (_NUM_NEG if _ryv < -0.005 else _NUM_ZERO)
            _chip(_ry_cl, _ry_cr, _ryv, _ry_max, True, _CHIP_POS if _ryv >= 0 else _CHIP_NEG, _ry_nc)
        else:
            # Multi-target consensus: show "--" instead of misleading 0.00
            ax_ei.text(
                (_ry_cl + _ry_cr) / 2,
                _yc,
                "--",
                ha="center",
                va="center",
                transform=_tr,
                fontsize=8.5,
                fontweight="600",
                color=_NUM_ZERO,
                zorder=5,
                family="sans-serif",
            )


# Shared edge-rendering helper


def _render_edges(ax, G, pos, node_list, data_span):
    """
    Draw magnitude-based edges with statistical glow threshold.

    Returns dict of {(u,v): (mid_x, mid_y, weight, p1, p2)} for the top-N
    edges (used for on-figure weight labels with perpendicular offset).

    Geometry-based curve direction: edges always curve outward
      from the layout center, avoiding arcs that cross through node interiors.
    """
    if G.number_of_edges() == 0:
        return {}

    _edge_sorted = sorted(G.edges(data=True), key=lambda x: abs(x[2]["weight"]), reverse=True)

    _weights = [abs(d["weight"]) for _, _, d in _edge_sorted]
    _glow_thresh = np.mean(_weights) + 1.0 * np.std(_weights)
    _strong_edges = set()
    for _u, _v, _d in _edge_sorted:
        if abs(_d["weight"]) >= _glow_thresh:
            _strong_edges.add((_u, _v))
            _strong_edges.add((_v, _u))

    # compute layout center for outward-curve sign
    _all_x = [pos[nd][0] for nd in node_list]
    _all_y = [pos[nd][1] for nd in node_list]
    _cx = np.mean(_all_x)
    _cy = np.mean(_all_y)

    _edge_midpoints = {}

    for _eidx, (_u, _v) in enumerate(G.edges()):
        _pc = G.edges[(_u, _v)]["weight"]
        _aw = abs(_pc)

        # Power-law mapping (exponent 0.6) concentrates visual contrast in
        # the mid-to-strong range where interesting edges live, while
        # compressing near-zero edges into ghostlike hairlines.
        _t = min(_aw / _EDGE_MAXIMUM, 1.0) ** 0.6
        _lw = _LW_MIN + (_LW_MAX - _LW_MIN) * _t
        _alp = _ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * _t

        _base_ec = _EDGE_POS if _pc > 0 else _EDGE_NEG
        _ls = "-" if _pc > 0 else _DASH_NEG
        _nc_u = G.nodes[_u]["color"]
        _nc_v = G.nodes[_v]["color"]
        _is_str = (_u, _v) in _strong_edges or (_v, _u) in _strong_edges

        # compute outward curve sign from geometry
        # Chord midpoint relative to layout center; perpendicular direction
        # that points outward gets positive sign.
        _p1, _p2 = pos[_u], pos[_v]
        _emx = (_p1[0] + _p2[0]) / 2.0 - _cx
        _emy = (_p1[1] + _p2[1]) / 2.0 - _cy
        _edx = _p2[0] - _p1[0]
        _edy = _p2[1] - _p1[1]
        # Cross product of chord direction with radial direction determines
        # which perpendicular side is "outward"
        _cross = (-_edy) * _emx + _edx * _emy
        _csign = 1 if _cross >= 0 else -1

        ctrl = _draw_fancy_edge(
            ax, _p1, _p2, _nc_u, _nc_v, _base_ec, _lw, _alp, _ls, _eidx, _is_str, data_span=data_span, curve_sign=_csign
        )

        if (_u, _v) in _strong_edges and (_u, _v) not in _edge_midpoints:
            _bmx = 0.25 * _p1[0] + 0.5 * ctrl[0] + 0.25 * _p2[0]
            _bmy = 0.25 * _p1[1] + 0.5 * ctrl[1] + 0.25 * _p2[1]
            _edge_midpoints[(_u, _v)] = (_bmx, _bmy, _pc, _p1, _p2)

    return _edge_midpoints


def _render_edge_labels(ax, edge_midpoints, max_labels=3):
    """On-figure edge weight annotations for strongest edges.

    Labels are offset perpendicular to the chord so they don't
    overlap the arc or sit on top of nodes for short edges.
    """
    _sorted = sorted(edge_midpoints.items(), key=lambda kv: abs(kv[1][2]), reverse=True)[:max_labels]
    for (_u, _v), vals in _sorted:
        if len(vals) == 5:
            _mx, _my, _wt, _p1, _p2 = vals
            # Perpendicular to chord, nudge outward from arc midpoint
            _pdx = -(_p2[1] - _p1[1])
            _pdy = _p2[0] - _p1[0]
            _plen = max(np.sqrt(_pdx**2 + _pdy**2), 1e-6)
            _nudge = 0.25  # data units
            _lx = _mx + _nudge * _pdx / _plen
            _ly = _my + _nudge * _pdy / _plen
        else:
            _mx, _my, _wt = vals[:3]
            _lx, _ly = _mx, _my
        ax.text(
            _lx,
            _ly,
            f"{_wt:+.2f}",
            fontsize=7,
            color=_CLR_SLATE,
            ha="center",
            va="center",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.88, ec="#CFD8DC", lw=0.5),
        )


# Label repulsion helper: resolves collision on circle layouts


def _repel_labels(label_positions, node_positions, min_sep=1.5, node_pull=0.15, iterations=80):
    """
    Spring-repulsion pass to separate overlapping node labels.

    label_positions : list of [lx, ly], MUTATED in place
    node_positions  : list of (nx, ny), label anchor points (attract labels)
    min_sep         : minimum distance between label centers before push
    node_pull       : fraction of each step that pulls labels toward anchor
    iterations      : number of relaxation passes

    Returns the mutated label_positions.
    """
    lp = [list(p) for p in label_positions]
    n = len(lp)
    for _ in range(iterations):
        for i in range(n):
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = lp[i][0] - lp[j][0]
                dy = lp[i][1] - lp[j][1]
                dist = max(np.sqrt(dx**2 + dy**2), 0.01)
                if dist < min_sep:
                    push = (min_sep - dist) / dist * 0.30
                    fx += dx * push
                    fy += dy * push
            # Light pull back toward own node to prevent drift
            ax, ay = node_positions[i]
            fx += (ax - lp[i][0]) * node_pull
            fy += (ay - lp[i][1]) * node_pull
            lp[i][0] += fx
            lp[i][1] += fy
    return lp


# Main body function


def _run_for_current_target():

    # 1. Prerequisites
    _has_train = "X_train" in ctx and "y_train" in ctx and ctx.X_train is not None
    _has_within = (
        "exported_within_dict" in ctx
        and isinstance(ctx.exported_within_dict, dict)
        and len(ctx.exported_within_dict) > 0
    )
    _has_metrics = (
        "metrics_dict_full" in ctx
        and isinstance(ctx.metrics_dict_full, dict)
        and len(ctx.metrics_dict_full.get("category", [])) > 0
    )
    _has_model = (
        "last_fitted_model" in ctx
        and ctx.last_fitted_model is not None
        and hasattr(ctx.last_fitted_model, "base_models")
        and hasattr(ctx.last_fitted_model, "meta_model")
    )
    _split = ctx.split if "split" in ctx else "across_categories"
    _is_within = _split == "within_categories"
    _is_across = _split == "across_categories"
    _is_reg = (len(np.unique(ctx.y_train)) > 10) if _has_train else False
    _dn = DISPLAY_NAMES if ("DISPLAY_NAMES" in ctx and isinstance(DISPLAY_NAMES, dict)) else {}
    _target_lbl = ctx.target_options if "target_options" in ctx else "?"
    _tp_lbl = ctx.tp_option if "tp_option" in ctx else "?"

    if not _has_train:
        logger.info("Error:  X_train / y_train not found -- run the Main Analysis Runner first.")
        return

    logger.info("-" * 72)
    logger.info(f"  Partial Correlation Network (EBICglasso, γ = {_EBIC_GAMMA})")
    logger.info(f"  Mode   : {'Within' if _is_within else 'Across'}-categories")
    logger.info(f"  Target : {_target_lbl}  ·  T{_tp_lbl}")
    logger.info(f"  Train  : {ctx.X_train.shape[0]:,} × {ctx.X_train.shape[1]:,}")
    logger.info("-" * 72)

    # 2. Consensus importance
    def _abbrev_cat(cat, max_len=22):
        mapping = {
            "Interpersonal Relationships & Social Functioning": "Social Functioning",
            "Cognitive Task Outcomes": "Cognitive Tasks",
            "Multimodal Neuroimaging": "Neuroimaging",
            "Residential Characteristics": "Residential",
            "Family Dynamics & Parenting": "Family Dynamics",
            "Other Personality Features": "Personality",
            "Medical/Somatic Problems": "Medical",
            "Physical Activity/Features": "Physical Activity",
            "Task and Resting State": "Task/Rest fMRI",
            "School Dynamics": "School",
            "Social Relationship Quality": "Social Rel.",
            "Technology Use": "Tech Use",
            "Sleep Problems": "Sleep",
            "SES & Mobility": "SES",
        }
        for long, short in mapping.items():
            cat = cat.replace(long, short)
        cat = cat.replace("and ", "& ").replace("_", " ")
        return cat[:max_len] + ("…" if len(cat) > max_len else "")

    def _consensus_from_pretrained(model, Xtr, ytr):
        original_columns = Xtr.columns.tolist()
        feature_importances = {}
        all_features = set()
        if hasattr(model.meta_model, "coef_"):
            mc = model.meta_model.coef_
            if mc.ndim > 1:
                mc = np.mean(np.abs(mc), axis=0)
            bn = list(model.base_models.keys())
            mw = {bn[i]: abs(mc[i]) if i < len(mc) else 0 for i in range(len(bn))}
        else:
            mw = {n: 1.0 for n in model.base_models}
        ws = sum(mw.values()) or 1
        mw = {k: v / ws for k, v in mw.items()}
        for name, bm in model.base_models.items():
            try:
                if hasattr(bm, "feature_importances_"):
                    imp = bm.feature_importances_
                elif hasattr(bm, "coef_"):
                    imp = np.abs(bm.coef_)
                    if imp.ndim > 1:
                        imp = imp.mean(axis=0)
                else:
                    continue
                if len(imp) != len(original_columns):
                    continue
                mi = pd.Series(imp, index=original_columns)
                feature_importances[name] = mi
                all_features.update(mi.index)
            except Exception:
                pass
        if not feature_importances:
            return None
        cs = {}
        for feat in all_features:
            ws2 = 0.0
            for mn, w in mw.items():
                if mn in feature_importances:
                    iv = feature_importances[mn].get(feat, 0)
                    ws2 += (iv / (feature_importances[mn].max() + 1e-12)) * w
            cs[feat] = ws2
        return pd.Series(cs).sort_values(ascending=False)

    def _consensus_importance_refit(Xtr, ytr):
        imp_dict = {}
        _imp = SimpleImputer(strategy="median")
        Xcln = pd.DataFrame(_imp.fit_transform(Xtr.replace([np.inf, -np.inf], np.nan)), columns=Xtr.columns)
        Xcln = Xcln.loc[:, Xcln.nunique() > 1]
        ycln = np.array(ytr).ravel()
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            m = (CatBoostRegressor if _is_reg else CatBoostClassifier)(
                iterations=1000, learning_rate=0.05, verbose=0, random_state=42
            )
            m.fit(Xcln, ycln)
            r = m.feature_importances_
            imp_dict["catboost"] = r / (r.sum() + 1e-12)
        except Exception:
            pass
        try:
            from xgboost import XGBClassifier, XGBRegressor

            m = (XGBRegressor if _is_reg else XGBClassifier)(
                n_estimators=ctx._XGB_PARAMS["n_estimators"],
                max_depth=ctx._XGB_PARAMS["max_depth"],
                learning_rate=ctx._XGB_PARAMS["learning_rate"],
                random_state=42,
                verbosity=0,
                **({} if _is_reg else {"eval_metric": "logloss"}),
            )
            m.fit(Xcln, ycln)
            r = m.feature_importances_
            imp_dict["xgboost"] = r / (r.sum() + 1e-12)
        except Exception:
            pass
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            m = (RandomForestRegressor if _is_reg else RandomForestClassifier)(
                n_estimators=ctx._RF_PARAMS["n_estimators"],
                max_depth=ctx._RF_PARAMS["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
            m.fit(Xcln, ycln)
            r = m.feature_importances_
            imp_dict["random_forest"] = r / (r.sum() + 1e-12)
        except Exception:
            pass
        if not imp_dict:
            return None, Xcln
        return pd.DataFrame(imp_dict, index=Xcln.columns).mean(axis=1).sort_values(ascending=False), Xcln

    def _consensus_from_globals():
        try:
            _gdf = ctx.consensus_df if "consensus_df" in ctx else None
        except Exception:
            _gdf = None
        if _gdf is None or not isinstance(_gdf, pd.DataFrame) or "weighted_score" not in _gdf.columns:
            return None, None
        _rev = {v: k for k, v in _dn.items()}
        _s = _gdf.sort_values("weighted_score", ascending=False)
        _c = pd.Series(_s["weighted_score"].values, index=[_rev.get(f, f) for f in _s.index])
        _c = _c[~_c.index.duplicated(keep="first")]
        try:
            _mw = ctx.model_weights if "model_weights" in ctx and isinstance(ctx.model_weights, dict) else {}
        except Exception:
            _mw = {}
        return _c, _mw

    def _get_top10_consensus(Xtr, ytr):
        _cg, _mg = _consensus_from_globals()
        if _cg is not None and len(_cg) > 0:
            logger.info("\n>  Using consensus_df (exact match -- incl. TabPFN)")
            return _cg, "main_runner_consensus_df"
        if _has_model:
            c = _consensus_from_pretrained(ctx.last_fitted_model, Xtr, ytr)
            if c is not None and len(c) > 0:
                return c, "pretrained_ensemble"
        c, _ = _consensus_importance_refit(Xtr, ytr)
        return c, "matched_refit"

    def _get_top1_consensus_within(Xtr, ytr):
        c, _ = _consensus_importance_refit(Xtr, ytr)
        return c

    # 3. Feature selection
    _feat_list = []
    _cat_of_feat = {}
    _imp_of_feat = {}
    _label_of_feat = {}

    if _is_across:
        _consensus, _source = _get_top10_consensus(ctx.X_train, ctx.y_train)
        if _consensus is None:
            logger.info("Error:  No models converged.")
            return
        _top10 = _consensus.head(10)
        for _rk, (_f, _v) in enumerate(_top10.items(), 1):
            _feat_list.append(_f)
            _cat_of_feat[_f] = f"#{_rk}"
            _imp_of_feat[_f] = float(_v)
            _label_of_feat[_f] = _display_name(_f)
        _analysis_title = f"Top-10 Ensemble Features  ·  {_target_lbl}  (T{_tp_lbl})"

    elif _is_within:
        if not _has_within:
            logger.info("Error:  exported_within_dict not found.")
            return
        _perf_col = "r2" if _is_reg else "auc"
        _cat_perf = {}
        if _has_metrics:
            for _cat, _mdict in zip(ctx.metrics_dict_full["category"], ctx.metrics_dict_full["metrics"]):
                _m = _mdict.get("ensemble", _mdict.get("meta_model", {}))
                _v = _m.get(_perf_col, _m.get("accuracy", np.nan))
                if _cat is not None and not np.isnan(_v):
                    _cat_perf[_cat] = float(_v)
        _ranked = sorted(ctx.exported_within_dict.keys(), key=lambda c: _cat_perf.get(c, -np.inf), reverse=True)[:10]
        for _cat in _ranked:
            _Xtr, _Xte, _ytr, _yte = ctx.exported_within_dict[_cat]
            _c = _get_top1_consensus_within(_Xtr, _ytr)
            if _c is None or len(_c) == 0:
                continue
            _f = _c.index[0]
            _v = float(_c.iloc[0])
            _feat_list.append(_f)
            _cat_of_feat[_f] = _cat
            _imp_of_feat[_f] = _v
            _label_of_feat[_f] = _display_name(_f)
        _analysis_title = f"Top-10 Category Representatives  ·  {_target_lbl}  (T{_tp_lbl})"
    else:
        return

    # 4. Build cross-feature matrix
    _feat_list = [f for f in _feat_list if f in ctx.X_train.columns]
    if len(_feat_list) < 3:
        logger.info("Warning:  < 3 features available.")
        return

    _cross_df = ctx.X_train[_feat_list].copy().replace([np.inf, -np.inf], np.nan).dropna()
    _scaler = StandardScaler()
    _cross_scl = pd.DataFrame(_scaler.fit_transform(_cross_df), columns=_cross_df.columns)
    _labels_plot = [_label_of_feat.get(f, f) for f in _feat_list]
    n = len(_feat_list)

    _feat_colors, _domain_handles = _get_domain_colors(_feat_list, _cat_of_feat, _is_within)

    _corr = _cross_df.corr(method="pearson")
    _dist = (1 - _corr.abs()).clip(lower=0).to_numpy(copy=True)
    np.fill_diagonal(_dist, 0)
    _linkage = linkage(squareform(_dist, checks=False), method="average")
    _order = dendrogram(_linkage, no_plot=True)["leaves"]
    _corr_ord = _corr.iloc[_order, _order]
    _lbl_ord = [_labels_plot[i] for i in _order]

    try:
        _batch_name = _target_lbl
        if "batch_corr_dict" not in ctx:
            ctx["batch_corr_dict"] = {}
            ctx["batch_feat_list_corr"] = list(_feat_list)
        _common = ctx["batch_feat_list_corr"]
        _corr_sub = _corr.loc[_common, _common].copy()
        ctx["batch_corr_dict"][_batch_name] = _corr_sub
    except Exception:
        pass  # non-critical; logged upstream or optional visualization

    # report n/p ratio
    _np_ratio = len(_cross_df) / max(n, 1)
    logger.info(f"\n  Feature matrix: {_cross_df.shape[0]:,} × {n}  (n/p = {_np_ratio:.0f})")

    # 5. FIGURE 1: Correlation Heatmap
    logger.info("\n" + "=" * 72)
    logger.info("  Figure 1 -- Clustered Pearson Correlation Matrix")
    logger.info("=" * 72)

    _cell_px = 72
    _max_llen = max(len(l) for l in _lbl_ord)
    _lbl_inch = max(_max_llen * 0.078, 1.8)
    _fw1 = n * (_cell_px / 100) + _lbl_inch + 2.8
    _fh1 = n * (_cell_px / 100) + 3.5

    fig1, ax1 = plt.subplots(figsize=(_fw1, _fh1))
    fig1.patch.set_facecolor("white")
    _im1 = ax1.imshow(_corr_ord.values, cmap=_CORR_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(n):
        for j in range(i + 1, n):
            ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec="white", lw=0, zorder=2))
    _fs_ann = max(7, min(11, 14 - n * 0.35))
    for i in range(n):
        for j in range(i + 1):
            v = _corr_ord.values[i, j]
            tc = "white" if abs(v) > 0.50 else _CLR_CHARCOAL
            fw = "bold" if abs(v) > 0.28 else "normal"
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", color=tc, fontsize=_fs_ann, fontweight=fw, zorder=3)
    _tf = max(8.5, min(11, 130 / max(n, 1)))
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(_lbl_ord, rotation=45, ha="right", fontsize=_tf, color=_CLR_CHARCOAL)
    ax1.set_yticklabels(_lbl_ord, fontsize=_tf, color=_CLR_CHARCOAL)
    ax1.tick_params(axis="both", length=0, pad=8)
    cb1 = fig1.colorbar(_im1, ax=ax1, shrink=0.52, aspect=20, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1])
    cb1.set_label("Pearson r", fontsize=10, color=_CLR_CHARCOAL, labelpad=8)
    cb1.ax.tick_params(labelsize=9, colors=_CLR_CHARCOAL)
    cb1.outline.set_visible(False)
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.grid(False)
    ax1.set_title(
        f"{_analysis_title}\nClustered Pearson Correlation Matrix  (N = {len(_cross_df):,})",
        fontsize=12,
        fontweight="bold",
        color=_CLR_SLATE,
        pad=16,
    )
    plt.tight_layout()
    _save_fig(fig1, "cross_feature_correlation_matrix")
    plt.show()

    # 6. FIGURE 2: Partial Correlation Network
    logger.info("\n" + "=" * 72)
    logger.info("  Figure 2 -- Partial Correlation Network  (EBICglasso + Circular)")
    logger.info("=" * 72)

    _pcor_df = None
    _prec = None
    _best_alpha = None
    _sample_cov = None
    try:
        # EBICglasso
        _prec, _ebic_cov, _best_alpha, _gamma_used = _ebic_glasso(
            _cross_scl.values, gamma=_EBIC_GAMMA, n_alpha=_EBIC_N_ALPHA, max_iter=1000
        )
        if _prec is not None:
            _dsqrt = np.sqrt(np.diag(_prec))
            _pcor = -_prec / np.outer(_dsqrt, _dsqrt)
            np.fill_diagonal(_pcor, 1.0)
            _pcor_df = pd.DataFrame(_pcor, index=_feat_list, columns=_feat_list)
            # use sample covariance (not regularised) for R²
            _sample_cov = np.cov(_cross_scl.values, rowvar=False, bias=False)
            logger.info(f"  EBICglasso: γ = {_gamma_used}, λ (selected) = {_best_alpha:.6f}")
        else:
            raise ValueError("EBICglasso returned no valid solution")
    except Exception as _eg:
        logger.info(f"  Warning:  EBICglasso failed ({_eg}) -- pseudoinverse fallback.")
        try:
            _sample_cov = np.cov(_cross_scl.values, rowvar=False, bias=False)
            _prec = np.linalg.pinv(_sample_cov)
            _dsqrt = np.sqrt(np.clip(np.diag(_prec), 1e-12, None))
            _pcor = -_prec / np.outer(_dsqrt, _dsqrt)
            np.fill_diagonal(_pcor, 1.0)
            _pcor_df = pd.DataFrame(_pcor, index=_feat_list, columns=_feat_list)
        except Exception as _eg2:
            logger.info(f"  Error:  Fallback failed: {_eg2}")

    if _pcor_df is not None:
        import textwrap

        import networkx as nx

        try:
            _batch_name = _target_lbl
            if "batch_pcor_dict" not in ctx:
                ctx["batch_pcor_dict"] = {}
                ctx["batch_feat_list"] = list(_feat_list)
            _common = ctx["batch_feat_list"]
            _pcor_sub = _pcor_df.loc[_common, _common].copy()
            ctx["batch_pcor_dict"][_batch_name] = _pcor_sub
        except Exception:
            pass  # non-critical; logged upstream or optional visualization

        # Build FULL graph (all EBICglasso-retained edges, 1e-6 noise filter)
        # for centrality metrics (EI, Strength) per Robinaugh et al. (2016).
        G_full = nx.Graph()
        for _f in _feat_list:
            G_full.add_node(
                _f, label=_label_of_feat.get(_f, _f), color=_feat_colors.get(_f, "#888"), imp=_imp_of_feat.get(_f, 0.01)
            )
        _n_edges_full = 0
        for _i in range(n):
            for _j in range(_i + 1, n):
                _pc = _pcor_df.iloc[_i, _j]
                if abs(_pc) > 1e-6:
                    G_full.add_edge(_feat_list[_i], _feat_list[_j], weight=_pc, aw=abs(_pc))
                    _n_edges_full += 1

        # EI and Strength from full EBICglasso model (not display-filtered)
        _EI = _compute_EI_from_graph(G_full, _feat_list)
        _strength = _compute_strength_from_graph(G_full, _feat_list)

        # Build DISPLAY graph (|pcor| > _DISPLAY_THRESH for visual clarity)
        G = nx.Graph()
        for _f in _feat_list:
            G.add_node(
                _f, label=_label_of_feat.get(_f, _f), color=_feat_colors.get(_f, "#888"), imp=_imp_of_feat.get(_f, 0.01)
            )
        for _i in range(n):
            for _j in range(_i + 1, n):
                _pc = _pcor_df.iloc[_i, _j]
                if abs(_pc) > _DISPLAY_THRESH:
                    G.add_edge(_feat_list[_i], _feat_list[_j], weight=_pc, aw=abs(_pc))

        _nstats = _network_stats(G)
        _nstats_full = _network_stats(G_full)
        logger.info(
            f"  EBICglasso model: {_nstats_full['n_edges']} edges retained (density = {_nstats_full['density']:.3f})"
        )
        logger.info(f"  Displayed (|pcor| > {_DISPLAY_THRESH}): {_nstats['n_nodes']} nodes, {_nstats['n_edges']} edges")
        logger.info(f"  Avg |w| (displayed): {_nstats['avg_abs_weight']:.4f}")

        for _f in _feat_list:
            if _f in G.nodes:
                G.nodes[_f]["ei"] = _EI.get(_f, 0.0)

        # precision-implied R² using sample covariance
        if _prec is not None and _sample_cov is not None:
            _pred_r2 = _compute_predictability(_feat_list, prec=_prec, cov=_sample_cov)
        else:
            _pred_r2 = _compute_predictability(_feat_list, cross_df=_cross_df, pcor_df=_pcor_df, thresh=1e-6)
        logger.info(
            f"  Predictability (precision-implied, sample cov): avg R² = {np.mean(list(_pred_r2.values())):.3f}"
        )
        for _f in sorted(_pred_r2, key=_pred_r2.get, reverse=True):
            logger.info(f"    {_label_of_feat.get(_f, _f):<45} R² = {_pred_r2[_f]:.3f}")

        _biv_r = {}
        if _has_train:
            _ya_series = pd.Series(np.array(ctx.y_train).ravel(), index=ctx.X_train.index)
            _ya_cross = _ya_series.reindex(_cross_df.index).values
            for _f in _feat_list:
                _x = _cross_df[_f].values
                _valid = ~(np.isnan(_x) | np.isnan(_ya_cross))
                if _valid.sum() > 10:
                    from scipy.stats import pearsonr

                    _r, _ = pearsonr(_x[_valid], _ya_cross[_valid])
                    _biv_r[_f] = _r
                else:
                    _biv_r[_f] = 0.0

        # dynamic circle scale: spreads nodes for dense networks
        _circle_scale = 5.5 + n * 0.25
        _pos = _fiedler_circle_layout(G, scale=_circle_scale)
        _layout_lbl = "Spectral (Fiedler)"
        logger.info(f"  Layout: {_layout_lbl} (nodes ordered by Fiedler vector + 2-opt)")

        _node_list = list(G.nodes())
        _imp_arr = np.array([G.nodes[nd]["imp"] for nd in _node_list])
        _imp_rng = max(_imp_arr.max() - _imp_arr.min(), 1e-9)

        _xs = [_pos[nd][0] for nd in _node_list]
        _ys = [_pos[nd][1] for nd in _node_list]
        _data_span = max(max(_xs) - min(_xs), max(_ys) - min(_ys), 0.01)
        _base_r = _data_span * 0.032
        _radii = [
            _base_r * (0.4 + 2.6 * ((_imp_arr[i] - _imp_arr.min()) / _imp_rng) ** 0.5) for i in range(len(_node_list))
        ]

        _border_colors = []
        _border_widths = []
        _ei_arr = np.array([abs(_EI[nd]) for nd in _node_list])
        _ei_rng = max(_ei_arr.max(), 1e-9)
        for _i, _nd in enumerate(_node_list):
            _eiv = _EI[_nd]
            if abs(_eiv) < 0.01:
                _border_colors.append(_BORDER_ZERO)
            elif _eiv > 0:
                _border_colors.append(_BORDER_POS)
            else:
                _border_colors.append(_BORDER_NEG)
            # floor raised 1.2→2.0pt so low-EI borders are visible
            _border_widths.append(2.0 + 8.0 * (_ei_arr[_i] / _ei_rng))

        _net_w = max(11, n * 1.10)
        _ei_w = max(5.5, n * 0.46)
        _fig_h = max(10, n * 1.05)

        fig2 = plt.figure(figsize=(_net_w + _ei_w + 1.0, _fig_h))
        fig2.patch.set_facecolor("white")
        _gs = gridspec.GridSpec(
            1, 2, figure=fig2, width_ratios=[_net_w, _ei_w], wspace=0.14, left=0.01, right=0.99, top=0.91, bottom=0.16
        )
        ax2 = fig2.add_subplot(_gs[0, 0])
        ax_ei = fig2.add_subplot(_gs[0, 1])
        ax2.set_facecolor(_NET_BG)
        ax2.grid(False)
        ax2.axis("off")

        # Edges (shared helper)
        _edge_midpoints = _render_edges(ax2, G, _pos, _node_list, _data_span)

        # Nodes
        _node_idx = {nd: i for i, nd in enumerate(_node_list)}
        for _i, _nd in enumerate(_node_list):
            _x, _y = _pos[_nd]
            _r2v = _pred_r2.get(_nd, 0.0)
            _nc = G.nodes[_nd]["color"]
            _draw_predictability_node(ax2, _x, _y, _radii[_i], _r2v, _nc, _border_colors[_i], _border_widths[_i])

        # Labels (with repulsion to prevent collision)
        _cx = np.mean(_xs)
        _cy = np.mean(_ys)
        _off = _data_span * 0.40
        _fs_lbl = max(10, min(13, 16 - n * 0.14))
        _ww = 22
        _max_radius = max(_radii) if _radii else 1.0

        # collect initial radial positions, then repel
        _init_lbl_pos = []
        _node_anchors = []
        _lbl_texts = []
        for _nd in _node_list:
            _x, _y = _pos[_nd]
            _raw = G.nodes[_nd]["label"]
            _lbl_texts.append("\n".join(textwrap.wrap(_raw, width=_ww)))
            _dx = _x - _cx
            _dy = _y - _cy
            _d = max(np.sqrt(_dx**2 + _dy**2), 0.001)
            _init_lbl_pos.append([_x + _off * _dx / _d, _y + _off * _dy / _d])
            _node_anchors.append((_x + _off * 0.5 * _dx / _d, _y + _off * 0.5 * _dy / _d))

        _repelled = _repel_labels(
            _init_lbl_pos, _node_anchors, min_sep=_data_span * 0.12, node_pull=0.12, iterations=80
        )

        for _li, _nd in enumerate(_node_list):
            _x, _y = _pos[_nd]
            _lx, _ly = _repelled[_li]
            _ni = _node_idx[_nd]
            _shrinkA = 10 + 8 * (_radii[_ni] / _max_radius)
            ax2.annotate(
                _lbl_texts[_li],
                xy=(_x, _y),
                xytext=(_lx, _ly),
                fontsize=_fs_lbl,
                color=_CLR_SLATE,
                fontweight=600,
                ha="center",
                va="center",
                clip_on=False,
                arrowprops=dict(arrowstyle="-", color=_ARROW_CLR, lw=0.9, shrinkA=_shrinkA, shrinkB=4),
                bbox=dict(
                    boxstyle="round,pad=0.15", fc="white", alpha=0.93, ec=_feat_colors.get(_nd, _CLR_SILVER), lw=1.5
                ),
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white"), pe.Normal()],
                zorder=7,
            )

        _pad = _data_span * 0.30 + 2.2
        ax2.set_xlim(min(_xs) - _pad, max(_xs) + _pad)
        ax2.set_ylim(min(_ys) - _pad, max(_ys) + _pad)

        # Network key + edge scale (bottom-left)
        _legend_items = [
            plt.Line2D([0], [0], color=_EDGE_POS, lw=2.0, linestyle="-", label="Positive pcor (solid)"),
            plt.Line2D([0], [0], color=_EDGE_NEG, lw=2.0, linestyle=_DASH_NEG, label="Negative pcor (dashed)"),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#111111",
                markerfacecoloralt=_RING_UNFILLED,
                fillstyle="left",
                markeredgecolor=_RING_UNFILLED,
                markeredgewidth=1.2,
                markersize=9,
                label="Ring fill = R² predictability",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=_BORDER_POS,
                markeredgewidth=3.5,
                markersize=9,
                label="Blue border = +EI",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=_BORDER_NEG,
                markeredgewidth=3.5,
                markersize=9,
                label="Orange border = −EI",
            ),
        ]
        for _sr in [0.10, 0.20, 0.30]:
            _t_s = min(_sr / _EDGE_MAXIMUM, 1.0) ** 0.6
            _sw = max(_LW_MIN + (_LW_MAX - _LW_MIN) * _t_s, 0.4)
            _sa = max(_ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * _t_s, 0.10)
            _sl = f"|r| = {_sr:.2f}+" if _sr == 0.30 else f"|r| = {_sr:.2f}"
            _legend_items.append(
                plt.Line2D([0], [0], color=_EDGE_POS, lw=_sw, alpha=_sa, solid_capstyle="round", label=_sl)
            )
        if _domain_handles:
            _legend_items = _domain_handles + _legend_items
        fig2.legend(
            handles=_legend_items,
            fontsize=7.5,
            framealpha=0.92,
            edgecolor="#CFD8DC",
            fancybox=True,
            title="Key & Edge Scale",
            title_fontsize=8.5,
            borderpad=0.7,
            labelspacing=0.45,
            handlelength=1.8,
            handletextpad=0.5,
            columnspacing=1.2,
            ncol=4,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.04),
        )

        # report γ instead of CV-selected α
        _gamma_str = (
            f"γ = {_EBIC_GAMMA}, λ = {_best_alpha:.6f}"
            if _best_alpha
            else "Pseudoinverse fallback -- EBICglasso failed"
        )

        _ei_sorted = sorted(_EI.items(), key=lambda kv: abs(kv[1]), reverse=True)
        _render_centrality_sidebar(ax_ei, _ei_sorted, _label_of_feat, _feat_colors, _biv_r, _imp_of_feat)

        _suptitle_method = f"EBICglasso, {_gamma_str}" if _best_alpha else "Pseudoinverse fallback"
        fig2.suptitle(
            f"{_analysis_title}\nPartial Correlation Network  ({_suptitle_method})",
            fontsize=12,
            fontweight="bold",
            color=_CLR_SLATE,
            x=0.50,
            y=0.97,
            ha="center",
        )

        _save_fig(fig2, "cross_feature_partial_network", facecolor="white")
        plt.show()

        _print_network_caption(
            _analysis_title,
            len(_cross_df),
            n,
            _gamma_str,
            _layout_lbl,
            _nstats,
            _pred_r2,
            target_lbl=_target_lbl,
            nstats_full=_nstats_full,
        )

        # Partial Correlation Heatmap
        _pcor_ord = _pcor_df.iloc[_order, _order]
        _fw2 = n * (_cell_px / 100) + _lbl_inch + 2.8
        _fh2 = n * (_cell_px / 100) + 3.5
        fig2b, ax2b = plt.subplots(figsize=(_fw2, _fh2))
        fig2b.patch.set_facecolor("white")
        _im2b = ax2b.imshow(_pcor_ord.values, cmap=_CORR_CMAP, vmin=-0.5, vmax=0.5, aspect="auto")
        for i in range(n):
            for j in range(i + 1, n):
                ax2b.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec="white", lw=0, zorder=2))
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    continue
                v = _pcor_ord.values[i, j]
                if abs(v) > 0.01:
                    tc = "white" if abs(v) > 0.22 else _CLR_CHARCOAL
                    ax2b.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        color=tc,
                        fontsize=_fs_ann,
                        fontweight="bold" if abs(v) > 0.10 else "normal",
                        zorder=3,
                    )
        ax2b.set_xticks(range(n))
        ax2b.set_yticks(range(n))
        ax2b.set_xticklabels(_lbl_ord, rotation=45, ha="right", fontsize=_tf, color=_CLR_CHARCOAL)
        ax2b.set_yticklabels(_lbl_ord, fontsize=_tf, color=_CLR_CHARCOAL)
        ax2b.tick_params(axis="both", length=0, pad=8)
        cb2 = fig2b.colorbar(_im2b, ax=ax2b, shrink=0.52, aspect=20, pad=0.03, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
        cb2.set_label("Partial r (EBICglasso)", fontsize=10, color=_CLR_CHARCOAL, labelpad=8)
        cb2.ax.tick_params(labelsize=9, colors=_CLR_CHARCOAL)
        cb2.outline.set_visible(False)
        for sp in ax2b.spines.values():
            sp.set_visible(False)
        ax2b.grid(False)
        ax2b.set_title(
            f"{_analysis_title}\nClustered Partial Correlation Matrix  (EBICglasso, γ={_EBIC_GAMMA})  N = {len(_cross_df):,}",
            fontsize=12,
            fontweight="bold",
            color=_CLR_SLATE,
            pad=16,
        )
        plt.tight_layout()
        _save_fig(fig2b, "cross_feature_partial_heatmap")
        plt.show()

    # 6B. BOOTSTRAP CIs (optional)
    _boot_results = None
    if _pcor_df is not None and ctx.run_bootstrap_CI:
        logger.info("\n" + "=" * 72)
        logger.info(f"  Bootstrap Edge-Weight Stability  ({ctx.n_bootstrap:,} resamples, EBICglasso)")
        logger.info("=" * 72)
        try:
            _boot_results = _bootstrap_edge_weights(
                _cross_df.values, _feat_list, n_boot=ctx.n_bootstrap, seed=42, gamma=_EBIC_GAMMA
            )
            logger.info(f"  Converged: {_boot_results['n_converged']}/{ctx.n_bootstrap}")
            _edge_data = []
            for _i in range(n):
                for _j in range(_i + 1, n):
                    _w = _pcor_df.iloc[_i, _j]
                    _lo = _boot_results["ci_lower"].iloc[_i, _j]
                    _hi = _boot_results["ci_upper"].iloc[_i, _j]
                    _ip = _boot_results["inclusion_prob"].iloc[_i, _j]
                    if abs(_w) > 0.001 or _ip > 0.10:
                        _li = _label_of_feat.get(_feat_list[_i], _feat_list[_i])[:15]
                        _lj = _label_of_feat.get(_feat_list[_j], _feat_list[_j])[:15]
                        _edge_data.append(
                            {"label": f"{_li}--{_lj}", "weight": _w, "ci_lo": _lo, "ci_hi": _hi, "incl_prob": _ip}
                        )
            if _edge_data:
                _edge_data.sort(key=lambda x: x["weight"])
                fig_b, ax_b = plt.subplots(figsize=(10, max(6, len(_edge_data) * 0.35 + 2)))
                fig_b.patch.set_facecolor("white")
                for _ei, _ed in enumerate(_edge_data):
                    _c = _EDGE_POS if _ed["weight"] > 0 else _EDGE_NEG
                    ax_b.errorbar(
                        _ed["weight"],
                        _ei,
                        xerr=[[_ed["weight"] - _ed["ci_lo"]], [_ed["ci_hi"] - _ed["weight"]]],
                        fmt="o",
                        color=_c,
                        alpha=0.4 + 0.5 * _ed["incl_prob"],
                        markersize=6,
                        capsize=3,
                        capthick=1.2,
                        elinewidth=1.5,
                        zorder=2,
                    )
                ax_b.axvline(0, color=_CLR_SILVER, lw=1, ls="--", zorder=1)
                ax_b.set_yticks(range(len(_edge_data)))
                ax_b.set_yticklabels([d["label"] for d in _edge_data], fontsize=8.5, color=_CLR_CHARCOAL)
                ax_b.set_xlabel("Partial Correlation (95% Bootstrap CI)", fontsize=11, color=_CLR_CHARCOAL)
                ax_b.set_title(
                    f"{_analysis_title}\nEdge-Weight Bootstrap ({_boot_results['n_converged']}/{ctx.n_bootstrap:,}, EBICglasso γ={_EBIC_GAMMA})",
                    fontsize=12,
                    fontweight="bold",
                    color=_CLR_SLATE,
                    pad=14,
                )
                ax_b.spines["top"].set_visible(False)
                ax_b.spines["right"].set_visible(False)
                ax_b.grid(axis="x", linestyle=":", alpha=0.4, color=_CLR_SILVER)
                plt.tight_layout()
                _save_fig(fig_b, "cross_feature_bootstrap_CI")
                plt.show()
        except Exception as _e:
            logger.info(f"  Warning:  Bootstrap failed: {_e}")

    # 6C. Centrality stability (optional)
    _cs_ei = _cs_str = None
    if _pcor_df is not None and ctx.run_centrality_stability:
        logger.info("\n" + "=" * 72)
        logger.info(f"  Centrality Stability  (CS coefficient, {ctx.n_bootstrap:,} resamples, EBICglasso)")
        logger.info("=" * 72)
        try:
            # centrality computed from thresholded graph
            _cs_ei = _centrality_stability(
                _cross_df.values,
                _feat_list,
                lambda pdf, fnames: _compute_EI_thresholded(pdf, fnames, thresh=1e-6),
                n_boot=ctx.n_bootstrap,
                seed=42,
                gamma=_EBIC_GAMMA,
            )
            _cs_str = _centrality_stability(
                _cross_df.values,
                _feat_list,
                lambda pdf, fnames: _compute_strength_thresholded(pdf, fnames, thresh=1e-6),
                n_boot=ctx.n_bootstrap,
                seed=42,
                gamma=_EBIC_GAMMA,
            )
            logger.info(
                f"  CS(EI)       = {_cs_ei['CS']:.2f}  {'stable' if _cs_ei['CS'] >= 0.50 else 'Warning: marginal' if _cs_ei['CS'] >= 0.25 else 'Error: unstable'}"
            )
            logger.info(
                f"  CS(Strength) = {_cs_str['CS']:.2f}  {'stable' if _cs_str['CS'] >= 0.50 else 'Warning: marginal' if _cs_str['CS'] >= 0.25 else 'Error: unstable'}"
            )

            if _cs_ei["proportions"] and _cs_str["proportions"]:
                fig_cs, ax_cs = plt.subplots(figsize=(8, 5))
                fig_cs.patch.set_facecolor("white")

                _ei_avg = np.array(_cs_ei["avg_cors"], dtype=float)
                _ei_sd = np.array(_cs_ei["sd_cors"], dtype=float)
                _str_avg = np.array(_cs_str["avg_cors"], dtype=float)
                _str_sd = np.array(_cs_str["sd_cors"], dtype=float)
                _props = np.array(_cs_ei["proportions"])

                ax_cs.plot(
                    _props, _ei_avg, "o-", color=_EDGE_POS, lw=2, markersize=5, label=f"EI  (CS = {_cs_ei['CS']:.2f})"
                )
                _ei_valid = ~np.isnan(_ei_avg) & ~np.isnan(_ei_sd)
                ax_cs.fill_between(
                    _props[_ei_valid],
                    (_ei_avg - _ei_sd)[_ei_valid],
                    (_ei_avg + _ei_sd)[_ei_valid],
                    alpha=0.15,
                    color=_EDGE_POS,
                )

                ax_cs.plot(
                    _props,
                    _str_avg,
                    "s-",
                    color=_EDGE_NEG,
                    lw=2,
                    markersize=5,
                    label=f"Strength  (CS = {_cs_str['CS']:.2f})",
                )
                _str_valid = ~np.isnan(_str_avg) & ~np.isnan(_str_sd)
                ax_cs.fill_between(
                    _props[_str_valid],
                    (_str_avg - _str_sd)[_str_valid],
                    (_str_avg + _str_sd)[_str_valid],
                    alpha=0.15,
                    color=_EDGE_NEG,
                )

                ax_cs.axhline(0.7, color=_CLR_SILVER, ls="--", lw=1, label="cor = 0.70")
                ax_cs.set_xlabel("Proportion Dropped", fontsize=11, color=_CLR_CHARCOAL)
                ax_cs.set_ylabel("Avg Correlation w/ Original", fontsize=11, color=_CLR_CHARCOAL)
                ax_cs.set_ylim(0, 1.05)
                ax_cs.set_xlim(_props[0] - 0.02, _props[-1] + 0.02)
                ax_cs.legend(fontsize=9.5, framealpha=0.95, edgecolor="#CFD8DC")
                ax_cs.set_title(
                    f"{_analysis_title}\nCentrality Stability (case-dropping bootstrap, EBICglasso)",
                    fontsize=12,
                    fontweight="bold",
                    color=_CLR_SLATE,
                    pad=14,
                )
                ax_cs.spines["top"].set_visible(False)
                ax_cs.spines["right"].set_visible(False)
                ax_cs.grid(axis="y", linestyle=":", alpha=0.4, color=_CLR_SILVER)
                plt.tight_layout()
                _save_fig(fig_cs, "cross_feature_centrality_stability")
                plt.show()
        except Exception as _e:
            logger.info(f"  Warning:  CS failed: {_e}")

    # 7. Dendrogram
    logger.info("\n" + "=" * 72)
    logger.info("  Figure 3 -- Dendrogram")
    logger.info("=" * 72)

    _dend_labels = [_label_of_feat.get(_feat_list[i], _feat_list[i]) for i in range(n)]
    fig3, ax3 = plt.subplots(figsize=(max(13, n * 1.1 + 2), 7))
    fig3.patch.set_facecolor("white")
    ax3.set_facecolor("white")
    _dend = dendrogram(
        _linkage,
        labels=_dend_labels,
        leaf_rotation=45,
        leaf_font_size=max(8.5, min(11, 130 / max(n, 1))),
        color_threshold=0.65 * _linkage[:, 2].max(),
        above_threshold_color=_CLR_SILVER,
        ax=ax3,
    )
    for tl in ax3.get_xticklabels():
        for fi in _feat_list:
            if _label_of_feat.get(fi, fi) == tl.get_text():
                tl.set_color(_feat_colors.get(fi, _CLR_CHARCOAL))
                tl.set_fontweight(600)
                break
    ax3.set_ylabel("Correlation Distance (1−|r|)", fontsize=11, color=_CLR_CHARCOAL, labelpad=8)
    ax3.set_title(
        f"{_analysis_title}\nFeature Clustering by Correlation Distance",
        fontsize=12,
        fontweight="bold",
        color=_CLR_SLATE,
        pad=14,
    )
    ax3.spines["left"].set_color(_CLR_SILVER)
    ax3.spines["bottom"].set_color(_CLR_SILVER)
    ax3.yaxis.grid(True, linestyle=":", linewidth=0.5, color=_CLR_SILVER, alpha=0.6)
    ax3.set_axisbelow(True)
    _yb = -0.03 * _linkage[:, 2].max()
    _lxs = sorted(ax3.get_xticks())
    for xi, li in enumerate(_dend["leaves"]):
        fi = _feat_list[li]
        c = _feat_colors.get(fi, "#999")
        if xi < len(_lxs):
            ax3.plot(_lxs[xi], _yb, "s", color=c, markersize=9, clip_on=False, zorder=5)
    plt.tight_layout()
    _save_fig(fig3, "cross_feature_dendrogram")
    plt.show()

    # 8. Summary table
    logger.info("\n" + "=" * 72)
    logger.info("  Summary Table")
    logger.info("=" * 72)

    _biv_r_target = {}
    if _has_train:
        _ya_series = pd.Series(np.array(ctx.y_train).ravel(), index=ctx.X_train.index)
        _ya_cross = _ya_series.reindex(_cross_df.index).values
        for fi in _feat_list:
            _x = _cross_df[fi].values
            _valid = ~(np.isnan(_x) | np.isnan(_ya_cross))
            if _valid.sum() > 10:
                from scipy.stats import pearsonr

                r, p = pearsonr(_x[_valid], _ya_cross[_valid])
                _biv_r_target[fi] = (r, p)
            else:
                _biv_r_target[fi] = (np.nan, np.nan)

    _rows = []
    for fi in _feat_list:
        cat = _cat_of_feat.get(fi, "--")
        imp = _imp_of_feat.get(fi, np.nan)
        if fi in _corr.index:
            oth = _corr.loc[fi].drop(fi, errors="ignore").abs()
            mr = float(oth.mean())
            mxr = float(oth.max())
            mxp = _label_of_feat.get(oth.idxmax(), oth.idxmax())
        else:
            mr = mxr = np.nan
            mxp = "--"
        pcmx = np.nan
        pcpt = "--"
        if _pcor_df is not None and fi in _pcor_df.index:
            rpc = _pcor_df.loc[fi].drop(fi, errors="ignore").abs()
            if len(rpc) > 0:
                pcmx = float(rpc.max())
                pcpt = _label_of_feat.get(rpc.idxmax(), rpc.idxmax())
        ei = _EI.get(fi, np.nan) if _pcor_df is not None else np.nan
        st = _strength.get(fi, np.nan) if _pcor_df is not None else np.nan
        r2 = _pred_r2.get(fi, np.nan) if _pcor_df is not None else np.nan
        br, _bp = _biv_r_target.get(fi, (np.nan, np.nan))
        _rows.append(
            {
                "Category/Rank": _abbrev_cat(cat) if _is_within else cat,
                "Feature": _label_of_feat.get(fi, fi),
                "Imp.": imp,
                "r(target)": br,
                "EI": ei,
                "Str": st,
                "R²": r2,
                "Mean|r|": mr,
                "Max|r|": mxr,
                "Strongest Biv.": mxp,
                "Max|pcor|": pcmx,
                "Strongest Cond.": pcpt,
            }
        )

    _sdf = pd.DataFrame(_rows)
    _nc = ["Imp.", "r(target)", "EI", "Str", "R²", "Mean|r|", "Max|r|", "Max|pcor|"]
    _sty = (
        _sdf.style.format({c: "{:.4f}" for c in _nc})
        .background_gradient(cmap="YlOrRd", subset=["Imp."], vmin=0)
        .background_gradient(cmap="RdBu_r", subset=["r(target)"], vmin=-0.3, vmax=0.3)
        .background_gradient(cmap="RdBu_r", subset=["EI"], vmin=-0.5, vmax=0.5)
        .background_gradient(cmap="Blues", subset=["Str"], vmin=0, vmax=1)
        .background_gradient(cmap="Greens", subset=["R²"], vmin=0, vmax=0.5)
        .background_gradient(cmap="Blues", subset=["Mean|r|", "Max|r|"], vmin=0, vmax=0.5)
        .background_gradient(cmap="Purples", subset=["Max|pcor|"], vmin=0, vmax=0.3)
        .set_caption(f"Cross-Feature Summary -- {_target_lbl} (T{_tp_lbl})")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [("font-size", "14px"), ("font-weight", "700"), ("color", _CLR_SLATE)],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#EEF2F7"),
                        ("color", _CLR_CHARCOAL),
                        ("font-size", "10px"),
                        ("font-weight", "600"),
                        ("text-transform", "uppercase"),
                        ("padding", "6px 8px"),
                    ],
                },
                {"selector": "td", "props": [("padding", "5px 8px"), ("font-size", "11px")]},
            ]
        )
        .hide(axis="index")
    )
    display(_sty)

    _gamma_str = (
        f"γ = {_EBIC_GAMMA}, λ = {_best_alpha:.6f}" if _best_alpha else "Pseudoinverse fallback -- EBICglasso failed"
    )
    logger.info("\n Cross-feature analysis complete.")
    logger.info(f"   • {n} features · Layout: {_layout_lbl}" if _pcor_df is not None else f"   • {n} features")
    if _pcor_df is not None:
        logger.info(f"   • EBICglasso ({_gamma_str})")
        logger.info(f"   • Full model: {_nstats_full['n_edges']} edges (density = {_nstats_full['density']:.3f})")
        logger.info(
            f"   • Displayed (|pcor| > {_DISPLAY_THRESH}): {_nstats['n_edges']} edges (density = {_nstats['density']:.3f})"
        )
        logger.info("   • Centrality: EI (from full EBICglasso model, sorted by |EI|)")
        logger.info("   • r(target): bivariate Pearson r between each feature and target (same N as network)")
        logger.info(
            f"   • Predictability (precision-implied, sample cov): avg R² = {np.mean(list(_pred_r2.values())):.3f}"
        )
        logger.info(f"   • Edge scaling: magnitude-based, fixed maximum = {_EDGE_MAXIMUM} (cross-figure comparable)")
        logger.info(f"   • n/p ratio: {_np_ratio:.0f}")
    if _boot_results:
        logger.info(f"   • Bootstrap: {_boot_results['n_converged']}/{ctx.n_bootstrap}, 95% CIs (EBICglasso)")
    elif not ctx.run_bootstrap_CI:
        logger.info("   • Bootstrap: disabled (networks.run_bootstrap_CI)")
    if _cs_ei:
        logger.info(f"   • CS(EI) = {_cs_ei['CS']:.2f}, CS(Str) = {_cs_str['CS']:.2f}")
    elif not ctx.run_centrality_stability:
        logger.info("   • Centrality stability: disabled (networks.run_centrality_stability)")
    logger.info("   • Exports: PNG (300 DPI) + PDF (vector)")


# Consensus helpers: called after all batch runs complete


def _plot_consensus_pcor_network(
    batch_pcor_dict, feat_list, label_of_feat, imp_of_feat, feat_colors, thresh=None, batch_corr_dict=None
):
    """Consensus partial-correlation network across batches via Fisher-z mean.

    R² now computed via precision-implied method by inverting
    the Fisher-z averaged correlation matrix, making R² comparable with
    per-target networks.
    """
    import networkx as nx

    if not batch_pcor_dict:
        logger.info("No batch-specific partial-correlation matrices found.")
        return

    mats = []
    for _, pcor in batch_pcor_dict.items():
        p = pcor.values.copy()
        p = np.clip(p, -0.9999, 0.9999)
        mats.append(np.arctanh(p))
    z_mean = np.nanmean(np.stack(mats, axis=0), axis=0)
    r_mean = np.tanh(z_mean)
    np.fill_diagonal(r_mean, 1.0)
    pcor_cons = pd.DataFrame(r_mean, index=feat_list, columns=feat_list)

    n = len(feat_list)

    # Full graph (all nonzero consensus edges) for EI
    G_full = nx.Graph()
    for f in feat_list:
        G_full.add_node(f, label=label_of_feat.get(f, f), color=feat_colors.get(f, "#888"), imp=imp_of_feat.get(f, 1.0))
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = feat_list[i], feat_list[j]
            pc = pcor_cons.loc[fi, fj]
            if abs(pc) > 1e-6:
                G_full.add_edge(fi, fj, weight=pc, aw=abs(pc))

    EI = _compute_EI_from_graph(G_full, feat_list)
    nstats_full = _network_stats(G_full)

    # Display graph (|pcor| > _DISPLAY_THRESH)
    G = nx.Graph()
    for f in feat_list:
        G.add_node(f, label=label_of_feat.get(f, f), color=feat_colors.get(f, "#888"), imp=imp_of_feat.get(f, 1.0))
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = feat_list[i], feat_list[j]
            pc = pcor_cons.loc[fi, fj]
            if abs(pc) > _DISPLAY_THRESH:
                G.add_edge(fi, fj, weight=pc, aw=abs(pc))

    if G.number_of_edges() == 0:
        logger.info(f"No edges above |pcor| > {_DISPLAY_THRESH} in consensus network.")
        return

    nstats = _network_stats(G)
    logger.info(f"  Consensus model: {nstats_full['n_edges']} edges (density = {nstats_full['density']:.3f})")
    logger.info(f"  Displayed (|pcor| > {_DISPLAY_THRESH}): {nstats['n_nodes']} nodes, {nstats['n_edges']} edges")

    for f in feat_list:
        if f in G.nodes:
            G.nodes[f]["ei"] = EI.get(f, 0.0)

    _circle_scale = 5.5 + n * 0.25
    pos = _fiedler_circle_layout(G, scale=_circle_scale)
    node_list = list(G.nodes())
    imp_arr = np.array([G.nodes[nd]["imp"] for nd in node_list])
    imp_rng = max(imp_arr.max() - imp_arr.min(), 1e-9)

    xs = [pos[nd][0] for nd in node_list]
    ys = [pos[nd][1] for nd in node_list]
    data_span = max(max(xs) - min(xs), max(ys) - min(ys), 0.01)
    base_r = data_span * 0.032
    radii = [base_r * (0.4 + 2.6 * ((imp_arr[i] - imp_arr.min()) / imp_rng) ** 0.5) for i in range(len(node_list))]

    # precision-implied R² for consensus network
    # Invert the Fisher-z averaged correlation matrix to get an approximate
    # precision matrix, making R² comparable with per-target networks
    # (Haslbeck & Waldorp 2018).
    pred_r2 = None
    if batch_corr_dict and len(batch_corr_dict) > 0:
        try:
            corr_mats = []
            for _, corr in batch_corr_dict.items():
                c = corr.values.copy()
                c = np.clip(c, -0.9999, 0.9999)
                corr_mats.append(np.arctanh(c))
            z_corr_mean = np.nanmean(np.stack(corr_mats, axis=0), axis=0)
            r_corr_mean = np.tanh(z_corr_mean)
            np.fill_diagonal(r_corr_mean, 1.0)
            # For standardized data, Θ = R⁻¹.
            # Ridge regularization (1e-4) stabilises inversion of a potentially
            # near-singular averaged correlation matrix, preventing artifactual
            # R² ≈ 1.0 from inflated precision diagonals (pinv can "succeed"
            # numerically while producing garbage values).
            _n_feat = len(feat_list)
            r_corr_mean_reg = r_corr_mean + 1e-4 * np.eye(_n_feat)
            prec_cons = np.linalg.inv(r_corr_mean_reg)
            cov_cons = r_corr_mean  # correlation = covariance for standardized data
            pred_r2 = _compute_predictability(feat_list, prec=prec_cons, cov=cov_cons)
            # Sanity check: if ridge-regularised inversion still produces
            # implausible R² (e.g. > 0.99 for any node), fall back to OLS
            _max_r2 = max(pred_r2.values()) if pred_r2 else 0
            if _max_r2 > 0.99:
                logger.info(
                    f"  Warning:  Consensus precision inversion unstable (max R² = {_max_r2:.3f}), falling back to OLS"
                )
                pred_r2 = None
            else:
                logger.info("  Predictability: precision-implied from consensus correlation (comparable w/ per-target)")
        except Exception as _e:
            logger.info(f"  Warning:  Precision-implied R² failed for consensus ({_e}), falling back to OLS")
            pred_r2 = None

    if pred_r2 is None:
        # Fallback: OLS (only if correlation-based method fails)
        cross_df = ctx.X_train[feat_list].replace([np.inf, -np.inf], np.nan).dropna()
        pred_r2 = _compute_predictability(feat_list, cross_df=cross_df, pcor_df=pcor_cons, thresh=1e-6)

    border_colors = []
    border_widths = []
    ei_arr = np.array([abs(EI[nd]) for nd in node_list])
    ei_rng = max(ei_arr.max(), 1e-9)
    for i, nd in enumerate(node_list):
        eiv = EI[nd]
        if abs(eiv) < 0.01:
            border_colors.append(_BORDER_ZERO)
        elif eiv > 0:
            border_colors.append(_BORDER_POS)
        else:
            border_colors.append(_BORDER_NEG)
        # floor raised 1.2→2.0pt
        border_widths.append(2.0 + 8.0 * (ei_arr[i] / ei_rng))

    net_w = max(11, n * 1.10)
    ei_w = max(5.5, n * 0.46)
    fig_h = max(10, n * 1.05)

    fig = plt.figure(figsize=(net_w + ei_w + 1.0, fig_h))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[net_w, ei_w], wspace=0.14, left=0.01, right=0.99, top=0.91, bottom=0.16
    )
    ax_net = fig.add_subplot(gs[0, 0])
    ax_ei = fig.add_subplot(gs[0, 1])

    ax_net.set_facecolor(_NET_BG)
    ax_net.grid(False)
    ax_net.axis("off")

    # Edges (shared helper)
    _edge_midpoints = _render_edges(ax_net, G, pos, node_list, data_span)

    import textwrap as _tw

    _cx = np.mean(xs)
    _cy = np.mean(ys)
    _off = data_span * 0.40
    _fs_lbl = max(10, min(13, 16 - n * 0.14))
    _ww = 22
    _max_radius = max(radii) if radii else 1.0
    _node_idx = {nd: i for i, nd in enumerate(node_list)}

    # Draw nodes first
    for i, nd in enumerate(node_list):
        x, y = pos[nd]
        _draw_predictability_node(
            ax_net, x, y, radii[i], pred_r2.get(nd, 0.0), G.nodes[nd]["color"], border_colors[i], border_widths[i]
        )

    # collect initial radial positions, then repel
    _init_lbl_pos = []
    _node_anchors = []
    _lbl_texts = []
    for i, nd in enumerate(node_list):
        x, y = pos[nd]
        _raw = G.nodes[nd]["label"]
        _lbl_texts.append("\n".join(_tw.wrap(_raw, width=_ww)))
        _dx = x - _cx
        _dy = y - _cy
        _d = max(np.sqrt(_dx**2 + _dy**2), 0.001)
        _init_lbl_pos.append([x + _off * _dx / _d, y + _off * _dy / _d])
        _node_anchors.append((x + _off * 0.5 * _dx / _d, y + _off * 0.5 * _dy / _d))

    _repelled = _repel_labels(_init_lbl_pos, _node_anchors, min_sep=data_span * 0.12, node_pull=0.12, iterations=80)

    for i, nd in enumerate(node_list):
        x, y = pos[nd]
        _lx, _ly = _repelled[i]
        _shrinkA = 10 + 8 * (radii[i] / _max_radius)
        ax_net.annotate(
            _lbl_texts[i],
            xy=(x, y),
            xytext=(_lx, _ly),
            fontsize=_fs_lbl,
            color=_CLR_SLATE,
            fontweight=600,
            ha="center",
            va="center",
            clip_on=False,
            arrowprops=dict(arrowstyle="-", color=_ARROW_CLR, lw=0.9, shrinkA=_shrinkA, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.93, ec=feat_colors.get(nd, _CLR_SILVER), lw=1.5),
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white"), pe.Normal()],
            zorder=7,
        )

    _pad = data_span * 0.30 + 2.2
    ax_net.set_xlim(min(xs) - _pad, max(xs) + _pad)
    ax_net.set_ylim(min(ys) - _pad, max(ys) + _pad)

    # Network key + edge scale (bottom-left)
    _legend_items = [
        plt.Line2D([0], [0], color=_EDGE_POS, lw=2.0, linestyle="-", label="Positive pcor (solid)"),
        plt.Line2D([0], [0], color=_EDGE_NEG, lw=2.0, linestyle=_DASH_NEG, label="Negative pcor (dashed)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_RING_UNFILLED,
            markeredgecolor=_BORDER_POS,
            markeredgewidth=2.5,
            markersize=10,
            label="Ring fill = R² predictability",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=_BORDER_POS,
            markeredgewidth=2.5,
            markersize=10,
            label="Blue border = +EI, orange = −EI",
        ),
    ]
    for _sr in [0.10, 0.20, 0.30]:
        _t_s = min(_sr / _EDGE_MAXIMUM, 1.0) ** 0.6
        _sw = max(_LW_MIN + (_LW_MAX - _LW_MIN) * _t_s, 0.4)
        _sa = max(_ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * _t_s, 0.10)
        _sl = f"|r| = {_sr:.2f}+" if _sr == 0.30 else f"|r| = {_sr:.2f}"
        _legend_items.append(
            plt.Line2D([0], [0], color=_EDGE_POS, lw=_sw, alpha=_sa, solid_capstyle="round", label=_sl)
        )
    fig.legend(
        handles=_legend_items,
        fontsize=8.5,
        framealpha=0.92,
        edgecolor="#CFD8DC",
        fancybox=True,
        title="Key & Edge Scale",
        title_fontsize=10,
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.2,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.04),
    )

    n_batch = len(batch_pcor_dict)
    fig.suptitle(
        f"Consensus Partial Correlation Network\n"
        f"(Fisher-z averaged across {n_batch} targets, EBICglasso γ={_EBIC_GAMMA})",
        fontsize=12,
        fontweight="bold",
        color=_CLR_SLATE,
        x=0.50,
        y=0.97,
        ha="center",
    )

    ei_sorted_cons = sorted(EI.items(), key=lambda kv: abs(kv[1]), reverse=True)
    # pass None: consensus has no single target for bivariate r
    _render_centrality_sidebar(ax_ei, ei_sorted_cons, label_of_feat, feat_colors, None, imp_of_feat)

    _save_fig(fig, "consensus_partial_correlation_network")
    plt.show()
    logger.info(f"  Avg R² (predictability) = {np.mean(list(pred_r2.values())):.3f}")

    # consensus network was missing Burger et al. caption
    _cons_n_subj = ctx.X_train[feat_list].replace([np.inf, -np.inf], np.nan).dropna().shape[0]
    _cons_gamma_str = f"γ = {_EBIC_GAMMA} (consensus Fisher-z averaged, {n_batch} targets)"
    _print_network_caption(
        f"Consensus Network ({n_batch} targets)",
        _cons_n_subj,
        n,
        _cons_gamma_str,
        "Spectral (Fiedler)",
        nstats,
        pred_r2,
        target_lbl="All targets (consensus)",
        nstats_full=nstats_full,
    )
    # disclose that display threshold acts as sparsity mechanism
    # on the Fisher-z averaged matrix (individual EBICglasso models are sparse,
    # but averaging fills in zeros → dense consensus matrix)
    logger.info("  | NOTE (consensus-specific): Individual per-target partial correlation")
    logger.info("  |   matrices are EBICglasso-sparse, but Fisher-z averaging across targets")
    logger.info("  |   produces non-zero entries where not all targets zero the same edges.")
    logger.info(f"  |   Display threshold |pcor| > {_DISPLAY_THRESH} therefore acts as the")
    logger.info("  |   effective sparsity mechanism for this consensus network.")


def _plot_consensus_dendrogram(batch_corr_dict, feat_list, label_of_feat, feat_colors):
    """Consensus dendrogram from Fisher-z averaged Pearson correlations."""
    if not batch_corr_dict:
        logger.info("No batch-specific correlation matrices found.")
        return

    mats = []
    for _, corr in batch_corr_dict.items():
        c = corr.values.copy()
        c = np.clip(c, -0.9999, 0.9999)
        mats.append(np.arctanh(c))
    z_mean = np.nanmean(np.stack(mats, axis=0), axis=0)
    r_mean = np.tanh(z_mean)
    np.fill_diagonal(r_mean, 1.0)

    corr_cons = pd.DataFrame(r_mean, index=feat_list, columns=feat_list)
    dist = (1 - corr_cons.abs()).clip(lower=0).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)

    _link = linkage(squareform(dist, checks=False), method="average")
    leaves = dendrogram(_link, no_plot=True)["leaves"]
    feat_ord = [feat_list[i] for i in leaves]
    lbl_ord = [label_of_feat.get(f, f) for f in feat_ord]
    corr_ord = corr_cons.loc[feat_ord, feat_ord]

    n = len(feat_ord)
    n_batch = len(batch_corr_dict)

    cell_px = 72
    max_llen = max(len(l) for l in lbl_ord)
    lbl_inch = max(max_llen * 0.078, 1.8)
    fw = n * (cell_px / 100) + lbl_inch + 2.8
    fh = n * (cell_px / 100) + 3.5

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor("white")
    im = ax.imshow(corr_ord.values, cmap=_CORR_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(n):
        for j in range(i + 1, n):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec="white", lw=0, zorder=2))
    fs_ann = max(7, min(11, 14 - n * 0.35))
    for i in range(n):
        for j in range(i + 1):
            v = corr_ord.values[i, j]
            tc = "white" if abs(v) > 0.50 else _CLR_CHARCOAL
            fwgt = "bold" if abs(v) > 0.28 else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=tc, fontsize=fs_ann, fontweight=fwgt, zorder=3)
    tf = max(8.5, min(11, 130 / max(n, 1)))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(lbl_ord, rotation=45, ha="right", fontsize=tf, color=_CLR_CHARCOAL)
    ax.set_yticklabels(lbl_ord, fontsize=tf, color=_CLR_CHARCOAL)
    ax.tick_params(axis="both", length=0, pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.52, aspect=20, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(f"Pearson r (consensus across {n_batch} targets)", fontsize=10, color=_CLR_CHARCOAL, labelpad=8)
    cb.ax.tick_params(labelsize=9, colors=_CLR_CHARCOAL)
    cb.outline.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_title(
        f"Consensus Clustered Pearson Correlation Matrix\n(Fisher-z averaged across {n_batch} targets)",
        fontsize=12,
        fontweight="bold",
        color=_CLR_SLATE,
        pad=16,
    )
    plt.tight_layout()
    _save_fig(fig, "consensus_correlation_matrix_heatmap")
    plt.show()

    _dend_labels = [label_of_feat.get(feat_list[i], feat_list[i]) for i in range(n)]
    fig3, ax3 = plt.subplots(figsize=(max(13, n * 1.1 + 2), 7))
    fig3.patch.set_facecolor("white")
    ax3.set_facecolor("white")
    _dend = dendrogram(
        _link,
        labels=_dend_labels,
        leaf_rotation=45,
        leaf_font_size=max(8.5, min(11, 130 / max(n, 1))),
        color_threshold=0.65 * _link[:, 2].max(),
        above_threshold_color=_CLR_SILVER,
        ax=ax3,
    )
    for tl in ax3.get_xticklabels():
        for fi in feat_list:
            if label_of_feat.get(fi, fi) == tl.get_text():
                tl.set_color(feat_colors.get(fi, _CLR_CHARCOAL))
                tl.set_fontweight(600)
                break
    ax3.set_ylabel("Correlation Distance (1−|r|)", fontsize=11, color=_CLR_CHARCOAL, labelpad=8)
    ax3.set_title(
        f"Consensus Feature Dendrogram\n(Fisher-z averaged across {n_batch} targets)",
        fontsize=12,
        fontweight="bold",
        color=_CLR_SLATE,
        pad=14,
    )
    ax3.spines["left"].set_color(_CLR_SILVER)
    ax3.spines["bottom"].set_color(_CLR_SILVER)
    ax3.yaxis.grid(True, linestyle=":", linewidth=0.5, color=_CLR_SILVER, alpha=0.6)
    ax3.set_axisbelow(True)
    _yb = -0.03 * _link[:, 2].max()
    _lxs = sorted(ax3.get_xticks())
    for xi, li in enumerate(_dend["leaves"]):
        fi = feat_list[li]
        c = feat_colors.get(fi, "#999")
        if xi < len(_lxs):
            ax3.plot(_lxs[xi], _yb, "s", color=c, markersize=9, clip_on=False, zorder=5)
    plt.tight_layout()
    _save_fig(fig3, "consensus_feature_dendrogram")
    plt.show()


# MODULE-LEVEL _display_name


def _display_name(feat):
    _dn = DISPLAY_NAMES if ("DISPLAY_NAMES" in ctx and isinstance(DISPLAY_NAMES, dict)) else {}
    return _dn.get(feat, feat)


def run() -> None:
    """Run this stage (notebook cell 9)."""
    global Counter, LinearSegmentedColormap, np, pd
    warnings.filterwarnings("ignore")
    logger.info(
        f"  Software: sklearn={sklearn.__version__}, "
        f"networkx={_nx_ver.__version__}, matplotlib={matplotlib.__version__}"
    )

    try:
        _RS["section"] = "cross_category"
    except Exception:
        pass  # non-critical; logged upstream or optional visualization

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Inter", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "font.weight": "regular",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    # Batch dispatch
    _is_bv_11 = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_11:
        try:
            _cbv = cache_load("main_runner")
            if (
                _cbv
                and isinstance(_cbv, dict)
                and _cbv.get("_batch_archive")
                and len(_cbv["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv["_batch_archive"]
                _is_bv_11 = True
        except Exception:
            pass

    if _is_bv_11:
        _vgs = {
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
            ]
        }
        logger.info(f"\n{'=' * 72}\n  BATCH -- {len(ctx._batch_archive)} targets\n{'=' * 72}")
        for _vi, (_vk, _vd) in enumerate(ctx._batch_archive.items()):
            _vl = _vd.get("target", _vk)
            logger.info(f"\n{'-' * 68}\n  [{_vi + 1}/{len(ctx._batch_archive)}] {_vl}\n{'-' * 68}")
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
                "sample_valid",
                "sample_valid_test",
                "sample_valid_train",
            ]:
                if _vd.get(_gk) is not None:
                    ctx[_gk] = _vd[_gk]
            ctx["target_options"] = _vl
            if "timepoint" in _vd:
                ctx["tp_option"] = _vd["timepoint"]
            try:
                _run_for_current_target()
            except Exception as _e:
                logger.info(f"  Warning:  {_vl}: {_e}")
        for _k, _v in _vgs.items():
            ctx[_k] = _v
        logger.info(f"\n{'=' * 72}\n  BATCH COMPLETE -- {len(ctx._batch_archive)} targets\n{'=' * 72}")

        # Consensus network & dendrogram across all batch runs
        try:
            if "batch_pcor_dict" in ctx and ctx["batch_pcor_dict"]:
                logger.info(f"\n{'=' * 72}")
                logger.info("  Consensus Partial Correlation Network  (across all targets)")
                logger.info(f"{'=' * 72}")
                _dn = DISPLAY_NAMES if ("DISPLAY_NAMES" in ctx and isinstance(DISPLAY_NAMES, dict)) else {}
                _fl = ctx.get("batch_feat_list", [])
                _lof = {f: _dn.get(f, f) for f in _fl}
                _iof = {}
                _fc = {}
                for _bv in ctx._batch_archive.values():
                    _Xt = _bv.get("X_train")
                    if _Xt is not None:
                        _fc2, _ = _get_domain_colors(_fl, {f: "#1" for f in _fl}, False)
                        _fc.update(_fc2)
                        break
                # pass batch_corr_dict for precision-implied R²
                _plot_consensus_pcor_network(
                    ctx["batch_pcor_dict"], _fl, _lof, _iof, _fc, batch_corr_dict=ctx.get("batch_corr_dict")
                )
        except Exception as _e:
            logger.info(f"  Warning:  Consensus network error: {_e}")
        try:
            if "batch_corr_dict" in ctx and ctx["batch_corr_dict"]:
                logger.info(f"\n{'=' * 72}")
                logger.info("  Consensus Dendrogram  (across all targets)")
                logger.info(f"{'=' * 72}")
                _dn = DISPLAY_NAMES if ("DISPLAY_NAMES" in ctx and isinstance(DISPLAY_NAMES, dict)) else {}
                _fl = ctx.get("batch_feat_list_corr", [])
                _lof = {f: _dn.get(f, f) for f in _fl}
                _fc = {}
                for _bv in ctx._batch_archive.values():
                    _Xt = _bv.get("X_train")
                    if _Xt is not None:
                        _fc2, _ = _get_domain_colors(_fl, {f: "#1" for f in _fl}, False)
                        _fc.update(_fc2)
                        break
                _plot_consensus_dendrogram(ctx["batch_corr_dict"], _fl, _lof, _fc)
        except Exception as _e:
            logger.info(f"  Warning:  Consensus dendrogram error: {_e}")

    else:
        _run_for_current_target()

    # Batch cross-target feature importance heatmap

    try:
        if _is_bv_11 and len(ctx._batch_archive) >= 2:
            import numpy as np
            import pandas as pd
            from matplotlib.colors import LinearSegmentedColormap

            _TOP_N_PER_TARGET = 15
            _MAX_COLS = 35

            _feat_records = {}
            _target_labels = {}
            ctx._target_metrics = {}

            for _fk, _fv in ctx._batch_archive.items():
                _tname = _fv.get("target", _fk)
                _tlabel = TARGET_DISPLAY_NAMES.get(_tname, _tname) if "TARGET_DISPLAY_NAMES" in ctx else _tname
                _target_labels[_tname] = _tlabel

                _cdf = _fv.get("consensus_df")
                if _cdf is not None and hasattr(_cdf, "columns") and len(_cdf) > 0:
                    _imp_col = None
                    for _cc in ["consensus_importance", "mean_importance", "importance"]:
                        if _cc in _cdf.columns:
                            _imp_col = _cc
                            break
                    if _imp_col is None:
                        _num_cols = _cdf.select_dtypes(include="number").columns.tolist()
                        if _num_cols:
                            _imp_col = _num_cols[0]

                    if _imp_col and "feature" in _cdf.columns:
                        _sorted_feats = _cdf.nlargest(_TOP_N_PER_TARGET, _imp_col)
                        _imp_dict = dict(zip(_sorted_feats["feature"], _sorted_feats[_imp_col]))
                        _max_imp = max(_imp_dict.values()) if _imp_dict else 1
                        if _max_imp > 0:
                            _imp_dict = {k: v / _max_imp for k, v in _imp_dict.items()}
                        _feat_records[_tname] = _imp_dict

                _mm = _fv.get("metrics_dict_full", {})
                if _mm.get("category") and _mm.get("metrics"):
                    for _ci, _mi in zip(_mm["category"], _mm["metrics"]):
                        if isinstance(_mi, dict) and "meta_model" in _mi:
                            _meta = _mi["meta_model"]
                            if isinstance(_meta, dict):
                                ctx._target_metrics[_tname] = _meta.get("auc", _meta.get("r2", 0))

            if len(_feat_records) >= 2:
                from collections import Counter

                _all_feats = Counter()
                for _imp_dict in _feat_records.values():
                    for _f in _imp_dict:
                        _all_feats[_f] += 1
                _top_feats = [f for f, _ in _all_feats.most_common(_MAX_COLS)]

                _sorted_targets = sorted(
                    _feat_records.keys(), key=lambda t: ctx._target_metrics.get(t, 0), reverse=True
                )

                _matrix = pd.DataFrame(
                    index=[_target_labels[t] for t in _sorted_targets], columns=_top_feats, dtype=float
                )
                for _tname in _sorted_targets:
                    _tlabel = _target_labels[_tname]
                    for _f in _top_feats:
                        _matrix.loc[_tlabel, _f] = _feat_records[_tname].get(_f, 0.0)

                _n_tgts = len(_sorted_targets)
                _n_feats = len(_top_feats)
                _fig_w = max(14, _n_feats * 0.45 + 4)
                _fig_h = max(6, _n_tgts * 0.5 + 2)

                fig_fi, ax_fi = plt.subplots(figsize=(_fig_w, _fig_h))

                _cmap_fi = LinearSegmentedColormap.from_list(
                    "imp_cmap", ["#F8F9FB", "#BFDBFE", "#3B82F6", "#1E3A8A"], N=256
                )

                _im_fi = ax_fi.imshow(_matrix.values.astype(float), aspect="auto", cmap=_cmap_fi, vmin=0, vmax=1)

                ax_fi.set_xticks(range(_n_feats))
                ax_fi.set_xticklabels(_top_feats, rotation=55, ha="right", fontsize=7.5)
                ax_fi.set_yticks(range(_n_tgts))
                ax_fi.set_yticklabels(_matrix.index, fontsize=9)

                for _ri in range(_n_tgts):
                    for _ci in range(_n_feats):
                        _v = _matrix.values[_ri, _ci]
                        if _v > 0.15:
                            _tc = "white" if _v > 0.55 else "#1E3A8A"
                            ax_fi.text(
                                _ci,
                                _ri,
                                f"{_v:.2f}",
                                ha="center",
                                va="center",
                                fontsize=6.5,
                                color=_tc,
                                fontweight="bold" if _v > 0.7 else "normal",
                            )

                ax_fi.set_title(
                    f"Cross-Target Feature Importance -- Top {_n_feats} Features\n"
                    f"({_n_tgts} targets, normalized 0-1 within each target)",
                    fontsize=13,
                    fontweight="bold",
                    pad=14,
                )

                _cbar = plt.colorbar(_im_fi, ax=ax_fi, shrink=0.6, pad=0.02)
                _cbar.set_label("Normalized Importance", fontsize=10)

                plt.tight_layout()
                try:
                    fig_fi.savefig(
                        f"{results_dir_summary}/batch_feature_importance_heatmap.png",  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                        dpi=250,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    fig_fi.savefig(
                        f"{results_dir_summary}/batch_feature_importance_heatmap.pdf",  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    logger.info("   Saved: batch_feature_importance_heatmap.png / .pdf")
                except Exception as _e:
                    logger.info(f"Warning: {_e}")
                plt.show()
            else:
                logger.info("  Skipped:  Cross-target feature heatmap: <2 targets with importance data")

    except Exception as _e_fi:
        logger.info(f"  Warning:  Cross-target feature heatmap error: {_e_fi}")
