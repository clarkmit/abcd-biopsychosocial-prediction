"""Equivalence of the bpsp package with the published notebook (paper/, v1.0.0).

Both are run on the same simulated panel with the same settings: the notebook cell by cell
in an IPython shell (as Colab executes it), the package stage by stage. Every shared result
(train/test matrices, metrics, batch archives, validation tables) is fingerprinted after each
step, and the two runs are compared together with all files they write.

Command line (each run in its own process)::

    python tests/regression/equivalence.py panel  OUT.csv [--subjects 1200]
    python tests/regression/equivalence.py notebook PLAN.json PANEL.csv OUTDIR
    python tests/regression/equivalence.py package  PLAN.json PANEL.csv OUTDIR
    python tests/regression/equivalence.py compare  NOTEBOOK_OUTDIR PACKAGE_OUTDIR
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
import numbers
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK = REPO / "paper" / "ABCDMLPipeline_v1.0.0.ipynb"
SIMULATED = REPO / "data" / "simulated" / "CLEAN_ABCD_5_1_panel_simulated.csv"

# Results shared between notebook cells / package stages that are compared.
KEY_VARS = [
    "results_summary",
    "metrics_dict_full",
    "_batch_archive",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "sample_valid_train",
    "sample_valid_test",
    "consensus_df",
    "model_weights",
    "category_metrics_dict",
    "_within_ncv_results",
    "_perm_results",
    "_perm_df",
    "all_rows",
    "_site_score",
    "exported_within_dict",
    "base_learner_all_models_df",
    "base_learner_by_target_df",
    "calibration_classification_df",
    "calibration_regression_df",
    "fairness_race_religion_df",
    "fairness_partial_corr_df",
    "grouped_perm_features_df",
    "grouped_perm_clusters_df",
    "grouped_perm_groups_df",
]
CELL_TO_STAGE = {
    6: "main",
    7: "batch_summary",
    8: "sankey",
    9: "networks",
    10: "diagnostics",
    11: "smoke_test",
    12: "nested_cv",
    13: "stratified",
    14: "grouped_cv",
    15: "permutation",
    16: "stacking_audit",
    17: "baselines",
    18: "ablation",
    19: "trajectories",
    20: "temporal_gradient",
    21: "fairness",
    22: "calibration",
    23: "base_learners",
    24: "grouped_importance",
}


# ----------------------------------------------------------------------------- environment
def use_stubs_if_needed() -> None:
    """Use the test stand-ins for torch/tabpfn when the real packages are not installed."""
    for name in ("torch", "tabpfn"):
        try:
            __import__(name)
        except ImportError:
            sys.path.insert(0, str(HERE / "stubs"))
            return


# ----------------------------------------------------------------------------- test panel
def build_panel(out: str | Path, n_subjects: int = 1200, seed: int = 0, src: Path = SIMULATED) -> Path:
    """Write the simulated panel with a known signal (:func:`bpsp.simulation.make_example_panel`)."""
    from bpsp.simulation import make_example_panel

    out = Path(out)
    make_example_panel(src, n_subjects=n_subjects, seed=seed).to_csv(out, index=False)
    return out


# ----------------------------------------------------------------------------- fingerprints
def _h(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]


def summarize(o, depth: int = 0):
    """JSON-comparable fingerprint of a result object (values rounded to 1e-6)."""
    if depth > 5:
        return f"<{type(o).__name__}>"
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, numbers.Number):
        x = float(o)
        return None if math.isnan(x) else round(x, 6)
    if isinstance(o, pd.DataFrame):
        num = o.select_dtypes(include=[np.number])
        return {
            "_df": list(o.shape),
            "cols": _h("|".join(map(str, o.columns)).encode()),
            "idx": _h("|".join(map(str, o.index[:2000])).encode()),
            "num": _h(np.round(num.to_numpy(dtype=float, na_value=np.nan), 6).tobytes()) if num.shape[1] else None,
            "obj": _h(o.select_dtypes(exclude=[np.number]).astype(str).to_csv().encode()),
        }
    if isinstance(o, pd.Series):
        return summarize(o.to_frame(), depth)
    if isinstance(o, np.ndarray):
        if o.dtype.kind in "fiub":
            return {"_nd": list(o.shape), "v": _h(np.round(o.astype(float), 6).tobytes())}
        return {"_nd": list(o.shape), "v": _h(str(o.tolist()).encode())}
    if isinstance(o, dict):
        return {str(k): summarize(v, depth + 1) for k, v in list(o.items())[:200]}
    if isinstance(o, (list, tuple)):
        return [summarize(v, depth + 1) for v in list(o)[:200]]
    return f"<{type(o).__name__}>"


def snapshot(namespace) -> dict:
    return {k: summarize(namespace[k]) for k in KEY_VARS if k in namespace}


def _normalize(text: str) -> str:
    text = re.sub(r"\d+(\.\d+)?\s?(s|sec|seconds|min)\b", "<T>", text)
    text = re.sub(r"run_\d+", "run_<T>", text)
    return re.sub(r"0x[0-9a-f]{6,}", "0x<A>", text)


# ----------------------------------------------------------------------------- plans
def setting(value):
    """Plan values are Python literals as they would be typed into the notebook."""
    return ast.literal_eval(value)


def _set_param(src: str, name: str, value: str) -> str:
    pat = re.compile(rf"^(\s*){re.escape(name)}\s*=\s*.*?(#\s*@param.*)?$", re.M)
    new, n = pat.subn(lambda m: f"{m.group(1)}{name} = {value}  {m.group(2) or ''}", src, count=1)
    if n != 1:
        raise KeyError(f"setting {name} not found in notebook cell")
    return new


# The two v2.0.0 changes to computed values, applied to the notebook's source before it runs so
# that both versions compute the same thing (see "Deliberate source changes" in docs/equivalence.md):
# * cell 5: adjusted R² is undefined when the test rows do not exceed the number of predictors
#   plus one, and is then reported as NaN instead of a meaningless value (the formula changes sign);
# * cell 6: the main stage takes its CatBoost settings from PIPELINE_MODEL_CONFIGS instead of a
#   copy of the same values, so hyperparameter overrides reach it. The defaults are identical.
V1_SOURCE_EDITS = {
    5: [
        (
            "    adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))",
            "    if n_samples - n_features - 1 <= 0:\n"
            "        return np.nan\n"
            "    adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))",
        )
    ],
    6: [
        (
            "        best_params = {\n"
            "              'iterations': 1000,\n"
            "              'learning_rate': 0.05,\n"
            "              'depth': 6,\n"
            "              'l2_leaf_reg': 3,\n"
            "              'border_count': 128,\n"
            "              'thread_count': -1,\n"
            "              'verbose': False\n"
            "          }\n",
            "        best_params = dict(PIPELINE_MODEL_CONFIGS['catboost'])\n",
        )
    ],
}

# Labels and messages that v2.0.0 words differently in printed output and in figure titles:
# the "[POINT n]" labels (numbered after points in an earlier review round) name the stage,
# and messages no longer say "Enhanced". Because some sit in figure titles, they are applied
# to the notebook's source before it runs, like the adjusted R² change above.
V1_TEXT_EDITS = [
    ("[POINT 1]", "[nested_cv]"),
    ("[POINT 3]", "[grouped_cv]"),
    ("[POINT 4]", "[permutation]"),
    ("[POINT 5]", "[stacking_audit]"),
    ("for Points 1-5", "for the validation stages"),
    ("Points 2 & 3 may not work.", "Validation stages that use sample_valid may not work."),
    ("Points 1-5 will not have real pipeline data.", "The validation stages will not have real pipeline data."),
    (
        "[Ensemble] Enhanced methods attached to StackingEnsembleModel",
        "[Ensemble] Meta-learner interactions and consensus methods attached to StackingEnsembleModel",
    ),
    (
        "Enhanced feature importance visualization..",
        "Plotting meta-learner coefficients (base learners, pairwise products and mean).",
    ),
    ("ENHANCED SANKEY \\u2014 Transdiagnostic Feature Flow", "SANKEY: Transdiagnostic Feature Flow"),
]


def v1_source(cell: int, src: str) -> str:
    """The notebook cell's source with the documented v2.0.0 edits applied."""
    for old, new in V1_SOURCE_EDITS.get(cell, []):
        if src.count(old) != 1:
            raise ValueError(f"source edit for cell {cell} does not match exactly once: {old!r}")
        src = src.replace(old, new)
    for old, new in V1_TEXT_EDITS:
        src = src.replace(old, new)
    return src


# IPython prints a cell's traceback after a line of dashes (with color codes).
_IPYTHON_TRACEBACK = re.compile(r"^(?:\x1b\[[0-9;]*m)*-{20,}", re.M)


def _seed_global_rngs() -> None:
    """Seed the global random generators before every step, identically in both runs.

    A few v1.0.0 figures draw from NumPy's global generator, which the notebook never seeds
    (jitter in the nested-CV fold plot, Kernel SHAP sampling). Seeding it the same way before
    each step makes those figures comparable byte for byte; it does not touch the pipeline's
    own seeded random states.
    """
    import random

    random.seed(0)
    np.random.seed(0)


def run_notebook(plan: list[dict], panel: str, outdir: str) -> None:
    """Execute the published notebook's cells in one IPython namespace."""
    from IPython.core.interactiveshell import InteractiveShell

    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]]
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    shell = InteractiveShell.instance()
    snaps, log = {}, open("progress.txt", "w")
    for k, step in enumerate(plan):
        if "cell" in step:
            src = cells[step["cell"]].replace("!pip install", "# !pip install")
            # Python >= 3.12 inlines comprehensions; keep the notebook's global lookup explicit.
            src = re.sub(r"(for (\w+) in [\w\.]+ if \2 (?:not )?in )dir\(\)\]", r"\1globals()]", src)
            src = v1_source(step["cell"], src)
            params = dict(step.get("params", {}))
            if step["cell"] == 4:
                params["DATA_PATH"] = repr(str(Path(panel).resolve()))
            for name, value in params.items():
                src = _set_param(src, name, value)
            label = f"cell {step['cell']}"
        else:
            src, label = step["code"], step["label"]
        buf, t0 = io.StringIO(), time.time()
        _seed_global_rngs()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            res = shell.run_cell(src, store_history=False)
        text = buf.getvalue()
        if not res.success:  # keep the error itself; IPython's traceback layout differs from Python's
            exc = res.error_in_exec or res.error_before_exec
            text = _IPYTHON_TRACEBACK.split(text, maxsplit=1)[0] + f"\nERROR {type(exc).__name__}: {exc}\n"
        Path(f"out_{k:02d}.txt").write_text(_normalize(text))
        snaps[f"{k:02d}"] = snapshot(shell.user_ns)
        Path("snapshots.json").write_text(json.dumps(snaps, default=str))
        log.write(f"{k:02d} {label} {'OK' if res.success else 'FAILED'} {time.time() - t0:.0f}s\n")
        log.flush()


def run_package(plan: list[dict], panel: str, outdir: str) -> None:
    """Run the same plan through bpsp (setup + stages)."""
    import matplotlib

    matplotlib.use("Agg")
    from bpsp import Pipeline, PipelineConfig, ctx

    cfg = PipelineConfig()
    cfg.setup.DATA_PATH = str(Path(panel).resolve())
    cfg.setup.objective_subjective_spectrum = any(st.get("cell") == 3 for st in plan)
    cfg.output_dir = "."
    overrides: dict = {}
    for st in plan:
        if st.get("cell") == 4:
            for name, value in st.get("params", {}).items():
                setattr(cfg.setup, name, setting(value))
        if st.get("label") == "model sizes":
            for m in re.finditer(r"PIPELINE_MODEL_CONFIGS\['(\w+)'\]\['(\w+)'\]\s*=\s*(\d+)", st["code"]):
                overrides.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
    cfg.model_overrides = overrides
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    pipe = Pipeline(cfg)
    last_setup = max(i for i, st in enumerate(plan) if st.get("cell", 99) <= 5 or st.get("label") == "model sizes")
    steps = [("setup", {}, last_setup)] + [
        (CELL_TO_STAGE[st["cell"]], st.get("params", {}), i) for i, st in enumerate(plan) if st.get("cell", -1) >= 6
    ]
    snaps, log = {}, open("progress.txt", "w")
    for stage, params, k in steps:
        for name, value in params.items():
            setattr(getattr(cfg, stage), name, setting(value))
        buf, t0, err = io.StringIO(), time.time(), None
        _seed_global_rngs()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                pipe.setup() if stage == "setup" else pipe.run_stage(stage)
            except Exception as exc:  # recorded and compared, like a failing notebook cell
                import traceback

                Path(f"traceback_{k:02d}.txt").write_text(traceback.format_exc())
                print(f"\nERROR {type(exc).__name__}: {exc}")
                err = exc
        Path(f"out_{k:02d}.txt").write_text(_normalize(buf.getvalue()))
        snaps[f"{k:02d}"] = snapshot(dict(ctx))
        Path("snapshots.json").write_text(json.dumps(snaps, default=str))
        log.write(f"{k:02d} {stage} {'OK' if err is None else 'FAILED'} {time.time() - t0:.0f}s\n")
        log.flush()


# ----------------------------------------------------------------------------- comparison
def _diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one run")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not np.isclose(a, b, rtol=1e-6, atol=1e-9):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} vs {str(b)[:60]}")
    return out


def _result_files(d: Path) -> dict[str, Path]:
    skip = ("out_", "traceback_", "progress", "snapshots", "run.log", "config.resolved")
    return {
        str(p.relative_to(d)): p
        for p in d.rglob("*")
        if p.is_file() and not p.name.startswith(skip) and "catboost_info" not in p.parts
    }


# Lines left out of the printed-output comparison: stage start/end log lines, library
# warnings and tracebacks, progress bars, blank lines and IPython's echo of a cell's value.
_TRANSCRIPT_NOISE = re.compile(
    r"^stage \w+: (start|done)|Warning|warn\(|warnings\.|Traceback|File \"|^\s+\^|^\s*$|it/s|^Out\[\d+\]"
)


_WARNING_LINE = re.compile(r":\d+: \w*Warning: ")
_CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")


# Messages that v2.0.0 words differently (references to notebook cells and the Colab form
# replaced by stage and setting names) or no longer prints (a development note in the networks
# stage). They are applied to the notebook's output before comparison; nothing else is edited.
V2_MESSAGE_EDITS = [
    (
        "[VAL] Validation cells (12-16) enabled.",
        "[VAL] Validation stages nested_cv, stratified, grouped_cv, permutation and stacking_audit enabled.",
    ),
    (
        "[VAL] Validation cells (12-16) will be skipped.",
        "[VAL] Validation stages nested_cv, stratified, grouped_cv, permutation and stacking_audit will be skipped.",
    ),
    ("Bootstrap: disabled (enable in UI)", "Bootstrap: disabled (networks.run_bootstrap_CI)"),
    (
        "Centrality stability: disabled (enable in UI)",
        "Centrality stability: disabled (networks.run_centrality_stability)",
    ),
    ("ECE as in Cell 12 =", "ECE as in nested_cv ="),
    ("no Cell 12 out-of-fold probabilities", "no nested_cv out-of-fold probabilities"),
    ("Cell 18 domain map unavailable", "ablation domain map unavailable"),
    (
        "domain groups skipped: run Cell 18 in this session to add them",
        "domain groups skipped: run the ablation stage first to add them",
    ),
    ("Run Cell 6 in batch mode first.", "Run the main stage in batch mode first."),
    (
        "Ensure visualization_mode includes consensus in Cell 6.",
        "Set main.visualization_mode to a mode that includes the consensus.",
    ),
    (
        "bootstrap_ci not found -- ensure Cell 4 (Preprocessing) was executed.",
        "bootstrap_ci not found. Run setup first.",
    ),
    (
        "Pipeline score not found in globals -- run Cell 6 (Main Analysis Runner) first for comparison.",
        "pipeline score not found. Run the main stage first for comparison.",
    ),
    ("sample not found -- ensure Cell 4 (Preprocessing) was executed.", "sample not found. Run setup first."),
    ("not found. Run Cells 1-5 first.", "not found. Run setup first."),
    ("Make sure to run the data preparation cells first", "Run setup and the main stage first."),
    ("Cell 0 (Report State) has not been run in this session.", "setup has not been run in this session."),
    (
        "Run Cells 0-6 in order first (Cell 3 is optional), then re-run this cell.",
        "Run setup and the main stage first.",
    ),
]
V2_REMOVED_LINES = [  # the networks stage's note about an unimplemented target-inclusive network
    "=" * 72,
    "  TARGET-INCLUSIVE NETWORK",
    "  (Same methodology -- MDS, R² rings, fixed max)",
    "=" * 72,
    "    Target-inclusive network uses same upgraded methodology.",
    "     Append target-inclusive section from v1 with these substitutions:",
    "     • nx.circular_layout → _mds_layout()",
    "     • nx.draw_networkx_nodes → _draw_predictability_node()",
    "     • aw/ew_max → aw/_EDGE_MAXIMUM",
    "     • solid negative → dashed (ls = _DASH_NEG)",
]


# Single lines v2.0.0 no longer prints: the main stage looked up an "ensemble" entry that the
# metrics never contain, printed the KeyError and then used the "meta_model" entry it now reads directly.
V2_DROPPED_LINES = {"[WARNING] 'ensemble'"}


def _as_v2_output(lines: list[str]) -> list[str]:
    """Apply the documented v2.0.0 message edits to the notebook's printed output."""
    edited = []
    for line in lines:
        if line in V2_DROPPED_LINES:
            continue
        for old, new in V2_MESSAGE_EDITS:
            line = line.replace(old, new)
        edited.append(line)
    n = len(V2_REMOVED_LINES)
    i = 0
    while i <= len(edited) - n:
        if edited[i : i + n] == V2_REMOVED_LINES:
            del edited[i : i + n]
        else:
            i += 1
    return edited


def _transcript(path: Path) -> list[str]:
    text = path.read_text().replace(str(path.parent.resolve()), "<OUT>").replace(" object at 0x", " at 0x")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines, skip_source = [], False
    for line in text.split("\n"):
        if skip_source and line.startswith(" "):  # source line Python prints under a warning
            skip_source = False
            continue
        skip_source = bool(_WARNING_LINE.search(line))
        if not _TRANSCRIPT_NOISE.search(line):
            lines.append(line.rstrip())
    return lines


def compare(notebook_dir: str, package_dir: str) -> dict:
    """Compare two runs.

    Returns ``{'fingerprints': {step: [differences]}, 'transcripts': {step: status},
    'files': {name: status}}``. Transcripts are compared for the analysis stages; setup
    prints the same lines as notebook cells 0-5, partly in a different order, because
    module-level code runs when a module is imported.
    """
    nb, pk = Path(notebook_dir), Path(package_dir)
    s_nb = json.loads((nb / "snapshots.json").read_text())
    s_pk = json.loads((pk / "snapshots.json").read_text())
    report = {"fingerprints": {}, "files": {}}
    for step, snap in s_pk.items():
        report["fingerprints"][step] = _diff(snap, s_nb.get(step, {}))
    report["transcripts"] = {}
    for line in (pk / "progress.txt").read_text().splitlines():
        step, stage = line.split()[:2]
        if stage == "setup":
            continue
        a, b = _as_v2_output(_transcript(nb / f"out_{step}.txt")), _transcript(pk / f"out_{step}.txt")
        if a == b:
            report["transcripts"][step] = "identical"
        else:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            report["transcripts"][step] = (
                f"differs at line {first + 1}: {a[first : first + 1]} vs {b[first : first + 1]}"
            )
    f_nb, f_pk = _result_files(nb), _result_files(pk)
    for name in sorted(set(f_nb) | set(f_pk)):
        if name not in f_pk or name not in f_nb:
            report["files"][name] = "missing in package" if name not in f_pk else "only in package"
        elif name.endswith(".csv"):
            a, b = pd.read_csv(f_nb[name]), pd.read_csv(f_pk[name])
            same = (
                a.shape == b.shape
                and list(a.columns) == list(b.columns)
                and np.allclose(
                    a.select_dtypes("number").to_numpy(float),
                    b.select_dtypes("number").to_numpy(float),
                    rtol=1e-6,
                    atol=1e-9,
                    equal_nan=True,
                )
                and a.select_dtypes(exclude="number").astype(str).equals(b.select_dtypes(exclude="number").astype(str))
            )
            report["files"][name] = "identical" if same else "values differ"
        elif name.endswith(".pdf"):  # compared without the embedded creation timestamp
            same = _CREATION_DATE.sub(b"", f_nb[name].read_bytes()) == _CREATION_DATE.sub(b"", f_pk[name].read_bytes())
            report["files"][name] = "identical" if same else "bytes differ"
        else:
            same = hashlib.md5(f_nb[name].read_bytes()).digest() == hashlib.md5(f_pk[name].read_bytes()).digest()
            report["files"][name] = "identical" if same else "bytes differ"
    return report


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    cmd, *args = argv
    if cmd == "panel":
        n = int(args[args.index("--subjects") + 1]) if "--subjects" in args else 1200
        build_panel(args[0], n_subjects=n)
    elif cmd in ("notebook", "package"):
        use_stubs_if_needed()
        os.environ.setdefault("MPLBACKEND", "Agg")
        plan = json.loads(Path(args[0]).read_text())
        (run_notebook if cmd == "notebook" else run_package)(plan, args[1], args[2])
    elif cmd == "compare":
        print(json.dumps(compare(args[0], args[1]), indent=1))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
