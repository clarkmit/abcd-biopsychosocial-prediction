"""Regenerate the reference pages that are derived from the code.

* ``docs/configuration.md``: every setting, its default and allowed values (from ``bpsp.config``);
* ``docs/_generated/dataflow.md``: what each stage reads from and writes to the run context;
* ``docs/_generated/known_issues_table.md``: names v1.0.0 references but never defines
  (lines marked ``# noqa: F821 (inherited from v1.0.0 ...)``; needs ruff).

Run ``python tools/gen_docs.py`` after changing configuration or stage code. CI fails when the
committed pages are out of date.
"""

import dataclasses
import sys
from pathlib import Path


def configuration(out: Path) -> None:
    from bpsp import config

    L = [
        "# Configuration",
        "",
        "A run is configured by one YAML file (or a `PipelineConfig` object). Every key is optional; missing keys keep the",
        "published defaults. `bpsp init-config FILE` writes a file with every setting.",
        "",
        "```yaml",
        "stages: [main, nested_cv]          # stages to run after setup, in order",
        "output_dir: outputs/my_run         # tables, figures, run.log, config.resolved.yaml",
        "model_overrides:                   # optional hyperparameter overrides per base learner",
        "  catboost: {iterations: 300}",
        "setup:",
        "  DATA_PATH: /secure/abcd/CLEAN_ABCD_5.1_panel.csv",
        "main:",
        "  target_options: depress_D_p",
        "  split: within_categories",
        "nested_cv:",
        "  n_outer_folds: 10",
        "```",
        "",
        "Setting names are identical to the form fields of the published notebook, so each one can be traced to the",
        "paper version. Settings with a fixed set of values are validated when the configuration is loaded.",
        "",
        "## Settings you will change most",
        "",
        "| Setting | What it does |",
        "|---|---|",
        "| `setup.DATA_PATH` | the merged analysis panel (keep it outside the repository) |",
        "| `main.target_options` | the target of a single-target run |",
        "| `main.batch_mode` | a bundle of targets run one after another, instead of `target_options` |",
        "| `main.split` | `within_categories` (one model per measurement category) or `across_categories` |",
        "| `main.tp_option` | the wave of the target, `0` (baseline) to `4` |",
        "| `main.feature_tp_option` | `Match target` for concurrent prediction, or an earlier wave for prospective prediction |",
        "| `setup.objective_subjective_spectrum` | use the objective/subjective gradient instead of the measurement categories |",
        "| `model_overrides` | base-learner hyperparameters for this run |",
        "| `stages` | the stages to run after setup |",
        "",
        "Every setting is listed below.",
        "",
    ]
    top = {f.name: f for f in dataclasses.fields(config.PipelineConfig)}
    for name, f in top.items():
        default = f.default_factory() if f.default is dataclasses.MISSING else f.default
        if not dataclasses.is_dataclass(default):
            continue
        cls = type(default)
        fields = dataclasses.fields(cls)
        L += [f"## `{name}`", "", (cls.__doc__ or "").strip(), ""]
        if not fields:
            L += ["No settings.", ""]
            continue
        docs = _attribute_docs(cls)
        L += ["| Setting | Default | Description |", "|---|---|---|"]
        for fl in fields:
            d = docs.get(fl.name, "")
            choices = config.CHOICES.get((name, fl.name))
            if choices:
                d = d.replace(f"One of ``CHOICES[({name!r}, {fl.name!r})]``.", "").strip()
                shown = ", ".join(f"`{c}`" for c in choices[:8]) + (
                    f", ... ({len(choices)} values)" if len(choices) > 8 else ""
                )
                d = (d + " " if d else "") + f"Allowed: {shown}."
            L.append(f"| `{fl.name}` | `{fl.default!r}` | {d.replace('|', '/')} |")
        L.append("")
    out.write_text("\n".join(L))


def _attribute_docs(cls):
    """Attribute docstrings (the string literal after each field) of a dataclass."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    body = tree.body[0].body
    docs = {}
    for node, nxt in zip(body, body[1:]):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(nxt, ast.Expr)
            and isinstance(nxt.value, ast.Constant)
            and isinstance(nxt.value.value, str)
        ):
            docs[node.target.id] = " ".join(nxt.value.value.split())
    return docs


def _ctx_access(text):
    """Names a module reads from / writes to the run context (attribute, item and `in`/get access)."""
    import ast

    reads, writes = set(), set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ctx":
            (writes if isinstance(node.ctx, (ast.Store, ast.Del)) else reads).add(node.attr)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ctx"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            (writes if isinstance(node.ctx, (ast.Store, ast.Del)) else reads).add(node.slice.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ctx"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            reads.add(node.args[0].value)
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == "ctx"
            and isinstance(node.left, ast.Constant)
        ):
            reads.add(node.left.value)
    reads -= {"get", "items", "keys", "values", "update", "register", "reset", "variables"}
    return reads, writes


def dataflow(src_dir: Path, out: Path) -> None:
    from bpsp import config
    from bpsp.pipeline import SETUP_MODULES, STAGE_MODULES

    settings = set()
    for f in dataclasses.fields(config.PipelineConfig):
        default = f.default_factory() if f.default is dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default):
            settings |= {x.name for x in dataclasses.fields(default)}
    rows = []
    entries = [("setup: " + m.rsplit(".", 1)[-1], m) for m in SETUP_MODULES] + list(STAGE_MODULES.items())
    for stage, mod in entries:
        path = src_dir / (mod.replace(".", "/") + ".py")
        reads, writes = _ctx_access(path.read_text())
        public = lambda s: sorted(x for x in s if not x.startswith("_"))
        rows.append((stage, mod, public(reads - writes - settings), public(writes - settings)))
    L = ["| Stage | Module | Reads from `ctx` | Writes to `ctx` |", "|---|---|---|---|"]  # results only
    for stage, mod, reads, writes in rows:
        fmt = lambda xs: ", ".join(f"`{x}`" for x in xs) or "none"
        L.append(f"| `{stage}` | `{mod.replace('bpsp.', '')}` | {fmt(reads)} | {fmt(writes)} |")
    L += [
        "",
        "Results only: each stage also reads its own settings (see [Configuration](configuration.md)) through",
        "`ctx`. Names starting with an underscore (internal) are omitted.",
    ]
    out.write_text("\n".join(L) + "\n")


def known_issues(src_dir: Path, out: Path) -> None:
    import json
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if ruff is None:
        raise SystemExit("ruff is required (pip install -e '.[dev]')")
    res = subprocess.run(
        [
            ruff,
            "check",
            "--isolated",
            "--select",
            "F821,F823",
            "--ignore-noqa",
            "--output-format",
            "json",
            str(src_dir),
        ],
        capture_output=True,
        text=True,
    )
    rows = set()
    for item in json.loads(res.stdout or "[]"):
        name = item["message"].split("`")[1] if "`" in item["message"] else item["message"]
        rows.add(
            (Path(item["filename"]).resolve().relative_to(src_dir.resolve().parent).as_posix(), name, item["code"])
        )
    lines = ["| Module | Name | Rule |", "|---|---|---|"]
    lines += [f"| `{module}` | `{name}` | {code} |" for module, name, code in sorted(rows)]
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    gen = repo / "docs" / "_generated"
    gen.mkdir(parents=True, exist_ok=True)
    configuration(repo / "docs" / "configuration.md")
    dataflow(repo / "src", gen / "dataflow.md")
    known_issues(repo / "src", gen / "known_issues_table.md")
    print("docs generated")
