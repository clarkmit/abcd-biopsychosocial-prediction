"""Command-line interface: ``bpsp run``, ``bpsp stages``, ``bpsp init-config``, ``bpsp example-panel``.

Examples
--------
Run the demo on a simulated panel with a known signal::

    bpsp example-panel outputs/example_panel.csv
    bpsp run configs/demo.yaml

Write a configuration file with every setting at its published default::

    bpsp init-config my_run.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import html
import io
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import yaml

from bpsp._logging import configure_logging
from bpsp._version import __version__
from bpsp.config import ALL_STAGES, PipelineConfig

STAGE_HELP = {
    "main": "main analysis runner: stacking ensemble per target / feature category (cell 6)",
    "batch_summary": "batch metric dashboard across targets (cell 7)",
    "sankey": "Sankey visualization of predictive feature domains (cell 8)",
    "networks": "top-feature correlation / partial-correlation networks (cell 9)",
    "diagnostics": "model diagnostics: diversity, stability, learning curves, SHAP (cell 10)",
    "smoke_test": "pipeline smoke test (cell 11)",
    "nested_cv": "outer-fold nested CV metrics and calibration, Supp. S1.4/S1.11 (cell 12)",
    "stratified": "demographic-stratified performance (cell 13)",
    "grouped_cv": "family-grouped and site-grouped cross-validation (cell 14)",
    "permutation": "permutation testing and null distributions (cell 15)",
    "stacking_audit": "stacking audit: meta-learner weights and leakage checks (cell 16)",
    "baselines": "PCA baseline and temporal hold-out (cell 17)",
    "ablation": "domain ablation and feature-domain analysis (cell 18)",
    "trajectories": "cross-timepoint performance trajectories (cell 19)",
    "temporal_gradient": "temporal gradient testing (cell 20)",
    "fairness": "race/ethnicity and religion fairness, confound check (cell 21)",
    "calibration": "calibration check and Platt/isotonic recalibration (cell 22)",
    "base_learners": "base learners vs stacked ensemble on held-out data (cell 23)",
    "grouped_importance": "correlation-aware grouped permutation importance (cell 24)",
}


class _Tee(io.TextIOBase):
    """Write to several text streams at once (console and ``run.log``)."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


_BLOCK_END = re.compile(r"</(div|p|h[1-6]|li|tr|table)>|<br\s*/?>", re.I)
_CELL_END = re.compile(r"</t[dh]>", re.I)
_TAG = re.compile(r"<[^>]+>")


def _html_to_text(markup: str) -> str:
    """Plain-text rendering of the small HTML notes the stages display in Jupyter."""
    text = _CELL_END.sub("  ", _BLOCK_END.sub("\n", markup))
    text = html.unescape(_TAG.sub("", text))
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


@contextlib.contextmanager
def _rich_output_as_text() -> Iterator[None]:
    """Show tables and HTML notes as text while a command-line run is in progress.

    The stages call IPython's ``display`` for styled tables and HTML notes, which Jupyter
    renders; outside Jupyter ``display`` prints ``str(obj)``. For the duration of the run
    those objects get a readable text form; the original behavior is restored afterwards.
    """
    import pandas as pd
    from IPython.core.display import HTML
    from pandas.io.formats.style import Styler

    def styler_text(self: Styler) -> str:
        show_index = not all(getattr(self, "hide_index_", [False]))
        with pd.option_context("display.precision", 4, "display.width", 200, "display.max_columns", 50):
            text: str = self.data.to_string(index=show_index)
        return text

    def html_text(self: HTML) -> str:
        return _html_to_text(self.data or "")

    saved = {cls: cls.__dict__.get("__str__") for cls in (Styler, HTML)}
    setattr(Styler, "__str__", styler_text)
    setattr(HTML, "__str__", html_text)
    try:
        yield
    finally:
        for cls, original in saved.items():
            if original is None:
                delattr(cls, "__str__")
            else:
                setattr(cls, "__str__", original)


def _cmd_run(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")  # headless: figures are written to the output directory

    from bpsp.pipeline import Pipeline

    try:  # configuration problems are reported without a traceback
        cfg = PipelineConfig.from_yaml(args.config)
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.stages:
            cfg.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        pipe = Pipeline(cfg)  # validates the configuration
        if not pipe.data_path.is_file():
            raise FileNotFoundError(f"setup.DATA_PATH is not a file: {pipe.data_path}")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"bpsp: error: {exc}", file=sys.stderr)
        return 2
    out = pipe.output_dir
    out.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(cfg)
    resolved.output_dir = str(out)
    resolved.setup.DATA_PATH = str(pipe.data_path)
    resolved.to_yaml(out / "config.resolved.yaml")
    # run.log receives exactly what the console shows (log records, tables, warnings)
    with open(out / "run.log", "w", encoding="utf-8") as log, contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, log)))
        stack.enter_context(contextlib.redirect_stderr(_Tee(sys.stderr, log)))
        stack.enter_context(_rich_output_as_text())
        configure_logging(args.log_level)
        pipe.run(cfg.stages)
    return 0


def _cmd_stages(_: argparse.Namespace) -> int:
    width = max(map(len, ALL_STAGES))
    for name in ALL_STAGES:
        print(f"{name:<{width}}  {STAGE_HELP[name]}")
    return 0


def _cmd_init_config(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} exists; use --force to overwrite", file=sys.stderr)
        return 1
    PipelineConfig().to_yaml(path)
    print(f"wrote {path}")
    return 0


def _cmd_example_panel(args: argparse.Namespace) -> int:
    from bpsp.simulation import PLANTED_FEATURES, make_example_panel

    source = Path(args.source)
    if not source.is_file():
        msg = f"simulated panel not found: {source} (run from the repository root or pass --source)"
        print(f"bpsp: error: {msg}", file=sys.stderr)
        return 2
    out = Path(args.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_example_panel(source, n_subjects=args.subjects, seed=args.seed).to_csv(out, index=False)
    print(f"wrote {out}: {args.subjects} simulated subjects x 5 waves")
    print(f"signal planted at T2 in {', '.join(PLANTED_FEATURES)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpsp", description="ABCD biopsychosocial prediction pipeline")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run setup and the configured stages")
    r.add_argument("config", help="YAML configuration file (see configs/)")
    r.add_argument("--stages", help="comma-separated stages, overriding the config")
    r.add_argument("--output-dir", help="output directory, overriding the config")
    r.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    r.set_defaults(func=_cmd_run)

    s = sub.add_parser("stages", help="list available stages")
    s.set_defaults(func=_cmd_stages)

    c = sub.add_parser("init-config", help="write a configuration file with all defaults")
    c.add_argument("path")
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=_cmd_init_config)

    e = sub.add_parser("example-panel", help="write a simulated panel with a known signal (for the demo)")
    e.add_argument("path", help="output CSV, e.g. outputs/example_panel.csv")
    e.add_argument("--subjects", type=int, default=1000, help="number of simulated participants (default 1000)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument(
        "--source",
        default="data/simulated/CLEAN_ABCD_5_1_panel_simulated.csv",
        help="the simulated panel shipped with the repository",
    )
    e.set_defaults(func=_cmd_example_panel)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status: int = args.func(args)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
