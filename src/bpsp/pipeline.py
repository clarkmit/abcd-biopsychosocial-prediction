"""Run the pipeline: setup, then any sequence of analysis stages.

Stages correspond one-to-one to the cells of the published notebook (v1.0.0) and run
in the same order. Each stage reads its settings from :class:`~bpsp.config.PipelineConfig`
and exchanges results with other stages through :data:`bpsp.context.ctx`.

Example
-------
>>> from bpsp import Pipeline, PipelineConfig
>>> cfg = PipelineConfig.from_yaml("configs/demo.yaml")
>>> Pipeline(cfg).run()                      # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType

from bpsp._logging import ensure_logging
from bpsp.config import ALL_STAGES, PipelineConfig
from bpsp.context import ctx

log = logging.getLogger(__name__)

SETUP_MODULES: tuple[str, ...] = (
    "bpsp.reporting",
    "bpsp.data.registry",
    "bpsp.data.circularity",
    "bpsp.data.spectrum",
    "bpsp.data.preprocessing",
    "bpsp.models.stacking",
)
"""Library modules executed at session start (notebook cells 0-5)."""

STAGE_MODULES: dict[str, str] = {
    "main": "bpsp.analysis.runner",
    "batch_summary": "bpsp.reports.batch_summary",
    "sankey": "bpsp.reports.sankey",
    "networks": "bpsp.interpretation.networks",
    "diagnostics": "bpsp.interpretation.diagnostics",
    "smoke_test": "bpsp.validation.smoke_test",
    "nested_cv": "bpsp.validation.nested_cv",
    "stratified": "bpsp.validation.stratified",
    "grouped_cv": "bpsp.validation.grouped_cv",
    "permutation": "bpsp.validation.permutation",
    "stacking_audit": "bpsp.validation.stacking_audit",
    "baselines": "bpsp.validation.baselines",
    "ablation": "bpsp.validation.ablation",
    "trajectories": "bpsp.validation.trajectories",
    "temporal_gradient": "bpsp.validation.temporal_gradient",
    "fairness": "bpsp.validation.fairness",
    "calibration": "bpsp.validation.calibration",
    "base_learners": "bpsp.validation.base_learners",
    "grouped_importance": "bpsp.validation.grouped_importance",
}
"""Stage name -> module (notebook cells 6-24, in order)."""

assert tuple(STAGE_MODULES) == ALL_STAGES


@contextlib.contextmanager
def working_directory(path: str | os.PathLike[str]) -> Iterator[Path]:
    """Temporarily run inside ``path``; stages write their tables and figures there."""
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(previous)


class Pipeline:
    """Orchestrates one analysis session.

    Parameters
    ----------
    config:
        Full run configuration. ``config.setup.DATA_PATH`` must point at the merged
        multi-wave panel (or at the simulated panel shipped in ``data/simulated``).
        Relative paths (data and output directory) are resolved against the working
        directory at the time the pipeline is created.

    Notes
    -----
    The pipeline keeps a reference to ``config``: settings changed between stages apply
    to the next stage, as editing a form field before running a notebook cell did. All
    tables and figures are written to ``config.output_dir``.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config if config is not None else PipelineConfig()
        self.config.validate()
        self._base_dir = Path.cwd()
        self.timings: dict[str, float] = {}
        self._ready = False

    @property
    def output_dir(self) -> Path:
        """Absolute output directory of this session."""
        return (self._base_dir / Path(self.config.output_dir).expanduser()).resolve()

    @property
    def data_path(self) -> Path:
        """Absolute path of the analysis panel (``config.setup.DATA_PATH``)."""
        return (self._base_dir / Path(self.config.setup.DATA_PATH).expanduser()).resolve()

    # ------------------------------------------------------------------ public API
    def setup(self) -> None:
        """Start a fresh session: load data and build feature registries (cells 0-5)."""
        ensure_logging()
        data_path = self.data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"setup.DATA_PATH is not a file: {data_path}")
        ctx.reset()
        settings = self.config.stage_settings("setup")
        settings["DATA_PATH"] = str(data_path)
        ctx.update(settings)
        with working_directory(self.output_dir):
            for name in SETUP_MODULES:
                if name == "bpsp.data.spectrum" and not self.config.setup.objective_subjective_spectrum:
                    continue  # optional ontology (notebook cell 3)
                self._execute(name, label=name.rsplit(".", 1)[-1])
        self._apply_model_overrides()
        self._ready = True

    def run_stage(self, stage: str) -> None:
        """Run one analysis stage (equivalent to running its notebook cell)."""
        _check_stages([stage])
        if not self._ready:
            self.setup()
        ctx.update(self.config.stage_settings(stage))
        with working_directory(self.output_dir):
            self._execute(STAGE_MODULES[stage], label=stage)

    def run(self, stages: Iterable[str] | None = None) -> None:
        """Run setup and then ``stages`` (default: ``config.stages``)."""
        stages = list(stages if stages is not None else self.config.stages)
        _check_stages(stages)
        self.setup()
        for stage in stages:
            self.run_stage(stage)

    # ------------------------------------------------------------------ internals
    def _execute(self, module_name: str, label: str) -> ModuleType:
        module = importlib.import_module(module_name)
        ctx.register(module)
        runner = getattr(module, "run", None)
        if runner is not None:
            log.info("stage %s: start", label)
            t0 = time.perf_counter()
            runner()
            self.timings[label] = time.perf_counter() - t0
            log.info("stage %s: done in %.1fs", label, self.timings[label])
        return module

    def _apply_model_overrides(self) -> None:
        if not self.config.model_overrides:
            return
        PIPELINE_MODEL_CONFIGS = ctx.PIPELINE_MODEL_CONFIGS  # built by bpsp.models.stacking.run()
        for model, params in self.config.model_overrides.items():
            if model not in PIPELINE_MODEL_CONFIGS:
                raise ValueError(
                    f"model_overrides: unknown model {model!r}; choose from {list(PIPELINE_MODEL_CONFIGS)}"
                )
            PIPELINE_MODEL_CONFIGS[model].update(params)


def _check_stages(stages: Iterable[str]) -> None:
    unknown = [s for s in stages if s not in STAGE_MODULES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {list(STAGE_MODULES)}")
