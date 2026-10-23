"""Shared fixtures. Everything runs on the fully simulated panel shipped in data/simulated/."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
SIMULATED_PANEL = REPO / "data" / "simulated" / "CLEAN_ABCD_5_1_panel_simulated.csv"

# Small models so tests finish quickly; structure and code paths are unchanged.
TINY_MODELS = {
    "catboost": {"iterations": 20},
    "xgboost": {"n_estimators": 20},
    "random_forest": {"n_estimators": 20},
}


def tiny_config(output_dir: Path, **main):
    from bpsp import PipelineConfig

    cfg = PipelineConfig()
    cfg.setup.DATA_PATH = str(SIMULATED_PANEL)
    cfg.output_dir = str(output_dir)
    cfg.model_overrides = TINY_MODELS
    cfg.main.target_options = main.pop("target_options", "internal_D_p")
    cfg.main.split = main.pop("split", "across_categories")
    cfg.main.tp_option = main.pop("tp_option", "2")
    cfg.main.visualization_mode = main.pop("visualization_mode", "No Visuals")
    cfg.main.batch_mode = main.pop("batch_mode", "Single Target")
    for key, value in main.items():
        setattr(cfg.main, key, value)
    return cfg


@pytest.fixture(scope="session")
def session_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("bpsp_session")


@pytest.fixture(scope="session")
def ready_pipeline(session_dir):
    """A pipeline whose setup (registry, exclusions, data loading) has run on the simulated panel."""
    from bpsp import Pipeline

    pipe = Pipeline(tiny_config(session_dir))
    pipe.setup()
    return pipe


@pytest.fixture(scope="session")
def simulated_panel() -> Path:
    """The fully simulated panel shipped in data/simulated/ (independent columns, no signal)."""
    return SIMULATED_PANEL


@pytest.fixture
def make_config():
    """Factory for small configurations on the simulated panel: ``make_config(output_dir, **main_settings)``."""
    return tiny_config
