import importlib

import pytest

from bpsp import ctx
from bpsp.pipeline import SETUP_MODULES, STAGE_MODULES, Pipeline, working_directory


@pytest.mark.parametrize("module", list(SETUP_MODULES) + list(STAGE_MODULES.values()))
def test_every_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("stage,module", list(STAGE_MODULES.items()))
def test_every_stage_has_a_run_function(stage, module):
    assert callable(getattr(importlib.import_module(module), "run", None)), stage


def test_setup_loads_the_panel_and_registries(ready_pipeline):
    assert ctx.sample.shape[0] > 0
    assert set(ctx.sample["time"].unique()) <= {0, 1, 2, 3, 4}
    assert ctx.DATA_PATH.endswith("CLEAN_ABCD_5_1_panel_simulated.csv")
    assert len(ctx.t2_variables) > 100
    assert "catboost" in ctx.PIPELINE_MODEL_CONFIGS


def test_model_overrides_are_applied(ready_pipeline):
    assert ctx.PIPELINE_MODEL_CONFIGS["catboost"]["iterations"] == 20
    assert ctx._CB_PARAMS is ctx.PIPELINE_MODEL_CONFIGS["catboost"]


def test_unknown_stage_is_rejected_before_any_work():
    pipe = Pipeline()  # default DATA_PATH does not exist: the stage name is checked first
    with pytest.raises(ValueError, match="unknown stage"):
        pipe.run_stage("no_such_stage")
    with pytest.raises(ValueError, match="unknown stage"):
        pipe.run(["main", "no_such_stage"])


def test_missing_data_file_fails_early(tmp_path):
    from bpsp import PipelineConfig

    cfg = PipelineConfig()
    cfg.setup.DATA_PATH = str(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="DATA_PATH"):
        Pipeline(cfg).setup()


def test_unknown_model_override_is_rejected(make_config, tmp_path):
    cfg = make_config(tmp_path)
    cfg.model_overrides = {"no_such_model": {"n_estimators": 5}}
    with pytest.raises(ValueError, match="unknown model"):
        Pipeline(cfg).setup()


def test_relative_paths_resolve_against_the_creation_directory(make_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = make_config("outputs/run1")
    pipe = Pipeline(cfg)
    monkeypatch.chdir("/")
    assert pipe.output_dir == tmp_path.resolve() / "outputs" / "run1"
    assert cfg.output_dir == "outputs/run1"  # the caller's configuration is not modified


def test_setup_resets_state_between_sessions(ready_pipeline):
    ctx.leftover_from_previous_session = 1
    ready_pipeline.setup()
    assert "leftover_from_previous_session" not in ctx


def test_working_directory_restores_cwd(tmp_path):
    import os

    before = os.getcwd()
    with working_directory(tmp_path / "out") as target:
        assert os.getcwd() == str(target)
    assert os.getcwd() == before
