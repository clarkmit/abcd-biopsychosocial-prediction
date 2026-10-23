"""End-to-end runs on the simulated panel (`pytest -m integration`)."""

from pathlib import Path

import numpy as np
import pytest

from bpsp import Pipeline, ctx

pytestmark = pytest.mark.integration


def test_regression_run_and_validation_stages(tmp_path, make_config):
    cfg = make_config(tmp_path / "reg", target_options="internal_D_p")
    cfg.calibration.rcal_n_bootstrap = 20
    pipe = Pipeline(cfg)
    pipe.run(["main", "networks", "base_learners", "calibration", "stacking_audit"])

    assert ctx.target_options == "internal_D_p"
    assert ctx.X_train.shape[0] > ctx.X_test.shape[0] > 0
    assert "internal_D_p" not in ctx.X_train.columns
    metrics = ctx.metrics_dict_full["metrics"][0]
    assert set(metrics) >= {"catboost", "xgboost", "random_forest", "meta_model"}
    assert np.isfinite(metrics["meta_model"]["r2"])
    # every column of the shipped panel is drawn independently, so nothing should be predictable
    assert metrics["meta_model"]["r2"] < 0.1
    assert set(pipe.timings) >= {"main", "networks", "base_learners", "calibration", "stacking_audit"}
    # the structural audit runs with or without the optional TabPFN extra
    assert ctx.stacking_auditor.audit_results["validation"]["oof_sample_coverage"].startswith("PASS")
    assert any((tmp_path / "reg").rglob("*.csv")), "stages write their tables to the output directory"


def test_classification_batch(tmp_path, make_config):
    cfg = make_config(tmp_path / "clf", batch_mode="Child Depression Worsening")
    pipe = Pipeline(cfg)
    pipe.run(["main", "batch_summary", "base_learners"])
    assert len(ctx._batch_archive) == 4
    for entry in ctx._batch_archive.values():
        assert entry["X_test"].shape[0] > 0


def test_cli_run(tmp_path, make_config):
    import yaml

    from bpsp.cli import main

    cfg = make_config(tmp_path / "cli").to_dict()
    cfg["stages"] = ["main"]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["run", str(path)]) == 0
    log = (tmp_path / "cli" / "run.log").read_text()
    assert "stage main: done" in log
    assert "<pandas.io.formats.style.Styler" not in log and "<IPython.core.display.HTML" not in log
    resolved = yaml.safe_load((tmp_path / "cli" / "config.resolved.yaml").read_text())
    assert Path(resolved["setup"]["DATA_PATH"]).is_absolute()
