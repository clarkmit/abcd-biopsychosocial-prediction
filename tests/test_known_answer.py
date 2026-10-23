"""Known-answer check: the pipeline finds a signal planted in simulated data.

The opposite check, that nothing is found in pure noise, is part of
``test_integration.py::test_regression_run_and_validation_stages``.
"""

import pytest

from bpsp import Pipeline, ctx
from bpsp.simulation import PLANTED_FEATURES, make_example_panel

pytestmark = pytest.mark.integration


def test_planted_predictors_rank_first_and_the_ensemble_explains_the_signal(tmp_path, make_config, simulated_panel):
    panel = tmp_path / "example_panel.csv"
    make_example_panel(simulated_panel, n_subjects=600).to_csv(panel, index=False)
    cfg = make_config(tmp_path / "out", visualization_mode="Ensemble Consensus + Best SHAP")
    cfg.setup.DATA_PATH = str(panel)
    cfg.model_overrides = {
        "catboost": {"iterations": 60},
        "xgboost": {"n_estimators": 40},
        "random_forest": {"n_estimators": 50},
    }

    Pipeline(cfg).run(["main"])

    planted = {ctx.DISPLAY_NAMES.get(name, name) for name in PLANTED_FEATURES}
    top = set(ctx.consensus_df["weighted_score"].nlargest(len(planted)).index)
    assert top == planted
    assert ctx.metrics_dict_full["metrics"][0]["meta_model"]["r2"] > 0.25
