import numpy as np
import pandas as pd
import pytest

from bpsp.simulation import PLANTED_FEATURES, make_example_panel


@pytest.fixture(scope="module")
def panel(simulated_panel):
    return make_example_panel(simulated_panel, n_subjects=500, seed=1)


def test_shape_and_determinism(panel, simulated_panel):
    assert panel.shape[0] == 500 * 5
    assert panel.groupby("time")["subject"].nunique().eq(500).all()
    again = make_example_panel(simulated_panel, n_subjects=500, seed=1)
    assert again.equals(panel)


def _corr(wave, feature):
    return pd.to_numeric(wave[feature], errors="coerce").corr(pd.to_numeric(wave["internal_D_p"], errors="coerce"))


def test_signal_is_planted_at_t2_only(panel):
    t0, t2 = panel[panel["time"] == 0], panel[panel["time"] == 2]
    for feature in PLANTED_FEATURES:
        assert _corr(t2, feature) > 0.25
        r0 = _corr(t0, feature)  # missing at baseline for some items
        assert np.isnan(r0) or abs(r0) < 0.15


def test_demographics_are_constant_within_participant(panel):
    assert panel.groupby("subject")["sex"].nunique().max() == 1
    assert panel.groupby("subject")["L_site_id"].nunique().max() == 1
