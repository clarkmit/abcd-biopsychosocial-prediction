"""Leakage guards: the checks that matter most for the paper's claims."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from bpsp.data.circularity import get_circular_exclusions
from bpsp.data.preprocessing import across_categories_data

TARGETS = ["internal_D_p", "depress_D_p", "adhd_D_p", "external_D_p"]


@pytest.mark.parametrize("target", TARGETS)
def test_target_and_circular_variables_never_enter_the_predictors(ready_pipeline, target):
    X_train, X_test, y_train, y_test = across_categories_data(2, [target])
    excluded, _ = get_circular_exclusions(target, timepoint=2)
    assert target not in X_train.columns
    assert not set(X_train.columns) & excluded
    assert list(X_train.columns) == list(X_test.columns)


def test_scaling_is_fitted_on_training_rows_only(ready_pipeline):
    X_train, X_test, _, _ = across_categories_data(2, ["internal_D_p"], scale_x=True)
    numeric = X_train.select_dtypes("number")
    assert np.allclose(numeric.mean(), 0, atol=1e-6)  # centered with training statistics
    assert not np.allclose(X_test[numeric.columns].mean(), 0, atol=1e-6)  # test rows were not used


def test_train_and_test_rows_are_disjoint(ready_pipeline):
    X_train, X_test, _, _ = across_categories_data(2, ["internal_D_p"])
    assert not set(X_train.index) & set(X_test.index)
    n = len(X_train) + len(X_test)
    assert abs(len(X_test) / n - 0.30) < 0.02


class _SeenRowProbe(BaseEstimator, RegressorMixin):
    """Predicts 1 for rows it was trained on, 0 otherwise (row id in column 'rid')."""

    def fit(self, X, y):
        self.seen_ = set(X["rid"])
        return self

    def predict(self, X):
        return X["rid"].isin(self.seen_).astype(float).to_numpy()


class _RecordingMeta(LinearRegression):
    def fit(self, X, y, sample_weight=None):
        self.recorded_ = np.asarray(X).copy()
        return super().fit(X, y)


def test_meta_learner_sees_only_out_of_fold_predictions(ready_pipeline):
    from bpsp.models.stacking import create_stacking_ensemble

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"rid": np.arange(200), "x": rng.normal(size=200)})
    y = pd.Series(X["x"] + rng.normal(size=200))
    model = create_stacking_ensemble("regression", cv=5, random_state=0)
    model.base_models = {"probe": _SeenRowProbe()}
    model.meta_model = _RecordingMeta()
    model.fit(X, y)
    # every training row's meta-feature came from a model that never saw that row
    assert np.all(model.meta_model.recorded_ == 0)


def test_permutation_p_values_follow_phipson_smyth():
    from bpsp.validation.permutation import PermutationTester

    tester = PermutationTester(
        pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.0, 1.0]), task_type="regression", n_permutations=99
    )
    tester.observed_score = 0.9
    tester.null_scores = np.zeros(99)
    assert tester._compute_p_value() == pytest.approx(1 / 100)  # never zero
    tester.null_scores = np.ones(99)
    assert tester._compute_p_value() == pytest.approx(1.0)


class _GroupProbe(BaseEstimator, RegressorMixin):
    overlaps: list = []

    def fit(self, X, y):
        self.groups_ = set(X["gid"])
        return self

    def predict(self, X):
        _GroupProbe.overlaps.append(len(self.groups_ & set(X["gid"])))
        return np.zeros(len(X))


def test_grouped_cv_keeps_families_together():
    from bpsp.validation.grouped_cv import GroupedCVCheck

    rng = np.random.default_rng(1)
    gid = np.repeat(np.arange(60), 3)
    X = pd.DataFrame({"gid": gid, "x": rng.normal(size=gid.size)})
    y = pd.Series(rng.normal(size=gid.size))
    _GroupProbe.overlaps = []
    GroupedCVCheck(X, y, task_type="regression", n_outer=5).run_group_cv(gid, "family", model_factory=_GroupProbe)
    assert len(_GroupProbe.overlaps) == 5
    assert sum(_GroupProbe.overlaps) == 0
