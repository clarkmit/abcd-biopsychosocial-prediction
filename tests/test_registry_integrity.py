"""Integrity of the variable registry and the per-target circularity exclusions."""

import pytest

from bpsp import ctx
from bpsp.data import circularity, registry


def test_outcome_lists_are_disjoint():
    reg = set(registry.continuous_outcome_variables)
    clf = set(registry.binary_or_categorical_outcome_variables)
    assert not reg & clf


@pytest.mark.parametrize("tp", [0, 1, 2, 3, 4])
def test_across_category_pools_are_non_empty(tp):
    assert len(getattr(registry, f"T{tp}_VARIABLES")) > 50


def test_child_categories_are_non_empty():
    for tp, categories in registry.CHILD_VARIABLES_BY_TIMEPOINT.items():
        for name, variables in categories.items():
            assert variables, f"T{tp} category {name!r} is empty"


def test_published_registry_is_not_modified_by_a_session(ready_pipeline):
    ctx.child_variables_for_each_time_point[2]["__probe__"] = ["x"]
    assert "__probe__" not in registry.CHILD_VARIABLES_BY_TIMEPOINT[2]
    del ctx.child_variables_for_each_time_point[2]["__probe__"]


def test_exclusions_return_two_sets():
    across, within = circularity.get_circular_exclusions("depress_D_p", timepoint=2)
    assert isinstance(across, set) and isinstance(within, set)
    assert across, "depression must exclude its own constituent items"


def test_unknown_target_has_no_exclusions():
    assert circularity.get_circular_exclusions("not_a_target") == (set(), set())


def test_parent_targets_are_recognised():
    assert circularity.is_parent_target("parent_depress_D_p")
    assert not circularity.is_parent_target("depress_D_p")
