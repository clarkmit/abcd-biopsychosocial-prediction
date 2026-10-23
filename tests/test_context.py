import types

import pytest

from bpsp.context import RunContext, UnsetContextName


def test_attribute_and_item_access_are_the_same_store():
    ctx = RunContext()
    ctx.X_train = [1, 2]
    assert ctx["X_train"] == [1, 2]
    ctx["split"] = "across_categories"
    assert ctx.split == "across_categories"
    assert "split" in ctx


def test_missing_name_raises_a_clear_error():
    ctx = RunContext()
    with pytest.raises(UnsetContextName, match="not defined in the run context"):
        _ = ctx.metrics_dict_full
    assert not hasattr(ctx, "metrics_dict_full")
    assert getattr(ctx, "metrics_dict_full", None) is None


def test_delete_and_reset():
    ctx = RunContext(a=1, b=2)
    del ctx.a
    assert "a" not in ctx
    with pytest.raises(UnsetContextName):
        del ctx.a
    ctx.reset()
    assert len(ctx) == 0


def test_register_exposes_module_definitions_but_not_dunders():
    mod = types.ModuleType("fake_stage")
    mod.helper = lambda: 42
    mod.CONSTANT = 7
    ctx = RunContext(CONSTANT=0)
    ctx.register(mod)
    assert ctx.helper() == 42
    assert ctx.CONSTANT == 7  # a stage's definitions replace earlier ones, as re-running a cell did
    assert "__name__" not in ctx


def test_variables_excludes_code():
    ctx = RunContext()
    ctx.register(types.ModuleType("m"))
    ctx.f = len
    ctx.g = lambda: 1
    ctx.mod = types
    ctx.cls = int
    ctx.X_train = [1]
    assert ctx.variables() == {"X_train": [1]}
