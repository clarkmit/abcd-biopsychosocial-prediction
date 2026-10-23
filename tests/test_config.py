import pytest

from bpsp.config import ALL_STAGES, CHOICES, PipelineConfig
from bpsp.pipeline import STAGE_MODULES


def test_defaults_are_valid_and_match_the_paper_notebook():
    cfg = PipelineConfig()
    cfg.validate()
    assert cfg.main.split == "across_categories"
    assert cfg.main.tp_option == "2"
    assert cfg.nested_cv.n_outer_folds == 10
    assert cfg.permutation.n_permutations == 1000
    assert cfg.setup.objective_subjective_spectrum is False


def test_every_stage_has_a_module_in_notebook_order():
    assert tuple(STAGE_MODULES) == ALL_STAGES
    assert ALL_STAGES[0] == "main"


def test_partial_dict_keeps_other_defaults():
    cfg = PipelineConfig.from_dict({"stages": ["main", "nested_cv"], "nested_cv": {"n_outer_folds": 5}})
    assert cfg.stages == ["main", "nested_cv"]
    assert cfg.nested_cv.n_outer_folds == 5
    assert cfg.main.split == "across_categories"


@pytest.mark.parametrize(
    "data, message",
    [
        ({"stages": ["main", "no_such_stage"]}, "unknown stage"),
        ({"main": {"no_such_setting": 1}}, "unknown setting"),
        ({"not_a_section": {}}, "unknown top-level"),
        ({"main": {"split": "sideways"}}, "not allowed; choose from 'across_categories'"),
        ({"main": 5}, "must be a mapping"),
        ({"nested_cv": {"n_outer_folds": "ten"}}, "should be of type int"),
        ({"main": {"skip_tabpfn": "yes"}}, "should be of type bool"),
    ],
)
def test_invalid_configuration_fails_fast(data, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig.from_dict(data)


def test_yaml_scalars_are_converted_to_the_notebook_types(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("stages: main\nmain:\n  tp_option: 2\nsetup:\n  ensemble_barplot_palette: null\n")
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.stages == ["main"]
    assert cfg.main.tp_option == "2"
    assert cfg.setup.ensemble_barplot_palette == "None"


def test_misspelt_settings_raise():
    cfg = PipelineConfig()
    with pytest.raises(AttributeError):
        cfg.main.target_option = "adhd_D_p"  # target_options


def test_choices_come_from_the_notebook_drop_downs():
    assert set(CHOICES[("main", "split")]) == {"across_categories", "within_categories"}
    assert "Single Target" in CHOICES[("main", "batch_mode")]


def test_yaml_round_trip(tmp_path):
    cfg = PipelineConfig.from_dict({"stages": ["main"], "main": {"target_options": "adhd_D_p"}})
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    again = PipelineConfig.from_yaml(path)
    assert again.to_dict() == cfg.to_dict()


def test_stage_settings_are_plain_dicts():
    settings = PipelineConfig().stage_settings("calibration")
    assert settings["rcal_bin_strategy"] == "quantile"
    assert PipelineConfig().stage_settings("smoke_test") == {}
