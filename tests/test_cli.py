import io

from bpsp.cli import STAGE_HELP, _html_to_text, _Tee, main
from bpsp.config import ALL_STAGES, PipelineConfig


def test_every_stage_has_help_text():
    assert set(STAGE_HELP) == set(ALL_STAGES)


def test_stages_option_and_errors_are_reported_without_traceback(tmp_path, capsys):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("main:\n  split: sideways\n")
    assert main(["run", str(cfg)]) == 2
    assert "bpsp: error: main.split='sideways' is not allowed" in capsys.readouterr().err
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2
    cfg.write_text(f"setup:\n  DATA_PATH: {tmp_path / 'nope.csv'}\n")
    assert main(["run", str(cfg), "--stages", "main, no_such_stage"]) == 2
    assert "unknown stage(s) ['no_such_stage']" in capsys.readouterr().err
    assert main(["run", str(cfg), "--stages", "main,"]) == 2
    assert "DATA_PATH is not a file" in capsys.readouterr().err


def test_stages_lists_every_stage(capsys):
    assert main(["stages"]) == 0
    out = capsys.readouterr().out
    for stage in ALL_STAGES:
        assert stage in out


def test_init_config_writes_a_loadable_file(tmp_path, capsys):
    path = tmp_path / "run.yaml"
    assert main(["init-config", str(path)]) == 0
    assert PipelineConfig.from_yaml(path).to_dict() == PipelineConfig().to_dict()
    assert main(["init-config", str(path)]) == 1  # refuses to overwrite
    assert main(["init-config", str(path), "--force"]) == 0


def test_html_notes_render_as_plain_text():
    markup = (
        '<div style="border:1px solid"><b>Note:</b> TabPFN&nbsp;skipped<br>second line</div>'
        "<table><tr><th>model</th><th>R2</th></tr><tr><td>catboost</td><td>0.41</td></tr></table>"
    )
    assert _html_to_text(markup) == "Note: TabPFN skipped\nsecond line\nmodel R2\ncatboost 0.41"


def test_tee_writes_to_every_stream():
    a, b = io.StringIO(), io.StringIO()
    tee = _Tee(a, b)
    tee.write("table\n")
    tee.flush()
    assert a.getvalue() == b.getvalue() == "table\n"


def test_rich_output_is_text_only_during_a_cli_run():
    import pandas as pd

    from bpsp.cli import _rich_output_as_text

    styled = pd.DataFrame({"model": ["catboost"], "R2": [0.41]}).style.hide(axis="index")
    with _rich_output_as_text():
        assert str(styled).split() == ["model", "R2", "catboost", "0.41"]
    assert str(styled).startswith("<pandas.io.formats.style.Styler")


def test_example_panel_command(tmp_path, simulated_panel, capsys):
    import pandas as pd

    out = tmp_path / "panel.csv"
    assert main(["example-panel", str(out), "--subjects", "50", "--source", str(simulated_panel)]) == 0
    assert len(pd.read_csv(out, low_memory=False)) == 50 * 5
    assert main(["example-panel", str(out), "--source", str(tmp_path / "missing.csv")]) == 2
    assert "simulated panel not found" in capsys.readouterr().err
