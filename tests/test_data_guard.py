"""The guard that keeps ABCD participant data out of the repository (tools/check_no_participant_data.py)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("guard", REPO / "tools" / "check_no_participant_data.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize("name", ["panel.csv", "export.xlsx", "subjects.sav", "dump.parquet", "data.csv.gz", "x.zip"])
def test_data_and_archive_files_are_blocked(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    Path(name).write_text("a,b\n1,2\n")
    assert guard.check([name])


def test_participant_identifiers_are_blocked_in_any_text_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_id = "NDAR_" + "INV" + "AB12CD34"  # assembled so this test file does not trip the guard itself
    Path("notes.md").write_text(f"subject {fake_id} looked odd")
    assert "participant identifier" in guard.check(["notes.md"])[0]


def test_large_files_are_blocked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("big.txt").write_bytes(b"0" * (guard.MAX_BYTES + 1))
    assert "larger than" in guard.check(["big.txt"])[0]


def test_allowed_and_ordinary_files_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("README.md").write_text("no data here")
    assert guard.check(["README.md"]) == []
    monkeypatch.chdir(REPO)
    assert guard.check(sorted(guard.ALLOWED)) == []


def test_every_tracked_file_passes():
    try:
        files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    if not files.strip():
        pytest.skip("no tracked files")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO)
        assert guard.check(files.split("\n")[:-1]) == []
