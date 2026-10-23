"""The package reproduces the published notebook (the full comparison is slow: `pytest -m slow`)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.regression.equivalence import NOTEBOOK, REPO, V1_SOURCE_EDITS, V1_TEXT_EDITS, compare

HERE = Path(__file__).resolve().parent
PLANS = sorted((HERE / "plans").glob("*.json"))


def test_documented_edits_match_the_published_notebook():
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]]
    for cell, edits in V1_SOURCE_EDITS.items():
        for old, _ in edits:
            assert cells[cell].count(old) == 1, old
    for old, _ in V1_TEXT_EDITS:
        assert any(old in c for c in cells), old


def test_package_carries_the_documented_edits():
    source = "".join(p.read_text() for p in (REPO / "src" / "bpsp").rglob("*.py"))
    for old, new in V1_TEXT_EDITS:
        assert old not in source and new in source, old


def _run(*args):
    env = {**os.environ, "PYTHONHASHSEED": "0"}  # same string hashing (set and dict order) in both runs
    subprocess.run([sys.executable, str(HERE / "equivalence.py"), *map(str, args)], check=True, env=env)


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    path = tmp_path_factory.mktemp("equivalence") / "panel.csv"
    _run("panel", path, "--subjects", 400)
    return path


@pytest.mark.slow
@pytest.mark.skipif(sys.version_info < (3, 12), reason="the v1.0.0 notebook uses Python 3.12 f-string syntax")
@pytest.mark.parametrize("plan", PLANS, ids=[p.stem for p in PLANS])
def test_package_matches_published_notebook(plan, panel, tmp_path):
    _run("notebook", plan, panel, tmp_path / "notebook")
    _run("package", plan, panel, tmp_path / "package")
    report = compare(tmp_path / "notebook", tmp_path / "package")
    mismatched = {step: d for step, d in report["fingerprints"].items() if d}
    assert not mismatched, mismatched
    printed = {step: status for step, status in report["transcripts"].items() if status != "identical"}
    assert not printed, printed
    bad_files = {f: s for f, s in report["files"].items() if s != "identical"}
    assert not bad_files, bad_files
