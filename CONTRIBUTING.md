# Contributing

Thanks for your interest. Issues and pull requests are welcome.

## Ground rules for data

* Never commit ABCD data or anything derived from it at participant level (predictions, residuals, row-level tables), and never paste such data into issues. Use the simulated panel in `data/simulated/` to reproduce problems.
* Keep real-data runs outside the repository folder.
* The pre-commit hook and CI block data files and ABCD participant identifiers. Do not bypass them.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
ruff check . && ruff format --check .
mypy                                         # the hand-written modules
pytest -m "not integration and not slow"   # unit tests, seconds
pytest -m integration                        # end-to-end on the simulated panel, a few minutes
pytest -m slow                               # equivalence with the published notebook, under an hour
```

`make help` lists shortcuts for the same checks and for the demo, the notebook and the docs.

## Changing analysis code

The stage modules are a faithful port of the published notebook, and the equivalence test (`tests/regression/`) checks that they still reproduce it. When a change is meant to keep results identical, run `pytest -m slow` before opening a pull request. When a change is meant to alter results (a bug fix, a new method), say so in the pull request and in `CHANGELOG.md`, explain the effect on the simulated panel, and add the change to the documented edits in `tests/regression/equivalence.py`.

Style: ruff (configured in `pyproject.toml`), NumPy-style docstrings for public functions, type hints checked by mypy for the hand-written modules, and `logger` instead of `print`.

After changing configuration fields or what a stage reads and writes, regenerate the reference pages with `python tools/gen_docs.py` (CI checks that they are current).

## Releases

1. Update `src/bpsp/_version.py`, `CHANGELOG.md` and the `version` and `date-released` fields of `CITATION.cff` (semantic versioning).
2. Tag `vX.Y.Z` on `main` and publish a GitHub release with the changelog entry.
3. Archive the release on Zenodo with New version on the existing record, so all versions share one concept DOI.

The tag `v1.0.0` and `paper/` always hold the exact code used for the article.
