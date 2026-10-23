# Changelog

Changes are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-23

Version 2 changes transformed notebook into an installable & tested Python package. On the equivalence runs described in `docs/equivalence.md` it gives the same results as v1.0.0, apart from the deliberate changes listed under Changed.

### Added
- `bpsp` package (`src/bpsp/`): one module per notebook cell, grouped into `data`, `models`, `analysis`, `reports`, `interpretation` and `validation`.
- `Pipeline` orchestration with setup, ordered stages, per-stage timings and an output directory per run.
- Typed configuration (`PipelineConfig`) for every former Colab form field, with the notebook's values as defaults. Drop-down choices and types are validated on load, misspelled settings are rejected, and configurations are read and written as YAML.
- Command-line interface: `bpsp run`, `bpsp stages`, `bpsp init-config`, `bpsp example-panel`.
- Explicit run context (`bpsp.ctx`) for state shared between stages.
- Logging through the `bpsp` logger. `bpsp run` prints tables as text and saves the console output (`run.log`) and the resolved configuration with each run.
- Tests: unit, end-to-end on simulated data, known-answer, and equivalence with the published notebook. GitHub Actions CI with ruff, mypy, coverage and actionlint.
- Documentation site, including Concepts and Methods in brief pages, a model card, a data governance guide, a contribution guide and a citation file.
- Data-protection guards: `.gitignore` rules, a pre-commit hook and a CI check that block data, statistics-package and archive files, large files and ABCD participant identifiers.
- Optional `tabpfn` extra. CPU-only installs leave TabPFN out of the ensemble and the stacking audit automatically.
- `setup.objective_subjective_spectrum` switch for the objective/subjective gradient (notebook cell 3).
- `bpsp.simulation`: a simulated panel with a known signal, used by the demo, the quickstart notebook and the known-answer tests.

### Changed
- The main stage takes its CatBoost settings from `PIPELINE_MODEL_CONFIGS`, so `model_overrides` reach it. v1.0.0 kept a second copy of the same values in the runner. Defaults are unchanged.
- The main stage reads the ensemble's metrics from `meta_model` directly. v1.0.0 first looked up an entry that never exists and printed `[WARNING] 'ensemble'` on every run.
- Messages and figure titles name stages and settings instead of notebook cells, the Colab form or review points (`[POINT 1]` became `[nested_cv]`), and no longer say "Enhanced". The permutation report is titled after the CatBoost model it tests. A development note printed by the networks stage was removed.
- Comments and docstrings were tidied, and docstrings that described the permutation test as refitting the stacking ensemble were corrected.
- Each session works on its own copies of the variable registry, so runs in one process do not affect each other.
- All tables and figures are written to the configured output directory.
- `CatBoostPruningCallback` is imported from `optuna_integration` (the same class). The `optuna.integration` path is deprecated and will be removed in Optuna 6.
- Printed output goes through logging, with the same text on the console.
- The versions v1.0.0 was tested with moved from `requirements.txt` to `paper/requirements.txt`, next to the v1.0.0 notebook. `requirements-lock.txt` pins the environment in which v2.0.0 was verified.

## [1.0.0] - 2026-10-21

Code used for the article, as a Colab notebook (`paper/ABCDMLPipeline_v1.0.0.ipynb`).
Archived on Zenodo: [10.5281/zenodo.23540757](https://doi.org/10.5281/zenodo.23540757).

[2.0.0]: https://github.com/clarkmit/abcd-biopsychosocial-prediction/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/clarkmit/abcd-biopsychosocial-prediction/releases/tag/v1.0.0
