# Equivalence with v1.0.0

Version 2.0.0 restructures the published notebook into a package. It does not re-implement it. This page records how that was checked and what the checks found.

## Method

Both versions were run on the same data with the same settings and compared after every step.

The data were panels of simulated subjects over five waves, resampled from the fully simulated panel in `data/simulated/`, with a planted signal so that the models learn something and any difference in the code path shows up in the numbers. The four test plans in `tests/regression/plans/` use 400 subjects. No ABCD data were used.

v1.0.0 is `paper/ABCDMLPipeline_v1.0.0.ipynb`, executed cell by cell in one IPython namespace, as Colab runs it, with the form fields set to the values of the run plan before each cell. v2.0.0 runs the same plan through `bpsp.Pipeline`, stage by stage.

Both runs use the same seeds (the pipeline's own), `PYTHONHASHSEED=0` (set by the test harness) and the CPU only, with smaller base learners and fewer resamples so that a run takes minutes rather than days. TabPFN and PyTorch are not installed in the test environment, so on both sides TabPFN is replaced by a small linear stand-in (`tests/regression/stubs/`), which only the within-category plan fits. The test plans also seed Python's and NumPy's global random generators identically before every step, because two v1.0.0 figures draw from them unseeded: `ncv_fold_dist.png` (random jitter of the outer-fold points) and `stratified_feature_importance.png` (Kernel SHAP sampling). Without that seeding these two figures change from run to run in v1.0.0 itself, while the tables behind them stay the same.

After each step three things are compared:

1. Shared results. Every object that later stages read (train and test matrices and targets, metrics, batch archives, consensus importance, ensemble weights, validation tables) is fingerprinted: arrays and data frames are hashed after rounding to six decimals, and scalars are compared with a relative tolerance of 10<sup>-6</sup>.
2. Printed output. The full text output of each analysis stage, line by line. Left out are stage start and end lines, lines that contain "Warning" or come from Python's warnings (including the pipeline's own warning messages), traceback layout (the error itself is compared), progress bars, blank lines and IPython's echo of a cell's value. Setup output is not compared, because importing the modules prints the same lines as notebook cells 0 to 5 in a different order.
3. Files. Every table and figure written: CSV files by value, PNG files byte for byte, PDF files byte for byte apart from their embedded creation date.

The deliberate changes listed below are applied to v1.0.0 before the comparison, exactly as listed in `tests/regression/equivalence.py`. Nothing else is edited.

## Results

| Test plan | Configuration | Steps compared | Shared results | Printed output | Files |
|---|---|---|---|---|---|
| `regression_across` | `internal_D_p` at T2, across categories | setup, `main`, `networks`, `smoke_test`, `nested_cv`, `grouped_cv`, `permutation`, `fairness`, `calibration`, `base_learners`, `grouped_importance` | all match | identical | all 30 identical |
| `validation_suite` | `internal_D_p` at T2, across categories | setup, `main`, `diagnostics`, `stratified`, `stacking_audit`, `baselines`, `ablation`, `trajectories`, `temporal_gradient` | all match | identical | all 23 identical |
| `classification_batch` | "Child Depression Worsening" batch: four binary targets at T2, across categories | setup, `main`, `batch_summary`, `sankey`, `nested_cv`, `fairness`, `calibration`, `base_learners`, `grouped_importance` | all match | identical | all 26 identical |
| `within_spectrum` | `internal_D_p` at T2, within categories (one model per category, with TabPFN), objective/subjective gradient | setup, `main`, `base_learners` | all match | identical | all 4 identical |

Between them the four plans cover every stage, and all four pass with `pytest -m slow` on the released code (Python 3.12, the environment of `requirements-lock.txt`).

One stage stops with an error in both versions. In the classification test plan, the `fairness` stage stops at the third of four targets with `ValueError: Axis limits cannot be NaN or Inf` while drawing its forest plot, because the 400-subject panel leaves the groups too small for confidence limits. v1.0.0 stops at the same point with the same error, and the stages after it run normally in both. It is listed in [Known issues](known_issues.md).

None of the runs reaches one of the undefined names listed in [Known issues](known_issues.md).

## Deliberate source changes

These changes were made on purpose and each is covered by the runs above. The first seven leave every result and printed line as it was. The last four change printed text, figure titles or values in some cases, so the equivalence harness applies them to v1.0.0 before comparing (`V1_SOURCE_EDITS`, `V1_TEXT_EDITS`, `V2_MESSAGE_EDITS` and `V2_DROPPED_LINES` in `tests/regression/equivalence.py`).

| Change | Why |
|---|---|
| Each session works on copies of the variable registry | The optional objective/subjective gradient edits the registry in place. Copies keep sessions in one process independent. |
| TabPFN and PyTorch imports are optional | CPU-only installs work. Without them TabPFN is skipped and the reason logged, as the notebook already did for across-category runs, and the stacking audit checks the other base learners. |
| `print` became logging through the `bpsp` logger | The console shows the same text, and `run.log` keeps it. Checked by comparing syntax trees (identical apart from the logging calls) and by the printed-output comparison. |
| Imports inside a stage's main code are declared global | In the notebook they bound globals. Inside a function they would otherwise shadow the module-level import on paths where they do not run. |
| `CatBoostPruningCallback` imported from `optuna_integration` instead of `optuna.integration` | It is the same class. The old path is deprecated and will be removed in Optuna 6. |
| Box-drawing characters in two f-strings written literally | Python 3.11 does not accept backslash escapes inside f-string expressions. The printed text is identical. |
| Comments and docstrings tidied, a block of commented-out code removed, and the private `_run_val_NN` helpers renamed `_run_for_current_target` | Readability. A syntax-tree comparison that ignores docstrings shows no other change. |
| The main stage takes its CatBoost settings from `PIPELINE_MODEL_CONFIGS` | v1.0.0 kept a second copy of the same values in the main runner, so changing the central dictionary did not reach the main stage. The defaults are identical, so results only change when settings are overridden (`model_overrides`). |
| The main stage reads the ensemble's metrics from `meta_model` directly | v1.0.0 first looked up an `ensemble` entry the metrics never contain, printed `[WARNING] 'ensemble'` on every run and then used `meta_model`. The numbers are the same and the misleading line is gone. |
| Adjusted R² is reported as missing (NaN) when the test rows do not exceed the number of predictors plus one | The formula is undefined there, and v1.0.0 printed values above 1. This happens on small panels like the demo's. At the article's sample sizes the values are unchanged. |
| Messages and figure titles name stages and settings instead of notebook cells, the Colab form or review points (`[POINT 1]` became `[nested_cv]`), "Enhanced" was dropped from messages, the permutation report is titled after the CatBoost model it tests, and the networks stage no longer prints a development note about a target-inclusive network that v1.0.0 does not implement | A package has stages, not cells or review points. |

## What the conversion found

Converting the notebook surfaced issues that v1.0.0 has as well. They are kept as they are, so that the two versions agree, and are listed in [Known issues](known_issues.md):

* 15 names are referenced but never defined in v1.0.0, and one is used before it is assigned in its function, all in rarely used branches
* three label dictionaries repeat keys (the later label is used)
* two figures use unseeded random numbers, and the fairness forest plot fails when confidence limits cannot be computed
* TabPFN does not contribute to the consensus importance of within-category models

## Running the check yourself

```bash
pip install -e ".[dev]"
pytest -m slow    # four smaller runs on a 400-subject panel
```

or step by step with `tests/regression/equivalence.py` (`panel`, `notebook`, `package`, `compare`). The equivalence workflow on GitHub Actions runs the same tests weekly.
