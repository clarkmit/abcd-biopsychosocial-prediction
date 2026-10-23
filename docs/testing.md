# Testing

| Suite | Command | What it checks | Time |
|---|---|---|---|
| Unit | `pytest -m "not integration and not slow"` | run context, configuration and its validation, CLI, logging, stage registry, variable registry and circularity exclusions, leakage guards, the simulated example panel, the data guard, and that the documented v2 edits match the published notebook | seconds |
| Integration | `pytest -m integration` | complete runs on simulated data (regression and classification), the CLI, and the known-answer checks below | about 2 minutes |
| Equivalence | `pytest -m slow` | the package gives the same results as the published notebook (`paper/`) on the same simulated panel | about 45 minutes |

CI runs these checks on every pull request and every push to `main`:

* ruff, the data guard, and actionlint on the workflow files
* mypy on the hand-written modules
* the unit tests on Python 3.11, 3.12 and 3.13, with a coverage report
* the integration tests and a CLI demo
* the quickstart notebook, a wheel build and the documentation build

The equivalence workflow runs weekly and on demand, in the pinned environment of `requirements-lock.txt`. Times are for a two-core machine.

Unit-test coverage of the hand-written modules (pipeline, configuration, run context, CLI, logging, simulation) is 94%. The stage modules are exercised by the integration and equivalence tests instead.

## What the tests show

### Leakage guards

`tests/test_leakage_guards.py`, on the simulated panel, checks that:

* the target and every variable on its circularity-exclusion list are absent from the predictors
* scaling statistics come from training rows only, and training and test rows are disjoint (70/30 split)
* the meta-learner is fitted on out-of-fold base-learner predictions, never on predictions for rows a base learner was trained on
* family-grouped cross-validation keeps all members of a family in the same fold
* permutation p-values use the Phipson-Smyth correction, (b + 1) / (m + 1), so they are never zero

### Known answers

On a panel with a signal planted in four predictors (`bpsp.simulation`), those four rank first in the consensus importance and the ensemble's held-out R² is above 0.25 (`tests/test_known_answer.py`). On the shipped panel, whose columns are independent, the held-out R² stays below 0.1 (`tests/test_integration.py`).

### Registry integrity

Outcome lists do not overlap, feature pools are not empty, and a session does not change the published variable registry (`tests/test_registry_integrity.py`).

### Equivalence with v1.0.0

The frozen notebook runs cell by cell in an IPython shell and the package stage by stage, on the same simulated panel. After every step the two runs are compared on fingerprints of every shared result (splits, metrics, archives, validation tables), their printed output and every file they write. Four plans in `tests/regression/plans/` cover every stage between them: a regression run with most validation stages, a second regression run with the remaining validation stages and diagnostics, a classification batch, and a within-category run with the objective/subjective gradient. The results are in [Equivalence](equivalence.md).

## What the tests do not show

They do not reproduce the article's findings. Every test uses simulated data. The scientific results depend on ABCD data, which cannot be used in CI, and were produced with v1.0.0 (`paper/`).

They do not show that v1.0.0 itself is right. The equivalence tests show that v2.0.0 does what v1.0.0 does, including its [known issues](known_issues.md), not that every analysis choice is correct.

They do not cover full scale. The tests use small panels (300 to 1,200 simulated participants) and small models, so runtime, memory use and TabPFN on a GPU at the scale of the published analyses are not tested.

Apart from the guards above, the validation stages (calibration, fairness, ablation and others) are checked for running end to end and for matching v1.0.0, not against independently computed reference values.
