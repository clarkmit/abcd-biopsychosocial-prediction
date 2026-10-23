# Stages

A run is setup followed by any sequence of stages. Each stage is one module in `src/bpsp/` and corresponds to one cell of the published notebook (`paper/ABCDMLPipeline_v1.0.0.ipynb`). Stages communicate through the run context (`bpsp.ctx`, see [Architecture](architecture.md)), so they should run in the order below. Validation stages need `main` first.

## Setup (always runs)

| Module | Notebook cell | Role |
|---|---|---|
| `bpsp.reporting` | 0 | report state and cache hooks |
| `bpsp.data.registry` | 1 | variable registry: the measurement categories and their variables for adolescents and parents at T0 to T4; outcome lists |
| `bpsp.data.circularity` | 2 | per-target circularity exclusions (variables subsumed in the target, construct overlap, reverse-causation confounds) |
| `bpsp.data.spectrum` | 3 | optional objective/subjective gradient (`setup.objective_subjective_spectrum`) |
| `bpsp.data.preprocessing` | 4 | load the panel, task detection, within/across-category matrices, train/test partitioning, display names |
| `bpsp.models.stacking` | 5 | stacking ensemble and model hyperparameters |

## Analysis stages

| Stage | Module | Cell | What it produces | Paper |
|---|---|---|---|---|
| `main` | `analysis.runner` | 6 | stacking ensemble per target and feature category; batch mode over target bundles; consensus importance, SHAP | Figs 3 to 8 |
| `batch_summary` | `reports.batch_summary` | 7 | metric dashboard across a classification batch | |
| `sankey` | `reports.sankey` | 8 | feature domain to target Sankey diagrams | Figs 7, 8; Fex1, Fex2, Fex4 |
| `networks` | `interpretation.networks` | 9 | correlation matrices, partial-correlation networks and dendrograms of top features | S1.7.5 |
| `diagnostics` | `interpretation.diagnostics` | 10 | model diversity, bootstrap stability, learning curves, SHAP beeswarm, effect sizes | |
| `smoke_test` | `validation.smoke_test` | 11 | quick end-to-end check of the fitted pipeline | |
| `nested_cv` | `validation.nested_cv` | 12 | outer-fold nested CV with 95% CIs, calibration and decision curves | S1.4, S1.11.3, S1.11.10 |
| `stratified` | `validation.stratified` | 13 | performance by sex, parent education, SES, adverse life events | S1.17, S1.11.7 |
| `grouped_cv` | `validation.grouped_cv` | 14 | family-grouped and site-grouped CV, with whole families or sites held out | S1.5, S1.18.4 |
| `permutation` | `validation.permutation` | 15 | permutation tests (single CatBoost model) with Phipson-Smyth p-values | S1.4.4, S1.11.5 |
| `stacking_audit` | `validation.stacking_audit` | 16 | base-model contributions, meta-learner weights, leakage checks | S1.1 |
| `baselines` | `validation.baselines` | 17 | PCA baseline and temporal hold-out | S1.3, S1.6 |
| `ablation` | `validation.ablation` | 18 | domain ablation and feature-domain analysis | S1.19 |
| `trajectories` | `validation.trajectories` | 19 | concurrent performance across timepoints | S1.16.1 |
| `temporal_gradient` | `validation.temporal_gradient` | 20 | concurrent versus prospective decay | S1.16.2 |
| `fairness` | `validation.fairness` | 21 | performance by race/ethnicity and religion, partial-correlation confound check | S5.1 |
| `calibration` | `validation.calibration` | 22 | calibration and Platt/isotonic recalibration | S5.2 |
| `base_learners` | `validation.base_learners` | 23 | each base learner versus the stacked ensemble on held-out data | S5.3 |
| `grouped_importance` | `validation.grouped_importance` | 24 | correlation-aware grouped permutation importance | S5.4 |

Supplement numbers follow the article. The published notebook's cell titles give the last four stages under earlier numbers (S1.21 to S1.24). Settings for every stage are listed in [Configuration](configuration.md), and the model each validation stage uses in [Methods in brief](methods.md).
