# Reproducing the paper

The analyses in the paper were produced with the notebook frozen in `paper/` (release v1.0.0, Zenodo DOI [10.5281/zenodo.23540757](https://doi.org/10.5281/zenodo.23540757)). This package (v2) gives the same results as that notebook at every stage when the two are run side by side on simulated panels ([Equivalence](equivalence.md)), so either can be used. All model fitting needs approved ABCD data access. Figure numbers follow the article, and Fex numbers refer to its extended figures.

| Results | Stages | Key settings |
|---|---|---|
| Within-category models (Figs 3, 5, 6; S1.13, S1.14) | `main` | `main.split: within_categories` |
| Objective/subjective gradient (Fig 4, Fex6; S1.12) | `main` | `setup.objective_subjective_spectrum: true`, `main.split: within_categories`, `main.tp_option` 0 or 2 |
| Across-category models and Sankey diagrams (Figs 7, 8; Fex1, Fex2, Fex4) | `main`, `batch_summary`, `sankey` | `main.split: across_categories`, `main.batch_mode` |
| Prospective prediction (Fex1; S1.15) | `main` | `main.feature_tp_option` set to an earlier wave than `main.tp_option` |
| Correlation matrices and partial-correlation networks (S1.7.5) | `networks` | |
| Nested CV and calibration (S1.4, S1.11.3, S1.11.10) | `nested_cv` | `nested_cv.n_outer_folds: 5` or `10` |
| Demographic stratification (S1.17, S1.11.7) | `stratified` | |
| Family- and site-grouped CV (S1.5, S1.18.4) | `grouped_cv` | `grouped_cv.robustness_mode` |
| Permutation testing (S1.4.4, S1.11.5) | `permutation` | `permutation.n_permutations: 1000` |
| Stacking audit (S1.1) | `stacking_audit` | |
| PCA baseline, temporal hold-out (S1.3, S1.6) | `baselines` | |
| Domain ablation (S1.19) | `ablation` | |
| Concurrent trajectories, temporal gradient (S1.16.1, S1.16.2) | `trajectories`, `temporal_gradient` | |
| Race/ethnicity and religion, calibration, base learners, grouped permutation importance (S5.1 to S5.4) | `fairness`, `calibration`, `base_learners`, `grouped_importance` | |

`configs/paper_defaults.yaml` contains every setting at its default in the published notebook. Copy it, set `setup.DATA_PATH`, the target (`main.target_options` or `main.batch_mode`) and the stages, then:

```bash
bpsp run my_paper_run.yaml
```

Edge bundling (Fig 2), the cross-lagged panel networks (Fig 10) and SHAP graph visualizations are in the companion repository [abcd-visualizations-and-data](https://github.com/ra2yamamoto/abcd-visualizations-and-data), together with the code that builds the merged analysis panel. The Bayesian mixed models (Fig 9) are not part of this repository.

## Runtime on real data

Within-category single-target runs vary with the data available at each wave. An across-category single-target run (~420 features, N ≈ 10,000) took about 15 to 45 minutes on an A100. The validation stages from `nested_cv` to `temporal_gradient` took 3 to 30 hours for one target, mostly depending on the number of permutations and folds.
