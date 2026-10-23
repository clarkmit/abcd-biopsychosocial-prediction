# Methods in brief

What the pipeline does, following the Methods of the article. Terms are explained in [Concepts](concepts.md), the settings in [Configuration](configuration.md), and the article's REFORMS-ML checklist is in Supplement S1.1.

## Features and targets

About 1,078 unique variables were selected or created from 69 ABCD scales on three criteria: potential theoretical relevance to psychopathology, being most informative for comparing hierarchical differences and informing later research, and prior empirical evidence of association. They are organized into measurement categories per wave, with change-based (delta) variables where the previous wave was available. Each target gets its own circularity exclusions before any model is fitted.

Neuroimaging features were prepared in the analysis dataset before this pipeline runs: per modality, columns and rows with more than 20% missing data were dropped, remaining gaps imputed with MICE, and the data reduced to 75 principal components. That preparation used the whole sample, not only the training rows. Genetic ancestry components were previously derived from the ABCD genetic data with PC-AiR (Fan et al., 2023).

Inside the pipeline, for each target and wave:

1. Variables with more than 70% missing values are dropped.
2. The rows are split 70/30 into training and test sets.
3. Missing feature values are imputed by k-nearest neighbors (k = 5), fitted on the training rows.
4. Constant features are removed, and numeric features are clipped at five standard deviations from their mean. The mean and standard deviation are computed within the training rows for the training set and within the test rows for the test set.
5. Features are standardized with a scaler fitted on the training rows.

## Stacking ensemble

| Base learner | Settings (fixed a priori) | Used in |
|---|---|---|
| CatBoost | 1,000 iterations, learning rate 0.05, depth 6, `l2_leaf_reg` 3, `border_count` 128 | all models |
| XGBoost | 500 estimators, learning rate 0.1, maximum depth 6, `min_child_weight` 1, `subsample` 0.8, `colsample_bytree` 0.8 | all models |
| Random forest | 200 estimators, maximum depth 10 (for regression also `min_samples_split` 5, `min_samples_leaf` 2) | all models |
| TabPFN v2 | its own defaults | within-category models |

For classification, CatBoost and random forest use balanced class weights, and for binary targets XGBoost's `scale_pos_weight` is the square root of the ratio of negative to positive cases. The main settings live in one dictionary, `PIPELINE_MODEL_CONFIGS` in `bpsp.models.stacking`, and a run can override them with `model_overrides`. The remaining settings above are fixed in `StackingEnsembleModel`. TabPFN is left out of across-category models because of its known instability above ~500 features, little added performance at that scale and high computational cost. When a within-category feature set has more than 100 features, TabPFN gets the 100 ranked highest by a small CatBoost model (100 iterations) fitted on the training partition.

Each base learner produces out-of-fold predictions on the training partition through an internal five-fold split. The meta-learner, linear regression for continuous targets and logistic regression for binary targets, is fitted on those out-of-fold predictions, their pairwise products and their mean. For classification its regularization strength is chosen by five-fold cross-validation within the training data. No other hyperparameters are tuned. The base learners are then refit on the full training partition. A global random seed (42) is used throughout, with shuffled splits (stratified for classification).

An optional hyperparameter search from the notebook (`main.hyperparameter_tuning`, which for the stacking ensemble searches CatBoost settings with Optuna when `main.with_optuna` is also on) is off by default and was not used for the article.

## Evaluation

Primary estimates come from a 70/30 train/test split, stratified for classification targets. The held-out 30% is not used to fit any model. Regression is scored with R², adjusted R², MAE and RMSE, and classification with accuracy, precision, recall, F1, AUC-ROC and AUC-PR, plus thresholds and calibration in the validation stages.

Adjusted R² applies the usual penalty for the number of candidate predictors to the held-out R², so it is a conservative reference value. It is undefined when the test rows do not exceed the number of predictors plus one, and is then reported as missing.

## Feature importance

Consensus importance is each base learner's feature importance, scaled to its largest value, multiplied by that learner's weight in the meta-learner and summed across learners. A learner's weight is the absolute meta-learner coefficient on its out-of-fold prediction, as a share of the total over learners. In within-category models TabPFN does not contribute: the consensus step computes no importance for it, so the consensus comes from CatBoost, XGBoost and random forest with their meta-learner weights ([Known issues](known_issues.md)). SHAP values are computed as well. Because tree-based and SHAP importance can split credit among correlated predictors, importance within highly correlated clusters should be read at the cluster level (`networks` for correlations among top features, `ablation` for domain permutation importance and top-K ablation, `grouped_importance` for grouped permutation importance).

## Validation suite

Each validation stage reuses the setup and the `main` stage of the same run.

| Stage | What it checks | Model | Supplement |
|---|---|---|---|
| `nested_cv` | nested cross-validation within the training partition, 5 or 10 outer folds, 95% CIs; calibration and decision curves for classification | the full ensemble, rebuilt in every outer fold | S1.4, S1.11.3, S1.11.10 |
| `stratified` | held-out performance by child sex, parent sex and education, SES and adverse life events, 1,000 bootstrap resamples | the full ensemble, refit on the training rows | S1.17, S1.11.7 |
| `grouped_cv` | family-grouped and site-grouped cross-validation, with whole families or whole sites held out in each fold | the full ensemble | S1.5, S1.18.4 |
| `permutation` | 1,000 label permutations, upper-tailed p-values with the Phipson-Smyth correction | one CatBoost model (250 iterations), for the observed score and every permutation, by five-fold CV on all rows | S1.4.4, S1.11.5 |
| `stacking_audit` | in every outer fold, each row falls in exactly one inner validation fold and the meta-learner sees only out-of-fold predictions | its own base learners (main settings) and a meta-learner on their out-of-fold predictions, rebuilt in each outer fold | S1.1 |
| `baselines` | PCA baseline and temporal hold-out | one CatBoost model (300 iterations) | S1.3, S1.6 |
| `ablation` | objective versus subjective ablation, per-domain ablation, domain permutation importance, top-K ablation | one CatBoost model (300 iterations) | S1.19 |
| `trajectories` | concurrent performance at each wave | one CatBoost model (300 iterations) | S1.16.1 |
| `temporal_gradient` | transfer between waves and decay with temporal distance | one CatBoost model (300 iterations) | S1.16.2 |
| `fairness` | performance by race/ethnicity and religion, with a partial-correlation adjustment | the fitted ensemble, not refit | S5.1 |
| `calibration` | calibration and Platt and isotonic recalibration | the fitted ensemble, not refit | S5.2 |
| `base_learners` | each base learner against the ensemble on the held-out data | the fitted models | S5.3 |
| `grouped_importance` | permutation importance of correlated clusters and of change-based versus level features | the fitted ensemble, not refit | S5.4 |

In the article, held-out and nested estimates differed by at most 0.036 R² and 0.044 AUC (Supplement S1.3, S1.4), and Supplement S5.3 compares the ensemble with each base learner.

Analyses that are not part of this repository are listed in [Reproducing the paper](reproducing.md).
