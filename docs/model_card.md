# Model card

## Model details

A two-stage stacking ensemble. The base learners are CatBoost, XGBoost, random forest and TabPFN (TabPFN only in within-category models). The meta-learner is linear regression for continuous targets and logistic regression for binary targets, fitted on the base learners' out-of-fold predictions, their pairwise products and their mean. Details are in [Methods in brief](methods.md).

| Field | Details |
|---|---|
| Contributors | Clark Roberts (conceptualization, feature-engineering, ML pipeline programming, implementation of manuscript models, statistical analysis, visualization), Zhafira E. Fawnia (Ensemble ML programming) |
| Versions | v1.0.0, the notebook used for the article; v2.0.0, this package, with the same results on simulated panels ([Equivalence](equivalence.md)) |
| License | MIT for the code. ABCD data are governed by the ABCD Data Use Certification. |

## Intended use

Research on the comparative predictive utility of biological, psychological, social and environmental measurement domains for psychiatric, cognitive and social outcomes, at the population level.

The models are not meant for clinical decisions, diagnosis, screening or individual risk prediction. They were fitted and evaluated in a research cohort and have not been validated for any clinical population or setting.

## Training data

ABCD Study Release 5.1, five annual waves (T0 to T4): 11,864 adolescents aged 9 to 14 and about 11,859 parents and caregivers aged 23 to 75. About 1,078 variables from 69 scales, organized into 37 adolescent and 32 parent measurement categories across the waves. Imputation and scaling inside the pipeline are fitted on training rows only. Neuroimaging components were computed in the analysis dataset beforehand, on the whole sample ([Methods in brief](methods.md)).

## Evaluation

Primary estimates come from a held-out 30% test split. Nested cross-validation (5 or 10 outer folds) with 95% confidence intervals, family-grouped and site-grouped cross-validation with whole families or sites held out, temporal hold-out, PCA baselines and 1,000-permutation tests add to it. Regression is scored with R², adjusted R², MAE and RMSE. Classification is scored with AUC-ROC, AUC-PR, accuracy, precision, recall and F1, with calibration (expected calibration error, Brier score, Platt and isotonic recalibration) and decision curves.

## Subgroups and fairness

Performance is reported by child sex, parent sex and education, SES, adverse life events, race/ethnicity and religion, with a partial-correlation check for demographic confounds (stages `stratified` and `fairness`).

## Limitations

Many predictors and outcomes share raters and measurement methods, and shared rater and method variance remain central to interpretation. The predictive advantage of subjective and informant ratings is partly inseparable from shared rater variance, shared method variance, prior symptom stability and construct covariance.

The circularity protocol reduces direct leakage but does not eliminate all rater, method, or ambiguous latent construct covariance.

Associations are observational. Feature importance is exploratory and descriptive, and directionality cannot be inferred from it.

Later waves overlap the COVID-19 pandemic, so developmental effects cannot be fully separated from pandemic-period perturbations.

Several domains (neuroimaging, experimental tasks, Fitbit) were collected only in adolescents, so the adolescent and parent analyses are not fully symmetric.

All models were developed and evaluated within ABCD. How they generalize to other cohorts is untested.

## Ethical considerations

The models and their outputs must not be used to make decisions about individuals. Participant privacy is protected by the Data Use Certification and by the safeguards in [Data access and governance](data.md).
