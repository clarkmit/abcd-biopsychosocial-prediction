# Concepts

The terms used across the pipeline, the article and these pages.

## Waves (T0 to T4)

The five annual ABCD assessment waves used here. T0 is baseline (ages 9 to 10) and T4 is the fourth annual follow-up (ages 13 to 14). The setting `main.tp_option` picks the wave of the target. Parent self-report (ASR) was only collected at T0, T2 and T4, two years apart, while parent ratings of the child (CBCL) were collected at every wave.

## Measurement categories

Variables were feature-engineered into theoretically relevant, commonly used, or potentially informative measurement categories (domains), such as Other Personality Features, Family Dynamics & Parenting, Technology Use or Structural and DTI. The variable registry (`bpsp.data.registry`) holds 37 adolescent and 32 parent categories across the five waves, 27 and 22 of them at baseline, because some categories only exist at some waves. Figure 1 of the article shows the 30 primary adolescent and 20 parent categories. The registry lists every category and its variables per wave (`CHILD_VARIABLES_BY_TIMEPOINT`, `PARENT_VARIABLES_BY_TIMEPOINT`).

## Variable names and figure labels

The analysis panel uses interpretable variable names. `data/variable_map.csv` gives each one's source ABCD table and original variable name (derived composites, change scores and thresholds are marked `derived`) and its figure label. For accessibility, a display-name dictionary translating coded variables into natural-language labels is applied to the feature-importance outputs and figures (`DISPLAY_NAMES` and `TARGET_DISPLAY_NAMES` in `bpsp.data.preprocessing`).

## Within-category and across-category models

Within-category models fit one stacking ensemble per measurement category, so categories can be compared with each other (Figs 3, 5 and 6). Across-category models use all non-circular measures together, to find the most predictive features and interactions across categories (Figs 7 and 8, Fex1 and Fex2). The setting is `main.split` (`within_categories` or `across_categories`).

## Circularity exclusions

For every target, a structured protocol removes predictors on three criteria: variables already subsumed in the target (for example the CBCL sadness item when predicting depression), predictors with theoretical or ontological overlap with the target that raters could plausibly conflate with it, and likely or ambiguous reverse-causation confounds. Within-category models drop specific features and sometimes whole categories. Across-category models drop features per target. The lists are in `bpsp.data.circularity`. The protocol reduces direct leakage but does not eliminate all rater, method, or ambiguous latent construct covariance.

## Concurrent and prospective prediction

Concurrent prediction uses features from the same wave as the target. Prospective prediction uses features from an earlier wave, for example baseline features (age ~9) predicting outcomes at age 11 (2-year) or age 13 (4-year). The setting is `main.feature_tp_option`: `Match target` for concurrent prediction, or an earlier wave.

## Trait-like and change-based predictors

Trait-like predictors are dispositional characteristics such as personality or cognitive style. Change-based (delta) predictors are changes since the previous wave, computed for theoretically important measures when the previous wave was available (for example other psychopathology, social problems, family conflict, adverse life events and school problems). Delta variables start with `delta_`.

## Targets

The article's Methods pre-specify the primary outcomes as adolescent depression (syndrome and primary-feature composites), anxiety, ADHD and social problems, and parent depression (ASR syndrome and single-item self-reported depressed mood). Other targets, such as externalizing, cognitive and academic outcomes, depression subtypes, happiness and relationship problems, are exploratory contrasts. Core depression targets leave out the secondary features of sleep, appetite and self-harm to focus on the hallmark features of low mood. The higher-order internalizing and externalizing spectra are the CBCL (adolescents) and ASR (parents) broadband scales. In the configuration, the bundles named `Child P-Factor` and `Parent P-Factor` hold these spectra and keep their v1.0.0 names.

## Objective/subjective gradient

An alternative parcellation of the same variables along a seven-level objective-to-subjective gradient, separating child and parent report, from Objective No-Report measures (neuroimaging, experimental tasks, physical measurements, genetic ancestry components) to parent self-report and parent-reported family measures (Fig 4). It is turned on with `setup.objective_subjective_spectrum`.

## Stacking ensemble

CatBoost, XGBoost, random forest and, in within-category models, TabPFN are the base learners. Each produces out-of-fold predictions on the training data, and a meta-learner (linear regression for continuous targets, logistic regression for binary ones) combines them. See [Methods in brief](methods.md).

## Consensus importance

Each base learner's feature importance, normalized to its largest value, multiplied by that learner's contribution to the meta-learner and summed across learners. In within-category models TabPFN does not contribute to it (see [Methods in brief](methods.md)). Because tree-based and SHAP importance can split credit among correlated predictors, importance within highly correlated feature clusters is best read at the cluster level (the `grouped_importance` stage, Supplement S5.4).
