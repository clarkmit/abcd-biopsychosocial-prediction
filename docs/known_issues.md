# Known issues inherited from v1.0.0

Version 2.0.0 keeps the behavior of v1.0.0, including 15 names (16 occurrences) that v1.0.0 references but never defines (lint rule F821) and one name used before it is assigned in its function (F823). They sit in rarely used branches (legacy model types, optional analyses, fallbacks inside `try` blocks), and none is reached by the configurations in the equivalence runs. Each occurrence is marked in the code with `# noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)` or the F823 equivalent, so linting stays strict everywhere else.

Fixing them changes behavior on those branches, so they are left for a separate release with its own equivalence report rather than mixed into the restructuring.

--8<-- "docs/_generated/known_issues_table.md"

## Other findings kept as written

Three label dictionaries repeat keys: `delta_not_liked_p` in `DISPLAY_NAMES`, `anhedonia_B_k` and `depressed_mood_B_k` in `TARGET_DISPLAY_NAMES`, and `suicidal_p` in `_BATCH_TARGET_COLORS`. As in v1.0.0 the later entry is used, which only affects figure labels and colors.

Comparisons such as `df["high_ale"] == True` are element-wise pandas operations and are correct as written. The corresponding lint rule is switched off for the stage modules.

Two figures draw from NumPy's global random generator, which v1.0.0 never seeds: the horizontal jitter of the outer-fold points in `ncv_fold_dist.png` (nested CV) and the Kernel SHAP sampling behind `stratified_feature_importance.png` (stratified stage, when no tree explainer applies). They differ slightly from run to run. The tables behind them matched exactly in the equivalence runs.

The diagnostics stage's effect-size and cross-validation versus held-out comparisons stop with a feature-count mismatch (for example 550 versus 519 features). They score the last fitted model on the run context's test matrix as it was before the main stage's own preprocessing (categorical encoding and removal of constant and duplicate columns). The stage logs the error and continues.

The fairness stage's forest plot raises `ValueError: Axis limits cannot be NaN or Inf` when no confidence limits can be computed for a target, which happens when groups are very small (seen on the 400-subject test panel). The stage then stops at that target, and later stages are unaffected.

When a within-category feature set has more than 100 features, the CatBoost screen that picks TabPFN's 100 features is fitted on the whole training partition before the internal five-fold split. TabPFN's out-of-fold predictions, and so the meta-learner's inputs, have therefore seen that screening. The held-out test estimates are not affected.

In within-category models, TabPFN does not contribute to the consensus importance. The consensus step has no importance method for it, logs "Successfully calculated importance for tabpfn", then fails on that model and logs "Could not calculate importance for tabpfn". The consensus therefore comes from CatBoost, XGBoost and random forest, and TabPFN's share of the meta-learner weight is left out rather than redistributed. TabPFN's permutation importance is computed separately (`get_feature_importance`).

Outlier clipping (at five standard deviations from the mean) uses statistics computed within each set: the training rows for the training set and the test rows for the test set. No outcome information is involved, but the test set's bounds come from the test rows themselves. Imputation and scaling are fitted on the training rows only.

Messages are kept as v1.0.0 printed them, apart from the documented edits, so that the two versions can be compared line by line. A few overstate what runs, such as "Successfully calculated importance for tabpfn" above.
