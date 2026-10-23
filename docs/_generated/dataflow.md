| Stage | Module | Reads from `ctx` | Writes to `ctx` |
|---|---|---|---|
| `setup: reporting` | `reporting` | none | none |
| `setup: registry` | `data.registry` | none | `child_variables_for_each_time_point`, `parent_variables_for_each_time_point`, `t0_variables`, `t1_variables`, `t2_variables`, `t3_variables`, `t4_variables`, `variables_for_each_time_point` |
| `setup: circularity` | `data.circularity` | none | none |
| `setup: spectrum` | `data.spectrum` | `child_variables_for_each_time_point`, `parent_variables_for_each_time_point` | `t0_variables`, `t2_variables`, `variables_for_each_time_point` |
| `setup: preprocessing` | `data.preprocessing` | `child_variables_for_each_time_point`, `is_regression`, `parent_variables_for_each_time_point`, `t0_variables`, `t1_variables`, `t2_variables`, `t3_variables`, `t4_variables` | `sample`, `variables_for_each_time_point` |
| `setup: stacking` | `models.stacking` | `DISPLAY_NAMES`, `TARGET_DISPLAY_NAMES` | `ENSEMBLE_VIZ_CONFIG`, `PIPELINE_MODEL_CONFIGS`, `category_results` |
| `main` | `analysis.runner` | `DISPLAY_NAMES`, `ENSEMBLE_VIZ_CONFIG`, `PIPELINE_MODEL_CONFIGS`, `category_metrics`, `feature_importance`, `sample` | `X_test`, `X_train`, `category_metrics_dict`, `category_results`, `consensus_df`, `exported_within_dict`, `group`, `isolate`, `last_fitted_model`, `metrics_dict_full`, `model_weights`, `results_summary`, `run_all`, `run_visualizations`, `sample_valid`, `sample_valid_test`, `sample_valid_train`, `show_feature_importance_plots`, `y_test`, `y_train` |
| `batch_summary` | `reports.batch_summary` | `RESULTS_DIR`, `TARGET_DISPLAY_NAMES`, `is_regression` | none |
| `sankey` | `reports.sankey` | `TARGET_DISPLAY_NAMES` | none |
| `networks` | `interpretation.networks` | `DISPLAY_NAMES`, `FEATURE_TO_CATEGORY`, `TARGET_DISPLAY_NAMES`, `X_train`, `consensus_df`, `exported_within_dict`, `last_fitted_model`, `metrics_dict_full`, `model_weights`, `y_train` | `batch_corr_dict`, `batch_feat_list`, `batch_feat_list_corr`, `batch_pcor_dict` |
| `diagnostics` | `interpretation.diagnostics` | `DISPLAY_NAMES`, `X_test`, `X_train`, `consensus_df`, `create_lightweight_model`, `detect_task_type`, `last_fitted_model`, `model_weights`, `y_test`, `y_train` | none |
| `smoke_test` | `validation.smoke_test` | `X_test`, `X_train`, `y_test`, `y_train` | `create_validation_model` |
| `nested_cv` | `validation.nested_cv` | `PIPELINE_MODEL_CONFIGS`, `X_test`, `X_train`, `binary_or_categorical_outcome_variables`, `continuous_outcome_variables`, `create_validation_model`, `exported_within_dict`, `make_run_key`, `metrics_dict_full`, `y_test`, `y_train` | `is_classification`, `is_regression`, `reporter_point1` |
| `stratified` | `validation.stratified` | `DISPLAY_NAMES`, `PIPELINE_MODEL_CONFIGS`, `TARGET_DISPLAY_NAMES`, `X_test`, `X_train`, `binary_or_categorical_outcome_variables`, `continuous_outcome_variables`, `exported_within_dict`, `make_run_key`, `metrics_dict_full`, `sample_valid_test`, `y_test`, `y_train` | `all_rows`, `is_classification`, `is_regression` |
| `grouped_cv` | `validation.grouped_cv` | `PIPELINE_MODEL_CONFIGS`, `X_test`, `X_train`, `binary_or_categorical_outcome_variables`, `continuous_outcome_variables`, `exported_within_dict`, `make_run_key`, `metrics_dict_full`, `sample_valid`, `y_test`, `y_train` | `is_classification`, `is_regression`, `robust_reporter` |
| `permutation` | `validation.permutation` | `X_test`, `X_train`, `binary_or_categorical_outcome_variables`, `continuous_outcome_variables`, `exported_within_dict`, `make_run_key`, `metrics_dict_full`, `y_test`, `y_train` | `is_classification`, `is_regression`, `perm_reporter` |
| `stacking_audit` | `validation.stacking_audit` | `X_train`, `binary_or_categorical_outcome_variables`, `continuous_outcome_variables`, `exported_within_dict`, `make_run_key`, `metrics_dict_full`, `model`, `pipeline_model`, `y_train` | `is_classification`, `is_regression`, `stacking_auditor` |
| `baselines` | `validation.baselines` | `DISPLAY_NAMES`, `X_test`, `X_train`, `bootstrap_ci`, `group`, `metrics_dict_full`, `results_summary`, `y_test`, `y_train` | none |
| `ablation` | `validation.ablation` | `X_test`, `X_train`, `bootstrap_ci`, `y_test`, `y_train` | none |
| `trajectories` | `validation.trajectories` | `across_categories_data`, `group`, `sample` | none |
| `temporal_gradient` | `validation.temporal_gradient` | `group`, `sample` | none |
| `fairness` | `validation.fairness` | `TARGET_DISPLAY_NAMES`, `X_test`, `sample`, `y_test` | `fairness_partial_corr_df`, `fairness_race_religion_df` |
| `calibration` | `validation.calibration` | `TARGET_DISPLAY_NAMES`, `X_test`, `y_test` | `calibration_classification_df`, `calibration_regression_df` |
| `base_learners` | `validation.base_learners` | `TARGET_DISPLAY_NAMES`, `metrics_dict_full` | `base_learner_all_models_df`, `base_learner_by_target_df` |
| `grouped_importance` | `validation.grouped_importance` | `DISPLAY_NAMES`, `TARGET_DISPLAY_NAMES`, `X_test`, `y_test` | `grouped_perm_clusters_df`, `grouped_perm_features_df`, `grouped_perm_groups_df` |

Results only: each stage also reads its own settings (see [Configuration](configuration.md)) through
`ctx`. Names starting with an underscore (internal) are omitted.
