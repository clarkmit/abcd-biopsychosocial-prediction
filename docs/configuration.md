# Configuration

A run is configured by one YAML file (or a `PipelineConfig` object). Every key is optional; missing keys keep the
published defaults. `bpsp init-config FILE` writes a file with every setting.

```yaml
stages: [main, nested_cv]          # stages to run after setup, in order
output_dir: outputs/my_run         # tables, figures, run.log, config.resolved.yaml
model_overrides:                   # optional hyperparameter overrides per base learner
  catboost: {iterations: 300}
setup:
  DATA_PATH: /secure/abcd/CLEAN_ABCD_5.1_panel.csv
main:
  target_options: depress_D_p
  split: within_categories
nested_cv:
  n_outer_folds: 10
```

Setting names are identical to the form fields of the published notebook, so each one can be traced to the
paper version. Settings with a fixed set of values are validated when the configuration is loaded.

## Settings you will change most

| Setting | What it does |
|---|---|
| `setup.DATA_PATH` | the merged analysis panel (keep it outside the repository) |
| `main.target_options` | the target of a single-target run |
| `main.batch_mode` | a bundle of targets run one after another, instead of `target_options` |
| `main.split` | `within_categories` (one model per measurement category) or `across_categories` |
| `main.tp_option` | the wave of the target, `0` (baseline) to `4` |
| `main.feature_tp_option` | `Match target` for concurrent prediction, or an earlier wave for prospective prediction |
| `setup.objective_subjective_spectrum` | use the objective/subjective gradient instead of the measurement categories |
| `model_overrides` | base-learner hyperparameters for this run |
| `stages` | the stages to run after setup |

Every setting is listed below.

## `setup`

Data location, shared validation settings and ensemble plot styling (notebook cells 4-5).

| Setting | Default | Description |
|---|---|---|
| `objective_subjective_spectrum` | `False` | Replace the T0/T2 feature ontology with the objective/subjective measurement spectrum (notebook cell 3; used for Fig 4, Fex6 and S1.12). Off for all other analyses. |
| `DATA_PATH` | `'CLEAN_ABCD_5.1_panel_20241022.csv'` | Merged multi-wave analysis panel (CSV). Keep ABCD data outside the repository. |
| `VAL_N_OUTER` | `5` | Default number of outer folds used by validation helpers. |
| `VAL_N_BOOTSTRAP` | `1000` | Default number of bootstrap resamples for confidence intervals. |
| `ensemble_barplot_palette` | `'rocket'` | Colormap for feature-importance bars. "None" (or YAML null) uses a gradient of ensemble_barplot_color instead. Allowed: `Purples_r`, `Blues_r`, `Reds_r`, `YlOrBr_r`, `Greens_r`, `None`, `YlOrRd`, `YlGnBu`, ... (32 values). |
| `ensemble_barplot_color` | `'sienna'` | Single color → gradient (only active when palette = None; auto-set per-target in batch mode). Allowed: `crimson`, `firebrick`, `tomato`, `darkorange`, `orange`, `gold`, `yellow`, `greenyellow`, ... (30 values). |
| `ensemble_heatmap_cmap` | `'Spectral'` | Colormap for model-comparison heatmaps. Allowed: `YlOrRd`, `coolwarm`, `RdYlBu`, `viridis`, `plasma`, `inferno`, `magma`, `Blues`, ... (18 values). |
| `ensemble_top_n` | `15` | Number of top features shown in the consensus importance, feature-importance and SHAP plots. |
| `ensemble_ylabel_fontsize` | `10` | Font size of the feature labels in the consensus importance plot. |
| `ensemble_save_plots` | `True` | Save the consensus importance plot to a file. |
| `ensemble_save_prefix` | `'ensemble'` | File name prefix for the saved consensus importance plot. |

## `main`

Main analysis runner: target, timepoint, feature split, batch mode (notebook cell 6).

| Setting | Default | Description |
|---|---|---|
| `target_options` | `'parent_dep_increase_1.5sd'` | Outcome variable to predict (ignored when batch_mode ≠ Single Target). Allowed: `GAD_present`, `MDD_KSADS_C`, `OCD_present`, `adhd_D_p`, `anhedonia_B_k`, `anorexia_B_k`, `anxdisord_D_p`, `area_deprivation_idx`, ... (147 values). |
| `split` | `'across_categories'` | across_categories = full predictor space; within_categories = loop over feature-set categories separately. Allowed: `across_categories`, `within_categories`. |
| `tp_option` | `'2'` | ABCD assessment wave: 0 (baseline, ages 9-10), 1 (ages 10-11), 2 (ages 11-12), 3 (ages 12-13), 4 (ages 13-14). Allowed: `0`, `1`, `2`, `3`, `4`, `All`. |
| `feature_tp_option` | `'Match target'` | Timepoint to draw features from. Match target = same wave as tp_option (default; preserves standard behavior). Set to a different wave for cross-timepoint prediction (e.g. T0 features → T2 target). Allowed: `Match target`, `0`, `1`, `2`, `3`, `4`. |
| `subgroup` | `'Full Sample'` | Subsample filter. Full Sample = no filter. Allowed: `Full Sample`, `High ALE`, `High ALE + Severe Parent MH`, `Low ALE`, `Female`, `Male`. |
| `ensemble_version` | `True` | Use 4-model stacking ensemble (True) or single base-learner (False, legacy). |
| `ensemble_type` | `'stacking'` | stacking (reported in manuscript) or bagging (legacy, not recommended). Allowed: `bagging`, `stacking`. |
| `skip_tabpfn` | `False` | Drop TabPFN from ensemble (~20-30% faster; auto-skipped for across_categories). |
| `batch_mode` | `'Single Target'` | Run a predefined bundle of targets, overriding target_options. Single Target uses target_options. Allowed: `Single Target`, `Academic/Cognitive`, `ADHD/Impulsivity/Compulsivity`, `All Classification`, `All Manuscript Figures (F3-F6)`, `Child Classification -- All`, `Child Classification -- Anxiety`, `Child Classification -- Depression`, ... (45 values). |
| `run_all_grid` | `False` | Full grid of every wave, subgroup, target and split (~840 runs); overrides the target, wave, subgroup and split settings. |
| `single_category_mode` | `False` | Run only isolate_category (skip the full category loop). |
| `isolate_combinations` | `False` | Run all pairwise feature-set combinations anchored on the selected category. |
| `isolate_category` | `'Cognitive Task Parameters'` | Category to target (used by single_category_mode and isolate_combinations). Allowed: `None`, `Inter`, `Cognitive Task Parameters`, `Dynamic Cognitive Control Parameters`, `Delta Other Psychopathology`, `Other Personality Features`, `Medical/Somatic Problems`, `Sleep Problems`, ... (62 values). |
| `visualization_mode` | `'Ensemble Consensus + Best SHAP'` | Plots generated during the analysis. All Visuals: importance bar plots and SHAP for all models, consensus, PDP and GAM. Ensemble Consensus + Best SHAP: ensemble comparison, consensus and SHAP for the best model. Ensemble Plots Only: model comparison charts (fastest with visuals). No Visuals: metrics only (fastest; for validation and batch runs). Allowed: `All Visuals`, `Ensemble Consensus + Best SHAP`, `Ensemble Plots Only`, `No Visuals`. |
| `run_validation_cells` | `True` | Turn off to make the smoke_test, nested_cv, stratified, grouped_cv, permutation and stacking_audit stages skip their work. |
| `deal_with_na_values` | `'impute_median'` | Strategy for missing values in train/test features. Allowed: `fill_with_zero`, `drop`, `impute_mean`, `impute_median`. |
| `hyperparameter_tuning` | `False` | Search hyperparameters instead of using the fixed settings (slow; not used for the article). For the stacking ensemble the search needs with_optuna. |
| `with_optuna` | `False` | Search the CatBoost settings with Optuna (requires hyperparameter_tuning). |
| `num_of_optuna_trials` | `5` | Number of Optuna trials when with_optuna is on. |
| `selection_method` | `'None'` | Feature selection before model fitting. None = skip. Allowed: `rfecv`, `interaction`, `h-stats`, `clustering`, `None`. |
| `conformal_prediction` | `False` | Add conformal prediction intervals to outputs. |
| `quantile_segment` | `False` | Segment predictions by target-variable quantile bins. |
| `simple_test_model` | `False` | Replace ensemble with a single linear model (lasso or ridge). |
| `model_type` | `'lasso'` | Linear model used when simple_test_model is on. Allowed: `lasso`, `ridge`. |

## `sankey`

Sankey batch visualization (notebook cell 8).

| Setting | Default | Description |
|---|---|---|
| `show_sankey` | `True` | Draw the Sankey diagrams. Off: the stage only lists the archived targets. |
| `sankey_top_features` | `2` | Number of top consensus features drawn per target (15 in Fig 7). |
| `sankey_style` | `'child_adhd_impcomp'` | Style preset: target grouping, colors, layout and which archived results are drawn (presets listed in `bpsp.reports.sankey`). `run_all_styles` draws and saves every preset. |
| `sankey_landscape` | `False` | Landscape page layout for the diagrams. |

## `networks`

Top-feature correlation and partial-correlation networks (notebook cell 9).

| Setting | Default | Description |
|---|---|---|
| `run_bootstrap_CI` | `False` | Bootstrap confidence intervals for the network edge weights (slow). |
| `run_centrality_stability` | `False` | Case-dropping stability of node centrality (slow). |
| `n_bootstrap` | `1000` | Resamples for the edge-weight bootstrap and the centrality stability analysis. |

## `diagnostics`

Model diagnostics (notebook cell 10).

| Setting | Default | Description |
|---|---|---|
| `run_diversity_analysis` | `True` | Base-learner diversity: pairwise correlations of the base learners' predictions. |
| `run_bootstrap_stability` | `True` | Bootstrap stability of the feature importance rankings. |
| `run_learning_curves` | `True` | Learning curves: held-out performance against training set size. |
| `run_shap_beeswarm` | `True` | SHAP beeswarm plots, per model and weighted by the ensemble. |
| `run_supplementary` | `True` | Supplementary diagnostics: effect sizes, feature overlap across targets and prediction distributions. |
| `show_scatter_matrix` | `True` | Scatter plots of predictions for the most diverse base-learner pairs. |
| `top_n_scatter_pairs` | `3` | Number of base-learner pairs in those scatter plots. |
| `n_bootstraps` | `100` | Bootstrap resamples for the importance stability analysis. |
| `top_k_features` | `20` | Number of top features tracked in the importance stability analysis. |
| `show_violin_plot` | `True` | Violin plot of the rank distributions of those features. |
| `n_fractions` | `10` | Number of training set fractions, from 10% to 100%, in the learning curves. |
| `n_repeats` | `5` | Repeats per training set fraction in the learning curves. |
| `show_marginal_gains` | `True` | Plot the performance gained with each increase in training data. |
| `top_n_shap` | `25` | Number of features in the ensemble-weighted SHAP beeswarm. |
| `show_weight_chart` | `True` | Bar chart of mean absolute SHAP values beside the beeswarm. |
| `use_display_names` | `True` | Natural-language labels instead of variable names in the diagnostics. |
| `run_effect_sizes` | `True` | Effect size summary for the current target (part of the supplementary diagnostics). |
| `run_feature_overlap` | `True` | Overlap of the top consensus features across targets, as Jaccard similarity. |
| `run_prediction_diagnostics` | `True` | Distribution diagnostics of the ensemble's predictions. |
| `top_k_overlap` | `20` | Number of top consensus features compared across targets. |
| `diagnostics_target` | `'last'` | `last` for the most recent target, or an archive key (e.g. `internal_D_p__tp2`) to diagnose another target of the batch. |

## `nested_cv`

Outer-fold nested CV metrics and calibration, Supplement S1.4, S1.11.3 and S1.11.10 (notebook cell 12).

| Setting | Default | Description |
|---|---|---|
| `n_outer_folds` | `10` | Number of outer folds (5 or 10 in the article). |

## `stratified`

Demographic-stratified performance (notebook cell 13).

| Setting | Default | Description |
|---|---|---|
| `n_bootstrap` | `1000` | Bootstrap resamples for the subgroup confidence intervals. |

## `grouped_cv`

Family-grouped and site-grouped cross-validation (notebook cell 14).

| Setting | Default | Description |
|---|---|---|
| `n_outer_folds` | `5` | Number of GroupKFold folds. |
| `robustness_mode` | `'Family Only'` | Family-grouped CV, site-grouped CV or both. Allowed: `Family Only`, `Site Only`, `Both`. |

## `permutation`

Permutation testing and null distributions (notebook cell 15).

| Setting | Default | Description |
|---|---|---|
| `n_permutations` | `1000` | Number of label permutations. |
| `n_outer_folds` | `5` | Cross-validation folds used to score the CatBoost model on the true and the permuted labels. |

## `stacking_audit`

Stacking audit: base-model contributions and leakage checks (notebook cell 16).

| Setting | Default | Description |
|---|---|---|
| `n_outer_folds` | `5` | Number of outer folds in the audit. |

## `baselines`

PCA baseline and temporal hold-out (notebook cell 17).

| Setting | Default | Description |
|---|---|---|
| `run_pca_baseline` | `True` | Compare the original features with a PCA baseline. |
| `run_temporal_holdout` | `True` | Train on earlier waves and test on later waves. |

## `ablation`

Domain ablation and feature-domain analysis (notebook cell 18).

| Setting | Default | Description |
|---|---|---|
| `run_binary_ablation` | `True` | Objective versus subjective ablation, with bootstrap confidence intervals. |
| `run_spectrum_ablation` | `True` | Ablation over the levels of the objective/subjective gradient, one domain at a time. |
| `run_domain_permutation` | `True` | Domain permutation importance: each domain shuffled as a block on the full model. |
| `run_topk_ablation` | `True` | Top-K ablation curve: the highest-ranked features removed progressively. |

## `temporal_gradient`

Temporal gradient testing (notebook cell 20).

| Setting | Default | Description |
|---|---|---|
| `run_temporal_gradient` | `True` | Run the analysis. Off: the stage is skipped. |

## `fairness`

Race/ethnicity and religion fairness, partial-correlation confound check (notebook cell 21).

| Setting | Default | Description |
|---|---|---|
| `frr_targets` | `'all'` | Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`. |
| `frr_n_bootstrap` | `1000` | Bootstrap resamples for the group estimates, their differences and the partial correlations. |
| `frr_min_group_n` | `100` | Religious affiliations with fewer test participants are pooled; any group still this small is listed but not scored. |
| `frr_min_group_pos` | `10` | Classification: groups with fewer positive or negative cases are listed but not scored. |
| `frr_religion_grouping` | `'affiliation'` | `affiliation` = named preferences (small ones pooled); `broad tradition` = Christian / other religion / atheist or agnostic / nothing in particular. Allowed: `affiliation`, `broad tradition`. |
| `frr_include_service_attendance` | `True` | Also stratify by attendance at religious services. |
| `frr_run_partial_corr` | `True` | Run the partial-correlation confound check (part 2). |
| `frr_include_site_in_partial` | `True` | Include site in the extended confound set of that check. |
| `frr_make_plots` | `True` | Draw the forest plots and the across-target difference map. |

## `calibration`

Calibration check and Platt/isotonic recalibration (notebook cell 22).

| Setting | Default | Description |
|---|---|---|
| `rcal_targets` | `'all'` | Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`. |
| `rcal_n_bins` | `10` | Number of bins for ECE and the reliability diagrams. |
| `rcal_bin_strategy` | `'quantile'` | `quantile` = equal-count bins (recommended for 2-10% base rates); `uniform` = equal-width bins. Allowed: `quantile`, `uniform`. |
| `rcal_min_bin_n` | `10` | Bins with fewer cases are not drawn (they still count in ECE). |
| `rcal_n_bootstrap` | `1000` | Bootstrap resamples for the Brier score confidence interval. |
| `rcal_crossfit_folds` | `5` | Folds for cross-fitting the calibrators within the held-out set. |
| `rcal_heldout_calibrator` | `'crossfit'` | `crossfit` = calibrators cross-fitted within the held-out set; `nested_oof` = fitted on the nested_cv stage's out-of-fold probabilities. Allowed: `crossfit`, `nested_oof`. |
| `rcal_include_nested_oof` | `True` | Also evaluate the nested-CV out-of-fold probabilities, when the nested_cv stage has run in this session. |
| `rcal_include_regression` | `True` | Include regression targets (calibration slope and intercept). |
| `rcal_make_plots` | `True` | Draw the calibration figures. |

## `base_learners`

Base learners vs stacked ensemble on held-out data (notebook cell 23).

| Setting | Default | Description |
|---|---|---|
| `blp_targets` | `'all'` | Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`. |
| `blp_gain_tolerance` | `0.005` | The ensemble counts as "matching" the best base learner if within this margin. |
| `blp_make_plots` | `True` | Draw the comparison figure. |

## `grouped_importance`

Correlation-aware grouped permutation importance (notebook cell 24).

| Setting | Default | Description |
|---|---|---|
| `gpi_targets` | `'all'` | Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`. |
| `gpi_top_k_features` | `30` | Number of top features (by weighted consensus importance) clustered and tested. |
| `gpi_cluster_r_threshold` | `0.5` | Features join a cluster when their average /Spearman r/ is at least this. |
| `gpi_n_repeats` | `5` | Permutation repeats per feature or group. |
| `gpi_include_domain_groups` | `True` | Also shuffle the measurement domains defined by the ablation stage (it must have run in this session). |
| `gpi_make_plots` | `True` | Draw the figures. |
