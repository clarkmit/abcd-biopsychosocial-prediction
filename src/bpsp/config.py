"""Typed configuration for every pipeline stage.

Each dataclass mirrors one notebook cell's Colab form fields; defaults are the values used
for the published analyses. Field names are kept identical to the notebook so that every
setting can be traced to the paper version (v1.0.0).

Setting a name that does not exist raises ``AttributeError``, so typos fail loudly.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ALL_STAGES: tuple[str, ...] = (
    "main",
    "batch_summary",
    "sankey",
    "networks",
    "diagnostics",
    "smoke_test",
    "nested_cv",
    "stratified",
    "grouped_cv",
    "permutation",
    "stacking_audit",
    "baselines",
    "ablation",
    "trajectories",
    "temporal_gradient",
    "fairness",
    "calibration",
    "base_learners",
    "grouped_importance",
)
"""Analysis stages in notebook order (setup always runs first)."""

CHOICES: dict[tuple[str, str], tuple[Any, ...]] = {}
"""Allowed values for fields that were drop-downs in the notebook."""


@dataclass(slots=True)
class SetupConfig:
    """Data location, shared validation settings and ensemble plot styling (notebook cells 4-5)."""

    objective_subjective_spectrum: bool = False
    """Replace the T0/T2 feature ontology with the objective/subjective measurement spectrum
    (notebook cell 3; used for Fig 4, Fex6 and S1.12). Off for all other analyses."""
    DATA_PATH: str = "CLEAN_ABCD_5.1_panel_20241022.csv"
    """Merged multi-wave analysis panel (CSV). Keep ABCD data outside the repository."""
    VAL_N_OUTER: int = 5
    """Default number of outer folds used by validation helpers."""
    VAL_N_BOOTSTRAP: int = 1000
    """Default number of bootstrap resamples for confidence intervals."""
    ensemble_barplot_palette: str = "rocket"
    """Colormap for feature-importance bars. "None" (or YAML null) uses a gradient of ensemble_barplot_color instead. One of ``CHOICES[('setup', 'ensemble_barplot_palette')]``."""
    ensemble_barplot_color: str = "sienna"
    """Single color → gradient (only active when palette = None; auto-set per-target in batch mode). One of ``CHOICES[('setup', 'ensemble_barplot_color')]``."""
    ensemble_heatmap_cmap: str = "Spectral"
    """Colormap for model-comparison heatmaps. One of ``CHOICES[('setup', 'ensemble_heatmap_cmap')]``."""
    ensemble_top_n: int = 15
    """Number of top features shown in the consensus importance, feature-importance and SHAP plots."""
    ensemble_ylabel_fontsize: int = 10
    """Font size of the feature labels in the consensus importance plot."""
    ensemble_save_plots: bool = True
    """Save the consensus importance plot to a file."""
    ensemble_save_prefix: str = "ensemble"
    """File name prefix for the saved consensus importance plot."""


CHOICES[("setup", "ensemble_barplot_palette")] = (
    "Purples_r",
    "Blues_r",
    "Reds_r",
    "YlOrBr_r",
    "Greens_r",
    "None",
    "YlOrRd",
    "YlGnBu",
    "RdYlBu",
    "RdYlGn",
    "coolwarm",
    "Spectral",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Set1",
    "Set2",
    "Set3",
    "Paired",
    "Dark2",
    "tab10",
    "tab20",
    "mako",
    "rocket",
    "Wistia",
    "cubehelix",
    "flare",
    "crest",
    "turbo",
    "rainbow",
)

CHOICES[("setup", "ensemble_barplot_color")] = (
    "crimson",
    "firebrick",
    "tomato",
    "darkorange",
    "orange",
    "gold",
    "yellow",
    "greenyellow",
    "limegreen",
    "forestgreen",
    "darkgreen",
    "teal",
    "darkcyan",
    "deepskyblue",
    "royalblue",
    "navy",
    "darkblue",
    "blueviolet",
    "purple",
    "mediumpurple",
    "orchid",
    "deeppink",
    "hotpink",
    "saddlebrown",
    "sienna",
    "peru",
    "black",
    "dimgray",
    "gray",
    "slategray",
)

CHOICES[("setup", "ensemble_heatmap_cmap")] = (
    "YlOrRd",
    "coolwarm",
    "RdYlBu",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "Blues",
    "Reds",
    "Greens",
    "Spectral",
    "RdBu",
    "PiYG",
    "BrBG",
    "mako",
    "rocket",
    "flare",
    "crest",
)


@dataclass(slots=True)
class MainAnalysisConfig:
    """Main analysis runner: target, timepoint, feature split, batch mode (notebook cell 6)."""

    target_options: str = "parent_dep_increase_1.5sd"
    """Outcome variable to predict (ignored when batch_mode ≠ Single Target). One of ``CHOICES[('main', 'target_options')]``."""
    split: str = "across_categories"
    """across_categories = full predictor space; within_categories = loop over feature-set categories separately. One of ``CHOICES[('main', 'split')]``."""
    tp_option: str = "2"
    """ABCD assessment wave: 0 (baseline, ages 9-10), 1 (ages 10-11), 2 (ages 11-12), 3 (ages 12-13), 4 (ages 13-14). One of ``CHOICES[('main', 'tp_option')]``."""
    feature_tp_option: str = "Match target"
    """Timepoint to draw features from. Match target = same wave as tp_option (default; preserves standard behavior). Set to a different wave for cross-timepoint prediction (e.g. T0 features → T2 target). One of ``CHOICES[('main', 'feature_tp_option')]``."""
    subgroup: str = "Full Sample"
    """Subsample filter. Full Sample = no filter. One of ``CHOICES[('main', 'subgroup')]``."""
    ensemble_version: bool = True
    """Use 4-model stacking ensemble (True) or single base-learner (False, legacy)."""
    ensemble_type: str = "stacking"
    """stacking (reported in manuscript) or bagging (legacy, not recommended). One of ``CHOICES[('main', 'ensemble_type')]``."""
    skip_tabpfn: bool = False
    """Drop TabPFN from ensemble (~20-30% faster; auto-skipped for across_categories)."""
    batch_mode: str = "Single Target"
    """Run a predefined bundle of targets, overriding target_options. Single Target uses target_options. One of ``CHOICES[('main', 'batch_mode')]``."""
    run_all_grid: bool = False
    """Full grid of every wave, subgroup, target and split (~840 runs); overrides the target, wave, subgroup and split settings."""
    single_category_mode: bool = False
    """Run only isolate_category (skip the full category loop)."""
    isolate_combinations: bool = False
    """Run all pairwise feature-set combinations anchored on the selected category."""
    isolate_category: str = "Cognitive Task Parameters"
    """Category to target (used by single_category_mode and isolate_combinations). One of ``CHOICES[('main', 'isolate_category')]``."""
    visualization_mode: str = "Ensemble Consensus + Best SHAP"
    """Plots generated during the analysis. All Visuals: importance bar plots and SHAP for all models, consensus, PDP and GAM. Ensemble Consensus + Best SHAP: ensemble comparison, consensus and SHAP for the best model. Ensemble Plots Only: model comparison charts (fastest with visuals). No Visuals: metrics only (fastest; for validation and batch runs). One of ``CHOICES[('main', 'visualization_mode')]``."""
    run_validation_cells: bool = True
    """Turn off to make the smoke_test, nested_cv, stratified, grouped_cv, permutation and stacking_audit stages skip their work."""
    deal_with_na_values: str = "impute_median"
    """Strategy for missing values in train/test features. One of ``CHOICES[('main', 'deal_with_na_values')]``."""
    hyperparameter_tuning: bool = False
    """Search hyperparameters instead of using the fixed settings (slow; not used for the article). For the stacking ensemble the search needs with_optuna."""
    with_optuna: bool = False
    """Search the CatBoost settings with Optuna (requires hyperparameter_tuning)."""
    num_of_optuna_trials: int = 5
    """Number of Optuna trials when with_optuna is on."""
    selection_method: str = "None"
    """Feature selection before model fitting. None = skip. One of ``CHOICES[('main', 'selection_method')]``."""
    conformal_prediction: bool = False
    """Add conformal prediction intervals to outputs."""
    quantile_segment: bool = False
    """Segment predictions by target-variable quantile bins."""
    simple_test_model: bool = False
    """Replace ensemble with a single linear model (lasso or ridge)."""
    model_type: str = "lasso"
    """Linear model used when simple_test_model is on. One of ``CHOICES[('main', 'model_type')]``."""


CHOICES[("main", "target_options")] = (
    "GAD_present",
    "MDD_KSADS_C",
    "OCD_present",
    "adhd_D_p",
    "anhedonia_B_k",
    "anorexia_B_k",
    "anxdisord_D_p",
    "area_deprivation_idx",
    "asd_ssrs_sum",
    "asdadhd_c",
    "bad_grades",
    "bdefs_distract_upset_p",
    "bulimia_B_k",
    "compulsive_p",
    "delta_anxdisord_D_p",
    "delta_depress_D_p",
    "delta_parent_depress_D_p",
    "delta_parent_depressed_D_p",
    "delta_parent_depressed_p",
    "delta_parent_worries_a_lot_p",
    "delta_sbt_core_depression",
    "dep_decrease_1.5sd",
    "dep_decrease_1sd",
    "dep_decrease_2sd",
    "dep_increase_1.5sd",
    "dep_increase_1sd",
    "dep_increase_2sd",
    "dep_onset_rci_1.7",
    "dep_onset_rci_1.96",
    "dep_onset_rci_2.3",
    "dep_remission_rci_1.7",
    "dep_remission_rci_1.96",
    "dep_remission_rci_2.3",
    "depadhd_c",
    "depress_D_p",
    "depress_D_p_rev",
    "depress_update_p",
    "depressed_mood_B_k",
    "doesnt_get_along_p",
    "elevated_mood_B_k",
    "enjoys_little_p",
    "excessive_worry_B_k",
    "external_D_p",
    "fatigue_present_B_k",
    "gd_riskybets",
    "hopeless_B_k",
    "hopeless_B_p",
    "impulsive_p",
    "internal_D_p",
    "ksads_1_840_t",
    "latent_class_depression",
    "lying_B_k",
    "nb_correct_nt_2back",
    "not_liked_p",
    "obsessions_p",
    "ocd_D_p",
    "pa_sum_k",
    "parent_adhd_D_p",
    "parent_anhedonia_B_p",
    "parent_anxdisord_D_p",
    "parent_avoidant_person_D_p",
    "parent_bad_family_relationship_p",
    "parent_bad_relationships_p",
    "parent_concentration_issues_B_p",
    "parent_concentration_trouble_p",
    "parent_dep_decrease_1.5sd",
    "parent_dep_decrease_2sd",
    "parent_dep_increase_1.5sd",
    "parent_dep_increase_2sd",
    "parent_dep_onset_rci_1.96",
    "parent_dep_onset_rci_2.3",
    "parent_dep_remission_rci_1.96",
    "parent_dep_remission_rci_2.3",
    "parent_depress_D_p",
    "parent_depressed_mood_B_p",
    "parent_depressed_p",
    "parent_distress_due_to_social_anxiety_B_p",
    "parent_drinks_frequency_p",
    "parent_drunk_days_p",
    "parent_dysthymia_B_p",
    "parent_easily_annoyed_B_p",
    "parent_excessive_worry_B_p",
    "parent_external_D_p",
    "parent_friendship_trouble_p",
    "parent_happy_person_p",
    "parent_impulsive_p",
    "parent_income",
    "parent_internal_D_p",
    "parent_low_self_esteem_B_p",
    "parent_not_liked_by_others_p",
    "parent_obsessions_present_B_p",
    "parent_obsessive_thoughts_p",
    "parent_repeats_acts_p",
    "parent_sleep_problem_B_p",
    "parent_social_anxiety_disorder_B_p",
    "parent_socially_anxious_B_p",
    "parent_suicidal_thoughts_p",
    "parent_tobacco_use_frequency_p",
    "parent_two_more_depression_B_p",
    "parent_wish_dead_present_B_p",
    "parent_worries_a_lot_p",
    "perfectionist_p",
    "ptsd_diagnosis_present_B_p",
    "sbt_academic_cognitive",
    "sbt_aggression_irritability",
    "sbt_anxiety_depression",
    "sbt_avoidance_fear",
    "sbt_core_dep_decrease_1.5sd",
    "sbt_core_dep_decrease_2sd",
    "sbt_core_dep_increase_1.5sd",
    "sbt_core_dep_increase_2sd",
    "sbt_core_dep_onset_rci_1.7",
    "sbt_core_dep_onset_rci_1.96",
    "sbt_core_dep_onset_rci_2.3",
    "sbt_core_dep_remission_rci_1.7",
    "sbt_core_dep_remission_rci_1.96",
    "sbt_core_dep_remission_rci_2.3",
    "sbt_core_depression",
    "sbt_core_depression_delta_t0_t3",
    "sbt_core_depression_raw",
    "sbt_emotional_dysregulation",
    "sbt_fatigue_sleep",
    "sbt_guilt_hopelessness",
    "sbt_perfectionism_achievement",
    "sbt_social_withdrawal",
    "sbt_somatic_depression",
    "sbt_well_being",
    "sleep_problem_B_k",
    "social_anxiety_disorder_B_k",
    "social_problems_D_p",
    "socially_anxious_B_k",
    "suicidal_ideation_present_B_k",
    "suicidal_p",
    "suicide_attempt_past_B_k",
    "suicide_attempt_present_B_k",
    "tb_cryst",
    "tb_flanker",
    "tb_fluid",
    "tb_reading",
    "thought_disorder_D_p",
    "top_10_depression",
    "top_10_sbt_core_depression",
    "top_5_depression",
    "top_5_sbt_core_depression",
    "two_more_depression_B_k",
    "weight",
    "wish_dead_present_B_k",
)

CHOICES[("main", "split")] = (
    "across_categories",
    "within_categories",
)

CHOICES[("main", "tp_option")] = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "All",
)

CHOICES[("main", "feature_tp_option")] = (
    "Match target",
    "0",
    "1",
    "2",
    "3",
    "4",
)

CHOICES[("main", "subgroup")] = (
    "Full Sample",
    "High ALE",
    "High ALE + Severe Parent MH",
    "Low ALE",
    "Female",
    "Male",
)

CHOICES[("main", "ensemble_type")] = (
    "bagging",
    "stacking",
)

CHOICES[("main", "batch_mode")] = (
    "Single Target",
    "Academic/Cognitive",
    "ADHD/Impulsivity/Compulsivity",
    "All Classification",
    "All Manuscript Figures (F3-F6)",
    "Child Classification -- All",
    "Child Classification -- Anxiety",
    "Child Classification -- Depression",
    "Child Classification -- Other",
    "Child Classification -- Suicidality",
    "Child Dep Classification",
    "Child Mood & Anxiety (with Deltas)",
    "Child Only Deltas",
    "Child P-Factor",
    "Child Suicide Ideation",
    "Parent P-Factor",
    "Parent Suicide Ideation",
    "Exhaustive Target Union",
    "F3: Child Psychopathology & Cognition",
    "F4: Depression & Anxiety Developmental Sweep",
    "F4c: Subjective vs. Objective Outcomes",
    "F5: Parent Depression Trajectory + T2 Contrasts",
    "F6: Child and Parent M/D & Deltas",
    "Mood/Depression Subtypes Children",
    "OCD/Impulsivity & Parent Substance Use",
    "Parent Classification -- All",
    "Parent Classification -- Anxiety",
    "Parent Classification -- Depression",
    "Parent Classification -- Other",
    "Parent Dep Classification",
    "Parent Only Deltas",
    "Parent Psychopathology (Mood/ADHD/Social)",
    "Parent Social/Impulsivity/Compulsivity",
    "Primary Child & Parent ADHD/Anxiety/Mood Targets",
    "SBT Core Triad",
    "Suicidal Ideation (Child & Parent Dyad)",
    "Child Depression Worsening",
    "Parent Depression Worsening",
    "Parent Targets Fx",
    "Primary Targets Fin",
    "Formal DepAnx NeuroCog",
    "Social Problems",
    "Child Social Problems Developmental Sweep",
    "Parent Social Problems Developmental Sweep",
    "T0 Cognitive Task Battery",
)

CHOICES[("main", "isolate_category")] = (
    "None",
    "Inter",
    "Cognitive Task Parameters",
    "Dynamic Cognitive Control Parameters",
    "Delta Other Psychopathology",
    "Other Personality Features",
    "Medical/Somatic Problems",
    "Sleep Problems",
    "Diet/Nutrition",
    "Physical Activity/Features",
    "Technology Use",
    "Social Relationship Quality",
    "Family Dynamics & Parenting",
    "Parent Anxiety",
    "Parent Anxiety and Fear-Related Issues",
    "Parent Mood Issues",
    "Parent Impulsivity and Behavior Regulation",
    "Parent Social Functioning",
    "Parent Interpersonal Relationships and Social Functioning",
    "Parent Cognitive and Attention Issues",
    "Parent Personality",
    "Parent Delta Psychopathology",
    "Parent Other Psychopathology",
    "Parent Psychopathology Composite",
    "Parent Drug Use",
    "SES & Mobility",
    "Residential Characteristics",
    "School Dynamics",
    "Anxiety",
    "ADHD",
    "Externalizing",
    "Family Psychopathology Aggregated",
    "Family Drug Use",
    "Ancestral-Genetic PCs/Ethnicity",
    "Task and Resting State",
    "Structural and DTI",
    "Adverse Life Events",
    "Familial Adverse Life Events",
    "Both Adverse Life Events",
    "Parent/Child Adverse Life Events",
    "Religion",
    "Religion Proxy",
    "Cognitive Style",
    "Emotional Regulation Strategies",
    "Ethnicity/Nationality",
    "Child ADHD",
    "Child Anxiety",
    "Child Externalizing",
    "Child Cognitive Task Outcomes",
    "Child Cognitive Style",
    "Child Delta",
    "Child Emotional Regulation Strategies",
    "Child Mood Issues",
    "Child Other Personality Features",
    "Child Personality Features",
    "Child Medical/Somatic Problems",
    "Child Sleep Problems",
    "Child Physical Activity/Features",
    "Child Technology Use",
    "Child Social Relationship Quality",
    "Child Social Personality Psychopathology Sum",
    "Child School Dynamics",
)

CHOICES[("main", "visualization_mode")] = (
    "All Visuals",
    "Ensemble Consensus + Best SHAP",
    "Ensemble Plots Only",
    "No Visuals",
)

CHOICES[("main", "deal_with_na_values")] = (
    "fill_with_zero",
    "drop",
    "impute_mean",
    "impute_median",
)

CHOICES[("main", "selection_method")] = (
    "rfecv",
    "interaction",
    "h-stats",
    "clustering",
    "None",
)

CHOICES[("main", "model_type")] = (
    "lasso",
    "ridge",
)


@dataclass(slots=True)
class SankeyConfig:
    """Sankey batch visualization (notebook cell 8)."""

    show_sankey: bool = True
    """Draw the Sankey diagrams. Off: the stage only lists the archived targets."""
    sankey_top_features: int = 2
    """Number of top consensus features drawn per target (15 in Fig 7)."""
    sankey_style: str = "child_adhd_impcomp"
    """Style preset: target grouping, colors, layout and which archived results are drawn (presets listed in `bpsp.reports.sankey`). `run_all_styles` draws and saves every preset."""
    sankey_landscape: bool = False
    """Landscape page layout for the diagrams."""


@dataclass(slots=True)
class NetworksConfig:
    """Top-feature correlation and partial-correlation networks (notebook cell 9)."""

    run_bootstrap_CI: bool = False
    """Bootstrap confidence intervals for the network edge weights (slow)."""
    run_centrality_stability: bool = False
    """Case-dropping stability of node centrality (slow)."""
    n_bootstrap: int = 1000
    """Resamples for the edge-weight bootstrap and the centrality stability analysis."""


@dataclass(slots=True)
class DiagnosticsConfig:
    """Model diagnostics (notebook cell 10)."""

    run_diversity_analysis: bool = True
    """Base-learner diversity: pairwise correlations of the base learners' predictions."""
    run_bootstrap_stability: bool = True
    """Bootstrap stability of the feature importance rankings."""
    run_learning_curves: bool = True
    """Learning curves: held-out performance against training set size."""
    run_shap_beeswarm: bool = True
    """SHAP beeswarm plots, per model and weighted by the ensemble."""
    run_supplementary: bool = True
    """Supplementary diagnostics: effect sizes, feature overlap across targets and prediction distributions."""
    show_scatter_matrix: bool = True
    """Scatter plots of predictions for the most diverse base-learner pairs."""
    top_n_scatter_pairs: int = 3
    """Number of base-learner pairs in those scatter plots."""
    n_bootstraps: int = 100
    """Bootstrap resamples for the importance stability analysis."""
    top_k_features: int = 20
    """Number of top features tracked in the importance stability analysis."""
    show_violin_plot: bool = True
    """Violin plot of the rank distributions of those features."""
    n_fractions: int = 10
    """Number of training set fractions, from 10% to 100%, in the learning curves."""
    n_repeats: int = 5
    """Repeats per training set fraction in the learning curves."""
    show_marginal_gains: bool = True
    """Plot the performance gained with each increase in training data."""
    top_n_shap: int = 25
    """Number of features in the ensemble-weighted SHAP beeswarm."""
    show_weight_chart: bool = True
    """Bar chart of mean absolute SHAP values beside the beeswarm."""
    use_display_names: bool = True
    """Natural-language labels instead of variable names in the diagnostics."""
    run_effect_sizes: bool = True
    """Effect size summary for the current target (part of the supplementary diagnostics)."""
    run_feature_overlap: bool = True
    """Overlap of the top consensus features across targets, as Jaccard similarity."""
    run_prediction_diagnostics: bool = True
    """Distribution diagnostics of the ensemble's predictions."""
    top_k_overlap: int = 20
    """Number of top consensus features compared across targets."""
    diagnostics_target: str = "last"
    """`last` for the most recent target, or an archive key (e.g. `internal_D_p__tp2`) to diagnose another target of the batch."""


@dataclass(slots=True)
class NestedCVConfig:
    """Outer-fold nested CV metrics and calibration, Supplement S1.4, S1.11.3 and S1.11.10 (notebook cell 12)."""

    n_outer_folds: int = 10
    """Number of outer folds (5 or 10 in the article)."""


@dataclass(slots=True)
class StratifiedConfig:
    """Demographic-stratified performance (notebook cell 13)."""

    n_bootstrap: int = 1000
    """Bootstrap resamples for the subgroup confidence intervals."""


@dataclass(slots=True)
class GroupedCVConfig:
    """Family-grouped and site-grouped cross-validation (notebook cell 14)."""

    n_outer_folds: int = 5
    """Number of GroupKFold folds."""
    robustness_mode: str = "Family Only"
    """Family-grouped CV, site-grouped CV or both. One of ``CHOICES[('grouped_cv', 'robustness_mode')]``."""


CHOICES[("grouped_cv", "robustness_mode")] = (
    "Family Only",
    "Site Only",
    "Both",
)


@dataclass(slots=True)
class PermutationConfig:
    """Permutation testing and null distributions (notebook cell 15)."""

    n_permutations: int = 1000
    """Number of label permutations."""
    n_outer_folds: int = 5
    """Cross-validation folds used to score the CatBoost model on the true and the permuted labels."""


@dataclass(slots=True)
class StackingAuditConfig:
    """Stacking audit: base-model contributions and leakage checks (notebook cell 16)."""

    n_outer_folds: int = 5
    """Number of outer folds in the audit."""


@dataclass(slots=True)
class BaselinesConfig:
    """PCA baseline and temporal hold-out (notebook cell 17)."""

    run_pca_baseline: bool = True
    """Compare the original features with a PCA baseline."""
    run_temporal_holdout: bool = True
    """Train on earlier waves and test on later waves."""


@dataclass(slots=True)
class AblationConfig:
    """Domain ablation and feature-domain analysis (notebook cell 18)."""

    run_binary_ablation: bool = True
    """Objective versus subjective ablation, with bootstrap confidence intervals."""
    run_spectrum_ablation: bool = True
    """Ablation over the levels of the objective/subjective gradient, one domain at a time."""
    run_domain_permutation: bool = True
    """Domain permutation importance: each domain shuffled as a block on the full model."""
    run_topk_ablation: bool = True
    """Top-K ablation curve: the highest-ranked features removed progressively."""


@dataclass(slots=True)
class TemporalGradientConfig:
    """Temporal gradient testing (notebook cell 20)."""

    run_temporal_gradient: bool = True
    """Run the analysis. Off: the stage is skipped."""


@dataclass(slots=True)
class FairnessConfig:
    """Race/ethnicity and religion fairness, partial-correlation confound check (notebook cell 21)."""

    frr_targets: str = "all"
    """Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`."""
    frr_n_bootstrap: int = 1000
    """Bootstrap resamples for the group estimates, their differences and the partial correlations."""
    frr_min_group_n: int = 100
    """Religious affiliations with fewer test participants are pooled; any group still this small is listed but not scored."""
    frr_min_group_pos: int = 10
    """Classification: groups with fewer positive or negative cases are listed but not scored."""
    frr_religion_grouping: str = "affiliation"
    """`affiliation` = named preferences (small ones pooled); `broad tradition` = Christian / other religion / atheist or agnostic / nothing in particular. One of ``CHOICES[('fairness', 'frr_religion_grouping')]``."""
    frr_include_service_attendance: bool = True
    """Also stratify by attendance at religious services."""
    frr_run_partial_corr: bool = True
    """Run the partial-correlation confound check (part 2)."""
    frr_include_site_in_partial: bool = True
    """Include site in the extended confound set of that check."""
    frr_make_plots: bool = True
    """Draw the forest plots and the across-target difference map."""


CHOICES[("fairness", "frr_religion_grouping")] = (
    "affiliation",
    "broad tradition",
)


@dataclass(slots=True)
class CalibrationConfig:
    """Calibration check and Platt/isotonic recalibration (notebook cell 22)."""

    rcal_targets: str = "all"
    """Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`."""
    rcal_n_bins: int = 10
    """Number of bins for ECE and the reliability diagrams."""
    rcal_bin_strategy: str = "quantile"
    """`quantile` = equal-count bins (recommended for 2-10% base rates); `uniform` = equal-width bins. One of ``CHOICES[('calibration', 'rcal_bin_strategy')]``."""
    rcal_min_bin_n: int = 10
    """Bins with fewer cases are not drawn (they still count in ECE)."""
    rcal_n_bootstrap: int = 1000
    """Bootstrap resamples for the Brier score confidence interval."""
    rcal_crossfit_folds: int = 5
    """Folds for cross-fitting the calibrators within the held-out set."""
    rcal_heldout_calibrator: str = "crossfit"
    """`crossfit` = calibrators cross-fitted within the held-out set; `nested_oof` = fitted on the nested_cv stage's out-of-fold probabilities. One of ``CHOICES[('calibration', 'rcal_heldout_calibrator')]``."""
    rcal_include_nested_oof: bool = True
    """Also evaluate the nested-CV out-of-fold probabilities, when the nested_cv stage has run in this session."""
    rcal_include_regression: bool = True
    """Include regression targets (calibration slope and intercept)."""
    rcal_make_plots: bool = True
    """Draw the calibration figures."""


CHOICES[("calibration", "rcal_bin_strategy")] = (
    "quantile",
    "uniform",
)

CHOICES[("calibration", "rcal_heldout_calibrator")] = (
    "crossfit",
    "nested_oof",
)


@dataclass(slots=True)
class BaseLearnersConfig:
    """Base learners vs stacked ensemble on held-out data (notebook cell 23)."""

    blp_targets: str = "all"
    """Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`."""
    blp_gain_tolerance: float = 0.005
    """The ensemble counts as "matching" the best base learner if within this margin."""
    blp_make_plots: bool = True
    """Draw the comparison figure."""


@dataclass(slots=True)
class GroupedImportanceConfig:
    """Correlation-aware grouped permutation importance (notebook cell 24)."""

    gpi_targets: str = "all"
    """Comma-separated archive keys or target names (e.g. `internal_D_p__tp2, top_10_depression`), or `all`."""
    gpi_top_k_features: int = 30
    """Number of top features (by weighted consensus importance) clustered and tested."""
    gpi_cluster_r_threshold: float = 0.5
    """Features join a cluster when their average |Spearman r| is at least this."""
    gpi_n_repeats: int = 5
    """Permutation repeats per feature or group."""
    gpi_include_domain_groups: bool = True
    """Also shuffle the measurement domains defined by the ablation stage (it must have run in this session)."""
    gpi_make_plots: bool = True
    """Draw the figures."""


@dataclass(slots=True)
class PipelineConfig:
    """Complete run configuration (one YAML file)."""

    stages: list[str] = field(default_factory=lambda: ["main"])
    """Stages to run after setup, in order (see ``ALL_STAGES``)."""
    output_dir: str = "outputs"
    """Directory that receives all tables, figures and logs of the run."""
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-model hyperparameter overrides, e.g. ``{"catboost": {"iterations": 300}}``."""
    setup: SetupConfig = field(default_factory=SetupConfig)
    main: MainAnalysisConfig = field(default_factory=MainAnalysisConfig)
    sankey: SankeyConfig = field(default_factory=SankeyConfig)
    networks: NetworksConfig = field(default_factory=NetworksConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    nested_cv: NestedCVConfig = field(default_factory=NestedCVConfig)
    stratified: StratifiedConfig = field(default_factory=StratifiedConfig)
    grouped_cv: GroupedCVConfig = field(default_factory=GroupedCVConfig)
    permutation: PermutationConfig = field(default_factory=PermutationConfig)
    stacking_audit: StackingAuditConfig = field(default_factory=StackingAuditConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    temporal_gradient: TemporalGradientConfig = field(default_factory=TemporalGradientConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    base_learners: BaseLearnersConfig = field(default_factory=BaseLearnersConfig)
    grouped_importance: GroupedImportanceConfig = field(default_factory=GroupedImportanceConfig)

    # ------------------------------------------------------------------ helpers
    def stage_settings(self, stage: str) -> dict[str, Any]:
        """Settings applied to the run context before ``stage`` runs."""
        section = getattr(self, stage, None)
        return asdict(section) if section is not None else {}

    def validate(self) -> None:
        """Check stages, setting types and drop-down values; raise ``ValueError`` on the first problem."""
        if not isinstance(self.stages, list) or not all(isinstance(s, str) for s in self.stages):
            raise ValueError(f"stages must be a list of stage names, got {self.stages!r}")
        unknown = [s for s in self.stages if s not in ALL_STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; choose from {list(ALL_STAGES)}")
        for section_field in fields(self):
            section = getattr(self, section_field.name)
            if not is_dataclass(section):
                continue
            for f in fields(section):
                _check_type(f"{section_field.name}.{f.name}", getattr(section, f.name), f.default)
        for (stage, name), allowed in CHOICES.items():
            value = getattr(getattr(self, stage), name)
            if value not in allowed:
                shown = ", ".join(repr(a) for a in allowed[:12]) + (", ..." if len(allowed) > 12 else "")
                raise ValueError(f"{stage}.{name}={value!r} is not allowed; choose from {shown}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build a config from a (possibly partial) nested dict; missing keys keep defaults.

        YAML numbers given for text settings (``tp_option: 2``) and ``null`` for settings
        whose allowed values include ``"None"`` are converted to the expected strings.
        """
        if not isinstance(data or {}, dict):
            raise ValueError(f"configuration must be a mapping, got {type(data).__name__}")
        data = dict(data or {})
        cfg = cls()
        if isinstance(data.get("stages"), str):
            data["stages"] = [data["stages"]]
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data.pop(f.name)
            current = getattr(cfg, f.name)
            if is_dataclass(current):
                if value is None:
                    continue
                if not isinstance(value, dict):
                    raise ValueError(f"{f.name} must be a mapping of settings, got {value!r}")
                known = {x.name: x for x in fields(current)}
                extra = set(value) - set(known)
                if extra:
                    raise ValueError(f"unknown setting(s) for {f.name}: {sorted(extra)}")
                for name, v in value.items():
                    setattr(current, name, _coerce(v, known[name].default, CHOICES.get((f.name, name))))
            else:
                setattr(cfg, f.name, value)
        if data:
            raise ValueError(f"unknown top-level key(s): {sorted(data)}")
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _coerce(value: Any, default: Any, choices: tuple | None) -> Any:
    """Convert YAML scalars to the type the notebook used for this setting."""
    if isinstance(default, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if value is None and choices and "None" in choices:
        return "None"
    return value


def _check_type(name: str, value: Any, default: Any) -> None:
    if default is MISSING or default is None or value is None:
        return
    expected = type(default)
    if expected is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, expected)
    if not ok:
        raise ValueError(f"{name}={value!r} should be of type {expected.__name__}")
