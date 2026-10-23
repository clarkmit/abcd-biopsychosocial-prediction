"""Main analysis runner (the `main` stage).

Runs the stacking ensemble for the selected target, wave, subgroup and
split. Within-category mode fits each feature-set category independently
to isolate domain-specific predictive power. Across-category mode uses the
full predictor space minus circular exclusions. Cross-timepoint mode
(feature wave different from the target wave) draws features from one wave
to predict outcomes at another. Batch mode runs preconfigured target
bundles and archives each target's results for the reporting and
validation stages. Subgroup filters cover sex and adverse life event (ALE)
strata, and isolate_category runs a single category in within-category
mode.
"""

import logging
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from IPython.display import display
from mapie.classification import SplitConformalClassifier
from mapie.regression import SplitConformalRegressor
from optuna_integration import CatBoostPruningCallback
from pygam import GAM, s
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bpsp.context import ctx
from bpsp.data.preprocessing import (
    DISPLAY_NAMES,
    across_categories_data,
    analyze_category_quantiles,
    calculate_top_quartile_weights,
    inverse_map_classes,
    is_regression,
    map_classes,
    within_categories_data,
    within_categories_data_combinations,
)
from bpsp.models.stacking import (
    TABPFN_AVAILABLE,
    _resolve_ensemble_palette,
    create_stacking_ensemble,
    modify_ensemble_for_boolean,
    preprocess_boolean_target,
)
from bpsp.reporting import (
    _RS,
    _pipeline_cleanup,
    cache_load,
    cache_save,
    make_run_key,
    report_init,
    report_load_state,
    report_log_data_info,
    report_log_table,
)

logger = logging.getLogger(__name__)


# Map setting values to internal variable names
_SUBGROUP_MAP = {
    "Full Sample": None,
    "High ALE": "high_ale",
    "High ALE + Severe Parent MH": "high_ale_severe_p_mh",
    "Low ALE": "low_ale",
    "Female": "Female",
    "Male": "Male",
}
imputer = "knn"  # feature imputation in data.preprocessing: k-nearest neighbors (k = 5), fitted on training rows

_TARGET_BUNDLES = {
    # All Manuscript Figures (F3-F6) (figure order: F3 → F4 → F4b → F5 → F6)
    "All Manuscript Figures (F3-F6)": [
        # F3: Child Psychopathology + Cognitive (T2)
        "adhd_D_p",
        "external_D_p",
        "internal_D_p",
        "social_problems_D_p",
        "bad_grades",
        "tb_flanker",
        "tb_reading",
        # F4: Mood & Anxiety across development
        "sbt_core_depression",
        "anxdisord_D_p",
        # F4b: Mixed Anxiety/Depression subtype
        "sbt_anxiety_depression",
        # F5: Parent targets
        "parent_depress_D_p",
        "parent_anxdisord_D_p",
        "parent_bad_relationships_p",
        "parent_happy_person_p",
        "parent_adhd_D_p",
        # F6: Additional operationalizations (child & parent)
        "top_5_sbt_core_depression",
        "delta_sbt_core_depression",
        "sbt_core_dep_onset_rci_2.3",
        "delta_parent_depressed_p",
        "delta_parent_worries_a_lot_p",
    ],
    "F3: Child Psychopathology & Cognition": [
        "suicidal_p",
        "adhd_D_p",
        "external_D_p",
        "internal_D_p",
        "social_problems_D_p",
        "bad_grades",
        "tb_flanker",
    ],
    "Child Mood & Anxiety (with Deltas)": [
        "sbt_core_depression",
        "delta_sbt_core_depression",
        "anxdisord_D_p",
        "delta_anxdisord_D_p",
    ],
    "Child P-Factor": [
        "internal_D_p",
        "external_D_p",
    ],
    "Parent P-Factor": [
        "parent_internal_D_p",
        "parent_external_D_p",
    ],
    "Child Suicide Ideation": [
        "suicidal_p",
    ],
    "Parent Suicide Ideation": [
        "parent_suicidal_thoughts_p",
    ],
    "Parent (ADHD/Social/DSyndrome)": [
        "parent_adhd_D_p",
        "parent_bad_relationships_p",
        "parent_depressed_p",
    ],
    "F6: Child and Parent M/D & Deltas": [
        "sbt_core_depression",
        "delta_sbt_core_depression",
        "delta_anxdisord_D_p",
        "parent_depress_D_p",
        "parent_happy_person_p",
        "delta_parent_depressed_p",
        "delta_parent_worries_a_lot_p",
        "parent_adhd_D_p",
        "parent_bad_relationships_p",
        "parent_depressed_p",
    ],
    "OCD/Impulsivity & Parent Substance Use": [
        "compulsions_p",
        "obsessions_p",
        "perfectionist_p",
        "impulsive_p",
        "parent_impulsive_p",
        "parent_tobacco_use_frequency_p",
        "parent_drinks_frequency_p",
    ],
    "Suicidal Ideation (Child & Parent Dyad)": [
        "suicidal_p",
        "parent_suicidal_thoughts_p",
    ],
    # F4c: Subjective vs. Objective Outcomes
    # Depression, Anxiety, ADHD, Educational Performance + additional
    # objective targets (crystallized intelligence, body weight).
    "F4c: Subjective vs. Objective Outcomes": [
        "sbt_core_depression",
        "anxdisord_D_p",
        "adhd_D_p",
        "bad_grades",
        "tb_cryst",
        "weight",
    ],
    # Mood/Depression Subtypes Children
    # All 12 SBT symptom-cluster subscales capturing distinct
    # dimensions of child mood/depression phenomenology.
    "Mood/Depression Subtypes Children": [
        "sbt_core_depression",
        "sbt_anxiety_depression",
        "sbt_guilt_hopelessness",
        "sbt_fatigue_sleep",
        "sbt_somatic_depression",
        "sbt_emotional_dysregulation",
        "sbt_social_withdrawal",
        "sbt_avoidance_fear",
        "sbt_aggression_irritability",
        "sbt_academic_cognitive",
        "sbt_perfectionism_achievement",
    ],
    # SBT Core Triad (Core Depression + Academic/Cognitive + Perfectionism)
    "SBT Core Triad": [
        "sbt_core_depression",
        "sbt_academic_cognitive",
        "sbt_perfectionism_achievement",
    ],
    # Primary Child & Parent ADHD/Anxiety/Mood Targets
    # Core triad (ADHD, anxiety, depression) for both child and parent.
    "Primary Child & Parent ADHD/Anxiety/Mood Targets": [
        # Child
        "adhd_D_p",
        "anxdisord_D_p",
        "sbt_core_depression",
        # Parent
        "parent_adhd_D_p",
        "parent_anxdisord_D_p",
        "parent_depressed_p",
    ],
    # Classification Operationalization Comparisons
    "Child Dep Classification": [
        "dep_onset_rci_2.3",
        "dep_onset_rci_1.96",
        "dep_onset_rci_1.7",
        "dep_increase_2sd",
        "dep_increase_1.5sd",
        "dep_increase_1sd",
        "dep_remission_rci_2.3",
        "dep_remission_rci_1.96",
        "dep_remission_rci_1.7",
        "dep_decrease_2sd",
        "dep_decrease_1.5sd",
        "dep_decrease_1sd",
        "top_10_depression",
        "top_5_depression",
        "latent_class_depression",
        "MDD_KSADS_C",
    ],
    "Parent Dep Classification": [
        "parent_dep_onset_rci_2.3",
        "parent_dep_onset_rci_1.96",
        "parent_dep_increase_2sd",
        "parent_dep_increase_1.5sd",
        "parent_dep_remission_rci_2.3",
        "parent_dep_remission_rci_1.96",
        "parent_dep_decrease_2sd",
        "parent_dep_decrease_1.5sd",
        "top_10_depression_parent",
        "top_5_depression_parent",
    ],
    # Child Only Deltas
    "Child Only Deltas": [
        "delta_depress_D_p",
        "delta_sbt_core_depression",
        "delta_anxdisord_D_p",
        "delta_adhd_D_p",
        "delta_social_problems_D_p",
        "delta_somatic_problems_D_p",
        "delta_bad_grades",
        "delta_family_conflict_ss_k",
    ],
    # Academic/Cognitive
    # Concurrent academic performance, change score in academic performance,
    # and NIH Toolbox crystallized intelligence composite. Captures the
    # objective (test-based) and subjective (grade-based) axes of academic/
    # cognitive functioning. All three are regression targets.
    "Academic/Cognitive": [
        "bad_grades",
        "delta_bad_grades",
        "tb_cryst",
    ],
    # ADHD/Impulsivity/Compulsivity
    # Child CBCL DSM-oriented ADHD composite + individual impulsivity item
    # (cbcl_q41) + individual compulsivity item (cbcl_q66). Captures attention/
    # hyperactive dysregulation alongside two narrower dyscontrol dimensions
    # (stop-failure vs repetitive-action). All three are regression targets.
    "ADHD/Impulsivity/Compulsivity": [
        "adhd_D_p",
        "impulsive_p",
        "compulsions_p",
    ],
    # Parent Social/Impulsivity/Compulsivity
    # Parent ASR analog of the child ADHD/Impulsivity/Compulsivity batch:
    # Avoidant Personality DSM composite (closest analog of CBCL Social
    # Problems syndrome at the parent self-report level), plus four ASR
    # item-level mirrors of the corresponding child CBCL items: not-liked-
    # by-others, impulsivity, repeats acts (compulsions), obsessive thoughts.
    # All five are regression targets. Excludes parent_adhd_D_p and parent
    # relationship-trouble items by design (already covered in prior batches).
    "Parent Social/Impulsivity/Compulsivity": [
        "parent_avoidant_person_D_p",
        "parent_not_liked_by_others_p",
        "parent_impulsive_p",
        "parent_repeats_acts_p",
        "parent_obsessive_thoughts_p",
    ],
    # Parent Only Deltas
    "Parent Only Deltas": [
        "delta_parent_depress_D_p",
        "delta_parent_sleep_trouble_p",
        "delta_parent_worries_about_family_p",
        "delta_parent_not_liked_by_others_p",
        "delta_parent_feels_unloved_p",
        "delta_family_conflict_ss_p",
        "delta_parent_worries_about_future_p",
        "delta_parent_concentration_trouble_p",
        "delta_parent_bad_relationships_p",
        "delta_parent_financial_trouble_p",
        "delta_parent_somatic_problems_D_p",
    ],
    # Child Depression Worsening (Classification)
    # Three threshold operationalisations of child depression worsening:
    # 1.5 SD, 2 SD, and RCI ≥ 2.3 (all binary classification targets).
    # Top 10th and 5th percentile cross-sectional severity cutoffs also included.
    "Child Depression Worsening": [
        "dep_increase_1.5sd",
        "dep_increase_2sd",
        "top_10_depression",
        "top_5_depression",
    ],
    # Parent Depression Worsening (Classification)
    # Three threshold operationalisations of parent depression worsening:
    # 1.5 SD, 2 SD, and RCI ≥ 2.3 (all binary classification targets).
    # Top 10th and 5th percentile cross-sectional severity cutoffs also included.
    "Parent Depression Worsening": [
        "parent_dep_increase_1.5sd",
        "parent_dep_increase_2sd",
        "top_10_depression_parent",
        "top_5_depression_parent",
    ],
    # Parent Targets Fx
    "Parent Targets Fx": [
        "parent_suicidal_thoughts_p",
        "parent_bad_relationships_p",
        "parent_adhd_D_p",
        "parent_happy_person_p",
        "parent_external_D_p",
        "parent_internal_D_p",
    ],
    # Child KSADS Classification
    "Child Classification -- All": [
        "anhedonia_B_k",
        "anorexia_B_k",
        "bulimia_B_k",
        "depressed_mood_B_k",
        "elevated_mood_B_k",
        "excessive_worry_B_k",
        "fatigue_present_B_k",
        "GAD_present",
        "hopeless_B_k",
        "ksads_1_840_t",
        "latent_class_depression",
        "lying_B_k",
        "MDD_KSADS_C",
        "OCD_present",
        "sleep_problem_B_k",
        "social_anxiety_disorder_B_k",
        "socially_anxious_B_k",
        "suicidal_ideation_present_B_k",
        "suicide_attempt_past_B_k",
        "suicide_attempt_present_B_k",
        "top_10_depression",
        "top_5_depression",
        "two_more_depression_B_k",
    ],
    "Child Classification -- Anxiety": [
        "excessive_worry_B_k",
        "GAD_present",
        "OCD_present",
        "social_anxiety_disorder_B_k",
        "socially_anxious_B_k",
    ],
    "Child Classification -- Depression": [
        "anhedonia_B_k",
        "depressed_mood_B_k",
        "elevated_mood_B_k",
        "fatigue_present_B_k",
        "hopeless_B_k",
        "ksads_1_840_t",
        "latent_class_depression",
        "MDD_KSADS_C",
        "top_10_depression",
        "top_5_depression",
        "two_more_depression_B_k",
    ],
    "Child Classification -- Other": [
        "anorexia_B_k",
        "bulimia_B_k",
        "lying_B_k",
        "sleep_problem_B_k",
    ],
    "Child Classification -- Suicidality": [
        "suicidal_ideation_present_B_k",
        "suicide_attempt_past_B_k",
        "suicide_attempt_present_B_k",
    ],
    # Parent KSADS Classification
    "Parent Classification -- All": [
        "hopeless_B_p",
        "parent_anhedonia_B_p",
        "parent_concentration_issues_B_p",
        "parent_depressed_mood_B_p",
        "parent_distress_due_to_social_anxiety_B_p",
        "parent_dysthymia_B_p",
        "parent_easily_annoyed_B_p",
        "parent_excessive_worry_B_p",
        "parent_low_self_esteem_B_p",
        "parent_obsessions_present_B_p",
        "parent_sleep_problem_B_p",
        "parent_social_anxiety_disorder_B_p",
        "parent_socially_anxious_B_p",
        "parent_two_more_depression_B_p",
        "ptsd_diagnosis_present_B_p",
    ],
    "Parent Classification -- Anxiety": [
        "parent_distress_due_to_social_anxiety_B_p",
        "parent_excessive_worry_B_p",
        "parent_obsessions_present_B_p",
        "parent_social_anxiety_disorder_B_p",
        "parent_socially_anxious_B_p",
    ],
    "Parent Classification -- Depression": [
        "hopeless_B_p",
        "parent_anhedonia_B_p",
        "parent_depressed_mood_B_p",
        "parent_dysthymia_B_p",
        "parent_low_self_esteem_B_p",
        "parent_two_more_depression_B_p",
    ],
    "Parent Classification -- Other": [
        "parent_concentration_issues_B_p",
        "parent_easily_annoyed_B_p",
        "parent_sleep_problem_B_p",
        "ptsd_diagnosis_present_B_p",
    ],
    # Combined
    "All Classification": [
        "anhedonia_B_k",
        "anorexia_B_k",
        "bulimia_B_k",
        "depressed_mood_B_k",
        "elevated_mood_B_k",
        "excessive_worry_B_k",
        "fatigue_present_B_k",
        "GAD_present",
        "hopeless_B_k",
        "hopeless_B_p",
        "ksads_1_840_t",
        "latent_class_depression",
        "lying_B_k",
        "MDD_KSADS_C",
        "OCD_present",
        "parent_anhedonia_B_p",
        "parent_concentration_issues_B_p",
        "parent_depressed_mood_B_p",
        "parent_distress_due_to_social_anxiety_B_p",
        "parent_dysthymia_B_p",
        "parent_easily_annoyed_B_p",
        "parent_excessive_worry_B_p",
        "parent_low_self_esteem_B_p",
        "parent_obsessions_present_B_p",
        "parent_sleep_problem_B_p",
        "parent_social_anxiety_disorder_B_p",
        "parent_socially_anxious_B_p",
        "parent_two_more_depression_B_p",
        "ptsd_diagnosis_present_B_p",
        "sleep_problem_B_k",
        "social_anxiety_disorder_B_k",
        "socially_anxious_B_k",
        "suicidal_ideation_present_B_k",
        "suicide_attempt_past_B_k",
        "suicide_attempt_present_B_k",
        "top_10_depression",
        "top_5_depression",
        "two_more_depression_B_k",
    ],
    # Primary Targets (Final Validation)
    # Matched child CBCL → parent ASR DSM-oriented scales + social/relational.
    "Primary Targets Fin": [
        # Child (CBCL DSM-oriented)
        "depress_D_p",
        "anxdisord_D_p",
        "adhd_D_p",
        "social_problems_D_p",
        # Parent (ASR analogues)
        "parent_depress_D_p",
        "parent_anxdisord_D_p",
        "parent_adhd_D_p",
        "parent_bad_relationships_p",
    ],
    # Social Problems (Cross-Child/Parent)
    # Matched child CBCL Social Problems items + parent ASR social/relational
    # targets. Designed for joint analysis across informant and instrument.
    # Pairs with the child/parent Social Problems Developmental Sweep bundles
    # in _TP_SWEEP_BUNDLES for longitudinal (T0-T4) Sankey rendering.
    "Social Problems": [
        # Child (CBCL item + DSM domain composite)
        "not_liked_p",
        "social_problems_D_p",
        # Parent (ASR item + relational composite)
        "parent_not_liked_by_others_p",
        "parent_bad_relationships_p",
    ],
    # Formal Depression & Anxiety NeuroCog Classification
    # Mixed child depression severity/diagnosis (CBCL percentile, KSADS)
    # + anxiety classification (KSADS GAD, excessive worry, social anxiety).
    # All targets are binary classification.
    "Formal DepAnx NeuroCog": [
        # Depression severity / diagnosis
        "top_10_depression",
        "top_5_depression",
        "two_more_depression_B_k",
        "MDD_KSADS_C",
        "anhedonia_B_k",
        # Anxiety classification
        "GAD_present",
        "excessive_worry_B_k",
        "social_anxiety_disorder_B_k",
    ],
    # T0 Cognitive Task Battery (Regression)
    # Seven non-redundant cognitive task targets covering distinct constructs
    # at baseline (T0): crystallized intelligence (verbal knowledge), fluid
    # intelligence (composite EF/speed/WM/episodic memory), Flanker (NIH
    # inhibitory control), SST SSRT (motor response inhibition), 2-back number
    # correct (emotional working memory), RAVLT long-delay total (verbal
    # episodic memory), and LMT efficiency (visuospatial mental rotation).
    # All targets are continuous regression and have full CIRCULAR_EXCLUSIONS
    # entries. T0 baseline coverage verified against CogTimePointsExample.csv.
    "T0 Cognitive Task Battery": [
        # NIH Toolbox composites
        "tb_cryst",
        "tb_fluid",
        # NIH Toolbox subscale (executive control)
        "tb_flanker",
        # fMRI behavioral: response inhibition (SST)
        "sst_mean_ssrt",
        # fMRI behavioral: emotional working memory (EN-back)
        "nb_correct_nt_2back",
        # Verbal episodic memory (RAVLT)
        "ravlt_l_total",
        # Visuospatial mental rotation (LMT)
        "lmt_efficiency",
    ],
}

# "full Target Union" is the deduplicated union of all other bundles.
# Built programmatically so adding a target to any bundle above
# automatically includes it here.
_TARGET_BUNDLES["Exhaustive Target Union"] = list(
    dict.fromkeys(t for bundle in _TARGET_BUNDLES.values() for t in bundle)
)

# Timepoint-sweep bundles: (target, timepoint) tuples
# Unlike _TARGET_BUNDLES which run all targets at a single tp_option,
# these bundles override BOTH target and timepoint per iteration.
# Archive keys use f"{target}__tp{tp}" to avoid overwrites.
_TP_SWEEP_BUNDLES = {
    # F4: Depression & Anxiety Developmental Sweep
    "F4: Depression & Anxiety Developmental Sweep": [
        (target, str(tp))
        for target in [
            "sbt_core_depression",
            "anxdisord_D_p",
        ]
        for tp in range(5)  # T0 through T4
    ],
    # F5: Parent Depression Trajectory + T2 Contrasts
    "F5: Parent Depression Trajectory + T2 Contrasts": [
        # Primary: parent depression at T0, T2, T4
        ("parent_depress_D_p", "0"),
        ("parent_depress_D_p", "2"),
        ("parent_depress_D_p", "4"),
        # Secondary: contrast targets at T2 only
        ("parent_adhd_D_p", "2"),
        ("parent_bad_relationships_p", "2"),
        ("parent_anxdisord_D_p", "2"),
        ("parent_happy_person_p", "2"),
    ],
    # Child Social Problems Developmental Sweep
    # CBCL item (not_liked_p) + DSM domain composite (social_problems_D_p)
    # across T0-T4. Use with sankey_style = "developmental_sweep".
    "Child Social Problems Developmental Sweep": [
        (target, str(tp))
        for target in [
            "not_liked_p",
            "social_problems_D_p",
        ]
        for tp in range(5)  # T0 through T4
    ],
    # Parent Social Problems Developmental Sweep
    # ASR item (parent_not_liked_by_others_p) + relational composite
    # (parent_bad_relationships_p) across T0-T4. Use with sankey_style =
    # "parent_developmental_sweep".
    "Parent Social Problems Developmental Sweep": [
        (target, str(tp))
        for target in [
            "parent_not_liked_by_others_p",
            "parent_bad_relationships_p",
        ]
        for tp in range(5)  # T0 through T4
    ],
}

# Per-target FI barplot colors for batch mode
# When batch_mode != "Single Target", the batch loop overrides
# ensemble_barplot_color before each run_analysis() call so every
# target gets a visually distinct feature-importance palette.
# Convention:
# Child Psychopathology+Cog  -- domain-coded
# Child Dep and Anx          -- depression = blue, anxiety = red
# Parent Psychopathology     -- analogous hues to child counterparts
_BATCH_TARGET_COLORS = {
    # F3: Child Psychopathology & Cognition
    "internal_D_p": "mediumpurple",  # internalizing → purple
    "adhd_D_p": "gold",  # ADHD → yellow
    "external_D_p": "forestgreen",  # externalizing → green
    "social_problems_D_p": "darkorange",  # social problems → orange
    "not_liked_p": "chocolate",  # child CBCL social item → warm brown-orange
    "bad_grades": "slategray",  # cognitive/academic → gray
    "tb_flanker": "turquoise",  # cognitive/academic → turquoise (match Sankey)
    "suicidal_p": "#0A0A3C",  # suicidality → very dark blue
    "tb_reading": "silver",  # cognitive/academic → light gray
    # Child Mood & Anxiety (with Deltas)
    "sbt_core_depression": "royalblue",  # depression → blue
    "delta_sbt_core_depression": "deepskyblue",  # delta depression → lighter blue
    "anxdisord_D_p": "crimson",  # anxiety → red
    "delta_anxdisord_D_p": "tomato",  # delta anxiety → lighter red
    "sbt_anxiety_depression": "orchid",  # mixed anx/dep → purple-pink
    "top_5_sbt_core_depression": "steelblue",  # severe depression → steel blue
    "sbt_core_dep_onset_rci_2.3": "cornflowerblue",  # onset RCI → lighter cornflower
    # Parent Psychopathology (Mood/ADHD/Social)
    "parent_depress_D_p": "navy",  # parent depression → darker blue
    "parent_depressed_p": "midnightblue",  # parent depressed item → deep blue
    "parent_anxdisord_D_p": "firebrick",  # parent anxiety → darker red
    "parent_adhd_D_p": "orange",  # parent ADHD → warm yellow
    "parent_bad_relationships_p": "peru",  # parent social → warm brown/orange
    "parent_not_liked_by_others_p": "darkcyan",  # parent ASR social item → teal (complements peru)
    "parent_happy_person_p": "salmon",  # parent happiness → warm salmon
    "delta_parent_depressed_p": "deepskyblue",  # delta depression → lighter blue
    "delta_parent_worries_a_lot_p": "tomato",  # delta worry/anx → lighter red
    # Child Depression Worsening
    # Three thresholds: graded blue, softer for looser criterion, richer for stricter
    "dep_increase_1.5sd": "steelblue",  # child ≥1.5SD → mid blue
    "dep_increase_2sd": "dodgerblue",  # child ≥2SD → vivid cobalt
    "dep_onset_rci_2.3": "royalblue",  # child RCI 2.3 → deep royal blue
    # Parent Depression Worsening
    # Parallel graded scheme, shifted darker to distinguish parent from child
    "parent_dep_increase_1.5sd": "slateblue",  # parent ≥1.5SD → slate blue
    "parent_dep_increase_2sd": "darkslateblue",  # parent ≥2SD → deep navy
    "parent_dep_onset_rci_2.3": "midnightblue",  # parent RCI 2.3 → midnight blue
    # OCD/Impulsivity & Parent Substance Use
    "compulsions_p": "darkviolet",  # OCD compulsions → deep violet
    "obsessions_p": "darkorchid",  # OCD obsessions → adjacent orchid
    "perfectionist_p": "plum",  # OCD perfectionism → lighter purple
    "impulsive_p": "hotpink",  # impulsivity → high-energy pink
    "parent_impulsive_p": "deeppink",  # parent analogue → darker pink
    "parent_tobacco_use_frequency_p": "saddlebrown",  # tobacco → earthy brown
    "parent_drinks_frequency_p": "teal",  # alcohol → cool teal
    "parent_drunk_days_p": "darkcyan",  # alcohol variant → adjacent cyan
    # Parent Social/Impulsivity/Compulsivity
    # Colors mirror the sankey palette for this batch (sankey stage).
    # parent_not_liked_by_others_p and parent_impulsive_p already defined above.
    "parent_avoidant_person_D_p": "#4B6BBE",  # parent ASR avoidant personality → steel blue-violet (matches sankey)
    "parent_repeats_acts_p": "darkturquoise",  # parent ASR compulsions item → turquoise (matches sankey, mirrors child compulsions_p)
    "parent_obsessive_thoughts_p": "#8E5DBF",  # parent ASR obsessions item → violet (matches sankey, distinct from child darkorchid)
    # Suicidal Ideation (Child & Parent Dyad)
    "suicidal_p": "#0A0A3C",  # very dark blue
    "parent_suicidal_thoughts_p": "#0A0A3C",  # very dark blue (match child)
    # Child P-Factor & Parent P-Factor
    "parent_internal_D_p": "rebeccapurple",  # parent internalizing → deeper purple (child = mediumpurple)
    "parent_external_D_p": "seagreen",  # parent externalizing → muted green (child = forestgreen)
    # F4c: Subjective vs. Objective Outcomes
    "tb_cryst": "cadetblue",  # crystallized intelligence → cool blue-green
    "weight": "sienna",  # body weight → earthy brown
    # Mood/Depression Subtypes Children (SBT subscales)
    # sbt_core_depression already defined above as "royalblue"
    # sbt_anxiety_depression already defined above as "orchid"
    "sbt_guilt_hopelessness": "indigo",  # guilt/hopelessness → deep blue-purple
    "sbt_fatigue_sleep": "slateblue",  # fatigue/sleep → cool muted tone
    "sbt_somatic_depression": "rosybrown",  # somatic/physical → earthy flesh-adjacent
    "sbt_emotional_dysregulation": "orangered",  # volatile affect → hot, unstable
    "sbt_social_withdrawal": "cadetblue",  # retreat/isolation → cool, receding
    "sbt_avoidance_fear": "mediumpurple",  # anxiety-adjacent → lighter purple
    "sbt_aggression_irritability": "firebrick",  # anger/hostility → deep aggressive red
    "sbt_academic_cognitive": "#B8960B",  # academic/cognitive → golden amber (match Sankey)
    "sbt_perfectionism_achievement": "#C44A80",  # perfectionism/achievement → rose magenta (match Sankey)
    # Child Only Deltas
    "delta_adhd_D_p": "gold",  # ADHD → warm yellow (matches adhd_D_p)
    "delta_family_conflict_ss_k": "sienna",  # family conflict → burnt orange
    "delta_bad_grades": "slategray",  # academic → gray (matches bad_grades)
    "delta_social_problems_D_p": "darkorange",  # social → orange (matches social_problems_D_p)
    "delta_somatic_problems_D_p": "peru",  # somatic → warm brown
    "delta_depress_D_p": "royalblue",  # depression → blue
    # Parent Only Deltas
    "delta_parent_depress_D_p": "navy",  # parent depression syndrome → deep blue
    "delta_parent_sleep_trouble_p": "darkslateblue",  # sleep/insomnia → cool indigo
    "delta_parent_worries_about_family_p": "indianred",  # anxiety domain → warm red
    "delta_parent_worries_about_future_p": "firebrick",  # anxiety domain → deeper red
    "delta_parent_not_liked_by_others_p": "darkcyan",  # social domain → teal
    "delta_parent_feels_unloved_p": "steelblue",  # depression/mood → muted blue
    "delta_family_conflict_ss_p": "sienna",  # family conflict → burnt orange
    "delta_parent_concentration_trouble_p": "mediumpurple",  # ADHD/attention → purple
    "delta_parent_bad_relationships_p": "teal",  # social domain → teal-green
    "delta_parent_financial_trouble_p": "dimgray",  # SES/financial → stone gray
    "delta_parent_somatic_problems_D_p": "peru",  # somatic complaints → warm brown
    # Formal DepAnx NeuroCog
    "top_10_depression": "cornflowerblue",  # depression severity → soft blue
    "top_5_depression": "royalblue",  # severe depression → richer blue
    "two_more_depression_B_k": "steelblue",  # ≥2 symptoms → cool steel blue
    "MDD_KSADS_C": "midnightblue",  # MDD diagnosis → deepest blue
    "anhedonia_B_k": "slateblue",  # anhedonia → violet-blue
    "GAD_present": "crimson",  # GAD → anxiety red
    "excessive_worry_B_k": "tomato",  # excessive worry → lighter red
    "social_anxiety_disorder_B_k": "indianred",  # social anxiety → muted red
    # T0 Cognitive Task Battery
    # Palette aligned with the child_cogtask_t0 Sankey style, which itself
    # tracks the Objective/Subjective-Report Spectrum heatmap column hues.
    # tb_cryst already defined above as "cadetblue" (shared with F4c batch).
    # tb_flanker already defined above as "turquoise" (shared with F3 / child_alts).
    "tb_fluid": "midnightblue",  # NIH fluid composite → deep indigo navy (heatmap "Comp")
    "sst_mean_ssrt": "firebrick",  # SST response inhibition → dark red (heatmap "SST")
    "nb_correct_nt_2back": "teal",  # emotional WM → dark teal (heatmap "N-Back")
    "ravlt_l_total": "royalblue",  # verbal episodic memory → royal blue (heatmap "RAVLT")
    "lmt_efficiency": "saddlebrown",  # visuospatial mental rotation → brown (heatmap "LMT")
}

_mode_summary = {
    "All Visuals": "FI + SHAP (all models) + consensus + PDP + GAM",
    "Ensemble Consensus + Best SHAP": "Ensemble plots + consensus + SHAP (best model)",
    "Ensemble Plots Only": "Model comparison only -- fastest with visuals",
    "No Visuals": "Metrics only -- maximum speed",
}

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


# Function to create and tune Lasso or Ridge regression model
def create_tuned_linear_model(model_type, X_train, y_train, alpha_range=None):
    """
    Create and tune Lasso or Ridge regression model.

    Parameters:
    model_type : str
        'lasso' or 'ridge'
    X_train : DataFrame
        Training features
    y_train : Series/array
        Training target
    alpha_range : list, optional
        Range of alpha values to search. If None, uses default range.

    Returns:
    --------
    best_model : sklearn model
        Tuned Lasso or Ridge model
    """
    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 20)

    if model_type == "lasso":
        model = Lasso(random_state=42, max_iter=10000)
    else:  # ridge
        model = Ridge(max_iter=10000)

    param_grid = {"alpha": alpha_range}

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

    grid_search.fit(X_train, y_train)

    logger.info(f"Best alpha for {model_type}: {grid_search.best_params_['alpha']}")
    logger.info(f"Best score: {-grid_search.best_score_:.4f} (MSE)")

    return grid_search.best_estimator_


# Function to run linear model analysis with visualization
def run_linear_model_analysis(model_type, X_train, X_test, y_train, y_test, target_options, tp, category=None):
    """
    Run analysis with Lasso or Ridge regression

    Parameters:
    model_type : str
        'lasso' or 'ridge'
    X_train, X_test : DataFrame
        Training and test features
    y_train, y_test : Series/array
        Training and test targets
    target_options, tp, category : str/int
        Target variable, time point, and category for reporting
    """
    logger.info(f"\nRunning {model_type.upper()} Regression Analysis")

    # Train model with cross-validation to select alpha
    model = create_tuned_linear_model(model_type, X_train, y_train)

    # Get coefficients
    coefficients = pd.Series(model.coef_, index=X_train.columns)

    # Plot top coefficients
    plt.figure(figsize=(12, 8))
    top_coefficients = coefficients.abs().sort_values(ascending=False).head(20)
    top_coefficients = coefficients[top_coefficients.index]

    sns.barplot(x=top_coefficients.values, y=top_coefficients.index, palette="coolwarm")
    title = f"Top 20 {model_type.capitalize()} Coefficients ({target_options}, Timepoint {tp}"
    title += f", Category {category})" if category else ")"
    plt.title(title, fontsize=16)
    plt.xlabel("Coefficient Value", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    evaluation_metrics = pd.DataFrame({"Metric": ["MSE", "RMSE", "MAE", "R² Score"], "Value": [mse, rmse, mae, r2]})

    logger.info("\nEvaluation Metrics:")
    display(evaluation_metrics)

    # Metrics visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        evaluation_metrics.set_index("Metric").T, annot=True, fmt=".4f", cmap="coolwarm", linewidths=1.5, cbar=False
    )
    ax.set_title(f"Evaluation Metrics for {target_options} (Timepoint {tp})", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_type.capitalize()} Regression: Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    # Add to category results for spider plot (for regression)
    if category is not None:
        if category not in ctx.category_results:
            ctx.category_results[category] = {}

        ctx.category_results[category]["score"] = r2
        # Calculate confidence intervals using bootstrap
        bootstrap_r2_scores = []
        n_bootstraps = 100

        for _ in range(n_bootstraps):
            indices = np.random.choice(len(y_test), len(y_test), replace=True)
            bootstrap_r2 = r2_score(y_test.iloc[indices], y_pred[indices])
            bootstrap_r2_scores.append(bootstrap_r2)

        ci = 1.96 * np.std(bootstrap_r2_scores)
        ctx.category_results[category]["score_lower"] = r2 - ci
        ctx.category_results[category]["score_upper"] = r2 + ci
        ctx.category_results[category]["r2"] = r2

    # Return model and metrics
    return {"model": model, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "coefficients": coefficients}


def ensure_unique_feature_names(df):
    """
    Ensure all feature names in the DataFrame are unique by appending a suffix to duplicates.
    """
    feature_names = list(df.columns)
    unique_names = []
    name_counts = {}

    for name in feature_names:
        if name in name_counts:
            name_counts[name] += 1
            unique_names.append(f"{name}_{name_counts[name]}")
        else:
            name_counts[name] = 0
            unique_names.append(name)

    df.columns = unique_names
    return df


# Winsorise numeric features at ±n_sigmas to limit the influence of
# extreme values on the scaler and downstream learners.
def handle_outliers(X, n_sigmas=5):
    for column in X.select_dtypes(include=[np.number]).columns:
        mean = X[column].mean()
        std = X[column].std()
        X[column] = X[column].clip(lower=mean - n_sigmas * std, upper=mean + n_sigmas * std)
    return X


# Resolve NaN and infinite values in both X and y.
# Methods: 'remove' (listwise deletion), 'fill_0' (zero-fill),
# 'impute' (median/mean via SimpleImputer).
# y can arrive as either a Series (from within_categories scale_y) or a
# single-column DataFrame; both are handled transparently.
def clean_data(X, y, method="fill_0", imputer_strategy=None):
    # Normalize y to DataFrame so axis=1 operations and .columns work
    # uniformly; restore original type before returning.
    _y_was_series = isinstance(y, pd.Series)
    _y_series_name = y.name if _y_was_series else None
    if _y_was_series:
        y = y.to_frame(name=_y_series_name or "target")

    if method == "remove":
        # Remove rows with infinite values
        mask_X = ~np.isinf(X).any(axis=1)
        mask_y = ~np.isinf(y).any(axis=1)
        mask = mask_X & mask_y

        # Remove rows with NaN values
        mask_X = ~np.isnan(X).any(axis=1)
        mask_y = ~np.isnan(y).any(axis=1)
        mask = mask & mask_X & mask_y

        X_clean = X[mask]
        y_clean = y[mask]

    elif method == "fill_0":
        # Replace infinite values with 0
        X_clean = X.replace([np.inf, -np.inf], 0)
        y_clean = y.replace([np.inf, -np.inf], 0)

        # Replace NaN values with 0
        X_clean = X_clean.fillna(0)
        y_clean = y_clean.fillna(0)

    elif method == "impute":
        # Replace infinite values with NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        y_clean = y.replace([np.inf, -np.inf], np.nan)

        # Impute NaN values
        if imputer_strategy in ["mean", "median", "most_frequent"]:
            imputer = SimpleImputer(strategy=imputer_strategy)
            X_clean = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)

            # Fit a separate imputer for y to avoid data leakage from X
            y_imputer = SimpleImputer(strategy=imputer_strategy)
            y_clean = pd.DataFrame(y_imputer.fit_transform(y_clean), columns=y_clean.columns)

        else:
            raise ValueError("Invalid imputer_strategy. Choose 'mean', 'median', or 'most_frequent'.")

    else:
        raise ValueError("Invalid method. Choose 'remove', 'fill_0', or 'impute'.")

    # Restore y to its original type
    if _y_was_series and isinstance(y_clean, pd.DataFrame):
        y_clean = y_clean.iloc[:, 0]
        y_clean.name = _y_series_name

    return X_clean, y_clean


class LightweightInteractionAnalyzer:
    def __init__(self, model, metric="rmse", n_jobs=-1):
        """
        Initialize the analyzer

        Parameters:
        model : fitted model object
            Should have predict method
        metric : str, default='rmse'
            Metric to use for interaction strength ('rmse' or 'r2')
        n_jobs : int, default=-1
            Number of jobs for parallel processing
        """
        self.model = model
        self.metric = metric
        self.n_jobs = n_jobs
        self.interaction_scores = {}

    def _calculate_feature_importance(self, X, y, feature_name):
        """Calculate single feature importance using permutation"""
        baseline_pred = self.model.predict(X)
        baseline_score = root_mean_squared_error(y, baseline_pred)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)

        X_permuted = X.copy()
        X_permuted[feature_name] = np.random.permutation(X_permuted[feature_name])
        permuted_pred = self.model.predict(X_permuted)
        permuted_score = root_mean_squared_error(y, permuted_pred)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)

        return abs(permuted_score - baseline_score)

    def _calculate_interaction_strength(self, X, y, feat1, feat2):
        """Calculate interaction strength between two features"""
        # Calculate individual importance
        imp1 = self._calculate_feature_importance(X, y, feat1)
        imp2 = self._calculate_feature_importance(X, y, feat2)

        # Calculate joint importance
        X_permuted = X.copy()
        X_permuted[feat1] = np.random.permutation(X_permuted[feat1])
        X_permuted[feat2] = np.random.permutation(X_permuted[feat2])

        baseline_pred = self.model.predict(X)
        baseline_score = root_mean_squared_error(y, baseline_pred)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)

        permuted_pred = self.model.predict(X_permuted)
        joint_score = root_mean_squared_error(y, permuted_pred)  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)

        # Interaction strength is the difference between joint importance
        # and the sum of individual importances
        interaction = abs(joint_score - baseline_score) - (imp1 + imp2)
        return max(interaction, 0)  # non-negative

    def analyze_interactions(self, X, y, max_features=10):
        """
        Analyze feature interactions

        Parameters:
        X : pandas DataFrame
            Input features
        y : array-like
            Target variable
        max_features : int, default=10
            Maximum number of top features to consider for interactions
        """
        logger.info("Calculating feature importances...")
        # First, get individual feature importances
        importances = {}
        for feature in tqdm(X.columns):
            importances[feature] = self._calculate_feature_importance(X, y, feature)

        # Select top features based on individual importance
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in top_features[:max_features]]

        logger.info("\nAnalyzing feature interactions...")
        # Calculate interactions between top features
        for feat1, feat2 in tqdm(list(combinations(top_features, 2))):
            interaction_strength = self._calculate_interaction_strength(X, y, feat1, feat2)
            self.interaction_scores[(feat1, feat2)] = interaction_strength

        return self.interaction_scores

    def plot_interactions(self, X, top_n=10):
        """Plot interaction heatmap"""
        # Create interaction matrix
        features = list(set([f for pair in self.interaction_scores.keys() for f in pair]))
        n_features = len(features)
        interaction_matrix = np.zeros((n_features, n_features))

        for (feat1, feat2), strength in self.interaction_scores.items():
            i = features.index(feat1)
            j = features.index(feat2)
            interaction_matrix[i, j] = strength
            interaction_matrix[j, i] = strength

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(interaction_matrix, xticklabels=features, yticklabels=features, cmap="YlOrRd")
        plt.title("Feature Interaction Strengths")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Plot top interactions as bar chart
        sorted_interactions = sorted(self.interaction_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        plt.figure(figsize=(12, 6))
        interaction_names = [" × ".join(k) for k, _ in sorted_interactions]
        interaction_strengths = [v for _, v in sorted_interactions]

        plt.bar(range(len(sorted_interactions)), interaction_strengths)
        plt.xticks(range(len(sorted_interactions)), interaction_names, rotation=45, ha="right")
        plt.title("Top Feature Interactions")
        plt.ylabel("Interaction Strength")
        plt.tight_layout()
        plt.show()


def analyze_model_interactions(model, X_train, X_test, y_train, max_features=10):
    """Analyze feature interactions using lightweight methods"""
    analyzer = LightweightInteractionAnalyzer(model)
    interaction_scores = analyzer.analyze_interactions(X_train, y_train, max_features=max_features)
    analyzer.plot_interactions(X_train)
    return interaction_scores


# Optuna objective
def objective(trial, X, y, is_reg, num_classes):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-7, 1, log=True),
        "random_state": 42,
    }

    if is_reg:
        model = CatBoostRegressor(**params)
        eval_metric = "RMSE"
    else:
        if num_classes > 2:
            params["loss_function"] = "MultiClass"
            eval_metric = "MultiClass"
        else:
            params["loss_function"] = "Logloss"
            eval_metric = "Logloss"
        model = CatBoostClassifier(**params)

    pruning_callback = CatBoostPruningCallback(trial, eval_metric)
    model.fit(X, y, eval_set=[(X, y)], early_stopping_rounds=100, verbose=False, callbacks=[pruning_callback])

    if is_reg:
        return model.best_score_["validation"][eval_metric]
    else:
        return model.best_score_["validation"][eval_metric]


# Interaction feature constructor
def create_interaction_features(X, max_interactions=1000):
    initial_features = X.columns
    n_features = len(initial_features)

    # Calculate total possible interactions
    total_interactions = n_features * (n_features - 1) // 2

    # Generate all combinations
    all_combinations = list(combinations(range(n_features), 2))

    # If total interactions exceed max_interactions, randomly select interactions
    if total_interactions > max_interactions:
        indices = np.random.RandomState(42).permutation(total_interactions)[:max_interactions]
        feature_pairs = [all_combinations[i] for i in indices]
    else:
        feature_pairs = all_combinations

    # Create interaction features
    interaction_features = {}
    for i, j in feature_pairs:
        feat_A, feat_B = initial_features[i], initial_features[j]
        interaction_name = f"{feat_A}_x_{feat_B}"
        interaction_features[interaction_name] = X.iloc[:, i] * X.iloc[:, j]

    return pd.DataFrame(interaction_features)


def visualize_feature_importance(X_train, y_train, top_n=20):
    """
    Create visualizations for feature importance based on correlation with target

    Parameters:
    X_train : pandas DataFrame
        Training features including interaction terms
    y_train : pandas Series
        Target variable
    top_n : int
        Number of top features to show in plots
    """
    # y_train is a Series with a name
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if y_train.name is None:
        y_train.name = "target"

    # Calculate correlations with target
    correlations = X_train.apply(lambda x: abs(x.corr(y_train)) if x.name != y_train.name else 0)

    # Sort correlations
    top_correlations = correlations.sort_values(ascending=False).head(top_n)

    # Create figure with subplots
    plt.figure(figsize=(15, 10))

    # 1. Bar plot of top feature correlations
    plt.subplot(2, 1, 1)
    sns.barplot(x=top_correlations.values, y=top_correlations.index)
    plt.title(f"Top {top_n} Feature Correlations with Target")
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Features")

    # Separate interaction terms and original features
    interaction_features = [f for f in top_correlations.index if "_x_" in f]
    original_features = [f for f in top_correlations.index if "_x_" not in f]

    # Create color mapping
    colors = ["#ff9999" if "_x_" in f else "#66b3ff" for f in top_correlations.index]

    # 2. Horizontal bar plot with color-coded features
    plt.subplot(2, 1, 2)
    bars = plt.barh(range(len(top_correlations)), top_correlations.values, color=colors)
    plt.yticks(range(len(top_correlations)), top_correlations.index)
    plt.xlabel("Absolute Correlation")
    plt.title("Feature Importance (Blue: Original Features, Red: Interaction Terms)")

    plt.tight_layout()
    plt.show()


def visualize_interaction_matrix(X_train, y_train, top_n=20):
    """
    Create a heatmap of feature interactions

    Parameters:
    X_train : pandas DataFrame
        Training features including interaction terms
    y_train : pandas Series
        Target variable
    top_n : int
        Number of top features to include in heatmap
    """
    # Calculate correlations with target
    correlations = X_train.apply(lambda x: abs(x.corr(y_train)))

    # Get top features
    top_features = correlations.nlargest(top_n).index

    # Calculate correlation matrix for top features
    correlation_matrix = X_train[top_features].corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def calculate_safe_correlations(X_train, y_train):
    # Convert y_train to 1D array if it's not already
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0].values
    elif isinstance(y_train, pd.Series):
        y_train = y_train.values
    y_train = y_train.ravel()

    correlations = {}

    # Calculate correlations one at a time
    for feature in X_train.columns:
        feature_values = X_train[feature].values.ravel()
        # Remove any NaN values
        mask = ~(np.isnan(feature_values) | np.isnan(y_train))
        if mask.sum() > 0:
            correlation = np.corrcoef(feature_values[mask], y_train[mask])[0, 1]
            correlations[feature] = abs(correlation) if not np.isnan(correlation) else 0
        else:
            correlations[feature] = 0

    return pd.Series(correlations)


def analyze_and_visualize_interactions(X_train, y_train, top_n=20):
    # 1. Feature importance visualization
    logger.info("Plotting feature importance...")
    visualize_feature_importance(X_train, y_train, top_n)

    # 2. Interaction matrix visualization
    logger.info("\nPlotting interaction matrix...")

    # 3. Print top interaction terms
    interaction_features = [f for f in X_train.columns if "_x_" in f]

    if interaction_features:
        X_interactions = X_train[interaction_features]
        correlations = calculate_safe_correlations(X_interactions, y_train)

        top_interactions = correlations.sort_values(ascending=False).head(top_n)

        logger.info("\nTop 10 Interaction Features:")
        logger.info("-" * 50)
        for feat, corr in top_interactions.items():
            # Split feature name for better readability
            feat1, feat2 = feat.split("_x_")
            logger.info(f"{feat1} × {feat2}: {corr:.4f}")

        # Optional: Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_interactions)), top_interactions.values)
        plt.xticks(
            range(len(top_interactions)),
            [f"{f.split('_x_')[0]}\n×\n{f.split('_x_')[1]}" for f in top_interactions.index],
            rotation=45,
        )
        plt.title("Top Feature Interactions by Correlation with Target")
        plt.ylabel("Absolute Correlation")
        plt.tight_layout()
        plt.show()

    return None


def plot_specific_interaction(model, X, feature1, feature2, n_samples=1000):
    """Plot detailed interaction between two specific features."""
    if len(X) > n_samples:
        X = X.sample(n_samples, random_state=42)

    # Create 2D grid of feature values
    x1_range = np.linspace(X[feature1].min(), X[feature1].max(), 20)
    x2_range = np.linspace(X[feature2].min(), X[feature2].max(), 20)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    # Create predictions for grid
    grid_data = X.copy()
    predictions = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            grid_data[feature1] = xx1[i, j]
            grid_data[feature2] = xx2[i, j]
            predictions[i, j] = model.predict(grid_data).mean()

    # Plot interaction
    plt.figure(figsize=(10, 8))
    plt.contourf(xx1, xx2, predictions, levels=20, cmap="viridis")
    plt.colorbar(label="Predicted Value")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"Interaction Effect: {feature1} × {feature2}")
    plt.tight_layout()
    plt.show()


def calculate_h_statistics_batch(model, X, feature_pairs, n_samples=1000, batch_size=100):
    """Calculate H-statistics for multiple feature pairs in batches."""
    if len(X) > n_samples:
        X = X.sample(n_samples, random_state=42)

    results = []

    # Process feature pairs in batches
    for i in range(0, len(feature_pairs), batch_size):
        batch_pairs = feature_pairs[i : i + batch_size]
        batch_results = []

        # Get base predictions once for this batch (each batch one base)
        base_pred = model.predict(X)

        for feat1, feat2 in batch_pairs:
            # Create all permutations at once
            X_perm = X.copy()
            X_perm[feat1] = np.random.permutation(X_perm[feat1].values)
            pred1 = model.predict(X_perm)

            X_perm[feat2] = np.random.permutation(X_perm[feat2].values)
            pred_both = model.predict(X_perm)

            X_perm = X.copy()
            X_perm[feat2] = np.random.permutation(X_perm[feat2].values)
            pred2 = model.predict(X_perm)

            # Calculate H-statistic
            effect1 = np.mean(np.abs(base_pred - pred1))
            effect2 = np.mean(np.abs(base_pred - pred2))
            effect_both = np.mean(np.abs(base_pred - pred_both))

            h_stat = max(0, effect_both - (effect1 + effect2))
            batch_results.append(h_stat)

        results.extend(batch_results)

    return results


def select_features_with_h_statistics(X_train, X_test, y_train, is_reg=True, max_features=1000, top_interactions=50):
    """
    Select features based on H-statistics for interaction strength.

    Parameters:
    X_train : pandas DataFrame
        Training features
    X_test : pandas DataFrame
        Test features
    y_train : array-like
        Training target
    is_reg : bool
        Whether this is a regression (True) or classification (False) problem
    max_features : int
        Maximum number of features to return after selection
    top_interactions : int
        Number of top interactions to consider for feature creation

    Returns:
    --------
    X_train_selected : pandas DataFrame
        Training data with selected original and interaction features
    X_test_selected : pandas DataFrame
        Test data with selected original and interaction features
    """

    if is_reg:
        model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=42)
    else:
        model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=42)

    logger.info("Training initial model...")
    model.fit(X_train, y_train)

    # Generate feature pairs
    features = list(X_train.columns)
    feature_pairs = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]

    # Calculate H-statistics with batching and sampling
    logger.info("Calculating H-statistics for feature pairs...")
    h_stats = calculate_h_statistics_batch(
        model,
        X_train,
        feature_pairs,
        n_samples=500,  # Reduced sample size
        batch_size=100,
    )  # Batch processing

    # Create interaction strengths list
    interaction_strengths = [(pair[0], pair[1], stat) for pair, stat in zip(feature_pairs, h_stats)]

    # Sort by interaction strength
    interaction_strengths.sort(key=lambda x: x[2], reverse=True)

    # Create interaction features for top pairs
    logger.info(f"\nCreating interaction features for top {top_interactions} pairs...")
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()

    for feat1, feat2, strength in interaction_strengths[:top_interactions]:
        feat_name = f"{feat1}_x_{feat2}"
        X_train_new[feat_name] = X_train[feat1] * X_train[feat2]
        X_test_new[feat_name] = X_test[feat1] * X_test[feat2]

    # Calculate feature importance including new interaction features
    logger.info("Calculating final feature importance...")
    model.fit(X_train_new, y_train)
    feature_importance = model.get_feature_importance()

    # Select top features based on importance
    feature_ranks = pd.Series(feature_importance, index=X_train_new.columns)
    selected_features = feature_ranks.nlargest(max_features).index

    # Visualize feature importance and interactions
    plt.figure(figsize=(15, 10))

    # Plot feature importance
    plt.subplot(2, 1, 1)
    top_20_features = feature_ranks.nlargest(20)
    sns.barplot(x=top_20_features.values, y=top_20_features.index)
    plt.title("Top 20 Features (Including Interactions)")
    plt.xlabel("Feature Importance")

    # Plot top interactions
    plt.subplot(2, 1, 2)
    top_interactions_df = pd.DataFrame(interaction_strengths[:10], columns=["Feature 1", "Feature 2", "H-Statistic"])
    sns.barplot(
        data=top_interactions_df,
        x="H-Statistic",
        y=top_interactions_df.apply(lambda x: f"{x['Feature 1']}\n×\n{x['Feature 2']}", axis=1),
    )
    plt.title("Top 10 Feature Interactions (H-Statistics)")
    plt.tight_layout()
    plt.show()

    # Create heatmap of top interactions
    n_top = 10  # Number of top features to show in heatmap
    top_features = list(set([f for pair in interaction_strengths[:n_top] for f in pair[:2]]))
    n_features = len(top_features)
    interaction_matrix = np.zeros((n_features, n_features))

    # Fill interaction matrix
    for i, feat1 in enumerate(top_features):
        for j, feat2 in enumerate(top_features):
            if i < j:
                # Find this pair in interaction_strengths if it exists
                strength = next(
                    (
                        s
                        for f1, f2, s in interaction_strengths
                        if (f1 == feat1 and f2 == feat2) or (f1 == feat2 and f2 == feat1)
                    ),
                    0,
                )
                interaction_matrix[i, j] = strength
                interaction_matrix[j, i] = strength

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        interaction_matrix,
        xticklabels=top_features,
        yticklabels=top_features,
        cmap="YlOrRd",
        annot=True,
        fmt=".2e",
        annot_kws={"size": 6},
    )  # Smaller font for in-cell heatmap annotations

    plt.title("Feature Interaction Strength Heatmap (H-Statistics)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Print top interaction details
    logger.info("\nTop 10 Feature Interactions (H-Statistics):")
    logger.info("-" * 50)
    for feat1, feat2, strength in interaction_strengths[:10]:
        logger.info(f"{feat1} × {feat2}: {strength:.4e}")

    # Return selected features
    X_train_selected = X_train_new[selected_features]
    X_test_selected = X_test_new[selected_features]

    logger.info(f"\nFinal number of features: {len(selected_features)}")
    logger.info(f"Number of interaction features: {sum('_x_' in f for f in selected_features)}")

    return X_train_selected, X_test_selected


def select_features_with_shap_interactions(
    X_train, X_test, y_train, split, is_reg=True, max_features=1000, top_interactions=50
):
    """
    Select features based on SHAP interaction values.

    Parameters:
    X_train : pandas DataFrame
        Training features
    X_test : pandas DataFrame
        Test features
    y_train : array-like
        Training target
    is_reg : bool
        Whether this is a regression (True) or classification (False) problem
    max_features : int
        Maximum number of features to return after selection
    top_interactions : int
        Number of top interactions to consider for feature creation

    Returns:
    --------
    X_train_selected : pandas DataFrame
        Training data with selected original and interaction features
    X_test_selected : pandas DataFrame
        Test data with selected original and interaction features
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    # Train a simple CatBoost model for SHAP analysis
    if is_reg:
        model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=42)
    else:
        model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=42)

    # Fit the model
    logger.info("Training initial model for SHAP analysis...")
    model.fit(X_train, y_train)

    # Calculate SHAP interaction values
    logger.info("Calculating SHAP interaction values...")
    explainer = shap.TreeExplainer(model)

    # Use a sample of the training data for interaction calculations
    sample_size = min(1000, len(X_train))
    X_train_sample = X_train.iloc[:sample_size]
    shap_interaction_values = explainer.shap_interaction_values(X_train_sample)

    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[0]

    # Calculate mean absolute SHAP interaction values
    mean_interactions = np.abs(shap_interaction_values).mean(axis=0)

    # Get feature pairs with strongest interactions
    interaction_strengths = []
    features = X_train.columns

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            strength = (mean_interactions[i, j] + mean_interactions[j, i]) / 2
            interaction_strengths.append((features[i], features[j], strength))

    # Sort interactions by strength
    interaction_strengths.sort(key=lambda x: x[2], reverse=True)

    # Create new interaction features for top pairs
    logger.info(f"Creating interaction features for top {top_interactions} pairs...")
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()

    for feat1, feat2, strength in interaction_strengths[:top_interactions]:
        feat_name = f"{feat1}_x_{feat2}"
        X_train_new[feat_name] = X_train[feat1] * X_train[feat2]
        X_test_new[feat_name] = X_test[feat1] * X_test[feat2]

    # Calculate feature importance including new interaction features
    logger.info("Calculating final feature importance...")
    model.fit(X_train_new, y_train)
    feature_importance = model.get_feature_importance()

    # Select top features based on importance
    feature_ranks = pd.Series(feature_importance, index=X_train_new.columns)
    selected_features = feature_ranks.nlargest(max_features).index

    # Visualize feature importance and interactions
    plt.figure(figsize=(15, 10))

    # Plot feature importance
    plt.subplot(2, 1, 1)
    top_20_features = feature_ranks.nlargest(20)
    sns.barplot(x=top_20_features.values, y=top_20_features.index)
    plt.title("Top 20 Features (Including Interactions)")
    plt.xlabel("Feature Importance")

    # Plot top interactions
    plt.subplot(2, 1, 2)
    top_interactions_df = pd.DataFrame(
        interaction_strengths[:10], columns=["Feature 1", "Feature 2", "Interaction Strength"]
    )
    sns.barplot(
        data=top_interactions_df,
        x="Interaction Strength",
        y=top_interactions_df.apply(lambda x: f"{x['Feature 1']}\n×\n{x['Feature 2']}", axis=1),
    )
    plt.title("Top 10 Feature Interactions")

    plt.tight_layout()
    plt.show()

    # Calculate feature importance including new interaction features
    logger.info("Calculating final feature importance...")
    model.fit(X_train_new, y_train)
    feature_importance = model.get_feature_importance()

    # Select top features based on importance
    feature_ranks = pd.Series(feature_importance, index=X_train_new.columns)
    selected_features = feature_ranks.nlargest(max_features).index

    # Create heatmap of top interactions

    n_top = 10  # Number of top features to show in heatmap
    top_features = list(set([f for pair in interaction_strengths[:n_top] for f in pair[:2]]))
    n_features = len(top_features)
    interaction_matrix = np.zeros((n_features, n_features))

    # Fill interaction matrix
    for i, feat1 in enumerate(top_features):
        for j, feat2 in enumerate(top_features):
            if i < j:
                # Find this pair in interaction_strengths if it exists
                strength = next(
                    (
                        s
                        for f1, f2, s in interaction_strengths
                        if (f1 == feat1 and f2 == feat2) or (f1 == feat2 and f2 == feat1)
                    ),
                    0,
                )
                interaction_matrix[i, j] = strength
                interaction_matrix[j, i] = strength

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        interaction_matrix, xticklabels=top_features, yticklabels=top_features, cmap="YlOrRd", annot=True, fmt=".2e"
    )
    plt.title("Feature Interaction Strength Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Print top interaction details
    if split == "within_categories":
        logger.info("\nTop 10 Feature Interactions:")
        logger.info("-" * 50)
        for feat1, feat2, strength in interaction_strengths[:10]:
            logger.info(f"{feat1} × {feat2}: {strength:.4e}")
    else:
        logger.info("\nTop 50 Feature Interactions:")
        logger.info("-" * 50)
        for feat1, feat2, strength in interaction_strengths[:50]:
            logger.info(f"{feat1} × {feat2}: {strength:.4e}")

    # Return selected features
    X_train_selected = X_train_new[selected_features]
    X_test_selected = X_test_new[selected_features]

    logger.info(f"\nFinal number of features: {len(selected_features)}")
    logger.info(f"Number of interaction features: {sum('_x_' in f for f in selected_features)}")

    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap

    # Create network visualization of top interactions
    logger.info("Creating interaction network graph...")

    # Create graph
    G = nx.Graph()

    # Get top N interactions for the graph
    top_n = min(20, len(interaction_strengths))  # Adjust based on readability
    top_interactions_for_graph = sorted(interaction_strengths[:top_n], key=lambda x: x[2], reverse=True)

    # Add nodes and edges
    for feat1, feat2, strength in top_interactions_for_graph:
        # Add nodes if they don't exist
        if feat1 not in G:
            G.add_node(feat1)
        if feat2 not in G:
            G.add_node(feat2)

        # Add edge with weight based on interaction strength
        G.add_edge(feat1, feat2, weight=strength)

    # Calculate node size based on individual feature importance
    node_importance = {}
    feature_importance = model.get_feature_importance()
    feature_dict = dict(zip(X_train.columns, feature_importance))

    for node in G.nodes():
        node_importance[node] = feature_dict.get(node, 0)

    # Normalize node sizes for visualization
    max_size = 800
    min_size = 300
    if max(node_importance.values()) > 0:  # Avoid division by zero
        node_sizes = [
            min_size + (max_size - min_size) * node_importance[node] / max(node_importance.values())
            for node in G.nodes()
        ]
    else:
        node_sizes = [min_size for node in G.nodes()]

    # Create figure
    plt.figure(figsize=(16, 12), facecolor="white")

    # Use a shell layout for more square-like appearance
    pos = nx.kamada_kawai_layout(G)  # This often gives more evenly spaced nodes

    # Create custom colormap for edges - from light blue to pink
    colors = ["#a6cffe", "#ff9dc0"]  # Light blue to pink
    edge_cmap = LinearSegmentedColormap.from_list("edge_colormap", colors)

    # Get max weight for normalization
    max_weight = max([G[u][v]["weight"] for u, v in G.edges()])

    # Draw edges with color based on weight and width proportional to weight
    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        normalized_weight = weight / max_weight
        edge_width = 1 + 8 * normalized_weight  # Scale width between 1 and 9
        edge_color = edge_cmap(normalized_weight)

        # Draw each edge individually with its own color and width
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_width, edge_color=[edge_color], alpha=0.8)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="#4a90e2",  # Medium blue
        alpha=0.8,
        linewidths=1,
        edgecolors="navy",
    )

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_family="sans-serif")

    # Add edge labels with weights
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2e}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title("Feature Interaction Network Graph", fontsize=18, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    from matplotlib import cm

    return X_train_selected, X_test_selected


def plot_shap_interaction_violin_grid(model, X, top_n_features=4):
    """
    Create a grid of violin plots showing SHAP interaction values between top features.

    Parameters:
    model : fitted CatBoost model
        The trained model to analyze
    X : pandas DataFrame
        Feature data to analyze
    top_n_features : int
        Number of top features to include in the grid
    """
    # Calculate SHAP interaction values
    explainer = shap.TreeExplainer(model)

    # Use a sample of data if the dataset is large
    sample_size = min(1000, len(X))
    X_sample = X.iloc[:sample_size]

    logger.info("Calculating SHAP interaction values...")
    shap_interaction_values = explainer.shap_interaction_values(X_sample)

    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[0]

    # Get feature importance based on mean absolute SHAP values
    feature_importance = np.abs(shap_interaction_values).mean(axis=0).sum(axis=1)
    top_indices = np.argsort(-feature_importance)[:top_n_features]
    top_features = X.columns[top_indices]

    # Create figure
    fig, axes = plt.subplots(top_n_features, top_n_features, figsize=(3 * top_n_features, 3 * top_n_features))

    # Create violin plots for each feature pair
    for i, feat1 in enumerate(top_features):
        for j, feat2 in enumerate(top_features):
            ax = axes[i, j]

            # Get indices for the features
            idx1 = list(X.columns).index(feat1)
            idx2 = list(X.columns).index(feat2)

            # Get interaction values
            interaction_vals = shap_interaction_values[:, idx1, idx2]

            # Create violin plot
            if i != j:  # Off-diagonal plots
                # Split data based on feature value being above/below median
                median_val = np.median(X_sample[feat2])
                high_mask = X_sample[feat2] > median_val

                # Create violin plot with two sides
                parts = ax.violinplot(
                    [interaction_vals[~high_mask], interaction_vals[high_mask]],
                    positions=[0, 0],
                    vert=True,
                    widths=0.7,
                    showmeans=True,
                    showextrema=True,
                )

                # Color the different sides
                for pc in parts["bodies"]:
                    if parts["bodies"].index(pc) == 0:
                        pc.set_facecolor("red")
                        pc.set_alpha(0.4)
                    else:
                        pc.set_facecolor("blue")
                        pc.set_alpha(0.4)
            else:  # Diagonal plots
                # Single violin for main effects
                parts = ax.violinplot(
                    interaction_vals, positions=[0], vert=True, widths=0.7, showmeans=True, showextrema=True
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor("blue")
                    pc.set_alpha(0.4)

            # Set plot limits and labels
            ax.set_ylim(-np.abs(interaction_vals).max(), np.abs(interaction_vals).max())
            ax.set_xlim(-1, 1)

            # Remove x-axis ticks
            ax.set_xticks([])

            # Add feature labels
            if i == 0:
                ax.set_title(feat2, fontsize=10)
            if j == 0:
                ax.set_ylabel(feat1, fontsize=10)

            # Add grid
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle("SHAP Interaction Values Distribution", y=1.02, fontsize=14)
    plt.show()

    # Print feature importance summary
    logger.info("\nFeature Importance Based on Interactions:")
    logger.info("-" * 50)
    for feat, importance in zip(top_features, feature_importance[top_indices]):
        logger.info(f"{feat}: {importance:.4e}")


def analyze_interactions_with_violin(model, X_train, X_test, y_train, is_reg=True, top_n_features=25):
    """
    Wrapper function to analyze interactions and create violin plot grid.
    """
    # Train model if not already fitted
    if not hasattr(model, "feature_importances_"):
        logger.info("Training model...")
        model.fit(X_train, y_train)

    # Create violin plot grid
    logger.info("Creating interaction violin plot grid...")
    plot_shap_interaction_violin_grid(model, X_test, top_n_features)

    return model


# SHAP beeswarm summary
def plot_shap_interaction_grid(model, X, top_n_features=25):
    """
    Create a grid of beeswarm plots showing SHAP interaction values between top features.
    """
    try:
        # Calculate SHAP interaction values
        explainer = shap.TreeExplainer(model)

        # Use a sample of data if the dataset is large
        sample_size = min(1000, len(X))
        X_sample = X.iloc[:sample_size]

        logger.info("Calculating SHAP interaction values...")
        shap_interaction_values = explainer.shap_interaction_values(X_sample)

        if isinstance(shap_interaction_values, list):
            shap_interaction_values = shap_interaction_values[0]

        # Get feature importance based on mean absolute SHAP values
        feature_importance = np.abs(shap_interaction_values).mean(axis=0).sum(axis=1)
        top_indices = np.argsort(-feature_importance)[:top_n_features]
        top_features = X.columns[top_indices]

        # Create figure
        fig, axes = plt.subplots(top_n_features, top_n_features, figsize=(2 * top_n_features, 2 * top_n_features))

        # Create beeswarm plots for each feature pair
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features):
                ax = axes[i, j]

                # Get indices for the features
                idx1 = list(X.columns).index(feat1)
                idx2 = list(X.columns).index(feat2)

                # Get interaction values and feature values
                interaction_vals = shap_interaction_values[:, idx1, idx2]
                feature_vals = X_sample[feat2].values

                if i != j:  # Off-diagonal plots
                    # Create scatter plot with jitter
                    if len(interaction_vals) > 0:
                        # Add jitter to x-axis
                        x_jitter = np.random.normal(0, 0.1, size=len(interaction_vals))

                        # Color points based on feature value
                        colors = feature_vals
                        scatter = ax.scatter(x_jitter, interaction_vals, c=colors, cmap="coolwarm", alpha=0.6, s=20)

                else:  # Diagonal plots
                    if len(interaction_vals) > 0:
                        x_jitter = np.random.normal(0, 0.1, size=len(interaction_vals))
                        ax.scatter(x_jitter, interaction_vals, color="blue", alpha=0.6, s=20)

                # Set plot limits and labels
                if len(interaction_vals) > 0:
                    ax.set_ylim(-np.abs(interaction_vals).max(), np.abs(interaction_vals).max())
                ax.set_xlim(-1, 1)

                # Remove x-axis ticks
                ax.set_xticks([])

                # Add feature labels
                if i == 0:
                    ax.set_title(feat2, fontsize=8, rotation=45, ha="right")
                if j == 0:
                    ax.set_ylabel(feat1, fontsize=8)

                # Add grid
                ax.grid(True, alpha=0.3)

                # Add colorbar for off-diagonal plots
                if i != j and i == top_n_features - 1 and j == top_n_features - 1:
                    plt.colorbar(scatter, ax=ax, label=feat2)

        plt.tight_layout()
        plt.suptitle("SHAP Interaction Values Distribution", y=1.02, fontsize=14)
        plt.show()

        # Print feature importance summary
        logger.info("\nFeature Importance Based on Interactions:")
        logger.info("-" * 50)
        for feat, importance in zip(top_features, feature_importance[top_indices]):
            logger.info(f"{feat}: {importance:.4e}")

    except Exception as e:
        logger.info(f"Error in plotting SHAP interactions: {str(e)}")
        logger.info("Continuing with the rest of the analysis...")


def analyze_interactions_with_beeswarm(model, X_train, X_test, y_train, is_reg=True, top_n_features=25):
    """
    Wrapper function to analyze interactions and create beeswarm plot grid.
    Parameters:
    model : CatBoost model
        The model to analyze
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series/array
        Training target
    is_reg : bool
        Whether this is a regression task
    top_n_features : int
        Number of top features to show in the interaction grid
    """
    logger.info("Training model...")
    # Fit the model first
    model.fit(X_train, y_train)

    # Create beeswarm plot grid
    logger.info("Creating interaction beeswarm plot grid...")
    plot_shap_interaction_grid(model, X_test, top_n_features)

    return model


def compare_metrics_across_categories(metrics_dict):
    """Compare and visualize metrics across different categories."""
    # Create DataFrame for comparison
    categories = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for cat, metrics in metrics_dict.items():
        categories.append(cat)
        accuracies.append(metrics["accuracy"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])
        if "auc" in metrics:
            aucs.append(metrics["auc"])
        else:
            aucs.append(None)

    comparison_df = pd.DataFrame(
        {
            "Category": categories,
            "Accuracy": accuracies,
            "Precision": precisions,
            "Recall": recalls,
            "F1": f1s,
            "AUC-ROC": aucs,
        }
    )

    # Visualize comparison
    plt.figure(figsize=(15, 8))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
    if not all(auc is None for auc in aucs):
        metrics_to_plot.append("AUC-ROC")

    comparison_df_melted = comparison_df.melt(
        id_vars=["Category"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score"
    )

    sns.barplot(data=comparison_df_melted, x="Category", y="Score", hue="Metric")
    plt.title("Comparison of Metrics Across Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print comparison table
    logger.info("\nMetrics Comparison Across Categories:")
    logger.info(comparison_df.to_string(index=False))


# Flatten the nested metrics_dict_full structure into a single DataFrame
# with one row per (category × model), sorted by R² descending.
def create_metrics_dataframe(metrics_dict):
    rows = []

    # Iterate through each category and its corresponding metrics
    for category, metrics in zip(metrics_dict["category"], metrics_dict["metrics"]):
        # For each model and its scores in the metrics
        for model, scores in metrics.items():
            # Create a row with category, model, and all metrics
            row = {
                "category": category,
                "model": model,
                **scores,  # Unpack all metrics (mse, rmse, mae, r2)
            }
            rows.append(row)

    # Create DataFrame from the rows
    df = pd.DataFrame(rows)

    # Sort by R² score in descending order
    try:
        df_sorted = df.sort_values("r2", ascending=False)
    except Exception as e:
        logger.info(f"[WARNING] {e}")
        df_sorted = df

    return df_sorted


def run_analysis(
    tp_option,
    group,
    target_options,
    split,
    isolate=False,
    isolate_category=None,
    quantile_segment=False,
    feature_tp=None,
):
    """
    Run analysis with option to isolate a category and analyze all its combinations.

    Parameters:
    tp_option : str
        Time point option
    group : str
        Group filter
    target_options : str
        Target variable
    split : str
        Split type ('across_categories' or 'within_categories')
    isolate : bool, default=False
        Whether to run isolation analysis
    isolate_category : str, default=None
        Category to combine with all others when isolate=True
    """
    tp = tp_option if tp_option else 2

    ctx.results_summary = []
    ctx.consensus_df = None
    ctx.model_weights = None
    ctx.category_results = {}  # Score accumulator for spider plot; reset each run_analysis call.

    # Stores evaluation metrics for each category and time point
    ctx.metrics_dict_full = {"category": [], "metrics": []}

    if tp_option == 0 and group == "high_ale":
        logger.info("No TP = 0 with high_ale=True")
        return

    if split == "across_categories":
        ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = across_categories_data(
            int(tp_option), [f"{target_options}"], group=group, scale_y=False, verbose=True, feature_tp=feature_tp
        )
        _, ctx.consensus_df, ctx.model_weights = run_catboost_analysis(
            ctx.X_train,
            ctx.X_test,
            ctx.y_train,
            ctx.y_test,
            target_options,
            tp,
            hyperparameter_tuning=ctx.hyperparameter_tuning,
            conformal_prediction=ctx.conformal_prediction,
        )

    elif split == "within_categories":
        if ctx.single_category_mode and isolate_category and isolate_category != "None":
            # Quick single-category mode
            # Runs only the selected category instead of looping all.
            logger.info(f"\n Single-category mode: running only '{isolate_category}'")
            within_categories_results = within_categories_data(
                int(tp_option), [f"{target_options}"], group=group, scale_y=False, verbose=True, feature_tp=feature_tp
            )

            if isolate_category in within_categories_results:
                ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = within_categories_results[isolate_category]
                logger.info(f"\nAnalyzing category: {isolate_category}")
                logger.info(f"Features shape: {ctx.X_train.shape}")

                _, _cdf, _mw = run_catboost_analysis(
                    ctx.X_train,
                    ctx.X_test,
                    ctx.y_train,
                    ctx.y_test,
                    target_options,
                    tp,
                    category=isolate_category,
                    hyperparameter_tuning=ctx.hyperparameter_tuning,
                    conformal_prediction=ctx.conformal_prediction,
                )
                if _cdf is not None:
                    ctx.consensus_df = _cdf
                    ctx.model_weights = _mw

                if ctx.ensemble_version:
                    results_ensemble = create_metrics_dataframe(ctx.metrics_dict_full)
                    try:
                        results_meta = results_ensemble[results_ensemble["model"] == "meta_model"]
                        results_meta.to_csv("results_ensemble_meta_model.csv", index=False)
                        display(results_meta)
                    except Exception as e:
                        logger.info(f"[WARNING] {e}")
                        display(results_ensemble)
                        results_ensemble.to_csv("results_ensemble.csv", index=False)
                    logger.info("Results ensemble saved.")
                else:
                    try:
                        results_summary_df = pd.DataFrame(ctx.results_summary)
                        display(results_summary_df)
                        results_summary_df.to_csv("results_summary.csv", index=False)
                        logger.info("Results summary saved.")
                    except Exception as e:
                        logger.info("No results summary to save.")
            else:
                _available = list(within_categories_results.keys())
                logger.info(f"Warning:  Category '{isolate_category}' not found at T{tp_option}.")
                logger.info(f"   Available categories: {_available}")

        elif isolate:
            # Get combinations for isolated category
            logger.info(f"\nAnalyzing all combinations for {isolate_category}:")
            combinations_results = within_categories_data_combinations(
                int(tp_option),
                [f"{target_options}"],
                base_category=isolate_category,
                group=group,
                scale_y=False,
                verbose=True,
            )

            # quantile_results = analyze_category_quantiles(combinations_data)
            # print_quantile_analysis(quantile_results)

            # Process each combination
            for combination_name, (ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test) in combinations_results.items():
                logger.info(f"\nAnalyzing combination: {combination_name}")
                logger.info(f"Features shape: {ctx.X_train.shape}")

                _, _cdf, _mw = run_catboost_analysis(
                    ctx.X_train,
                    ctx.X_test,
                    ctx.y_train,
                    ctx.y_test,
                    target_options,
                    tp,
                    category=combination_name,
                    hyperparameter_tuning=ctx.hyperparameter_tuning,
                    conformal_prediction=ctx.conformal_prediction,
                )
                if _cdf is not None:
                    ctx.consensus_df = _cdf
                    ctx.model_weights = _mw

            try:
                results_summary_df = pd.DataFrame(ctx.results_summary)
                display(results_summary_df)
                results_summary_df.to_csv("results_summary.csv", index=False)
                logger.info("Results summary saved.")
            except Exception as e:
                logger.info("No results summary to save.")

        else:
            within_categories_results = within_categories_data(
                int(tp_option), [f"{target_options}"], group=group, scale_y=False, verbose=True, feature_tp=feature_tp
            )

            # Quantile analysis
            quantile_results = analyze_category_quantiles(within_categories_results)

            for category, data in within_categories_results.items():
                logger.info(f"\nAnalyzing category: {category}")
                ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = data

                if category == "social_relationships" or category == "other_psychopathology":
                    ctx.X_train = ctx.X_train.loc[:, ~ctx.X_train.columns.duplicated()]
                    ctx.X_test = ctx.X_test.loc[:, ~ctx.X_test.columns.duplicated()]

                _, _cdf, _mw = run_catboost_analysis(
                    ctx.X_train,
                    ctx.X_test,
                    ctx.y_train,
                    ctx.y_test,
                    target_options,
                    tp,
                    category,
                    hyperparameter_tuning=ctx.hyperparameter_tuning,
                    conformal_prediction=ctx.conformal_prediction,
                    quantile_segment=quantile_segment,
                )
                if _cdf is not None:
                    ctx.consensus_df = _cdf
                    ctx.model_weights = _mw

            if not ctx.ensemble_version:
                try:
                    results_summary_df = pd.DataFrame(ctx.results_summary)
                    display(results_summary_df)
                    results_summary_df.to_csv("results_summary.csv", index=False)
                    logger.info("Results summary saved.")
                except Exception as e:
                    logger.info("No results summary to save.")
            else:
                results_ensemble = create_metrics_dataframe(ctx.metrics_dict_full)
                try:
                    results_meta = results_ensemble[results_ensemble["model"] == "meta_model"]
                    results_meta.to_csv("results_ensemble_meta_model.csv", index=False)
                    display(results_meta)
                except Exception as e:
                    logger.info(f"[WARNING] {e}")
                    display(results_ensemble)
                    results_ensemble.to_csv("results_ensemble.csv", index=False)

                logger.info("Results ensemble saved.")


# Main per-category modeling loop. Preprocesses the category slice
# (encoding, outlier handling, scaling), fits either the stacking
# ensemble or a standalone CatBoost model, and collects performance
# metrics, SHAP importance, partial dependence, and optional
# conformal prediction intervals.
def run_catboost_analysis(
    X_train,
    X_test,
    y_train,
    y_test,
    target_options,
    tp,
    category=None,
    hyperparameter_tuning=False,
    conformal_prediction=False,
    quantile_segment=False,
):

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import RandomizedSearchCV, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    try:
        from sklearn.metrics import root_mean_squared_error  # sklearn ≥ 1.4
    except ImportError:
        from sklearn.metrics import mean_squared_error as _mse

        def root_mean_squared_error(y_true, y_pred):
            return _mse(y_true, y_pred) ** 0.5

    import matplotlib.pyplot as plt
    import shap
    from imblearn.over_sampling import SMOTE, SMOTENC
    from scipy.stats import randint, uniform
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import OneHotEncoder

    logger.info(f"Original shape: {X_train.shape}")

    # One-hot encode categorical features before modeling.
    one_hot_features = ["marital_status", "child_religion", "sex_orient_y", "trans_id_y", "crysflu_c"]

    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

    # Step 1: Encode categorical variables
    le = LabelEncoder()
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        X_train[column] = le.fit_transform(X_train[column].astype(str))
        X_test[column] = le.transform(X_test[column].astype(str))
    logger.info("Categorical variables encoded")

    # Step 2: Remove constant features
    try:
        constant_columns = [col for col in X_train.columns if X_train[col].nunique() == 1]
        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)
        logger.info(f"Removed {len(constant_columns)} constant columns")
    except Exception as e:
        logger.info("No constant features to remove.")

    # Step 3: Handle outliers
    try:
        X_train = handle_outliers(X_train)
        X_test = handle_outliers(X_test)
        logger.info("Outliers handled")
    except Exception as e:
        logger.info("No outliers needed to be handled.")

    is_reg = is_regression(target_options)

    if not is_reg and (
        target_options == "latent_class_depression" or target_options == "depadhd_c" or target_options == "asdadhd_c"
    ):
        y_train = map_classes(y_train)
        y_test = map_classes(y_test)

    # Step 4a: Clean NaN / inf BEFORE fitting the scaler (NaN in fit → NaN-scaled columns).
    na_method_mapping = {
        "fill_with_zero": ("fill_0", None),
        "drop": ("remove", None),
        "impute_mean": ("impute", "mean"),
        "impute_median": ("impute", "median"),
    }
    method, imputer_strategy = na_method_mapping[ctx.deal_with_na_values]
    X_train, y_train = clean_data(X_train, y_train, method=method, imputer_strategy=imputer_strategy)
    X_test, y_test = clean_data(X_test, y_test, method=method, imputer_strategy=imputer_strategy)
    X_train = X_train.loc[:, ~X_train.columns.duplicated(keep="first")]
    X_test = X_test.loc[:, ~X_test.columns.duplicated(keep="first")]

    # Step 4b: Feature scaling, fit exclusively on clean training data.
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    logger.info("Features scaled")

    logger.info(f"Shape after preprocessing: {X_train.shape}")

    # Feature selection (if enabled)
    if ctx.selection_method != "None" and ctx.selection_method is not None:
        if ctx.selection_method == "rfecv":
            # Recursive Feature Elimination with Cross-Validation
            if is_reg:
                estimator = RandomForestRegressor(random_state=42)
                logger.info("Using RandomForestRegressor for feature selection.")
            else:
                estimator = RandomForestClassifier(random_state=42)
                logger.info("Using RandomForestClassifier for feature selection.")
            selector = RFECV(estimator, step=1, cv=5)
            selector = selector.fit(X_train, y_train)  # Use y.ravel to make it to 1d from 2d

            # Transform X_train and X_test using the selector
            X_train_transformed = selector.transform(X_train)
            X_test_transformed = selector.transform(X_test)

            # Convert numpy arrays back to DataFrame and update column names
            selected_features = [X_train.columns[i] for i in selector.get_support(indices=True)]
            X_train = pd.DataFrame(X_train_transformed, columns=selected_features)
            X_test = pd.DataFrame(X_test_transformed, columns=selected_features)
            logger.info(f"Features reduced to {selector.n_features_} using RFECV.")

        elif ctx.selection_method == "interaction":
            # Feature Interaction: Add new features that are interactions of existing features

            if is_reg:
                initial_model = CatBoostRegressor(
                    iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=42
                )
            else:
                initial_model = CatBoostClassifier(
                    iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=42
                )

            logger.info("Performing SHAP interaction-based feature selection...")
            X_train, X_test = select_features_with_shap_interactions(
                X_train, X_test, y_train, split=str(ctx.split), is_reg=is_reg, max_features=1000, top_interactions=50
            )

        elif ctx.selection_method == "h-stats":
            logger.info("Performing H-Stats interaction-based feature selection...")

            X_train, X_test = select_features_with_h_statistics(
                X_train, X_test, y_train, is_reg=is_reg, max_features=1000, top_interactions=50
            )

        elif ctx.selection_method == "clustering":
            # Feature Clustering: Cluster features and select cluster representatives
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_train.T)
            cluster_labels = kmeans.labels_
            selected_features = [X_train.columns[cluster_labels == i][0] for i in range(n_clusters)]
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            logger.info(f"Features reduced to {len(selected_features)} by clustering.")

    num_classes = len(np.unique(y_train)) if not is_reg else 0

    if quantile_segment:
        logger.info("Calculating quantile-based sample weights...")
        # This one is for the dynamic 4 quantile segmentation
        # sample_weights = calculate_quantile_weights(y_train)

        # This one only for top 25%
        sample_weights = calculate_top_quartile_weights(y_train)

    else:
        sample_weights = None

    if hyperparameter_tuning:
        logger.info("Tuning the hyperparameters")

        if ctx.with_optuna:
            logger.info("Tuning hyperparameters with Optuna")

            # unique feature names
            X_train = ensure_unique_feature_names(X_train)
            X_test = ensure_unique_feature_names(X_test)

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, is_reg, num_classes), n_trials=ctx.num_of_optuna_trials
            )

            best_params = study.best_params
            best_params["random_state"] = 42
            logger.info("%s %s", "Best hyperparameters:", best_params)

            if ctx.ensemble_version:
                if ctx.ensemble_type == "bagging":
                    # Create ensemble model (existing bagging implementation)
                    model_configs = {
                        "catboost": best_params,
                        "random_forest": dict(ctx._RF_PARAMS),
                        "xgboost": dict(ctx._XGB_PARAMS),
                        "elasticnet": {"alpha": 1.0, "l1_ratio": 0.5},  # Only for regression
                    }

                    model = create_ensemble_model(  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                        task_type="regression" if is_reg else "classification", model_configs=model_configs
                    )
                else:  # ensemble_type == 'stacking'
                    # Create stacking ensemble model

                    model_configs = {
                        "catboost": best_params,
                        "random_forest": dict(ctx._RF_PARAMS),
                        "xgboost": dict(ctx._XGB_PARAMS),
                        #'svm': {'kernel': 'rbf', 'C': 1.0},
                        "tabpfn": dict(ctx.PIPELINE_MODEL_CONFIGS["tabpfn"]),
                    }
                    if ctx._SKIP_TABPFN:
                        model_configs.pop("tabpfn", None)

                    if ctx.split == "within_categories" and category == "Multimodal Neuroimaging":
                        logger.info("Change to CPU for Neuroimaging.")
                        model_configs["tabpfn"] = {"device": "cpu", "max_features": 50, "version": "v2"}
                        if ctx._SKIP_TABPFN:
                            model_configs.pop("tabpfn", None)

                    # Stacking uses Linear Regression as the meta-learner for regression
                    # and LogisticRegression for classification
                    if is_reg:
                        meta_model = LinearRegression()
                    else:
                        meta_model = LogisticRegression(random_state=42)

                    model = create_stacking_ensemble(
                        task_type="regression" if is_reg else "classification",
                        model_configs=model_configs,
                        meta_model=meta_model,
                        cv=5,
                        random_state=42,
                    )

                    model.explain_meta_model(X_train, y_train)
                    # Expose fitted model to later stages
                    ctx.last_fitted_model = model

            else:
                if is_reg:
                    model = CatBoostRegressor(**best_params)
                else:
                    model = CatBoostClassifier(**best_params)

            # Optuna provides feature importance for hyperparameters
            importance = optuna.importance.get_param_importances(study)

            logger.info("\nHyperparameter importance:")
            for param, score in importance.items():
                logger.info(f"{param}: {score:.4f}")
        else:
            if ctx.ensemble_version:
                if ctx.ensemble_type == "bagging":
                    raise ValueError("Bagging ensemble path removed — use ensemble_type='stacking'")
                else:  # ensemble_type == 'stacking'
                    # Create and tune stacking ensemble
                    if is_reg:
                        meta_model = LinearRegression()
                    else:
                        meta_model = LogisticRegression(random_state=42)

                    model = create_stacking_ensemble(
                        task_type="regression" if is_reg else "classification",
                        meta_model=meta_model,
                        cv=5,
                        random_state=42,
                    )

                # Fit the model (stacking uses .fit; tuned bagging uses .tune_and_fit)
                if hasattr(model, "tune_and_fit"):
                    categorical_features = list(range(len(categorical_columns))) if categorical_columns else None
                    model = model.tune_and_fit(X_train, y_train, categorical_features=categorical_features)
                else:
                    model.fit(X_train, y_train)
                    model.explain_meta_model(X_train, y_train)
                # Expose fitted model to the networks stage
                if ctx.ensemble_type == "stacking":
                    ctx.last_fitted_model = model
            else:
                # Hyperparameter tuning for CatBoost
                if is_reg:
                    param_dist = {
                        "iterations": randint(100, 1000),
                        "learning_rate": uniform(0.01, 0.3),
                        "depth": randint(4, 10),
                        "l2_leaf_reg": uniform(1, 10),
                        "border_count": randint(32, 255),
                    }
                    model = CatBoostRegressor(random_state=42)
                    scoring = "neg_mean_squared_error"
                else:
                    param_dist = {
                        "iterations": randint(100, 1000),
                        "learning_rate": uniform(0.01, 0.3),
                        "depth": randint(4, 10),
                        "l2_leaf_reg": uniform(1, 10),
                        "border_count": randint(32, 255),
                    }
                    model = CatBoostClassifier(random_state=42)
                    scoring = "accuracy"

                random_search = RandomizedSearchCV(
                    model, param_distributions=param_dist, n_iter=100, cv=5, scoring=scoring, random_state=42, n_jobs=-1
                )
                random_search.fit(X_train, y_train)

                logger.info("%s %s", "Best hyperparameters:", random_search.best_params_)
                model = random_search.best_estimator_
    else:
        logger.info("Not tuning the hyperparameters")

        # CatBoost settings from PIPELINE_MODEL_CONFIGS (models.stacking), so model_overrides apply here too
        best_params = dict(ctx._CB_PARAMS)

        # Use optimal parameters
        if ctx.ensemble_version:
            if ctx.ensemble_type == "bagging":
                # Create bagging ensemble model
                model_configs = {
                    "catboost": best_params,
                    "random_forest": dict(ctx._RF_PARAMS),
                    "xgboost": dict(ctx._XGB_PARAMS),
                    "elasticnet": {"alpha": 1.0, "l1_ratio": 0.5},
                }

                model = create_ensemble_model(  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
                    task_type="regression" if is_reg else "classification", model_configs=model_configs
                )
            else:  # ensemble_type == 'stacking'
                # Create stacking ensemble model
                model_configs = {
                    "catboost": best_params,
                    "random_forest": dict(ctx._RF_PARAMS),
                    "xgboost": dict(ctx._XGB_PARAMS),
                    #'svm': {'kernel': 'rbf', 'C': 1.0},
                    "tabpfn": dict(ctx.PIPELINE_MODEL_CONFIGS["tabpfn"]),
                }
                if ctx._SKIP_TABPFN:
                    model_configs.pop("tabpfn", None)

                if ctx.split == "within_categories" and category == "Multimodal Neuroimaging":
                    logger.info("Switch to CPU for neuroimaging.")
                    model_configs["tabpfn"] = {"device": "cpu", "max_features": 50, "version": "v2"}
                    if ctx._SKIP_TABPFN:
                        model_configs.pop("tabpfn", None)

                if is_reg:
                    meta_model = LinearRegression()
                else:
                    meta_model = LogisticRegression(random_state=42)

                model = create_stacking_ensemble(
                    task_type="regression" if is_reg else "classification",
                    model_configs=model_configs,
                    meta_model=meta_model,
                    cv=5,
                    random_state=42,
                )

        else:
            if is_reg:
                model = CatBoostRegressor(**best_params, random_state=42)
            else:
                model = CatBoostClassifier(**best_params, random_state=42)
    if is_reg:
        scoring = "r2"
    else:
        scoring = "roc_auc"

    # Add boolean target processing here
    if target_options == "dep_onset_rci_2.3" or (len(np.unique(y_train)) == 2 and not is_reg):
        y_train = preprocess_boolean_target(y_train)
        y_test = preprocess_boolean_target(y_test)
        model, y_train = modify_ensemble_for_boolean(model, X_train, y_train)

    # Inner-loop CV handled by stacking OOF; standalone CV skipped.
    cv_scores = []

    # Create CatBoost Pool objects
    # Identify categorical columns
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Single authoritative Pool construction.
    try:
        train_pool = Pool(X_train, y_train, cat_features=categorical_columns, weight=sample_weights)
        test_pool = Pool(X_test, y_test, cat_features=categorical_columns)
    except Exception as e:
        logger.info(f"[WARNING] Pool with cat_features failed ({e}); retrying without.")
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]
        train_pool = Pool(X_train, y_train, weight=sample_weights)
        test_pool = Pool(X_test, y_test)

    if ctx.simple_test_model:
        if ctx.model_type == "lasso" or ctx.model_type == "ridge":
            if not is_regression(target_options):
                logger.info(
                    f"Warning: {ctx.model_type} regression is only for regression tasks. Using CatBoost for classification."
                )
            else:
                return (
                    run_linear_model_analysis(
                        ctx.model_type, X_train, X_test, y_train, y_test, target_options, tp, category
                    ),
                    None,
                    None,
                )

    # Only for classification
    if not is_reg and not ctx.ensemble_version:
        from sklearn.utils.class_weight import compute_class_weight

        # Calculate class weights for this category's data
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train.values.ravel())
        class_weights_dict = dict(zip(np.unique(y_train), class_weights))
        model.set_params(class_weights=class_weights_dict)

    # Fit the model
    if ctx.ensemble_version:
        # Ensemble path: fit on raw arrays rather than CatBoost Pool objects
        # Check if we need to reduce features for TabPFN
        if X_train.shape[1] >= 500 and "model_configs" in dir() and "tabpfn" in model_configs:
            logger.info("\nFeature count exceeds TabPFN limit. Selecting top features to ensure it's under 500...")
            # Use CatBoost to select features
            if is_reg:
                feature_selector = CatBoostRegressor(iterations=100, learning_rate=0.1, verbose=False, random_state=42)
            else:
                feature_selector = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=False, random_state=42)

            # Fit CatBoost and get feature importance
            try:
                feature_selector.fit(X_train, y_train)
            except Exception as e:
                logger.info("For TabPFN feature selection, y_train needs to not be multidimentional")
                feature_selector.fit(X_train, y_train.iloc[:, 0])

            importance = pd.Series(feature_selector.feature_importances_, index=X_train.columns)
            selected_features = importance.nlargest(490).index

            # Reduce features for all models
            X_train = X_train[selected_features]
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test[selected_features]

            logger.info(f"Reduced feature set from {X_train.shape[1]} to {len(selected_features)} features")

            categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

            # Create Pool objects with potentially reduced features
            if categorical_columns:
                train_pool = Pool(X_train, y_train, cat_features=categorical_columns, weight=sample_weights)
                test_pool = Pool(X_test, y_test, cat_features=categorical_columns)
            else:
                try:
                    train_pool = Pool(X_train, y_train, weight=sample_weights)
                    test_pool = Pool(X_test, y_test)
                except Exception as e:
                    logger.info(f"[WARNING] {e}")
                    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
                    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
                    train_pool = Pool(X_train, y_train, weight=sample_weights)
                    test_pool = Pool(X_test, y_test)

        # Fit the model on the (possibly reduced) feature set
        if categorical_columns:
            model.fit(X_train, y_train, categorical_features=categorical_columns)
        else:
            model.fit(X_train, y_train)
        # Expose fitted model to the networks stage
        if ctx.ensemble_type == "stacking":
            ctx.last_fitted_model = model
    else:
        # CatBoost fitting with Pool objects
        model.fit(train_pool, eval_set=test_pool, use_best_model=True, early_stopping_rounds=50)

    if ctx._run_consensus:
        # Local vars: will be returned to caller for global assignment
        _local_consensus_df = None
        _local_model_weights = None
        try:
            consensus_df, model_weights, feature_comparison = model.analyze_feature_consensus(
                X_train, y_train, target_options, tp, category, top_n=ctx.ENSEMBLE_VIZ_CONFIG["top_n"]
            )
            _local_consensus_df = consensus_df
            _local_model_weights = model_weights

            # Access specific results if needed
            logger.info("\nTop features with their consensus scores:")
            logger.info(consensus_df.sort_values("weighted_score", ascending=False))
        except Exception as e:
            logger.info("%s %s", "Failed to do consensus_df, model_weights, and feature comparison, because of: ", e)
            logger.info("Only works for Ensemble: Stacking - Consensus Analysis")
    else:
        _local_consensus_df = None
        _local_model_weights = None
        logger.info("    Consensus analysis skipped (visualization_mode does not include consensus).")

    if sample_weights is not None and len(sample_weights) > 0:
        logger.info(f"Sample weights included: n={len(sample_weights)}, unique={len(set(sample_weights))}")

    # Convert y_test to numpy array if it's a DataFrame or Series
    y_test_np = y_test.values if isinstance(y_test, (pd.DataFrame, pd.Series)) else np.array(y_test)
    y_test_np = y_test_np.ravel()  # it's 1-dimensional

    # Visualize selection_method interaction with SHAP

    # After model fitting, add conformal prediction if enabled
    if conformal_prediction:
        logger.info("\nPerforming Conformal Prediction Analysis...")

        # Split validation set for conformal prediction
        X_train_conf, X_val_conf, y_train_conf, y_val_conf = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        if is_reg:
            # Regression conformal prediction
            mapie = SplitConformalRegressor(estimator=model, random_state=42)

            # Fit MAPIE
            mapie.fit(X_train_conf, y_train_conf)

            # Get prediction intervals
            y_pred_conf, y_pis = mapie.predict(X_test, alpha=[0.05, 0.32])

            # Visualize prediction intervals
            plt.figure(figsize=(15, 8))
            plt.fill_between(
                range(len(y_test)),
                y_pis[:, 0, 0],  # Lower bound for alpha=0.05
                y_pis[:, 1, 0],  # Upper bound for alpha=0.05
                alpha=0.3,
                label="95% Prediction Interval",
            )
            plt.fill_between(
                range(len(y_test)),
                y_pis[:, 0, 1],  # Lower bound for alpha=0.32
                y_pis[:, 1, 1],  # Upper bound for alpha=0.32
                alpha=0.3,
                label="68% Prediction Interval",
            )
            plt.plot(range(len(y_test)), y_test, "ko", label="Actual", markersize=4)
            plt.plot(range(len(y_test)), y_pred_conf, "r-", label="Predicted")
            if ctx.split == "within_categories":
                plt.title(f"Conformal Prediction Intervals\n({target_options}, Timepoint {tp}, for {category})")
            else:
                plt.title(f"Conformal Prediction Intervals\n({target_options}, Timepoint {tp})")

            plt.xlabel("Sample Index")
            plt.ylabel("Target Value")
            plt.legend()
            plt.tight_layout()
            plt.show()

        else:
            # Classification conformal prediction
            mapie = SplitConformalClassifier(estimator=model, random_state=42)

            # Fit MAPIE
            mapie.fit(X_train_conf, y_train_conf)

            # Get prediction sets for different alpha values
            alpha_values = [0.05, 0.32, 0.5]  # 0.5 included to approximate predicted probabilities
            y_pred_conf, y_ps = mapie.predict(X_test, alpha=alpha_values)

            # Determine if it's binary or multiclass
            n_classes = len(np.unique(y_test))
            is_binary = n_classes == 2

            # Calculate prediction set sizes
            set_sizes = np.sum(y_ps, axis=2)

            # Calculate coverage using numpy arrays
            coverage_95 = np.mean([y_test_np[i] in np.where(y_ps[i, :, 0])[0] for i in range(len(y_test_np))])
            coverage_68 = np.mean([y_test_np[i] in np.where(y_ps[i, :, 1])[0] for i in range(len(y_test_np))])

            # Approximate class probabilities
            if is_binary:
                class_probs = 1 - y_ps[:, 0, :]  # Probability of class 1
                class_probs = np.column_stack((1 - class_probs, class_probs))  # Add probability of class 0
            else:
                class_probs = y_ps[:, :, -1]  # Use the last alpha value for multiclass

            # 1. Class Probability Distributions
            plt.figure(figsize=(15, 8))
            for i in range(class_probs.shape[1]):
                sns.kdeplot(class_probs[:, i], label=f"Class {i}")

            if ctx.split == "within_categories":
                plt.title(f"Class Probability Distributions\n({target_options}, Timepoint {tp}, for {category})")
            else:
                plt.title(f"Class Probability Distributions\n({target_options}, Timepoint {tp})")
            plt.xlabel("Probability")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 2. Prediction Set Sizes Distribution

            try:
                plt.figure(figsize=(15, 8))
                for i, alpha in enumerate(alpha_values):
                    sns.histplot(set_sizes[:, i], kde=True, label=f"α = {alpha}", stat="density", alpha=0.6)
            except Exception as e:
                logger.info("%s %s", "Full Alpha Distribution of Prediction Sets not available for ", target_options)

            if ctx.split == "within_categories":
                plt.title(f"Distribution of Prediction Set Sizes\n({target_options}, Timepoint {tp}, for {category})")
            else:
                plt.title(f"Distribution of Prediction Set Sizes\n({target_options}, Timepoint {tp})")
            plt.xlabel("Number of Classes in Prediction Set")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 3. Coverage vs Alpha
            coverages = [
                np.mean([y_test_np[i] in np.where(y_ps[i, :, j])[0] for i in range(len(y_test_np))])
                for j in range(len(alpha_values))
            ]
            plt.figure(figsize=(10, 6))
            plt.plot(alpha_values, coverages, "bo-")
            plt.plot(alpha_values, alpha_values, "r--", label="Ideal coverage")

            if ctx.split == "within_categories":
                plt.title(f"Coverage vs. Alpha\n({target_options}, Timepoint {tp}, for {category})")
            else:
                plt.title(f"Coverage vs. Alpha\n({target_options}, Timepoint {tp})")
            plt.xlabel("Alpha")
            plt.ylabel("Empirical coverage")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 4. Confusion Matrix (for binary classification)
            if is_binary:
                y_pred = (class_probs[:, 1] > 0.5).astype(int)
                cm = confusion_matrix(y_test_np, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix\n({target_options}, Timepoint {tp})")
                plt.xlabel("Predicted Class")
                plt.ylabel("True Class")
                plt.tight_layout()
                plt.show()

            # 5. ROC Curve (for binary classification)
            try:
                if is_binary:
                    fpr, tpr, _ = roc_curve(y_test_np, class_probs[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(10, 8))
                    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
                    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"Receiver Operating Characteristic (ROC) Curve\n({target_options}, Timepoint {tp})")
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                logger.info("%s %s", "Cannot convert AUC variables because len y unique != 2: ", e)

            logger.info(f"Coverage at 95% confidence: {coverage_95:.4f}")
            logger.info(f"Coverage at 68% confidence: {coverage_68:.4f}")

    # Get feature importance
    if ctx.ensemble_version:
        # Get and plot feature importance from all models
        if not ctx.run_visualizations:
            logger.info("    run_visualizations=False -- all FI/SHAP/PDP plots skipped.")
        elif ctx.show_feature_importance_plots:
            _fi_models = {k: v for k, v in model.base_models.items() if k in ctx._VIZ_ENABLED_MODELS}
            if _fi_models:
                _orig_base = model.base_models
                model.base_models = _fi_models
                logger.info(f"  FI plots for: {sorted(_fi_models.keys())}")
                try:
                    # Apply DISPLAY_NAMES to ensemble FI
                    if "DISPLAY_NAMES" in ctx and DISPLAY_NAMES:
                        model._display_names = DISPLAY_NAMES  # Pass to plotting
                    model.plot_feature_importance(X_train, y_train, target_options, tp, category)
                finally:
                    model.base_models = _orig_base
            else:
                logger.info("    No models enabled for FI plots.")
        else:
            logger.info("    show_feature_importance_plots=False -- FI plots suppressed.")
            logger.info("     SHAP will be shown for the best enabled model only.")
    else:
        feature_importance = model.get_feature_importance()
        feature_names = X_train.columns
        feature_importance = pd.Series(feature_importance, index=feature_names).sort_values(ascending=False)

        # Apply display names if available
        if "DISPLAY_NAMES" in dir() and isinstance(DISPLAY_NAMES, dict) and len(DISPLAY_NAMES) > 0:
            feature_importance.index = feature_importance.index.map(lambda x: DISPLAY_NAMES.get(x, x))

        # Plot top 20 OR 30 feature importances
        if ctx.split == "across_categories":
            top_20_features = feature_importance.head(30).sort_values(ascending=True)
            top_str = 30
        else:
            top_20_features = feature_importance.head(20).sort_values(ascending=True)
            top_str = 20

        plt.figure(figsize=(8, 8))
        sns.barplot(x=top_20_features.values, y=top_20_features.index, palette="coolwarm")
        title = f"Top {top_str} CatBoost Important Features for {'Regression' if is_reg else 'Classification'} ({target_options}, Timepoint {tp}"
        title += f", Category {category})" if category else ")"
        plt.title(title, fontsize=16)
        plt.xlabel("Feature Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        for i, v in enumerate(top_20_features.values):
            plt.text(v, i, f" {v:.4f}", va="center", fontsize=10)
        plt.tight_layout()
        plt.show()

        # Partial Dependence Plots
        top_10_features = feature_importance.head(10).index
        if is_reg and ctx._run_pdp_gam:
            fig, ax = plt.subplots(2, 5, figsize=(24, 12))
            for i, feature in enumerate(top_10_features):
                try:
                    PartialDependenceDisplay.from_estimator(model, X_train, [feature], ax=ax[i // 5, i % 5])
                except Exception as e:
                    logger.info("Percentiles too close together, skipping this one as it is un-map-able.")
            plt.suptitle(
                f"Partial Dependence Plots for Top 10 Features ({target_options}, Timepoint {tp})", fontsize=20
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            classes = model.classes_
            for target_class in classes:
                fig, ax = plt.subplots(2, 5, figsize=(24, 12))
                for i, feature in enumerate(top_10_features):
                    try:
                        PartialDependenceDisplay.from_estimator(
                            model, X_train, [feature], target=target_class, ax=ax[i // 5, i % 5]
                        )
                    except ValueError:
                        logger.info("Percentiles too close together, skipping this one as it is un-map-able.")
                plt.suptitle(
                    f"Partial Dependence Plots for Top 10 Features (Class: {target_class}, {target_options}, Timepoint {tp})",
                    fontsize=20,
                )
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

    # Calculate test predictions and metrics

    y_pred = model.predict(X_test)

    if ctx.ensemble_version:
        # Get metrics for all models and ensemble
        if not ctx.run_visualizations:
            import matplotlib

            _was_interactive = matplotlib.is_interactive()
            matplotlib.interactive(False)
        metrics_dict = model.evaluate_models(X_test, y_test, target_options, tp, category)
        if not ctx.run_visualizations:
            plt.close("all")
            matplotlib.interactive(_was_interactive)
            logger.info("    run_visualizations=False -- model comparison plots suppressed.")
        ctx.metrics_dict_full["metrics"].append(metrics_dict)

        try:
            ctx.metrics_dict_full["category"].append(category)
        except Exception as e:
            logger.info(f"[WARNING] {e}")
            ctx.metrics_dict_full["category"].append("All")

        # Overall evaluation uses the stacking ensemble, stored by evaluate_models as "meta_model"
        if is_reg:
            evaluation_metrics = pd.DataFrame(
                {
                    "Metric": ["MSE", "RMSE", "MAE", "R² Score"],
                    "Value": [
                        metrics_dict["meta_model"]["mse"],
                        metrics_dict["meta_model"]["rmse"],
                        metrics_dict["meta_model"]["mae"],
                        metrics_dict["meta_model"]["r2"],
                    ],
                }
            )
            score = metrics_dict["meta_model"]["r2"]
        else:
            metrics = metrics_dict["meta_model"]
            evaluation_metrics = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"]
                    + (["AUC-ROC"] if "auc" in metrics else []),
                    "Value": [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
                    + ([metrics["auc"]] if "auc" in metrics else []),
                }
            )
            score = metrics["auc"] if "auc" in metrics else metrics["accuracy"]

    else:
        if is_reg:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            score = r2

            evaluation_metrics = pd.DataFrame(
                {"Metric": ["MSE", "RMSE", "MAE", "R² Score"], "Value": [mse, rmse, mae, r2]}
            )
        else:
            # Get predictions for just this category's test data
            y_pred_category = model.predict(X_test)
            proba_category = model.predict_proba(test_pool)

            # Calculate metrics using only this category's predictions
            accuracy = accuracy_score(y_test, y_pred_category)
            precision = precision_score(y_test, y_pred_category, average="weighted")
            recall = recall_score(y_test, y_pred_category, average="weighted")
            f1 = f1_score(y_test, y_pred_category, average="weighted")

            # Store metrics for this category in a dictionary
            category_metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, model.predict_proba(test_pool)[:, 1])
                category_metrics["auc"] = auc
                evaluation_metrics = pd.DataFrame(
                    {
                        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
                        "Value": [accuracy, precision, recall, f1, auc],
                    }
                )
                score = auc
            else:
                try:
                    auc = roc_auc_score(y_test, proba_category, multi_class="ovr", average="macro")
                    score = auc
                    category_metrics["auc"] = auc
                except Exception as e:
                    logger.info(f"Cannot define AUC for category {category} - {len(np.unique(y_test))} unique classes")
                    score = accuracy  # Fallback to accuracy as the score

                evaluation_metrics = pd.DataFrame(
                    {
                        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                        "Value": [accuracy, precision, recall, f1],
                    }
                )

            # Store metrics in global dictionary for tracking across categories
            if "category_metrics" not in ctx:
                ctx.category_metrics_dict = {}

            ctx.category_metrics_dict[category] = category_metrics

            logger.info(f"\nClassification Report for category {category}:")
            logger.info(classification_report(y_test, y_pred))

            # Confusion Matrix with category label
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {category}")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.show()

            # Compare metrics across categories if we have multiple categories
            if category and len(ctx.category_metrics_dict) > 1:
                compare_metrics_across_categories(ctx.category_metrics_dict)

        if target_options != "latent_class_depression":
            ci = stats.sem(cv_scores) * stats.t.ppf((1 + 0.95) / 2.0, len(cv_scores) - 1)
        # CI assumes approximate normality of fold scores; may be less
        # precise for AUC with few folds but is standard practice.

        # The spider-plot score variable may be undefined for multiclass
        # targets where per-fold CV scoring is not applicable.
        try:
            ctx.category_results[category]["score"] = score
            ctx.category_results[category]["score_lower"] = score - ci
            ctx.category_results[category]["score_upper"] = score + ci
        except Exception as e:
            logger.info(f"Warning: {e}")

        # Print metrics as a table
        logger.info("\nEvaluation Metrics:")
        display(evaluation_metrics)

        display(
            evaluation_metrics.style.format({"Value": "{:.4f}"})
            .background_gradient(cmap="coolwarm", subset=["Value"])
            .set_caption(f"Evaluation Metrics -- {target_options} (Timepoint {tp})")
            .set_table_styles([{"selector": "caption", "props": [("font-size", "14px"), ("font-weight", "bold")]}])
        )

        # After calculating metrics for spider plot (regression only)
        if is_reg:
            ctx.category_results[category]["r2"] = r2

    # (Calibration is handled separately in the nested_cv stage.)

    # GAM Analysis: uses CatBoost feature_importance (non-ensemble path only).
    _fi_available = isinstance(ctx.get("feature_importance", None), pd.Series)
    if not ctx._run_pdp_gam:
        logger.info("    GAM/PDP analysis skipped (visualization_mode does not include PDP/GAM).")
    else:
        try:
            if not _fi_available:
                raise RuntimeError("feature_importance not available in ensemble mode; skipping GAM.")
            top_10_features = feature_importance.head(10).index
            X_train_top10 = X_train[top_10_features]
            X_test_top10 = X_test[top_10_features]

            # Create GAM formula using feature names
            gam_formula = " + ".join([f"s({i}, label='{name}')" for i, name in enumerate(top_10_features)])

            # Create GAM formula with just the indices
            if is_reg:
                gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9))
            else:
                gam = GAM(
                    s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9),
                    distribution="binomial",
                    link="logit",
                )

            gam.fit(X_train_top10, y_train)

            # Assign term names after fitting
            feature_mapping = {i: name for i, name in enumerate(top_10_features)}
            gam.terms.feature_names = feature_mapping

            # GAM Visualization
            fig, axs = plt.subplots(2, 5, figsize=(25, 12))
            fig.suptitle(
                f"GAM Partial Dependence Plots for Top 10 Features\n({target_options}, Timepoint {tp}{', ' + category if category else ''})",
                fontsize=20,
                y=1.02,
            )

            for i, (term, feature) in enumerate(zip(gam.terms, top_10_features)):
                if term.isintercept:
                    continue

                XX = gam.generate_X_grid(term=i)
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

                ax = axs[i // 5, i % 5]
                ax.plot(XX[:, term.feature], pdep, color="#ff7f0e", label="Partial Effect")
                ax.fill_between(XX[:, term.feature], confi[:, 0], confi[:, 1], alpha=0.2, color="#ff7f0e")

                feature_values = X_train_top10.iloc[:, i]
                partial_dep = gam.partial_dependence(term=i, X=X_train_top10.values)
                ax.scatter(feature_values, partial_dep, alpha=0.1, color="#1f77b4", label="Data")

                ax.set_title(feature, fontsize=12)
                ax.set_xlabel("Value", fontsize=10)
                ax.set_ylabel("Partial Effect", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.tick_params(axis="both", which="major", labelsize=8)

                importance = feature_importance[feature]
                ax.text(
                    0.05,
                    0.95,
                    f"CatBoost Importance: {importance:.4f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

                if i == 0:
                    ax.legend(fontsize=8, loc="lower right")

            plt.tight_layout()
            plt.show()

            # Print GAM summary with feature names(? How)

            logger.info("\nGAM Summary:")
            logger.info(gam.summary())

            # Print feature mapping

            logger.info("\nFeature Mapping:")
            logger.info("-" * 50)
            for i, feature in enumerate(top_10_features):
                logger.info(f"s({i}) = {feature}")

            # GAM Performance
            y_pred_gam = gam.predict(X_test_top10)
            if is_reg:
                mse_gam = mean_squared_error(y_test, y_pred_gam)
                r2_gam = r2_score(y_test, y_pred_gam)
                logger.info(f"\nGAM Performance:\nMSE: {mse_gam:.4f}\nR2: {r2_gam:.4f}")
            else:
                accuracy_gam = accuracy_score(y_test, (y_pred_gam > 0.5).astype(int))
                logger.info(f"\nGAM Performance:\nAccuracy: {accuracy_gam:.4f}")
                logger.info("\nGAM Classification Report:")
                logger.info(classification_report(y_test, (y_pred_gam > 0.5).astype(int)))

        except Exception as e:
            logger.info(f"Error in GAM analysis: {str(e)}")
            logger.info("Not enough dimensionality to do a GAM FI.")

    # SHAP Values Visualization
    if ctx.ensemble_version:
        # Per-model gating: only run SHAP for models enabled in _VIZ_ENABLED_MODELS
        _shap_models = {k: v for k, v in model.base_models.items() if k in ctx._VIZ_ENABLED_MODELS}
        if not ctx.run_visualizations or not _shap_models:
            logger.info("    SHAP skipped (run_visualizations=False or no models enabled).")
        elif not ctx.show_feature_importance_plots and "metrics_dict" in dir():
            # Pick the best enabled non-TabPFN model for a single SHAP run
            _metric_k = "r2" if is_reg else "auc"
            _best_shap_mn = None
            _best_shap_score = -np.inf
            _skip_models = {"tabpfn", "ensemble", "meta_model"}
            for _mn, _mvals in metrics_dict.items():
                if _mn in _skip_models or _mn not in _shap_models:
                    continue
                _sc = _mvals.get(_metric_k, _mvals.get("accuracy", np.nan))
                if not (isinstance(_sc, float) and np.isnan(_sc)) and _sc > _best_shap_score:
                    _best_shap_score = _sc
                    _best_shap_mn = _mn
            if _best_shap_mn and _best_shap_mn in model.base_models:
                logger.info(
                    f"\n  SHAP: best enabled model → {_best_shap_mn}  ({_metric_k.upper()} = {_best_shap_score:.4f})"
                )
                _orig_base = model.base_models
                model.base_models = {_best_shap_mn: _orig_base[_best_shap_mn]}
                try:
                    shap_values_dict = model.explain_with_shap(X_train, X_test)
                finally:
                    model.base_models = _orig_base
            else:
                logger.info("  Warning:  No enabled model found for SHAP -- skipping.")
        else:
            # Run SHAP only for enabled models
            _orig_base = model.base_models
            model.base_models = _shap_models
            logger.info(f"  SHAP running for: {sorted(_shap_models.keys())}")
            try:
                shap_values_dict = model.explain_with_shap(X_train, X_test)
            finally:
                model.base_models = _orig_base
    elif not ctx.run_visualizations:
        logger.info("    SHAP skipped (run_visualizations=False).")
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_values = explainer.shap_values(X_test)

        if shap_values.ndim > 2:
            # individual x features x class

            # 2. SHAP values come as a list of arrays, one per class
            # For multiclass problems, shap_values is a list of arrays (one array per class)
            logger.info(shap_values.shape)
            shap_values_abs = np.apply_along_axis(np.abs, 0, shap_values)

            # 3. Calculate mean SHAP values for each feature per class
            shap_values_mean = np.mean(shap_values_abs, axis=0)  # Average over instances for each class

            sorted_shap_values_mean_sum_idxs = np.argsort(np.sum(shap_values_mean, axis=1))

            idxs = sorted_shap_values_mean_sum_idxs[-10:]

            shap_values_mean = shap_values_mean[idxs, :]

            # 4. Prepare data for stacked bar plot
            n_classes = shap_values_mean.shape[1]
            n_features = 10
            class_names = inverse_map_classes(pd.Series(model.classes_))  # Replace with actual class names if available
            feature_names = X_test.columns[idxs]  # Feature names from your dataset

            # 5. Create the stacked bar plot
            plt.figure(figsize=(12, 8))
            colors = plt.get_cmap("tab10").colors[:n_classes]  # Colors for each class

            # Initialize bottom for stacking
            bottom = np.zeros(n_features)

            # Plot each class's contribution as a stacked bar
            for i, class_name in enumerate(class_names):
                logger.info(feature_names.shape)
                logger.info(shap_values_mean.shape)
                try:
                    plt.barh(feature_names, shap_values_mean[:, i], left=bottom, color=colors[i], label=class_name)
                except Exception as e:
                    logger.info("Shape mismatch, going to the shap values mean for bottom variable name..")
                    bottom = shap_values_mean.shape[1]
                    plt.barh(feature_names, shap_values_mean[:, i], left=bottom, color=colors[i], label=class_name)

                bottom += shap_values_mean[:, i]  # Update bottom for stacking

            # 6. Add labels and title
            plt.xlabel("Features", fontsize=14)
            plt.ylabel("Mean SHAP Value (Absolute)", fontsize=14)
            plt.title("Stacked Global SHAP Feature Importance Across Classes", fontsize=16)
            plt.xticks(rotation=90)
            plt.legend(title="Classes", loc="upper right")
            plt.tight_layout()

            # 7. Display the plot
            plt.show()

            # For multi-class problems
            for i in range(0, 4):
                shap.summary_plot(shap_values[:, :, i], X_test, plot_type="bar", show=False)
                plt.title(f"SHAP Feature Importance Class {inverse_map_classes(int(model.classes_[i]))}", fontsize=20)
                plt.tight_layout()
                plt.show()

        if (
            target_options == "latent_class_depression"
            or target_options == "depadhd_c"
            or target_options == "asdadhd_c"
            or isinstance(shap_values, list)
        ):
            shap_values_raw = explainer.shap_values(X_test)
            # For multiclass problems
            n_classes = len(shap_values_raw)
            class_names = inverse_map_classes(pd.Series(range(n_classes))).tolist()

            # Global feature importance bar plot
            plt.figure(figsize=(12, 8))
            shap_values_all = np.abs(np.array(shap_values_raw)).mean(0)
            shap_values_exp = shap.Explanation(values=shap_values_all, data=X_test.values, feature_names=X_test.columns)
            shap.plots.bar(shap_values_exp, show=False)
            plt.title("Global Feature Importance (All Classes)", fontsize=16)
            plt.tight_layout()
            plt.show()

            # Bar plots for each class
            for i, class_name in enumerate(class_names):
                if i >= 4:
                    break
                else:
                    plt.figure(figsize=(12, 8))
                    shap_values_exp = shap.Explanation(
                        values=shap_values_raw[i], data=X_test.values, feature_names=X_test.columns
                    )
                    shap.plots.bar(shap_values_exp, show=False)
                    plt.title(f"Feature Importance for Class: {class_name}", fontsize=16)
                    plt.tight_layout()
                    plt.show()

                    # Individual SHAP value plot for the first instance of each class
                    if len(shap_values_exp) > 0:
                        try:
                            plt.figure(figsize=(12, 8))
                            shap.plots.bar(shap_values_exp[0], show=False)
                            plt.title(f"Individual SHAP Values for Class: {class_name} (First Instance)", fontsize=16)
                            plt.tight_layout()
                            plt.show()
                        except Exception as e:
                            logger.info("%s %s", "SHAP Values Error:", e)

        else:
            try:
                # For regression or binary classification
                plt.figure(figsize=(20, 16))
                shap.summary_plot(shap_values, X_test, plot_type="violin", show=False, max_display=20)
                plt.title("SHAP Feature Importance and Impact", fontsize=24)
                plt.xlabel("SHAP value (impact on model output)", fontsize=18)
                plt.ylabel("Features", fontsize=18)
                plt.tick_params(axis="both", which="major", labelsize=14)

                # Adjust colorbar
                cbar = plt.gcf().axes[-1]
                cbar.set_ylabel("Feature value", fontsize=18, labelpad=20)
                cbar.tick_params(labelsize=14)

                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.info("Not enough data to do SHAP feature importance on this class/group prediction.")

    # Compute metrics based on task type
    try:
        if not is_reg:
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = {
                "Category": category,
                "Time Point": f"T{tp}",
                "Task": "Classification",
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1-Score": f1_score(y_test, y_pred, average="weighted"),
                "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            }
        else:
            metrics = {
                "Category": category,
                "Time Point": f"{tp}",
                "Task": "Regression",
                "MSE": mean_squared_error(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }

        # Append per-category metrics to the running summary list.
        if ctx.split == "within_categories":
            ctx.results_summary.append(metrics)
            logger.info(f"Results summary appended with {metrics}")
    except Exception as e:
        logger.info("Not showing summary_metrics table in computing metrics for classifier ensembles.")

    run_shapiq = False
    if ctx.ensemble_version and ctx.ensemble_type == "stacking" and run_shapiq:
        # Run ShapIQ analysis

        logger.info("\nRunning advanced ShapIQ interaction analysis...")

        shapiq_results = integrate_shapiq_analysis(  # noqa: F821 (inherited from v1.0.0, see docs/known_issues.md)
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model=model,
            target_options=target_options,
            tp=tp,
            category=category,
        )

    return model, _local_consensus_df, _local_model_weights


def run() -> None:
    """Run this stage (notebook cell 6)."""
    global copy, traceback
    ctx.group = _SUBGROUP_MAP.get(ctx.subgroup, None)
    # Resolve feature timepoint: None = match target (default), int = cross-tp mode
    # When tp_option="All", _feat_tp must be None (match target) so each
    # TP iteration uses same-wave features. Explicit cross-tp feature
    # selection is incompatible with the All-TP sweep.
    if ctx.tp_option == "All" and ctx.feature_tp_option != "Match target":
        logger.info("Warning:  feature_tp_option is ignored when tp_option='All' (features always match target wave).")
    _feat_tp = (
        None if (ctx.feature_tp_option == "Match target" or ctx.tp_option == "All") else int(ctx.feature_tp_option)
    )
    _tp_all_mode = ctx.tp_option == "All"  # flag used by dispatch logic below
    ctx.run_all = ctx.run_all_grid
    ctx.isolate = ctx.isolate_combinations

    # Derive internal flags from dropdown
    ctx.run_visualizations = ctx.visualization_mode != "No Visuals"
    ctx.show_feature_importance_plots = ctx.visualization_mode == "All Visuals"
    ctx._run_consensus = ctx.visualization_mode in ("All Visuals", "Ensemble Consensus + Best SHAP")
    ctx._run_pdp_gam = ctx.visualization_mode == "All Visuals"

    ctx._VIZ_ENABLED_MODELS = set()
    if ctx.visualization_mode == "All Visuals":
        ctx._VIZ_ENABLED_MODELS = {"catboost", "xgboost", "random_forest", "tabpfn", "meta_model"}
    elif ctx.visualization_mode == "Ensemble Consensus + Best SHAP":
        # Include base models so the best-model SHAP selector has candidates
        ctx._VIZ_ENABLED_MODELS = {"catboost", "xgboost", "random_forest", "meta_model"}
    logger.info(f"[VIZ] {ctx.visualization_mode}: {_mode_summary[ctx.visualization_mode]}")
    if ctx._VIZ_ENABLED_MODELS:
        logger.info(f"      SHAP/FI candidates: {sorted(ctx._VIZ_ENABLED_MODELS)}")

    if not ctx.run_validation_cells:
        logger.info(
            "[VAL] Validation stages nested_cv, stratified, grouped_cv, permutation and stacking_audit will be skipped."
        )
    else:
        logger.info(
            "[VAL] Validation stages nested_cv, stratified, grouped_cv, permutation and stacking_audit enabled."
        )

    # Apply skip_tabpfn: remove TabPFN from any model_configs built below
    # TabPFN is auto-excluded for across-category analyses (d≈500-600,
    # N×d ≈ 4.5M far exceeds TabPFN's d≤500 / N×d≤1M sweet spot).
    # Within-category (d≈20-150) remains in TabPFN's optimal range.
    ctx._SKIP_TABPFN = ctx.skip_tabpfn or (ctx.split == "across_categories") or not TABPFN_AVAILABLE
    if ctx._SKIP_TABPFN:
        _tabpfn_reason = []
        if ctx.skip_tabpfn:
            _tabpfn_reason.append("skip_tabpfn=True")
        if ctx.split == "across_categories":
            _tabpfn_reason.append(f"split={ctx.split!r} (d too high for TabPFN)")
        if not TABPFN_AVAILABLE:
            _tabpfn_reason.append("tabpfn not installed")
        logger.info(f"[MODEL] TabPFN excluded from ensemble ({'; '.join(_tabpfn_reason)}).")

    # Cache system: Main Analysis Runner
    # On cache hit: restores all results instantly, skips model training.
    _KEY = make_run_key()
    _cached = cache_load("main_runner", _KEY)
    if _cached:
        ctx.results_summary = _cached["results_summary"]
        ctx.metrics_dict_full = _cached["metrics_dict_full"]
        ctx.X_train = _cached["X_train"]
        ctx.X_test = _cached["X_test"]
        ctx.y_train = _cached["y_train"]
        ctx.y_test = _cached["y_test"]
        ctx.exported_within_dict = _cached.get("exported_within_dict", {})
        ctx.sample_valid = _cached.get("sample_valid")
        ctx.sample_valid_train = _cached.get("sample_valid_train")
        ctx.sample_valid_test = _cached.get("sample_valid_test")
        ctx._batch_archive = _cached.get("_batch_archive", {})  # restore batch archive from cache
        ctx.batch_mode = _cached.get("_batch_mode", "Single Target")
        report_load_state(_KEY)
        logger.info("  Main runner loaded from cache -- skipping model training.")
        _run_main = False
    else:
        report_init(_KEY)
        _RS["section"] = "main_runner"
        _run_main = True

    if _run_main:
        # _batch_archive stores per-target snapshots so later stages can
        # display cross-target summaries. Cleared each run.
        ctx._batch_archive = {}  # {target_name: {results_summary, metrics_dict_full}}

        # Helper: create sample_valid for current target
        def _create_sample_valid_for_target():
            """Reconstruct sample_valid aligned with current X_train/X_test."""
            import numpy as _np
            import pandas as _pd

            try:
                _tp = int(ctx.tp_option)
                _grp_val = ctx.group if "group" in dir() else None
                _filt = ctx.sample[ctx.sample["time"] == _tp].copy()
                if _grp_val and _grp_val != "None":
                    if _grp_val == "Female":
                        _filt = _filt[_filt["sex"] == 2]
                    elif _grp_val == "Male":
                        _filt = _filt[_filt["sex"] == 1]
                    elif _grp_val == "high_ale":
                        _filt = _filt[_filt["high_ale"] == True]
                    elif _grp_val == "high_ale_severe_p_mh":
                        _filt = _filt[_filt["high_ale_severe_p_mh"] == True]
                    elif _grp_val == "low_ale":
                        _filt = _filt[_filt["low_ale_children_p"] == True]
                if ctx.target_options in _filt.columns:
                    _tmask = _filt[ctx.target_options].notna()
                    if _filt[ctx.target_options].dtype == object:
                        _tmask = _tmask & (_filt[ctx.target_options].astype(str).str.strip() != "")
                    _filt = _filt[_tmask]
                _filt = _filt.reset_index(drop=True)
                _expected = len(ctx.X_train) + len(ctx.X_test)
                if len(_filt) != _expected:
                    return None, None, None
                _sv_train = _filt.iloc[ctx.X_train.index.values].reset_index(drop=True)
                _sv_test = _filt.iloc[ctx.X_test.index.values].reset_index(drop=True)
                _sv_full = _pd.concat([_sv_train, _sv_test], ignore_index=True)
                return _sv_full, _sv_train, _sv_test
            except Exception:
                return None, None, None

        if ctx.run_all:
            for ctx.tp_option in [0, 1, 2, 3, 4]:
                for ctx.group in ["high_ale", "high_ale_severe_p_mh", "Female", "Male", "low_ale", None]:
                    for ctx.target_options in [
                        "depress_D_p_rev",
                        "depress_D_p",
                        "tb_fluid",
                        "latent_class_depression",
                        "dep_onset_rci_1.96",
                        "dep_onset_rci_2.3",
                        "dep_remission_rci_1.96",
                        "dep_remission_rci_2.3",
                        "dep_increase_2sd",
                        "dep_increase_1.5sd",
                        "dep_decrease_2sd",
                        "dep_decrease_1.5sd",
                        "top_10_depression",
                        "top_5_depression",
                    ]:
                        for ctx.split in ["across_categories", "within_categories"]:
                            logger.info(
                                f"  [grid] tp={ctx.tp_option} group={ctx.group} target={ctx.target_options} split={ctx.split}"
                            )
                            run_analysis(
                                ctx.tp_option,
                                ctx.group,
                                ctx.target_options,
                                ctx.split,
                                ctx.isolate,
                                ctx.isolate_category,
                                feature_tp=_feat_tp,
                            )
                            _pipeline_cleanup(label=f"run_all tp{ctx.tp_option} {ctx.target_options}")

        elif not _tp_all_mode and ctx.batch_mode != "Single Target" and ctx.batch_mode in _TARGET_BUNDLES:
            # Batch mode: run each target in the selected bundle
            _batch_targets = _TARGET_BUNDLES[ctx.batch_mode]
            _batch_summary = []
            _n_batch = len(_batch_targets)

            logger.info(f"\n{'═' * 70}")
            logger.info(f"  BATCH MODE: {ctx.batch_mode}")
            logger.info(f"  Targets ({_n_batch}): {_batch_targets}")
            logger.info(f"{'═' * 70}")

            for _bi, _bt in enumerate(_batch_targets, 1):
                logger.info(f"\n{'━' * 70}")
                logger.info(f"  [{_bi}/{_n_batch}]  TARGET: {_bt}")
                logger.info(f"{'━' * 70}")

                # Override the global target_options for this iteration
                ctx.target_options = _bt

                # Set target-specific FI barplot color
                if _bt in _BATCH_TARGET_COLORS:
                    ctx.ensemble_barplot_color = _BATCH_TARGET_COLORS[_bt]
                    ctx.ensemble_barplot_palette = "None"  # force monochrome mode
                    ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"] = _resolve_ensemble_palette
                    logger.info(f"  FI colour: {ctx.ensemble_barplot_color}")

                try:
                    run_analysis(
                        ctx.tp_option,
                        ctx.group,
                        ctx.target_options,
                        ctx.split,
                        ctx.isolate,
                        ctx.isolate_category,
                        ctx.quantile_segment,
                        feature_tp=_feat_tp,
                    )

                    # Create sample_valid for this target
                    ctx.sample_valid, ctx.sample_valid_train, ctx.sample_valid_test = _create_sample_valid_for_target()

                    # Archive this target's results for later stages
                    import copy

                    ctx._batch_archive[_bt] = {
                        "results_summary": copy.deepcopy(ctx.results_summary),
                        "metrics_dict_full": copy.deepcopy(ctx.metrics_dict_full),
                        "consensus_df": copy.deepcopy(ctx.consensus_df)
                        if ("consensus_df" in ctx and ctx.consensus_df is not None)
                        else None,
                        "model_weights": copy.deepcopy(ctx.model_weights)
                        if ("model_weights" in ctx and ctx.model_weights is not None)
                        else None,
                        "X_train": ctx.X_train.copy() if ("X_train" in ctx and ctx.X_train is not None) else None,
                        "y_train": ctx.y_train.copy() if ("y_train" in ctx and ctx.y_train is not None) else None,
                        "X_test": ctx.X_test.copy() if ("X_test" in ctx and ctx.X_test is not None) else None,
                        "y_test": ctx.y_test.copy() if ("y_test" in ctx and ctx.y_test is not None) else None,
                        "exported_within_dict": copy.deepcopy(ctx.exported_within_dict)
                        if ("exported_within_dict" in ctx and isinstance(ctx.exported_within_dict, dict))
                        else {},
                        "split": ctx.split if "split" in ctx else "across_categories",
                        "target": _bt,
                        "category_results": copy.deepcopy(ctx.category_results)
                        if ("category_results" in ctx and isinstance(ctx.category_results, dict))
                        else {},
                        "sample_valid": ctx.sample_valid.copy()
                        if ("sample_valid" in ctx and ctx.sample_valid is not None)
                        else None,
                        "sample_valid_test": ctx.sample_valid_test.copy()
                        if ("sample_valid_test" in ctx and ctx.sample_valid_test is not None)
                        else None,
                        "sample_valid_train": ctx.sample_valid_train.copy()
                        if ("sample_valid_train" in ctx and ctx.sample_valid_train is not None)
                        else None,
                        "last_fitted_model": copy.deepcopy(ctx.last_fitted_model)
                        if ("last_fitted_model" in ctx and ctx.last_fitted_model is not None)
                        else None,
                    }

                    # Harvest the best meta-model result for the summary table
                    if ctx.metrics_dict_full.get("category"):
                        _best_score = -1e9
                        _best_entry = None
                        for _cat, _mets in zip(ctx.metrics_dict_full["category"], ctx.metrics_dict_full["metrics"]):
                            if isinstance(_mets, dict) and "meta_model" in _mets:
                                _mm = _mets["meta_model"]
                                if isinstance(_mm, dict):
                                    _sc = _mm.get("r2", _mm.get("auc", -1e9))
                                    if _sc > _best_score:
                                        _best_score = _sc
                                        _best_entry = dict(_mm)
                                        _best_entry["best_category"] = _cat

                        if _best_entry:
                            _best_entry["target"] = _bt
                            _batch_summary.append(_best_entry)

                    _pipeline_cleanup(label=f"batch {_bi}/{_n_batch}")

                except Exception as _be:
                    logger.info(f"  Warning: Failed for {_bt}: {_be}")
                    import traceback

                    traceback.print_exc()
                    _pipeline_cleanup(label="post-error")  # Cleanup even on failure

            # Print cross-target summary table
            logger.info(f"\n{'═' * 70}")
            logger.info(f"  BATCH COMPLETE: {ctx.batch_mode}")
            logger.info(f"{'═' * 70}")

            if _batch_summary:
                _batch_df = pd.DataFrame(_batch_summary)
                _priority_cols = ["target", "best_category"]
                _other_cols = [c for c in _batch_df.columns if c not in _priority_cols]
                _batch_df = _batch_df[[c for c in _priority_cols if c in _batch_df.columns] + _other_cols]
                display(_batch_df)

                try:
                    _batch_csv = f"batch_{ctx.batch_mode.replace(' ', '_').replace('+', '_')}.csv"
                    _batch_df.to_csv(_batch_csv, index=False)
                    logger.info(f"\n  Saved: {_batch_csv}")
                except Exception as _e:
                    logger.info(f"Warning: {_e}")

                batch_results_df = _batch_df
            else:
                logger.info("  No results collected.")
                batch_results_df = pd.DataFrame()

            # After batch, target_options / results_summary / metrics_dict_full
            # all reflect the LAST target. Later stages show detail for that
            # target. _batch_archive holds ALL targets for the batch summary.

        elif ctx.batch_mode in _TP_SWEEP_BUNDLES:
            # Timepoint-sweep batch: each entry is a (target, tp) tuple
            _sweep_targets = _TP_SWEEP_BUNDLES[ctx.batch_mode]
            _batch_summary = []
            _n_batch = len(_sweep_targets)

            logger.info(f"\n{'═' * 70}")
            logger.info(f"  TP-SWEEP BATCH: {ctx.batch_mode}")
            logger.info(f"  Runs ({_n_batch}): {_sweep_targets}")
            logger.info(f"{'═' * 70}")

            for _bi, (_bt, _btp) in enumerate(_sweep_targets, 1):
                logger.info(f"\n{'━' * 70}")
                logger.info(f"  [{_bi}/{_n_batch}]  TARGET: {_bt}  |  TP: {_btp}")
                logger.info(f"{'━' * 70}")

                # Override both global target_options and tp_option
                ctx.target_options = _bt
                ctx.tp_option = _btp

                # Set target-specific FI barplot color
                if _bt in _BATCH_TARGET_COLORS:
                    ctx.ensemble_barplot_color = _BATCH_TARGET_COLORS[_bt]
                    ctx.ensemble_barplot_palette = "None"
                    ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"] = _resolve_ensemble_palette
                    logger.info(f"  FI colour: {ctx.ensemble_barplot_color}")

                try:
                    run_analysis(
                        ctx.tp_option,
                        ctx.group,
                        ctx.target_options,
                        ctx.split,
                        ctx.isolate,
                        ctx.isolate_category,
                        ctx.quantile_segment,
                        feature_tp=_feat_tp,
                    )

                    # Create sample_valid for this target
                    ctx.sample_valid, ctx.sample_valid_train, ctx.sample_valid_test = _create_sample_valid_for_target()

                    # Archive with target__tp key to avoid overwrites
                    import copy

                    _archive_key = f"{_bt}__tp{_btp}"
                    ctx._batch_archive[_archive_key] = {
                        "results_summary": copy.deepcopy(ctx.results_summary),
                        "metrics_dict_full": copy.deepcopy(ctx.metrics_dict_full),
                        "consensus_df": copy.deepcopy(ctx.consensus_df)
                        if ("consensus_df" in ctx and ctx.consensus_df is not None)
                        else None,
                        "model_weights": copy.deepcopy(ctx.model_weights)
                        if ("model_weights" in ctx and ctx.model_weights is not None)
                        else None,
                        "X_train": ctx.X_train.copy() if ("X_train" in ctx and ctx.X_train is not None) else None,
                        "y_train": ctx.y_train.copy() if ("y_train" in ctx and ctx.y_train is not None) else None,
                        "X_test": ctx.X_test.copy() if ("X_test" in ctx and ctx.X_test is not None) else None,
                        "y_test": ctx.y_test.copy() if ("y_test" in ctx and ctx.y_test is not None) else None,
                        "exported_within_dict": copy.deepcopy(ctx.exported_within_dict)
                        if ("exported_within_dict" in ctx and isinstance(ctx.exported_within_dict, dict))
                        else {},
                        "split": ctx.split if "split" in ctx else "across_categories",
                        "target": _bt,
                        "timepoint": _btp,
                        "category_results": copy.deepcopy(ctx.category_results)
                        if ("category_results" in ctx and isinstance(ctx.category_results, dict))
                        else {},
                        "sample_valid": ctx.sample_valid.copy()
                        if ("sample_valid" in ctx and ctx.sample_valid is not None)
                        else None,
                        "sample_valid_test": ctx.sample_valid_test.copy()
                        if ("sample_valid_test" in ctx and ctx.sample_valid_test is not None)
                        else None,
                        "sample_valid_train": ctx.sample_valid_train.copy()
                        if ("sample_valid_train" in ctx and ctx.sample_valid_train is not None)
                        else None,
                        "last_fitted_model": copy.deepcopy(ctx.last_fitted_model)
                        if ("last_fitted_model" in ctx and ctx.last_fitted_model is not None)
                        else None,
                    }

                    # Harvest the best meta-model result for the summary table
                    if ctx.metrics_dict_full.get("category"):
                        _best_score = -1e9
                        _best_entry = None
                        for _cat, _mets in zip(ctx.metrics_dict_full["category"], ctx.metrics_dict_full["metrics"]):
                            if isinstance(_mets, dict) and "meta_model" in _mets:
                                _mm = _mets["meta_model"]
                                if isinstance(_mm, dict):
                                    _sc = _mm.get("r2", _mm.get("auc", -1e9))
                                    if _sc > _best_score:
                                        _best_score = _sc
                                        _best_entry = dict(_mm)
                                        _best_entry["best_category"] = _cat

                        if _best_entry:
                            _best_entry["target"] = _bt
                            _best_entry["timepoint"] = _btp
                            _batch_summary.append(_best_entry)

                    _pipeline_cleanup(label=f"tp-sweep {_bt} tp{_btp}")

                except Exception as _be:
                    logger.info(f"  Warning: Failed for {_bt} at tp{_btp}: {_be}")
                    import traceback

                    traceback.print_exc()
                    _pipeline_cleanup(label="post-error")

            # Print cross-target summary table
            logger.info(f"\n{'═' * 70}")
            logger.info(f"  TP-SWEEP BATCH COMPLETE: {ctx.batch_mode}")
            logger.info(f"{'═' * 70}")

            if _batch_summary:
                _batch_df = pd.DataFrame(_batch_summary)
                _priority_cols = ["target", "timepoint", "best_category"]
                _other_cols = [c for c in _batch_df.columns if c not in _priority_cols]
                _batch_df = _batch_df[[c for c in _priority_cols if c in _batch_df.columns] + _other_cols]
                display(_batch_df)

                try:
                    _batch_csv = f"batch_{ctx.batch_mode.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
                    _batch_df.to_csv(_batch_csv, index=False)
                    logger.info(f"\n  Saved: {_batch_csv}")
                except Exception as _e:
                    logger.info(f"Warning: {_e}")

                batch_results_df = _batch_df
            else:
                logger.info("  No results collected.")
                batch_results_df = pd.DataFrame()

        elif _tp_all_mode and ctx.batch_mode == "Single Target":
            # All-TP sweep for a single target
            # Iterates T0→T4, running the selected target at every available
            # timepoint. Archives results with target__tpN keys so downstream
            # stages render cross-TP batch panels automatically.
            _all_tps = ["0", "1", "2", "3", "4"]
            _batch_summary = []
            _n_batch = len(_all_tps)

            logger.info(f"\n{'=' * 70}")
            logger.info(f"  ALL-TP SWEEP (Single Target): {ctx.target_options}")
            logger.info(f"  Timepoints: {_all_tps}")
            logger.info(f"{'=' * 70}")

            for _bi, _btp in enumerate(_all_tps, 1):
                logger.info(f"\n{'-' * 70}")
                logger.info(f"  [{_bi}/{_n_batch}]  TARGET: {ctx.target_options}  |  TP: {_btp}")
                logger.info(f"{'-' * 70}")

                ctx.tp_option = _btp  # override global for this iteration

                # Set target-specific FI barplot color
                if ctx.target_options in _BATCH_TARGET_COLORS:
                    ctx.ensemble_barplot_color = _BATCH_TARGET_COLORS[ctx.target_options]
                    ctx.ensemble_barplot_palette = "None"
                    ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"] = _resolve_ensemble_palette
                    logger.info(f"    FI colour: {ctx.ensemble_barplot_color}")

                try:
                    run_analysis(
                        ctx.tp_option,
                        ctx.group,
                        ctx.target_options,
                        ctx.split,
                        ctx.isolate,
                        ctx.isolate_category,
                        ctx.quantile_segment,
                        feature_tp=_feat_tp,
                    )

                    ctx.sample_valid, ctx.sample_valid_train, ctx.sample_valid_test = _create_sample_valid_for_target()

                    import copy

                    _archive_key = f"{ctx.target_options}__tp{_btp}"
                    ctx._batch_archive[_archive_key] = {
                        "results_summary": copy.deepcopy(ctx.results_summary),
                        "metrics_dict_full": copy.deepcopy(ctx.metrics_dict_full),
                        "consensus_df": copy.deepcopy(ctx.consensus_df)
                        if ("consensus_df" in ctx and ctx.consensus_df is not None)
                        else None,
                        "model_weights": copy.deepcopy(ctx.model_weights)
                        if ("model_weights" in ctx and ctx.model_weights is not None)
                        else None,
                        "X_train": ctx.X_train.copy() if ("X_train" in ctx and ctx.X_train is not None) else None,
                        "y_train": ctx.y_train.copy() if ("y_train" in ctx and ctx.y_train is not None) else None,
                        "X_test": ctx.X_test.copy() if ("X_test" in ctx and ctx.X_test is not None) else None,
                        "y_test": ctx.y_test.copy() if ("y_test" in ctx and ctx.y_test is not None) else None,
                        "exported_within_dict": copy.deepcopy(ctx.exported_within_dict)
                        if ("exported_within_dict" in ctx and isinstance(ctx.exported_within_dict, dict))
                        else {},
                        "split": ctx.split if "split" in ctx else "across_categories",
                        "target": ctx.target_options,
                        "timepoint": _btp,
                        "category_results": copy.deepcopy(ctx.category_results)
                        if ("category_results" in ctx and isinstance(ctx.category_results, dict))
                        else {},
                        "sample_valid": ctx.sample_valid.copy() if (ctx.sample_valid is not None) else None,
                        "sample_valid_test": ctx.sample_valid_test.copy()
                        if (ctx.sample_valid_test is not None)
                        else None,
                        "sample_valid_train": ctx.sample_valid_train.copy()
                        if (ctx.sample_valid_train is not None)
                        else None,
                        "last_fitted_model": copy.deepcopy(ctx.last_fitted_model)
                        if ("last_fitted_model" in ctx and ctx.last_fitted_model is not None)
                        else None,
                    }

                    if ctx.metrics_dict_full.get("category"):
                        _best_score = -1e9
                        _best_entry = None
                        for _cat, _mets in zip(ctx.metrics_dict_full["category"], ctx.metrics_dict_full["metrics"]):
                            if isinstance(_mets, dict) and "meta_model" in _mets:
                                _mm = _mets["meta_model"]
                                if isinstance(_mm, dict):
                                    _sc = _mm.get("r2", _mm.get("auc", -1e9))
                                    if _sc > _best_score:
                                        _best_score = _sc
                                        _best_entry = dict(_mm)
                                        _best_entry["best_category"] = _cat
                        if _best_entry:
                            _best_entry["target"] = ctx.target_options
                            _best_entry["timepoint"] = _btp
                            _batch_summary.append(_best_entry)

                    _pipeline_cleanup(label=f"all-tp {ctx.target_options} tp{_btp}")

                except Exception as _be:
                    logger.info(f"  Warning:  Failed for {ctx.target_options} at tp{_btp}: {_be}")
                    import traceback

                    traceback.print_exc()
                    _pipeline_cleanup(label="post-error")

            # Summary table
            logger.info(f"\n{'=' * 70}")
            logger.info(f"  ALL-TP SWEEP COMPLETE: {ctx.target_options}")
            logger.info(f"{'=' * 70}")

            if _batch_summary:
                _batch_df = pd.DataFrame(_batch_summary)
                _priority_cols = ["target", "timepoint", "best_category"]
                _other_cols = [c for c in _batch_df.columns if c not in _priority_cols]
                _batch_df = _batch_df[[c for c in _priority_cols if c in _batch_df.columns] + _other_cols]
                display(_batch_df)
                try:
                    _batch_csv = f"alltp_{ctx.target_options.replace(' ', '_')}.csv"
                    _batch_df.to_csv(_batch_csv, index=False)
                    logger.info(f"\n  Saved: {_batch_csv}")
                except Exception as _e:
                    logger.info(f"Warning: {_e}")
                batch_results_df = _batch_df
            else:
                logger.info("  No results collected.")
                batch_results_df = pd.DataFrame()

        elif _tp_all_mode and ctx.batch_mode != "Single Target" and ctx.batch_mode in _TARGET_BUNDLES:
            # All-TP sweep for a target bundle
            # Runs every target in the bundle × every timepoint (T0→T4).
            _all_tps = ["0", "1", "2", "3", "4"]
            _batch_targets = _TARGET_BUNDLES[ctx.batch_mode]
            _sweep_pairs = [(t, tp) for t in _batch_targets for tp in _all_tps]
            _batch_summary = []
            _n_batch = len(_sweep_pairs)

            logger.info(f"\n{'=' * 70}")
            logger.info(f"  ALL-TP × BATCH SWEEP: {ctx.batch_mode}")
            logger.info(f"  Targets ({len(_batch_targets)}) × TPs ({len(_all_tps)}) = {_n_batch} runs")
            logger.info(f"{'=' * 70}")

            for _bi, (_bt, _btp) in enumerate(_sweep_pairs, 1):
                logger.info(f"\n{'-' * 70}")
                logger.info(f"  [{_bi}/{_n_batch}]  TARGET: {_bt}  |  TP: {_btp}")
                logger.info(f"{'-' * 70}")

                ctx.target_options = _bt
                ctx.tp_option = _btp

                if _bt in _BATCH_TARGET_COLORS:
                    ctx.ensemble_barplot_color = _BATCH_TARGET_COLORS[_bt]
                    ctx.ensemble_barplot_palette = "None"
                    ctx.ENSEMBLE_VIZ_CONFIG["palette_fn"] = _resolve_ensemble_palette
                    logger.info(f"    FI colour: {ctx.ensemble_barplot_color}")

                try:
                    run_analysis(
                        ctx.tp_option,
                        ctx.group,
                        ctx.target_options,
                        ctx.split,
                        ctx.isolate,
                        ctx.isolate_category,
                        ctx.quantile_segment,
                        feature_tp=_feat_tp,
                    )

                    ctx.sample_valid, ctx.sample_valid_train, ctx.sample_valid_test = _create_sample_valid_for_target()

                    import copy

                    _archive_key = f"{_bt}__tp{_btp}"
                    ctx._batch_archive[_archive_key] = {
                        "results_summary": copy.deepcopy(ctx.results_summary),
                        "metrics_dict_full": copy.deepcopy(ctx.metrics_dict_full),
                        "consensus_df": copy.deepcopy(ctx.consensus_df)
                        if ("consensus_df" in ctx and ctx.consensus_df is not None)
                        else None,
                        "model_weights": copy.deepcopy(ctx.model_weights)
                        if ("model_weights" in ctx and ctx.model_weights is not None)
                        else None,
                        "X_train": ctx.X_train.copy() if ("X_train" in ctx and ctx.X_train is not None) else None,
                        "y_train": ctx.y_train.copy() if ("y_train" in ctx and ctx.y_train is not None) else None,
                        "X_test": ctx.X_test.copy() if ("X_test" in ctx and ctx.X_test is not None) else None,
                        "y_test": ctx.y_test.copy() if ("y_test" in ctx and ctx.y_test is not None) else None,
                        "exported_within_dict": copy.deepcopy(ctx.exported_within_dict)
                        if ("exported_within_dict" in ctx and isinstance(ctx.exported_within_dict, dict))
                        else {},
                        "split": ctx.split if "split" in ctx else "across_categories",
                        "target": _bt,
                        "timepoint": _btp,
                        "category_results": copy.deepcopy(ctx.category_results)
                        if ("category_results" in ctx and isinstance(ctx.category_results, dict))
                        else {},
                        "sample_valid": ctx.sample_valid.copy() if (ctx.sample_valid is not None) else None,
                        "sample_valid_test": ctx.sample_valid_test.copy()
                        if (ctx.sample_valid_test is not None)
                        else None,
                        "sample_valid_train": ctx.sample_valid_train.copy()
                        if (ctx.sample_valid_train is not None)
                        else None,
                        "last_fitted_model": copy.deepcopy(ctx.last_fitted_model)
                        if ("last_fitted_model" in ctx and ctx.last_fitted_model is not None)
                        else None,
                    }

                    if ctx.metrics_dict_full.get("category"):
                        _best_score = -1e9
                        _best_entry = None
                        for _cat, _mets in zip(ctx.metrics_dict_full["category"], ctx.metrics_dict_full["metrics"]):
                            if isinstance(_mets, dict) and "meta_model" in _mets:
                                _mm = _mets["meta_model"]
                                if isinstance(_mm, dict):
                                    _sc = _mm.get("r2", _mm.get("auc", -1e9))
                                    if _sc > _best_score:
                                        _best_score = _sc
                                        _best_entry = dict(_mm)
                                        _best_entry["best_category"] = _cat
                        if _best_entry:
                            _best_entry["target"] = _bt
                            _best_entry["timepoint"] = _btp
                            _batch_summary.append(_best_entry)

                    _pipeline_cleanup(label=f"alltp-batch {_bt} tp{_btp}")

                except Exception as _be:
                    logger.info(f"  Warning:  Failed for {_bt} at tp{_btp}: {_be}")
                    import traceback

                    traceback.print_exc()
                    _pipeline_cleanup(label="post-error")

            # Summary table
            logger.info(f"\n{'=' * 70}")
            logger.info(f"  ALL-TP × BATCH SWEEP COMPLETE: {ctx.batch_mode}")
            logger.info(f"{'=' * 70}")

            if _batch_summary:
                _batch_df = pd.DataFrame(_batch_summary)
                _priority_cols = ["target", "timepoint", "best_category"]
                _other_cols = [c for c in _batch_df.columns if c not in _priority_cols]
                _batch_df = _batch_df[[c for c in _priority_cols if c in _batch_df.columns] + _other_cols]
                display(_batch_df)
                try:
                    _batch_csv = f"batch_alltp_{ctx.batch_mode.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
                    _batch_df.to_csv(_batch_csv, index=False)
                    logger.info(f"\n  Saved: {_batch_csv}")
                except Exception as _e:
                    logger.info(f"Warning: {_e}")
                batch_results_df = _batch_df
            else:
                logger.info("  No results collected.")
                batch_results_df = pd.DataFrame()

        else:
            run_analysis(
                ctx.tp_option,
                ctx.group,
                ctx.target_options,
                ctx.split,
                ctx.isolate,
                ctx.isolate_category,
                ctx.quantile_segment,
                feature_tp=_feat_tp,
            )

        # After an All-TP sweep, tp_option holds the last iteration's value
        # (e.g. "4"). Restore a valid numeric string for the export section.
        if _tp_all_mode:
            if not isinstance(ctx.tp_option, str) or ctx.tp_option == "All":
                ctx.tp_option = "4"  # fallback to last wave

        logger.info("\n" + "=" * 80)
        logger.info("EXPORTING PIPELINE VARIABLES FOR VALIDATION")
        logger.info("=" * 80)

        # Safety init: ensure sample_valid vars exist even if export fails
        if "sample_valid_test" not in ctx:
            ctx.sample_valid_test = None
            ctx.sample_valid_train = None
            ctx.sample_valid = None

        try:
            t = int(ctx.tp_option)
            targets = [ctx.target_options]

            if ctx.split == "across_categories":
                exported_data = across_categories_data(
                    t, targets, group=ctx.group if ctx.group else None, scale_x=True, scale_y=False, verbose=True
                )
                ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = exported_data
            else:
                exported_data = within_categories_data(
                    t, targets, group=ctx.group if ctx.group else None, scale_x=True, scale_y=False, verbose=True
                )
                # Category selector for within_categories runs.
                _available_cats = list(exported_data.keys())
                logger.info(
                    f"   Available categories ({len(_available_cats)}): {_available_cats[:5]}{'...' if len(_available_cats) > 5 else ''}"
                )

                # Use isolate_category if valid, otherwise first key
                _ic = (
                    ctx.isolate_category
                    if ("isolate_category" in ctx and ctx.isolate_category and ctx.isolate_category != "None")
                    else None
                )
                if _ic and _ic in exported_data:
                    _export_cat = _ic
                else:
                    _export_cat = _available_cats[0]
                    if _ic:
                        logger.info(f"   Warning:  isolate_category='{_ic}' not found in categories, using first")

                logger.info(f"   Exporting category: '{_export_cat}' for the validation stages")
                ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = exported_data[_export_cat]
                ctx.exported_within_dict = exported_data  # Store full dict for optional looped validation

            # Create sample_valid (the rows kept for this target) for the validation stages
            # Must happen BEFORE clean_data (which resets indices)
            # X_train.index currently contains scattered positions from
            # RangeIndex(0, N-1) assigned during across_categories_data().
            # These positions map to rows in the filtered data.
            _train_positions = ctx.X_train.index.values.copy()
            _test_positions = ctx.X_test.index.values.copy()

            # Reconstruct the same row filtering as across_categories_data
            # to get a demographic lookup table with matching row order
            _t_export = int(ctx.tp_option)
            _grp = ctx.group if ctx.group else None

            _filtered_rows = ctx.sample[ctx.sample["time"] == _t_export].copy()

            if _grp and _grp != "None":
                if _grp == "Female":
                    _filtered_rows = _filtered_rows[_filtered_rows["sex"] == 2]
                elif _grp == "Male":
                    _filtered_rows = _filtered_rows[_filtered_rows["sex"] == 1]
                elif _grp == "high_ale":
                    _filtered_rows = _filtered_rows[_filtered_rows["high_ale"] == True]
                elif _grp == "high_ale_severe_p_mh":
                    _filtered_rows = _filtered_rows[_filtered_rows["high_ale_severe_p_mh"] == True]
                elif _grp == "low_ale":
                    _filtered_rows = _filtered_rows[_filtered_rows["low_ale_children_p"] == True]
                # else: unknown group string, no additional filter

            # Mirror the row-level filtering applied inside the data functions.
            if (
                "depress_D_p" not in targets
                and "depress_D_p_rev" not in targets
                and "external_D_p" not in targets
                and "adhd_D_p" not in targets
            ):
                if "latent_class_depression" in targets and "latent_class_depression" in _filtered_rows.columns:
                    _before = len(_filtered_rows)
                    _filtered_rows = _filtered_rows[_filtered_rows["latent_class_depression"] != ""]
                    logger.info(f"   Applied latent_class filter: {_before} → {len(_filtered_rows)} rows")

            # Drop rows with missing target values.
            # The data functions (across/within_categories_data) drop target-NaN
            # rows before train_test_split. We must replicate that here so the
            # row count of _filtered_rows matches len(X_train) + len(X_test).
            if ctx.target_options in _filtered_rows.columns:
                _target_mask = _filtered_rows[ctx.target_options].notna()
                # Also handle empty-string entries (e.g. categorical targets)
                if _filtered_rows[ctx.target_options].dtype == object:
                    _target_mask = _target_mask & (_filtered_rows[ctx.target_options].astype(str).str.strip() != "")
                if (~_target_mask).any():
                    _n_target_na = (~_target_mask).sum()
                    logger.info(f"   Dropped {_n_target_na} rows with missing/empty target '{ctx.target_options}'")
                    _filtered_rows = _filtered_rows[_target_mask]

            _filtered_rows = _filtered_rows.reset_index(drop=True)

            # Verify alignment: row count must match
            _expected_n = len(ctx.X_train) + len(ctx.X_test)
            if len(_filtered_rows) != _expected_n:
                logger.info(
                    f"   Warning:  sample_valid row mismatch: filtered={len(_filtered_rows)}, "
                    f"data={_expected_n}. Validation stages that use sample_valid may not work."
                )
                ctx.sample_valid = None
                ctx.sample_valid_train = None
                ctx.sample_valid_test = None
            else:
                ctx.sample_valid_train = _filtered_rows.iloc[_train_positions].reset_index(drop=True)
                ctx.sample_valid_test = _filtered_rows.iloc[_test_positions].reset_index(drop=True)
                ctx.sample_valid = pd.concat([ctx.sample_valid_train, ctx.sample_valid_test], ignore_index=True)
                logger.info(f"   sample_valid created: {ctx.sample_valid.shape}")
                logger.info(f"      train: {ctx.sample_valid_train.shape}, test: {ctx.sample_valid_test.shape}")

                # Quick sanity check: sex distribution should be reasonable
                if "sex" in ctx.sample_valid.columns:
                    _sex_counts = ctx.sample_valid_train["sex"].value_counts().to_dict()
                    logger.info(f"      Sex distribution (train): {_sex_counts}")

            # Apply same NaN/inf cleaning as main pipeline (BEFORE flattening y)
            # WARNING: This resets X_train/X_test to RangeIndex(0, N-1)
            na_method_mapping = {
                "fill_with_zero": ("fill_0", None),
                "drop": ("remove", None),
                "impute_mean": ("impute", "mean"),
                "impute_median": ("impute", "median"),
            }
            method, imputer_strategy = na_method_mapping[ctx.deal_with_na_values]
            ctx.X_train, ctx.y_train = clean_data(
                ctx.X_train, ctx.y_train, method=method, imputer_strategy=imputer_strategy
            )
            ctx.X_test, ctx.y_test = clean_data(
                ctx.X_test, ctx.y_test, method=method, imputer_strategy=imputer_strategy
            )
            ctx.X_train = ctx.X_train.loc[:, ~ctx.X_train.columns.duplicated(keep="first")]
            ctx.X_test = ctx.X_test.loc[:, ~ctx.X_test.columns.duplicated(keep="first")]

            # Post-clean_data safety check: if clean_data dropped rows (method='remove'),
            # sample_valid is now misaligned. Detect and warn.
            if ctx.sample_valid_train is not None and len(ctx.sample_valid_train) != len(ctx.X_train):
                logger.info(
                    f"   Warning:  clean_data changed row count: sample_valid_train={len(ctx.sample_valid_train)}, "
                    f"X_train={len(ctx.X_train)}. Resetting sample_valid to None."
                )
                ctx.sample_valid = None
                ctx.sample_valid_train = None
                ctx.sample_valid_test = None

            # Flatten y if it's a DataFrame with one column (AFTER cleaning)
            if hasattr(ctx.y_train, "columns") and len(ctx.y_train.columns) == 1:
                ctx.y_train = ctx.y_train.iloc[:, 0]
            if hasattr(ctx.y_test, "columns") and len(ctx.y_test.columns) == 1:
                ctx.y_test = ctx.y_test.iloc[:, 0]

            logger.info("\nExported pipeline variables:")
            logger.info(f"   Target: {ctx.target_options}")
            logger.info(f"   Timepoint: {ctx.tp_option}")
            logger.info(f"   Split: {ctx.split}")
            logger.info(f"   X_train: {ctx.X_train.shape}")
            logger.info(f"   X_test:  {ctx.X_test.shape}")
            logger.info(
                f"   y_train: {ctx.y_train.shape}, unique values: {len(ctx.y_train.unique()) if hasattr(ctx.y_train, 'unique') else len(set(ctx.y_train))}"
            )
            logger.info(f"   y_test:  {ctx.y_test.shape}")

        except Exception as e:
            logger.info(f"\nWarning:  Could not export pipeline variables: {e}")
            logger.info("   The validation stages will not have real pipeline data.")
            import traceback

            traceback.print_exc()
        # [Cache save]
        _RS["section"] = "uncategorized"
        try:
            if "X_train" in ctx and ctx.X_train is not None:
                report_log_data_info(ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test)
            if "results_summary" in ctx and isinstance(ctx.results_summary, list) and ctx.results_summary:
                try:
                    report_log_table(
                        pd.DataFrame(ctx.results_summary), "Per-Category Performance", section="main_runner"
                    )
                except Exception:
                    pass  # non-critical; logged upstream or optional visualization
            cache_save(
                {
                    "results_summary": ctx.get("results_summary", []),
                    "metrics_dict_full": ctx.get("metrics_dict_full", {}),
                    "X_train": ctx.get("X_train"),
                    "X_test": ctx.get("X_test"),
                    "y_train": ctx.get("y_train"),
                    "y_test": ctx.get("y_test"),
                    "exported_within_dict": ctx.get("exported_within_dict", {}),
                    "sample_valid": ctx.get("sample_valid"),
                    "sample_valid_train": ctx.get("sample_valid_train"),
                    "sample_valid_test": ctx.get("sample_valid_test"),
                    "_batch_archive": ctx._batch_archive if "_batch_archive" in ctx else {},
                    "_batch_mode": ctx.batch_mode if "batch_mode" in ctx else "Single Target",
                },
                cell_tag="main_runner",
                key=_KEY,
                label=f"{ctx.target_options}_tp{ctx.tp_option}_{ctx.split}",
            )
        except Exception as _e:
            logger.info(f"[Cache] Could not save main_runner: {_e}")
