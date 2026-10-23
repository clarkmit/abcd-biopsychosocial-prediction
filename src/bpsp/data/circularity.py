"""Target circularity exclusions.

Builds the per-target circularity exclusion dictionary and defines
get_circular_exclusions(). Removes construct-overlapping, tautological,
and reverse-causation predictors independently for each target, which is
necessary given high covariation among CBCL items, where raters may
conflate related constructs (e.g. anxiety/depression, ADHD/externalizing).
Applies to both within-category (category-level skips) and across-category
(variable-level drops) analysis modes.

Shared exclusion sets for composite targets (defined in the registry unless marked):

Internalizing / Depression
_GENERAL_INTERNALIZING_EXCLUDE     _ANXDEP_SYNDROME
_WITHDRAWN_SYNDROME                _SOMATIC_SYNDROME
_DELTA_INTERNALIZING_EXCLUDE       _DEPRESSION_SBT
_DEPRESSION_T1T3T4_EXTRA           _ANXIETY_T1T3T4_EXTRA

Externalizing / ADHD
_DELTA_EXTERNALIZING_EXCLUDE       _AGGRESSIVE_SYNDROME
_RULEBREAKING_SYNDROME             _EXTERNALIZING_THEORETICAL
_ADHD_EXTRA_GENERAL                _ADHD_T1T3T4_EXTRA  (defined in this module)

SBT subtype item sets
_SBT_GUILT_HOPELESSNESS_ITEMS      _SBT_ANXIETY_DEPRESSION_ITEMS
_SBT_SOCIAL_WITHDRAWAL_ITEMS       _SBT_SOMATIC_DEPRESSION_ITEMS
_SBT_AVOIDANCE_FEAR_ITEMS          _SBT_PERFECTIONISM_ITEMS
_SBT_WELL_BEING_ITEMS              _SBT_EMOTIONAL_DYSREGULATION_ITEMS
_SBT_AGGRESSION_IRRITABILITY_ITEMS _SBT_ACADEMIC_COGNITIVE_ITEMS

Targets defined in this module:
CHILD TARGETS: PRIMARY
1.  Internalizing          internal_D_p
2.  Depression             depress_D_p + 28 targets, SBT subtypes
3.  Anxiety                anxdisord_D_p
4.  Externalizing          external_D_p
5.  ADHD                   adhd_D_p, depadhd_c, asdadhd_c
6.  Social Problems        social_problems_D_p, not_liked_p, doesnt_get_along_p
CHILD TARGETS: SECONDARY
7.  Grade Outcomes         bad_grades
8.  NIH Toolbox            tb_flanker, tb_reading, tb_fluid, tb_cryst
9.  Thought Disorder       thought_disorder_D_p
10.  Child OCD              ocd_D_p
11.  Child Obsessions       obsessions_p
12.  Child Perfectionism    perfectionist_p
13.  Child Impulsivity      impulsive_p
14.  Child Compulsivity     compulsions_p
PARENT TARGETS: PRIMARY
15.  Parent Depression      parent_depress_D_p + 2 targets, delta variant
16.  Parent Anxiety         parent_anxdisord_D_p, parent_worries_a_lot_p, delta variant
17.  Parent ADHD            parent_adhd_D_p, parent_concentration_trouble_p
PARENT TARGETS: SECONDARY
18.  Parent Social Quality  parent_bad_relationships_p + 3 targets
19.  Parent Happiness       parent_happy_person_p
20.  Parent Suicidal        parent_suicidal_thoughts_p
21.  Parent Impulsivity     parent_impulsive_p
22.  Parent Substance Use   parent_tobacco_use_frequency_p, parent_drinks_frequency_p,
parent_drunk_days_p
OTHER TARGETS
23.  SES / Neighborhood    parent_income, area_deprivation_idx
24.  Miscellaneous          gd_riskybets, nb_correct_nt_2back, pa_sum_k,
bdefs_distract_upset_p, asd_ssrs_sum
25.  Body Weight            weight
"""

from bpsp.data.registry import (
    _ADHD_EXTRA_GENERAL,
    _AGGRESSIVE_SYNDROME,
    _ANXDEP_SYNDROME,
    _ANXIETY_T1T3T4_EXTRA,
    _DELTA_EXTERNALIZING_EXCLUDE,
    _DELTA_INTERNALIZING_EXCLUDE,
    _DEPRESSION_SBT,
    _DEPRESSION_T1T3T4_EXTRA,
    _EXTERNALIZING_THEORETICAL,
    _GENERAL_INTERNALIZING_EXCLUDE,
    _PARENT_GENERAL_MOOD_EXCLUDE,
    _RULEBREAKING_SYNDROME,
    _SBT_ACADEMIC_COGNITIVE_ITEMS,
    _SBT_AGGRESSION_IRRITABILITY_ITEMS,
    _SBT_ANXIETY_DEPRESSION_ITEMS,
    _SBT_AVOIDANCE_FEAR_ITEMS,
    _SBT_EMOTIONAL_DYSREGULATION_ITEMS,
    _SBT_GUILT_HOPELESSNESS_ITEMS,
    _SBT_PERFECTIONISM_ITEMS,
    _SBT_SOCIAL_WITHDRAWAL_ITEMS,
    _SBT_SOMATIC_DEPRESSION_ITEMS,
    _SBT_WELL_BEING_ITEMS,
    _SOMATIC_SYNDROME,
    _WITHDRAWN_SYNDROME,
)

CIRCULAR_EXCLUSIONS = {}

# Extend helper sets from the registry based on full registry audit. These augment
# the original sets in-place before they are used by target entries below.

# Depression: 10 T3/T4 child_upset_* items not in original set --
# irritable/dysphoric affect, loss-of-control, and emotional-overwhelm items
# that operationalise DSM youth MDD criteria in the ERC-P instrument.
_DEPRESSION_T1T3T4_EXTRA = _DEPRESSION_T1T3T4_EXTRA | {
    "child_upset_angry_p",
    "child_upset_irritated_p",
    "child_upset_embarrassed_p",
    "child_upset_emotions_overwhelm_p",
    "child_upset_control_p",
    "child_upset_no_control_p",
    "child_upset_lose_control_p",
    "child_upset_out_control_p",
    "child_upset_acknowledge_p",
    "child_upset_better_p",
}

# Anxiety: 3 T3/T4 items not in original set --
# angry/irritable arousal, avoidant acknowledgement, and emotional flooding
# are indistinguishable from anxious agitation at the parent-report level.
_ANXIETY_T1T3T4_EXTRA = _ANXIETY_T1T3T4_EXTRA | {
    "child_upset_angry_p",
    "child_upset_acknowledge_p",
    "child_upset_emotions_overwhelm_p",
}

# ADHD: new T1T3T4 set, ERC-P attention/control items and BDEFS executive-
# function items that load directly on the ADHD EF/dysregulation pathway.
_ADHD_T1T3T4_EXTRA = {
    "child_upset_bad_focus_p",
    "child_upset_poor_concentrate_p",
    "child_upset_control_p",
    "child_upset_no_control_p",
    "child_upset_lose_control_p",
    "child_upset_out_control_p",
    "bdefs_explain_idea_p",
    "bdefs_explain_pt_p",
    "bdefs_explain_seq_p",
    "bdefs_inconsistant_p",
    "bdefs_lazy_p",
    "bdefs_process_info_p",
    "bdefs_rechannel_p",
    "bdefs_consequences_p",
}

# Child targets: primary

# 1. Internalizing
# All three CBCL syndrome subscales constituting the internalizing broadband are
# excluded; SBT composites are excluded to prevent indirect reconstruction.
_INTERNAL_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _ANXDEP_SYNDROME
    | _WITHDRAWN_SYNDROME
    | _SOMATIC_SYNDROME
    | _DELTA_INTERNALIZING_EXCLUDE
    | _DEPRESSION_SBT
    | {"clings_to_adults_p"}  # cbcl_q11 Anxious/Depressed constituent (safety net; see registry _ANXDEP_SYNDROME)
    | {
        "social_problems_D_p",
        "self_conscious_k",
        "anxious_fearful_k",
        "feels_worthless_k",
        "difficulty_goingtosleep_p",
        "difficulty_wakingup_p",
        "sleep_problems_p",
        "sleeps_little_p",
        "sleeps_alot_p",
        "overeating_p",
        "sociable_p",
        "adhd_d_p",
        "parent_depress_D_p",
        "bdefs_calm_down_p",
        "bdefs_consequences_p",
        "bdefs_distract_upset_p",
        "bdefs_explain_idea_p",
        "bdefs_explain_pt_p",
        "bdefs_explain_seq_p",
        "bdefs_impulsive_action_p",
        "bdefs_inconsistant_p",
        "bdefs_lazy_p",
        "bdefs_process_info_p",
        "bdefs_rechannel_p",
        "bdefs_sense_time_p",
        "bdefs_shortcuts_p",
        "bdefs_stop_think_p",
        "child_emotion_overwhelm_p",
        "child_clear_feelings_p",
        "child_feelings_attentive_p",
        "child_feelings_care_p",
        "child_feelings_know_p",
        "child_upset_acknowledge_p",
        "child_upset_angry_p",
        "child_upset_ashamed_p",
        "child_upset_bad_behavior_p",
        "child_upset_bad_esteem_p",
        "child_upset_bad_focus_p",
        "child_upset_better_p",
        "child_upset_control_p",
        "child_upset_embarrassed_p",
        "child_upset_emotions_overwhelm_p",
        "child_upset_fixation_p",
        "child_upset_guilty_p",
        "child_upset_irritated_p",
        "child_upset_longtime_better_p",
        "child_upset_longtime_p",
        "child_upset_lose_control_p",
        "child_upset_no_control_p",
        "child_upset_nothing_better_p",
        "child_upset_out_control_p",
        "child_upset_poor_concentrate_p",
        "child_upset_unproductive_p",
        "child_upset_weak_p",
        "nervous_general_p",
        "paranoid_p",
        "scared_dark_p",
        "nervous_twitching_p",
    }
    | {"parent_anhedonia_B_p"}  # DSM MDD core criterion; subsumed by internalizing composite
)
_INTERNAL_WITHIN = {"Anxiety", "Child Anxiety", "ADHD", "Child ADHD", "Externalizing", "Child Externalizing"}
CIRCULAR_EXCLUSIONS["internal_D_p"] = {"across_exclude": _INTERNAL_ACROSS, "within_exclude": _INTERNAL_WITHIN}

# 2. Depression
# Covers CBCL Withdrawn and Anxious/Depressed items that operationalise DSM depression
# criteria, sleep/appetite disturbance items, child self-report mood indices, and all
# SBT composites. Anxiety category excluded within-category due to overlapping affect items.
_DEPRESSION_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _DELTA_INTERNALIZING_EXCLUDE
    | _DEPRESSION_SBT
    | {"social_problems_D_p", "delta_parent_depressed_p"}
    | {"parent_anhedonia_B_p", "parent_dysthymia_B_p"}  # DSM MDD core criterion / depressive disorder diagnosis
    | {
        "enjoys_little_p",
        "cries_p",
        "selfharm_p",
        "doesnt_eat_well_p",
        "feels_unloved_p",
        "feels_inferior_p",
        "guilty_p",
        "overtired_p",
        "sleeps_less_p",
        "sleeps_more_p",
        "suicidal_p",
        "low_energy_p",
        "sad_p",
        "parent_wish_dead_present_B_p",
    }
    | {"withdrawn_p", "prefer_alone_p", "doesnt_talk_p", "secretive_p", "shy_p", "lonely_p"}
    | {"sleeps_little_p", "sleeps_alot_p", "sleep_problems_p", "overeating_p"}
    | {"feels_worthless_k", "anxious_fearful_k"}
    | {"difficulty_goingtosleep_p", "difficulty_wakingup_p", "paranoid_p", "nervous_twitching_p", "worries_p"}
    # KSADS depression-domain predictors in child feature pool
    | {"parent_depressed_mood_B_p", "parent_suicide_past_B_p", "nightmares_B_p", "ptsd_diagnosis_present_B_p"}
    # Parent mood items allowing indirect depression severity reconstruction
    | {"parent_suicidal_thoughts_p", "parent_self_harm_p"}
    # Emotional regulation / clarity items circular with depression construct
    | {"bdefs_rechannel_p", "child_emotion_overwhelm_p", "child_clear_feelings_p", "parent_depress_D_p"}
    # Executive-function emotion-regulation items that index depression severity
    | {"bdefs_calm_down_p", "bdefs_distract_upset_p", "bdefs_lazy_p", "bdefs_stop_think_p"}
    # Child emotion regulation scale: items directly indexing mood/affect dysregulation
    | {
        "child_upset_angry_p",
        "child_upset_ashamed_p",
        "child_upset_bad_behavior_p",
        "child_upset_bad_esteem_p",
        "child_upset_bad_focus_p",
        "child_upset_better_p",
        "child_upset_control_p",
        "child_upset_embarrassed_p",
        "child_upset_emotions_overwhelm_p",
        "child_upset_fixation_p",
        "child_upset_guilty_p",
        "child_upset_irritated_p",
        "child_upset_longtime_better_p",
        "child_upset_longtime_p",
        "child_upset_lose_control_p",
        "child_upset_no_control_p",
        "child_upset_nothing_better_p",
        "child_upset_out_control_p",
        "child_upset_poor_concentrate_p",
        "child_upset_unproductive_p",
        "child_upset_weak_p",
        "child_upset_acknowledge_p",
        "child_feelings_attentive_p",
        "child_feelings_care_p",
        "child_feelings_know_p",
    }
)
_DEPRESSION_WITHIN = {"Anxiety", "Child Anxiety"}

# Extended within_exclude for subtypes that draw on cross-domain CBCL items.
_SBT_EMODSYREG_AGGRESSION_WITHIN = _DEPRESSION_WITHIN | {"Externalizing", "Child Externalizing"}
_SBT_ACADEMIC_WITHIN = _DEPRESSION_WITHIN | {"ADHD", "Child ADHD"}

_DEP_BASE = {
    "across_exclude": _DEPRESSION_ACROSS,
    "within_exclude": _DEPRESSION_WITHIN,
    "across_exclude_t1t3t4": _DEPRESSION_T1T3T4_EXTRA,
}

for target in [
    "depress_D_p",
    "depress_D_p_rev",
    "depress_update_p",
    "delta_depress_D_p",
    "delta_sbt_core_depression",
    "sbt_core_depression",
    "sbt_core_depression_raw",
    "top_10_sbt_core_depression",
    "top_5_sbt_core_depression",
    "sbt_core_depression_delta_t0_t3",
    "sbt_fatigue_sleep",
    "dep_onset_rci_2.3",
    "dep_remission_rci_2.3",
    "dep_increase_2sd",
    "dep_increase_1.5sd",
    "dep_increase_1sd",
    "dep_decrease_2sd",
    "dep_decrease_1.5sd",
    "dep_decrease_1sd",
    "sbt_core_dep_onset_rci_2.3",
    "sbt_core_dep_decrease_2sd",
    "sbt_core_dep_remission_rci_2.3",
    "sbt_core_dep_decrease_1.5sd",
    "sbt_core_dep_increase_2sd",
    "sbt_core_dep_increase_1.5sd",
    "top_10_depression",
    "top_5_depression",
    "latent_class_depression",
    "enjoys_little_p",
    "suicidal_p",
    "MDD_KSADS_C",
    "depressed_mood_B_k",
    "anhedonia_B_k",
    "parent_wish_dead_present_B_p",
    "wish_dead_present_B_k",
    "hopeless_B_k",
    "two_more_depression_B_k",
    "ksads_1_840_t",
]:
    CIRCULAR_EXCLUSIONS[target] = _DEP_BASE

# Delta-specific overrides (depression family)
# The bulk loop above assigned _DEP_BASE (whose across_exclude is _DEPRESSION_ACROSS)
# to every depression target, including the two depression deltas. But
# _DEPRESSION_ACROSS does not contain the raw baseline composites used to compute
# those deltas (depress_D_p, sbt_core_depression at baseline trivially reconstruct
# the change score), nor the broadband internalizing raw composite. Override the
# two delta entries here with expanded exclusions. Non-delta depression targets
# keep the plain _DEP_BASE assigned above.
CIRCULAR_EXCLUSIONS["delta_depress_D_p"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS
    | {
        "depress_D_p",  # raw baseline/endpoint source of the delta
        "depress_update_p",  # depression severity update item
        "internal_D_p",  # broadband internalizing contains depress_D_p
    },
}
CIRCULAR_EXCLUSIONS["delta_sbt_core_depression"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS
    | {
        "depress_D_p",  # shares constituent items with sbt_core_depression
        "internal_D_p",  # broadband internalizing contains depression content
    },
}

# Per-subtype: base _DEPRESSION_ACROSS plus the subtype's unique constituent items.
CIRCULAR_EXCLUSIONS["sbt_guilt_hopelessness"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS | _SBT_GUILT_HOPELESSNESS_ITEMS,
}
CIRCULAR_EXCLUSIONS["sbt_anxiety_depression"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS | _SBT_ANXIETY_DEPRESSION_ITEMS,
}
CIRCULAR_EXCLUSIONS["sbt_social_withdrawal"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS | _SBT_SOCIAL_WITHDRAWAL_ITEMS,
}
CIRCULAR_EXCLUSIONS["sbt_somatic_depression"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS | _SBT_SOMATIC_DEPRESSION_ITEMS,
}
CIRCULAR_EXCLUSIONS["sbt_avoidance_fear"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS | _SBT_AVOIDANCE_FEAR_ITEMS,
}
CIRCULAR_EXCLUSIONS["sbt_perfectionism_achievement"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS | _SBT_PERFECTIONISM_ITEMS,
}
CIRCULAR_EXCLUSIONS["sbt_well_being"] = {**_DEP_BASE, "across_exclude": _DEPRESSION_ACROSS | _SBT_WELL_BEING_ITEMS}
CIRCULAR_EXCLUSIONS["sbt_emotional_dysregulation"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS | _SBT_EMOTIONAL_DYSREGULATION_ITEMS | {"breaks_rules_p"},
    "within_exclude": _SBT_EMODSYREG_AGGRESSION_WITHIN,
}
CIRCULAR_EXCLUSIONS["sbt_aggression_irritability"] = {
    **_DEP_BASE,
    "across_exclude": _DEPRESSION_ACROSS
    | _SBT_AGGRESSION_IRRITABILITY_ITEMS
    | {"impulsive_p", "not_liked_p", "doesnt_get_along_p"},
    "within_exclude": _SBT_EMODSYREG_AGGRESSION_WITHIN,
}
CIRCULAR_EXCLUSIONS["sbt_academic_cognitive"] = {
    **_DEP_BASE,
    "across_exclude": (
        _DEPRESSION_ACROSS
        | _SBT_ACADEMIC_COGNITIVE_ITEMS
        | {
            "goal_continuity_p",
            "bad_grades",
            "delta_bad_grades",
            "disobedient_school_p",
            "disobedient_home_p",
            "breaks_rules_p",
            "grades_dropped",
            "up_lackperseverance_ss_k",
            "trouble_concentrating_k",
            "strange_ideas_p",
        }
    ),
    "within_exclude": _SBT_ACADEMIC_WITHIN,
}

# 3. Anxiety
# CBCL Anxious/Depressed items plus specific-fear, somatic-anxiety, and avoidance
# items excluded. KSADS construct measures and ABCD PK items excluded where present.
_ANXIETY_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _WITHDRAWN_SYNDROME
    | _DELTA_INTERNALIZING_EXCLUDE
    | {"social_problems_D_p", "self_conscious_k", "anxious_fearful_k"}
    | {
        "clings_to_adults_p",
        "fears_excl_school_p",
        "fears_school_p",
        "nervous_general_p",
        "anxious_p",
        "worries_p",
        "secretive_p",
    }
    | {
        "fears_being_bad_p",
        "paranoid_p",
        "nervous_twitching_p",
        "perfectionist_p",
        "easily_embarrassed_p",
        "Doesnt_finish_p",
        "nightmares_p",
    }
    | {
        "social_fear_present_PK",
        "scared_dark_p",
        "fear_becoming_obese_present_PK",
        "obsessions_present_B_p",
        "difficulty_goingtosleep_p",
        "difficulty_wakingup_p",
    }
    | {
        "parent_social_anxiety_disorder_B_p",
        "parent_intrusive_thoughts_D_p",
        "parent_excessive_worry_B_p",
    }  # direct anxiety constructs / KSADS predictors
    # Emotional regulation / clarity items circular with anxiety construct
    | {"bdefs_rechannel_p", "child_emotion_overwhelm_p", "child_clear_feelings_p", "parent_depress_D_p"}
    # Construct overlap ambiguity
    | {"bdefs_calm_down_p", "bdefs_distract_upset_p", "bdefs_stop_think_p"}
    # Child emotion regulation scale: items directly indexing mood/affect dysregulation
    | {
        "child_upset_angry_p",
        "child_upset_ashamed_p",
        "child_upset_bad_behavior_p",
        "child_upset_bad_esteem_p",
        "child_upset_bad_focus_p",
        "child_upset_better_p",
        "child_upset_control_p",
        "child_upset_embarrassed_p",
        "child_upset_emotions_overwhelm_p",
        "child_upset_fixation_p",
        "child_upset_guilty_p",
        "child_upset_irritated_p",
        "child_upset_longtime_better_p",
        "child_upset_longtime_p",
        "child_upset_lose_control_p",
        "child_upset_no_control_p",
        "child_upset_nothing_better_p",
        "child_upset_out_control_p",
        "child_upset_poor_concentrate_p",
        "child_upset_unproductive_p",
        "child_upset_weak_p",
        "child_upset_acknowledge_p",
        "child_feelings_attentive_p",
        "child_feelings_care_p",
        "child_feelings_know_p",
    }
)
CIRCULAR_EXCLUSIONS["anxdisord_D_p"] = {
    "across_exclude": _ANXIETY_ACROSS,
    "within_exclude": {"Anxiety", "Child Anxiety"},
    "across_exclude_t1t3t4": _ANXIETY_T1T3T4_EXTRA,
}
# Child Only Deltas: inherits anxiety base
CIRCULAR_EXCLUSIONS["delta_anxdisord_D_p"] = {
    # Base _ANXIETY_ACROSS excludes delta-counterpart and broadband-delta composites
    # but NOT the raw baseline composite used to compute the delta. For a delta
    # target, the raw composite (at either timepoint) trivially reconstructs the
    # target, so it must be explicitly excluded. Same for broadband internalizing
    # (which contains anxdisord_D_p) and cross-informant parent_anxdisord_D_p.
    "across_exclude": _ANXIETY_ACROSS
    | {
        "anxdisord_D_p",  # raw baseline/endpoint source of the delta
        "parent_anxdisord_D_p",  # cross-informant anxiety composite (highly correlated)
        "internal_D_p",  # broadband internalizing contains anxdisord_D_p
    },
    "within_exclude": {"Anxiety", "Child Anxiety"},
    "across_exclude_t1t3t4": _ANXIETY_T1T3T4_EXTRA,
}

# 4. Externalizing
# three CBCL externalizing syndrome scales are excluded; ABCD youth
# and parent report analogues are excluded alongside CBCL-derived broadband deltas.
_EXTERNAL_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _DELTA_EXTERNALIZING_EXCLUDE
    | _AGGRESSIVE_SYNDROME
    | _RULEBREAKING_SYNDROME
    | _EXTERNALIZING_THEORETICAL
    | {
        "acts_immature_k",
        "destroys_others_things_k",
        "disobeys_parents_k",
        "stubborn_k",
        "hot_temper_k",
        "disobeys_at_school_k",
        "secretive_p",
        "social_problems_D_p",
        "easily_offended_p",
        "bdefs_lazy_p",
        "child_clear_feelings_p",
        "disagreeable_p",
        "blames_others_p",
        "prosocial_ss_p",
        "not_critical_others_p",
        "parent_depress_D_p",
        "bdefs_calm_down_p",
        "bdefs_consequences_p",
        "bdefs_distract_upset_p",
        "bdefs_explain_idea_p",
        "bdefs_explain_pt_p",
        "bdefs_explain_seq_p",
        "bdefs_impulsive_action_p",
        "bdefs_inconsistant_p",
        "bdefs_process_info_p",
        "bdefs_rechannel_p",
        "bdefs_sense_time_p",
        "bdefs_shortcuts_p",
        "bdefs_stop_think_p",
        "child_emotion_overwhelm_p",
        "child_feelings_attentive_p",
        "child_feelings_care_p",
        "child_feelings_know_p",
        "child_upset_acknowledge_p",
        "child_upset_angry_p",
        "child_upset_ashamed_p",
        "child_upset_bad_behavior_p",
        "child_upset_bad_esteem_p",
        "child_upset_bad_focus_p",
        "child_upset_better_p",
        "child_upset_control_p",
        "child_upset_embarrassed_p",
        "child_upset_emotions_overwhelm_p",
        "child_upset_fixation_p",
        "child_upset_guilty_p",
        "child_upset_irritated_p",
        "child_upset_longtime_better_p",
        "child_upset_longtime_p",
        "child_upset_lose_control_p",
        "child_upset_no_control_p",
        "child_upset_nothing_better_p",
        "child_upset_out_control_p",
        "child_upset_poor_concentrate_p",
        "child_upset_unproductive_p",
        "child_upset_weak_p",
        "doesnt_finish_p",
        "adhd_d_p",
        "sociable_p",
        "bis_behav_inhibition_ss_k",
    }
    | {"delta_doesnt_get_along_p", "delta_not_liked_p", "not_liked_p"}
    | {"conduct_physical_fights_present_PK", "bragadocious_p", "easily_jealous_p", "impulsive_p"}
    | {"demands_attention_p"}
    | {"hyperactive_p"}
)
CIRCULAR_EXCLUSIONS["external_D_p"] = {
    "across_exclude": _EXTERNAL_ACROSS,
    "within_exclude": {"Externalizing", "Child Externalizing"},
}

# 5. ADHD
# All nine CBCL ADHD DSM-5 scale items excluded; ODD/CD-adjacent items excluded
# due to inclusion in the scale's hyperactive-impulsive subscale definition.
# Child self-report attention/impulse items excluded for construct-level overlap.
_ADHD_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _ADHD_EXTRA_GENERAL
    | _DELTA_EXTERNALIZING_EXCLUDE
    | {
        "acts_young_p",
        "doesnt_finish_p",
        "cant_concentrate_p",
        "hyperactive_p",
        "impulsive_p",
        "easily_distracted_p",
        "loquacious_p",
        "goal_continuity_p",
    }
    | {
        "stubborn_p",
        "temper_tantrums_p",
        "social_problems_D_p",
        "disobedient_home_p",
        "disobedient_school_p",
        "breaks_rules_p",
        "poor_schoolwork_p",
    }
    | {
        "bad_grades",
        "grades_dropped",
        "delta_cant_concentrate_p",
        "delta_bad_grades",
        "bdefs_impulsive_action_p",
        "bdefs_sense_time_p",
    }
    | {
        "trouble_concentrating_k",
        "difficulty_finishing_tasks_k",
        "easily_distracted_k",
        "acts_immature_k",
        "destroys_others_things_k",
        "disobeys_parents_k",
        "up_lackperseverance_ss_k",
        "up_lackofplanning_ss_k",
        "stubborn_k",
        "hot_temper_k",
        "disobeys_at_school_k",
    }
)
for target in ["adhd_D_p", "depadhd_c", "asdadhd_c"]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _ADHD_ACROSS,
        "within_exclude": {"ADHD", "Child ADHD", "Externalizing", "Child Externalizing"},
        "across_exclude_t1t3t4": _ADHD_T1T3T4_EXTRA,
    }

# Child Only Deltas: needs T1T3T4 extras for child_upset + BDEFS items
CIRCULAR_EXCLUSIONS["delta_adhd_D_p"] = {
    # See delta_anxdisord_D_p note: base _ADHD_ACROSS excludes delta composites
    # but not the raw baseline ones. Add raw source + broadband externalizing +
    # the combined-class targets (depadhd_c, asdadhd_c) that include ADHD status.
    "across_exclude": _ADHD_ACROSS
    | {
        "adhd_D_p",  # raw baseline/endpoint source of the delta
        "external_D_p",  # broadband externalizing contains adhd_D_p
        "depadhd_c",  # latent class: combined depression+ADHD
        "asdadhd_c",  # latent class: combined ASD+ADHD
    },
    "within_exclude": {"ADHD", "Child ADHD", "Externalizing", "Child Externalizing"},
    "across_exclude_t1t3t4": _ADHD_T1T3T4_EXTRA,
}

# 6. Social Problems
# Targets (social_problems_D_p, not_liked_p, doesnt_get_along_p) shares
# enough item overlap that all three are mutually excluded across the group.
_SOCIAL_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | {
        "social_problems_D_p",
        "delta_social_problems_D_p",
        "not_liked_p",
        "regarded_weird_p",
        "difficulty_making_friends_p",
        "doesnt_get_along_p",
        "delta_not_liked_p",
        "delta_doesnt_get_along_p",
    }
    | {
        "clings_to_adults_p",
        "lonely_p",
        "easily_jealous_p",
        "paranoid_p",
        "Doesnt_finish_p",
        "gets_hurt_p",
        "teased_p",
        "clumsy_p",
        "prefer_younger_p",
        "speech_problems_p",
    }
    | {"shy_p", "prefer_alone_p", "disobedient_friends_p", "self_conscious_k"}
)
for target in [
    "social_problems_D_p",
    "not_liked_p",
    "doesnt_get_along_p",
    # Child Only Deltas
    "delta_social_problems_D_p",
]:
    CIRCULAR_EXCLUSIONS[target] = {"across_exclude": _SOCIAL_ACROSS, "within_exclude": set()}

# Child targets: secondary

# 7. Grade Outcomes
# bad_grades and grades_dropped are near-identical constructs; ADHD dropped given
# parents likely proxy with attention features.
# T1/T3/T4 extras: BDEFS executive-function items and ERC-P concentration items
# that operationalise the cognitive substrate of academic performance at later waves.
_BAD_GRADES_T1T3T4_EXTRA = {
    "bdefs_explain_idea_p",
    "bdefs_explain_pt_p",
    "bdefs_explain_seq_p",  # verbal/working-memory executive
    "bdefs_inconsistant_p",
    "bdefs_lazy_p",
    "bdefs_process_info_p",  # effortful-control executive
    "bdefs_rechannel_p",
    "bdefs_sense_time_p",
    "bdefs_shortcuts_p",  # time-management / task-switching executive
    "child_upset_bad_focus_p",
    "child_upset_poor_concentrate_p",  # ERC-P concentration under distress
    "child_upset_unproductive_p",  # task-completion under distress
    "finds_schoolboring_k",
    "enjoys_school_k",  # school disengagement proxies
}
CIRCULAR_EXCLUSIONS["bad_grades"] = {
    "across_exclude": {
        "bad_grades",
        "grades_dropped",
        "delta_bad_grades",
        "poor_schoolwork_p",
        "skips_school_p",
        "cant_concentrate_p",
        "disobedient_school_p",
        "doesnt_finish_p",
        "goal_continuity_p",
        "atschool_total_problems_ss_t",
        "atschool_attention_ss_t",
        "delta_cant_concentrate_p",
        "school_detension_suspension",
        "atschool_external_ss_t",
        "atschool_internal_ss_t",
        "delta_atschool_total_problems_ss_t",
    },
    "within_exclude": set(),
    "across_exclude_t1t3t4": _BAD_GRADES_T1T3T4_EXTRA,
}
# Child Only Deltas (shared dict reference: inherits T1T3T4 extras automatically)
CIRCULAR_EXCLUSIONS["delta_bad_grades"] = CIRCULAR_EXCLUSIONS["bad_grades"]

# 8. NIH Toolbox Composite Scores
# tb_total aggregates all TB scores.
# Constituent scores are mutually excluded within each composite.
# tasks measuring largely same general constructs excluded from each other
CIRCULAR_EXCLUSIONS["tb_flanker"] = {
    "across_exclude": {
        "tb_flanker",
        "tb_fluid",
        "tb_fluid_grouped",
        "tb_total",
        "lmt_mrt",
        "sst_mean_ssrt",
        "sst_median_ssrt",
        "sst_ssrt_int_est",
        "sst_ssrt_mean_est",
        "tb_pattern",  # Pattern Comparison Processing Speed (NIH): shared latent construct
    },
    "within_exclude": set(),
}
CIRCULAR_EXCLUSIONS["tb_reading"] = {
    # tb_reading and tb_picvocab are the two constituent scores of tb_cryst.
    "across_exclude": {"tb_reading", "tb_picvocab", "tb_cryst", "tb_total"},
    "within_exclude": set(),
}
CIRCULAR_EXCLUSIONS["tb_fluid"] = {
    # tb_fluid = tb_flanker + tb_pattern + tb_cardsort + tb_picture + tb_list
    "across_exclude": {
        "tb_fluid",
        "tb_fluid_grouped",
        "tb_total",
        "tb_flanker",
        "tb_cryst",
        "tb_pattern",
        "tb_cardsort",
        "tb_picture",
        "tb_list",
        "lmt_mrt",
        "tb_reading",
        "tb_picvocab",
    },
    "within_exclude": set(),
}
CIRCULAR_EXCLUSIONS["tb_cryst"] = {
    # tb_cryst = tb_reading + tb_picvocab
    "across_exclude": {"tb_cryst", "tb_total", "tb_reading", "tb_picvocab"},
    "within_exclude": set(),
}

# 8b. Verbal Episodic Memory (RAVLT)
# RAVLT short- and long-delay recall (total / repetitions / intrusions) all
# index the same auditory-verbal LTM construct, so all six subscales are
# mutually excluded. tb_picture (NIH Picture Sequence Memory) shares the
# episodic-memory construct via a different modality and is also excluded,
# along with composite parents (tb_fluid, tb_total).
CIRCULAR_EXCLUSIONS["ravlt_l_total"] = {
    "across_exclude": {
        "ravlt_l_total",
        "ravlt_l_repitition",
        "ravlt_l_intrusions",
        "ravlt_s_total",
        "ravlt_s_repitition",
        "ravlt_s_intrusions",
        "tb_picture",  # NIH Picture Sequence Memory: shared episodic-memory construct
        "tb_fluid",
        "tb_fluid_grouped",  # composite parent
        "tb_total",  # top-level composite
    },
    "within_exclude": set(),
}

# 8c. Visuospatial Mental Rotation (LMT)
# All LMT subscales (accuracy, efficiency, RT variants, correct-trial counts)
# index the same mental-rotation construct and are mutually excluded.
# Matrix Reasoning (mr_*) is the closest visuospatial-reasoning analogue and
# tb_pattern (NIH Pattern Comparison) shares a processing-speed component, so
# both families are excluded.
CIRCULAR_EXCLUSIONS["lmt_efficiency"] = {
    "across_exclude": {
        "lmt_efficiency",
        "lmt_accuracy",
        "lmt_correct_nt",
        "lmt_correct_mrt",
        "lmt_incorrect_mrt",
        "lmt_mrt",
        "mr_total",
        "mr_matrix",
        "mr_serial",  # Matrix Reasoning: shared visuospatial-reasoning construct
        "tb_pattern",  # NIH Pattern Comparison: shared processing-speed component
    },
    "within_exclude": set(),
}

# 8d. Stop Signal Task (SST)
# All SST behavioral summaries and computational-model parameters index
# the same response-inhibition construct (or its variability/uncertainty),
# so the full SST family is mutually excluded. tb_flanker (NIH inhibitory
# control via conflict monitoring) shares the response-inhibition construct
# via a different paradigm and is excluded, along with the fluid composite
# parents and lmt_mrt (shared RT/speed confound).
CIRCULAR_EXCLUSIONS["sst_mean_ssrt"] = {
    "across_exclude": {
        # Primary SSRT estimates
        "sst_mean_ssrt",
        "sst_median_ssrt",
        "sst_ssrt_mean_est",
        "sst_ssrt_int_est",
        # QC flag
        "sst_acceptable_performance",
        # Computational-model parameters (BEESTS / general SST modeling)
        "sst_theta0",
        "sst_theta1",
        "sst_mu",
        "sst_aS",
        "sst_dS",
        "sst_gamma0",
        "sst_gamma1",
        "sst_dG",
        "sst_aG1",
        "sst_aG2",
        "sst_kappa0",
        "sst_xM",
        "sst_sM",
        "sst_bG",
        "sst_pp",
        "sst_factor1",
        "sst_factor2",
        "sst_factor3",
        # Trial-level central-tendency summaries
        "sst_mean_PDR",
        "sst_median_PDR",
        "sst_mean_SD",
        "sst_median_SD",
        "sst_mean_SDr",
        "sst_median_SDr",
        "sst_mean_PDRg",
        "sst_median_PDRg",
        "sst_mean_betaS",
        "sst_median_betaS",
        "sst_mean_bS2",
        "sst_median_bS2",
        "sst_mean_absdelta",
        "sst_median_absdelta",
        # Variance / SD / CV summaries
        "sst_pdrV",
        "sst_pdrSD",
        "sst_pdrCV",
        "sst_sdRV",
        "sst_sdSD",
        "sst_sdCV",
        "sst_sdrRV",
        "sst_sdrSD",
        "sst_sdrCV",
        "sst_pdrgV",
        "sst_pdrgSD",
        "sst_pdrgCV",
        "sst_betasV",
        "sst_betasSD",
        "sst_betasCV",
        "sst_absdeltaRV",
        "sst_absdeltaSD",
        "sst_absdeltaCV",
        "sst_bs2V",
        "sst_bs2SD",
        "sst_bs2CV",
        # Cross-paradigm overlap
        "tb_flanker",  # NIH inhibitory control: shared response-inhibition construct
        "tb_fluid",
        "tb_fluid_grouped",  # composite parent (contains tb_flanker)
        "tb_total",  # top-level composite
        "lmt_mrt",  # shared RT / processing-speed confound
    },
    "within_exclude": set(),
}

# 8e. EN-back Working Memory
# All n-back behavioral metrics (0-/2-back accuracy, RT, and 2-back signal-
# detection indices across positive/neutral/negative valence conditions) index
# the same emotional-WM construct and are mutually excluded. tb_list (NIH
# List Sorting Working Memory) shares the WM construct via a non-emotional
# paradigm and is also excluded, along with composite parents.
CIRCULAR_EXCLUSIONS["nb_correct_nt_2back"] = {
    "across_exclude": {
        # 0-back / 2-back number correct & mean RT (overall + by valence)
        "nb_correct_nt_2back",
        "nb_correct_mrt_2back",
        "nb_correct_nt",
        "nb_correct_mrt",
        "nb_correct_nt_pos",
        "nb_correct_mrt_pos",
        "nb_correct_nt_neutral",
        "nb_correct_mrt_neutral",
        "nb_correct_nt_neg",
        "nb_correct_mrt_neg",
        # 2-back signal-detection metrics by valence
        "nb2_accuracy_pos",
        "nb2_resp_bias_pos",
        "nb2_D_prime_pos",
        "nb2_accuracy_neg",
        "nb2_resp_bias_neg",
        "nb2_D_prime_neg",
        # QC flag
        "nb_acceptable_performance",
        # Cross-paradigm overlap
        "tb_list",  # NIH List Sorting Working Memory: shared WM construct
        "tb_fluid",
        "tb_fluid_grouped",  # composite parent (contains tb_list)
        "tb_total",  # top-level composite
    },
    "within_exclude": set(),
}

# 9. Thought Disorder
# All nine CBCL Thought Problems syndrome items excluded; this subscale has no
# overlapping items with internalizing/externalizing syndromes, so within_exclude is empty.
CIRCULAR_EXCLUSIONS["thought_disorder_D_p"] = {
    "across_exclude": {
        "thought_disorder_D_p",
        "delta_thought_disorder_D_p",
        "strange_behavior_p",
        "strange_ideas_p",
        "hears_voices_p",
        "sees_things_p",
        "repetitive_acts_p",
        "hoarding_p",
        "daydreams_p",
        "stares_blankly_p",
        "sleep_walks_p",
        "nervous_twitching_p",
        "picks_things_p",
        "plays_genitals_public_p",
    },
    "within_exclude": set(),
}

# 10. Child OCD
# Additional exploratory target
# CBCL OCD scale comprises cbcl_q09 (obsessions), cbcl_q66 (compulsions), cbcl_q83 (hoarding).
# Thought Problems broadband excluded as its constituent items substantially overlap with OCD.
# Body-focused repetitive behavior and tic-adjacent items excluded for redundancy/overlap
CIRCULAR_EXCLUSIONS["ocd_D_p"] = {
    "across_exclude": (
        {"ocd_D_p", "delta_ocd_D_p"}
        | {"obsessions_p", "repetitive_acts_p", "hordes_p"}  # CBCL OCD scale items (q09, q66, q83)
        | {"thought_disorder_D_p", "delta_thought_disorder_D_p"}  # broadband containing OCD scale
        | {
            "picks_skin_p",
            "nervous_twitching_p",
            "picks_things_p",  # body-focused repetitive / tic-adjacent
            "blank_stare_p",
            "strange_ideas_p",
            "strange_behavior_p",
        }
        | {"obsessions_present_B_p"}  # KSADS direct-construct item
        | _DEPRESSION_SBT
    ),
    "within_exclude": set(),
}

# 11. Child Obsessions
# cbcl_q09 ("Can't get his/her mind off certain thoughts; obsessions") is a
# constituent of the CBCL OCD scale and Thought Problems broadband. OCD co-
# constituents (compulsions_p, hordes_p) and body-focused repetitive/tic-adjacent
# items excluded for phenotypic proximity.
# hyperactive_p excluded: parent raters cannot cleanly distinguish restless/driven
# behavior from obsessive mental preoccupation.
CIRCULAR_EXCLUSIONS["obsessions_p"] = {
    "across_exclude": (
        {"obsessions_p", "delta_obsessions_p"}
        | {"compulsions_p", "repetitive_acts_p", "hordes_p"}  # OCD scale co-constituents (q66, q83)
        | {"ocd_D_p", "delta_ocd_D_p"}  # OCD composite sums this item
        | {"thought_disorder_D_p", "delta_thought_disorder_D_p"}  # Thought Problems broadband contains q09
        | {
            "nervous_twitching_p",
            "picks_skin_p",
            "picks_things_p",
            "strange_behavior_p",
            "strange_ideas_p",
            "blank_stare_p",
        }  # body-focused repetitive / tic-adjacent
        | {"obsessions_present_B_p"}  # KSADS direct-construct item
        | {"hyperactive_p"}  # theoretical overlap: restless/driven vs obsessive preoccupation
        | {"parent_picks_skin_p"}  # body-focused repetitive behavior proxy (parent analogue)
        | {
            "narrow_interests_p",
            "concentration_on_parts_p",  # ASD cognitive style: rater conflation with OCD/obsessive style
            "sensory_sensitivity_p",
            "avoids_eyecontact_p",
            "bad_conversational_flow_p",
            "regarded_weird_p",
        }
        | {"mania_7up_ss_k"}  # racing/driven cognition indistinguishable from obsessive ideation
        | _DEPRESSION_SBT
    ),
    "within_exclude": set(),
}

# 12. Child Perfectionism
# cbcl_q32 ("Feels he/she has to be perfect") is a constituent of the CBCL
# Anxious/Depressed syndrome (→ anxdisord_D_p, internal_D_p) and the SBT
# Perfectionism/Achievement subtype. Co-constituents of sbt_perfectionism
# (fears_being_bad_p, poor_schoolwork_p_reverse, confused_p) are excluded to
# prevent indirect reconstruction. Self-evaluative items (easily_embarrassed_p,
# self_conscious_k) excluded for construct overlap with evaluative concern
# perfectionism.
# hyperactive_p excluded: parent raters cannot cleanly distinguish restless/driven
# behavior from perfectionistic rigidity.
CIRCULAR_EXCLUSIONS["perfectionist_p"] = {
    "across_exclude": (
        {"perfectionist_p", "delta_perfectionist_p"}
        | {
            "fears_being_bad_p",
            "confused_p",
            "worries_p",
            "poor_schoolwork_p",
            "poor_schoolwork_p_reverse",
        }  # sbt_perfectionism_achievement co-constituents
        | {"sbt_perfectionism_achievement"}  # SBT subtype that sums this item directly
        | {"easily_embarrassed_p", "self_conscious_k"}  # evaluative-concern perfectionism overlap
        | {"anxdisord_D_p", "delta_anxdisord_D_p"}  # Anxious/Depressed syndrome contains this item
        | {"internal_D_p", "delta_internal_D_p"}  # Internalizing broadband (via anxdep subscale)
        | {"hyperactive_p"}  # theoretical overlap: driven/restless vs perfectionistic rigidity
        | {"mania_7up_ss_k"}  # perfectionistic striving vs manic grandiosity indistinguishable at item level
        | _DEPRESSION_SBT
    ),
    "within_exclude": {"Anxiety", "Child Anxiety"},
}

# 13. Child Impulsivity
# cbcl_q41 ("Acts without thinking") is a constituent of the CBCL ADHD DSM-5 scale
# (hyperactive/impulsive subscale). The exclusion set is intentionally narrower than
# _ADHD_ACROSS: acts_young_p and goal_continuity_p are excluded because they belong
# to the same ADHD subscale grouping and share variance with impulsive dyscontrol.
# _DEPRESSION_SBT excluded because sbt_emotional_dysregulation aggregates
# impulse-control failure items, creating indirect circularity.
_IMPULSIVE_T1T3T4_EXTRA = {
    "bdefs_impulsive_action_p",
    "bdefs_stop_think_p",
    "bdefs_shortcuts_p",
    "child_upset_bad_behavior_p",
    "child_upset_lose_control_p",
    "child_upset_out_control_p",
    "child_upset_no_control_p",
}
CIRCULAR_EXCLUSIONS["impulsive_p"] = {
    "across_exclude": (
        {"impulsive_p", "delta_impulsive_p"}
        | {
            "cant_concentrate_p",
            "doesnt_finish_p",
            "hyperactive_p",
            "easily_distracted_p",
            "loquacious_p",
            "acts_young_p",
            "goal_continuity_p",
        }
        | {"adhd_D_p", "delta_adhd_D_p", "external_D_p", "delta_external_D_p", "depadhd_c", "asdadhd_c"}
        | {
            "demands_attention_p",
            "temper_tantrums_p",
            "argues_p",
            "screams_p",
            "breaks_rules_p",
            "no_guilt_p",
            "stubborn_p",
            "disobedient_home_p",
            "disobedient_school_p",
        }
        | {
            "social_problems_D_p",
            "poor_schoolwork_p",
            "bad_grades",
            "grades_dropped",
            "delta_cant_concentrate_p",
            "delta_bad_grades",
            "bdefs_impulsive_action_p",
            "bdefs_sense_time_p",
        }
        | {
            "trouble_concentrating_k",
            "difficulty_finishing_tasks_k",
            "easily_distracted_k",
            "acts_immature_k",
            "destroys_others_things_k",
            "disobeys_parents_k",
            "up_lackperseverance_ss_k",
            "up_lackofplanning_ss_k",
            "stubborn_k",
            "hot_temper_k",
            "disobeys_at_school_k",
        }
        | _DELTA_EXTERNALIZING_EXCLUDE
        | _DEPRESSION_SBT
    ),
    "within_exclude": {"ADHD", "Child ADHD", "Externalizing", "Child Externalizing"},
    "across_exclude_t1t3t4": _IMPULSIVE_T1T3T4_EXTRA,
}

# 14. Child Compulsivity
# cbcl_q66 ("Repeats certain acts over and over; compulsions") is a CBCL OCD scale
# constituent. repetitive_acts_p is excluded as a likely dataset alias for the same
# item. Thought Problems broadband is excluded because it sums q66 directly.
# Body-focused repetitive and tic-adjacent items excluded for phenotypic proximity.
# hyperactive_p excluded: parent raters cannot cleanly distinguish restless/driven
# behavior from compulsive repetition.
# T1/T3/T4 extras: ERC-P emotional-fixation/control items and BDEFS executive-function
# items that operationalise perseverative/stuck cognition at the later waves.
_COMPULSIONS_T1T3T4_EXTRA = {
    "child_upset_fixation_p",  # emotional fixation/perseveration
    "child_upset_longtime_p",
    "child_upset_longtime_better_p",  # prolonged emotional stuck state
    "child_upset_nothing_better_p",  # inability to disengage from distress
    "bdefs_stop_think_p",
    "bdefs_rechannel_p",  # executive inhibition / shifting
    "bdefs_consequences_p",  # repetitive behavior despite aversive consequences
}
CIRCULAR_EXCLUSIONS["compulsions_p"] = {
    "across_exclude": (
        {"compulsions_p", "delta_compulsions_p"}
        | {"repetitive_acts_p"}  # dataset alias for cbcl_q66
        | {"obsessions_p", "impulsive_p", "hordes_p"}  # CBCL OCD scale co-constituents (q09, q83)
        | {"ocd_D_p", "delta_ocd_D_p"}  # OCD composite sums this item directly
        | {"thought_disorder_D_p", "delta_thought_disorder_D_p"}  # Thought Problems broadband contains q66
        | {
            "nervous_twitching_p",
            "picks_skin_p",
            "picks_things_p",
            "strange_behavior_p",
            "strange_ideas_p",
            "blank_stare_p",
        }  # body-focused repetitive / tic-adjacent
        | {"obsessions_present_B_p"}  # KSADS direct-construct item
        | {
            "goal_continuity_p",
            "up_lackperseverance_ss_k",
            "up_lackofplanning_ss_k",
        }  # stickiness/perseveration proxies
        | {"hyperactive_p"}  # theoretical overlap: restless/driven vs compulsive repetition
        | {"parent_picks_skin_p"}  # body-focused repetitive behavior proxy (parent analogue)
        | {
            "narrow_interests_p",
            "concentration_on_parts_p",  # ASD cognitive style: rater conflation with compulsive/OCD style
            "sensory_sensitivity_p",
            "avoids_eyecontact_p",
            "bad_conversational_flow_p",
            "regarded_weird_p",
        }
        | _DEPRESSION_SBT
    ),
    "within_exclude": set(),
    "across_exclude_t1t3t4": _COMPULSIONS_T1T3T4_EXTRA,
}

# Shared batch-wide exclusions  (Parent Social/Impulsivity/Compulsivity batch)
# Applied to all 5 targets in the Parent Social/Impulsivity/Compulsivity
# bundle (parent_avoidant_person_D_p, parent_not_liked_by_others_p,
# parent_impulsive_p, parent_repeats_acts_p, parent_obsessive_thoughts_p).
# These broad ASR composites and the mood-swing item are removed from every
# predictor pool in this batch even where item-level circularity is absent.
# Rationale: composite-subsumption / reconstruction-via-subtraction risk
# (Internalizing + Externalizing + Other = Total Problems), construct-overlap
# at the broadband level, and Mood Swings as a transdiagnostic affect-
# dysregulation confound that cuts across all 5 target dimensions.
_PARENT_SOCIAL_IMPCOMP_BATCH_EXCLUDE = {
    # ASR Externalizing broadband + sub-syndromes
    "parent_external_D_p",
    "parent_aggressive_D_p",
    "parent_antisocial_D_p",
    "parent_hyperactive_D_p",
    # ASR Attention/ADHD composites
    "parent_attention_D_p",
    "parent_adhd_D_p",
    # ASR Depression Syndrome composite (parent self-report)
    "parent_depress_D_p",
    "parent_depressed_p",
    # ASR 87 Mood Swings: transdiagnostic affect-dysregulation confound
    "parent_sudden_mood_changes_p",
    # Delta derivatives (future-proof; no-op if not present in predictor pool)
    "delta_parent_external_D_p",
    "delta_parent_aggressive_D_p",
    "delta_parent_antisocial_D_p",
    "delta_parent_hyperactive_D_p",
    "delta_parent_attention_D_p",
    "delta_parent_adhd_D_p",
    "delta_parent_depress_D_p",
    "delta_parent_sudden_mood_changes_p",
}

# Parent targets: primary

# 15. Parent Depression
# ASR internalizing broadband and subscale composites excluded; ASR items that
# operationalise DSM MDD criteria (appetite, sleep, energy, anhedonia, hopelessness)
# are individually excluded. Change scores for mood and somatic subscales included.
_PARENT_DEP_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _PARENT_GENERAL_MOOD_EXCLUDE
    | {
        "parent_depressed_p",
        "parent_doesnt_eat_well_p",
        "parent_depress_D_p",
        "parent_self_harm_p",
        "parent_suicidal_thoughts_p",
    }
    | {
        "parent_anxdep_D_p",
        "parent_internal_D_p",
        "parent_external_D_p",
        "parent_somatic_problems_D_p",
        "parent_depressed_mood_B_p",
    }  # ASR broadband/subscale composites
    | {"parent_anhedonia_B_p", "parent_dysthymia_B_p"}  # DSM MDD core criterion / depressive disorder diagnosis
    | {"parent_paranoid_p"}  # paranoid ideation co-occurs with psychotic depression; rater conflation
    | {
        "parent_enjoys_little_p",
        "parent_low_energy_p",
        "parent_tired_no_reason_p",
        "parent_avoidant_person_D_p",
        "parent_attention_D_p",
        "parent_cries_a_lot_p",
        "parent_lonely_p",
        "parent_aggressive_D_p",
        "parent_stubborn_irritable_p",
        "parent_feels_unsuccessful_p",
        "parent_happy_person_p",
        "parent_obsessive_thoughts_p",
        "parent_restless_p",
        "parent_worries_about_future_p",
        "parent_planning_trouble_p",
        "parent_personal_strength_D_p",
        "parent_worries_a_lot_p",
        "parent_decision_trouble_p",
        "delta_parent_stubborn_irritable_p",
        "parent_secretive_p",
        "parent_feels_inferior_p",
        "parent_sudden_mood_changes_p",
        "parent_antisocial_D_p",
        "parent_hyperactive_D_p",
        "parent_adhd_D_p",
        "parent_therapy_emo",
        "family_emotionprob_p",
        "parent_strange_thoughts_p",
        "parent_intrusive_thoughts_D_p",
        "delta_parent_worries_about_future_p",
        "parent_self_conscious_p",
        "parent_clumsy_p",
        "parent_sleep_trouble_p",
    }
    | {
        "delta_parent_depressed_p",
        "delta_parent_feels_overwhelmed_p",
        "delta_parent_worries_a_lot_p",
        "delta_parent_internal_D_p",
        "tb_picvocab",
        "delta_parent_somatic_problems_D_p",
        "delta_parent_stubborn_irritable_p",
        "delta_parent_planning_trouble_p",
        "delta_parent_sleep_trouble_p",
    }
)
_PARENT_DEP_WITHIN = {"Parent Mood Issues", "Parent Anxiety", "Parent Anxiety and Fear-Related Issues"}

for target in [
    "parent_depress_D_p",
    "parent_depressed_p",
    "delta_parent_depress_D_p",
    "delta_parent_depressed_D_p",
    "parent_dep_onset_rci_2.3",
    "parent_dep_onset_rci_1.96",
    "parent_dep_remission_rci_2.3",
    "parent_dep_remission_rci_1.96",
    "parent_dep_increase_2sd",
    "parent_dep_increase_1.5sd",
    "parent_dep_decrease_2sd",
    "parent_dep_decrease_1.5sd",
    "top_10_depression_parent",
    "top_5_depression_parent",
    # Parent Only Deltas: mood-domain change scores
    "delta_parent_sleep_trouble_p",
    "delta_parent_feels_unloved_p",
]:
    CIRCULAR_EXCLUSIONS[target] = {"across_exclude": _PARENT_DEP_ACROSS, "within_exclude": _PARENT_DEP_WITHIN}

# delta_parent_depressed_p gets additional SST exclusion (sst_bG identified as
# circular predictor for depression change scores via inhibitory-control confound).
CIRCULAR_EXCLUSIONS["delta_parent_depressed_p"] = {
    "across_exclude": _PARENT_DEP_ACROSS | {"sst_bG"},
    "within_exclude": _PARENT_DEP_WITHIN,
}

# 16. Parent Anxiety
# ASR anxiety subscale items excluded individually; obsessive-compulsive and
# avoidance items included due to their role in generalized/social anxiety constructs.
_PARENT_ANX_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _PARENT_GENERAL_MOOD_EXCLUDE
    | {"parent_anxdisord_D_p", "parent_anxdep_D_p", "parent_internal_D_p", "parent_sleep_trouble_p"}
    | {
        "parent_fearful_or_anxious_p",
        "parent_specific_fears_p",
        "parent_depress_D_p",
        "parent_restless_p",
        "parent_fear_of_bad_thoughts_p",
        "parent_worries_about_future_p",
        "delta_parent_depressed_p",
        "parent_worries_about_family_p",
        "parent_worries_a_lot_p",
        "parent_avoidant_person_D_p",
        "parent_aggressive_D_p",
        "parent_happy_person_p",
        "parent_stubborn_irritable_p",
        "parent_therapy_emo",
        "family_emotionprob_p",
        "parent_intrusive_thoughts_D_p",
        "parent_personal_strength_D_p",
        "parent_strange_thoughts_p",
        "parent_self_conscious_p",
        "parent_relationship_concerns_p",
    }
    | {"parent_obsessive_thoughts_p", "parent_repeats_acts_p"}
    | {
        "parent_social_anxiety_disorder_B_p",
        "parent_intrusive_thoughts_D_p",
    }  # direct anxiety disorder construct measures
    | {"parent_paranoid_p"}  # fearful/paranoid ideation indistinguishable from severe anxiety
    | {
        "delta_parent_worries_a_lot_p",
        "delta_parent_worries_about_future_p",
        "delta_parent_worries_about_family_p",
        "delta_parent_feels_overwhelmed_p",
        "delta_parent_sleep_trouble_p",
    }
)
_PARENT_ANX_WITHIN = {"Parent Anxiety", "Parent Anxiety and Fear-Related Issues", "Parent Mood Issues"}

for target in [
    "parent_anxdisord_D_p",
    "parent_worries_a_lot_p",
    "delta_parent_worries_a_lot_p",
    # Parent Only Deltas: anxiety-domain change scores
    "delta_parent_worries_about_family_p",
    "delta_parent_worries_about_future_p",
]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _PARENT_ANX_ACROSS,
        "within_exclude": _PARENT_ANX_WITHIN,
    }

# parent_anxdisord_D_p-specific additions: avoidant coping, inferiority, and depressed
# affect are excluded as ASR items with construct overlap that parent raters cannot
# readily distinguish from anxiety disorder symptomatology (see S1.1 audit).
CIRCULAR_EXCLUSIONS["parent_anxdisord_D_p"]["across_exclude"] = _PARENT_ANX_ACROSS | {
    "parent_avoidant_person_D_p",
    "parent_feels_inferior_p",
    "parent_depressed_p",
}

# 17. Parent ADHD
# Both inattentive and hyperactive/impulsive ASR subscale items are excluded, along
# with behavioral-consequence items (risky decisions, time management) that are
# functionally dependent on executive dysregulation rather than independent predictors.
_PARENT_ADHD_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | {"parent_adhd_D_p", "parent_attention_D_p", "parent_hyperactive_D_p"}
    | {
        "parent_concentration_trouble_p",
        "parent_forgetful_p",
        "parent_planning_trouble_p",
        "parent_not_good_at_details_p",
        "parent_max_effort_p",
        "parent_disorganized_p",
        "parent_depress_D_p",
        "parent_internal_D_p",
        "parent_loses_things_p",
        "parent_decision_trouble_p",
        "parent_priority_trouble_p",
        "parent_doesnt_finish_tasks_p",
    }
    | {
        "parent_hyperactive_p",
        "parent_talks_too_much_p",
        "parent_restless_p",
        "parent_impulsive_p",
        "parent_intrusive_thoughts_D_p",
        "parent_easily_bored_p",
        "parent_tardy_often_p",
    }
    | {
        "parent_stubborn_irritable_p",
        "parent_behavior_changeable_p",
        "parent_strange_behavior_p",
        "parent_antisocial_D_p",
        "family_organisation_ss_p",
        "parent_money_management_trouble_p",
        "parent_personal_strength_D_p",
        "parent_attention_seeking_p",
        "parent_risky_decisions_p",
        "parent_drives_too_fast_p",
    }
    | {
        "delta_parent_concentration_trouble_p",
        "delta_parent_poor_work_performance_p",
        "delta_parent_planning_trouble_p",
        "delta_parent_stubborn_irritable_p",
    }
    | {
        "parent_aggressive_D_p",
        "parent_clumsy_p",
        "parent_impulsive_p",  # ASR items with construct overlap
        "parent_confused_p",
        "parent_external_D_p",
    }  # ASR externalizing broadband aggregates ADHD items
    | {"parent_hot_temper_p", "parent_yells_a_lot_p"}  # emotional dysregulation / ADHD emotional impulsivity subdomain
    | {
        "parent_destroys_own_things_p",
        "parent_destroys_others_things_p",
    }  # downstream ADHD impulse-dyscontrol consequences
    | {"parent_elevated_mood_B_p"}  # mania vs ADHD hyperactivity diagnostic ambiguity
    | {"parent_sees_things_p", "parent_hears_voices_p"}  # psychosis items misclassified in cognitive/attention category
)
for target in [
    "parent_adhd_D_p",
    "parent_concentration_trouble_p",
    # Parent Only Deltas: ADHD-domain change score
    "delta_parent_concentration_trouble_p",
]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _PARENT_ADHD_ACROSS,
        "within_exclude": {"Parent Cognitive and Attention Issues", "Parent Impulsivity and Behavior Regulation"},
    }

# Parent targets: secondary

# 18. Parent Social Quality
_PARENT_SOCIAL_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | {
        "parent_bad_relationships_p",
        "parent_bad_family_relationship_p",
        "parent_not_liked_by_others_p",
        "parent_friendship_trouble_p",
        "parent_external_D_p",
        "parent_bad_opposite_sex_relationship_p",
        "parent_meets_family_duties_p",
        "parent_enjoys_little_p",
        "parent_feels_unloved_p",
        "parent_depress_D_p",
        "parent_internal_D_p",
        "parent_clowns_or_shows_off_p",
        "parent_teases_others_p",
        "parent_antisocial_D_p",
        "parent_stands_up_rights_p",
        "parent_associates_with_trouble_p",
        "parent_avoidant_person_D_p",
        "parent_therapy_emo",
        "family_emotionprob_p",
        "parent_intrusive_thoughts_D_p",
        "parent_aggressive_D_p",
        "parent_strange_thoughts_p",
        "parent_prefers_older_people_p",
    }
    | {"parent_prefers_to_be_alone_p", "parent_shy_or_timid_p", "parent_avoids_talking_p", "parent_speech_problems_p"}
    | {
        "delta_parent_bad_relationships_p",
        "delta_parent_bad_family_relationship_p",
        "delta_parent_not_liked_by_others_p",
        "parent_personal_strength_D_p",
        "delta_parent_friendship_trouble_p",
        "delta_parent_meets_family_duties_p",
    }
    | _PARENT_SOCIAL_IMPCOMP_BATCH_EXCLUDE  # batch-wide broad composites + mood-swings
)
_PARENT_SOCIAL_WITHIN = {"Parent Social Functioning", "Parent Interpersonal Relationships and Social Functioning"}

for target in [
    "parent_bad_relationships_p",
    "parent_not_liked_by_others_p",
    "parent_friendship_trouble_p",
    "parent_bad_family_relationship_p",
    # Parent Only Deltas: social-domain change scores
    "delta_parent_not_liked_by_others_p",
    "delta_parent_bad_relationships_p",
]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _PARENT_SOCIAL_ACROSS,
        "within_exclude": _PARENT_SOCIAL_WITHIN,
    }

# 19. Parent Happiness / Wellbeing
# parent_happy_person_p is an inverse-mood item; parent depression and internalizing
# composites are excluded as they represent the same latent dimension in reverse.
_PARENT_HAPPY_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _PARENT_GENERAL_MOOD_EXCLUDE
    | {
        "parent_happy_person_p",
        "parent_enjoys_little_p",
        "parent_doesnt_eat_well_p",
        "parent_therapy_emo",
        "parent_internal_D_p",
        "family_emotionprob_p",
        "parent_strange_thoughts_p",
        "parent_avoidant_person_D_p",
        "parent_self_conscious_p",
        "parent_intrusive_thoughts_D_p",
        "parent_self_harm_p",
    }
    | {
        "parent_depressed_p",
        "parent_internal_D_p",
        "parent_depress_D_p",
        "parent_anxdep_D_p",
        "parent_somatic_problems_D_p",
    }
    | {
        "delta_parent_depressed_p",
        "parent_personal_strength_D_p",
        "delta_parent_feels_overwhelmed_p",
        "delta_parent_internal_D_p",
        "delta_parent_somatic_problems_D_p",
    }
)
CIRCULAR_EXCLUSIONS["parent_happy_person_p"] = {
    "across_exclude": _PARENT_HAPPY_ACROSS,
    "within_exclude": {"Parent Mood Issues", "Parent Anxiety", "Parent Anxiety and Fear-Related Issues"},
}

# 20. Parent Suicidal Thoughts
_PARENT_SUICIDAL_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _PARENT_GENERAL_MOOD_EXCLUDE
    | {
        "parent_suicidal_thoughts_p",
        "parent_self_harm_p",
        "parent_depressed_p",
        "parent_feels_unloved_p",
        "parent_depress_D_p",
        "parent_feels_overwhelmed_p",
        "parent_therapy_emo",
        "parent_intrusive_thoughts_D_p",
        "family_emotionprob_p",
        "parent_strange_thoughts_p",
        "parent_fear_of_bad_thoughts_p",
        "parent_self_conscious_p",
        "parent_doesnt_eat_well_p",
        "parent_sudden_mood_changes_p",
        "parent_obsessive_thoughts_p",
        "parent_aggressive_D_p",
        "parent_avoidant_person_D_p",
        "parent_internal_D_p",
        "parent_external_D_p",
        "parent_anxdep_D_p",
        "parent_depressed_mood_B_p",
        "parent_happy_person_p",
        "delta_parent_internal_D_p",
        "parent_tired_no_reason_p",
        "parent_personal_strength_D_p",
        "parent_confused_p",
    }
    | {"delta_parent_depressed_p", "delta_parent_feels_overwhelmed_p", "delta_parent_feels_unloved_p"}
    | {"parent_suicide_past_B_p"}  # past suicidal act is near-equivalent to current ideation (criterion 1/2)
    | {
        "parent_dysthymia_B_p",
        "parent_anhedonia_B_p",
    }  # depressive disorder features causally upstream of suicidal ideation
    | {"parent_paranoid_p"}  # paranoid ideation co-present with suicidality; rater conflation
)
CIRCULAR_EXCLUSIONS["parent_suicidal_thoughts_p"] = {
    "across_exclude": _PARENT_SUICIDAL_ACROSS,
    "within_exclude": {"Parent Mood Issues", "Parent Anxiety", "Parent Anxiety and Fear-Related Issues"},
}

# 21. Parent Impulsivity
# Intentionally narrower than parent_adhd_D_p: inattentive-subscale items
# (concentration, planning, forgetfulness) are retained because they do not
# directly constitute impulsivity. Only the hyperactive/impulsive subscale, its
# behavioral-consequence items, and the ADHD broadband composites that aggregate
# this item are excluded.
CIRCULAR_EXCLUSIONS["parent_impulsive_p"] = {
    "across_exclude": (
        {"parent_impulsive_p", "delta_parent_impulsive_p"}
        | {
            "parent_hyperactive_p",
            "parent_talks_too_much_p",
            "parent_restless_p",
            "parent_concentration_trouble_p",
            "parent_external_D_p",
            "parent_aggressive_D_p",
            "parent_easily_bored_p",
            "parent_tardy_often_p",
        }  # ASR hyperactive/impulsive subscale co-constituents
        | {
            "parent_risky_decisions_p",
            "parent_personal_strength_D_p",
            "parent_drives_too_fast_p",
            "parent_money_management_trouble_p",
            "parent_intrusive_thoughts_D_p",
            "parent_attention_seeking_p",
        }  # downstream impulsivity consequences
        | {
            "parent_stubborn_irritable_p",
            "parent_behavior_changeable_p",
            "delta_parent_stubborn_irritable_p",
        }  # emotional dysregulation overlap
        | {"parent_adhd_D_p", "parent_hyperactive_D_p"}  # ASR composites that aggregate this item
        | {
            "parent_hot_temper_p",
            "parent_yells_a_lot_p",  # emotional/behavioral impulsivity proxies
            "parent_louder_than_others_p",
        }
        | {
            "parent_destroys_own_things_p",
            "parent_destroys_others_things_p",  # impulse-dyscontrol consequences
            "parent_illegal_behavior_p",
        }
        | {"parent_no_guilt_p"}  # antisocial/ADHD emotional blunting overlap
        | {"parent_strange_behavior_p"}  # disinhibited acts: impulsivity vs thought disorder (conservative)
        | _PARENT_SOCIAL_IMPCOMP_BATCH_EXCLUDE  # batch-wide broad composites + mood-swings
    ),
    "within_exclude": {"Parent Impulsivity and Behavior Regulation"},
}

# 22. Parent Substance Use
# ASR substance use frequency items. Tobacco and alcohol are mutually excluded
# across each other's entries: they occupy the same Parent Substance Use feature
# category, and their high co-occurrence introduces near-circularity in within-category
# models even absent a direct item-composite relationship.
_SUBSTANCE_SHARED = {
    "parent_druguse_other_p",
    "parent_rulebreaking_D_p",
    "parent_external_D_p",
    "parent_aggressive_D_p",
    "parent_intrusive_thoughts_D_p",
    "delta_parent_rulebreaking_D_p",
    "delta_parent_external_D_p",
}
_SUBSTANCE_WITHIN = {"Parent Drug Use", "Parent Externalizing"}

CIRCULAR_EXCLUSIONS["parent_tobacco_use_frequency_p"] = {
    "across_exclude": (
        {"parent_tobacco_use_frequency_p", "cigs_during_pregnancy_p", "delta_parent_tobacco_use_frequency_p"}
        | {"parent_tobacco_use_p"}  # binary tobacco use item: near-circular with frequency
        | {"parent_drinks_frequency_p", "delta_parent_drinks_frequency_p"}
        | _SUBSTANCE_SHARED
    ),
    "within_exclude": _SUBSTANCE_WITHIN,
}
CIRCULAR_EXCLUSIONS["parent_drinks_frequency_p"] = {
    "across_exclude": (
        {"parent_drinks_frequency_p", "delta_parent_drinks_frequency_p"}
        | {"parent_tobacco_use_frequency_p", "delta_parent_tobacco_use_frequency_p"}
        | {
            "parent_drinks_too_much_p",
            "parent_drunk_days_p",  # alcohol quantity/severity items: near-circular with frequency
            "parent_alcohol_excess_p",
            "parent_alcohol_freq_p",
            "parent_alcohol_days_drunk_p",
        }
        | _SUBSTANCE_SHARED
    ),
    "within_exclude": _SUBSTANCE_WITHIN,
}
CIRCULAR_EXCLUSIONS["parent_drunk_days_p"] = {
    "across_exclude": (
        {"parent_drunk_days_p"}
        | {"parent_drinks_frequency_p", "delta_parent_drinks_frequency_p"}
        | {"parent_tobacco_use_frequency_p", "delta_parent_tobacco_use_frequency_p"}
        | {
            "parent_drinks_too_much_p",
            "parent_alcohol_excess_p",  # alcohol severity items: near-circular with days drunk
            "parent_alcohol_freq_p",
            "parent_alcohol_days_drunk_p",
        }
        | _SUBSTANCE_SHARED
    ),
    "within_exclude": _SUBSTANCE_WITHIN,
}

# Other targets

# 22b. Parent Avoidant Personality (ASR DSM Composite)
# parent_avoidant_person_D_p is the ASR DSM-oriented Avoidant Personality scale,
# the closest available analog of the CBCL Social Problems syndrome at the
# parent self-report measurement level. Its item constituents span social
# inhibition, self-consciousness, social withdrawal, and feelings of inferiority.
# Exclusions cover (a) the constituent items, (b) ASR Internalizing/Withdrawn
# syndrome composites that subsume it, (c) social-functioning items that index
# the same latent dimension, and (d) KSADS social-anxiety diagnostic proxies.
_PARENT_AVOIDANT_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _PARENT_GENERAL_MOOD_EXCLUDE
    | {"parent_avoidant_person_D_p", "delta_parent_avoidant_person_D_p"}
    # ASR DSM composites that subsume or substantially overlap with Avoidant Personality
    | {
        "parent_internal_D_p",
        "parent_anxdep_D_p",
        "parent_anxdisord_D_p",
        "parent_personal_strength_D_p",
        "parent_intrusive_thoughts_D_p",
        "parent_somatic_problems_D_p",
        "parent_depress_D_p",
        "parent_depressed_mood_B_p",
    }
    # ASR Avoidant Personality scale primary item constituents
    | {
        "parent_self_conscious_p",  # ASR 71 Self-conscious / easily embarrassed
        "parent_shy_or_timid_p",  # ASR 75 Shy or timid
        "parent_feels_inferior_p",  # ASR 35 Feels worthless or inferior
        "parent_avoids_talking_p",  # ASR 65 Refuses to talk
        "parent_prefers_to_be_alone_p",  # ASR 42 Prefers to be alone
        "parent_relationship_concerns_p",  # ASR 113 Worries about relationships
        "parent_fearful_or_anxious_p",
    }  # ASR 50 Too fearful or anxious
    # ASR Withdrawn syndrome co-constituents (composite subsumption + construct overlap)
    | {
        "parent_bad_relationships_p",  # ASR 25
        "parent_bad_opposite_sex_relationship_p",  # ASR 30
        "parent_not_liked_by_others_p",  # ASR 48
        "parent_friendship_trouble_p",  # ASR 67
        "parent_secretive_p",  # ASR 69
        "parent_enjoys_little_p",
    }  # ASR 60
    # KSADS social-anxiety diagnostic proxies
    | {"parent_social_anxiety_disorder_B_p"}
    # Adaptive functioning / reverse-coded social wellbeing
    | {
        "parent_happy_person_p",
        "parent_meets_family_duties_p",
        "parent_stands_up_rights_p",
        "parent_uses_opportunities_p",
    }
    # Treatment-seeking / family environment
    | {"parent_therapy_emo", "family_emotionprob_p", "parent_strange_thoughts_p"}
    # Delta derivatives
    | {
        "delta_parent_internal_D_p",
        "delta_parent_not_liked_by_others_p",
        "delta_parent_friendship_trouble_p",
        "delta_parent_bad_relationships_p",
        "delta_parent_meets_family_duties_p",
        "delta_parent_bad_opposite_sex_relationship_p",
    }
    | _PARENT_SOCIAL_IMPCOMP_BATCH_EXCLUDE  # batch-wide broad composites + mood-swings
    | _DEPRESSION_SBT
)
CIRCULAR_EXCLUSIONS["parent_avoidant_person_D_p"] = {
    "across_exclude": _PARENT_AVOIDANT_ACROSS,
    "within_exclude": {
        "Parent Mood Issues",
        "Parent Anxiety",
        "Parent Anxiety and Fear-Related Issues",
        "Parent Social Functioning",
        "Parent Interpersonal Relationships and Social Functioning",
        "Parent Personality",
        "Parent Other Psychopathology",
        "Parent Psychopathology Composite",
    },
}

# 23. SES / Neighborhood
# parent_income and area_deprivation_idx are bidirectionally confounded at the
# census-tract level; each is excluded from the other's predictor pool.
CIRCULAR_EXCLUSIONS["parent_income"] = {
    "across_exclude": {"parent_income", "area_deprivation_idx"},
    "within_exclude": set(),
}
CIRCULAR_EXCLUSIONS["area_deprivation_idx"] = {
    "across_exclude": {"area_deprivation_idx", "parent_income"},
    "within_exclude": set(),
}

# 24. Miscellaneous Single-Item Targets
CIRCULAR_EXCLUSIONS["gd_riskybets"] = {"across_exclude": {"gd_riskybets"}, "within_exclude": set()}

CIRCULAR_EXCLUSIONS["nb_correct_nt_2back"] = {
    "across_exclude": {
        "nb_correct_nt_2back",
        "nb_correct_nt",
        "nb_correct_nt_pos",
        "nb_correct_nt_neg",
        "nb_correct_nt_neutral",
        "nb_correct_mrt_2back",
        "nb_correct_mrt",
        "nb_correct_mrt_pos",
        "nb_correct_mrt_neg",
        "nb_correct_mrt_neutral",
        "nb2_accuracy_pos",
        "nb2_accuracy_neg",
        "nb2_D_prime_pos",
        "nb2_D_prime_neg",
        "nb2_resp_bias_pos",
        "nb2_resp_bias_neg",
    },
    "within_exclude": set(),
}

CIRCULAR_EXCLUSIONS["pa_sum_k"] = {"across_exclude": {"pa_sum_k"}, "within_exclude": set()}

# bdefs_distract_upset_p is only present in the T3 predictor pool;
# bdefs_calm_down_p and bdefs_stop_think_p reflect overlapping emotion-regulation strategies.
CIRCULAR_EXCLUSIONS["bdefs_distract_upset_p"] = {
    "across_exclude": {"bdefs_distract_upset_p", "bdefs_calm_down_p", "bdefs_stop_think_p"},
    "within_exclude": set(),
}

# asd_ssrs_sum was missing: without this entry SSRS subscale items
# could circularly predict the SSRS composite.
CIRCULAR_EXCLUSIONS["asd_ssrs_sum"] = {
    "across_exclude": {"asd_ssrs_sum"},
    "within_exclude": set(),
}

# 25. Body Weight
# height, waist, and birth_weight_p excluded as near-collinear body-size indicators.
CIRCULAR_EXCLUSIONS["weight"] = {
    "across_exclude": {
        "weight",
        "height",
        "waist",
        "birth_weight_p",
        "weight_grouped",
        "sex",
        "bad_diet_p",
    },
    "within_exclude": set(),
}

# 26. RCI Threshold Aliases (1.96 ↔ 2.3)
# RCI-1.96 and RCI-2.3 variants measure the same construct at different confidence
# thresholds; they share identical exclusion sets.
for _t, _b in [
    ("dep_onset_rci_1.96", "dep_onset_rci_2.3"),
    ("dep_onset_rci_1.7", "dep_onset_rci_2.3"),
    ("dep_remission_rci_1.96", "dep_remission_rci_2.3"),
    ("dep_remission_rci_1.7", "dep_remission_rci_2.3"),
    ("sbt_core_dep_onset_rci_1.96", "sbt_core_dep_onset_rci_2.3"),
    ("sbt_core_dep_onset_rci_1.7", "sbt_core_dep_onset_rci_2.3"),
    ("sbt_core_dep_remission_rci_1.96", "sbt_core_dep_remission_rci_2.3"),
    ("sbt_core_dep_remission_rci_1.7", "sbt_core_dep_remission_rci_2.3"),
]:
    CIRCULAR_EXCLUSIONS[_t] = CIRCULAR_EXCLUSIONS[_b].copy()

# 27. Child KSADS classification targets (19 targets)
# Targets already inheriting _DEP_BASE above (section 2):
# depressed_mood_B_k, anhedonia_B_k, hopeless_B_k,
# two_more_depression_B_k, ksads_1_840_t, MDD_KSADS_C

# 27a. Child Suicidality (KSADS)

_CHILD_SUICIDALITY_ACROSS = _DEPRESSION_ACROSS | {"parent_suicidal_thoughts_p", "parent_self_harm_p"}

for target in ["suicidal_ideation_present_B_k", "suicide_attempt_present_B_k", "suicide_attempt_past_B_k"]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _CHILD_SUICIDALITY_ACROSS,
        "within_exclude": _DEPRESSION_WITHIN,
        "across_exclude_t1t3t4": _DEPRESSION_T1T3T4_EXTRA,
    }

# 27b. Child Fatigue (KSADS)

CIRCULAR_EXCLUSIONS["fatigue_present_B_k"] = {
    "across_exclude": (
        _DEPRESSION_ACROSS
        | {"parent_tired_no_reason_p", "parent_low_energy_p"}
        | {"chronotype", "fallsleeptime", "wakeuptime", "wakesleepcalc"}
        | {"no_sports_activities_p", "fitbit_sedentary_mins", "fitbit_resting_hr"}
    ),
    "within_exclude": _DEPRESSION_WITHIN,
    "across_exclude_t1t3t4": _DEPRESSION_T1T3T4_EXTRA,
}

# 27c. Elevated Mood (KSADS)

CIRCULAR_EXCLUSIONS["elevated_mood_B_k"] = {
    "across_exclude": (
        _DEPRESSION_ACROSS
        | {"hyperactive_p", "impulsive_p", "loquacious_p", "demands_attention_p", "temper_tantrums_p"}
        | {
            "bragadocious_p",
            "easily_jealous_p",
            "up_positiveurgency_ss_k",
            "up_sensationseeking_ss_k",
            "bis_funseeking_ss_k",
            "bis_drive_ss_k",
        }
        | {"mania_7up_ss_k"}
        | {
            "parent_elevated_mood_B_p",
            "parent_hyperactive_D_p",
            "parent_restless_p",
            "parent_behavior_changeable_p",
            "parent_sudden_mood_changes_p",
        }
    ),
    "within_exclude": {"Anxiety", "Child Anxiety", "ADHD", "Child ADHD"},
    "across_exclude_t1t3t4": _DEPRESSION_T1T3T4_EXTRA,
}

# 27d. Child Anxiety-domain (KSADS)

CIRCULAR_EXCLUSIONS["excessive_worry_B_k"] = {
    "across_exclude": _ANXIETY_ACROSS,
    "within_exclude": {"Anxiety", "Child Anxiety"},
    "across_exclude_t1t3t4": _ANXIETY_T1T3T4_EXTRA,
}

CIRCULAR_EXCLUSIONS["GAD_present"] = {
    "across_exclude": (
        _ANXIETY_ACROSS
        | {"body_aches_p", "frequent_headaches_p", "frequent_stomachaches_p"}
        | {"difficulty_goingtosleep_p", "sleep_problems_p"}
    ),
    "within_exclude": {"Anxiety", "Child Anxiety"},
    "across_exclude_t1t3t4": _ANXIETY_T1T3T4_EXTRA,
}

_SOCIAL_ANXIETY_CHILD_EXTRA = {
    "shy_p",
    "prefer_alone_p",
    "lonely_p",
    "poor_eye_contact_B_p",
    "difficulty_making_friends_p",
    "doesnt_feel_accepted_k",
    "parent_shy_or_timid_p",
    "parent_prefers_to_be_alone_p",
    "parent_avoids_talking_p",
}

for target in ["socially_anxious_B_k", "social_anxiety_disorder_B_k"]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _ANXIETY_ACROSS | _SOCIAL_ANXIETY_CHILD_EXTRA,
        "within_exclude": {"Anxiety", "Child Anxiety"},
        "across_exclude_t1t3t4": _ANXIETY_T1T3T4_EXTRA,
    }

# 27e. Child OCD (KSADS)

CIRCULAR_EXCLUSIONS["OCD_present"] = {
    "across_exclude": (
        CIRCULAR_EXCLUSIONS["ocd_D_p"]["across_exclude"]
        | {"parent_obsessive_thoughts_p", "parent_repeats_acts_p", "perfectionist_p"}
    ),
    "within_exclude": set(),
}

# 27f. Child Sleep Problems (KSADS)

CIRCULAR_EXCLUSIONS["sleep_problem_B_k"] = {
    "across_exclude": (
        _DEPRESSION_ACROSS
        | {"nightmares_p", "nightmares_B_p", "parent_sleep_problem_B_p"}
        | {"parent_sleep_trouble_p", "parent_high_sleep_duration_p"}
        | {"parent_tired_no_reason_p", "parent_low_energy_p"}
        | {"chronotype", "fallsleeptime", "wakeuptime", "wakesleepcalc"}
        | {"fitbit_sedentary_mins", "fitbit_resting_hr"}
    ),
    "within_exclude": {"Anxiety", "Child Anxiety"},
    "across_exclude_t1t3t4": _DEPRESSION_T1T3T4_EXTRA,
}

# 27g. Child Lying (KSADS)

CIRCULAR_EXCLUSIONS["lying_B_k"] = {
    "across_exclude": (
        _EXTERNAL_ACROSS
        | {"parent_lying_B_p", "parent_antisocial_D_p"}
        | {"no_guilt_p", "parent_no_guilt_p", "parent_honest_p"}
        | {"school_detension_suspension"}
    ),
    "within_exclude": {"Externalizing", "Child Externalizing"},
}

# 27h. Child Eating Disorders (KSADS)

_EATING_ACROSS = (
    {"anorexia_B_k", "bulimia_B_k"}
    | {"doesnt_eat_well_p", "overeating_p"}
    | {"nausea_p", "vomiting_p", "frequent_stomachaches_p", "constipated_p"}
    | {"somatic_problems_D_p", "delta_somatic_problems_D_p"}
    | {"fear_becoming_obese_present_PK", "weight", "waist", "bad_diet_p"}
    | {"height", "saliva_DHEA", "saliva_testosterone", "puberty_k"}
    | {"perfectionist_p", "fears_being_bad_p"}
    | {
        "total_calories",
        "protein_intake",
        "carbohydrate_intake",
        "fiber_intake",
        "sodium_intake",
        "potassium_intake",
        "total_sugar",
        "saturated_fat",
        "fruit_intake",
        "vegetable_intake",
        "sugary_beverage_freq",
        "protein_sources_intake",
        "legume_intake",
        "added_sugar",
        "dairy_intake",
        "whole_grain_intake",
    }
    | {"parent_bulimia_B_p", "parent_doesnt_eat_well_p"}
)

for target in ["anorexia_B_k", "bulimia_B_k"]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _EATING_ACROSS,
        "within_exclude": {"Diet/Nutrition"},
    }

# 28. Parent KSADS classification targets (15 targets)

# 28a. Parent Depression-domain (KSADS)

_PARENT_DEP_KSADS_ACROSS = (
    _PARENT_DEP_ACROSS
    | {"parent_two_more_depression_B_p", "hopeless_B_p", "parent_low_self_esteem_B_p"}
    | {"parent_wish_dead_present_B_p", "parent_suicide_past_B_p", "ptsd_diagnosis_present_B_p"}
    | {"parent_shy_or_timid_p", "parent_prefers_to_be_alone_p", "parent_avoids_talking_p"}
    | {"parent_confused_p", "parent_clumsy_p"}
    | {"parent_heart_racing_p", "parent_numbness_p"}
    | {"delta_parent_poor_work_performance_p"}
)

for target in [
    "parent_two_more_depression_B_p",
    "parent_depressed_mood_B_p",
    "parent_anhedonia_B_p",
    "hopeless_B_p",
    "parent_dysthymia_B_p",
    "parent_low_self_esteem_B_p",
]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _PARENT_DEP_KSADS_ACROSS,
        "within_exclude": _PARENT_DEP_WITHIN,
    }

# 28b. Parent Anxiety-domain (KSADS)

_PARENT_ANX_KSADS_ACROSS = (
    _PARENT_ANX_ACROSS
    | {
        "parent_excessive_worry_B_p",
        "parent_socially_anxious_B_p",
        "parent_distress_due_to_social_anxiety_B_p",
        "parent_obsessions_present_B_p",
    }
    | {"nightmares_B_p", "obsessions_present_B_p"}
    | {"parent_shy_or_timid_p", "parent_prefers_to_be_alone_p", "parent_avoids_talking_p", "parent_self_conscious_p"}
    | {"parent_heart_racing_p", "parent_numbness_p"}
)

_PARENT_SA_SOCIAL_CONTAMINATION = {
    "parent_bad_relationships_p",
    "parent_friendship_trouble_p",
    "parent_not_liked_by_others_p",
    "parent_bad_opposite_sex_relationship_p",
    "delta_parent_friendship_trouble_p",
    "delta_parent_not_liked_by_others_p",
    "delta_parent_bad_relationships_p",
}

CIRCULAR_EXCLUSIONS["parent_excessive_worry_B_p"] = {
    "across_exclude": _PARENT_ANX_KSADS_ACROSS,
    "within_exclude": _PARENT_ANX_WITHIN,
}

for target in ["parent_socially_anxious_B_p", "parent_social_anxiety_disorder_B_p"]:
    CIRCULAR_EXCLUSIONS[target] = {
        "across_exclude": _PARENT_ANX_KSADS_ACROSS | _PARENT_SA_SOCIAL_CONTAMINATION,
        "within_exclude": _PARENT_ANX_WITHIN
        | {"Parent Social Functioning", "Parent Interpersonal Relationships and Social Functioning"},
    }

CIRCULAR_EXCLUSIONS["parent_distress_due_to_social_anxiety_B_p"] = {
    "across_exclude": (
        _PARENT_ANX_KSADS_ACROSS
        | _PARENT_SA_SOCIAL_CONTAMINATION
        | {
            "parent_bad_family_relationship_p",
            "parent_meets_family_duties_p",
            "delta_parent_meets_family_duties_p",
            "delta_parent_poor_work_performance_p",
            "parent_prefers_older_people_p",
            "parent_clowns_or_shows_off_p",
        }
    ),
    "within_exclude": _PARENT_ANX_WITHIN
    | {"Parent Social Functioning", "Parent Interpersonal Relationships and Social Functioning"},
}

# 28c. Parent Concentration (KSADS)

CIRCULAR_EXCLUSIONS["parent_concentration_issues_B_p"] = {
    "across_exclude": (
        _PARENT_ADHD_ACROSS
        | {"parent_concentration_issues_B_p"}
        | {
            "bdefs_explain_idea_p",
            "bdefs_explain_pt_p",
            "bdefs_explain_seq_p",
            "bdefs_inconsistant_p",
            "bdefs_process_info_p",
            "bdefs_impulsive_action_p",
        }
        | {"delta_parent_poor_work_performance_p"}
    ),
    "within_exclude": {"Parent Cognitive and Attention Issues", "Parent Impulsivity and Behavior Regulation"},
}

# 28d. Parent Irritability (KSADS)

CIRCULAR_EXCLUSIONS["parent_easily_annoyed_B_p"] = {
    "across_exclude": (
        _GENERAL_INTERNALIZING_EXCLUDE
        | {"parent_easily_annoyed_B_p"}
        | {
            "parent_stubborn_irritable_p",
            "delta_parent_stubborn_irritable_p",
            "parent_hot_temper_p",
            "parent_yells_a_lot_p",
            "parent_louder_than_others_p",
        }
        | {"parent_sudden_mood_changes_p", "parent_behavior_changeable_p"}
        | {"parent_aggressive_D_p", "parent_external_D_p", "parent_depressed_p", "parent_anxdep_D_p"}
        | {"parent_impulsive_p", "parent_hyperactive_p", "parent_restless_p"}
        | {"parent_physical_attacks_p", "parent_destroys_own_things_p", "parent_destroys_others_things_p"}
        | {"parent_elevated_mood_B_p"}
        | {"delta_parent_depressed_p", "delta_parent_feels_overwhelmed_p"}
        | {
            "frequent_family_conflict_p",
            "family_conflict_ss_p",
            "family_conflict_ss_k",
            "family_lose_temper_rare_p",
            "family_believe_not_raise_voice_p",
            "parents_argue_more_p",
        }
        | {"fam_fight_often_k", "fam_no_lose_temps_k", "fam_criticize_often_k", "fam_no_raise_voices_k"}
        | {"argues_p", "stubborn_p", "temper_tantrums_p", "disobedient_home_p", "whines_p"}
    ),
    "within_exclude": {"Parent Impulsivity and Behavior Regulation", "Parent Mood Issues"},
}

# 28e. Parent Sleep Problems (KSADS)

CIRCULAR_EXCLUSIONS["parent_sleep_problem_B_p"] = {
    "across_exclude": (
        _PARENT_DEP_ACROSS
        | {"parent_sleep_problem_B_p", "parent_high_sleep_duration_p", "nightmares_B_p"}
        | {"parent_tired_no_reason_p", "parent_low_energy_p"}
        | {"parent_confused_p", "parent_forgetful_p", "parent_concentration_trouble_p"}
        | {"difficulty_goingtosleep_p", "difficulty_wakingup_p", "nightmares_p"}
        | {"parent_heart_racing_p"}
    ),
    "within_exclude": _PARENT_DEP_WITHIN,
}

# 28f. Parent PTSD (KSADS)

CIRCULAR_EXCLUSIONS["ptsd_diagnosis_present_B_p"] = {
    "across_exclude": (
        _PARENT_DEP_ACROSS
        | _PARENT_ANX_ACROSS
        | {"ptsd_diagnosis_present_B_p"}
        | {
            "parent_heart_racing_p",
            "parent_numbness_p",
            "parent_physical_attacks_p",
            "parent_hot_temper_p",
            "parent_yells_a_lot_p",
        }
        | {"nightmares_B_p"}
        | {"parent_prefers_to_be_alone_p", "parent_shy_or_timid_p"}
        | {"parent_drinks_too_much_p", "parent_drunk_days_p", "parent_drug_use_p", "parent_drug_days_p"}
        | {"parent_strange_behavior_p", "parent_strange_thoughts_p"}
        | {
            "car_accident_hurt_p",
            "big_accident_need_treatment_p",
            "fire_victim_p",
            "natural_disaster_victim_p",
            "terrorism_victim_p",
            "war_death_witness_p",
            "stabbing_shooting_witness_p",
            "stabbing_shooting_victim_community_p",
            "stabbing_shooting_victim_home_p",
            "beating_victim_home_p",
            "stranger_threatened_child_victim_p",
            "family_threatened_child_victim_p",
            "adult_family_fighting_victim_p",
            "domestic_child_sexually_abuse_victim_p",
            "foreign_child_sexually_abuse_victim_p",
            "peer_child_sexually_abuse_victim_p",
            "sudden_death_in_family_p",
        }
    ),
    "within_exclude": (_PARENT_DEP_WITHIN | _PARENT_ANX_WITHIN | {"Parent Impulsivity and Behavior Regulation"}),
}

# 28g. Parent Obsessions (KSADS)

CIRCULAR_EXCLUSIONS["parent_obsessions_present_B_p"] = {
    "across_exclude": (
        {"parent_obsessions_present_B_p"}
        | {"parent_obsessive_thoughts_p", "parent_repeats_acts_p"}
        | {"obsessions_present_B_p"}
        | {"parent_anxdep_D_p", "parent_internal_D_p", "parent_intrusive_thoughts_D_p"}
        | {"parent_strange_thoughts_p", "parent_strange_behavior_p", "parent_sees_things_p", "parent_hears_voices_p"}
        | {"parent_picks_skin_p"}
        | {"parent_hyperactive_p", "parent_restless_p", "parent_max_effort_p"}
        | _DEPRESSION_SBT
    ),
    "within_exclude": {"Parent Cognitive and Attention Issues"},
}

# 28h. Parent Compulsions (ASR Repeats Acts)
# parent_repeats_acts_p is the ASR analog of child cbcl_q66 ("Repeats certain
# acts over and over"). Mirrors child compulsions_p (§14) adapted to the
# parent self-report measurement level.
CIRCULAR_EXCLUSIONS["parent_repeats_acts_p"] = {
    "across_exclude": (
        {"parent_repeats_acts_p", "delta_parent_repeats_acts_p"}
        | {
            "parent_obsessive_thoughts_p",  # ASR OCD scale co-constituent
            "parent_picks_skin_p",
        }  # body-focused repetitive behavior
        | {"parent_intrusive_thoughts_D_p"}  # ASR DSM OCD composite that aggregates this item
        | {"parent_obsessions_present_B_p"}  # KSADS OCD diagnostic proxy
        | {
            "parent_strange_behavior_p",
            "parent_strange_thoughts_p",  # thought disorder / OCD-anxiety overlap
            "parent_sees_things_p",
            "parent_hears_voices_p",
        }
        | {"parent_fear_of_bad_thoughts_p"}  # ASR 31: intrusive-thoughts construct overlap
        | {
            "parent_internal_D_p",
            "parent_anxdep_D_p",
        }  # ASR Internalizing broadband (subsumes OCD scale via Thought Problems overlap)
        | {
            "parent_hyperactive_p",
            "parent_restless_p",  # restless/driven vs compulsive repetition
            "parent_max_effort_p",
        }
        | {"compulsions_p"}  # cross-rater child analog
        | _PARENT_SOCIAL_IMPCOMP_BATCH_EXCLUDE  # batch-wide broad composites + mood-swings
        | _DEPRESSION_SBT
    ),
    "within_exclude": {
        "Parent Cognitive and Attention Issues",
        "Parent Anxiety and Fear-Related Issues",
        "Parent Other Psychopathology",
        "Parent Psychopathology Composite",
    },
}

# 28i. Parent Obsessions (ASR Obsessive Thoughts)
# parent_obsessive_thoughts_p is the ASR analog of child cbcl_q09
# ("Obsessions / can't get mind off certain thoughts"). Mirrors child
# obsessions_p (§11) adapted to the parent self-report measurement level.
# Attention items (concentration, decision trouble) intentionally NOT excluded
# to align with child §11 convention: attention/rumination conflation is
# permitted as cross-domain prediction signal.
CIRCULAR_EXCLUSIONS["parent_obsessive_thoughts_p"] = {
    "across_exclude": (
        {"parent_obsessive_thoughts_p", "delta_parent_obsessive_thoughts_p"}
        | {
            "parent_repeats_acts_p",  # ASR OCD scale co-constituent
            "parent_picks_skin_p",
        }  # body-focused repetitive behavior
        | {"parent_intrusive_thoughts_D_p"}  # ASR DSM OCD composite that aggregates this item
        | {"parent_obsessions_present_B_p"}  # KSADS OCD diagnostic proxy
        | {
            "parent_strange_thoughts_p",
            "parent_strange_behavior_p",  # thought disorder overlap
            "parent_sees_things_p",
            "parent_hears_voices_p",
        }
        | {"parent_fear_of_bad_thoughts_p"}  # ASR 31: intrusive-thoughts construct overlap
        | {
            "parent_internal_D_p",
            "parent_anxdep_D_p",
        }  # ASR Internalizing broadband (subsumes OCD via Thought Problems overlap)
        | {
            "parent_hyperactive_p",
            "parent_restless_p",  # restless/driven vs obsessive preoccupation
            "parent_max_effort_p",
        }
        | {"obsessions_p"}  # cross-rater child analog
        | _PARENT_SOCIAL_IMPCOMP_BATCH_EXCLUDE  # batch-wide broad composites + mood-swings
        | _DEPRESSION_SBT
    ),
    "within_exclude": {
        "Parent Cognitive and Attention Issues",
        "Parent Anxiety and Fear-Related Issues",
        "Parent Other Psychopathology",
        "Parent Psychopathology Composite",
    },
}

# Parent Internalizing (ASR Broadband)
# ASR Internalizing = Anxious/Depressed (18 items) + Withdrawn (9 items)
# + Somatic Complaints (12 items).
# ALL constituent items of these three subscales must be excluded, plus
# broadband composites, diagnostic proxies, and change-score derivatives.
# _PARENT_GENERAL_MOOD_EXCLUDE (registry) already covers:
# ASR Anxious/Depressed:  parent_self_conscious_p (71), parent_fearful_or_anxious_p (50),
# parent_confused_p (13)
# ASR Withdrawn:          parent_prefers_to_be_alone_p (42), parent_avoids_talking_p (65)
# ASR Somatic Complaints: parent_heart_racing_p (56h), parent_numbness_p (56i)
# Other Problems:         parent_shy_or_timid_p (75), parent_high_sleep_duration_p (77)
_PARENT_INTERNAL_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    | _PARENT_GENERAL_MOOD_EXCLUDE
    # ASR broadband & syndrome composites
    | {
        "parent_internal_D_p",
        "parent_anxdep_D_p",
        "parent_adhd_D_p",
        "parent_somatic_problems_D_p",
        "parent_depressed_mood_B_p",
    }  # ASR internalizing subscales
    | {
        "parent_external_D_p",
        "parent_aggressive_D_p",
        "parent_antisocial_D_p",
        "parent_hyperactive_D_p",
    }  # ASR externalizing broadband (composite subsumption: prevents reconstruction via subtraction)
    | {"parent_depress_D_p", "parent_anxdisord_D_p"}  # CBCL-style DSM syndrome composites
    | {
        "parent_avoidant_person_D_p",
        "parent_personal_strength_D_p",
        "parent_attention_D_p",
    }  # ASR adaptive/attention composites with internalizing loading
    # ASR Anxious/Depressed subscale (18 items → Internalizing)
    # Items already covered by _PARENT_GENERAL_MOOD_EXCLUDE above:
    # parent_self_conscious_p (71), parent_fearful_or_anxious_p (50),
    # parent_confused_p (13)
    # Remaining Anxious/Depressed items:
    | {
        "parent_depressed_p",  # ASR 103 Unhappy, sad, or depressed
        "parent_cries_a_lot_p",  # ASR 14 Cries a lot
        "parent_lonely_p",  # ASR 12 Lonely
        "parent_feels_unloved_p",  # ASR 33 Feels no one loves me
        "parent_feels_inferior_p",  # ASR 35 Feels worthless or inferior
        "parent_feels_unsuccessful_p",  # ASR 107 Feels can't succeed
        "parent_paranoid_p",  # ASR 34 Others are out to get me
        "parent_worries_a_lot_p",  # ASR 112 Worries
        "parent_worries_about_future_p",  # ASR 22 Worries about future
        "parent_relationship_concerns_p",  # ASR 113 Worries about opposite sex relations
        "parent_fear_of_bad_thoughts_p",  # ASR 31 Afraid might think/do something bad
        "parent_suicidal_thoughts_p",
    }  # ASR 91 Thinks about killing self
    # ASR Withdrawn subscale (9 items → Internalizing)
    # Items already covered by _PARENT_GENERAL_MOOD_EXCLUDE above:
    # parent_prefers_to_be_alone_p (42), parent_avoids_talking_p (65)
    # Remaining Withdrawn items:
    | {
        "parent_bad_relationships_p",  # ASR 25 Don't get along with other people
        "parent_bad_opposite_sex_relationship_p",  # ASR 30 Relations with opposite sex are poor
        "parent_not_liked_by_others_p",  # ASR 48 Not liked by others
        "parent_friendship_trouble_p",  # ASR 67 Trouble making or keeping friends
        "parent_enjoys_little_p",  # ASR 60 Very little that I enjoy
        "parent_secretive_p",
    }  # ASR 69 Secretive, keeps things to self
    # ASR 111 (Withdrawn) not in ABCD dataset
    # ASR Somatic Complaints subscale (12 items → Internalizing)
    # Items already covered by _PARENT_GENERAL_MOOD_EXCLUDE above:
    # parent_heart_racing_p (56h), parent_numbness_p (56i)
    # Remaining Somatic items:
    | {
        "parent_tired_no_reason_p",  # ASR 54 Tired without good reason
        "parent_sleep_trouble_p",  # ASR 100 Trouble sleeping
        "parent_picks_skin_p",
    }  # ASR 58 Picks skin (Other Problems, but somatic/BFR-adjacent)
    # parent_doesnt_eat_well_p (ASR 24, Other Problems) intentionally
    # NOT excluded: it is not an Internalizing subscale constituent.
    # ASR 51 (dizzy), 56a-56g (aches/headaches/nausea/eye/skin/
    # stomach/vomiting) not individually named in ABCD: they are
    # captured via the parent_somatic_problems_D_p composite above.
    # Cross-scale items excluded for construct overlap
    # These are NOT Internalizing subscale constituents but index mood/
    # energy/anhedonia constructs that are inseparable from depression.
    | {
        "parent_happy_person_p",  # Adaptive: reverse-coded anhedonia proxy
        "parent_sudden_mood_changes_p",  # ASR 87 Aggressive Behavior: mood dysregulation confound
        "parent_decision_trouble_p",  # ASR 78 Attention Problems: depression-linked indecisiveness
        "parent_restless_p",  # Other Problems: anxious agitation confound
        "parent_low_energy_p",
    }  # ASR 102 Attention Problems: fatigue/anergia confound
    # Thought disorder / OCD-anxiety overlap
    | {
        "parent_intrusive_thoughts_D_p",
        "parent_obsessive_thoughts_p",
    }  # ASR 9 Thought Problems: obsessional anxiety overlap
    # KSADS / diagnostic items
    | {"parent_anhedonia_B_p", "parent_dysthymia_B_p"}  # DSM MDD core criterion / depressive disorder diagnosis
    | {
        "parent_self_harm_p",  # suicidality subsumed by internalizing
        "parent_suicide_past_B_p",
    }  # KSADS past suicide attempt
    | {"ptsd_diagnosis_present_B_p"}  # trauma/PTSD diagnostic: internalizing spectrum
    | {"parent_social_anxiety_disorder_B_p"}  # KSADS social anxiety diagnosis
    | {
        "parent_specific_fears_p",  # ASR 29 Other Problems: specific phobia-adjacent
        "parent_worries_about_family_p",
    }  # ASR 72 Other Problems: generalized worry proxy
    # Family/environment confounds
    | {
        "parent_therapy_emo",
        "family_emotionprob_p",
        "parent_strange_thoughts_p",
    }  # treatment-seeking / family distress confounds
    # Delta (change-score) derivatives
    | {
        "delta_parent_internal_D_p",
        "delta_parent_depressed_p",
        "delta_parent_worries_a_lot_p",
        "delta_parent_somatic_problems_D_p",
        "delta_parent_feels_overwhelmed_p",
        "delta_parent_sleep_trouble_p",
        "delta_parent_feels_unloved_p",
        "delta_parent_stubborn_irritable_p",
        "delta_parent_worries_about_family_p",
        "delta_parent_worries_about_future_p",
        "delta_parent_aches_pains_p",
        "delta_parent_planning_trouble_p",
        "delta_parent_bad_relationships_p",  # Withdrawn deltas
        "delta_parent_bad_opposite_sex_relationship_p",
        "delta_parent_not_liked_by_others_p",
        "delta_parent_friendship_trouble_p",
    }
    | _DEPRESSION_SBT
)
CIRCULAR_EXCLUSIONS["parent_internal_D_p"] = {
    "across_exclude": _PARENT_INTERNAL_ACROSS,
    "within_exclude": {
        "Parent Mood Issues",
        "Parent Anxiety",
        "Parent Anxiety and Fear-Related Issues",
        "Parent Other Psychopathology",
        "Parent Psychopathology Composite",
    },
}

# Parent Externalizing (ASR Broadband)
# ASR Externalizing = Aggressive Behavior (15 items) + Rule-Breaking
# Behavior (14 items) + Intrusive (6 items).
# ALL constituent items of these three subscales must be excluded, plus
# broadband composites, diagnostic proxies, and change-score derivatives.
_PARENT_EXTERNAL_ACROSS = (
    _GENERAL_INTERNALIZING_EXCLUDE
    # ASR broadband & syndrome composites
    | {
        "parent_external_D_p",
        "parent_aggressive_D_p",
        "parent_antisocial_D_p",
        "parent_hyperactive_D_p",
    }  # ASR externalizing subscales
    | {
        "parent_internal_D_p",
        "parent_anxdep_D_p",
        "parent_depress_D_p",
        "parent_somatic_problems_D_p",
    }  # ASR internalizing broadband (composite subsumption)
    | {"parent_adhd_D_p", "parent_attention_D_p"}  # ADHD composites (externalizing ADHD item overlap)
    | {
        "parent_personal_strength_D_p",
        "parent_intrusive_thoughts_D_p",
        "parent_avoidant_person_D_p",
    }  # ASR composites with externalizing loading
    # ASR Aggressive Behavior items (15 items → Externalizing)
    | {
        "parent_stubborn_irritable_p",  # ASR 86 Stubborn, sullen, or irritable
        "parent_hot_temper_p",  # ASR 95 Hot temper
        "parent_yells_a_lot_p",  # ASR 68 Screams or yells a lot
        "parent_sudden_mood_changes_p",  # ASR 87 Moods or feelings change suddenly
        "parent_behavior_changeable_p",  # ASR 81 Behavior is very changeable
        "parent_destroys_own_things_p",  # ASR 20 Destroys own things
        "parent_destroys_others_things_p",  # ASR 21 Destroys others' things (via ASR item 21)
        "parent_physical_attacks_p",  # ASR 57 Physically attacks people
        "parent_depressed_p",
    }  # ASR 103: excluded for Aggressive mood-swing confound
    # remaining Aggressive items (3-argues, 5-blames, 16-mean,
    # 28-badly_family, 37-fights, 55-elate_depress, 97-threaten,
    # 116-upset, 118-impatient) may not be individually named in ABCD
    # but are captured via parent_aggressive_D_p composite above.
    # ASR Rule-Breaking Behavior items (14 items → Externalizing)
    | {
        "parent_impulsive_p",  # ASR 41 Impulsive, acts without thinking
        "parent_no_guilt_p",  # ASR 26 Doesn't feel guilty
        "parent_illegal_behavior_p",  # ASR 92 Does things that may cause trouble with law
        "parent_drug_use_p",  # ASR 6 Uses drugs for nonmedical purposes
        "parent_drinks_too_much_p",  # ASR 90 Drinks too much / gets drunk
        "parent_money_management_trouble_p",  # ASR 117 Trouble managing money or credit cards
        "parent_doesnt_finish_tasks_p",  # ASR 59 Fails to finish things (also Attention)
        "parent_risky_decisions_p",  # ASR 76 Behavior is irresponsible
        "parent_strange_behavior_p",
    }  # ASR 84 Does things others think are strange (also Thought)
    # parent_fails_to_pay_debts_p (ASR 114) and
    # parent_drives_too_fast_p (ASR 120, Other) not reliably in ABCD
    # but are named below for safety-net coverage.
    | {
        "parent_fails_to_pay_debts_p",  # ASR 114 Fails to pay debts (Rule-Breaking)
        "parent_drives_too_fast_p",
    }  # ASR 120 Other Problems: reckless-impulsivity confound
    # parent_tardy_often_p (ASR 121, Attention) intentionally NOT
    # excluded: symmetrical with PI treatment.
    # ASR Intrusive items (6 items → Externalizing)
    | {
        "parent_bragging_p",  # ASR 7 Brags
        "parent_attention_seeking_p",  # ASR 19 Tries to get a lot of attention
        "parent_clowns_or_shows_off_p",  # ASR 74 Shows off or clowns
        "parent_talks_too_much_p",  # ASR 93 Talks too much
        "parent_teases_others_p",  # ASR 94 Teases others
        "parent_louder_than_others_p",
    }  # ASR 104 Louder than others
    # ASR Withdrawn items (Internalizing: excluded for composite subsumption)
    # These are Internalizing constituents, not Externalizing. Excluding
    # them prevents indirect reconstruction of Externalizing via
    # Total − Internalizing arithmetic and removes social-functioning
    # confounds that covary with externalizing consequences.
    | {
        "parent_bad_relationships_p",  # ASR 25 Withdrawn → INT
        "parent_bad_family_relationship_p",  # ASR 28 Aggressive → EXT (gets along badly with family)
        "parent_not_liked_by_others_p",  # ASR 48 Withdrawn → INT
        "parent_friendship_trouble_p",  # ASR 67 Withdrawn → INT
        "parent_bad_opposite_sex_relationship_p",  # ASR 30 Withdrawn → INT
        "parent_meets_family_duties_p",  # Adaptive functioning (reverse-coded): interpersonal confound
        "parent_stands_up_rights_p",  # Adaptive functioning: assertiveness confound
        "parent_associates_with_trouble_p",  # ASR 39 Rule-Breaking → EXT
        "parent_prefers_older_people_p",
    }  # ASR 63 Thought Problems: social-deviance confound
    # Cross-scale items excluded for construct overlap
    | {
        "parent_secretive_p",  # ASR 69 Withdrawn: covert antisocial confound
        "parent_restless_p",  # Other Problems: hyperactive-impulsivity confound
        "parent_hyperactive_p",  # Other Problems: ADHD-hyperactivity confound
        "parent_strange_thoughts_p",
    }  # ASR 85 Thought Problems: impulsive-chaotic cognition
    # parent_priority_trouble_p (ASR 64) and parent_easily_bored_p
    # (ASR 83) intentionally NOT excluded: symmetrical with PI treatment.
    # ASR Attention Problems items (parent_concentration_trouble_p,
    # parent_forgetful_p, parent_planning_trouble_p, parent_disorganized_p,
    # parent_loses_things_p) intentionally NOT excluded: they are not
    # Externalizing subscale constituents and are allowed as cross-domain
    # predictors (symmetrical with Parent Internalizing treatment).
    # KSADS / diagnostic items with externalizing overlap
    | {"parent_lying_B_p", "parent_elevated_mood_B_p"}  # antisocial/mania-hyperactivity ambiguity
    # Family/environment confounds
    | {
        "family_conflict_ss_p",
        "family_organisation_ss_p",
        "family_believe_not_raise_voice_p",
        "family_lose_temper_rare_p",
        "frequent_family_conflict_p",
    }  # family behavioral environment confounds
    # Delta (change-score) derivatives
    | {
        "delta_parent_stubborn_irritable_p",
        "delta_parent_bad_relationships_p",
        "delta_parent_bad_family_relationship_p",
        "delta_parent_not_liked_by_others_p",
        "delta_parent_friendship_trouble_p",
        "delta_parent_meets_family_duties_p",
        "delta_parent_poor_work_performance_p",
        "delta_parent_bad_opposite_sex_relationship_p",  # Withdrawn delta
        "delta_parent_drug_use_p",
        "delta_parent_drinks_too_much_p",  # Rule-Breaking substance deltas
        "delta_parent_fails_to_pay_debts_p",  # Rule-Breaking delta
        "delta_family_conflict_ss_p",
    }
)
CIRCULAR_EXCLUSIONS["parent_external_D_p"] = {
    "across_exclude": _PARENT_EXTERNAL_ACROSS,
    "within_exclude": {
        "Parent Impulsivity and Behavior Regulation",
        "Parent Interpersonal Relationships and Social Functioning",
        "Parent Social Functioning",
        "Parent Cognitive and Attention Issues",
        "Parent Other Psychopathology",
    },
}

# Lookup functions


# Child Only Deltas: entries

# delta_family_conflict_ss_k: Youth-report family conflict change score.
# Same FES conflict items as the parent-report version.
CIRCULAR_EXCLUSIONS["delta_family_conflict_ss_k"] = {
    "across_exclude": (
        {
            "family_conflict_ss_p",
            "family_conflict_ss_k",
            "delta_family_conflict_ss_p",
            "delta_family_conflict_ss_k",
            "frequent_family_conflict_p",
            "family_lose_temper_rare_p",
            "family_peaceful_p",
            "family_believe_not_raise_voice_p",
            "parents_argue_more_p",
        }
        | {
            "fam_fight_often_k",
            "fam_no_open_anger_k",
            "fam_no_lose_temps_k",
            "fam_criticize_often_k",
            "fam_hit_each_other_k",
            "fam_throw_things_k",
            "fam_keep_peace_k",
            "fam_try_one_up_k",
            "fam_no_raise_voices_k",
        }
    ),
    "within_exclude": {"Family Dynamics & Parenting"},
}

# delta_somatic_problems_D_p: CBCL Somatic Complaints syndrome change score.
# Exclude all somatic syndrome constituent items plus the internalizing broadband
# composites that subsume somatic complaints.
CIRCULAR_EXCLUSIONS["delta_somatic_problems_D_p"] = {
    "across_exclude": (
        _SOMATIC_SYNDROME
        | {
            "somatic_problems_D_p",
            "delta_somatic_problems_D_p",
            "internal_D_p",
            "delta_internal_D_p",
        }  # broadband composites containing somatic
        | {"pain_last_month_k"}  # youth self-report pain item
        | _GENERAL_INTERNALIZING_EXCLUDE
    ),
    "within_exclude": {"Medical/Somatic Problems"},
}

# Parent Only Deltas: entries

# delta_family_conflict_ss_p: Family conflict summary score change.
# Exclude family conflict items and related family-dynamics predictors that
# directly constitute or reconstruct the composite.
CIRCULAR_EXCLUSIONS["delta_family_conflict_ss_p"] = {
    "across_exclude": (
        {
            "family_conflict_ss_p",
            "family_conflict_ss_k",
            "delta_family_conflict_ss_p",
            "delta_family_conflict_ss_k",
            "frequent_family_conflict_p",
            "family_lose_temper_rare_p",
            "family_peaceful_p",
            "family_believe_not_raise_voice_p",
            "parents_argue_more_p",
        }  # parent arguing: directly taps family conflict
        | {
            "fam_fight_often_k",
            "fam_no_open_anger_k",
            "fam_no_lose_temps_k",
            "fam_criticize_often_k",
            "fam_hit_each_other_k",
            "fam_throw_things_k",
            "fam_keep_peace_k",
            "fam_try_one_up_k",
            "fam_no_raise_voices_k",
        }  # all 9 FES conflict subscale items (youth)
    ),
    "within_exclude": {"Family Dynamics & Parenting"},
}

# delta_parent_financial_trouble_p: SES / financial hardship change score.
# Exclude financial hardship items that operationalise the same construct.
CIRCULAR_EXCLUSIONS["delta_parent_financial_trouble_p"] = {
    "across_exclude": (
        {
            "parent_financial_trouble_p",
            "delta_parent_financial_trouble_p",
            "parent_financial_failures_p",
            "delta_parent_financial_failures_p",
        }
        | {
            "struggle_food_expenses",
            "couldnt_afford_rent_mortgage",
            "couldnt_afford_phone",
            "evicted",
            "gas_electric_oil_turned_off",
        }  # all material-hardship indicators
        | {
            "parent_fails_to_pay_debts_p",
            "parent_money_management_trouble_p",
            "positive_finance_p",
        }  # ASR financial items + inverse construct
        | {"parent_income", "area_deprivation_idx"}  # SES composites that subsume financial hardship
    ),
    "within_exclude": {"SES & Mobility"},
}

# delta_parent_somatic_problems_D_p: ASR Somatic Complaints broadband change.
# Inherits from _PARENT_INTERNAL_ACROSS since it is a constituent of the
# internalizing broadband; within-exclude covers mood and somatic categories.
CIRCULAR_EXCLUSIONS["delta_parent_somatic_problems_D_p"] = {
    "across_exclude": _PARENT_INTERNAL_ACROSS,
    "within_exclude": {
        "Parent Mood Issues",
        "Parent Anxiety",
        "Parent Anxiety and Fear-Related Issues",
        "Parent Other Psychopathology",
        "Parent Psychopathology Composite",
    },
}


def get_circular_exclusions(target, timepoint=None):
    """
    Return (across_exclude_set, within_exclude_set) for a given target.

    Targets absent from CIRCULAR_EXCLUSIONS return empty sets: a safe default
    that applies no exclusions rather than failing silently on novel targets.

    Parameters
    target    : str       Target variable name (e.g. 'depress_D_p', 'adhd_D_p').
    timepoint : int|None  When provided, T1/T3/T4-specific extras are merged in.

    Returns
    across_exclude : set  Variable names to remove from the flat predictor pool.
    within_exclude : set  Category names to skip in within-category analysis.
    """
    entry = CIRCULAR_EXCLUSIONS.get(target, {"across_exclude": set(), "within_exclude": set()})
    across = set(entry.get("across_exclude", set()))
    within = set(entry.get("within_exclude", set()))
    if timepoint in (1, 3, 4):
        across |= entry.get("across_exclude_t1t3t4", set())
    return across, within


_PARENT_TARGET_OVERRIDES = {"hopeless_B_p", "ptsd_diagnosis_present_B_p"}


def is_parent_target(target):
    """Return True if the target uses the parent feature structure (ASR-derived).
    NOTE: hopeless_B_p and ptsd_diagnosis_present_B_p are parent targets that
    do not follow the parent_ prefix convention."""
    return target.startswith("parent_") or target.startswith("delta_parent_") or target in _PARENT_TARGET_OVERRIDES
