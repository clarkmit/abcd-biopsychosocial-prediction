"""Domain ablation and feature-domain analysis (the `ablation` stage, Supplement S1.19).

Domain ablation quantifying each measurement domain's marginal
contribution by retraining models per ablation condition. Four analyses:

1. Binary ablation (objective vs subjective) with bootstrap CIs of the
   difference
2. Spectrum ablation over the seven levels of the objective/subjective
   gradient and its cross-domain set, each domain on its own
3. Domain permutation importance (shuffle within a domain, rescore the
   full model)
4. Top-K feature ablation curve (remove ranked features progressively)

The domain definitions are a fixed copy of the T2 objective/subjective
gradient, so the stage does not need `setup.objective_subjective_spectrum`.
It requires the `main` stage of the same run.
"""

import copy
import logging
import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.metrics import r2_score, roc_auc_score

from bpsp.context import ctx
from bpsp.data.preprocessing import bootstrap_ci, create_lightweight_model, detect_task_type
from bpsp.reporting import _RS, cache_load

logger = logging.getLogger(__name__)

# Domain registry (fixed copy of the T2 objective/subjective gradient, _obj_sub_t2)
# Each entry: short label → {full_name, vars (set of base names), ohe_bases}
# One-hot encoded bases (sex, L_site_id, etc.) are listed separately so
# the feature classifier can prefix-match their expanded column names.

_DOMAIN_REGISTRY = {
    "Obj No-Report": {
        "full_name": "Objective No-Report (Cog Tasks, Neuro, Physio, Ances Genes)",
        "vars": {
            "birth_weight_p",
            "desc_african_AFR_B",
            "desc_alaska_native_AMR_B",
            "desc_asian_indian_SAS_B",
            "desc_chinese_EAS_B",
            "desc_european_EUR_B",
            "desc_japanese_EAS_B",
            "desc_korean_EAS_B",
            "desc_latin_B",
            "desc_native_american_AMR_B",
            "desc_other_south_asian_SAS_B",
            "desc_vietnamese_EAS_B",
            "fitbit_fairlyactive_mins",
            "fitbit_lightlyactive_mins",
            "fitbit_resting_hr",
            "fitbit_sedentary_mins",
            "fitbit_steps",
            "fitbit_veryactive_mins",
            "gd_riskybets",
            "gd_safebets",
            "height",
            "lmt_accuracy",
            "lmt_correct_mrt",
            "lmt_correct_nt",
            "lmt_efficiency",
            "lmt_mrt",
            "mid_acceptable_performance",
            "mid_mrt_lgrw",
            "mid_mrt_smrw",
            "mid_num_trials",
            "mid_total_payout",
            "nb2_D_prime_neg",
            "nb2_D_prime_pos",
            "nb2_accuracy_neg",
            "nb2_accuracy_pos",
            "nb2_resp_bias_neg",
            "nb2_resp_bias_pos",
            "nb_correct_mrt",
            "nb_correct_mrt_2back",
            "nb_correct_mrt_neg",
            "nb_correct_mrt_pos",
            "nb_correct_nt",
            "nb_correct_nt_2back",
            "nb_correct_nt_neg",
            "nb_correct_nt_pos",
            "no_sports_activities_p",
            "pc_DTI1",
            "pc_DTI10",
            "pc_DTI11",
            "pc_DTI12",
            "pc_DTI13",
            "pc_DTI14",
            "pc_DTI15",
            "pc_DTI16",
            "pc_DTI17",
            "pc_DTI18",
            "pc_DTI19",
            "pc_DTI2",
            "pc_DTI20",
            "pc_DTI21",
            "pc_DTI22",
            "pc_DTI23",
            "pc_DTI24",
            "pc_DTI25",
            "pc_DTI26",
            "pc_DTI27",
            "pc_DTI28",
            "pc_DTI29",
            "pc_DTI3",
            "pc_DTI30",
            "pc_DTI31",
            "pc_DTI32",
            "pc_DTI33",
            "pc_DTI34",
            "pc_DTI35",
            "pc_DTI36",
            "pc_DTI37",
            "pc_DTI38",
            "pc_DTI39",
            "pc_DTI4",
            "pc_DTI40",
            "pc_DTI41",
            "pc_DTI42",
            "pc_DTI43",
            "pc_DTI44",
            "pc_DTI45",
            "pc_DTI46",
            "pc_DTI47",
            "pc_DTI48",
            "pc_DTI49",
            "pc_DTI5",
            "pc_DTI50",
            "pc_DTI51",
            "pc_DTI52",
            "pc_DTI53",
            "pc_DTI54",
            "pc_DTI55",
            "pc_DTI56",
            "pc_DTI57",
            "pc_DTI58",
            "pc_DTI59",
            "pc_DTI6",
            "pc_DTI60",
            "pc_DTI61",
            "pc_DTI62",
            "pc_DTI63",
            "pc_DTI64",
            "pc_DTI65",
            "pc_DTI66",
            "pc_DTI67",
            "pc_DTI68",
            "pc_DTI69",
            "pc_DTI7",
            "pc_DTI70",
            "pc_DTI71",
            "pc_DTI72",
            "pc_DTI73",
            "pc_DTI74",
            "pc_DTI75",
            "pc_DTI8",
            "pc_DTI9",
            "pc_gene_aces1",
            "pc_gene_aces10",
            "pc_gene_aces11",
            "pc_gene_aces12",
            "pc_gene_aces13",
            "pc_gene_aces14",
            "pc_gene_aces15",
            "pc_gene_aces16",
            "pc_gene_aces17",
            "pc_gene_aces18",
            "pc_gene_aces19",
            "pc_gene_aces2",
            "pc_gene_aces20",
            "pc_gene_aces21",
            "pc_gene_aces22",
            "pc_gene_aces23",
            "pc_gene_aces24",
            "pc_gene_aces25",
            "pc_gene_aces26",
            "pc_gene_aces27",
            "pc_gene_aces28",
            "pc_gene_aces29",
            "pc_gene_aces3",
            "pc_gene_aces30",
            "pc_gene_aces31",
            "pc_gene_aces32",
            "pc_gene_aces4",
            "pc_gene_aces5",
            "pc_gene_aces6",
            "pc_gene_aces7",
            "pc_gene_aces8",
            "pc_gene_aces9",
            "pc_rsFMRI1",
            "pc_rsFMRI10",
            "pc_rsFMRI11",
            "pc_rsFMRI12",
            "pc_rsFMRI13",
            "pc_rsFMRI14",
            "pc_rsFMRI15",
            "pc_rsFMRI16",
            "pc_rsFMRI17",
            "pc_rsFMRI18",
            "pc_rsFMRI19",
            "pc_rsFMRI2",
            "pc_rsFMRI20",
            "pc_rsFMRI21",
            "pc_rsFMRI22",
            "pc_rsFMRI23",
            "pc_rsFMRI24",
            "pc_rsFMRI25",
            "pc_rsFMRI26",
            "pc_rsFMRI27",
            "pc_rsFMRI28",
            "pc_rsFMRI29",
            "pc_rsFMRI3",
            "pc_rsFMRI30",
            "pc_rsFMRI31",
            "pc_rsFMRI32",
            "pc_rsFMRI33",
            "pc_rsFMRI34",
            "pc_rsFMRI35",
            "pc_rsFMRI36",
            "pc_rsFMRI37",
            "pc_rsFMRI38",
            "pc_rsFMRI39",
            "pc_rsFMRI4",
            "pc_rsFMRI40",
            "pc_rsFMRI41",
            "pc_rsFMRI42",
            "pc_rsFMRI43",
            "pc_rsFMRI44",
            "pc_rsFMRI45",
            "pc_rsFMRI46",
            "pc_rsFMRI47",
            "pc_rsFMRI48",
            "pc_rsFMRI49",
            "pc_rsFMRI5",
            "pc_rsFMRI50",
            "pc_rsFMRI51",
            "pc_rsFMRI52",
            "pc_rsFMRI53",
            "pc_rsFMRI54",
            "pc_rsFMRI55",
            "pc_rsFMRI56",
            "pc_rsFMRI57",
            "pc_rsFMRI58",
            "pc_rsFMRI59",
            "pc_rsFMRI6",
            "pc_rsFMRI60",
            "pc_rsFMRI61",
            "pc_rsFMRI62",
            "pc_rsFMRI63",
            "pc_rsFMRI64",
            "pc_rsFMRI65",
            "pc_rsFMRI66",
            "pc_rsFMRI67",
            "pc_rsFMRI68",
            "pc_rsFMRI69",
            "pc_rsFMRI7",
            "pc_rsFMRI70",
            "pc_rsFMRI71",
            "pc_rsFMRI72",
            "pc_rsFMRI73",
            "pc_rsFMRI74",
            "pc_rsFMRI75",
            "pc_rsFMRI8",
            "pc_rsFMRI9",
            "pc_struct1",
            "pc_struct10",
            "pc_struct11",
            "pc_struct12",
            "pc_struct13",
            "pc_struct14",
            "pc_struct15",
            "pc_struct16",
            "pc_struct17",
            "pc_struct18",
            "pc_struct19",
            "pc_struct2",
            "pc_struct20",
            "pc_struct21",
            "pc_struct22",
            "pc_struct23",
            "pc_struct24",
            "pc_struct25",
            "pc_struct26",
            "pc_struct27",
            "pc_struct28",
            "pc_struct29",
            "pc_struct3",
            "pc_struct30",
            "pc_struct31",
            "pc_struct32",
            "pc_struct33",
            "pc_struct34",
            "pc_struct35",
            "pc_struct36",
            "pc_struct37",
            "pc_struct38",
            "pc_struct39",
            "pc_struct4",
            "pc_struct40",
            "pc_struct41",
            "pc_struct42",
            "pc_struct43",
            "pc_struct44",
            "pc_struct45",
            "pc_struct46",
            "pc_struct47",
            "pc_struct48",
            "pc_struct49",
            "pc_struct5",
            "pc_struct50",
            "pc_struct51",
            "pc_struct52",
            "pc_struct53",
            "pc_struct54",
            "pc_struct55",
            "pc_struct56",
            "pc_struct57",
            "pc_struct58",
            "pc_struct59",
            "pc_struct6",
            "pc_struct60",
            "pc_struct61",
            "pc_struct62",
            "pc_struct63",
            "pc_struct64",
            "pc_struct65",
            "pc_struct66",
            "pc_struct67",
            "pc_struct68",
            "pc_struct69",
            "pc_struct7",
            "pc_struct70",
            "pc_struct71",
            "pc_struct72",
            "pc_struct73",
            "pc_struct74",
            "pc_struct75",
            "pc_struct8",
            "pc_struct9",
            "ravlt_l_intrusions",
            "ravlt_l_repitition",
            "ravlt_l_total",
            "ravlt_s_intrusions",
            "ravlt_s_repitition",
            "ravlt_s_total",
            "sst_aG1",
            "sst_aG2",
            "sst_aS",
            "sst_absdeltaCV",
            "sst_absdeltaRV",
            "sst_absdeltaSD",
            "sst_acceptable_performance",
            "sst_bG",
            "sst_betasCV",
            "sst_betasSD",
            "sst_betasV",
            "sst_bs2CV",
            "sst_bs2SD",
            "sst_bs2V",
            "sst_dG",
            "sst_dS",
            "sst_factor1",
            "sst_factor2",
            "sst_factor3",
            "sst_gamma0",
            "sst_gamma1",
            "sst_kappa0",
            "sst_mean_PDR",
            "sst_mean_PDRg",
            "sst_mean_SD",
            "sst_mean_SDr",
            "sst_mean_absdelta",
            "sst_mean_bS2",
            "sst_mean_betaS",
            "sst_mean_ssrt",
            "sst_median_PDR",
            "sst_median_PDRg",
            "sst_median_SD",
            "sst_median_SDr",
            "sst_median_absdelta",
            "sst_median_bS2",
            "sst_median_betaS",
            "sst_median_ssrt",
            "sst_mu",
            "sst_pdrCV",
            "sst_pdrSD",
            "sst_pdrV",
            "sst_pdrgCV",
            "sst_pdrgSD",
            "sst_pdrgV",
            "sst_pp",
            "sst_sM",
            "sst_sdCV",
            "sst_sdRV",
            "sst_sdSD",
            "sst_sdrCV",
            "sst_sdrRV",
            "sst_sdrSD",
            "sst_ssrt_int_est",
            "sst_ssrt_mean_est",
            "sst_theta0",
            "sst_theta1",
            "sst_xM",
            "tb_cardsort",
            "tb_flanker",
            "tb_list",
            "tb_pattern",
            "tb_picture",
            "tb_picvocab",
            "tb_reading",
            "waist",
            "weight",
        },
        "ohe_bases": ["sex"],
    },
    "Obj Rated": {
        "full_name": "Objective Rated (SES, Resid, Grades, Diet)",
        "vars": {
            "added_sugar",
            "carbohydrate_intake",
            "dairy_intake",
            "fiber_intake",
            "fruit_intake",
            "legume_intake",
            "neighborhood_safe_y",
            "neighborhood_safety_ss_p",
            "num_brothers_p",
            "num_sisters_p",
            "parent_age",
            "parent_education",
            "parent_fails_to_pay_debts_p",
            "parent_financial_trouble_p",
            "parent_income",
            "parent_work_absences_p",
            "positive_finance_p",
            "potassium_intake",
            "protein_intake",
            "protein_sources_intake",
            "relig_importance",
            "religious_service_frequency",
            "resid_crime_drug",
            "resid_crime_dui",
            "resid_crime_tot",
            "resid_crime_violent",
            "resid_density",
            "resid_immigrant_bias",
            "resid_lead_risk",
            "resid_lead_risk_houses_perc",
            "resid_lead_risk_poverty",
            "resid_no2_avg",
            "resid_pm25_avg",
            "resid_prox_roads",
            "resid_racism",
            "resid_sex_orient_bias",
            "resid_sexism",
            "resid_walkability",
            "saturated_fat",
            "sodium_intake",
            "struggle_food_expenses",
            "sugary_beverage_freq",
            "total_calories",
            "total_sugar",
            "vegetable_intake",
            "whole_grain_intake",
        },
        "ohe_bases": ["L_site_id", "child_religion", "marital_status", "sex_P"],
    },
    "Child Self": {
        "full_name": "Child-Self Measures (School, Family, Social)",
        "vars": {
            "bis_behav_inhibition_ss_k",
            "bis_drive_ss_k",
            "bis_funseeking_ss_k",
            "bis_reward_responsive_ss_k",
            "bullied_on_internet_k",
            "child_newschool_p",
            "close_boy_friends_k",
            "close_girl_friends_k",
            "discrimination_ss_k",
            "disobeys_at_school_k",
            "doesnt_feel_accepted_k",
            "enjoys_school_k",
            "excluded_k",
            "fam_criticize_often_k",
            "fam_fight_often_k",
            "fam_hit_each_other_k",
            "fam_keep_peace_k",
            "fam_no_lose_temps_k",
            "fam_no_open_anger_k",
            "fam_no_raise_voices_k",
            "fam_throw_things_k",
            "fam_try_one_up_k",
            "feels_discriminated_adults_not_school_k",
            "feels_discriminated_americans_k",
            "feels_discriminated_k",
            "feels_discriminated_students_k",
            "feels_discriminated_teachers_k",
            "feels_leftout_k",
            "feels_smart_k",
            "feels_threatned_k",
            "feels_unwanted_american_society_k",
            "feelsafe_at_school_k",
            "finds_schoolboring_k",
            "getalong_teachers_k",
            "grades_important_k",
            "not_invited_k",
            "otherkids_gossip_k",
            "otherkids_saymeanthings_k",
            "otherkids_spreadneg_rumors_k",
            "overt_aggression_ss_k",
            "overt_victimization_ss_k",
            "p_comm_cohesion_ss",
            "p_comm_collective_efficacy_ss",
            "p_comm_ctrl_ss",
            "peer_net_protective_ss_k",
            "peers_beh_delinquent_ss_k",
            "peers_beh_prosocial_ss_k",
            "prosocial_ss_k",
            "prosocial_ss_p",
            "relational_aggression_ss_k",
            "relational_victimization_ss_k",
            "repeated_grade",
            "reputational_aggression_ss_k",
            "reputational_victimization_ss_k",
            "saysmeanthings_others_k",
            "school_disengagement_ss_k",
            "school_environment_ss_k",
            "school_involvement_ss_k",
            "senses_racism_k",
            "socialinfluence_meanfinal_k",
            "up_lackofplanning_ss_k",
            "up_lackperseverance_ss_k",
            "up_negative_urgency_ss_k",
            "up_positiveurgency_ss_k",
            "up_sensationseeking_ss_k",
        },
        "ohe_bases": [],
    },
    "Parent→Child": {
        "full_name": "Parent-to-Child Measures (Psychopathology, Personality, Social)",
        "vars": {
            "argues_p",
            "attacks_others_p",
            "blames_others_p",
            "body_aches_p",
            "bragadocious_p",
            "breaks_rules_p",
            "bullies_others_p",
            "cant_concentrate_p",
            "clings_to_adults_p",
            "constipated_p",
            "demands_attention_p",
            "destroys_others_things_p",
            "destroys_own_things_p",
            "disagreeable_p",
            "disobedient_home_p",
            "disobedient_school_p",
            "doesnt_finish_p",
            "doesnt_get_along_p",
            "easily_embarrassed_p",
            "easily_jealous_p",
            "easily_offended_p",
            "eye_problems_p",
            "fears_being_bad_p",
            "fears_excl_school_p",
            "fears_school_p",
            "fights_p",
            "frequent_headaches_p",
            "frequent_stomachaches_p",
            "goal_continuity_p",
            "hyperactive_p",
            "impulsive_p",
            "loquacious_p",
            "lying_p",
            "medhx_doctorvisit_p",
            "medhx_emergencyroom_p",
            "medhx_p",
            "nausea_p",
            "nervous_general_p",
            "nervous_twitching_p",
            "not_critical_others_p",
            "not_liked_p",
            "paranoid_p",
            "perfectionist_p",
            "scared_dark_p",
            "school_excitement_p",
            "secretive_p",
            "skin_problems_p",
            "sociable_p",
            "social_fear_present_PK",
            "steals_home_p",
            "steals_outside_p",
            "stubborn_p",
            "temper_tantrums_p",
            "threatens_others_p",
            "vomiting_p",
            "whines_p",
            "wishes_other_sex_p",
            "worries_p",
        },
        "ohe_bases": [],
    },
    "Adverse Life": {
        "full_name": "Child & Parent Ratings (Adverse Life Events)",
        "vars": {
            "adult_family_fighting_victim_p",
            "b_lifeevents_affected_ss_k",
            "b_lifeevents_affected_ss_p",
            "b_lifeevents_ss_k",
            "b_lifeevents_ss_p",
            "beating_victim_home_p",
            "big_accident_need_treatment_p",
            "car_accident_hurt_p",
            "domestic_child_sexually_abuse_victim_p",
            "experienced_crime_p",
            "family_threatened_child_victim_p",
            "fire_victim_p",
            "foreign_child_sexually_abuse_victim_p",
            "g_lifeevents_ss_k",
            "g_lifeevents_ss_p",
            "natural_disaster_victim_p",
            "peer_child_sexually_abuse_victim_p",
            "stabbing_shooting_victim_community_p",
            "stabbing_shooting_victim_home_p",
            "stabbing_shooting_witness_p",
            "stranger_threatened_child_victim_p",
            "sudden_death_in_family_p",
            "terrorism_victim_p",
            "war_death_witness_p",
        },
        "ohe_bases": [],
    },
    "Parent Self (Psych)": {
        "full_name": "Parent-Self Measures (Psychopathology, Personality, Social)",
        "vars": {
            "delta_parent_aches_pains_p",
            "delta_parent_bad_family_relationship_p",
            "delta_parent_bad_relationships_p",
            "delta_parent_concentration_trouble_p",
            "delta_parent_drinks_too_much_p",
            "delta_parent_drug_use_p",
            "delta_parent_feels_overwhelmed_p",
            "delta_parent_feels_unloved_p",
            "delta_parent_financial_failures_p",
            "delta_parent_friendship_trouble_p",
            "delta_parent_meets_family_duties_p",
            "delta_parent_not_liked_by_others_p",
            "delta_parent_planning_trouble_p",
            "delta_parent_poor_work_performance_p",
            "delta_parent_sleep_trouble_p",
            "delta_parent_somatic_problems_D_p",
            "delta_parent_stubborn_irritable_p",
            "delta_parent_worries_a_lot_p",
            "delta_parent_worries_about_family_p",
            "parent_associates_with_trouble_p",
            "parent_attention_seeking_p",
            "parent_avoids_talking_p",
            "parent_bad_family_relationship_p",
            "parent_bad_opposite_sex_relationship_p",
            "parent_bad_relationships_p",
            "parent_behavior_changeable_p",
            "parent_bragging_p",
            "parent_clowns_or_shows_off_p",
            "parent_clumsy_p",
            "parent_concentration_trouble_p",
            "parent_confused_p",
            "parent_cries_a_lot_p",
            "parent_decision_trouble_p",
            "parent_depressed_p",
            "parent_destroys_others_things_p",
            "parent_destroys_own_things_p",
            "parent_disorganized_p",
            "parent_doesnt_eat_well_p",
            "parent_doesnt_finish_tasks_p",
            "parent_drives_too_fast_p",
            "parent_easily_bored_p",
            "parent_enjoys_little_p",
            "parent_fear_of_bad_thoughts_p",
            "parent_fearful_or_anxious_p",
            "parent_feels_inferior_p",
            "parent_feels_unloved_p",
            "parent_feels_unsuccessful_p",
            "parent_forgetful_p",
            "parent_friendship_trouble_p",
            "parent_happy_person_p",
            "parent_hears_voices_p",
            "parent_heart_racing_p",
            "parent_high_sleep_duration_p",
            "parent_honest_p",
            "parent_hot_temper_p",
            "parent_hyperactive_p",
            "parent_illegal_behavior_p",
            "parent_impulsive_p",
            "parent_lonely_p",
            "parent_loses_things_p",
            "parent_louder_than_others_p",
            "parent_low_energy_p",
            "parent_max_effort_p",
            "parent_meets_family_duties_p",
            "parent_money_management_trouble_p",
            "parent_no_guilt_p",
            "parent_not_good_at_details_p",
            "parent_not_liked_by_others_p",
            "parent_numbness_p",
            "parent_obsessive_thoughts_p",
            "parent_opposite_sex_wish_p",
            "parent_paranoid_p",
            "parent_physical_attacks_p",
            "parent_picks_skin_p",
            "parent_planning_trouble_p",
            "parent_prefers_older_people_p",
            "parent_prefers_to_be_alone_p",
            "parent_priority_trouble_p",
            "parent_relationship_concerns_p",
            "parent_repeats_acts_p",
            "parent_restless_p",
            "parent_risky_decisions_p",
            "parent_secretive_p",
            "parent_sees_things_p",
            "parent_self_conscious_p",
            "parent_self_harm_p",
            "parent_sense_of_fairness_p",
            "parent_shy_or_timid_p",
            "parent_sleep_trouble_p",
            "parent_specific_fears_p",
            "parent_speech_problems_p",
            "parent_stands_up_rights_p",
            "parent_strange_behavior_p",
            "parent_strange_thoughts_p",
            "parent_stubborn_irritable_p",
            "parent_sudden_mood_changes_p",
            "parent_suicidal_thoughts_p",
            "parent_talks_too_much_p",
            "parent_tardy_often_p",
            "parent_teases_others_p",
            "parent_tired_no_reason_p",
            "parent_uses_opportunities_p",
            "parent_worries_a_lot_p",
            "parent_worries_about_family_p",
            "parent_worries_about_future_p",
            "parent_yells_a_lot_p",
        },
        "ohe_bases": [],
    },
    "Parent Self (Fam)": {
        "full_name": "Parent-Self Measures (Family Style/Dynamics, Drug Use)",
        "vars": {
            "alcohol_before_pregnancy_p",
            "alcohol_during_pregnancy_p",
            "caffeine_during_pregnancy_p",
            "cigs_before_pregnancy_p",
            "cigs_during_pregnancy_p",
            "cocaine_before_pregnancy_p",
            "cocaine_during_pregnancy_p",
            "death_in_family_p",
            "drinksperweek_during_pregnancy_p",
            "drugs_before_pregnancy_p",
            "drugs_during_pregnancy_p",
            "family_activities_ss_p",
            "family_believe_not_raise_voice_p",
            "family_conflict_ss_k",
            "family_conflict_ss_p",
            "family_expression_ss_p",
            "family_intellectual_ss_p",
            "family_lose_temper_rare_p",
            "family_move_p",
            "family_not_talk_aboutfeelings_p",
            "family_open_discussing_anything_p",
            "family_organisation_ss_p",
            "family_peaceful_p",
            "father_alcohol",
            "father_druguse",
            "frequent_family_conflict_p",
            "hallucinogen_current_B_p",
            "hallucinogen_use_history_B_p",
            "heroin_before_pregnancy_p",
            "heroin_during_pregnancy_p",
            "mother_alcohol",
            "mother_druguse",
            "parent_drinks_frequency_p",
            "parent_drinks_too_much_p",
            "parent_drug_days_nonmedical_p",
            "parent_drug_use_p",
            "parent_drunk_days_p",
            "parent_family_responsibilities_p",
            "parent_monitoring_ss_k",
            "parent_tobacco_use_frequency_p",
            "parents_divorced_p",
            "prescriptionmed_pregnancy_p",
            "sedative_hypnotic_anxiolytic_use_B_p",
            "weed_before_pregnancy_p",
            "weed_during_pregnancy_p",
        },
        "ohe_bases": [],
    },
    "Cross-Domain": {
        "full_name": "Cross-Domain / Inter-Category Variables",
        "vars": {
            "b_lifeevents_affected_ss_k",
            "b_lifeevents_ss_k",
            "b_lifeevents_ss_p",
            "bad_grades",
            "cct",
            "child_white",
            "close_boy_friends_k",
            "close_girl_friends_k",
            "delta_not_liked_p",
            "family_conflict_ss_k",
            "feels_leftout_k",
            "gd_riskybets",
            "mid_total_payout",
            "nb_correct_nt_2back",
            "num_brothers_p",
            "num_sisters_p",
            "parent_age_grouped",
            "parent_income",
            "puberty_k",
            "relig_importance",
            "screentime_daysperweek_k",
            "screentime_weekday_ss_k",
            "social_problems_D_p",
            "sst_ssrt_mean_est",
            "tb_cryst",
            "weight",
        },
        "ohe_bases": ["L_site_id", "child_religion", "sex", "sex_P", "sex_orient_y", "trans_id_y"],
    },
}

# Objective = first two categories; Subjective = all others
_OBJECTIVE_DOMAINS = {"Obj No-Report", "Obj Rated"}
_SUBJECTIVE_DOMAINS = {k for k in _DOMAIN_REGISTRY if k not in _OBJECTIVE_DOMAINS}

# Domain colors (ordered from most objective → most subjective)
_DOMAIN_COLORS = {
    "Obj No-Report": "#1565C0",
    "Obj Rated": "#0097A7",
    "Child Self": "#2E7D32",
    "Parent→Child": "#F57F17",
    "Adverse Life": "#E65100",
    "Parent Self (Psych)": "#AD1457",
    "Parent Self (Fam)": "#6A1B9A",
    "Cross-Domain": "#78909C",
    "Objective": "#1565C0",
    "Subjective": "#AD1457",
    "Combined (Full)": "#37474F",
}

# Feature classifier
# OHE collision map: sex_ must not match sex_orient_y_, trans_id_y_, etc.
_OHE_SAFE_PREFIXES = {
    "L_site_id": "L_site_id_",
    "sex_P": "sex_P_",
    "marital_status": "marital_status_",
    "child_religion": "child_religion_",
}
# sex needs special guard: sex_ but not sex_orient*
_OHE_SEX_PREFIX = "sex_"
_OHE_SEX_EXCLUDE = ("sex_orient", "sex_P")
# Subjective OHE bases
_OHE_SUBJ_PREFIXES = {
    "sex_orient_y": "sex_orient_y_",
    "trans_id_y": "trans_id_y_",
    "crysflu_c": "crysflu_c_",
    "anxadhd_c": "anxadhd_c_",
}


def _classify_feature(f):
    """Return the domain label for a feature column name.
    Handles one-hot expanded columns via prefix matching.
    Returns None if no domain claims the feature."""
    # 1. Exact match across all domains
    for domain, info in _DOMAIN_REGISTRY.items():
        if f in info["vars"]:
            return domain
    # 2. OHE prefix match: check each domain's ohe_bases
    for domain, info in _DOMAIN_REGISTRY.items():
        for base in info["ohe_bases"]:
            if base == "sex":
                if f.startswith(_OHE_SEX_PREFIX) and not f.startswith(_OHE_SEX_EXCLUDE):
                    return domain
            elif base in _OHE_SAFE_PREFIXES:
                if f.startswith(_OHE_SAFE_PREFIXES[base]):
                    return domain
            elif base in _OHE_SUBJ_PREFIXES:
                if f.startswith(_OHE_SUBJ_PREFIXES[base]):
                    return domain
    return None


def _build_feature_domain_map(feature_list):
    """Map every feature in feature_list to a domain. Returns dict[feature] → domain."""
    f2d = {}
    for f in feature_list:
        domain = _classify_feature(f)
        f2d[f] = domain if domain else "Unclassified"
    return f2d


# Analytics utilities


def _bootstrap_delta_ci(y_true, y_pred_a, y_pred_b, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap CI for the difference metric(A) - metric(B), paired samples."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            sa = metric_fn(y_true[idx], y_pred_a[idx])
            sb = metric_fn(y_true[idx], y_pred_b[idx])
            deltas.append(sa - sb)
        except Exception:
            continue
    deltas = np.array(deltas)
    alpha = 1 - ci
    return np.median(deltas), np.percentile(deltas, 100 * alpha / 2), np.percentile(deltas, 100 * (1 - alpha / 2))


def _domain_permutation_importance(model, X_test, y_test, f2d, metric_fn, is_clf=True, n_repeats=10, seed=42):
    """Permute all features in each domain, measure score drop."""
    rng = np.random.RandomState(seed)
    # Baseline score
    if is_clf:
        base_pred = model.predict_proba(X_test)[:, 1]
    else:
        base_pred = model.predict(X_test)
    base_score = metric_fn(y_test, base_pred)

    # Group features by domain
    domain_features = {}
    for f, d in f2d.items():
        if f in X_test.columns:
            domain_features.setdefault(d, []).append(f)

    results = {}
    for domain, feats in domain_features.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            for f in feats:
                X_perm[f] = rng.permutation(X_perm[f].values)
            if is_clf:
                perm_pred = model.predict_proba(X_perm)[:, 1]
            else:
                perm_pred = model.predict(X_perm)
            perm_score = metric_fn(y_test, perm_pred)
            drops.append(base_score - perm_score)
        results[domain] = {
            "mean_drop": np.mean(drops),
            "std_drop": np.std(drops, ddof=1),
            "drops": drops,
            "n_features": len(feats),
        }
    return base_score, results


# Main body function


def _run_domain_ablation():
    """Run domain ablation analyses for CURRENT globals."""

    # Gate check
    _skip = False
    for _req in ["X_train", "X_test", "y_train", "y_test"]:
        if _req not in ctx:
            logger.info(f"Error: {_req} not found. Run the Main Analysis Runner first.")
            _skip = True
    if "bootstrap_ci" not in ctx:
        logger.info("Warning: bootstrap_ci not found. Run setup first.")
        _skip = True
    if _skip:
        return

    # Clear stale _RS sub-key
    _RS.pop("domain_ablation", None)

    # Detect within vs across-category mode
    _split = "across_categories"
    if "split" in ctx and isinstance(ctx.split, str):
        _split = ctx.split
    # Also check _batch_archive entry
    if "_vd_da" in dir() and isinstance(ctx._vd_da, dict):
        _split = ctx._vd_da.get("split", _split)

    _is_within = _split == "within_categories"
    if _is_within:
        logger.info("  Warning: Within-category mode detected -- domain spectrum analyses will be")
        logger.info("    skipped (feature set is from a single construct domain).")
        logger.info("    Running: Top-K Feature Ablation only.\n")

    _target = ctx.target_options if "target_options" in ctx else "target"
    _tp = ctx.tp_option if "tp_option" in ctx else "?"
    _task = detect_task_type(ctx.y_train)
    _is_clf = _task == "classification"
    _metric = "AUC" if _is_clf else "R²"
    _metric_fn = roc_auc_score if _is_clf else r2_score
    _y_tr = np.asarray(ctx.y_train).ravel()
    _y_te = np.asarray(ctx.y_test).ravel()
    _all_features = list(ctx.X_train.columns)

    logger.info("=" * 72)
    logger.info("   DOMAIN ABLATION & FEATURE-DOMAIN ANALYSIS")
    logger.info(f"  Target: {_target}  |  Timepoint: T{_tp}  |  Task: {_task}")
    logger.info("=" * 72)

    _analyses = []
    if ctx.run_binary_ablation and not _is_within:
        _analyses.append("Binary")
    if ctx.run_spectrum_ablation and not _is_within:
        _analyses.append("Spectrum")
    if ctx.run_domain_permutation and not _is_within:
        _analyses.append("Permutation")
    if ctx.run_topk_ablation:
        _analyses.append("Top-K")
    logger.info(f"  Running: {', '.join(_analyses) if _analyses else 'None'}")
    logger.info(f"  Features: {len(_all_features)}  |  N_train: {len(ctx.X_train)}  |  N_test: {len(ctx.X_test)}")

    # Build feature→domain map
    _f2d = _build_feature_domain_map(_all_features)
    _domain_counts = {}
    for f, d in _f2d.items():
        _domain_counts[d] = _domain_counts.get(d, 0) + 1
    logger.info("\n  Feature-domain mapping:")
    for d in list(_DOMAIN_REGISTRY.keys()) + ["Unclassified"]:
        if d in _domain_counts:
            logger.info(f"    {d:>22s}: {_domain_counts[d]:>4d} features")

    # Collect features by binary bucket
    _obj_features = [f for f, d in _f2d.items() if d in _OBJECTIVE_DOMAINS]
    _sub_features = [f for f, d in _f2d.items() if d in _SUBJECTIVE_DOMAINS or d == "Unclassified"]

    # Storage dicts
    _binary_results = {}
    _spectrum_results = {}
    _perm_results = {}
    _topk_results = {}
    _delta_cis = {}
    _predictions = {}  # store predictions for bootstrap delta

    # 1. Binary ablation (Obj vs Subj vs Full) + Bootstrap Δ CIs
    if ctx.run_binary_ablation and not _is_within:
        display(
            HTML(
                '<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #7B1FA2;'
                "background:linear-gradient(135deg,#F3E5F5,#F5F5F5);padding:14px 20px;"
                'margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
                '<span style="font-size:15px;font-weight:700;color:#7B1FA2;">Binary Domain Ablation</span><br>'
                '<span style="font-size:12px;color:#546E7A;">Objective vs Subjective vs Combined (Full) -- with bootstrap Δ CIs</span>'
                "</div>"
            )
        )

        for _label, _feat_list in [
            ("Subjective", _sub_features),
            ("Objective", _obj_features),
            ("Combined (Full)", _all_features),
        ]:
            if len(_feat_list) < 2:
                logger.info(f"  Warning: {_label}: only {len(_feat_list)} features -- skipping")
                continue

            _X_tr_sub = ctx.X_train[[f for f in _feat_list if f in ctx.X_train.columns]]
            _X_te_sub = ctx.X_test[[f for f in _feat_list if f in ctx.X_test.columns]]

            _m = create_lightweight_model(is_classification=_is_clf, n_features=len(_feat_list))
            _m.fit(_X_tr_sub, _y_tr)

            if _is_clf:
                _pred = _m.predict_proba(_X_te_sub)[:, 1]
            else:
                _pred = _m.predict(_X_te_sub)

            _score = _metric_fn(_y_te, _pred)
            _point, _lo, _hi = bootstrap_ci(_y_te, _pred, _metric_fn, n_boot=1000, seed=42)

            _binary_results[_label] = {"score": _score, "ci_lo": _lo, "ci_hi": _hi, "n_features": _X_tr_sub.shape[1]}
            _predictions[_label] = _pred.copy()
            logger.info(
                f"  {_label:>20s}: {_metric} = {_score:.4f} [{_lo:.4f}, {_hi:.4f}]  ({_X_tr_sub.shape[1]} features)"
            )

        # Bootstrap Δ CIs (paired)
        if "Combined (Full)" in _predictions and "Subjective" in _predictions:
            _d_med, _d_lo, _d_hi = _bootstrap_delta_ci(
                _y_te, _predictions["Combined (Full)"], _predictions["Subjective"], _metric_fn, n_boot=1000, seed=42
            )
            _delta_cis["Full_vs_Subj"] = {"median": _d_med, "ci_lo": _d_lo, "ci_hi": _d_hi}
            logger.info(f"\n  Δ (Full − Subj):  {_d_med:+.4f} [{_d_lo:+.4f}, {_d_hi:+.4f}]")

        if "Combined (Full)" in _predictions and "Objective" in _predictions:
            _d_med, _d_lo, _d_hi = _bootstrap_delta_ci(
                _y_te, _predictions["Combined (Full)"], _predictions["Objective"], _metric_fn, n_boot=1000, seed=42
            )
            _delta_cis["Full_vs_Obj"] = {"median": _d_med, "ci_lo": _d_lo, "ci_hi": _d_hi}
            logger.info(f"  Δ (Full − Obj):   {_d_med:+.4f} [{_d_lo:+.4f}, {_d_hi:+.4f}]")

        if "Subjective" in _predictions and "Objective" in _predictions:
            _d_med, _d_lo, _d_hi = _bootstrap_delta_ci(
                _y_te, _predictions["Subjective"], _predictions["Objective"], _metric_fn, n_boot=1000, seed=42
            )
            _delta_cis["Subj_vs_Obj"] = {"median": _d_med, "ci_lo": _d_lo, "ci_hi": _d_hi}
            logger.info(f"  Δ (Subj − Obj):   {_d_med:+.4f} [{_d_lo:+.4f}, {_d_hi:+.4f}]")

        # Figure 1: Horizontal lollipop
        if len(_binary_results) >= 2:
            fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
            _labels_sorted = ["Combined (Full)", "Subjective", "Objective"]
            _labels_sorted = [l for l in _labels_sorted if l in _binary_results]
            _y_pos = list(range(len(_labels_sorted)))

            for _i, _lab in enumerate(_labels_sorted):
                _r = _binary_results[_lab]
                _c = _DOMAIN_COLORS.get(_lab, "#78909C")
                ax.barh(_i, _r["score"], height=0.45, color=_c, alpha=0.75, zorder=3, edgecolor="white", linewidth=0.6)
                ax.errorbar(
                    _r["score"],
                    _i,
                    xerr=[[_r["score"] - _r["ci_lo"]], [_r["ci_hi"] - _r["score"]]],
                    fmt="o",
                    color=_c,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=1.2,
                    capsize=4,
                    capthick=1.0,
                    ecolor="#333333",
                    zorder=4,
                )
                ax.text(
                    _r["ci_hi"] + 0.008,
                    _i,
                    f"{_r['score']:.4f}  [{_r['ci_lo']:.3f}, {_r['ci_hi']:.3f}]  ({_r['n_features']} feat.)",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=_c,
                )

            # Delta bracket annotations
            if "Subj_vs_Obj" in _delta_cis:
                _dc = _delta_cis["Subj_vs_Obj"]
                _sig = "●" if _dc["ci_lo"] > 0 or _dc["ci_hi"] < 0 else "○"
                ax.annotate(
                    f"Δ = {_dc['median']:+.4f} [{_dc['ci_lo']:+.3f}, {_dc['ci_hi']:+.3f}] {_sig}",
                    xy=(0.02, 0.02),
                    xycoords="axes fraction",
                    fontsize=7.5,
                    color="#546E7A",
                    fontstyle="italic",
                )

            ax.set_yticks(_y_pos)
            ax.set_yticklabels(_labels_sorted, fontsize=9.5, fontweight="bold")
            ax.set_xlabel(_metric, fontsize=9)
            ax.set_title(f"Binary Domain Ablation -- {_target} (T{_tp})", fontsize=10, fontweight="bold")
            ax.set_xlim(left=max(0, min(r["ci_lo"] for r in _binary_results.values()) - 0.05))
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.15, linewidth=0.5)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.tight_layout()

            try:
                _fig_dir = os.path.join(os.getcwd(), "domain_ablation_figures")
                os.makedirs(_fig_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(_fig_dir, f"binary_ablation_{_target}_T{_tp}.png"), dpi=300, bbox_inches="tight"
                )
                fig.savefig(
                    os.path.join(_fig_dir, f"binary_ablation_{_target}_T{_tp}.pdf"), dpi=300, bbox_inches="tight"
                )
            except Exception as _e:
                logger.info(f"Warning: {_e}")
            plt.show()

    # 2. 8-CATEGORY Spectrum ablation
    if ctx.run_spectrum_ablation and not _is_within:
        display(
            HTML(
                '<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #0097A7;'
                "background:linear-gradient(135deg,#E0F7FA,#F5F5F5);padding:14px 20px;"
                'margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
                '<span style="font-size:15px;font-weight:700;color:#0097A7;">8-Category Spectrum Ablation</span><br>'
                '<span style="font-size:12px;color:#546E7A;">Each domain independently -- Standalone predictive performance</span>'
                "</div>"
            )
        )

        for _dom in _DOMAIN_REGISTRY:
            _dom_feats = [f for f, d in _f2d.items() if d == _dom]
            if len(_dom_feats) < 2:
                logger.info(f"  {_dom:>22s}: Warning: only {len(_dom_feats)} features -- skipping")
                _spectrum_results[_dom] = {"score": None, "n_features": len(_dom_feats), "skipped": True}
                continue

            _X_tr_d = ctx.X_train[[f for f in _dom_feats if f in ctx.X_train.columns]]
            _X_te_d = ctx.X_test[[f for f in _dom_feats if f in ctx.X_test.columns]]

            _md = create_lightweight_model(is_classification=_is_clf, n_features=len(_dom_feats))
            _md.fit(_X_tr_d, _y_tr)
            if _is_clf:
                _pred_d = _md.predict_proba(_X_te_d)[:, 1]
            else:
                _pred_d = _md.predict(_X_te_d)

            _score_d = _metric_fn(_y_te, _pred_d)
            _pt, _lo, _hi = bootstrap_ci(_y_te, _pred_d, _metric_fn, n_boot=500, seed=42)

            _spectrum_results[_dom] = {"score": _score_d, "ci_lo": _lo, "ci_hi": _hi, "n_features": _X_tr_d.shape[1]}
            logger.info(f"  {_dom:>22s}: {_metric} = {_score_d:.4f} [{_lo:.4f}, {_hi:.4f}]  ({_X_tr_d.shape[1]} feat.)")

        # Figure 2: Spectrum lollipop (sorted by score)
        _valid_spectrum = {
            k: v for k, v in _spectrum_results.items() if v.get("score") is not None and not v.get("skipped")
        }
        if len(_valid_spectrum) >= 2:
            _sorted_doms = sorted(_valid_spectrum.keys(), key=lambda d: _valid_spectrum[d]["score"], reverse=True)

            fig, ax = plt.subplots(figsize=(10, max(4, len(_sorted_doms) * 0.55 + 1)), dpi=300)
            _y_pos = list(range(len(_sorted_doms)))

            for _i, _dom in enumerate(_sorted_doms):
                _r = _valid_spectrum[_dom]
                _c = _DOMAIN_COLORS.get(_dom, "#78909C")
                ax.barh(_i, _r["score"], height=0.5, color=_c, alpha=0.6, zorder=2, edgecolor="white", linewidth=0.5)
                ax.errorbar(
                    _r["score"],
                    _i,
                    xerr=[[_r["score"] - _r["ci_lo"]], [_r["ci_hi"] - _r["score"]]],
                    fmt="o",
                    color=_c,
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    capsize=3,
                    capthick=0.8,
                    ecolor="#444444",
                    zorder=3,
                )
                ax.text(
                    _r["ci_hi"] + 0.008,
                    _i,
                    f"{_r['score']:.4f}  ({_r['n_features']})",
                    va="center",
                    fontsize=8,
                    color=_c,
                    fontweight="bold",
                )

            # Full model reference line
            if "Combined (Full)" in _binary_results:
                _full_s = _binary_results["Combined (Full)"]["score"]
                ax.axvline(x=_full_s, color="#37474F", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)
                ax.text(
                    _full_s + 0.005,
                    len(_sorted_doms) - 0.3,
                    f"Full: {_full_s:.4f}",
                    fontsize=7.5,
                    color="#37474F",
                    fontstyle="italic",
                )

            ax.set_yticks(_y_pos)
            ax.set_yticklabels(_sorted_doms, fontsize=9, fontweight="bold")
            ax.set_xlabel(f"Standalone {_metric}", fontsize=9)
            ax.set_title(f"8-Category Spectrum Ablation -- {_target} (T{_tp})", fontsize=10, fontweight="bold")
            _min_score = min(r["ci_lo"] for r in _valid_spectrum.values())
            ax.set_xlim(left=max(0, _min_score - 0.05))
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.15, linewidth=0.5)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.tight_layout()

            try:
                _fig_dir = os.path.join(os.getcwd(), "domain_ablation_figures")
                os.makedirs(_fig_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(_fig_dir, f"spectrum_ablation_{_target}_T{_tp}.png"), dpi=300, bbox_inches="tight"
                )
                fig.savefig(
                    os.path.join(_fig_dir, f"spectrum_ablation_{_target}_T{_tp}.pdf"), dpi=300, bbox_inches="tight"
                )
            except Exception as _e:
                logger.info(f"Warning: {_e}")
            plt.show()

    # 3. Domain permutation importance
    if ctx.run_domain_permutation and not _is_within:
        display(
            HTML(
                '<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #E65100;'
                "background:linear-gradient(135deg,#FFF3E0,#F5F5F5);padding:14px 20px;"
                'margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
                '<span style="font-size:15px;font-weight:700;color:#E65100;">Domain Permutation Importance</span><br>'
                '<span style="font-size:12px;color:#546E7A;">Shuffle each domain within the full model -- marginal contribution in context</span>'
                "</div>"
            )
        )

        # Train full model
        _full_model = create_lightweight_model(is_classification=_is_clf, n_features=len(_all_features))
        _full_model.fit(ctx.X_train, _y_tr)

        _base_score, _perm_raw = _domain_permutation_importance(
            _full_model, ctx.X_test, _y_te, _f2d, _metric_fn, is_clf=_is_clf, n_repeats=10, seed=42
        )

        logger.info(f"\n  Full model {_metric} (baseline): {_base_score:.4f}")
        logger.info(f"  Domain permutation importance (Δ{_metric} on shuffle):\n")

        _sorted_perm = sorted(_perm_raw.items(), key=lambda x: x[1]["mean_drop"], reverse=True)
        for _dom, _pr in _sorted_perm:
            _perm_results[_dom] = _pr
            _sig = (
                "***" if _pr["mean_drop"] > 2 * _pr["std_drop"] else ("*" if _pr["mean_drop"] > _pr["std_drop"] else "")
            )
            logger.info(
                f"    {_dom:>22s}: Δ{_metric} = {_pr['mean_drop']:+.4f} ± {_pr['std_drop']:.4f}  "
                f"({_pr['n_features']} feat.) {_sig}"
            )

        # Figure 3: Permutation importance lollipop
        if len(_perm_results) >= 2:
            _sorted_doms_p = [d for d, _ in _sorted_perm if d != "Unclassified"]

            fig, ax = plt.subplots(figsize=(10, max(4, len(_sorted_doms_p) * 0.55 + 1)), dpi=300)
            _y_pos = list(range(len(_sorted_doms_p)))

            for _i, _dom in enumerate(_sorted_doms_p):
                _pr = _perm_results[_dom]
                _c = _DOMAIN_COLORS.get(_dom, "#78909C")
                _drop = _pr["mean_drop"]
                ax.barh(_i, _drop, height=0.5, color=_c, alpha=0.6, zorder=2, edgecolor="white", linewidth=0.5)
                ax.errorbar(
                    _drop,
                    _i,
                    xerr=_pr["std_drop"],
                    fmt="o",
                    color=_c,
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    capsize=3,
                    capthick=0.8,
                    ecolor="#444444",
                    zorder=3,
                )
                _x_ann = max(_drop + _pr["std_drop"], 0) + 0.003
                ax.text(
                    _x_ann,
                    _i,
                    f"Δ = {_drop:+.4f}  ({_pr['n_features']})",
                    va="center",
                    fontsize=8,
                    color=_c,
                    fontweight="bold",
                )

            ax.axvline(x=0, color="#999999", linewidth=0.8, zorder=1)
            _max_drop = max(r["mean_drop"] + r["std_drop"] for r in _perm_results.values())
            ax.set_xlim(left=min(-0.005, -_max_drop * 0.05))
            ax.set_yticks(_y_pos)
            ax.set_yticklabels(_sorted_doms_p, fontsize=9, fontweight="bold")
            ax.set_xlabel(f"Δ{_metric} (drop on permutation)", fontsize=9)
            ax.set_title(f"Domain Permutation Importance -- {_target} (T{_tp})", fontsize=10, fontweight="bold")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.15, linewidth=0.5)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.tight_layout()

            try:
                _fig_dir = os.path.join(os.getcwd(), "domain_ablation_figures")
                os.makedirs(_fig_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(_fig_dir, f"domain_permutation_{_target}_T{_tp}.png"), dpi=300, bbox_inches="tight"
                )
                fig.savefig(
                    os.path.join(_fig_dir, f"domain_permutation_{_target}_T{_tp}.pdf"), dpi=300, bbox_inches="tight"
                )
            except Exception as _e:
                logger.info(f"Warning: {_e}")
            plt.show()

    # 4. Top-k feature ablation curve
    if ctx.run_topk_ablation:
        display(
            HTML(
                '<div style="font-family:sans-serif;max-width:750px;border-left:4px solid #2E7D32;'
                "background:linear-gradient(135deg,#E8F5E9,#F5F5F5);padding:14px 20px;"
                'margin:16px 0 12px 0;border-radius:0 6px 6px 0;">'
                '<span style="font-size:15px;font-weight:700;color:#2E7D32;">Top-K Feature Ablation Curve</span><br>'
                '<span style="font-size:12px;color:#546E7A;">Progressive removal of the top-ranked features</span>'
                "</div>"
            )
        )

        # Train full model and get feature importances
        _full_model_tk = create_lightweight_model(is_classification=_is_clf, n_features=len(_all_features))
        _full_model_tk.fit(ctx.X_train, _y_tr)

        _feat_imp = None
        if hasattr(_full_model_tk, "feature_importances_"):
            _feat_imp = pd.Series(_full_model_tk.feature_importances_, index=ctx.X_train.columns)
        elif hasattr(_full_model_tk, "get_feature_importance"):
            _feat_imp = pd.Series(_full_model_tk.get_feature_importance(), index=ctx.X_train.columns)

        if _feat_imp is not None:
            _feat_imp_sorted = _feat_imp.sort_values(ascending=False)

            # Full model baseline
            if _is_clf:
                _full_pred_tk = _full_model_tk.predict_proba(ctx.X_test)[:, 1]
            else:
                _full_pred_tk = _full_model_tk.predict(ctx.X_test)
            _full_score_tk = _metric_fn(_y_te, _full_pred_tk)

            _ks = [0, 1, 5, 10, 20, 50, 100, min(200, len(_all_features) // 2)]
            _ks = sorted(set(k for k in _ks if k < len(_all_features)))

            logger.info(f"\n  Full model {_metric}: {_full_score_tk:.4f} ({len(_all_features)} features)")
            logger.info("  Removing top-K features by importance:\n")

            _topk_curve = []
            for _k in _ks:
                if _k == 0:
                    _topk_curve.append({"k": 0, "score": _full_score_tk, "n_remaining": len(_all_features)})
                    continue

                _drop_feats = _feat_imp_sorted.index[:_k].tolist()
                _keep_feats = [f for f in _all_features if f not in _drop_feats]

                if len(_keep_feats) < 2:
                    break

                _X_tr_k = ctx.X_train[_keep_feats]
                _X_te_k = ctx.X_test[_keep_feats]
                _mk = create_lightweight_model(is_classification=_is_clf, n_features=len(_keep_feats))
                _mk.fit(_X_tr_k, _y_tr)
                if _is_clf:
                    _pred_k = _mk.predict_proba(_X_te_k)[:, 1]
                else:
                    _pred_k = _mk.predict(_X_te_k)
                _score_k = _metric_fn(_y_te, _pred_k)

                _topk_curve.append({"k": _k, "score": _score_k, "n_remaining": len(_keep_feats)})
                _delta_k = _score_k - _full_score_tk
                logger.info(
                    f"    Remove top-{_k:>3d}: {_metric} = {_score_k:.4f}  "
                    f"(Δ = {_delta_k:+.4f}, {len(_keep_feats)} remaining)"
                )

                # Domain composition of removed features
                _removed_domains = {}
                for _rf in _drop_feats:
                    _rd = _f2d.get(_rf, "Unclassified")
                    _removed_domains[_rd] = _removed_domains.get(_rd, 0) + 1

            _topk_results = _topk_curve

            # Figure 4: Ablation curve
            if len(_topk_curve) >= 3:
                _df_tk = pd.DataFrame(_topk_curve)
                fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

                ax.plot(
                    _df_tk["k"],
                    _df_tk["score"],
                    "o-",
                    color="#2E7D32",
                    markersize=6,
                    linewidth=1.8,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=3,
                )
                ax.fill_between(_df_tk["k"], _df_tk["score"], _full_score_tk, alpha=0.08, color="#C62828", zorder=1)
                ax.axhline(
                    y=_full_score_tk,
                    color="#37474F",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.5,
                    zorder=2,
                    label=f"Full model ({_full_score_tk:.4f})",
                )

                for _, _row in _df_tk.iterrows():
                    if _row["k"] > 0:
                        _delta_ann = _row["score"] - _full_score_tk
                        ax.annotate(
                            f"{_row['score']:.3f}\n(Δ{_delta_ann:+.3f})",
                            xy=(_row["k"], _row["score"]),
                            textcoords="offset points",
                            xytext=(8, -12),
                            fontsize=7.5,
                            color="#2E7D32",
                            fontweight="bold",
                        )

                # Top-5 feature names as annotation
                _top5 = _feat_imp_sorted.index[:5].tolist()
                _top5_str = "\n".join([f"{i + 1}. {f}" for i, f in enumerate(_top5)])
                ax.text(
                    0.98,
                    0.98,
                    f"Top 5 features:\n{_top5_str}",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                    ha="right",
                    color="#546E7A",
                    fontstyle="italic",
                    bbox=dict(
                        boxstyle="round,pad=0.4", facecolor="#f8f8f8", alpha=0.9, edgecolor="#cccccc", linewidth=0.5
                    ),
                )

                ax.set_xlabel("Number of top features removed", fontsize=9)
                ax.set_ylabel(_metric, fontsize=9)
                ax.set_title(f"Top-K Feature Ablation -- {_target} (T{_tp})", fontsize=10, fontweight="bold")
                ax.legend(fontsize=8, loc="lower left", framealpha=0.9, edgecolor="#cccccc")
                ax.grid(axis="y", alpha=0.15, linewidth=0.5)
                for sp in ["top", "right"]:
                    ax.spines[sp].set_visible(False)
                plt.tight_layout()

                try:
                    _fig_dir = os.path.join(os.getcwd(), "domain_ablation_figures")
                    os.makedirs(_fig_dir, exist_ok=True)
                    fig.savefig(
                        os.path.join(_fig_dir, f"topk_ablation_{_target}_T{_tp}.png"), dpi=300, bbox_inches="tight"
                    )
                    fig.savefig(
                        os.path.join(_fig_dir, f"topk_ablation_{_target}_T{_tp}.pdf"), dpi=300, bbox_inches="tight"
                    )
                except Exception as _e:
                    logger.info(f"Warning: {_e}")
                plt.show()
        else:
            logger.info("  Warning: Could not extract feature importances from lightweight model.")

    # Consolidated summary dashboard
    _summary_rows = []
    if _binary_results:
        for _lab, _r in _binary_results.items():
            _summary_rows.append(
                {
                    "Analysis": "Binary",
                    "Domain": _lab,
                    _metric: f"{_r['score']:.4f}",
                    "CI": f"[{_r['ci_lo']:.3f}, {_r['ci_hi']:.3f}]",
                    "N Feat": _r["n_features"],
                }
            )
    if _spectrum_results:
        for _dom, _r in sorted(_spectrum_results.items(), key=lambda x: x[1].get("score", 0) or 0, reverse=True):
            if _r.get("score") is not None:
                _summary_rows.append(
                    {
                        "Analysis": "Spectrum",
                        "Domain": _dom,
                        _metric: f"{_r['score']:.4f}",
                        "CI": f"[{_r['ci_lo']:.3f}, {_r['ci_hi']:.3f}]",
                        "N Feat": _r["n_features"],
                    }
                )
    if _perm_results:
        for _dom, _pr in sorted(_perm_results.items(), key=lambda x: x[1]["mean_drop"], reverse=True):
            _summary_rows.append(
                {
                    "Analysis": "Permutation",
                    "Domain": _dom,
                    _metric: f"Δ = {_pr['mean_drop']:+.4f}",
                    "CI": f"± {_pr['std_drop']:.4f}",
                    "N Feat": _pr["n_features"],
                }
            )
    if _delta_cis:
        for _cmp, _dc in _delta_cis.items():
            _sig = "●" if _dc["ci_lo"] > 0 or _dc["ci_hi"] < 0 else "○"
            _summary_rows.append(
                {
                    "Analysis": "Δ Bootstrap",
                    "Domain": _cmp.replace("_", " "),
                    _metric: f"Δ = {_dc['median']:+.4f}",
                    "CI": f"[{_dc['ci_lo']:+.3f}, {_dc['ci_hi']:+.3f}] {_sig}",
                    "N Feat": "--",
                }
            )

    if _summary_rows:
        _sdf = pd.DataFrame(_summary_rows)
        display(
            HTML(
                '<div style="font-family:sans-serif;max-width:800px;margin:20px 0;">'
                '<div style="background:#37474F;color:white;padding:10px 16px;'
                'font-size:13px;font-weight:700;border-radius:6px 6px 0 0;">'
                f"Domain Ablation Summary -- {_target} (T{_tp})</div>"
                "</div>"
            )
        )
        display(
            _sdf.style.set_properties(
                **{
                    "font-size": "12px",
                    "text-align": "center",
                }
            ).set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("font-size", "11px"),
                            ("background", "#ECEFF1"),
                            ("text-align", "center"),
                            ("padding", "6px 10px"),
                        ],
                    },
                    {"selector": "td", "props": [("padding", "5px 10px")]},
                ]
            )
        )

    # Store results in the report state
    _RS["domain_ablation"] = {
        "results": {k: v for k, v in _binary_results.items()},
        "target": _target,
        "metric": _metric,
        "spectrum_results": {k: v for k, v in _spectrum_results.items()},
        "permutation_results": {k: v for k, v in _perm_results.items()},
        "topk_results": _topk_results,
        "delta_cis": _delta_cis,
        "feature_domain_map_counts": _domain_counts,
        "split": _split,
    }

    logger.info("\n  Domain ablation complete. Results stored in _RS['domain_ablation'].")


def run() -> None:
    """Run this stage (notebook cell 18)."""
    global traceback
    warnings.filterwarnings("ignore")

    # Batch dispatch
    _is_bv_da = (
        "_batch_archive" in ctx
        and isinstance(ctx._batch_archive, dict)
        and len(ctx._batch_archive) > 1
        and any(v.get("X_train") is not None for v in ctx._batch_archive.values())
    )
    if not _is_bv_da:
        try:
            _cbv_da = cache_load("main_runner")
            if (
                _cbv_da
                and isinstance(_cbv_da, dict)
                and _cbv_da.get("_batch_archive")
                and len(_cbv_da["_batch_archive"]) > 1
                and any(v.get("X_train") is not None for v in _cbv_da["_batch_archive"].values())
            ):
                ctx._batch_archive = _cbv_da["_batch_archive"]
                _is_bv_da = True
        except Exception:
            pass  # non-critical; logged upstream or optional visualization

    if _is_bv_da:
        _vgs_da = {
            k: ctx.get(k)
            for k in [
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "target_options",
                "tp_option",
                "metrics_dict_full",
                "exported_within_dict",
                "consensus_df",
                "model_weights",
                "split",
                "last_fitted_model",
                "sample_valid",
                "sample_valid_test",
                "sample_valid_train",
                "results_summary",
            ]
        }
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH -- Domain Ablation: processing {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)

        for _vi_da, (_vk_da, ctx._vd_da) in enumerate(ctx._batch_archive.items()):
            _vl_da = ctx._vd_da.get("target", _vk_da)
            logger.info(f"\n{'-' * 68}")
            logger.info(f"  [{_vi_da + 1}/{len(ctx._batch_archive)}] Target: {_vl_da}")
            logger.info(f"{'-' * 68}")
            for _gk in [
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "metrics_dict_full",
                "exported_within_dict",
                "consensus_df",
                "model_weights",
                "split",
                "last_fitted_model",
                "results_summary",
                "sample_valid",
                "sample_valid_test",
                "sample_valid_train",
            ]:
                if ctx._vd_da.get(_gk) is not None:
                    ctx[_gk] = ctx._vd_da[_gk]
            ctx["target_options"] = _vl_da
            if "timepoint" in ctx._vd_da:
                ctx["tp_option"] = ctx._vd_da["timepoint"]
            if "split" in ctx._vd_da:
                ctx["split"] = ctx._vd_da["split"]
            try:
                _run_domain_ablation()
            except Exception as _e_da:
                logger.info(f"  Warning: Error for {_vl_da}: {_e_da}")
                import traceback

                traceback.print_exc()
            # Archive results back into _batch_archive
            _rs_ref = ctx.get("_RS")
            if _vk_da in ctx._batch_archive and _rs_ref is not None:
                if "domain_ablation" in _rs_ref:
                    ctx._batch_archive[_vk_da]["_RS_domain_ablation"] = copy.deepcopy(_rs_ref["domain_ablation"])

        for _k, _v in _vgs_da.items():
            ctx[_k] = _v
        logger.info("\n" + "=" * 72)
        logger.info(f"  BATCH COMPLETE -- Domain Ablation: {len(ctx._batch_archive)} targets")
        logger.info("=" * 72)
    else:
        _run_domain_ablation()
