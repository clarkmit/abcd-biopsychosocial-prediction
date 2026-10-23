"""Preprocessing, natural language variable renaming and train-test partitioning (setup).

Data loading, imputation, scaling, train-test partitioning, and natural
language variable renaming. All preprocessing is fit on training data only
and applied after the split. Provides within_categories_data() and
across_categories_data() slicing functions that apply circular exclusions
at runtime based on the active target. Delta variables enter as
precomputed columns from the source data.

Functions are grouped as follows:

Task-type detection  is_regression(), is_classification()
Class encoding       map_classes(), inverse_map_classes()
Class rebalancing    simple_balance_classes()
Imputation factory   get_imputer()
Missing-data QC      drop_high_missing_columns()
Data splitting       within_categories_data(), across_categories_data(),
                     within_categories_data_combinations()
K-fold variants      within_categories_data_kfold(),
                     across_categories_data_kfold(),
                     train_and_evaluate_model_kfold()
Quantile utilities   segment_data_by_quantiles(),
                     calculate_quantile_weights()
"""

import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from bpsp.context import ctx
from bpsp.data.circularity import get_circular_exclusions, is_parent_target

logger = logging.getLogger(__name__)


# Registry of continuous-outcome targets. Dimensional targets (CBCL subscales,
# SBT depression composites, cognitive scores) are modeled as regressions;
# binary/categorical targets default to classification with stratified splits.
def is_regression(target_options):
    regression_targets = [
        "depress_D_p",
        "delta_depress_D_p",
        "internal_D_p",
        "external_D_p",
        "anxdisord_D_p",
        "adhd_D_p",
        "social_problems_D_p",
        "thought_disorder_D_p",
        "sbt_core_depression",
        "delta_sbt_core_depression",
        "sbt_core_depression_delta_t0_t3",
        "sbt_anxiety_depression",
        "sbt_guilt_hopelessness",
        "sbt_emotional_dysregulation",
        "sbt_well_being",
        "sbt_fatigue_sleep",
        "sbt_somatic_depression",
        "sbt_academic_cognitive",
        "sbt_perfectionism_achievement",
        "sbt_social_withdrawal",
        "sbt_avoidance_fear",
        "sbt_aggression_irritability",
        "suicidal_p",
        "wish_dead_present_B_k",
        "asd_ssrs_sum",
        "nb_correct_nt_2back",
        "bdefs_distract_upset_p",
        "gd_riskybets",
        "pa_sum_k",
        "tb_fluid",
        "tb_cryst",
        "tb_reading",
        "tb_flanker",
        # T0 Cognitive Task Battery additions:
        "ravlt_l_total",
        "lmt_efficiency",
        "sst_mean_ssrt",
        "bad_grades",
        "area_deprivation_idx",
        "weight",
        "parent_income",
        "compulsions_p",
        "obsessions_p",
        "impulsive_p",
        "perfectionist_p",
        # Child CBCL Social Problems items
        "not_liked_p",
        "doesnt_get_along_p",
        "enjoys_little_p",
        "parent_depressed_p",
        "parent_suicidal_thoughts_p",
        "parent_depress_D_p",
        "parent_adhd_D_p",
        "parent_anxdisord_D_p",
        "delta_anxdisord_D_p",
        "parent_worries_a_lot_p",
        "parent_concentration_trouble_p",
        "parent_happy_person_p",
        "parent_bad_relationships_p",
        "parent_not_liked_by_others_p",
        "delta_parent_depressed_p",
        "delta_parent_worries_a_lot_p",
        "parent_impulsive_p",
        "parent_tobacco_use_frequency_p",
        "parent_drinks_frequency_p",
        "parent_drunk_days_p",
        # Parent Social/Impulsivity/Compulsivity batch
        "parent_avoidant_person_D_p",
        "parent_repeats_acts_p",
        "parent_obsessive_thoughts_p",
        # Parent broadband (ASR Internalizing / Externalizing)
        "parent_internal_D_p",
        "parent_external_D_p",
        "delta_parent_depress_D_p",  # parent broadband Depression Syndrome change score
        # Child Only Deltas batch
        "delta_adhd_D_p",
        "delta_family_conflict_ss_k",
        "delta_bad_grades",
        "delta_social_problems_D_p",
        "delta_somatic_problems_D_p",
        # Parent Only Deltas batch
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
    ]
    return target_options in regression_targets


def is_classification(y):
    """Heuristic check: targets with ≤10 unique observed values are treated
    as classification tasks.  Used for stratified splitting and metric selection."""
    try:
        unique_values = y.nunique() if hasattr(y, "nunique") else len(set(y.values.flatten()))
        if isinstance(unique_values, pd.Series):
            unique_values = unique_values.iloc[0]
        return bool(unique_values <= 10)
    except Exception as e:
        logger.info(f"[WARNING] {e}")
        return False


# Canonical task-type detector used by the validation and reporting stages.
# Defined here so there is a single authoritative copy; later stages
# rely on the registry having been loaded first.
# Canonical task-type detector
def detect_task_type(y, threshold=10):
    """Determine whether a target is regression or classification.

    Uses the same logic as is_regression / is_classification defined below in this module,
    but returns a string for downstream branching.

    Parameters
    y : array-like
        Target variable (Series, ndarray, or DataFrame column).
    threshold : int, default 10
        If the number of unique non-NaN values ≤ threshold, treat as
        classification.  Otherwise regression.

    Returns
    str : 'classification' or 'regression'
    """
    # Honour whitelist for targets with known types
    try:
        if "is_regression" in ctx and "target_options" in ctx:
            if is_regression(ctx.target_options):
                return "regression"
    except Exception:
        pass
    _arr = np.asarray(y).ravel()
    _arr = _arr[~np.isnan(_arr)] if np.issubdtype(_arr.dtype, np.floating) else _arr
    n_unique = len(np.unique(_arr))
    return "classification" if n_unique <= threshold else "regression"


def map_classes(y):
    if ctx.target_options == "depadhd_c":
        class_mapping = {
            "other": 0,
            "lowdep_lowadhd": 1,
            "highdep_highadhd": 2,
            "highdep_lowadhd": 3,
            "lowdep_highadhd": 4,
        }
    elif ctx.target_options == "asdadhd_c":
        class_mapping = {
            "other": 0,
            "lowasd_lowadhd": 1,
            "highasd_highadhd": 2,
            "highasd_lowadhd": 3,
            "lowasd_highadhd": 4,
        }
    else:
        class_mapping = {"falling": 0, "low": 1, "rising": 2, "high": 3}

    logger.info(f"For {ctx.target_options}, mapping of classes are as follows: {class_mapping}")

    if isinstance(y, pd.DataFrame):
        return y.map(lambda x: class_mapping.get(x, x))
    elif isinstance(y, pd.Series):
        return y.map(class_mapping)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame")


# Reverse the integer encoding produced by map_classes() for display and
# downstream evaluation against the original string labels.
def inverse_map_classes(y):
    if ctx.target_options == "depadhd_c":
        inverse_class_mapping = {
            0: "other",
            1: "lowdep_lowadhd",
            2: "highdep_highadhd",
            3: "highdep_lowadhd",
            4: "lowdep_highadhd",
        }
    elif ctx.target_options == "asdadhd_c":
        inverse_class_mapping = {
            0: "other",
            1: "lowasd_lowadhd",
            2: "highasd_highadhd",
            3: "highasd_lowadhd",
            4: "lowasd_highadhd",
        }
    else:
        inverse_class_mapping = {0: "falling", 1: "low", 2: "rising", 3: "high"}

    if isinstance(y, pd.DataFrame):
        return y.map(lambda x: inverse_class_mapping.get(x, x))
    elif isinstance(y, pd.Series):
        return y.map(inverse_class_mapping)
    elif isinstance(y, (int, np.integer)):  # Handle integer input
        return inverse_class_mapping.get(y, y)
    else:
        raise TypeError("Input must be a pandas Series, DataFrame, or integer")


# Rebalance class frequencies for multi-class trajectory targets (latent_class_
# depression, depadhd_c, asdadhd_c). Majority classes are downsampled and
# minority classes upsampled with replacement to equalize representation.
# Applied post-split to training data only in each data-preparation function.
def simple_balance_classes(df, target_column="latent_class_depression"):
    # Note: In the main pipeline (within/across_categories_data), this function
    # is called AFTER the train/test split on training data only.
    # The kfold variant still applies rebalancing pre-split.

    # Get the current class distribution
    class_counts = df[target_column].value_counts()
    logger.info("Original class distribution:")
    logger.info(class_counts)

    # Determine the desired number of samples per class
    n_samples = len(df)
    n_classes = len(class_counts)
    samples_per_class = n_samples // n_classes

    # Collect resampled class chunks; build DataFrame once to avoid pandas 2.x FutureWarning
    chunks = []

    for class_label in class_counts.index:
        class_df = df[df[target_column] == class_label]

        if len(class_df) > samples_per_class:
            # Undersample majority class
            resampled = resample(class_df, replace=False, n_samples=samples_per_class, random_state=42)
        else:
            # Oversample minority class
            resampled = resample(class_df, replace=True, n_samples=samples_per_class, random_state=42)

        chunks.append(resampled)

    balanced_df = pd.concat(chunks, ignore_index=True)

    # Shuffle the balanced dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("\nBalanced class distribution:")
    logger.info(balanced_df[target_column].value_counts())

    return balanced_df


# Factory function that instantiates the selected missing-data imputer.
# All options are fit exclusively on training data and applied to both
# train and test splits to prevent information leakage.
# "mean"   -- unconditional mean imputation
# "median": unconditional median imputation
# "knn"    -- k-nearest-neighbor imputation (k = 5)
# "mice"   -- multiple imputation by chained equations
def get_imputer(imputer_type):
    if imputer_type == "mean":
        return SimpleImputer(strategy="mean")
    elif imputer_type == "median":
        return SimpleImputer(strategy="median")
    elif imputer_type == "knn":
        return KNNImputer(n_neighbors=5)
    elif imputer_type == "mice":
        return IterativeImputer(random_state=0)
    else:
        raise ValueError(f"Unknown imputer type: {imputer_type}")


def drop_high_missing_columns(df, threshold=0.7, verbose=False):
    """
    Drop columns with more than threshold% missing values

    Args:
        df: pandas DataFrame
        threshold: float between 0 and 1, representing maximum allowed missing percentage
        verbose: bool, whether to print dropped columns

    Returns:
        DataFrame with high-missing columns dropped
    """
    # Calculate percentage of missing values
    missing_percentages = df.isna().sum() / len(df)

    # Get columns to drop
    cols_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()

    if verbose and cols_to_drop:
        logger.info(f"\nDropping {len(cols_to_drop)} columns with >{threshold * 100}% missing values:")
        for col in cols_to_drop:
            logger.info(f"{col}: {missing_percentages[col] * 100:.1f}%")

    return df.drop(columns=cols_to_drop)


# within_categories_data()
# Prepares train/test splits for within-category analyses. Each key in the
# returned dictionary corresponds to one predictor category (e.g., "Anxiety",
# "Task fMRI"); the value is the (X_train, X_test, y_train, y_test) tuple for
# that category's variable subset.
# Processing order (leakage-free):
# 1. Select child or parent variable structure based on target.
# 2. Apply category-level circular exclusions from CIRCULAR_EXCLUSIONS.
# 3. Drop columns with >70 % missing values.
# 4. One-hot-encode nominal covariates.
# 5. Split raw data 70/30 (stratified for classification).
# 6. Fit imputer and optional scaler on training fold only; transform both.
# 7. Return per-category slices of the imputed/scaled arrays.
def within_categories_data(
    t, targets, group=None, scale_x=True, scale_y=True, verbose=False, class_rebalancing=False, feature_tp=None
):
    from bpsp.analysis.runner import imputer

    # Resolve feature timepoint
    feat_t = feature_tp if feature_tp is not None else t
    _cross_tp = feature_tp is not None and feature_tp != t
    target_name = targets[0] if targets else ""
    if is_parent_target(target_name):
        ctx.variables_for_each_time_point = ctx.parent_variables_for_each_time_point
        logger.info(f"[VARS] Using PARENT within-category structure for target: {target_name}")
    else:
        ctx.variables_for_each_time_point = ctx.child_variables_for_each_time_point
        logger.info(f"[VARS] Using CHILD within-category structure for target: {target_name}")

    # Build active categories from the FEATURE timepoint
    # Circular exclusions are always keyed on the TARGET timepoint (t).
    across_excl, within_excl = get_circular_exclusions(target_name, timepoint=t)
    active_vfet_tp = {
        cat: vars_list for cat, vars_list in ctx.variables_for_each_time_point[feat_t].items() if cat not in within_excl
    }
    if within_excl:
        skipped = within_excl & set(ctx.variables_for_each_time_point[feat_t].keys())
        if skipped:
            logger.info(f"[CIRCULAR] Excluded within-categories for {target_name}: {skipped}")

    # Flatten retained categories to a single variable list, removing any
    # individually excluded circular variables within those categories.
    within_categories = [x for xs in active_vfet_tp.values() for x in xs if x not in across_excl] + targets
    if across_excl:
        individual_removed = len([x for xs in active_vfet_tp.values() for x in xs if x in across_excl])
        if individual_removed > 0:
            logger.info(
                f"[CIRCULAR] Also removed {individual_removed} individual circular variables from within-category features for {target_name}"
            )

    within_categories = list(set(within_categories))

    # Get available columns in the DataFrame
    available_columns = set(ctx.sample.columns)

    # Remove columns from within_categories that are not in available_columns
    within_categories = [var for var in within_categories if var in available_columns]

    # If any columns were removed, notify the user
    dropped_columns = set(
        [x for xs in ctx.variables_for_each_time_point[feat_t].values() for x in xs] + targets
    ).difference(available_columns)

    if dropped_columns:
        logger.info(
            f"WARNING: The following columns were not found in the dataset and have been dropped: {dropped_columns}"
        )

    # Remove peer-victimisation variables absent/unreliable at T1 and T3 (feature side)
    if int(feat_t) == 3 or int(feat_t) == 1:
        variables_to_remove = [
            "peq_ss_relational_victim",
            "peq_ss_reputation_aggs",
            "peq_ss_reputation_victim",
            "peq_ss_overt_aggression",
            "peq_ss_overt_victim",
            "peq_ss_relational_aggs",
        ]
        within_categories = [var for var in within_categories if var not in variables_to_remove]

    # Apply circular exclusions so kfold path is same as standard path.
    _across_excl_kw, _within_excl_kw = get_circular_exclusions(targets[0] if targets else "", timepoint=t)
    if _within_excl_kw:
        within_categories = [
            x
            for x in within_categories
            if x in targets
            or not any(x in ctx.variables_for_each_time_point[feat_t].get(cat, []) for cat in _within_excl_kw)
        ]
    if _across_excl_kw:
        within_categories = [x for x in within_categories if x not in _across_excl_kw or x in targets]
        if verbose:
            logger.info(
                f"[CIRCULAR/kfold] Excluded {len(_across_excl_kw)} circular variables for {targets[0] if targets else '?'}"
            )

    # Build all_data
    _feat_only_vars = [v for v in within_categories if v not in targets]

    if _cross_tp:
        # Cross-timepoint path: features from feat_t, targets from t
        # Group filter applied to TARGET wave so subgroup membership is
        # defined at the outcome timepoint.
        tgt_rows = ctx.sample[ctx.sample["time"] == t].copy()
        if group:
            if group == "high_ale":
                tgt_rows = tgt_rows[tgt_rows["high_ale"] == True]
            elif group == "high_ale_severe_p_mh":
                tgt_rows = tgt_rows[tgt_rows["high_ale_severe_p_mh"] == True]
            elif group == "low_ale":
                tgt_rows = tgt_rows[tgt_rows["low_ale_children_p"] == True]
            elif group == "Female":
                tgt_rows = tgt_rows[tgt_rows["sex"] == 2]
            elif group == "Male":
                tgt_rows = tgt_rows[tgt_rows["sex"] == 1]

        valid_subjects = tgt_rows["subject"].values

        # Feature-wave rows, restricted to group-filtered subjects
        feat_rows = ctx.sample[(ctx.sample["time"] == feat_t) & (ctx.sample["subject"].isin(valid_subjects))].copy()

        # Merge: features from feat_t, targets from t, on subject
        feat_cols_avail = [c for c in _feat_only_vars if c in feat_rows.columns]
        tgt_cols_avail = [c for c in targets if c in tgt_rows.columns]

        _feat_df = feat_rows[["subject"] + feat_cols_avail].drop_duplicates("subject")
        _tgt_df = tgt_rows[["subject"] + tgt_cols_avail].drop_duplicates("subject")

        all_data = _feat_df.merge(_tgt_df, on="subject", how="inner").drop(columns="subject")

        if verbose:
            logger.info(
                f"[Cross-TP within] feat_tp={feat_t}, target_tp={t}, "
                f"group={group if group else 'full'}, merged N={len(all_data)}"
            )

    else:
        # Same-timepoint path: original logic preserved exactly
        if group:
            if group == "high_ale":
                all_vars = within_categories + [f"{group}"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data[all_data["high_ale"] == True]  # high_ale (old)

            elif group == "high_ale_severe_p_mh":
                all_vars = within_categories + [f"{group}"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data[all_data["high_ale_severe_p_mh"] == True]  # high_ale (old)

            elif group == "low_ale":
                group = "low_ale_children_p"
                all_vars = within_categories + [f"{group}"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data[all_data["low_ale_children_p"] == True]

            elif group == "Female":
                all_vars = within_categories + ["sex"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data.loc[:, ~all_data.columns.duplicated()]
                all_data = all_data[all_data["sex"] == 2]

            elif group == "Male":
                all_vars = within_categories + ["sex"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data.loc[:, ~all_data.columns.duplicated()]
                all_data = all_data[all_data["sex"] == 1]

            else:
                all_vars = within_categories
                all_data = ctx.sample.query(f"time == {t}")[all_vars]

        else:
            all_vars = within_categories
            all_data = ctx.sample.query(f"time == {t}")[all_vars]

    if verbose:
        logger.info(
            f"Gathered within category data from {all_data.shape[0]} subjects for {group if group else 'full'} group at t = {t}"
        )

    if (
        "depress_D_p" not in targets
        and "depress_D_p_rev" not in targets
        and "external_D_p" not in targets
        and "adhd_D_p" not in targets
    ):  # ("latent_class_depression" in targets):
        if "latent_class_depression" in targets:
            # latent_class_depression has sporadic missing/empty entries; drop them.
            all_data = all_data[(all_data["latent_class_depression"] != "")]
        # Class rebalancing deferred to AFTER split
        # to prevent oversampled duplicates from appearing in both train and test.
        _needs_rebalance = class_rebalancing

    available = drop_high_missing_columns(all_data, threshold=0.7, verbose=verbose)
    dropped_columns = set(all_data.columns).difference(set(available.columns))

    if dropped_columns:
        logger.info(f"WARNING: Dropped columns at t = {t} due to entirely missing data: {dropped_columns}")

    raw_X = available.drop(targets, axis=1)
    raw_y = available[targets]

    # Drop rows where target is NaN (parent ASR items can have sporadic missingness)
    _target_notna = raw_y.iloc[:, 0].notna()
    if (~_target_notna).any():
        _n_dropped = (~_target_notna).sum()
        logger.info(f"Dropped {_n_dropped} rows with missing target values")
        raw_X = raw_X[_target_notna]
        raw_y = raw_y[_target_notna]

    # Before imputation and scaling
    # One Hot Encoding
    # OHE is fit on pre-split data. For categorical columns with
    # handle_unknown='ignore', this is not statistical leakage (no learned
    # statistics), but fit-on-train-only would be stricter.
    one_hot_features = [
        "sex",
        "sex_P",
        "L_site_id",
        "marital_status",
        "child_religion",
        "sex_orient_y",
        "trans_id_y",
        "crysflu_c",
        "anxadhd_c",
    ]
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

    for feature in one_hot_features:
        if feature in raw_X.columns:
            logger.info(f"Encoding {feature} for one hot encoding..")
            # Reshape data for encoding
            raw_X_cat = raw_X[[feature]].astype(str)

            # Fit and transform
            encoded_data = encoder.fit_transform(raw_X_cat)

            # Create column names for encoded features
            unique_values = encoder.categories_[0][1:]
            encoded_columns = [f"{feature}_{val}" for val in unique_values]

            # Convert to DataFrame
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=raw_X.index)

            # Replace original column with encoded columns
            raw_X = pd.concat([raw_X.drop(feature, axis=1), encoded_df], axis=1)
            logger.info(f"Encoded {feature} into {len(encoded_columns)} categories")

    # Preserve original index for downstream demographic lookups
    # Store the original index BEFORE resetting, so later stages can
    # align X_train/X_test rows back to the `sample` DataFrame.
    _original_index = raw_X.index.copy()
    raw_X = raw_X.reset_index(drop=True)
    raw_y = raw_y.reset_index(drop=True)

    # Leakage-free split → impute → scale
    # Raw data is split before any preprocessing estimator is fit. Imputation
    # and standardization parameters are estimated on X_train_raw only.
    logger.info("Splitting data prior to preprocessing...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        raw_X,
        raw_y,
        test_size=0.3,
        random_state=42,
        stratify=raw_y
        if (not is_regression(targets[0] if isinstance(targets, list) else targets) and is_classification(raw_y))
        else None,
    )
    # Apply class rebalancing to TRAINING data only
    if "_needs_rebalance" in dir() and _needs_rebalance:
        _train_combined = pd.concat([X_train_raw, y_train_raw], axis=1)
        _train_balanced = simple_balance_classes(
            _train_combined, target_column=str(targets[0]) if isinstance(targets, list) else str(targets)
        )
        X_train_raw = _train_balanced.drop(targets, axis=1, errors="ignore")
        y_train_raw = _train_balanced[targets] if isinstance(targets, list) else _train_balanced[[targets]]
        logger.info(f"  Class rebalancing applied to TRAINING data only ({len(X_train_raw)} samples)")

    # Fit imputer on training only
    selected_imputer = get_imputer(imputer)
    logger.info("Imputing the data (fit on train only)...")
    selected_imputer.fit(X_train_raw)

    X_train = pd.DataFrame(
        selected_imputer.transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index
    )
    X_test = pd.DataFrame(selected_imputer.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)

    # Handle y imputation (separate imputer: avoids X-fitted statistics on y)
    y_train = y_train_raw.copy()
    y_test = y_test_raw.copy()
    try:
        from sklearn.impute import SimpleImputer as _SI

        _y_imputer = _SI(strategy="median")
        _y_imputer.fit(y_train_raw)
        y_train = pd.DataFrame(_y_imputer.transform(y_train_raw), columns=y_train_raw.columns, index=y_train_raw.index)
        y_test = pd.DataFrame(_y_imputer.transform(y_test_raw), columns=y_test_raw.columns, index=y_test_raw.index)
    except Exception as e:
        logger.info("raw_y does not need to be imputed")

    # Scale X if requested
    logger.info("Scaling the data (fit on train only)...")
    if scale_x:
        scaler_x = StandardScaler()
        try:
            scaler_x.fit(X_train)
            X_train = pd.DataFrame(scaler_x.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler_x.transform(X_test), columns=X_test.columns, index=X_test.index)
        except Exception as e:
            logger.info("raw_X does not need to be scaled")

    # Scale y if requested (skip for classification targets)
    if scale_y and not is_classification(raw_y):
        scaler_y = StandardScaler()
        try:
            scaler_y.fit(y_train.values.reshape(-1, 1))
            y_train = pd.Series(
                scaler_y.transform(y_train.values.reshape(-1, 1)).flatten(),
                index=y_train.index,
                name=y_train.name if hasattr(y_train, "name") else None,
            )
            y_test = pd.Series(
                scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(),
                index=y_test.index,
                name=y_test.name if hasattr(y_test, "name") else None,
            )
        except Exception as e:
            logger.info("raw_y does not need to be scaled")

    ret = {}

    try:
        for category, vars in active_vfet_tp.items():
            available_vars = list(set(vars).difference(dropped_columns))
            ret[category] = (X_train[available_vars], X_test[available_vars], y_train, y_test)
    except KeyError:
        # Category variable list references a column not yet in imputed X; rebuild defensively.
        for category, vars in active_vfet_tp.items():
            # Get the available variables after removing dropped columns
            available_vars = list(set(vars).difference(dropped_columns))

            # Further filter to ensure available_vars exist in both X_train and X_test
            available_vars = [var for var in available_vars if var in X_train.columns and var in X_test.columns]

            if available_vars:
                ret[category] = (X_train[available_vars], X_test[available_vars], y_train, y_test)
            else:
                logger.info(f"WARNING: No available variables left for category: {category}")

    return ret


# across_categories_data()
# Prepares train/test splits for across-category analyses, drawing from the
# full flat predictor list (t0_variables … t4_variables) at the selected wave.
# Returns a single (X_train, X_test, y_train, y_test) tuple.
# Applies the same leakage-free preprocessing sequence as
# within_categories_data(): circular exclusion → missing-data filtering →
# one-hot encoding → raw split → imputation (train-only) → optional scaling.
def across_categories_data(
    t, targets, group=None, scale_x=True, scale_y=True, verbose=False, class_rebalancing=False, feature_tp=None
):
    from bpsp.analysis.runner import imputer

    # feature_tp: if set, pull FEATURES from this timepoint and TARGETS from t.
    # None (default) preserves existing behavior (features and targets same wave).
    feat_t = feature_tp if feature_tp is not None else t

    available_columns = set(ctx.sample.columns)

    if feat_t == 0:
        across_categories = list(ctx.t0_variables)
    elif feat_t == 1:
        across_categories = list(ctx.t1_variables)
    elif feat_t == 2:
        across_categories = list(ctx.t2_variables)
    elif feat_t == 3:
        across_categories = list(ctx.t3_variables)
    elif feat_t == 4:
        across_categories = list(ctx.t4_variables)
    else:
        raise ValueError(f"Invalid feature time point: {feat_t}")

    # Circular exclusions are always keyed on the TARGET timepoint (t), not feat_t.
    # Cross-tp runs (e.g. T0 features → T2 target) are inherently safe by design,
    # but any target-wave exclusions still apply to keep the pool clean.
    target_name = targets[0] if targets else ""
    across_excl, _ = get_circular_exclusions(target_name, timepoint=t)
    if across_excl:
        before_count = len(across_categories)
        across_categories = [v for v in across_categories if v not in across_excl]
        removed_count = before_count - len(across_categories)
        if removed_count > 0:
            logger.info(
                f"[CIRCULAR] Removed {removed_count} circular variables from across-category pool for target: {target_name}"
            )

    # Remove columns not present in dataset (based on feat_t variable space)
    across_categories = [var for var in across_categories if var in available_columns]

    # Warn about any expected variables missing from the dataset (use feat_t)
    dropped_columns = set(
        [x for xs in ctx.variables_for_each_time_point[feat_t].values() for x in xs] + targets
    ).difference(available_columns)
    if dropped_columns:
        logger.info(
            f"WARNING: The following columns were not found in the dataset and have been dropped: {dropped_columns}"
        )

    # Remove peer-victimisation variables absent/unreliable at T1 and T3 (feature side)
    if int(feat_t) == 3 or int(feat_t) == 1:
        variables_to_remove = [
            "peq_ss_relational_victim",
            "peq_ss_reputation_aggs",
            "peq_ss_reputation_victim",
            "peq_ss_overt_aggression",
            "peq_ss_overt_victim",
            "peq_ss_relational_aggs",
        ]
        across_categories = [var for var in across_categories if var not in variables_to_remove]

    # Build all_data
    # Cross-timepoint path: features from feat_t rows, targets from t rows,
    # merged on subject ID. Group filter is applied to the TARGET wave so that
    # subgroup membership (e.g. sex, ALE) is defined at the outcome timepoint.
    if feature_tp is not None and feature_tp != t:
        # 1. Get target-wave rows and apply any group filter
        tgt_rows = ctx.sample[ctx.sample["time"] == t].copy()
        if group == "high_ale":
            tgt_rows = tgt_rows[tgt_rows["high_ale"] == True]
        elif group == "high_ale_severe_p_mh":
            tgt_rows = tgt_rows[tgt_rows["high_ale_severe_p_mh"] == True]
        elif group == "low_ale":
            tgt_rows = tgt_rows[tgt_rows["low_ale_children_p"] == True]
        elif group == "Female":
            tgt_rows = tgt_rows[tgt_rows["sex"] == 2]
        elif group == "Male":
            tgt_rows = tgt_rows[tgt_rows["sex"] == 1]

        valid_subjects = tgt_rows["subject"].values

        # 2. Get feature-wave rows, restricted to the group-filtered subjects
        feat_rows = ctx.sample[(ctx.sample["time"] == feat_t) & (ctx.sample["subject"].isin(valid_subjects))].copy()

        # 3. Merge: features from feat_t, target from t, on subject
        feat_cols_avail = [c for c in across_categories if c in feat_rows.columns]
        tgt_cols_avail = [c for c in targets if c in tgt_rows.columns]

        _feat_df = feat_rows[["subject"] + feat_cols_avail].drop_duplicates("subject")
        _tgt_df = tgt_rows[["subject"] + tgt_cols_avail].drop_duplicates("subject")

        all_data = _feat_df.merge(_tgt_df, on="subject", how="inner").drop(columns="subject")

        if verbose:
            logger.info(
                f"[Cross-TP] feat_tp={feat_t}, target_tp={t}, "
                f"group={group if group else 'full'}, merged N={len(all_data)}"
            )

    # Same-timepoint path: original logic preserved exactly
    else:
        if group:
            if group == "high_ale":
                all_vars = across_categories + targets + [f"{group}"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data[all_data["high_ale"] == True]  # high_ale

            elif group == "high_ale_severe_p_mh":
                all_vars = across_categories + targets + [f"{group}"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data[all_data["high_ale_severe_p_mh"] == True]  # high_ale

            elif group == "low_ale":
                all_vars = across_categories + targets + ["low_ale_children_p"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data[all_data["low_ale_children_p"] == True]

            elif group == "Female":
                all_vars = across_categories + targets + ["sex"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data.loc[:, ~all_data.columns.duplicated()]
                all_data = all_data[all_data["sex"] == 2]

            elif group == "Male":
                all_vars = across_categories + targets + ["sex"]
                all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                all_data = all_data.loc[:, ~all_data.columns.duplicated()]
                all_data = all_data[all_data["sex"] == 1]

            else:
                all_vars = across_categories + targets
                all_data = ctx.sample.query(f"time == {t}")[all_vars]

        else:
            all_vars = across_categories + targets
            all_data = ctx.sample.query(f"time == {t}")[all_vars]

        if verbose:
            logger.info(
                f"Gathered across category data from {all_data.shape[0]} subjects for {group if group else 'full'} group at t = {t}"
            )

    # Downstream processing
    if (
        "depress_D_p" not in targets
        and "depress_D_p_rev" not in targets
        and "external_D_p" not in targets
        and "adhd_D_p" not in targets
    ):  # ("latent_class_depression" in targets):
        # latent_class_depression has sporadic missing/empty entries; drop them.
        if "latent_class_depression" in targets:
            all_data = all_data[(all_data["latent_class_depression"] != "")]
        # Class rebalancing deferred to AFTER split
        # to prevent oversampled duplicates from appearing in both train and test.
        _needs_rebalance = class_rebalancing

    available = drop_high_missing_columns(all_data, threshold=0.7, verbose=verbose)
    dropped_columns = set(all_data.columns).difference(set(available.columns))

    if dropped_columns:
        logger.info(f"WARNING: Dropped columns at t = {t} due to entirely missing data: {dropped_columns}")

    raw_X = available.drop(targets, axis=1)
    raw_y = available[targets]

    # Drop rows where target is NaN (parent ASR items can have sporadic missingness)
    _target_notna = raw_y.iloc[:, 0].notna()
    if (~_target_notna).any():
        _n_dropped = (~_target_notna).sum()
        logger.info(f"Dropped {_n_dropped} rows with missing target values")
        raw_X = raw_X[_target_notna]
        raw_y = raw_y[_target_notna]

    # Before imputation and scaling
    # One Hot Encoding
    one_hot_features = [
        "sex",
        "sex_P",
        "L_site_id",
        "marital_status",
        "child_religion",
        "sex_orient_y",
        "trans_id_y",
        "crysflu_c",
        "anxadhd_c",
    ]
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

    for feature in one_hot_features:
        if feature in raw_X.columns:
            logger.info(f"Encoding {feature} for one hot encoding..")
            # Reshape data for encoding
            raw_X_cat = raw_X[[feature]].astype(str)

            # Fit and transform
            encoded_data = encoder.fit_transform(raw_X_cat)

            # Create column names for encoded features
            unique_values = encoder.categories_[0][1:]
            encoded_columns = [f"{feature}_{val}" for val in unique_values]

            # Convert to DataFrame
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=raw_X.index)

            # Replace original column with encoded columns
            raw_X = pd.concat([raw_X.drop(feature, axis=1), encoded_df], axis=1)
            logger.info(f"Encoded {feature} into {len(encoded_columns)} categories")

    # Preserve original index for downstream demographic lookups
    _original_index = raw_X.index.copy()
    raw_X = raw_X.reset_index(drop=True)
    raw_y = raw_y.reset_index(drop=True)

    # Split before fitting any preprocessing estimator to prevent leakage.
    # Imputer and scaler are fit on X_train_raw only, then applied to both splits.

    # Split raw data FIRST
    logger.info("Splitting data (before preprocessing to prevent leakage)...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        raw_X,
        raw_y,
        test_size=0.3,
        random_state=42,
        stratify=raw_y
        if (not is_regression(targets[0] if isinstance(targets, list) else targets) and is_classification(raw_y))
        else None,
    )
    # Apply class rebalancing to TRAINING data only
    if "_needs_rebalance" in dir() and _needs_rebalance:
        _train_combined = pd.concat([X_train_raw, y_train_raw], axis=1)
        _train_balanced = simple_balance_classes(
            _train_combined, target_column=str(targets[0]) if isinstance(targets, list) else str(targets)
        )
        X_train_raw = _train_balanced.drop(targets, axis=1, errors="ignore")
        y_train_raw = _train_balanced[targets] if isinstance(targets, list) else _train_balanced[[targets]]
        logger.info(f"  Class rebalancing applied to TRAINING data only ({len(X_train_raw)} samples)")

    # Fit imputer on training only
    selected_imputer = get_imputer(imputer)
    logger.info("Imputing the data (fit on train only)...")
    selected_imputer.fit(X_train_raw)

    X_train = pd.DataFrame(
        selected_imputer.transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index
    )
    X_test = pd.DataFrame(selected_imputer.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)

    # Handle y imputation (separate imputer: avoids X-fitted statistics on y)
    y_train = y_train_raw.copy()
    y_test = y_test_raw.copy()
    try:
        from sklearn.impute import SimpleImputer as _SI

        _y_imputer = _SI(strategy="median")
        _y_imputer.fit(y_train_raw)
        y_train = pd.DataFrame(_y_imputer.transform(y_train_raw), columns=y_train_raw.columns, index=y_train_raw.index)
        y_test = pd.DataFrame(_y_imputer.transform(y_test_raw), columns=y_test_raw.columns, index=y_test_raw.index)
    except Exception as e:
        logger.info("raw_y does not need to be imputed")

    # Scale X if requested (fit on training only)
    logger.info("Scaling the data (fit on train only)...")
    if scale_x:
        scaler_x = StandardScaler()
        try:
            scaler_x.fit(X_train)
            X_train = pd.DataFrame(scaler_x.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler_x.transform(X_test), columns=X_test.columns, index=X_test.index)
        except Exception as e:
            logger.info("raw_X does not need to be scaled")

    # Scale y if requested (skip for classification targets)
    if scale_y and not is_classification(raw_y):
        scaler_y = StandardScaler()
        try:
            scaler_y.fit(y_train)
            y_train = pd.DataFrame(scaler_y.transform(y_train), columns=y_train.columns, index=y_train.index)
            y_test = pd.DataFrame(scaler_y.transform(y_test), columns=y_test.columns, index=y_test.index)
        except Exception as e:
            logger.info("raw_y does not need to be scaled")

    return X_train, X_test, y_train, y_test


# within_categories_data_combinations()
# Exploratory function: pairs a single base category with each remaining
# category in turn and prepares a merged train/test split for each pair.
# Useful for assessing the marginal contribution of individual categories
# when combined with a fixed reference domain (e.g., neuroimaging + each
# psychosocial domain).
def within_categories_data_combinations(
    t, targets, base_category, group=None, scale_x=True, scale_y=True, verbose=False, class_rebalancing=False
):
    from bpsp.analysis.runner import imputer

    target_name = targets[0] if targets else ""
    if is_parent_target(target_name):
        ctx.variables_for_each_time_point = ctx.parent_variables_for_each_time_point
        logger.info(f"[VARS] Using PARENT within-category structure for target: {target_name}")
    else:
        ctx.variables_for_each_time_point = ctx.child_variables_for_each_time_point
        logger.info(f"[VARS] Using CHILD within-category structure for target: {target_name}")

    # Apply circular exclusions
    across_excl, within_excl = get_circular_exclusions(target_name, timepoint=t)
    active_vfet_tp = {
        cat: vars_list for cat, vars_list in ctx.variables_for_each_time_point[t].items() if cat not in within_excl
    }
    if within_excl:
        skipped = within_excl & set(ctx.variables_for_each_time_point[t].keys())
        if skipped:
            logger.info(f"[CIRCULAR] Excluded within-categories for {target_name}: {skipped}")

    # Flatten retained categories, removing individually excluded circular variables.
    within_categories = [x for xs in active_vfet_tp.values() for x in xs if x not in across_excl] + targets
    if across_excl:
        individual_removed = len([x for xs in active_vfet_tp.values() for x in xs if x in across_excl])
        if individual_removed > 0:
            logger.info(
                f"[CIRCULAR] Also removed {individual_removed} individual circular variables from within-category features for {target_name}"
            )

    within_categories = list(set(within_categories))

    # Get available columns in the DataFrame
    available_columns = set(ctx.sample.columns)

    # Remove columns from within_categories that are not in available_columns
    within_categories = [var for var in within_categories if var in available_columns]

    # If any columns were removed, notify the user
    dropped_columns = set([x for xs in ctx.variables_for_each_time_point[t].values() for x in xs] + targets).difference(
        available_columns
    )

    if dropped_columns:
        logger.info(
            f"WARNING: The following columns were not found in the dataset and have been dropped: {dropped_columns}"
        )

    # Remove specific variables
    if int(t) == 3 or int(t) == 1:
        variables_to_remove = [
            "peq_ss_relational_victim",
            "peq_ss_reputation_aggs",
            "peq_ss_reputation_victim",
            "peq_ss_overt_aggression",
            "peq_ss_overt_victim",
            "peq_ss_relational_aggs",
        ]
        within_categories = [var for var in within_categories if var not in variables_to_remove]

    ret = {}
    if base_category:
        # Use actual timepoint t (not hardcoded 0) for category enumeration.
        # Filter out circular-excluded categories from pairing candidates.
        other_categories = [cat for cat in active_vfet_tp.keys() if cat != base_category]
        logger.info(f"Mixing {base_category} with: {other_categories}")

        # Avoid mutating the group parameter across loop iterations.
        _resolved_group = "low_ale_children_p" if group == "low_ale" else group

        for category in other_categories:
            if _resolved_group:
                if _resolved_group == "high_ale":
                    all_vars = within_categories + [f"{_resolved_group}"]
                    all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                    all_data = all_data[all_data["high_ale"] == True]

                elif _resolved_group == "high_ale_severe_p_mh":
                    all_vars = within_categories + [f"{_resolved_group}"]
                    all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                    all_data = all_data[all_data["high_ale_severe_p_mh"] == True]

                elif _resolved_group == "low_ale_children_p":
                    all_vars = within_categories + ["low_ale_children_p"]
                    all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                    all_data = all_data[all_data["low_ale_children_p"] == True]

                elif _resolved_group == "Female":
                    all_vars = within_categories + ["sex"]
                    all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                    all_data = all_data.loc[:, ~all_data.columns.duplicated()]
                    all_data = all_data[all_data["sex"] == 2]

                elif _resolved_group == "Male":
                    all_vars = within_categories + ["sex"]
                    all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
                    all_data = all_data.loc[:, ~all_data.columns.duplicated()]
                    all_data = all_data[all_data["sex"] == 1]

                else:
                    all_vars = within_categories
                    all_data = ctx.sample.query(f"time == {t}")[all_vars]

            else:
                all_vars = within_categories
                all_data = ctx.sample.query(f"time == {t}")[all_vars]

            try:
                variables_total = list(active_vfet_tp[f"{base_category}"]) + list(active_vfet_tp[f"{category}"])
            except Exception as e:
                logger.info(f"  Skipped: '{category}' not available for this tp/target/group combination.")
                continue

            # Apply across_excl to the per-combo variable list as well.
            if across_excl:
                variables_total = [v for v in variables_total if v not in across_excl]

            variables_total += targets

            if verbose:
                logger.info(f"\nProcessing {base_category} + {category}")
                logger.info(f"Total variables before filtering: {len(variables_total)}")

            # Filter out variables that aren't in the dataset
            available_vars = [var for var in variables_total if var in ctx.sample.columns]
            missing_vars = set(variables_total) - set(available_vars)

            if missing_vars and verbose:
                logger.info(f"WARNING: The following variables were not found in the dataset:\n{sorted(missing_vars)}")
                logger.info(f"Continuing with {len(available_vars)} available variables")

            # Subset only the available vars
            available_vars = list(dict.fromkeys(available_vars))  # Remove duplicates while preserving order
            all_data = all_data[available_vars]

            if (
                "depress_D_p" not in targets
                and "depress_D_p_rev" not in targets
                and "external_D_p" not in targets
                and "adhd_D_p" not in targets
            ):
                if "latent_class_depression" in targets:
                    all_data = all_data[(all_data["latent_class_depression"] != "")]
                if class_rebalancing:
                    all_data = simple_balance_classes(all_data, target_column=f"{ctx.target_options}")
                    display(all_data[f"{ctx.target_options}"].value_counts())

            available = drop_high_missing_columns(all_data, threshold=0.7, verbose=verbose)
            dropped_columns = set(all_data.columns).difference(set(available.columns))

            if dropped_columns:
                logger.info(f"WARNING: Dropped columns at t = {t} due to entirely missing data: {dropped_columns}")

            raw_X = available.drop(targets, axis=1)
            raw_y = available[targets]

            # Drop rows where target is NaN
            _target_notna = raw_y.iloc[:, 0].notna()
            if (~_target_notna).any():
                _n_dropped = (~_target_notna).sum()
                logger.info(f"Dropped {_n_dropped} rows with missing target values")
                raw_X = raw_X[_target_notna]
                raw_y = raw_y[_target_notna]

            # One Hot Encoding
            one_hot_features = [
                "sex",
                "sex_P",
                "L_site_id",
                "marital_status",
                "child_religion",
                "sex_orient_y",
                "trans_id_y",
                "crysflu_c",
                "anxadhd_c",
            ]
            encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

            for feature in one_hot_features:
                if feature in raw_X.columns:
                    logger.info(f"Encoding {feature} for one hot encoding..")
                    raw_X_cat = raw_X[[feature]].astype(str)
                    encoded_data = encoder.fit_transform(raw_X_cat)
                    unique_values = encoder.categories_[0][1:]
                    encoded_columns = [f"{feature}_{val}" for val in unique_values]
                    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=raw_X.index)
                    raw_X = pd.concat([raw_X.drop(feature, axis=1), encoded_df], axis=1)
                    logger.info(f"Encoded {feature} into {len(encoded_columns)} categories")

            # Reset to RangeIndex so positional indices align
            raw_X = raw_X.reset_index(drop=True)
            raw_y = raw_y.reset_index(drop=True)

            # Split raw data before fitting any preprocessing estimator.
            logger.info("Splitting data prior to preprocessing...")
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                raw_X,
                raw_y,
                test_size=0.3,
                random_state=42,
                stratify=raw_y
                if (
                    not is_regression(targets[0] if isinstance(targets, list) else targets) and is_classification(raw_y)
                )
                else None,
            )

            # Fit imputer on training only
            selected_imputer = get_imputer(imputer)
            logger.info("Imputing the data (fit on train only)...")
            selected_imputer.fit(X_train_raw)

            X_train = pd.DataFrame(
                selected_imputer.transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index
            )
            X_test = pd.DataFrame(
                selected_imputer.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index
            )

            # Handle y imputation (separate imputer)
            y_train = y_train_raw.copy()
            y_test = y_test_raw.copy()
            try:
                from sklearn.impute import SimpleImputer as _SI

                _y_imputer = _SI(strategy="median")
                _y_imputer.fit(y_train_raw)
                y_train = pd.DataFrame(
                    _y_imputer.transform(y_train_raw), columns=y_train_raw.columns, index=y_train_raw.index
                )
                y_test = pd.DataFrame(
                    _y_imputer.transform(y_test_raw), columns=y_test_raw.columns, index=y_test_raw.index
                )
            except Exception as e:
                logger.info("raw_y does not need to be imputed")

            # Scale X if requested (fit on training only)
            logger.info("Scaling the data (fit on train only)...")
            if scale_x:
                scaler_x = StandardScaler()
                try:
                    scaler_x.fit(X_train)
                    X_train = pd.DataFrame(scaler_x.transform(X_train), columns=X_train.columns, index=X_train.index)
                    X_test = pd.DataFrame(scaler_x.transform(X_test), columns=X_test.columns, index=X_test.index)
                except Exception as e:
                    logger.info("raw_X does not need to be scaled")

            # Scale y if requested (fit on training only)
            if scale_y:
                scaler_y = StandardScaler()
                try:
                    scaler_y.fit(y_train)
                    y_train = pd.DataFrame(scaler_y.transform(y_train), columns=y_train.columns, index=y_train.index)
                    y_test = pd.DataFrame(scaler_y.transform(y_test), columns=y_test.columns, index=y_test.index)
                except Exception as e:
                    logger.info("raw_y does not need to be scaled")

            # Store results
            combination_name = f"{base_category} + {category}"
            ret[combination_name] = (X_train, X_test, y_train, y_test)

    return ret


# Quantile Analysis Utilities
# These functions segment data by predictor or target quantiles to examine
# whether predictive performance varies across the distribution of a given
# feature or outcome. Used for exploratory heterogeneity analyses.


def analyze_category_quantiles(data_dict):
    """Compute Q1/Q2/Q3 quantile boundaries and bin counts for every feature
    within each category.  Returns a nested dict keyed by category → feature.

    Parameters
    data_dict : dict
        {category: (X_train, X_test, y_train, y_test)} as returned by
        within_categories_data().

    Returns
    dict
        Nested structure: {category: {'total_samples': int,
        'feature_quantiles': {feature: {Q1/Q2/Q3 counts and values}}}}
    """
    results = {}

    for category, (X_train, X_test, y_train, y_test) in data_dict.items():
        # Combine train and test data for complete picture
        X_combined = pd.concat([X_train, X_test])

        # Calculate quantiles
        quantiles = X_combined.quantile([0.25, 0.5, 0.75])

        # Count data points in each quantile range
        quantile_counts = {}

        # For each column in the dataset
        for column in X_combined.columns:
            q1, q2, q3 = quantiles.loc[0.25, column], quantiles.loc[0.5, column], quantiles.loc[0.75, column]

            counts = {
                "Q1 (0-25%)": len(X_combined[X_combined[column] <= q1]),
                "Q2 (25-50%)": len(X_combined[(X_combined[column] > q1) & (X_combined[column] <= q2)]),
                "Q3 (50-75%)": len(X_combined[(X_combined[column] > q2) & (X_combined[column] <= q3)]),
                "Q4 (75-100%)": len(X_combined[X_combined[column] > q3]),
                "quantile_values": {"Q1": q1, "Q2 (median)": q2, "Q3": q3},
            }
            quantile_counts[column] = counts

        results[category] = {"total_samples": len(X_combined), "feature_quantiles": quantile_counts}

    return results


def print_quantile_analysis(results):
    """
    Print the quantile analysis results in a readable format
    """
    for category, analysis in results.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Category: {category}")
        logger.info(f"Total samples: {analysis['total_samples']}")
        logger.info(f"{'=' * 80}")

        for feature, counts in analysis["feature_quantiles"].items():
            logger.info(f"\nFeature: {feature}")
            logger.info("Quantile values:")
            for q_name, value in counts["quantile_values"].items():
                logger.info(f"  {q_name}: {value:.3f}")
            logger.info("Data points in each quantile:")
            for q_name, count in {k: v for k, v in counts.items() if k != "quantile_values"}.items():
                logger.info(f"  {q_name}: {count}")
            logger.info("-" * 40)


def segment_data_by_quantiles(X_train, X_test, y_train, y_test, feature_name):
    """Split (X_train, X_test, y_train, y_test) into four quartile bins
    defined by the distribution of feature_name, preserving the original
    train/test membership within each bin.
    """
    # Reset indices before concatenation to avoid duplicates
    X_train_reset = X_train.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    # Create a marker column to identify train/test split
    X_train_reset["is_train"] = True
    X_test_reset["is_train"] = False

    # Combine train and test
    X_combined = pd.concat([X_train_reset, X_test_reset], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train_reset, y_test_reset], axis=0, ignore_index=True)

    # Calculate quantile values
    q1, q2, q3 = X_combined[feature_name].quantile([0.25, 0.5, 0.75])

    # Create masks for each quantile
    q1_mask = X_combined[feature_name] <= q1
    q2_mask = (X_combined[feature_name] > q1) & (X_combined[feature_name] <= q2)
    q3_mask = (X_combined[feature_name] > q2) & (X_combined[feature_name] <= q3)
    q4_mask = X_combined[feature_name] > q3

    segments = {}
    for q_name, mask in [("Q1", q1_mask), ("Q2", q2_mask), ("Q3", q3_mask), ("Q4", q4_mask)]:
        # Get the masked data
        X_segment = X_combined[mask].copy()
        y_segment = y_combined[mask].copy()

        if len(X_segment) == 0:
            logger.info(f"Warning: No data in {q_name} segment for feature {feature_name}")
            continue

        # Split back into train and test using the marker column
        train_mask = X_segment["is_train"]
        test_mask = ~train_mask

        # Remove the marker column
        X_segment = X_segment.drop("is_train", axis=1)

        segments[q_name] = {
            "X_train": X_segment[train_mask],
            "X_test": X_segment[test_mask],
            "y_train": y_segment[train_mask],
            "y_test": y_segment[test_mask],
        }

    return segments


def segment_data_by_target_quantiles(X_train, X_test, y_train, y_test):
    """
    Segments data into quantiles based on the target variable.
    """
    # Reset indices before concatenation to avoid duplicates
    X_train_reset = X_train.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    # Create a marker column to identify train/test split
    X_train_reset["is_train"] = True
    X_test_reset["is_train"] = False

    # Combine train and test
    X_combined = pd.concat([X_train_reset, X_test_reset], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train_reset, y_test_reset], axis=0, ignore_index=True)

    # Calculate quantile values for the target variable
    q1, q2, q3 = y_combined.iloc[:, 0].quantile([0.25, 0.5, 0.75])

    # Create masks for each quantile
    q1_mask = y_combined.iloc[:, 0] <= q1
    q2_mask = (y_combined.iloc[:, 0] > q1) & (y_combined.iloc[:, 0] <= q2)
    q3_mask = (y_combined.iloc[:, 0] > q2) & (y_combined.iloc[:, 0] <= q3)
    q4_mask = y_combined.iloc[:, 0] > q3

    segments = {}
    for q_name, mask in [("Q1", q1_mask), ("Q2", q2_mask), ("Q3", q3_mask), ("Q4", q4_mask)]:
        # Get the masked data
        X_segment = X_combined[mask].copy()
        y_segment = y_combined[mask].copy()

        if len(X_segment) == 0:
            logger.info(f"Warning: No data in {q_name} segment")
            continue

        # Split back into train and test using the marker column
        train_mask = X_segment["is_train"]
        test_mask = ~train_mask

        # Remove the marker column
        X_segment = X_segment.drop("is_train", axis=1)

        segments[q_name] = {
            "X_train": X_segment[train_mask],
            "X_test": X_segment[test_mask],
            "y_train": y_segment[train_mask],
            "y_test": y_segment[test_mask],
            "quantile_range": (q1, q2, q3),  # Store quantile boundaries for reference
        }

        logger.info(f"{q_name} segment ranges for target:")
        if q_name == "Q1":
            logger.info(f"  Range: <= {q1:.3f}")
        elif q_name == "Q2":
            logger.info(f"  Range: {q1:.3f} - {q2:.3f}")
        elif q_name == "Q3":
            logger.info(f"  Range: {q2:.3f} - {q3:.3f}")
        else:
            logger.info(f"  Range: > {q3:.3f}")
        logger.info(f"  Samples - Train: {len(segments[q_name]['X_train'])}, Test: {len(segments[q_name]['X_test'])}")

    return segments


def run_quantile_analysis(data_dict, tp, target_options, quantile_segment=False):
    """
    Runs CatBoost analysis on each category, segmenting by target variable quantiles.
    """
    from bpsp.analysis.runner import run_catboost_analysis

    results = {}

    for category, (X_train, X_test, y_train, y_test) in data_dict.items():
        logger.info(f"\nAnalyzing category: {category}")

        if not quantile_segment:
            # Regular analysis without quantile segmentation
            model, _, _ = run_catboost_analysis(X_train, X_test, y_train, y_test, target_options, tp, category)
            results[category] = {"model": model, "feature_importance": getattr(model, "feature_importances_", None)}
        else:
            logger.info(f"\nSegmenting by {target_options} quantiles")
            try:
                segments = segment_data_by_target_quantiles(X_train, X_test, y_train, y_test)

                quantile_results = {}
                for quantile, data in segments.items():
                    logger.info(f"\nAnalyzing {category} for {target_options} {quantile}")

                    # Check if we have enough samples in this segment
                    if len(data["X_train"]) < 50 or len(data["X_test"]) < 20:
                        logger.info(f"Skipping {quantile} due to insufficient samples")
                        continue

                    try:
                        model, _, _ = run_catboost_analysis(
                            data["X_train"],
                            data["X_test"],
                            data["y_train"],
                            data["y_test"],
                            target_options,
                            tp,
                            f"{category}_{quantile}",
                        )

                        quantile_results[quantile] = {
                            "model": model,
                            "feature_importance": getattr(model, "feature_importances_", None),
                            "n_samples": len(data["X_train"]) + len(data["X_test"]),
                            "quantile_range": data["quantile_range"],
                        }
                    except Exception as e:
                        logger.info(f"Error in analyzing {quantile}: {str(e)}")
                        continue

                if quantile_results:  # Only store if we have results
                    results[category] = quantile_results
            except Exception as e:
                logger.info(f"Error processing category {category}: {str(e)}")
                continue

    return results


# Assign sample weights based on target quartile membership so that higher
# outcome values receive greater emphasis during training. Weights are
# passed directly to the CatBoost model via the sample_weight parameter.
def calculate_quantile_weights(y_values):
    """Compute sample weights by target quartile with adaptive boundaries.

    Falls back to equal-size grouping if any bin would otherwise be empty.

    Parameters
    y_values : array-like
        Target variable values (regression outcomes).

    Returns
    numpy.ndarray
        Per-sample weights in {1.0, 2.0, 3.0, 4.0}.
    """
    # Convert to numpy array and flatten
    y_flat = np.array(y_values).ravel()

    # Get unique values and their counts
    unique_vals = np.unique(y_flat)

    if len(unique_vals) < 4:
        logger.info("Warning: Less than 4 unique values in target. Using available unique values for grouping.")
        weights = np.ones(len(y_flat))
        for i, val in enumerate(unique_vals, 1):
            weights[y_flat == val] = i

        # Print distribution
        logger.info("\nUnique value weights:")
        for i, val in enumerate(unique_vals, 1):
            count = np.sum(y_flat == val)
            logger.info(f"Value {val:.3f}: weight {i}.0, {count} samples")

        return weights

    # Find adaptive quantile boundaries that ensure samples in each group
    n_samples = len(y_flat)
    sorted_vals = np.sort(y_flat)

    # Try to get roughly equal groups while ensuring no empty groups
    target_size = n_samples // 4

    # Find boundaries that create non-empty groups
    q1_idx = max(target_size - target_size // 2, 0)
    q2_idx = 2 * target_size
    q3_idx = 3 * target_size

    # indices don't exceed array bounds
    q1_idx = min(q1_idx, n_samples - 3)
    q2_idx = min(q2_idx, n_samples - 2)
    q3_idx = min(q3_idx, n_samples - 1)

    # Get the actual values at these positions
    q1 = sorted_vals[q1_idx]
    q2 = sorted_vals[q2_idx]
    q3 = sorted_vals[q3_idx]

    # Initialize weights array
    weights = np.ones(len(y_flat))

    # Assign weights based on adaptive quantiles
    weights[y_flat > q1] = 2.0
    weights[y_flat > q2] = 3.0
    weights[y_flat > q3] = 4.0

    # Print distribution information
    logger.info("\nAdaptive quantile ranges and weights:")
    logger.info(f"Q1 (≤{q1:.3f}): weight 1.0, {np.sum(y_flat <= q1)} samples")
    logger.info(f"Q2 ({q1:.3f}-{q2:.3f}): weight 2.0, {np.sum((y_flat > q1) & (y_flat <= q2))} samples")
    logger.info(f"Q3 ({q2:.3f}-{q3:.3f}): weight 3.0, {np.sum((y_flat > q2) & (y_flat <= q3))} samples")
    logger.info(f"Q4 (>{q3:.3f}): weight 4.0, {np.sum(y_flat > q3)} samples")

    # Verify no empty groups
    for i, (lower, upper) in enumerate([(-np.inf, q1), (q1, q2), (q2, q3), (q3, np.inf)]):
        count = np.sum((y_flat > lower) & (y_flat <= upper)) if i < 3 else np.sum(y_flat > lower)
        if count == 0:
            logger.info(f"\nWarning: Found empty group {i + 1}, adjusting boundaries...")
            return calculate_quantile_weights_equal_size(y_flat)

    return weights


def calculate_quantile_weights_equal_size(y_values):
    """Fallback: rank samples and assign weights based on equal-size quantile bins."""
    n_samples = len(y_values)
    ranks = pd.Series(y_values).rank(method="first")
    group_size = n_samples // 4

    weights = np.ones(n_samples)
    weights[ranks > group_size] = 2.0
    weights[ranks > 2 * group_size] = 3.0
    weights[ranks > 3 * group_size] = 4.0

    # Print distribution information
    logger.info("\nEqual-size group weights:")
    logger.info(f"Group 1: weight 1.0, {np.sum(weights == 1.0)} samples")
    logger.info(f"Group 2: weight 2.0, {np.sum(weights == 2.0)} samples")
    logger.info(f"Group 3: weight 3.0, {np.sum(weights == 3.0)} samples")
    logger.info(f"Group 4: weight 4.0, {np.sum(weights == 4.0)} samples")

    return weights


def calculate_top_quartile_weights(y_values):
    """Apply a binary weighting scheme that up-weights samples in the top 20 %
    of the target distribution (weight = 1.25) relative to the remainder
    (weight = 1.0).  Intended for use when clinical interest is concentrated
    in the high-symptom tail.

    Parameters
    y_values : array-like
        Continuous target variable values.

    Returns
    numpy.ndarray
        Per-sample weights in {1.0, 1.25}.
    """
    # Convert to numpy array and flatten
    y_flat = np.array(y_values).ravel()

    # Calculate 80th percentile
    q3 = np.percentile(y_flat, 80)

    # Initialize all weights to 1.0
    weights = np.ones(len(y_flat))

    # Set weight 1.25 for top 20%
    weights[y_flat > q3] = 1.25

    # Print distribution information
    logger.info("\nBinary weighting scheme:")
    logger.info(f"Bottom 80% (≤{q3:.3f}): weight 1.0, {np.sum(y_flat <= q3)} samples")
    logger.info(f"Top 20% (>{q3:.3f}): weight 1.25, {np.sum(y_flat > q3)} samples")

    # Verify distribution
    weight_dist = pd.Series(weights).value_counts()
    logger.info("\nWeight distribution:")
    for weight, count in weight_dist.items():
        logger.info(f"Weight {weight:.1f}: {count} samples ({count / len(weights) * 100:.1f}%)")

    return weights


# K-Fold Variants
# These functions return raw (un-imputed) feature matrices together with a
# KFold/StratifiedKFold splitter object. Imputation, scaling, and model
# fitting are deferred to train_and_evaluate_model_kfold() so that the full
# preprocessing pipeline is applied independently within each fold, preventing
# any leakage from test data into fitted preprocessing parameters.

from sklearn.model_selection import KFold, StratifiedKFold


# Within-category variant: returns {category: (X_raw, y_raw, kf, scale_x, scale_y)}
def within_categories_data_kfold(
    t, targets, n_splits=5, group=None, scale_x=False, scale_y=False, verbose=False, class_rebalancing=False
):
    target_name = targets[0] if targets else ""
    if is_parent_target(target_name):
        ctx.variables_for_each_time_point = ctx.parent_variables_for_each_time_point
        logger.info(f"[VARS] Using PARENT within-category structure for target: {target_name}")
    else:
        ctx.variables_for_each_time_point = ctx.child_variables_for_each_time_point
        logger.info(f"[VARS] Using CHILD within-category structure for target: {target_name}")

    # Apply circular exclusions
    across_excl, within_excl = get_circular_exclusions(target_name, timepoint=t)
    active_vfet_tp = {
        cat: vars_list for cat, vars_list in ctx.variables_for_each_time_point[t].items() if cat not in within_excl
    }
    if within_excl:
        skipped = within_excl & set(ctx.variables_for_each_time_point[t].keys())
        if skipped:
            logger.info(f"[CIRCULAR] Excluded within-categories for {target_name}: {skipped}")

    # Flatten retained categories, removing individually excluded circular variables.
    within_categories = [x for xs in active_vfet_tp.values() for x in xs if x not in across_excl] + targets
    if across_excl:
        individual_removed = len([x for xs in active_vfet_tp.values() for x in xs if x in across_excl])
        if individual_removed > 0:
            logger.info(f"[CIRCULAR] Also removed {individual_removed} individual circular variables for {target_name}")

    within_categories = list(set(within_categories))

    # Get available columns in the DataFrame
    available_columns = set(ctx.sample.columns)

    # Remove columns from within_categories that are not in available_columns
    within_categories = [var for var in within_categories if var in available_columns]

    # If any columns were removed, notify the user
    dropped_columns = set([x for xs in ctx.variables_for_each_time_point[t].values() for x in xs] + targets).difference(
        available_columns
    )

    if dropped_columns:
        logger.info(
            f"WARNING: The following columns were not found in the dataset and have been dropped: {dropped_columns}"
        )

    # Remove specific variables
    if int(t) == 3 or int(t) == 1:
        variables_to_remove = [
            "peq_ss_relational_victim",
            "peq_ss_reputation_aggs",
            "peq_ss_reputation_victim",
            "peq_ss_overt_aggression",
            "peq_ss_overt_victim",
            "peq_ss_relational_aggs",
        ]
        within_categories = [var for var in within_categories if var not in variables_to_remove]

    # Avoid mutating the group parameter.
    _resolved_group = "low_ale_children_p" if group == "low_ale" else group

    if _resolved_group:
        if _resolved_group == "high_ale":
            all_vars = within_categories + [f"{_resolved_group}"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data[all_data["high_ale"] == True]

        elif _resolved_group == "high_ale_severe_p_mh":
            all_vars = within_categories + [f"{_resolved_group}"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data[all_data["high_ale_severe_p_mh"] == True]

        elif _resolved_group == "low_ale_children_p":
            all_vars = within_categories + ["low_ale_children_p"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data[all_data["low_ale_children_p"] == True]

        elif _resolved_group == "Female":
            all_vars = within_categories + ["sex"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data.loc[:, ~all_data.columns.duplicated()]
            all_data = all_data[all_data["sex"] == 2]

        elif _resolved_group == "Male":
            all_vars = within_categories + ["sex"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data.loc[:, ~all_data.columns.duplicated()]
            all_data = all_data[all_data["sex"] == 1]

        else:
            all_vars = within_categories
            all_data = ctx.sample.query(f"time == {t}")[all_vars]

    else:
        all_vars = within_categories
        all_data = ctx.sample.query(f"time == {t}")[all_vars]

    if verbose:
        logger.info(
            f"Gathered within category data from {all_data.shape[0]} subjects for {_resolved_group if _resolved_group else 'full'} group at t = {t}"
        )

    if (
        "depress_D_p" not in targets
        and "depress_D_p_rev" not in targets
        and "external_D_p" not in targets
        and "adhd_D_p" not in targets
    ):
        if "latent_class_depression" in targets:
            all_data = all_data[(all_data["latent_class_depression"] != "")]
        # Class rebalancing deferred to AFTER split
        _needs_rebalance = class_rebalancing or (
            ctx.target_options == "latent_class_depression"
            or ctx.target_options == "depadhd_c"
            or ctx.target_options == "asdadhd_c"
        )

    available = drop_high_missing_columns(all_data, threshold=0.7, verbose=verbose)
    dropped_columns = set(all_data.columns).difference(set(available.columns))

    if dropped_columns:
        logger.info(f"WARNING: Dropped columns at t = {t} due to entirely missing data: {dropped_columns}")

    raw_X = available.drop(targets, axis=1)
    raw_y = available[targets]

    # Drop rows where target is NaN (parent ASR items can have sporadic missingness)
    _target_notna = raw_y.iloc[:, 0].notna()
    if (~_target_notna).any():
        _n_dropped = (~_target_notna).sum()
        logger.info(f"Dropped {_n_dropped} rows with missing target values")
        raw_X = raw_X[_target_notna]
        raw_y = raw_y[_target_notna]

    # One-hot encoding is applied to the full dataset before folding because it
    # uses only category membership (deterministic; no data-dependent statistics).
    # Imputation and scaling are deferred to train_and_evaluate_model_kfold().
    one_hot_features = [
        "sex",
        "sex_P",
        "L_site_id",
        "marital_status",
        "child_religion",
        "sex_orient_y",
        "trans_id_y",
        "crysflu_c",
        "anxadhd_c",
    ]
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

    for feature in one_hot_features:
        if feature in raw_X.columns:
            logger.info(f"Encoding {feature} for one hot encoding..")
            raw_X_cat = raw_X[[feature]].astype(str)
            encoded_data = encoder.fit_transform(raw_X_cat)
            unique_values = encoder.categories_[0][1:]
            encoded_columns = [f"{feature}_{val}" for val in unique_values]
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=raw_X.index)
            raw_X = pd.concat([raw_X.drop(feature, axis=1), encoded_df], axis=1)
            logger.info(f"Encoded {feature} into {len(encoded_columns)} categories")

    ret = {}

    if is_regression(targets[0]):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Iterate over the circular-exclusion-filtered category dict (active_vfet_tp)
    # so that excluded categories never appear in the returned results.
    for category, cat_vars in active_vfet_tp.items():
        # Also filter individual across_excl variables from each category slice.
        filtered_vars = [v for v in cat_vars if v not in across_excl]
        available_vars = list(set(filtered_vars).difference(dropped_columns))
        available_vars = [var for var in available_vars if var in raw_X.columns]

        if available_vars:
            ret[category] = (raw_X[available_vars], raw_y, kf, scale_x, scale_y)
        else:
            logger.info(f"WARNING: No available variables left for category: {category}")

    return ret


# Across-category variant: returns (X_raw, y_raw, kf, scale_x, scale_y)
def across_categories_data_kfold(
    t, targets, n_splits=5, group=None, scale_x=False, scale_y=False, verbose=False, class_rebalancing=False
):
    available_columns = set(ctx.sample.columns)

    if t == 0:
        across_categories = list(ctx.t0_variables)
    elif t == 1:
        across_categories = list(ctx.t1_variables)
    elif t == 2:
        across_categories = list(ctx.t2_variables)
    elif t == 3:
        across_categories = list(ctx.t3_variables)
    elif t == 4:
        across_categories = list(ctx.t4_variables)
    else:
        raise ValueError(f"Invalid time point: {t}")

    # Apply circular exclusions
    target_name = targets[0] if targets else ""
    across_excl, _ = get_circular_exclusions(target_name, timepoint=t)
    if across_excl:
        before_count = len(across_categories)
        across_categories = [v for v in across_categories if v not in across_excl]
        removed_count = before_count - len(across_categories)
        if removed_count > 0:
            logger.info(
                f"[CIRCULAR] Removed {removed_count} circular variables from across-category pool for target: {target_name}"
            )

    # Remove columns from across_categories that are not in available_columns
    across_categories = [var for var in across_categories if var in available_columns]

    # If any columns were removed, notify the user
    dropped_columns = set([x for xs in ctx.variables_for_each_time_point[t].values() for x in xs] + targets).difference(
        available_columns
    )

    if dropped_columns:
        logger.info(
            f"WARNING: The following columns were not found in the dataset and have been dropped: {dropped_columns}"
        )

    # Remove specific variables when t == 1 or t == 3
    if int(t) == 3 or int(t) == 1:
        variables_to_remove = [
            "peq_ss_relational_victim",
            "peq_ss_reputation_aggs",
            "peq_ss_reputation_victim",
            "peq_ss_overt_aggression",
            "peq_ss_overt_victim",
            "peq_ss_relational_aggs",
        ]
        across_categories = [var for var in across_categories if var not in variables_to_remove]

    # Avoid mutating the group parameter.
    _resolved_group = "low_ale_children_p" if group == "low_ale" else group

    if _resolved_group:
        if _resolved_group == "high_ale":
            all_vars = across_categories + targets + [f"{_resolved_group}"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data[all_data["high_ale"] == True]

        elif _resolved_group == "high_ale_severe_p_mh":
            all_vars = across_categories + targets + [f"{_resolved_group}"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data[all_data["high_ale_severe_p_mh"] == True]

        elif _resolved_group == "low_ale_children_p":
            all_vars = across_categories + targets + ["low_ale_children_p"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data[all_data["low_ale_children_p"] == True]

        elif _resolved_group == "Female":
            all_vars = across_categories + targets + ["sex"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data.loc[:, ~all_data.columns.duplicated()]
            all_data = all_data[all_data["sex"] == 2]

        elif _resolved_group == "Male":
            all_vars = across_categories + targets + ["sex"]
            all_data = ctx.sample[ctx.sample["time"] == t][all_vars]
            all_data = all_data.loc[:, ~all_data.columns.duplicated()]
            all_data = all_data[all_data["sex"] == 1]

        else:
            all_vars = across_categories + targets
            all_data = ctx.sample.query(f"time == {t}")[all_vars]

    else:
        all_vars = across_categories + targets
        all_data = ctx.sample.query(f"time == {t}")[all_vars]

    if verbose:
        logger.info(
            f"Gathered across category data from {all_data.shape[0]} subjects for {_resolved_group if _resolved_group else 'full'} group at t = {t}"
        )

    if (
        "depress_D_p" not in targets
        and "depress_D_p_rev" not in targets
        and "external_D_p" not in targets
        and "adhd_D_p" not in targets
    ):
        if "latent_class_depression" in targets:
            all_data = all_data[(all_data["latent_class_depression"] != "")]
        # Class rebalancing deferred to AFTER split
        _needs_rebalance = class_rebalancing or (
            ctx.target_options == "latent_class_depression"
            or ctx.target_options == "depadhd_c"
            or ctx.target_options == "asdadhd_c"
        )

    available = drop_high_missing_columns(all_data, threshold=0.7, verbose=verbose)
    dropped_columns = set(all_data.columns).difference(set(available.columns))

    if dropped_columns:
        logger.info(f"WARNING: Dropped columns at t = {t} due to entirely missing data: {dropped_columns}")

    raw_X = available.drop(targets, axis=1)
    raw_y = available[targets]

    # Drop rows where target is NaN (parent ASR items can have sporadic missingness)
    _target_notna = raw_y.iloc[:, 0].notna()
    if (~_target_notna).any():
        _n_dropped = (~_target_notna).sum()
        logger.info(f"Dropped {_n_dropped} rows with missing target values")
        raw_X = raw_X[_target_notna]
        raw_y = raw_y[_target_notna]

    # Same encoding strategy as the within-category kfold function above.
    one_hot_features = [
        "sex",
        "sex_P",
        "L_site_id",
        "marital_status",
        "child_religion",
        "sex_orient_y",
        "trans_id_y",
        "crysflu_c",
        "anxadhd_c",
    ]
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

    for feature in one_hot_features:
        if feature in raw_X.columns:
            logger.info(f"Encoding {feature} for one hot encoding..")
            raw_X_cat = raw_X[[feature]].astype(str)
            encoded_data = encoder.fit_transform(raw_X_cat)
            unique_values = encoder.categories_[0][1:]
            encoded_columns = [f"{feature}_{val}" for val in unique_values]
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=raw_X.index)
            raw_X = pd.concat([raw_X.drop(feature, axis=1), encoded_df], axis=1)
            logger.info(f"Encoded {feature} into {len(encoded_columns)} categories")

    # Instead of train_test_split, we'll return the full dataset and a KFold object
    if is_regression(targets[0]):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    return raw_X, raw_y, kf, scale_x, scale_y


def train_and_evaluate_model_kfold(X, y, kf, model, scale_x=False, scale_y=False):
    """Train and evaluate a model under K-fold cross-validation with per-fold
    preprocessing to ensure no leakage from held-out folds.

    Imputation and optional standardization are fit on the training fold only
    and applied to the corresponding test fold within every iteration.

    Parameters
    X, y      : Feature matrix and target returned by a kfold data function.
    kf        : Pre-configured KFold or StratifiedKFold splitter.
    model     : Unfitted estimator (will be cloned each fold).
    scale_x   : Standardize features if True.
    scale_y   : Standardize target if True (regression only).

    Returns
    (mean_score, std_score) : Mean and standard deviation of fold-wise scores.
    """
    from bpsp.analysis.runner import imputer

    scores = []
    for train_index, test_index in kf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train_fold, y_test_fold = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        # Impute per fold (fit on train only)
        fold_imputer = get_imputer(imputer)
        fold_imputer.fit(X_train_fold)
        X_train_fold = pd.DataFrame(
            fold_imputer.transform(X_train_fold), columns=X_train_fold.columns, index=X_train_fold.index
        )
        X_test_fold = pd.DataFrame(
            fold_imputer.transform(X_test_fold), columns=X_test_fold.columns, index=X_test_fold.index
        )

        # Scale per fold if requested (fit on train only)
        if scale_x:
            fold_scaler_x = StandardScaler()
            fold_scaler_x.fit(X_train_fold)
            X_train_fold = pd.DataFrame(
                fold_scaler_x.transform(X_train_fold), columns=X_train_fold.columns, index=X_train_fold.index
            )
            X_test_fold = pd.DataFrame(
                fold_scaler_x.transform(X_test_fold), columns=X_test_fold.columns, index=X_test_fold.index
            )

        if scale_y:
            fold_scaler_y = StandardScaler()
            fold_scaler_y.fit(y_train_fold)
            y_train_fold = pd.DataFrame(
                fold_scaler_y.transform(y_train_fold), columns=y_train_fold.columns, index=y_train_fold.index
            )
            y_test_fold = pd.DataFrame(
                fold_scaler_y.transform(y_test_fold), columns=y_test_fold.columns, index=y_test_fold.index
            )

        model_clone = clone(model)
        model_clone.fit(X_train_fold, y_train_fold)
        score = model_clone.score(X_test_fold, y_test_fold)
        scores.append(score)

    return np.mean(scores), np.std(scores)


VAL_RANDOM_STATE = 42  # Master seed
VAL_DPI = 300  # Figure DPI for Nature
VAL_FIGSIZE_SINGLE = (3.50, 3.50)  # Nature single-column (89 mm)
VAL_FIGSIZE_DOUBLE = (7.20, 4.50)  # Nature double-column (183 mm)


# Lightweight validation model factory
def create_lightweight_model(is_classification=True, n_features=None):
    """Return a quick-training model suitable for validation sweeps.

    Uses CatBoost with moderate iterations (300) and early stopping.
    Intended for PCA baseline, temporal holdout, ablation, and TP sweeps
    where full stacking ensemble is too expensive.
    """
    try:
        if is_classification:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.08,
                early_stopping_rounds=30,
                verbose=0,
                random_state=42,
                thread_count=-1,
                auto_class_weights="Balanced",
            )
        else:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.08,
                early_stopping_rounds=30,
                verbose=0,
                random_state=42,
                thread_count=-1,
            )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        if is_classification:
            return GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
        else:
            return GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)


# Helper: bootstrap CI
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=None, ci=0.95, seed=VAL_RANDOM_STATE):
    """Compute bootstrap confidence interval for a metric.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    if n_boot is None:
        n_boot = ctx.VAL_N_BOOTSTRAP
    rng = np.random.RandomState(seed)
    point = metric_fn(y_true, y_pred)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    scores = np.array(scores)
    alpha = 1 - ci
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return point, lo, hi


logger.info("Shared Validation Utilities loaded (create_lightweight_model, bootstrap_ci)")

# Variable Display Names

# Translates raw ABCD variable names into readable labels for feature
# importance figures. The rename_dict is consumed by FI plotting functions
# in this module.
# DISPLAY_NAMES: T2 Within-Category Variable Renaming Guide
# Conventions:
# (Y) at end  = youth self-report: ONLY when raw name ends in _k
# (_p) parent ratings of children
# (_parent)  = parent self-report (mostly from ASR)

DISPLAY_NAMES = {
    # ADHD (8)
    "trouble_concentrating_k": "Concentration Trouble (Y)",
    "difficulty_finishing_tasks_k": "Task Continuity  (Y)",
    "easily_distracted_k": "Distracted Easily (Y)",
    "cant_concentrate_p": "Concentration Difficulty",
    "doesnt_finish_p": "Failure to Finish Tasks",
    "hyperactive_p": "Hyperactive",
    "impulsive_p": "Impulsive",
    "easily_distracted_p": "Distracted Easily",
    # Adverse Life Events (24)
    "g_lifeevents_ss_k": "Positive Events (Y)",
    "b_lifeevents_ss_k": "Adverse Life Events (Y)",
    "b_lifeevents_affected_ss_k": "Adverse Events Impact (Y)",
    "experienced_crime_p": "Crime Victim",
    "g_lifeevents_ss_p": "Positive Events",
    "b_lifeevents_ss_p": "Adverse Life Events",
    "b_lifeevents_affected_ss_p": "Adverse Events Impact",
    "car_accident_hurt_p": "Car Accident",
    "big_accident_need_treatment_p": "Major Accident",
    "fire_victim_p": "Fire Victim",
    "natural_disaster_victim_p": "Nat. Disaster",
    "terrorism_victim_p": "Terrorism Exp.",
    "war_death_witness_p": "War Witness",
    "stabbing_shooting_witness_p": "Violence Witness",
    "stabbing_shooting_victim_community_p": "Community Violence",
    "stabbing_shooting_victim_home_p": "Home Violence",
    "beating_victim_home_p": "Beaten at Home",
    "stranger_threatened_child_victim_p": "Stranger Threat",
    "family_threatened_child_victim_p": "Family Threat",
    "adult_family_fighting_victim_p": "Adult Fighting",
    "domestic_child_sexually_abuse_victim_p": "Dom. Sex Abuse",
    "foreign_child_sexually_abuse_victim_p": "Non-Dom. Sex Abuse",
    "peer_child_sexually_abuse_victim_p": "Peer Sex Abuse",
    "sudden_death_in_family_p": "Sudden Death",
    # Anxiety (12)
    "social_fear_present_PK": "Social Fear",
    "worries_p": "Worries",
    "clings_to_adults_p": "Clingy",
    "nervous_general_p": "Generally Nervous",
    "nervous_twitching_p": "Nervous Twitching",
    "fears_excl_school_p": "Fears Exclusion",
    "fears_school_p": "School Fear",
    "fears_being_bad_p": "Guilt Fear",
    "paranoid_p": "Paranoid",
    "fear_becoming_obese_present_PK": "Weight Fear",
    "self_conscious_k": "Self-Conscious (Y)",
    "anxious_fearful_k": "Anxious (Y)",
    # Cognitive Task Outcomes (44)
    "tb_picvocab": "Picture Vocabulary Task (NIH)",
    "tb_picture": "Picture Sequence Memory Task (NIH)",
    "tb_reading": "Oral Reading Recognition Task (NIH)",
    "tb_flanker": "Flanker Inhibitory Control Task (NIH)",
    "tb_list": "List Sorting Working Memory Task (NIH)",
    "tb_cardsort": "Dimensional Change Card Sort Task (NIH)",
    "tb_pattern": "Pattern Comparison Speed Task (NIH)",
    "gd_safebets": "Safe Bets (GDT)",
    "gd_riskybets": "Game of Dice Task -- Risky Bets",
    "ravlt_s_total": "RAVLT Short Tot.",
    "ravlt_s_repitition": "RAVLT Short Rep.",
    "ravlt_s_intrusions": "RAVLT Short Intr.",
    "ravlt_l_total": "RAVLT Long Tot.",
    "ravlt_l_repitition": "RAVLT Long Rep.",
    "ravlt_l_intrusions": "RAVLT Long Intr.",
    "nb_correct_nt": "N-Back Acc.",
    "nb_correct_mrt": "N-Back RT",
    "nb_correct_nt_2back": "2-Back Acc.",
    "nb_correct_mrt_2back": "2-Back RT",
    "nb_correct_nt_pos": "N-Back Acc. Pos",
    "nb_correct_mrt_pos": "N-Back RT Pos",
    "nb_correct_nt_neg": "N-Back Acc. Neg",
    "nb_correct_mrt_neg": "N-Back RT Neg",
    "nb2_accuracy_pos": "2B Acc. Pos",
    "nb2_resp_bias_pos": "2B Bias Pos",
    "nb2_D_prime_pos": "2B d' Pos",
    "nb2_accuracy_neg": "2B Acc. Neg",
    "nb2_resp_bias_neg": "2B Bias Neg",
    "nb2_D_prime_neg": "2B d' Neg",
    "sst_ssrt_mean_est": "SSRT Mean",
    "sst_ssrt_int_est": "SSRT Integration",
    "sst_acceptable_performance": "SST Task Valid",
    "mid_mrt_smrw": "MID RT Sm. Rew.",
    "mid_mrt_lgrw": "MID RT Lg. Rew.",
    "mid_total_payout": "MID Earnings",
    "mid_acceptable_performance": "MID Valid",
    "mid_num_trials": "MID Trials",
    "lmt_accuracy": "LMT Accuracy",
    "lmt_correct_nt": "LMT Correct N",
    "lmt_mrt": "LMT Mean RT",
    "lmt_correct_mrt": "LMT Correct RT",
    "lmt_efficiency": "LMT Efficiency",
    "nb_correct_mrt_neutral": "N-Back RT Neut",
    "nb_correct_nt_neutral": "N-Back Acc. Neut",
    # Delta (13)
    "delta_adhd_D_p": "Change in ADHD",
    "delta_not_liked_p": "Change in Unliked",
    "delta_doesnt_get_along_p": "Change in Peer Conflict",
    "delta_family_conflict_ss_p": "Change in Fam. Conflict",
    "delta_family_conflict_ss_k": "Change in Fam. Conflict (Y)",
    "delta_bad_grades": "Change in Bad Grades",
    "delta_social_problems_D_p": "Change in Social Problems",
    "delta_somatic_problems_D_p": "Change in Somatic Issues",
    "delta_bad_diet_p": "Change in Bad Diet",
    "delta_atschool_total_problems_ss_t": "Change in School Prob.",
    "delta_b_lifeevents_ss_p": "Change in Bad Events",
    "delta_anxdisord_D_p": "Change in Anxiety",
    "delta_cant_concentrate_p": "Change in Concentration Difficulty",
    # Diet / Nutrition (17)
    "fruit_intake": "Fruit Intake",
    "vegetable_intake": "Vegetable Intake",
    "protein_sources_intake": "Protein Sources",
    "legume_intake": "Legume Intake",
    "added_sugar": "Added Sugar",
    "sugary_beverage_freq": "Sugary Drinks",
    "dairy_intake": "Dairy Intake",
    "whole_grain_intake": "Whole Grains",
    "total_calories": "Total Calories",
    "protein_intake": "Protein (g)",
    "carbohydrate_intake": "Carbs (g)",
    "fiber_intake": "Fiber (g)",
    "sodium_intake": "Sodium (mg)",
    "potassium_intake": "Potassium (mg)",
    "total_sugar": "Total Sugar",
    "saturated_fat": "Saturated Fat",
    "bad_diet_p": "Bad Diet",
    # Dynamic Cognitive Control Parameters (55)
    # Computational Modeling of SST Task
    # from Percy Mistry (see https://www.biorxiv.org/content/10.1101/2024.10.01.615613v1.full for overview)
    # ALL prefixed with SST
    "sst_theta0": "SST Task Theta0",
    "sst_theta1": "SST Task Theta1",
    "sst_mu": "SST Task Mu",
    "sst_aS": "SST Task AlphaS",
    "sst_dS": "SST Task DeltaS",
    "sst_gamma0": "SST Task Gamma0",
    "sst_gamma1": "SST Task Gamma1",
    "sst_dG": "SST Task DeltaG",
    "sst_aG1": "SST Task AlphaG1",
    "sst_aG2": "SST Task AlphaG2",
    "sst_kappa0": "SST Task Kappa0",
    "sst_xM": "SST Task xM",
    "sst_sM": "SST Task sM",
    "sst_bG": "SST Task BetaG",
    "sst_pp": "SST Task pp",
    "sst_factor1": "SST Task Factor 1",
    "sst_factor2": "SST Task Factor 2",
    "sst_factor3": "SST Task Factor 3",
    "sst_mean_ssrt": "SST Task Mean SSRT",
    "sst_median_ssrt": "SST Task Median SSRT",
    "sst_mean_PDR": "SST Task Mean PDR",
    "sst_median_PDR": "SST Task Median PDR",
    "sst_mean_SD": "SST Task Mean SD",
    "sst_median_SD": "SST Task Median SD",
    "sst_mean_SDr": "SST Task Mean SDr",
    "sst_median_SDr": "SST Task Median SDr",
    "sst_mean_PDRg": "SST Task Mean PDRg",
    "sst_median_PDRg": "SST Task Median PDRg",
    "sst_mean_betaS": "SST Task Mean BetaS",
    "sst_median_betaS": "SST Task Median BetaS",
    "sst_mean_bS2": "SST Task Mean BetaS2",
    "sst_median_bS2": "SST Task Median BetaS2",
    "sst_mean_absdelta": "SST Task Mean AbsDelta",
    "sst_median_absdelta": "SST Task Med. AbsDelta",
    "sst_pdrV": "SST Task PDR Var.",
    "sst_pdrSD": "SST Task PDR SD",
    "sst_pdrCV": "SST Task PDR CV",
    "sst_sdRV": "SST Task SD Var.",
    "sst_sdSD": "SST Task SD StdDev",
    "sst_sdCV": "SST Task SD CV",
    "sst_sdrRV": "SST Task SDr Var.",
    "sst_sdrSD": "SST Task SDr SD",
    "sst_sdrCV": "SST Task SDr CV",
    "sst_pdrgV": "SST Task PDRg Var.",
    "sst_pdrgSD": "SST Task PDRg SD",
    "sst_pdrgCV": "SST Task PDRg CV",
    "sst_betasV": "SST Task BetaS Var.",
    "sst_betasSD": "SST Task BetaS SD",
    "sst_betasCV": "SST Task BetaS CV",
    "sst_absdeltaRV": "SST Task AbsDelta Var.",
    "sst_absdeltaSD": "SST Task AbsDelta SD",
    "sst_absdeltaCV": "SST Task AbsDelta CV",
    "sst_bs2V": "SST Task BetaS2 Var.",
    "sst_bs2SD": "SST Task BetaS2 SD",
    "sst_bs2CV": "SST Task BetaS2 CV",
    # Ethnicity / Nationality (43)
    "desc_african_AFR_B": "African Desc.",
    "desc_native_american_AMR_B": "Native Amer.",
    "desc_alaska_native_AMR_B": "Alaska Native",
    "desc_chinese_EAS_B": "Chinese Desc.",
    "desc_japanese_EAS_B": "Japanese Desc.",
    "desc_korean_EAS_B": "Korean Desc.",
    "desc_vietnamese_EAS_B": "Vietnamese Desc.",
    "desc_european_EUR_B": "European Desc.",
    "desc_asian_indian_SAS_B": "South Asian Desc.",
    "desc_other_south_asian_SAS_B": "Oth. S. Asian",
    "desc_latin_B": "Latino Desc.",
    **{f"pc_gene_aces{i}": f"Ancestry PC{i}" for i in range(1, 33)},
    # Externalizing / Other (23)
    "argues_p": "Argumentative",
    "stubborn_p": "Stubborn",
    "temper_tantrums_p": "Tantrums",
    "bullies_others_p": "Bullies Others",
    "destroys_own_things_p": "Destroys Own Stuff",
    "destroys_others_things_p": "Destroys Others Stuff",
    "disobedient_home_p": "Disobedient Home",
    "disobedient_school_p": "Disobedient School",
    "breaks_rules_p": "Rule-Breaking",
    "fights_p": "Fights",
    "lying_p": "Dishonest/Deceptive",
    "steals_home_p": "Steals Home",
    "steals_outside_p": "Steals Outside",
    "threatens_others_p": "Threatens",
    "whines_p": "Whines",
    "demands_attention_p": "Demands Attention",
    "conduct_physical_fights_present_PK": "Phys. Fights",
    "acts_immature_k": "Immature (Y)",
    "destroys_others_things_k": "Destroys Others (Y)",
    "disobeys_parents_k": "Par. Disobeys (Y)",
    "stubborn_k": "Stubborn (Y)",
    "hot_temper_k": "Hot Temper (Y)",
    # Family Drug Use (36)
    "hallucinogen_use_history_B_p": "Hallucinogen Hx",
    "hallucinogen_current_B_p": "Hallucinogen Cur.",
    "sedative_hypnotic_anxiolytic_use_B_p": "Sedative Use",
    "father_alcohol": "Father Alcohol",
    "mother_alcohol": "Mother Alcohol",
    "father_druguse": "Father Drugs",
    "mother_druguse": "Mother Drugs",
    "cigs_during_pregnancy_p": "Prenatal Cigs",
    "alcohol_during_pregnancy_p": "Prenatal Use Alcohol",
    "weed_during_pregnancy_p": "Prenatal Use Cannabis",
    "cocaine_during_pregnancy_p": "Prenatal Use Cocaine",
    "heroin_during_pregnancy_p": "Prenatal Use Heroin",
    "prescriptionmed_pregnancy_p": "Prenatal Use Rx",
    "cigs_before_pregnancy_p": "Pre-Preg. Cigs",
    "alcohol_before_pregnancy_p": "Pre-Preg. Alcohol",
    "weed_before_pregnancy_p": "Pre-Preg. Cannabis",
    "cocaine_before_pregnancy_p": "Pre-Preg. Cocaine",
    "heroin_before_pregnancy_p": "Pre-Preg. Heroin",
    "drugs_before_pregnancy_p": "Pre-Preg. Drugs",
    "drinksperweek_during_pregnancy_p": "Prenatal Use Drinks/Wk",
    "drugs_during_pregnancy_p": "Prenatal Use Drugs",
    "caffeine_during_pregnancy_p": "Prenatal Use Caffeine",
    "parent_tobacco_use_frequency_p": "Par. Tobacco Freq.",
    "parent_drug_use_p": "Par. Drug Use",
    "parent_drinks_too_much_p": "Par. Excess Drink",
    "parent_drinks_frequency_p": "Par. Drink Freq.",
    "parent_drunk_days_p": "Par. Drunk Days",
    "parent_drug_days_nonmedical_p": "Par. Rec. Drug Days",
    "parent_alcohol_days_drunk_p": "Par. Intox. Days",
    "parent_alcohol_excess_p": "Par. Binge Drink",
    "parent_alcohol_freq_p": "Par. Alcohol Freq.",
    "parent_drug_days_p": "Par. Drug Days",
    "parent_tobacco_use_p": "Par. Tobacco Use",
    "druguse_alcohol_p": "Child Alcohol",
    "druguse_other_p": "Child Oth. Drug",
    "druguse_tobacco": "Child Tobacco",
    # Family Dynamics & Parenting (38)
    # ALL prefixed with Fam.
    "p_comm_cohesion_ss": "Fam. Comm. Cohesion",
    "p_comm_ctrl_ss": "Fam. Comm. Control",
    "p_comm_collective_efficacy_ss": "Fam. Collect. Efficacy",
    "fam_fight_often_k": "Fam. Fight Often (Y)",
    "fam_no_open_anger_k": "Fam. No Open Anger (Y)",
    "fam_throw_things_k": "Fam. Throw Things (Y)",
    "fam_no_lose_temps_k": "Fam. Keep Temper (Y)",
    "fam_criticize_often_k": "Fam. Criticize (Y)",
    "fam_hit_each_other_k": "Fam. Hitting (Y)",
    "fam_keep_peace_k": "Fam. Keep Peace (Y)",
    "fam_try_one_up_k": "Fam. One-Upmanship (Y)",
    "fam_no_raise_voices_k": "Fam. No Yelling (Y)",
    "family_not_talk_aboutfeelings_p": "Fam. No Feeling Talk",
    "family_peaceful_p": "Fam. Peaceful",
    "family_open_discussing_anything_p": "Fam. Open Discussion",
    "family_lose_temper_rare_p": "Fam. Rare Temper",
    "family_believe_not_raise_voice_p": "Fam. No Yelling",
    "frequent_family_conflict_p": "Fam. Freq. Conflict",
    "family_conflict_ss_p": "Fam. Conflict SS",
    "family_expression_ss_p": "Fam. Expression SS",
    "family_intellectual_ss_p": "Fam. Intellectual SS",
    "family_activities_ss_p": "Fam. Activities SS",
    "family_organisation_ss_p": "Fam. Organization SS",
    "parents_argue_more_p": "Fam. Par. Argue",
    "family_emotionprob_p": "Fam. Emot. Prob.",
    "parents_divorced_p": "Fam. Par. Divorced",
    "death_in_family_p": "Fam. Death",
    "family_move_p": "Fam. Moved",
    "family_conflict_ss_k": "Fam. Conflict SS (Y)",
    "parent_monitoring_ss_k": "Fam. Par. Monitoring (Y)",
    "marital_status": "Fam. Marital Status",
    "parent_age": "Fam. Par. Age",
    "sex_P_2": "Female (Par. Sex)",
    "sex_P_3": "Intersex-Male (Par. Sex)",
    "sex_P_4": "Intersex-Female (Par. Sex)",
    "num_brothers_p": "Fam. Num. Brothers",
    "num_sisters_p": "Fam. Num. Sisters",
    "religious_service_frequency": "Fam. Service Freq.",
    "relig_importance": "Fam. Relig. Import.",
    "parent_family_responsibilities_p": "Fam. Par. Duties",
    # Family Psychopathology & Well-Being (30)
    "parent_internal_D_p": "Par. Internalizing",
    "parent_anhedonia_B_p": "Par. Anhedonia",
    "obsessions_present_B_p": "Par. Obsessions",
    "poor_eye_contact_B_p": "Par. Eye Contact",
    "nightmares_B_p": "Par. Nightmares",
    "parent_elevated_mood_B_p": "Par. Elev. Mood",
    "parent_excessive_worry_B_p": "Par. Chronic Worry",
    "parent_lying_B_p": "Par. Lying",
    "parent_social_anxiety_disorder_B_p": "Par. Social Anx.",
    "parent_sleep_problem_B_p": "Par. Sleep Prob.",
    "parent_bulimia_B_p": "Par. Bulimia",
    "parent_attention_D_p": "Par. Attention Dysregulation",
    "parent_aggressive_D_p": "Par. Aggressive/Impulsive",
    "parent_external_D_p": "Par. Externalizing",
    "parent_anxdisord_D_p": "Par. Anx. Disorder",
    "parent_antisocial_D_p": "Par. Antisocial",
    "parent_hyperactive_D_p": "Par. Hyperactivity",
    "parent_somatic_problems_D_p": "Par. Somatic Issues",
    "parent_intrusive_thoughts_D_p": "Par. Intrusive Thoughts",
    "parent_avoidant_person_D_p": "Par. Avoidant",
    "parent_personal_strength_D_p": "Par. Strengths",
    "d_grandfather_dep": "Pat. Grandpa Dep.",
    "d_grandmother_dep": "Pat. Grandma Dep.",
    "m_grandfather_dep": "Mat. Grandpa Dep.",
    "m_grandmother_dep": "Mat. Grandma Dep.",
    "father_mania": "Father Mania",
    "mother_mania": "Mother Mania",
    "father_trouble": "Father Trouble",
    "parent_hospitalized_emo": "Par. Hospitalized",
    "parent_therapy_emo": "Par. Therapy",
    "suicidal_p": "Suicidal Ideation",
    "parent_adhd_D_p": "Par. ADHD",
    "sex_orient_y_3.0": "Sex Orientation (3.0)",
    # Inter (35)
    "sex_2": "Female (Child Sex)",
    "sex_3": "Intersex-Male (Child Sex)",
    "sex_4": "Intersex-Female (Child Sex)",
    "parent_income": "Par. Income",
    "parent_income_new": "Par. Income Adj.",
    "L_site_id": "ABCD Site",
    "weight": "Weight",
    "social_problems_D_p": "Social Problems",
    "delta_not_liked_p": "Change in Unliked",
    "parent_age_grouped": "Par. Age Group",
    "child_white": "White",
    "feels_leftout_k": "Left Out (Y)",
    "sex_orient_y": "Sex Orientation",
    "trans_id_y": "Trans Identity",
    "screentime_daysperweek_k": "Screen Days/Wk (Y)",
    "screentime_weekday_ss_k": "Weekday Screen (Y)",
    "puberty_k": "Puberty Stage (Y)",
    "close_boy_friends_k": "Close Boy Friends (Y)",
    "close_girl_friends_k": "Close Girl Friends (Y)",
    "cct": "Cash Choice",
    "bad_grades": "Bad Grades",
    "tb_cryst": "Crystallized Composite (NIH)",
    "crysflu_c": "Cryst./Fluid Comp.",
    "child_religion": "Child Religion",
    # Medical / Somatic Problems (15)
    "medhx_p": "Med. History",
    "medhx_doctorvisit_p": "Doctor Visits",
    "medhx_emergencyroom_p": "ER Visits",
    "pain_last_month_k": "Recent Pain (Y)",
    "seriously_sick_lastyear_k": "Sick Last Year (Y)",
    "body_aches_p": "Body Aches",
    "frequent_headaches_p": "Headaches",
    "nausea_p": "Nausea",
    "eye_problems_p": "Eye Problems",
    "skin_problems_p": "Skin Problems",
    "frequent_stomachaches_p": "Stomachaches",
    "vomiting_p": "Vomiting",
    "constipated_p": "Constipation",
    "bad_toilet_habits_p": "Encopresis",
    "wets_bed_p": "Enuresis",
    # Other Personality Features (26)
    "easily_offended_p": "Easily Offended",
    "blames_others_p": "Blames Others",
    "sociable_p": "Sociable",
    "school_excitement_p": "School Excite.",
    "not_critical_others_p": "Non-Critical",
    "scared_dark_p": "Scared Dark",
    "disagreeable_p": "Disagreeable",
    "goal_continuity_p": "Goal Persistence",
    "up_negative_urgency_ss_k": "Neg. Urgency (Y)",
    "up_lackofplanning_ss_k": "Lack Planning (Y)",
    "up_sensationseeking_ss_k": "Sensation Seek. (Y)",
    "up_positiveurgency_ss_k": "Pos. Urgency (Y)",
    "up_lackperseverance_ss_k": "Lack Persever. (Y)",
    "bis_behav_inhibition_ss_k": "Behav. Inhibit. (Y)",
    "bis_reward_responsive_ss_k": "Reward Resp. (Y)",
    "bis_drive_ss_k": "Drive (Y)",
    "bis_funseeking_ss_k": "Fun Seeking (Y)",
    "loquacious_p": "Loquacious",
    "bragadocious_p": "Bragging",
    "easily_jealous_p": "Easily Jealous",
    "wishes_other_sex_p": "Other Gender Wish",
    "easily_embarrassed_p": "Easily Embarrassed",
    "secretive_p": "Secretive",
    "perfectionist_p": "Perfectionist",
    "strange_ideas_p": "Strange Ideas",
    # Parent Anxiety (7)
    "parent_fearful_or_anxious_p": "Par. Fearful",
    "parent_specific_fears_p": "Par. Phobias",
    "parent_fear_of_bad_thoughts_p": "Par. Bad Thoughts",
    "parent_worries_about_future_p": "Par. Future Worry",
    "parent_worries_about_family_p": "Par. Family Specific Worry",
    "parent_worries_a_lot_p": "Par. Worries",
    "parent_relationship_concerns_p": "Par. Relat. Specific Worry",
    # Parent Cognitive and Attention Issues (15)
    "parent_forgetful_p": "Par. Forgetful",
    "parent_concentration_trouble_p": "Par. Concentration Difficulty",
    "parent_confused_p": "Par. Confused",
    "parent_planning_trouble_p": "Par. Planing Trouble",
    "parent_not_good_at_details_p": "Par. Struggles with Details",
    "parent_obsessive_thoughts_p": "Par. Obsessive Thinking",
    "parent_repeats_acts_p": "Par. Compulsions",
    "parent_max_effort_p": "Par. Low Effort",
    "parent_disorganized_p": "Par. Disorganized",
    "parent_loses_things_p": "Par. Loses Things",
    "parent_decision_trouble_p": "Par. Indecisive",
    "parent_priority_trouble_p": "Par. Prioritization Difficulty",
    "parent_sees_things_p": "Par. Vis. Halluc.",
    "parent_hears_voices_p": "Par. Aud. Halluc.",
    "parent_speech_problems_p": "Par. Speech Prob.",
    # Parent Delta Psychopathology (24)
    "delta_parent_sleep_trouble_p": "Change in Par. Sleep",
    "delta_parent_worries_about_family_p": "Change in Par. Fam. Worry",
    "delta_parent_friendship_trouble_p": "Change in Par. Friend Prob.",
    "delta_parent_poor_work_performance_p": "Change in Par. Work Performance",
    "delta_parent_aches_pains_p": "Change in Par. Aches",
    "delta_parent_not_liked_by_others_p": "Change in Par. Unliked",
    "delta_parent_feels_overwhelmed_p": "Change in Par. Overwhelmed",
    "delta_parent_feels_unloved_p": "Change in Par. Feels Unloved",
    "delta_parent_bad_family_relationship_p": "Change in Par. Fam. Relations",
    "delta_parent_worries_about_future_p": "Change in Par. Future Worry",
    "delta_parent_worries_a_lot_p": "Change in Par. Worries",
    "delta_parent_depressed_p": "Change in Par. Depressed",
    "delta_parent_depress_D_p": "Parent Depression Syndrome (Change Score)",
    "delta_parent_concentration_trouble_p": "Change in Par. Concentration",
    "delta_parent_stubborn_irritable_p": "Change in Par. Stubborn/Irritable",
    "delta_parent_drinks_too_much_p": "Change in Par. Drinking",
    "delta_parent_financial_failures_p": "Change in Par. Financial Failures",
    "delta_parent_meets_family_duties_p": "Change in Par. Fulfills Fam. Responsibilities",
    "delta_parent_planning_trouble_p": "Change in Par. Planning Trouble",
    "delta_parent_bad_relationships_p": "Change in Par. General Relationship Quality",
    "delta_parent_drug_use_p": "Change in Par. Drug Use",
    "delta_parent_drug_days_nonmedical_p": "Change in Par. Rec. Drugs",
    "delta_parent_financial_trouble_p": "Change in Par. Finances",
    "delta_parent_internal_D_p": "Change in Par. Internal.",
    "delta_parent_somatic_problems_D_p": "Change in Par. Somatic",
    # Parent Impulsivity and Behavior Regulation (16)
    "parent_impulsive_p": "Par. Impulsive",
    "parent_risky_decisions_p": "Par. Risky Decisions",
    "parent_drives_too_fast_p": "Par. Drives Fast",
    "parent_tardy_often_p": "Par. Tardy",
    "parent_money_management_trouble_p": "Par. Money Management Prob.",
    "parent_behavior_changeable_p": "Par. Behavior Labile",
    "parent_hot_temper_p": "Par. Hot Temper",
    "parent_attention_seeking_p": "Par. Attention Seeking",
    "parent_destroys_own_things_p": "Par. Destroys Own",
    "parent_destroys_others_things_p": "Par. Destroys Oth.",
    "parent_doesnt_finish_tasks_p": "Par. Task Completion",
    "parent_strange_behavior_p": "Par. Strange Behavior",
    "parent_illegal_behavior_p": "Par. Illegal Behav.",
    "parent_self_harm_p": "Par. Self-Harm",
    "parent_doesnt_eat_well_p": "Par. Poor Eating",
    # Parent Mood Issues (14)
    "parent_cries_a_lot_p": "Par. Cries",
    "parent_lonely_p": "Par. Lonely",
    "parent_feels_unloved_p": "Par. Unloved",
    "parent_paranoid_p": "Par. Paranoid",
    "parent_feels_inferior_p": "Par. Feels Inferior",
    "parent_depressed_p": "Par. Depressed",
    "parent_feels_unsuccessful_p": "Par. Feels Unsuccessful",
    "parent_tired_no_reason_p": "Par. Fatigue",
    "parent_low_energy_p": "Par. Low Energy",
    "parent_sleep_trouble_p": "Par. Sleep Trouble",
    "parent_enjoys_little_p": "Par. No Pleasure",
    "parent_sudden_mood_changes_p": "Par. Mood Swings",
    "parent_suicidal_thoughts_p": "Par. Suicidal",
    "parent_happy_person_p": "Par. Happiness",
    # Parent Other (4)
    "parent_heart_racing_p": "Par. Heart Racing",
    "parent_numbness_p": "Par. Numbness",
    "parent_physical_attacks_p": "Par. Panic Attacks",
    "parent_picks_skin_p": "Par. Skin Picking",
    # Parent Personality (21)
    "parent_bragging_p": "Par. Bragging",
    "parent_honest_p": "Par. Honest",
    "parent_secretive_p": "Par. Secretive",
    "parent_stubborn_irritable_p": "Par. Stubborn/Irritable",
    "parent_clumsy_p": "Par. Clumsy",
    "parent_strange_thoughts_p": "Par. Odd Thoughts",
    "parent_self_conscious_p": "Par. Low Self Confidence",
    "parent_uses_opportunities_p": "Par. Opportunistic",
    "parent_louder_than_others_p": "Par. Loud",
    "parent_yells_a_lot_p": "Par. Yells",
    "parent_shy_or_timid_p": "Par. Shy",
    "parent_restless_p": "Par. Restless",
    "parent_easily_bored_p": "Par. Easily Bored",
    "parent_hyperactive_p": "Par. Hyperactive",
    "parent_talks_too_much_p": "Par. Talkative",
    "parent_avoids_talking_p": "Par. Reticent",
    "parent_prefers_to_be_alone_p": "Par. Prefers to be Alone",
    "parent_no_guilt_p": "Par. No Guilt",
    "parent_sense_of_fairness_p": "Par. Fairness",
    "parent_high_sleep_duration_p": "Par. Long Sleep Duration",
    "parent_opposite_sex_wish_p": "Par. Gender Wish",
    # Parent Psychopathology Composite (5)
    "parent_anxdep_D_p": "Par. Anx./Dep.",
    "parent_depressed_mood_B_p": "Par. Dep. Mood",
    "parent_dysthymia_B_p": "Par. Dysthymia",
    "parent_suicide_past_B_p": "Par. Past Suicide",
    "ptsd_diagnosis_present_B_p": "Par. PTSD",
    # Parent Social Functioning (11)
    "parent_bad_relationships_p": "Par. General Bad Relationships",
    "parent_bad_family_relationship_p": "Par. Bad Family Relations",
    "parent_not_liked_by_others_p": "Par. Unliked by Others",
    "parent_friendship_trouble_p": "Par. Friendship Problems",
    "parent_prefers_older_people_p": "Par. Pref. Older Friends",
    "parent_associates_with_trouble_p": "Par. Gets in Trouble",
    "parent_bad_opposite_sex_relationship_p": "Par. Opposite Sex Relationships",
    "parent_meets_family_duties_p": "Par. Fulfills Fam. Responsibilities",
    "parent_clowns_or_shows_off_p": "Par. Socially Performative",
    "parent_teases_others_p": "Par. Teases",
    "parent_stands_up_rights_p": "Par. Assertive",
    # Physical Activity / Features (13)
    "height": "Height",
    "waist": "Waist Circumference",
    "no_sports_activities_p": "No Sports",
    "birth_weight_p": "Birth Weight",
    "fitbit_resting_hr": "Resting HR",
    "fitbit_steps": "Daily Steps",
    "fitbit_sedentary_mins": "Sedentary Min.",
    "fitbit_lightlyactive_mins": "Light Active Min.",
    "fitbit_fairlyactive_mins": "Moderate Active Min.",
    "fitbit_veryactive_mins": "Vigorous Active Min.",
    # Positive Affect (9)
    "pa_at_ease": "At Ease",
    "pa_attentive": "Attentive",
    "pa_calm": "Calm",
    "pa_concentrate": "Concentrating",
    "pa_confident": "Confident",
    "pa_delighted": "Delighted",
    "pa_energetic": "Energetic",
    "pa_enthusiastic": "Enthusiastic",
    "pa_interested": "Interested",
    # Residential Characteristics (19)
    "neighborhood_safety_ss_p": "Nbhd. Safety",
    "neighborhood_safe_y": "Nbhd. Safe Youth",
    "resid_density": "Pop. Density",
    "resid_walkability": "Walkability",
    "resid_prox_roads": "Road Proximity",
    "resid_crime_tot": "Total Crime",
    "resid_crime_violent": "Violent Crime",
    "resid_crime_drug": "Drug Crime",
    "resid_crime_dui": "DUI Rate",
    "resid_lead_risk_poverty": "Lead Risk Pov.",
    "resid_lead_risk_houses_perc": "Lead Risk Hsg.",
    "resid_lead_risk": "Lead Risk",
    "resid_no2_avg": "NO2 Exposure",
    "resid_pm25_avg": "PM2.5 Exposure",
    "resid_sexism": "Area Sexism",
    "resid_sex_orient_bias": "Area SO Bias",
    "resid_immigrant_bias": "Immigrant Bias",
    "resid_racism": "Area Racism",
    # SES & Mobility (11)
    "parent_education": "Par. Education",
    "struggle_food_expenses": "Food Hardship",
    # this one was reverse coded on accident
    "positive_finance_p": "Negative Financial Dynamics",
    "parent_work_absences_p": "Par. Work Absences",
    "parent_financial_trouble_p": "Par. Fin. Trouble",
    "parent_fails_to_pay_debts_p": "Par. Debt Default",
    "couldnt_afford_phone": "No Phone",
    "couldnt_afford_rent_mortgage": "Housing Hardship",
    "evicted": "Evicted",
    "gas_electric_oil_turned_off": "Utilities Cutoff",
    # School Dynamics (15)
    "disobeys_at_school_k": "Disobeys School (Y)",
    "getalong_teachers_k": "Teacher Rapport (Y)",
    "feelsafe_at_school_k": "School Safety (Y)",
    "feels_smart_k": "Feels Smart (Y)",
    "enjoys_school_k": "Enjoys School (Y)",
    "grades_important_k": "Grades Matter (Y)",
    "school_environment_ss_k": "School Environ. (Y)",
    "school_involvement_ss_k": "School Involve. (Y)",
    "school_disengagement_ss_k": "Disengagement (Y)",
    "repeated_grade": "Repeated Grade",
    "grades_dropped": "Grades Dropped",
    "school_detension_suspension": "Detention/Susp.",
    "child_newschool_p": "New School",
    "finds_schoolboring_k": "School Boring (Y)",
    # Sleep Problems (12)
    "difficulty_goingtosleep_p": "Onset Difficulty",
    "difficulty_wakingup_p": "Wake Difficulty",
    "fallsleeptime": "Sleep Onset Time",
    "wakeuptime": "Wake Time",
    "wakesleepcalc": "Sleep Duration",
    "chronotype": "Chronotype",
    "nightmares_p": "Nightmares",
    "sleeps_more_p": "Sleeps More",
    "sleeps_less_p": "Sleeps Less",
    "sleeps_little_p": "Little Sleep",
    "sleeps_alot_p": "Excess Sleep",
    "sleep_problems_p": "Sleep Problems",
    # Social Relationship Quality (34)
    "not_liked_p": "Not Liked By Others",
    "doesnt_get_along_p": "Peer Conflict",
    "prosocial_ss_p": "Prosocial",
    "peer_net_protective_ss_k": "Peer Protection (Y)",
    "peers_beh_prosocial_ss_k": "Peer Prosocial (Y)",
    "peers_beh_delinquent_ss_k": "Peer Delinquency (Y)",
    "not_invited_k": "Not Invited (Y)",
    "excluded_k": "Excluded (Y)",
    "otherkids_spreadneg_rumors_k": "Rumors (Y)",
    "otherkids_gossip_k": "Gossip (Y)",
    "feels_threatned_k": "Threatened (Y)",
    "saysmeanthings_others_k": "Says Mean (Y)",
    "otherkids_saymeanthings_k": "Mean to Me (Y)",
    "discrimination_ss_k": "Discrimination (Y)",
    "feels_discriminated_k": "Feels Discrim. (Y)",
    "senses_racism_k": "Senses Racism (Y)",
    "doesnt_feel_accepted_k": "Not Accepted (Y)",
    "bullied_on_internet_k": "Cyberbullied (Y)",
    "prosocial_ss_k": "Prosocial (Y)",
    "socialinfluence_meanfinal_k": "Social Influence (Y)",
    "relational_victimization_ss_k": "Relat. Victim. (Y)",
    "reputational_aggression_ss_k": "Reput. Aggress. (Y)",
    "reputational_victimization_ss_k": "Reput. Victim. (Y)",
    "overt_aggression_ss_k": "Overt Aggress. (Y)",
    "overt_victimization_ss_k": "Overt Victim. (Y)",
    "relational_aggression_ss_k": "Relat. Aggress. (Y)",
    "feels_discriminated_teachers_k": "Discrim. Teachers (Y)",
    "feels_discriminated_adults_not_school_k": "Discrim. Adults (Y)",
    "feels_discriminated_students_k": "Discrim. Students (Y)",
    "feels_unwanted_american_society_k": "Feels Unwanted (Y)",
    "feels_discriminated_americans_k": "Discrim. Society (Y)",
    # Technology Use (4)
    "socialmedia_daysperweek_k": "Social Media (Y)",
    "videogames_daysperweek_k": "Gaming Days (Y)",
    "vgame_thinking": "Gaming Preoccup.",
    # Structural and DTI (150)
    **{f"pc_struct{i}": f"Struct PC{i}" for i in range(1, 76)},
    **{f"pc_DTI{i}": f"DTI PC{i}" for i in range(1, 76)},
    # Task and Resting State (300)
    **{f"pc_SSTmri{i}": f"SST-fMRI PC{i}" for i in range(1, 76)},
    **{f"pc_nbackmri{i}": f"NBack-fMRI PC{i}" for i in range(1, 76)},
    **{f"pc_midfmri{i}": f"MID-fMRI PC{i}" for i in range(1, 76)},
    **{f"pc_rsFMRI{i}": f"RS-fMRI PC{i}" for i in range(1, 76)},
    # Social Responsiveness (SSRS): Parent Report
    "avoids_eyecontact_p": "Avoids Eye Contact",
    "difficulty_making_friends_p": "Difficulty Making Friends",
    "regarded_weird_p": "Regarded as Odd",
    "bad_conversational_flow_p": "Poor Conversational Flow",
    "narrow_interests_p": "Narrow Interests",
    "sensory_sensitivity_p": "Sensory Sensitivity",
    "concentration_on_parts_p": "Detail Fixation",
    "face_understanding": "Poor Social Cue Reading",
    # Barkley Deficits in Executive Functioning (BDEFS): Parent Report
    "bdefs_calm_down_p": "Difficulty Calming Down",
    "bdefs_consequences_p": "Ignores Consequences",
    "bdefs_distract_upset_p": "Emotional Refocusing Deficit",
    "bdefs_explain_idea_p": "Poor Idea Expression",
    "bdefs_explain_pt_p": "Unfocused Explanations",
    "bdefs_explain_seq_p": "Poor Sequential Narration",
    "bdefs_impulsive_action_p": "Impulsive Actions",
    "bdefs_inconsistant_p": "Inconsistent Performance",
    "bdefs_lazy_p": "Appears Unmotivated",
    "bdefs_process_info_p": "Slow Info Processing",
    "bdefs_rechannel_p": "Poor Emotion Redirection",
    "bdefs_sense_time_p": "Poor Time Sense",
    "bdefs_shortcuts_p": "Takes Shortcuts",
    "bdefs_stop_think_p": "Acts Before Thinking",
    # Child Emotional Regulation
    "child_emotion_overwhelm_p": "Emotion Overwhelm",
    "child_clear_feelings_p": "Emotion Clarity",
    # Child Emotion Regulation (ERC/DERS): Parent Report
    "child_feelings_attentive_p": "Attentive to Feelings",
    "child_feelings_care_p": "Cares About Feelings",
    "child_feelings_know_p": "Knows Own Feelings",
    "child_upset_acknowledge_p": "Acknowledges Upset",
    "child_upset_angry_p": "Upset: Gets Angry",
    "child_upset_ashamed_p": "Upset: Feels Ashamed",
    "child_upset_bad_behavior_p": "Upset: Bad Behavior",
    "child_upset_bad_esteem_p": "Upset: Low Self-Esteem",
    "child_upset_bad_focus_p": "Upset: Poor Focus",
    "child_upset_better_p": "Upset: Recovers Quickly",
    "child_upset_control_p": "Upset: Maintains Control",
    "child_upset_embarrassed_p": "Upset: Embarrassed",
    "child_upset_emotions_overwhelm_p": "Upset: Emotionally Overwhelmed",
    "child_upset_fixation_p": "Upset: Ruminative Fixation",
    "child_upset_guilty_p": "Upset: Feels Guilty",
    "child_upset_irritated_p": "Upset: Irritable",
    "child_upset_longtime_better_p": "Upset: Slow to Recover",
    "child_upset_longtime_p": "Upset: Prolonged Duration",
    "child_upset_lose_control_p": "Upset: Loses Control",
    "child_upset_no_control_p": "Upset: No Self-Control",
    "child_upset_nothing_better_p": "Upset: Feels Hopeless",
    "child_upset_out_control_p": "Upset: Out of Control",
    "child_upset_poor_concentrate_p": "Upset: Can't Concentrate",
    "child_upset_unproductive_p": "Upset: Unproductive",
    "child_upset_weak_p": "Upset: Feels Weak",
    # Emotion Regulation Strategies: Youth Self-Report (T3/T4)
    "emoreg_reapp_ss_k": "Reappraisal (Sum)",
    "emoreg_reapp_happy_k": "Reappraises: Happy",
    "emoreg_reapp_less_bad_k": "Reappraises: Less Bad",
    "emoreg_reapp_thoughts_k": "Reappraises: Thoughts",
    "emoreg_sup_ss_k": "Suppression (Sum)",
    "emoreg_sup_control_k": "Suppresses: Control",
    "emoreg_sup_hide_k": "Suppresses: Hides",
    "emoreg_sup_self_k": "Suppresses: Self",
    "child_selfaware_p": "Self-Awareness",
    # Smartphone Use (T3/T4)
    "sphone_Interupt": "Phone Interruptions",
    "sphone_attachment": "Phone Attachment",
    "sphone_distress": "Phone Distress",
    "sphone_noreason": "Phone: No Reason",
    # Parental Monitoring: Youth Self-Report (T3)
    "parent_care_friends_k": "Par. Knows Friends (Y)",
    "parent_care_misbehave_k": "Par. Monitors Behavior (Y)",
    "parent_care_whereabouts_k": "Par. Knows Whereabouts (Y)",
    "parent_cares_ss_k": "Parental Monitoring (Y)",
    "parent_gotoschool_k": "Par. School Attendance (Y)",
    "parent_helphomework_k": "Par. Homework Help (Y)",
    "parent_helpunderstanding_k": "Par. Helps Understand (Y)",
    "parent_safeplay_k": "Par. Safe Play (Y)",
    "parent_troubleschool_k": "Par. School Trouble (Y)",
    # Parental Acceptance: Youth Report (T3/T4)
    "y_acceptance_ss_caregiver_crpbi": "Caregiver Acceptance (Y)",
    "y_acceptance_ss_p_crpbi": "Parent Acceptance (Y)",
    # Social / Friendship (T3/T4)
    "close_nonB_friends_k": "Close Friends (Y)",
    "nonBfriends_k": "Number of Friends (Y)",
    # Other T3/T4-Specific
    "mania_7up_ss_k": "Mania Symptoms (Y)",
    "problem_solving_ss_k": "Problem Solving (Y)",
    "enjoys_music_k": "Enjoys Music (Y)",
    # CBCL Items (present in T3/T4 flat lists)
    "enjoys_little_p": "Enjoys Little",
    "guilty_p": "Feels Guilty",
    "sad_p": "Sad",
    "withdrawn_p": "Withdrawn",
    # Cognitive Tasks: Matrix Reasoning (T0)
    "mr_matrix": "Matrix Reasoning",
    "mr_total": "Matrix Reasoning Total",
    # Stroop Task (T1)
    "str_accuracy": "Stroop Accuracy",
    "str_accuracy_c": "Stroop Accuracy (Cong.)",
    "str_accuracy_ic": "Stroop Accuracy (Incong.)",
    "str_mrt": "Stroop Reaction Time",
    "str_mrt_c": "Stroop RT (Congruent)",
    "str_mrt_ic": "Stroop RT (Incongruent)",
    "str_stroop_acc": "Stroop Effect (Acc.)",
    "str_stroop_mrt": "Stroop Effect (RT)",
    # Medical History: Parent Report (T1)
    "medhx_allergies_p": "Hx: Allergies",
    "medhx_asthma_p": "Hx: Asthma",
    "medhx_brain_p": "Hx: Brain Condition",
    "medhx_brokenbones_p": "Hx: Broken Bones",
    "medhx_diabetes_p": "Hx: Diabetes",
    "medhx_epilepsy_p": "Hx: Epilepsy",
    "medhx_headaches_p": "Hx: Headaches",
    "medhx_heart_p": "Hx: Heart Condition",
    "medhx_seriousinjury_p": "Hx: Serious Injury",
    # Biomarkers / Hormones (T0/T1)
    "saliva_DHEA": "Salivary DHEA",
    "saliva_testosterone": "Salivary Testosterone",
    # Other T0/T1-Specific
    "screentime_weekend_ss_k": "Weekend Screen Time (Y)",
    "sex_P": "Biological Sex",
}

# Validation
logger.info(f"DISPLAY_NAMES contains {len(DISPLAY_NAMES)} mappings")

# TARGET_DISPLAY_NAMES: Outcome Variable Renaming for Figures
# Translates raw target variable names into publication-ready labels for
# heatmap axes, figure titles and summary tables. Used by the reporting,
# interpretation and validation stages.
# Naming conventions:
# "Mood/Depression (Primary Features)"  -- SBT core depression
# "Depression Syndrome (w/ Secondary Features)"  -- CBCL depress_D_p
# "Depression Subtype (X)"  -- SBT subscales
# "(Change Score)"  -- delta / longitudinal change
# "(RCI ≥ X)"  -- Reliable Change Index thresholds

TARGET_DISPLAY_NAMES = {
    # Core Syndrome Scales (Continuous)
    "sbt_core_depression": "Mood/Depression (Primary Features)",
    "depress_D_p": "Depression Syndrome (w/ Secondary Features)",
    "depress_D_p_rev": "Depression Syndrome (Reversed)",
    "depress_update_p": "Depression (Updated Composite)",
    "internal_D_p": "Internalizing Spectra",
    "external_D_p": "Externalizing Spectra",
    "adhd_D_p": "ADHD (Syndrome)",
    "anxdisord_D_p": "Anxiety (Syndrome)",
    "social_problems_D_p": "Social Problems (Syndrome)",
    "thought_disorder_D_p": "Thought Disorder (Syndrome)",
    # SBT Depression Subtypes (Continuous)
    "sbt_guilt_hopelessness": "Depression Subtype (Guilt/Hopelessness)",
    "sbt_social_withdrawal": "Depression Subtype (Social Withdrawal)",
    "sbt_fatigue_sleep": "Depression Subtype (Fatigue/Sleep)",
    "sbt_somatic_depression": "Depression Subtype (Somatic)",
    "sbt_anxiety_depression": "Depression Subtype (Anxiety/Depression Overlap)",
    "sbt_emotional_dysregulation": "Depression Subtype (Emotional Dysregulation)",
    "sbt_avoidance_fear": "Depression Subtype (Avoidance/Fear)",
    "sbt_aggression_irritability": "Depression Subtype (Aggression/Irritability)",
    "sbt_academic_cognitive": "Depression Subtype (Poor Academic/Cognitive)",
    "sbt_perfectionism_achievement": "Depression Subtype (Perfectionism/High-Achievement)",
    "sbt_well_being": "Depression Subtype (Well-Being)",
    # Change Scores (Continuous)
    "delta_depress_D_p": "Depression Syndrome (Change Score)",
    "delta_sbt_core_depression": "Mood/Depression (Change Score)",
    "sbt_core_depression_delta_t0_t3": "Mood/Depression (Change T0→T3)",
    "delta_anxdisord_D_p": "Anxiety (Change Score)",
    "delta_parent_depressed_p": "Parent Depression (Change Score)",
    "delta_parent_worries_a_lot_p": "Parent Worry (Change Score)",
    # RCI-Based Binary Outcomes
    "dep_onset_rci_2.3": "Depression Onset (RCI ≥ 2.3)",
    "dep_onset_rci_1.96": "Depression Onset (RCI ≥ 1.96)",
    "dep_remission_rci_2.3": "Depression Remission (RCI ≥ 2.3)",
    "dep_remission_rci_1.96": "Depression Remission (RCI ≥ 1.96)",
    "sbt_core_dep_onset_rci_2.3": "Mood/Depression Onset (RCI ≥ 2.3)",
    "sbt_core_dep_onset_rci_1.96": "Mood/Depression Onset (RCI ≥ 1.96)",
    "sbt_core_dep_remission_rci_2.3": "Mood/Depression Remission (RCI ≥ 2.3)",
    "sbt_core_dep_remission_rci_1.96": "Mood/Depression Remission (RCI ≥ 1.96)",
    # Threshold-Based Binary Outcomes
    "dep_increase_2sd": "Depression Increase (≥ 2 SD)",
    "dep_increase_1.5sd": "Depression Increase (≥ 1.5 SD)",
    "dep_decrease_2sd": "Depression Decrease (≥ 2 SD)",
    "dep_decrease_1.5sd": "Depression Decrease (≥ 1.5 SD)",
    "sbt_core_dep_increase_2sd": "Mood/Depression Increase (≥ 2 SD)",
    "sbt_core_dep_increase_1.5sd": "Mood/Depression Increase (≥ 1.5 SD)",
    "sbt_core_dep_decrease_2sd": "Mood/Depression Decrease (≥ 2 SD)",
    "sbt_core_dep_decrease_1.5sd": "Mood/Depression Decrease (≥ 1.5 SD)",
    "top_10_depression": "Depression (Top 10th Percentile)",
    "top_5_depression": "Depression (Top 5th Percentile)",
    "top_10_depression_parent": "Parent Depression (Top 10th Percentile)",
    "top_5_depression_parent": "Parent Depression (Top 5th Percentile)",
    "top_10_sbt_core_depression": "Mood/Depression (Top 10th Percentile)",
    "top_5_sbt_core_depression": "Mood/Depression (Top 5th Percentile)",
    # Clinical / Diagnostic Binary Outcomes
    "MDD_KSADS_C": "MDD Diagnosis (KSADS)",
    "depressed_mood_B_k": "Depressed Mood (KSADS Item)",
    "anhedonia_B_k": "Anhedonia (KSADS Item)",
    "wish_dead_present_B_k": "Wish to Be Dead (Child Report)",
    "parent_wish_dead_present_B_p": "Wish to Be Dead (Parent Report)",
    "latent_class_depression": "Depression Latent Class",
    "depadhd_c": "Depression × ADHD Comorbidity Class",
    "asdadhd_c": "ASD × ADHD Comorbidity Class",
    # Parent Outcomes
    "parent_depress_D_p": "Parent Depression (Syndrome)",
    "parent_anxdisord_D_p": "Parent Anxiety",
    "parent_adhd_D_p": "Parent ADHD",
    # Parent Social/Impulsivity/Compulsivity batch
    "parent_avoidant_person_D_p": "Parent Avoidant Personality (Syndrome)",
    "parent_not_liked_by_others_p": "Parent Not Liked by Others",
    "parent_repeats_acts_p": "Parent Compulsivity (Repeats Acts)",
    "parent_obsessive_thoughts_p": "Parent Obsessions (Obsessive Thoughts)",
    "parent_dep_onset_rci_2.3": "Parent Depression Onset (RCI ≥ 2.3)",
    "parent_bad_relationships_p": "Parent Relationship Problems",
    "parent_suicidal_thoughts_p": "Parent Suicidal Ideation",
    "parent_depressed_p": "Parent Mood/Depression (Primary Features)",
    "parent_worries_a_lot_p": "Parent Worry (Item)",
    "parent_concentration_trouble_p": "Parent Concentration Trouble (Item)",
    "parent_happy_person_p": "Parent Happiness (Item)",
    # Other Targets
    "bad_grades": "Poor Academic Grades",
    "tb_flanker": "Cognitive Control (Flanker)",
    "tb_fluid": "Fluid Intelligence",
    "tb_cryst": "Crystallized Intelligence",
    "tb_reading": "Reading Ability",
    # T0 Cognitive Task Battery additions
    "ravlt_l_total": "Verbal Episodic Memory (RAVLT)",
    "lmt_efficiency": "Visuospatial Mental Rotation (LMT)",
    "sst_mean_ssrt": "Response Inhibition (SST SSRT)",
    "nb_correct_nt_2back": "Emotional Working Memory (2-Back)",
    "suicidal_p": "Suicidal Ideation (Child)",
    "compulsive_p": "Compulsive Behavior",
    "impulsive_p": "Impulsive Behavior",
    "parent_impulsive_p": "Parent Impulsive Behavior",
    "parent_tobacco_use_frequency_p": "Parent Tobacco Use",
    "parent_drinks_frequency_p": "Parent Alcohol Use",
    "parent_drunk_days_p": "Parent Heavy Drinking",
    "parent_income": "Parent Income",
    "area_deprivation_idx": "Area Deprivation Index",
    "asd_ssrs_sum": "ASD Symptoms (SSRS Total)",
    "bdefs_distract_upset_p": "Executive Dysfunction (Distract/Upset)",
    "gd_riskybets": "Risk-Taking (Game of Dice)",
    "pa_sum_k": "Physical Activity (Child Report)",
    "weight": "Body Weight",
    # Parent Spectra
    "parent_internal_D_p": "Parent Internalizing Spectra",
    "parent_external_D_p": "Parent Externalizing Spectra",
    # RCI 1.7 Variants (Child)
    "dep_onset_rci_1.7": "Depression Onset (RCI ≥ 1.7)",
    "dep_remission_rci_1.7": "Depression Remission (RCI ≥ 1.7)",
    "sbt_core_dep_onset_rci_1.7": "Mood/Depression Onset (RCI ≥ 1.7)",
    "sbt_core_dep_remission_rci_1.7": "Mood/Depression Remission (RCI ≥ 1.7)",
    # 1 SD Variants (Child)
    "dep_increase_1sd": "Depression Worsened (≥ 1 SD)",
    "dep_decrease_1sd": "Depression Improved (≥ 1 SD)",
    # Parent Classification Targets
    "parent_dep_onset_rci_1.96": "Parent Depression Onset (RCI ≥ 1.96)",
    "parent_dep_remission_rci_1.96": "Parent Depression Remission (RCI ≥ 1.96)",
    "parent_dep_remission_rci_2.3": "Parent Depression Remission (RCI ≥ 2.3)",
    "parent_dep_increase_2sd": "Parent Depression Changes (≥ 2 SD)",
    "parent_dep_increase_1.5sd": "Parent Depression Changes (≥ 1.5 SD)",
    "parent_dep_decrease_2sd": "Parent Depression Improved (≥ 2 SD)",
    "parent_dep_decrease_1.5sd": "Parent Depression Improved (≥ 1.5 SD)",
    # KSADS Child: Depression / Mood
    "anhedonia_B_k": "Anhedonia (KSADS)",
    "depressed_mood_B_k": "Depressed Mood (KSADS)",
    "elevated_mood_B_k": "Elevated Mood (KSADS)",
    "fatigue_present_B_k": "Fatigue (KSADS)",
    "hopeless_B_k": "Hopelessness (KSADS)",
    "ksads_1_840_t": "Major Depressive Disorder (KSADS)",
    "two_more_depression_B_k": "≥2 Depressive Symptoms (KSADS)",
    # KSADS Child: Suicidality
    "suicidal_ideation_present_B_k": "Suicidal Ideation (KSADS)",
    "suicide_attempt_past_B_k": "Past Suicide Attempt (KSADS)",
    "suicide_attempt_present_B_k": "Suicide Attempt (KSADS)",
    # KSADS Child: Anxiety
    "excessive_worry_B_k": "Excessive Worry (KSADS)",
    "GAD_present": "Generalized Anxiety Disorder (KSADS)",
    "OCD_present": "OCD (KSADS)",
    "social_anxiety_disorder_B_k": "Social Anxiety Disorder (KSADS)",
    "socially_anxious_B_k": "Social Anxiety Symptoms (KSADS)",
    # KSADS Child: Other
    "anorexia_B_k": "Anorexia Nervosa (KSADS)",
    "bulimia_B_k": "Bulimia Nervosa (KSADS)",
    "lying_B_k": "Frequent Lying (KSADS)",
    "sleep_problem_B_k": "Sleep Problems (KSADS)",
    # KSADS Parent: Depression
    "hopeless_B_p": "Parent Hopelessness (KSADS)",
    "parent_anhedonia_B_p": "Parent Anhedonia (KSADS)",
    "parent_depressed_mood_B_p": "Parent Depressed Mood (KSADS)",
    "parent_dysthymia_B_p": "Parent Dysthymia (KSADS)",
    "parent_low_self_esteem_B_p": "Parent Low Self-Esteem (KSADS)",
    "parent_two_more_depression_B_p": "Parent ≥2 Depressive Symptoms (KSADS)",
    # KSADS Parent: Anxiety
    "parent_distress_due_to_social_anxiety_B_p": "Parent Social Anxiety Distress (KSADS)",
    "parent_excessive_worry_B_p": "Parent Excessive Worry (KSADS)",
    "parent_obsessions_present_B_p": "Parent Obsessions (KSADS)",
    "parent_social_anxiety_disorder_B_p": "Parent Social Anxiety Disorder (KSADS)",
    "parent_socially_anxious_B_p": "Parent Social Anxiety Symptoms (KSADS)",
    # KSADS Parent: Other
    "parent_concentration_issues_B_p": "Parent Concentration Problems (KSADS)",
    "parent_easily_annoyed_B_p": "Parent Irritability (KSADS)",
    "parent_sleep_problem_B_p": "Parent Sleep Problems (KSADS)",
    "ptsd_diagnosis_present_B_p": "Parent PTSD (KSADS)",
}


# Helper: resolve target display name
def get_target_display_name(raw_name):
    """Return the publication-ready display name for a target variable.
    Falls back to the raw name if no mapping exists."""
    return TARGET_DISPLAY_NAMES.get(raw_name, raw_name)


logger.info(f"TARGET_DISPLAY_NAMES contains {len(TARGET_DISPLAY_NAMES)} mappings")


def run() -> None:
    """Run this stage (notebook cell 4)."""

    # Load cleaned ABCD 5.1 panel (N ~23,760 after merging adolescent + parent
    # data across 5 annual waves).
    ctx.sample = pd.read_csv(ctx.DATA_PATH, low_memory=False)
    # Drop legacy unnamed index columns from older exports (e.g., from df.to_csv(index=True))
    _unnamed_cols = [c for c in ctx.sample.columns if str(c).startswith("Unnamed:")]
    if _unnamed_cols:
        ctx.sample = ctx.sample.drop(columns=_unnamed_cols)
