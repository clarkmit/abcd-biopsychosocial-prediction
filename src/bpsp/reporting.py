"""Pipeline overview, report state and report hooks (setup).

ABCD Study: biopsychosocial predictive modeling pipeline.
Adolescent Brain Cognitive Development Study, Release 5.1: 11,864
adolescents (ages 9 to 14) and about 11,859 parents and caregivers (mean
age about 40).

Architecture  : stacking ensemble (CatBoost, XGBoost, random forest,
                TabPFN) with a linear or logistic meta-learner
Targets       : multiple depression operationalizations (dimensional,
                RCI-based, percentile thresholds), anxiety, ADHD,
                externalizing, social problems and cognitive/educational
                outcomes across five annual assessment waves (T0 to T4)
Parcellation  : ~1,078 biopsychosocial variables organized into child and
                parent ontological categories for comparison (Fig 1) and
                across-domain analyses with circular-feature exclusions

This module also holds the lightweight report state (_RS) and the cache
and report hooks that the validation stages call. Apart from make_run_key,
the hooks do nothing in the package.
"""

# Report state and pipeline hooks
# Lightweight report state container (_RS) used by later stages
# to track which section is active and collect figures/metrics.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# IterativeImputer is experimental; this enable import must precede the imputer import.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

CACHE_ROOT = Path(".")  # base directory for reports/figures

_RS = {
    "section": "init",
    "run_key": None,
    "figures": [],
    "metrics": {},
    "tables": [],
    "fig_seq": {},
    "data_info": {},
    "_patched": False,
}

# Hooks called by the validation stages (no-ops in the package, apart from make_run_key).


def make_run_key(extra=None):
    import time

    return f"run_{int(time.time())}"


def cache_load(tag, key=None):
    return None


def fold_load_all(key, tag):
    return {}


def cache_save(data, cell_tag=None, key=None, label=None):
    pass


def report_save_state(key=None):
    pass


def _pipeline_cleanup(label="", verbose=False):
    pass


def report_init(key=None):
    _RS["run_key"] = key
    _RS["section"] = "init"


def report_load_state(key=None):
    return False


def report_log_data_info(X_train=None, X_test=None, y_train=None, y_test=None):
    try:
        _RS["data_info"] = {
            "n_train": len(X_train) if X_train is not None else 0,
            "n_test": len(X_test) if X_test is not None else 0,
            "n_features": X_train.shape[1] if X_train is not None and hasattr(X_train, "shape") else 0,
        }
    except Exception:
        pass


def report_log_table(df, title="", section=""):
    try:
        _RS["tables"].append({"title": title, "df": df, "section": section})
    except Exception:
        pass
