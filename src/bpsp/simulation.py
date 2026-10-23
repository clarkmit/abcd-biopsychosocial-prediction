"""Simulated panels with a known answer, for demonstrations and tests.

The panel shipped in ``data/simulated/`` draws every column independently, so no outcome can
be predicted from it. :func:`make_example_panel` resamples it to a chosen number of subjects
and plants a known structure at wave T2 (``time == 2``), so that a run has a correct answer
to recover:

* a latent score drives three parent-report CBCL items (``stubborn_p``, ``disagreeable_p``,
  ``blames_others_p``, recorded as 0/1/2);
* the CBCL internalizing and depression T-scores (``internal_D_p``, ``depress_D_p``) are
  ``55 + 6 * (0.8 * latent + 0.6 * delta_not_liked_p + noise)``, and the externalizing score
  (``external_D_p``) uses 0.7 times the same signal;
* the binary depression targets of the "Child Depression Worsening" batch
  (``top_10_depression``, ``top_5_depression``, ``dep_increase_1.5sd``, ``dep_increase_2sd``)
  are thresholds of that signal;
* the noise standard deviation is 1.3 instead of 0.8 for participants with
  ``desc_latin_B == 1``, so the subgroup analyses have a real difference to detect.

Everything else carries no information, and baseline demographics are held constant within
subject. The data remain fully synthetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PLANTED_FEATURES: tuple[str, ...] = ("stubborn_p", "disagreeable_p", "blames_others_p", "delta_not_liked_p")
"""Predictors that carry the planted signal at T2."""

_BASELINE_CONSTANT = (
    "sex",
    "child_religion",
    "religious_service_frequency",
    "relig_importance",
    "parent_education",
    "parent_income",
    "sex_P",
    "L_site_id",
)
_BINARY_TARGETS = ("top_10_depression", "top_5_depression", "dep_increase_1.5sd", "dep_increase_2sd")


def make_example_panel(source: str | Path, n_subjects: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Return a simulated panel of ``n_subjects`` x 5 waves with the planted signal described above.

    Parameters
    ----------
    source:
        The simulated panel shipped with the repository,
        ``data/simulated/CLEAN_ABCD_5_1_panel_simulated.csv``.
    n_subjects:
        Number of simulated participants (rows per wave).
    seed:
        Seed for resampling and for the planted signal; the same arguments give the same panel.
    """
    rng = np.random.RandomState(seed)
    base = pd.read_csv(source, low_memory=False)
    base = base.drop(columns=[c for c in base.columns if c.startswith("Unnamed")])
    subjects = [f"SIM{i:05d}" for i in range(n_subjects)]

    # Resample each column independently within each wave.
    waves = []
    for t in sorted(base["time"].unique()):
        rows = base[base["time"] == t].reset_index(drop=True)
        wave = {"subject": subjects, "time": [t] * n_subjects}
        for c in base.columns:
            if c not in ("subject", "time"):
                wave[c] = rows[c].values[rng.randint(0, len(rows), n_subjects)]
        waves.append(pd.DataFrame(wave))
    panel = pd.concat(waves, ignore_index=True)

    # Demographics are properties of the participant, not of the visit.
    constant = list(_BASELINE_CONSTANT) + [c for c in base.columns if c.startswith("desc_")]
    first = panel[panel["time"] == 0].set_index("subject")
    for c in constant:
        if c in panel.columns:
            panel[c] = panel["subject"].map(first[c])

    t2 = panel["time"] == 2
    n2 = int(t2.sum())
    latent = rng.randn(n2)
    for c in ("stubborn_p", "disagreeable_p", "blames_others_p"):
        if c in panel.columns:
            panel.loc[t2, c] = np.clip(np.round(1 + 0.8 * (latent + 0.6 * rng.randn(n2))), 0, 2)
    delta = rng.randn(n2)
    if "delta_not_liked_p" in panel.columns:
        panel.loc[t2, "delta_not_liked_p"] = delta
    noisier = (panel.loc[t2, "desc_latin_B"] == 1).values
    noise = rng.randn(n2) * np.where(noisier, 1.3, 0.8)
    signal = 0.8 * latent + 0.6 * delta
    for target, scale in (("internal_D_p", 1.0), ("depress_D_p", 1.0), ("external_D_p", 0.7)):
        if target in panel.columns:
            panel.loc[t2, target] = 55 + 6 * (scale * signal + noise)

    depression = panel.loc[t2, "depress_D_p"].values
    for c in _BINARY_TARGETS:
        panel[c] = pd.to_numeric(
            panel[c].map(lambda v: {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}.get(v, v)), errors="coerce"
        )
    panel.loc[t2, "top_10_depression"] = (depression > np.quantile(depression, 0.90)).astype(float)
    panel.loc[t2, "top_5_depression"] = (depression > np.quantile(depression, 0.95)).astype(float)
    change = 0.9 * delta + 0.4 * latent + rng.randn(n2)
    panel.loc[t2, "dep_increase_1.5sd"] = (change > np.quantile(change, 0.965)).astype(float)
    panel.loc[t2, "dep_increase_2sd"] = (change > np.quantile(change, 0.98)).astype(float)
    return panel
