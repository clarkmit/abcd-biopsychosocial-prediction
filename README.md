# ABCD Biopsychosocial Predictive Modelling Pipeline

**High-Dimensional Ensemble Stacking Machine Learning Pipeline for Hierarchical, Interactive, and Dynamic Prediction of Psychiatric, Cognitive, and Social Outcomes in the ABCD Study**

This repository contains the analytic pipeline used in:

> Roberts, C. et al. * Psychiatry and Its Hierarchical & Dynamic Discontents: Biopsychosocial Prediction in Depression, Anxiety, ADHD, Social Relationships, and Cognition* (Submitted)

The pipeline organizes ~1,078 biopsychosocial variables extracted from the Adolescent Brain Cognitive Development (ABCD) Study, Release 5.1 (combined N ≈ 23,760 adolescents age 9–14 and their parents/caregivers, mean parent age ≈ 40), and uses a stacking machine learning ensemble to compare the hierarchical, interactive, concurrent, and prospective predictive capacities of a wide range of biological, psychological, social, and environmental measurement domains across multiple psychiatric and cognitive outcomes.

*\*\*Use of ABCD data requires a license through the NIMH Data Archive ([NDA](https://nda.nih.gov/abcd)). Variable names in the pipeline are renamed for confidentiality and interpretability. Please contact the corresponding author if you would like variable mapping for replication.\*\**


Runs optimally on a T4 or better, particularly for GPU for TabPFN functionality. If running locally, please ensure ≥ 10 GB GPU RAM to run TabPFN, and ≥ 16 GB system RAM for across-category feature combinations. Research article results ran on A100 and is recommended, particularly with much extended added validation functions (especially permutation testing). Several options in Main Analysis Runner to reduce timing needed. 

## Central Project Aim

Address the fundamental question: Which of the wide range of currently available biopsychosocial measurements best predict hallmark psychiatric outcomes in adolescents and adults, and where do they converge or diverge in their predictive architecture? Feature sets are organized for both Across and Within Domains. Latter use a diverse ontology of 37 theoretically driven, commonly used, or potentially informative categories (arranged along a rough biopsychosocial and objective-subjective gradient) which are arranged here into feature sets so as to assess hierarchical, interactive, concurrent, and various forms of temporal predictive power.

## Organizational Procedures

Feature-set categories include:

- **Biological:** Multimodal neuroimaging (structural MRI morphometry, DTI, resting-state fMRI, task-fMRI from MID/SST/N-back), ancestral genetic principal components, Fitbit-derived physical activity and physiology, pubertal/somatic measures.
- **Psychological:** Experimental cognitive assessments (NIH Toolbox, Stop-Signal, N-back, MID, RAVLT, Little Man Task, Game of Dice) including hierarchical Bayesian individual-level PRAD parameters, personality traits, emotion regulation style, coextensive individual and familial psychopathology dimensions (CBCL, ASR, KSADS), sleep, etc.
- **Social:** Parenting styles, family dynamics and conflict, quality and dynamics of peer relationships, social discrimination and exclusion, parent social functioning.
- **Environmental:** Adverse life events (child and parent report), residential characteristics and safety, neighborhood deprivation, school dynamics, cultural and religious factors, socioeconomic indicators.
- **Temporal:** Delta (change-score) variables are computed between adjacent available waves for theoretically relevant measures (other psychopathology, family conflict, social problems, adverse life events, school problems, grades, etc.) to assess the predictive utility of recent within-person change.

An alternative parcellation along a spectrum of classically more objective (genetic PCs, neuroimaging, cognitive tasks, Fitbit, physiological) to subjective (child-self, parent-self, parent-to-child, dyadic) measurement modalities is also provided for modality-discrepancy analyses.

Circularity criterion are already set up for all targets used in paper and many more. These can be used for adolescent (9-14 years of age) and their parents/adults(23-75) targets, including possible intergenerational relationships. Importantly though, as data acquisition was primarily centered around adolescents, several categories and their corresponding measures were only collected in children such as multimodal neuroimaging, experimental tasks, Fitbit data, etc.  


## Predictive Targets & Circularity Exclusion Setup for:

- **Many Hallmark psychiatric dimensions:** depression, anxiety, ADHD, externalizing, internalizing, social problems (child CBCL and parent ASR). Several more specific items are included also such as impulsivity, compulsivity, etc. 
- **Broader ‘P-Factor’ spectra:** CBCL/ASR-derived Internalizing and Externalizing spectra in children and parents across all available waves.
- **Multiple depression operationalizations:** dimensional syndrome and primary-features-only composites; top 5th/10th percentile severity classifications; Reliable Change Index (RCI, 1.7, 1.96, and 2.3 thresholds) and standardized-difference (≥1.5 SD, ≥2 SD) change-score classifications; KSADS-derived symptom and diagnostic targets.
- **Additional targets for stratification and contrast:** 11 theoretically derived depression subtypes (Fatigue/Sleep, Social Withdrawal, Guilt/Hopelessness, Anxiety/Depression Overlap, Avoidance/Fear, Emotional Dysregulation, Aggression/Irritability, Somatic, Poor Academic/Cognitive, Perfectionism/High-Achievement, and Well-Being) generated from permutations of CBCL items sharing the primary anhedonia/sadness core.
- **Contrasting non-psychopathological outcomes:** cognitive task performance (NIH Toolbox Flanker, Reading, N-back, etc.), academic grades, weight, combined family income, myriad of technology-use variables in adolescents where available. Note, unlike mental health outcomes from CBCL, some of these were more objective variables were only collected at 1 or 2 of the 5 available time points. Select tp_option all, and analyses will run on time points which were available. 


## Overview

The pipeline supports:

- Predictive and comparative modelling of hallmark psychiatric dimensions and a myriad of other social, cognitive, and academic outcome related targets in adolescents and adults.
- Feature-engineering of longitudinal data across five available timepoints (T0–T4), with categorized feature-sets matched to each wave's data acquisition scope. Note, parent targets were not collected at T1 or T3. 
- Multi-model stacking ensemble architecture.
- Weighted consensus feature importance across base learners and meta-model, and all base learner SHAP values. Additional option to only output SHAP from best-performing model. 
- Visualization, model diagnostics, and a validation suite (nested CV, permutation testing, family- and site-grouped CV, PCA baselines, temporal hold-out, demographic stratification, domain ablation).

## Key Features in the Analytic Pipeline

**Stacking Ensemble Models:**
- CatBoost
- XGBoost
- Random Forest
- TabPFN (GPU-accelerated; auto-excluded for across-category runs, whose feature counts typically exceed TabPFN's recommended d ≤ 500 ceiling)
- Linear meta-learner (regression targets) / Logistic meta-learner (classification targets)
- Additional simpler models (Elastic Net, Ridge) for quick analyses, validation, and performance contrast


**Visualization Options:**
- Within- and across-category feature importance plots (pre-defined targets are color coded and further options available)
- Sankey diagrams for across-category feature-to-target architecture (configurable styles for internalizing/externalizing spectra, developmental sweeps, depression worsening, severity thresholds, and subtype trajectories)
- SHAP summary and beeswarm plots
- Model performance ROC, PR, calibration, and hexbin OOF prediction plots
- Partial correlation network figures output with top 10 features of ML ensemble

**Validation Suite (Cells 12–20):**
- Nested cross-validation (5 or 10 outer folds) with 95% confidence intervals
- Extended demographic-stratified performance (child sex, parent sex, parent education tertile, SES tertile, ALE exposure)
- Family- and site-grouped CV, including leave-site-out cross-validation
- Permutation testing (1,000 permutations) with Phipson-Smyth correction
- Stacking audit (base-model contribution analysis, meta-learner weight inspection, leakage checks)
- PCA baseline comparisons and temporal hold-out generalization
- Domain ablation / feature-domain importance analysis
- Cross-timepoint performance trajectories
- Temporal gradient / concurrent-vs-prospective decay analysis

**Modelling Support:**
- Built-in preprocessing (scaling, imputation) fit exclusively on training partitions to prevent leakage
- Systematic per-target circularity exclusion protocol (documented in code and supplement) to remove tautological predictors, construct overlap, and plausible reverse-causation confounds
- 70/30 stratified train/test splitting with out-of-fold base-learner predictions as meta-learner input
- Handles mixed data types (continuous, categorical, ordinal) natively 
- Timepoint-aware feature-set construction (variables available at each wave organized for meaningful longitudinal comparison). T1/T3 not available for Parent Targets. 
- Configurable batch target bundles will run multiple targets at once (e.g., Child Dep Classification, Parent Dep Classification, F4: Depression & Anxiety Developmental Sweep)
- Several options available in Main Analysis for temporal prediction, isolating categories, visualization options, etc. 

## Requirements

Python 3.10+ recommended. Core dependencies (pinned versions for reproducibility):

```
catboost>=1.2
xgboost>=1.7
scikit-learn>=1.3
shap>=0.44
tabpfn>=2.0
pygam>=0.9
mapie>=1.0.1
optuna>=3.4
optuna-integration
imbalanced-learn>=0.11
smogn
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
networkx>=3.0
statsmodels>=0.14
```

## Installation

```
pip install -r requirements.txt
```

Or simply run the Setup/Initialization cell (Cell 0) in the notebook, which handles package installation on Colab automatically.


**Operating systems tested:** Ubuntu 22.04 (Colab), macOS 14 (local), Windows 11 via WSL2.

**Hardware requirements:**
- GPU: T4 or better (required for TabPFN). A100 recommended for most large feature sets or time points. 
- GPU RAM: ≥ 10 GB.
- System RAM: ≥ 16 GB (≥ 32 GB recommended for full within-category sweeps across all waves).

## Usage

The main analysis pipeline is in `ABCDMLPipeline.ipynb`. A minimal workflow:

1. Acquire ABCD data through NDA (license required) and place the merged multi-wave tabular file in the working directory, then update the `DATA_PATH` variable at the top of Cell 4 to match your filename.
2. Run Cell 0 (Setup) to install dependencies.
3. Run Cell 1 (Variable Registry) to load the child/parent categorized feature-sets for all timepoints.
4. Run Cells 2–4 to apply circularity exclusions, optionally load the objective/subjective spectrum feature-sets (Cell 3), and execute preprocessing and train-test partitioning.
5. Run Cell 5 to initialize the stacking ensemble architecture.
6. Configure the analysis in Cell 6 (Main Analysis Runner):

```python
tp_option       = '2'                    # Timepoint selection (0-4 or 'All')
target_options  = 'sbt_core_depression'  # Target variable
split           = 'within_categories'    # or 'across_categories'
single_category_mode = False             # True to skip the full category loop and run only isolate_category
isolate_category = 'None'                # e.g., 'Other Personality Features' (used when single_category_mode=True)
run_validation_cells = True              # Enables Cells 12-20 after main run
```

Sections 2 and 3 of the same UI panel control model and batch behaviour. `ensemble_version = True` with `ensemble_type = 'stacking'` selects the four-model stacking ensemble used in the manuscript; the alternatives are retained for legacy comparison only. `skip_tabpfn = True` drops TabPFN for an approximate 20 to 30 percent speedup and is auto-applied for across-category runs whose feature counts exceed TabPFN's recommended d ≤ 500 ceiling. `batch_mode` overrides single `target_options` mode and runs a curated bundle of targets together (e.g. "Child Dep Classification", "T0 Cognitive Task Battery"); set it to `"Single Target"` to ONLY prespecified target. The within-categories sub-block (`isolate_combinations`, `isolate_category`, plus `single_category_mode` above) is active only when `split = "within_categories"`, and either restricts the run to the selected category or runs every combination anchored on it.

7. Run Cell 6 to execute the analysis.
8. Run Cell 7 for the batch summary classification table (classification batch runs only), then Cells 8–10 for Sankey visualizations, partial correlation networks, and diagnostic plots.
9. (Optional) Run Cell 11 for a quick pipeline smoke test, followed by Cells 12–20 for additional validations.

## Demo

A synthetic-data demo notebook (`demo.ipynb`) and matching synthetic panel file (`CLEAN_ABCD_5_1_panel_synthetic.csv`) are included so users without ABCD access can verify that the pipeline installs and runs end-to-end. The demo loads the synthetic file, fits the stacking ensemble (CatBoost + XGBoost + Random Forest base learners with a linear meta-learner; TabPFN omitted for CPU compatibility) on a baseline (T0) feature set covering cognitive task scores, demographics, and genetic ancestry PCs, then reports test-set metrics and a predicted-vs-observed plot. Expected runtime: ~1–2 minutes on a Colab T4 runtime. Because the synthetic file is produced by independent column-wise permutation, predictive performance will be near zero; the demo verifies execution, not reproducibility of manuscript findings.

**Typical runtime on real data:**
- Within-category single-target run varies by time point data availability
- Across-category single-target run (~420 features, N ≈ 10,000): ~15–45 minutes on A100 GPU.
- Full validation suite (Cells 12–20) for one target: 3-30 hours on A100, largely depending on permutation testing count and CV fold setting.

## Data Format

Required data structure for full analysis:

- Longitudinal tabular data across timepoints (T0–T4), merged by subject ID, with wave indicator column.
- Features organized by ontological categories (27 child / 21 parent at T0; 37 / 32 unique across all five waves), with variable-to-category mappings hardcoded in Cell 1's registry; users working with matched datasets can modify this registry.
- Target variables for depression and related outcomes (CBCL/ASR raw and derived scores, percentile and RCI thresholds precomputed).
- Family-ID and site-ID columns for grouped cross-validation.
- Delta variables precomputed from adjacent waves and entered as static columns subject to the same train/test split-then-preprocess ordering.

Access to ABCD data requires a license through [NDA](https://nda.nih.gov/abcd). See the Data and Code Availability statement in the manuscript for the ABCD DOI reference.

## License

This code is released under the MIT License. See `LICENSE` for the full text. Use of the ABCD dataset itself is governed by the NDA Data Use Certification; obtaining a license is required prior to running this pipeline on real ABCD data.

## Citation

If you use this pipeline, please cite:

> Roberts, C. et al. *Hierarchical, Interactive, and Dynamic Predictive Capacity of Current Biopsychosocial Measures in Depression, Anxiety, Social Relationships, ADHD, and Cognition.* (Submitted). Zenodo DOI: *[to be inserted upon acceptance]*.

## Contributors

Clark Roberts (conceptualization, feature-engineering, ML pipeline programming, implementation of manuscript models, statistical analysis, visualization), Zhafira E. Fawnia (Ensemble ML programming).

## Contact

For questions about the code, please contact Clark Roberts (Clark001@mit.edu). For questions about data access, please consult the [NDA ABCD portal](https://nda.nih.gov/abcd). Note that reproduction of the full analyses requires an approved ABCD Data Use Certification.


