# Data access and governance

## ABCD data

The analyses use the Adolescent Brain Cognitive Development (ABCD) Study, Release 5.1: N = 11,864 adolescents (ages 9 to 14) and N ≈ 11,859 parents or caregivers, across five annual waves (T0 to T4).

ABCD data can only be used under an approved Data Use Certification (DUC). The article used Release 5.1, obtained through the NIMH Data Archive (NDA). The NDA no longer accepts new ABCD access requests: access now goes through the [NIH Brain Development Cohorts (NBDC) Data Hub](https://www.nbdc-datahub.org), which also hosts later releases. The variable registry uses Release 5.1 variable names.

This repository contains no ABCD data and must never contain any, including participant-level outputs (predictions, residuals, row-level tables) produced from ABCD data. The ABCD terms also prohibit putting ABCD data into generative AI tools.

## What is in the repository

| File | Contents |
|---|---|
| `data/simulated/CLEAN_ABCD_5_1_panel_simulated.csv` | Fully simulated panel: same columns and five-wave structure as the analysis file, values drawn at random from column-level summaries. No participant data, no real associations. |
| `data/variable_map.csv` | Each variable in the analysis file with its source ABCD table, original variable name and figure label: the dictionary the article provides with the code and in Supplement S1.1. Derived composites, change scores and thresholds are marked `derived`. |

## Safeguards

* `.gitignore` excludes common tabular, statistics-package (SPSS, Stata, SAS, R), array and archive formats, except the two files above.
* A pre-commit hook (`tools/check_no_participant_data.py`) blocks the same file types, any other file over 5 MB, and any file containing ABCD participant identifiers (`NDAR_INV...`). CI runs the same check on every pull request and every push to `main`. Archives are blocked because their contents cannot be checked.
* Run outputs go to `outputs/` by default, which is ignored by git, as are figures (PNG, PDF, SVG). Keep real-data outputs outside the repository folder altogether.

If ABCD data or participant-level outputs are ever committed by mistake, deleting them in a new commit is not enough: the history must be rewritten (for example with `git filter-repo`), forks checked, and GitHub Support asked to purge cached views. Check your DUC for reporting obligations.

## Expected input format

* One row per participant per wave, with `subject` and `time` (0 to 4) columns.
* Feature columns named as in `data/variable_map.csv`, with target variables, change scores and neuroimaging components already computed.
* Family ID (`rel_family_id`) and site ID (`L_site_id`) columns for grouped cross-validation.

The merged file is built with the extraction code in the companion repository [abcd-visualizations-and-data](https://github.com/ra2yamamoto/abcd-visualizations-and-data).

## Acknowledgment

The article, and the article figures shown in this documentation, carry the standard ABCD acknowledgment:

> Data used in the preparation of this article were obtained from the Adolescent Brain Cognitive Development (ABCD) Study (https://abcdstudy.org), held in the NIMH Data Archive (NDA). This is a multisite, longitudinal study designed to recruit more than 10,000 children age 9-10 and follow them over 10 years into early adulthood. The ABCD Study is supported by the National Institutes of Health and additional federal partners under award numbers U01DA041048, U01DA050989, U01DA051016, U01DA041022, U01DA051018, U01DA051037, U01DA050987, U01DA041174, U01DA041106, U01DA041117, U01DA041028, U01DA041134, U01DA050988, U01DA051039, U01DA041156, U01DA041025, U01DA041120, U01DA051038, U01DA041148, U01DA041093, U01DA041089, U24DA041123, U24DA041147. A full list of supporters is available at https://abcdstudy.org/federal-partners.html. A listing of participating sites and a complete listing of the study investigators can be found at https://abcdstudy.org/consortium_members/. ABCD consortium investigators designed and implemented the study and/or provided data but did not necessarily participate in the analysis or writing of this report. This manuscript reflects the views of the authors and may not reflect the opinions or views of the NIH or ABCD consortium investigators. ABCD DOI: https://doi.org/10.15154/abt7-2i22
