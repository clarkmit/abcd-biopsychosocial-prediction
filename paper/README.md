# Code version used for the article (v1.0.0)

The two notebooks are release v1.0.0, kept unchanged: the notebook used for the article's analyses and its demonstration notebook.

| File | Contents |
|---|---|
| `ABCDMLPipeline_v1.0.0.ipynb` | the analysis notebook (cells 0 to 24) |
| `demo_v1.0.0.ipynb` | the v1.0.0 demonstration notebook |
| `requirements.txt` | the package versions v1.0.0 was tested with (Python 3.13), including PyTorch and TabPFN, plus the three the `bpsp` package also needs |

The notebooks match release [v1.0.0](https://github.com/clarkmit/abcd-biopsychosocial-prediction/releases/tag/v1.0.0), archived on Zenodo: [10.5281/zenodo.23540757](https://doi.org/10.5281/zenodo.23540757).

The `bpsp` package in `src/` is the maintained version of the same code. Run side by side with this notebook on simulated panels, it gives the same results at every stage, apart from a few deliberate changes listed in `docs/equivalence.md`. `docs/stages.md` maps each notebook cell to its stage.

The notebook was written for Google Colab and uses f-string syntax introduced in Python 3.12, so it needs Python 3.12 or later. The package runs on Python 3.11 to 3.13.

## Running the notebook

1. With an approved ABCD Data Use Certification, place the merged multi-wave file where the notebook can read it and set `DATA_PATH` at the top of cell 4.
2. Run cell 0 (Setup) to install dependencies on Colab.
3. Run cell 1 (Variable Registry), cells 2 to 4 (circularity exclusions, the optional objective/subjective spectrum in cell 3, then preprocessing, natural language variable renaming and train-test partitioning) and cell 5 (stacking ensemble architecture).
4. Configure the analysis in the form of cell 6 (Main Analysis Runner: `tp_option`, `target_options`, `split`, `batch_mode` and the other fields) and run it.
5. Run cell 7 for the batch summary (classification batches), cells 8 to 10 for Sankey diagrams, partial correlation networks and diagnostic plots, cell 11 for a quick pipeline smoke test, and cells 12 to 24 for the validation suite. The cell titles of cells 21 to 24 use earlier supplement numbers (S1.21 to S1.24), which are S5.1 to S5.4 in the article.

Do not edit the notebooks. Changes go into the package.
