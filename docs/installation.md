# Installation

## Requirements

* Python 3.11 to 3.13
* CPU is enough for the demo and for across-category runs. TabPFN (within-category runs) needs a CUDA GPU (a T4 or better) with at least 10 GB of memory. The published analyses ran on an A100.
* At least 16 GB of system RAM for across-category feature combinations (32 GB recommended for full within-category sweeps across all waves).
* v1.0.0 was run on Ubuntu 22.04 (Colab), macOS 14 and Windows 11 via WSL2. v2.0.0 is tested on Linux in CI.

## Install

```bash
git clone https://github.com/clarkmit/abcd-biopsychosocial-prediction.git
cd abcd-biopsychosocial-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # core pipeline (CPU)
pip install -e ".[tabpfn]"       # optional: TabPFN base learner and GPU support
```

Optional extras:

| Extra | Installs | Use |
|---|---|---|
| `tabpfn` | `tabpfn`, `torch` | TabPFN base learner (within-category runs) |
| `notebooks` | JupyterLab | the notebooks in `notebooks/` |
| `docs` | MkDocs Material, mkdocstrings | building this site |
| `dev` | pytest, pytest-cov, mypy, ruff, pre-commit, nbstripout | development |

## Pinned environments

Two files pin exact versions:

* `requirements-lock.txt` is the complete environment in which v2.0.0 was verified (Python 3.12 on Linux x86-64, checked with the unit, integration and equivalence tests). Use it for exact reruns:

  ```bash
  pip install -r requirements-lock.txt
  pip install -e . --no-deps
  ```

* `paper/requirements.txt` lists the versions v1.0.0 was tested with (Python 3.13), including PyTorch and TabPFN.

`pyproject.toml` itself gives compatible ranges, and CI tests the newest releases on Python 3.11 to 3.13.

## Google Colab

Open `notebooks/01_quickstart.ipynb` or `notebooks/02_full_analysis.ipynb` and use the "Open in Colab" badge at the top of the notebook. Its first cell clones the repository and installs the package.

## Check the install

```bash
bpsp --version
bpsp stages
pip install -e ".[dev]" && pytest -m "not integration and not slow"
```
