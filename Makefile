.DEFAULT_GOAL := help
.PHONY: help install dev lint typecheck format test test-integration test-equivalence demo notebook docs clean

help:               ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-18s %s\n", $$1, $$2}'

install:            ## install the package
	pip install -e .

dev:                ## install with development tools and git hooks
	pip install -e ".[dev]"
	pre-commit install

lint:               ## static checks
	ruff check .
	ruff format --check .

typecheck:          ## type-check the hand-written modules
	mypy

format:             ## apply formatting and safe fixes
	ruff check --fix .
	ruff format .

test:               ## unit tests with coverage of the hand-written modules (seconds)
	pytest -m "not integration and not slow" --cov

test-integration:   ## end-to-end runs on the simulated panel (minutes)
	pytest -m integration

test-equivalence:   ## compare with the published notebook (long)
	pytest -m slow -v

demo:               ## build the example panel and run the demo configuration
	bpsp example-panel outputs/example_panel.csv
	bpsp run configs/demo.yaml

notebook:           ## re-execute the quickstart notebook and keep its outputs (needs the notebooks extra)
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_quickstart.ipynb

docs:               ## regenerate reference pages and build the site into site/
	python tools/gen_docs.py
	mkdocs build --strict

clean:              ## remove outputs, caches and build files
	rm -rf outputs site .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
