# Validation entry point: `make check` runs format -> lint -> typecheck -> test.
#
# CHECKED lists the paths held to the full pipeline. Most of the tree predates
# it (not black/isort/flake8 clean), so paths are added as they are cleaned up;
# `make test-all` runs the whole unittest suite.

PYTHON ?= python
CHECKED := src/lisatools/search tests/test_emri_search.py
TESTS := tests.test_emri_search

.PHONY: check format lint typecheck test test-all

check: format lint typecheck test
	@echo "All checks passed!"

format:
	black $(CHECKED)
	isort $(CHECKED)

lint:
	flake8 --max-line-length=100 --extend-ignore=E203,W503 $(CHECKED)

typecheck:
	mypy --ignore-missing-imports --follow-imports=silent $(filter src/%,$(CHECKED))

test:
	$(PYTHON) -m unittest $(TESTS)

test-all:
	$(PYTHON) -m unittest discover
