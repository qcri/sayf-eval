.PHONY: install style quality test build clean

install:
	python -m pip install -e ".[dev]"

# Auto-fix formatting + lint.
style:
	ruff format src tests
	ruff check --fix src tests

# CI gate: fail if anything is unformatted or lint errors remain.
quality:
	ruff format --check src tests
	ruff check src tests

test:
	pytest

build:
	python -m build

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
