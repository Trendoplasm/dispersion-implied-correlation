# Common tasks. `make help` lists them.
#
# Everything runs through uv, which manages both the Python version and the dependencies.
# See README.md for a plain-pip alternative.

PYTHON_VERSION := 3.13
# --extra dev so `make test` works on a fresh clone without `make setup` having been run
# first; uv resolves and installs what is missing rather than failing on an absent pytest.
RUN := uv run --extra dev --python $(PYTHON_VERSION)

.DEFAULT_GOAL := help
.PHONY: help setup data lint format typecheck test test-fast reproduce verify clean

help:  ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the environment and install the package with dev tools
	uv python install $(PYTHON_VERSION)
	uv sync --extra dev

data:  ## Download the implied-volatility and price histories
	$(RUN) python scripts/fetch_cboe_data.py
	$(RUN) python scripts/fetch_price_data.py

lint:  ## Check formatting and lint rules
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## Apply formatting and safe lint fixes
	$(RUN) ruff check . --fix
	$(RUN) ruff format .

typecheck:  ## Run static type checking
	$(RUN) mypy

test:  ## Run the full test suite
	$(RUN) pytest

test-fast:  ## Run only the tests that need no market data
	$(RUN) pytest -m "not golden"

reproduce:  ## Re-run the study and write to outputs/
	$(RUN) dispersion --output-dir outputs

verify:  ## Reproduce into a scratch directory and compare against outputs/
	$(RUN) dispersion --output-dir .verify --no-plots --quiet
	$(RUN) python -c "from pathlib import Path; from dispersion.verify import compare_output_dirs; \
r = compare_output_dirs(Path('outputs'), Path('.verify')); \
print(f'largest relative difference: {r.max_relative_difference:.2e} ({r.max_difference_field})'); \
print('note: columns whose correct value is zero (fit residuals) are compared on an absolute floor, so a large relative figure above is expected and not a failure'); \
print('MATCH' if r.matches else chr(10).join(r.discrepancies[:10])); \
raise SystemExit(0 if r.matches else 1)"

clean:  ## Remove caches and scratch reproduction directories
	rm -rf .pytest_cache .ruff_cache .mypy_cache .verify my_reproduction quick_check
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
