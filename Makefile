DATASET ?= ipl

.PHONY: help install lint format test data features report pipeline docker-data

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## Editable install with dev tools
	pip install -e ".[dev]"

lint:  ## Ruff lint and format check
	ruff check . && ruff format --check .

format:  ## Auto-fix lint and format
	ruff check --fix . && ruff format .

test:  ## Run tests
	pytest

data:  ## Download and build DATASET (default ipl)
	cricket-ai data download $(DATASET)
	cricket-ai data build $(DATASET)

features:  ## Win-probability chase states for DATASET
	cricket-ai data features $(DATASET)

report:  ## Analytics reports for DATASET
	cricket-ai data report $(DATASET)

pipeline: data features report  ## Full module 1 pipeline

docker-data:  ## Build the data pipeline image
	docker build -f docker/data.Dockerfile -t cricket-ai-data .
