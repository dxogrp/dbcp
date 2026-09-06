# Colors for pretty output
BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: venv
venv:
	@printf "$(BLUE)Creating virtual environment...$(RESET)\n"
	@uv venv --python 3.12

.PHONY: install
install: venv ## install all dependencies using uv
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	@uv sync --frozen

.PHONY: test
test: install ## run tests
	uv run pytest tests

.PHONY: sync-examples
sync-examples: ## install locked example dependencies
	@printf "$(BLUE)Installing example dependencies...$(RESET)\n"
	@uv sync --frozen --group dev --group examples

.PHONY: check-examples
check-examples: sync-examples ## check Marimo examples
	@printf "$(BLUE)Checking Marimo examples...$(RESET)\n"
	@uv run --frozen --group examples marimo check --strict examples

.PHONY: marimo
marimo: sync-examples ## start a Marimo server
	@printf "$(BLUE)Start Marimo server...$(RESET)\n"
	@uv run --frozen --group examples marimo edit examples

.PHONY: fmt
fmt: venv ## Run code formatting and linting
	@printf "$(BLUE)Running formatters and linters...$(RESET)\n"
	@uv pip install pre-commit
	@uv run pre-commit install
	@uv run pre-commit run --all-files

.PHONY: clean
clean: ## clean generated files and directories
	@printf "$(BLUE)Cleaning project...$(RESET)\n"
	@git clean -d -X -f

.PHONY: help
help: ## display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
