DATA ?= data/real
PRED ?=
N ?= 1000
FRONTEND_SRC := $(shell find frontend/src -type f) frontend/index.html frontend/package.json frontend/vite.config.ts

.PHONY: help install dev build annotate synth api-types test lint clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install Python and frontend dependencies
	uv sync
	cd frontend && npm ci

frontend/node_modules: frontend/package-lock.json
	cd frontend && npm ci
	@touch $@

dev: frontend/node_modules ## Dev servers with hot reload: UI on http://localhost:5173 (DATA=..., PRED=...)
	@trap 'kill 0' INT TERM EXIT; \
	CIRCLAPSE_DATA=$(DATA) CIRCLAPSE_PRED=$(PRED) uv run uvicorn --factory \
		circlapse.annotate.app:app_from_env --reload --reload-dir src --port 8000 & \
	(cd frontend && npm run dev) & \
	wait

frontend/dist/index.html: frontend/node_modules $(FRONTEND_SRC)
	cd frontend && npm run build

build: frontend/dist/index.html ## Build the annotator UI

annotate: build ## Annotate images in DATA on http://127.0.0.1:8000 (PRED=predictions.json to overlay)
	uv run circlapse annotate $(DATA) $(if $(PRED),--pred $(PRED))

synth: ## Generate N synthetic images into data/synth
	uv run circlapse synth --n $(N) --out data/synth

api-types: ## Regenerate frontend/src/api-types.ts from the FastAPI schema
	uv run python -c "import json, tempfile, pathlib; \
		from circlapse.annotate.app import create_app; \
		print(json.dumps(create_app(pathlib.Path(tempfile.mkdtemp())).openapi(), indent=2))" \
		> frontend/openapi.json
	cd frontend && npm run api-types

test: frontend/node_modules ## Run Python and frontend tests
	uv run pytest
	cd frontend && npm run typecheck && npm test

lint: frontend/node_modules ## Lint and type-check everything
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy
	cd frontend && npm run lint

clean: ## Remove build output and caches
	rm -rf frontend/dist .pytest_cache .mypy_cache .ruff_cache
	find data -name .thumbs -type d -prune -exec rm -rf {} + 2>/dev/null || true
