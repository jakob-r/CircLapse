DATA ?= data/real
PRED ?=
N ?= 1000
NAME ?= nn
CKPT ?= checkpoints/finetune.pt
NOTE ?=
ifndef RUN
RUN := $(shell date +%Y-%m-%d_%H%M)_$(shell git rev-parse --short HEAD)
endif
FRONTEND_SRC := $(shell find frontend/src -type f) frontend/index.html frontend/package.json frontend/vite.config.ts

.PHONY: help install install-ml dev build studio annotate synth synth-train pretrain finetune train experiment record use detect eval render api-types test lint clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python and frontend dependencies
	uv sync
	cd frontend && npm ci

install-ml: ## Install with the neural network extra (torch, timm)
	uv sync --extra ml
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

studio: build ## Open the studio: import a folder or Apple Photos, review and export (DATA=... skips the start page)
	uv run --extra ml --extra photos --extra app circlapse \
		$(if $(filter command line,$(origin DATA)),make $(DATA),studio)

annotate: build ## Annotate images in DATA on http://127.0.0.1:8000 (PRED=predictions.json to overlay)
	uv run circlapse annotate $(DATA) $(if $(PRED),--pred $(PRED))

synth: ## Generate N synthetic images into data/synth
	uv run circlapse synth --n $(N) --out data/synth

synth-train: ## Generate the NN training (20k) and validation (1k) synthetic sets
	uv run circlapse synth --n 20000 --out data/synth --seed 0
	uv run circlapse synth --n 1000 --out data/synth_val --seed 1

pretrain: ## Pretrain the NN on data/synth -> checkpoints/pretrain.pt
	uv run --extra ml circlapse train pretrain

finetune: ## Fine-tune on data/real labels with 5-fold CV -> checkpoints/finetune.pt
	uv run --extra ml circlapse train finetune

train: pretrain finetune ## Pretrain and fine-tune the NN

experiment: ## Train into runs/RUN, make it current and add its CV scores to RESULTS.md (NOTE="what changed")
	uv run --extra ml circlapse runs start runs/$(RUN) --note "$(NOTE)"
	uv run --extra ml circlapse train pretrain --out runs/$(RUN)/pretrain.pt > runs/$(RUN)/pretrain.log 2>&1
	uv run --extra ml circlapse train finetune --init runs/$(RUN)/pretrain.pt \
		--out runs/$(RUN)/finetune.pt --cv-out runs/$(RUN)/cv.json > runs/$(RUN)/finetune.log 2>&1
	uv run circlapse runs record runs/$(RUN)
	uv run circlapse runs use runs/$(RUN)

record: ## Re-score runs/RUN against the current labels and update its row in RESULTS.md
	uv run circlapse runs record runs/$(RUN)

use: ## Make runs/RUN the current model (copies its checkpoints to checkpoints/)
	uv run circlapse runs use runs/$(RUN)

detect: ## Predict DATA with CKPT -> data/predictions/NAME.json
	uv run --extra ml circlapse detect $(DATA) --checkpoint $(CKPT) --out data/predictions/$(NAME).json

render: ## Render the hyperlapse of DATA (mp4, webm, gif) into DATA/circlapse/
	uv run --extra ml circlapse render $(DATA)

eval: ## Evaluate data/predictions/NAME.json against DATA labels
	uv run circlapse eval --gt $(DATA) --pred data/predictions/$(NAME).json

api-types: ## Regenerate frontend/src/api-types.ts from the FastAPI schema
	uv run python -c "import json, tempfile, pathlib; \
		from circlapse.studio.launcher import openapi_schema; \
		from circlapse.studio.project import Project, save_project; \
		d = pathlib.Path(tempfile.mkdtemp()); save_project(d, Project()); \
		print(json.dumps(openapi_schema(d), indent=2))" \
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
