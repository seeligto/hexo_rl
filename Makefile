SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
MATURIN ?= .venv/bin/maturin

CHECKPOINT_BOOTSTRAP ?= checkpoints/bootstrap_model.pt
CHECKPOINT_LATEST ?= $(shell ls -1 checkpoints/checkpoint_*.pt 2>/dev/null | tail -n 1)
PRETRAIN_CKPT ?= $(shell ls -1 checkpoints/pretrain/pretrain_*.pt 2>/dev/null | tail -n 1)

# Named variant from configs/variants/ — deep-merged on top of selfplay.yaml.
# Usage: make train VARIANT=gumbel_full   (or gumbel_targets, baseline_puct)
VARIANT ?=
VARIANT_FLAG = $(if $(VARIANT),--variant $(VARIANT),)

# Set DASHBOARD=0 to disable the web+terminal dashboard (e.g. for scripted runs)
DASHBOARD ?= 1
_NODASH_FLAG = $(if $(filter 0,$(DASHBOARD)),--no-dashboard,)

# Eval parameters
CKPT ?= $(CHECKPOINT_LATEST)
N_GAMES ?= 100
THINK_TIME ?= 0.5
SIMS ?= 128

# Corpus parameters
CORPUS_N ?= 2500
MAX_POSITIONS ?= 50000

N_CORES ?= $(shell $(PY) -c "import os; print(os.cpu_count() or 4)")


.PHONY: help
help: ## Show all useful commands
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' Makefile | sort | awk 'BEGIN {FS = ":.*## "; printf "\nUsage: make <target>\n\nTargets:\n"} {printf "  %-30s %s\n", $$1, $$2}'; echo


# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Full first-time setup: env check → deps → engine → artifacts → test
	@bash scripts/install.sh

.PHONY: build
build: ## Build/install Rust extension via maturin (LTO + native CPU)
	$(MATURIN) develop --release -m engine/Cargo.toml

.PHONY: clean
clean: ## Remove all Rust build artifacts and Python caches
	cargo clean
	rm -rf .venv/lib/python*/site-packages/engine*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete. Run 'make build' to rebuild."

.PHONY: rebuild
rebuild: clean build ## Full clean + optimized rebuild
	@echo "Rebuild complete."


# ── Tests ─────────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run Rust + Python tests (excludes integration marker)
	cargo test
	$(PY) -m pytest -q -m "not slow and not integration" tests

.PHONY: test.slow
test.slow: ## Run slow/integration Python tests (~2-5 min)
	$(PY) -m pytest -v -m "integration" tests/test_train_lifecycle.py


# ── Benchmarks ────────────────────────────────────────────────────────────────

.PHONY: bench
bench: ## Higher-confidence benchmark (n=5, warm-up; full Phase 4.5 gate methodology)
	$(PY) scripts/benchmark.py --mcts-sims 50000 --pool-workers $(N_CORES) --pool-duration 60


# ── Training ──────────────────────────────────────────────────────────────────

.PHONY: train
train: ## Self-play RL from bootstrap checkpoint (VARIANT=..., DASHBOARD=0 to disable)
	MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(CHECKPOINT_BOOTSTRAP) $(_NODASH_FLAG) $(VARIANT_FLAG)

.PHONY: train.resume
train.resume: ## Resume training from latest checkpoint (VARIANT= supported)
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(CHECKPOINT_LATEST) $(_NODASH_FLAG) $(VARIANT_FLAG)

.PHONY: train.bg
train.bg: ## Self-play RL from bootstrap checkpoint, background (VARIANT= supported)
	@mkdir -p logs
	@nohup env MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(CHECKPOINT_BOOTSTRAP) $(VARIANT_FLAG) \
		> logs/train_$$(date +%Y%m%d_%H%M%S).log 2>&1 & \
		echo $$! > logs/train.pid; \
		echo "Training started (PID $$(cat logs/train.pid))"
	@echo "Logs: logs/train_*.log"
	@echo "Dashboard: http://localhost:5001"

.PHONY: train.stop
train.stop: ## Stop background training
	@stopped=0; \
	if [ -f logs/train.pid ]; then \
		PID=$$(cat logs/train.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID && echo "Stopped PID $$PID (from train.pid)"; \
			stopped=1; \
		else \
			echo "Stale train.pid (PID $$PID not running), removing"; \
		fi; \
		rm -f logs/train.pid; \
	fi; \
	pkill -f "scripts/train.py" 2>/dev/null && { echo "Killed train.py processes via pkill"; stopped=1; } || true; \
	if [ $$stopped -eq 0 ]; then echo "No training process found"; fi

.PHONY: train.status
train.status: ## Check background training status
	@if [ -f logs/train.pid ]; then \
		PID=$$(cat logs/train.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Training running (PID $$PID)"; \
			tail -5 $$(ls -t logs/train_*.log 2>/dev/null | head -1); \
		else \
			echo "PID $$PID not running (stale train.pid)"; \
		fi \
	else \
		echo "Not running (no train.pid)"; \
	fi

.PHONY: train.smoke
train.smoke: ## 20-step smoke test to verify training end-to-end
	@if [ -z "$(PRETRAIN_CKPT)" ]; then \
	    echo "Error: No pretrain checkpoint found. Run 'make pretrain' first."; \
	    exit 1; \
	fi
	@echo "Using checkpoint: $(PRETRAIN_CKPT)"
	MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(PRETRAIN_CKPT) --iterations 20

.PHONY: pretrain
pretrain: ## Full bootstrap pretrain (15 epochs)
	MALLOC_ARENA_MAX=2 $(PY) -m hexo_rl.bootstrap.pretrain --epochs 15


# ── Eval ──────────────────────────────────────────────────────────────────────

.PHONY: eval
eval: ## Evaluate checkpoint vs SealBot (CKPT=latest, N_GAMES=100, THINK_TIME=0.5, SIMS=128)
	@test -n "$(CKPT)" || (echo "No checkpoint found. Pass CKPT= or generate one first." && exit 1)
	$(PY) scripts/eval_vs_sealbot.py --checkpoint "$(CKPT)" --n-games $(N_GAMES) --time-limit $(THINK_TIME) --model-sims $(SIMS)


# ── Corpus ────────────────────────────────────────────────────────────────────

.PHONY: corpus.fetch
corpus.fetch: ## Scrape human games + generate SealBot corpus + update manifest (CORPUS_N=2500, THINK_TIME=0.5)
	bash scripts/scrape_daily.sh
	$(PY) -m hexo_rl.bootstrap.generate_corpus --bot sealbot --time-limit $(THINK_TIME) --n-games $(CORPUS_N) --output data/corpus/bot_games/sealbot_strong
	$(PY) scripts/update_manifest.py

.PHONY: corpus.export
corpus.export: ## Export optimized NPZ corpus for buffer prefill (MAX_POSITIONS=50000)
	$(PY) scripts/export_corpus_npz.py --max-positions $(MAX_POSITIONS) --no-compress


# ── Dashboard ─────────────────────────────────────────────────────────────────

.PHONY: dash.open
dash.open: ## Open web dashboard in browser
	@echo "Opening http://localhost:5001"
	@$(PY) -c "import webbrowser; webbrowser.open('http://localhost:5001')" \
		|| echo "Open manually: http://localhost:5001"
