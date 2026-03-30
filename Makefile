SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
MATURIN ?= .venv/bin/maturin

CONFIG_LITE ?= configs/fast_debug.yaml
CONFIG_FULL ?= configs/default.yaml
CONFIG_MULTI ?= configs/long_run_balanced.yaml

CHECKPOINT_BOOTSTRAP ?= checkpoints/bootstrap_model.pt
CHECKPOINT_LATEST ?= $(shell ls -1 checkpoints/checkpoint_*.pt 2>/dev/null | tail -n 1)

RAMORA_N ?= 100
RAMORA_TIME ?= 0.03
RAMORA_SIMS ?= 128

BOT_GAMES_LITE ?= 100
BOT_GAMES_FULL ?= 1000
HUMAN_PAGES_LITE ?= 5
HUMAN_PAGES_FULL ?= 50

.PHONY: help
help: ## Show all useful commands
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' Makefile | sort | awk 'BEGIN {FS = ":.*## "; printf "\nUsage: make <target>\n\nTargets:\n"} {printf "  %-30s %s\n", $$1, $$2}'; echo

.PHONY: env.check
env.check: ## Check virtualenv/python/native_core availability
	@test -x "$(PY)" || (echo "Missing $(PY). Create venv first." && exit 1)
	@$(PY) -c "from native_core import Board, MCTSTree; print('native_core ok')"

.PHONY: deps.install
deps.install: ## Install python deps into .venv
	$(PIP) install -r requirements.txt

.PHONY: native.build
native.build: ## Build/install Rust extension via maturin
	$(MATURIN) develop --release -m native_core/Cargo.toml

.PHONY: test.py
test.py: ## Run python test suite
	$(PY) -m pytest -q tests

.PHONY: test.focus
test.focus: ## Run focused inference/pool benchmark smoke tests
	$(PY) -m pytest -q tests/test_inference_server.py tests/test_worker_pool.py tests/test_benchmark_smoke.py

.PHONY: test.rust
test.rust: ## Run Rust tests
	cargo test

.PHONY: test.all
test.all: test.rust test.py ## Run rust + python tests

.PHONY: bench.lite
bench.lite: ## Quick local benchmark pass
	$(PY) scripts/benchmark.py --config $(CONFIG_LITE) --no-compile --mcts-sims 2000 --pool-workers $(N_CORES) --pool-duration 10

N_CORES ?= $(shell python3 -c "import os; print(os.cpu_count() or 4)")

.PHONY: bench.full
bench.full: ## Higher-confidence benchmark pass
	$(PY) scripts/benchmark.py --config $(CONFIG_FULL) --mcts-sims 50000 --pool-workers $(N_CORES) --pool-duration 30

.PHONY: bench.stress
bench.stress: ## Heavy stress test (5 mins, high sims)
	$(PY) scripts/benchmark.py --config $(CONFIG_FULL) --mcts-sims 100000 --pool-workers $(N_CORES) --pool-duration 300 --mcts-search-sims 800

.PHONY: bench.mcts
bench.mcts: ## Dedicated Rust MCTS micro-benchmark
	$(PY) scripts/benchmark_mcts.py

.PHONY: train.lite
train.lite: ## Fast debug training (short run)
	$(PY) scripts/train.py --config $(CONFIG_LITE) --iterations 100 --no-dashboard --no-compile

.PHONY: train.full
train.full: ## Standard training from bootstrap checkpoint
	$(PY) scripts/train.py --config $(CONFIG_FULL) --checkpoint $(CHECKPOINT_BOOTSTRAP)

.PHONY: train.multi
train.multi: ## Multi-hour training profile from bootstrap checkpoint
	$(PY) scripts/train.py --config $(CONFIG_MULTI) --checkpoint $(CHECKPOINT_BOOTSTRAP)

.PHONY: train.resume
train.resume: ## Resume multi-hour training from latest checkpoint
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	$(PY) scripts/train.py --config $(CONFIG_MULTI) --checkpoint $(CHECKPOINT_LATEST)

.PHONY: plot.train.latest
plot.train.latest: ## Plot latest training log
	$(PY) scripts/plot_training.py --latest

.PHONY: plot.train
plot.train: ## Plot a specific training log (LOG=logs/xxxx.jsonl)
	@test -n "$(LOG)" || (echo "Usage: make plot.train LOG=logs/<file>.jsonl" && exit 1)
	$(PY) scripts/plot_training.py --log-file "$(LOG)"

.PHONY: plot.ramora.latest
plot.ramora.latest: ## Plot latest Ramora eval log
	$(PY) scripts/plot_ramora_eval.py --latest

.PHONY: plot.ramora.all
plot.ramora.all: ## Plot aggregated Ramora eval trend
	$(PY) scripts/plot_ramora_eval.py --all

.PHONY: eval.ramora.latest
eval.ramora.latest: ## Evaluate latest checkpoint vs Ramora
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	$(PY) scripts/eval_vs_ramora.py --checkpoint $(CHECKPOINT_LATEST) --n-games $(RAMORA_N) --time-limit $(RAMORA_TIME) --model-sims $(RAMORA_SIMS)

.PHONY: eval.ramora
eval.ramora: ## Evaluate specific checkpoint vs Ramora (CKPT=...)
	@test -n "$(CKPT)" || (echo "Usage: make eval.ramora CKPT=checkpoints/checkpoint_XXXXXXXX.pt" && exit 1)
	$(PY) scripts/eval_vs_ramora.py --checkpoint "$(CKPT)" --n-games $(RAMORA_N) --time-limit $(RAMORA_TIME) --model-sims $(RAMORA_SIMS)

.PHONY: eval.ramora.quick
eval.ramora.quick: ## Quick Ramora eval (10 games)
	$(MAKE) eval.ramora.latest RAMORA_N=10 RAMORA_TIME=0.01 RAMORA_SIMS=64

.PHONY: eval.ramora.full
eval.ramora.full: ## Full Ramora eval (100 games)
	$(MAKE) eval.ramora.latest RAMORA_N=100 RAMORA_TIME=0.03 RAMORA_SIMS=128

.PHONY: corpus.lite
corpus.lite: ## Build small bootstrap corpus (bot + human)
	$(PY) python/bootstrap/generate_corpus.py --bot-games $(BOT_GAMES_LITE) --human-pages $(HUMAN_PAGES_LITE)

.PHONY: corpus.full
corpus.full: ## Build larger bootstrap corpus (bot + human)
	$(PY) python/bootstrap/generate_corpus.py --bot-games $(BOT_GAMES_FULL) --human-pages $(HUMAN_PAGES_FULL)

.PHONY: corpus.fresh
corpus.fresh: ## Rebuild corpus cache from scratch
	$(PY) python/bootstrap/generate_corpus.py --bot-games $(BOT_GAMES_FULL) --human-pages $(HUMAN_PAGES_FULL) --force-regenerate

.PHONY: pretrain.lite
pretrain.lite: ## Short bootstrap pretrain
	$(PY) -m python.bootstrap.pretrain --epochs 5 --use-cache

.PHONY: pretrain.full
pretrain.full: ## Full bootstrap pretrain
	$(PY) -m python.bootstrap.pretrain --epochs 15 --force-regenerate

.PHONY: bootstrap.lite
bootstrap.lite: corpus.lite pretrain.lite ## Lite corpus + pretrain pipeline

.PHONY: bootstrap.full
bootstrap.full: corpus.full pretrain.full ## Full corpus + pretrain pipeline
