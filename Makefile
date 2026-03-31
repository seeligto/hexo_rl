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

.PHONY: test.py.fast
test.py.fast: ## Python unit tests only — skips slow self-play integration (~20s)
	$(PY) -m pytest -q tests --ignore=tests/test_phase1_exit_criteria.py

.PHONY: test.focus
test.focus: ## Run focused buffer/inference/pool smoke tests
	$(PY) -m pytest -q tests/test_rust_replay_buffer.py tests/test_inference_server.py tests/test_worker_pool.py tests/test_benchmark_smoke.py

.PHONY: test.rust
test.rust: ## Run Rust tests
	cargo test

.PHONY: test.all
test.all: test.rust test.py ## Run rust + python tests

.PHONY: ci
ci: test.all bench.quick ## Full pre-push gate: all tests + quick benchmark

.PHONY: bench.quick
bench.quick: ## Fast sanity benchmark — did I break anything? (~30s)
	$(PY) scripts/benchmark.py --config $(CONFIG_LITE) --no-compile --quick --mcts-sims 2000 --pool-workers $(N_CORES) --pool-duration 10

.PHONY: bench.lite
bench.lite: ## Quick local benchmark pass
	$(PY) scripts/benchmark.py --config $(CONFIG_LITE) --no-compile --mcts-sims 2000 --pool-workers $(N_CORES) --pool-duration 10

N_CORES ?= $(shell python3 -c "import os; print(os.cpu_count() or 4)")

.PHONY: bench.full
bench.full: ## Higher-confidence benchmark pass
	$(PY) scripts/benchmark.py --config $(CONFIG_FULL) --mcts-sims 50000 --pool-workers $(N_CORES) --pool-duration 60

.PHONY: bench.stress
bench.stress: ## Heavy stress test (5 mins, high sims)
	$(PY) scripts/benchmark.py --config $(CONFIG_FULL) --mcts-sims 100000 --pool-workers $(N_CORES) --pool-duration 300 --mcts-search-sims 800

.PHONY: bench.mcts
bench.mcts: ## Dedicated Rust MCTS micro-benchmark
	$(PY) scripts/benchmark_mcts.py

DASHBOARD_PORT ?= 5001
DASHBOARD_URL  ?= http://localhost:$(DASHBOARD_PORT)

.PHONY: dashboard
dashboard: ## Start the web dashboard server (DASHBOARD_PORT=5001 by default)
	$(PY) dashboard.py $(DASHBOARD_PORT)

.PHONY: train.lite
train.lite: ## Fast debug training — short run, no dashboard
	$(PY) scripts/train.py --config $(CONFIG_LITE) --iterations 100 --no-dashboard --no-compile

.PHONY: train.lite.dashboard
train.lite.dashboard: ## Fast debug training with web dashboard (start dashboard separately first)
	$(PY) scripts/train.py --config $(CONFIG_LITE) --iterations 100 --no-compile \
	    --web-dashboard --web-dashboard-url $(DASHBOARD_URL)

.PHONY: train.full
train.full: ## Standard training from bootstrap checkpoint, no dashboard
	$(PY) scripts/train.py --config $(CONFIG_FULL) --checkpoint $(CHECKPOINT_BOOTSTRAP)

.PHONY: train.full.dashboard
train.full.dashboard: ## Standard training with web dashboard (start dashboard separately first)
	$(PY) scripts/train.py --config $(CONFIG_FULL) --checkpoint $(CHECKPOINT_BOOTSTRAP) \
	    --web-dashboard --web-dashboard-url $(DASHBOARD_URL)

.PHONY: train.multi
train.multi: ## Multi-hour training profile from bootstrap checkpoint
	$(PY) scripts/train.py --config $(CONFIG_MULTI) --checkpoint $(CHECKPOINT_BOOTSTRAP)

.PHONY: train.multi.dashboard
train.multi.dashboard: ## Multi-hour training with web dashboard
	$(PY) scripts/train.py --config $(CONFIG_MULTI) --checkpoint $(CHECKPOINT_BOOTSTRAP) \
	    --web-dashboard --web-dashboard-url $(DASHBOARD_URL)

.PHONY: train.resume
train.resume: ## Resume multi-hour training from latest checkpoint
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	$(PY) scripts/train.py --config $(CONFIG_MULTI) --checkpoint $(CHECKPOINT_LATEST)

.PHONY: train.resume.dashboard
train.resume.dashboard: ## Resume training from latest checkpoint with web dashboard
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	$(PY) scripts/train.py --config $(CONFIG_MULTI) --checkpoint $(CHECKPOINT_LATEST) \
	    --web-dashboard --web-dashboard-url $(DASHBOARD_URL)

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

.PHONY: pretrain.lite
pretrain.lite: ## Short bootstrap pretrain
	$(PY) -m python.bootstrap.pretrain --epochs 5 --use-cache

.PHONY: pretrain.full
pretrain.full: ## Full bootstrap pretrain
	$(PY) -m python.bootstrap.pretrain --epochs 15 --force-regenerate

