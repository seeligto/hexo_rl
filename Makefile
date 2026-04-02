SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
MATURIN ?= .venv/bin/maturin

# Override config files — applied on top of base configs (model+training+selfplay).
# Omit --config to use base configs unmodified (production settings).
CONFIG_LITE ?= configs/fast_debug.yaml
CONFIG_MULTI ?= configs/long_run_balanced.yaml

CHECKPOINT_BOOTSTRAP ?= checkpoints/bootstrap_model.pt
CHECKPOINT_LATEST ?= $(shell ls -1 checkpoints/checkpoint_*.pt 2>/dev/null | tail -n 1)

SEALBOT_N ?= 100
SEALBOT_TIME ?= 0.03
SEALBOT_SIMS ?= 128


.PHONY: help
help: ## Show all useful commands
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' Makefile | sort | awk 'BEGIN {FS = ":.*## "; printf "\nUsage: make <target>\n\nTargets:\n"} {printf "  %-30s %s\n", $$1, $$2}'; echo

.PHONY: install
install: ## Full first-time setup: venv → deps → submodules → SealBot → engine → test.all
	@echo "==> Creating virtualenv..."
	python3 -m venv .venv
	@echo "==> Upgrading pip and installing maturin + pybind11..."
	$(PIP) install --upgrade pip maturin pybind11
	@echo "==> Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo "==> Initialising git submodules..."
	git submodule update --init --recursive
	@echo "==> Building SealBot C++ extensions..."
	cd vendor/bots/sealbot/best    && $(PY) setup.py build_ext --inplace --quiet
	cd vendor/bots/sealbot/current && $(PY) setup.py build_ext --inplace --quiet
	@echo "==> Building engine Rust extension..."
	$(MATURIN) develop --release -m engine/Cargo.toml
	@echo "==> Verifying environment..."
	$(MAKE) env.check
	@echo "==> Running full test suite..."
	$(MAKE) test.all
	@echo ""
	@echo "Install complete. Run 'make corpus.scrape' to fetch the latest games."

.PHONY: env.check
env.check: ## Check virtualenv/python/engine availability
	@test -x "$(PY)" || (echo "Missing $(PY). Create venv first." && exit 1)
	@$(PY) -c "from engine import Board, MCTSTree; print('engine ok')"

.PHONY: deps.install
deps.install: ## Install python deps into .venv
	$(PIP) install -r requirements.txt

.PHONY: native.build
native.build: ## Build/install Rust extension via maturin (LTO + native CPU — see .cargo/config.toml)
	$(MATURIN) develop --release -m engine/Cargo.toml

.PHONY: clean
clean: ## Remove all Rust build artifacts and Python caches
	cargo clean
	rm -rf .venv/lib/python*/site-packages/engine*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete. Run 'make native.build' to rebuild."

.PHONY: rebuild
rebuild: clean native.build ## Full clean + optimized rebuild
	@echo "Rebuild complete."

.PHONY: test.py
test.py: ## Run python test suite
	$(PY) -m pytest -q tests

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
bench.lite: ## Quick benchmark (n=3, no CPU pin, warm-up)
	$(PY) scripts/benchmark.py --config $(CONFIG_LITE) --no-compile --mcts-sims 2000 --pool-workers $(N_CORES) --pool-duration 10 --mode lite

N_CORES ?= $(shell python3 -c "import os; print(os.cpu_count() or 4)")

.PHONY: bench.full
bench.full: ## Higher-confidence benchmark (n=5, CPU pin attempted, warm-up)
	$(PY) scripts/benchmark.py --mcts-sims 50000 --pool-workers $(N_CORES) --pool-duration 60 --mode full

.PHONY: bench.stress
bench.stress: ## Heavy stress test (n=10, CPU pin required, 5-min pool runs)
	$(PY) scripts/benchmark.py --mcts-sims 100000 --pool-workers $(N_CORES) --pool-duration 300 --mcts-search-sims 800 --mode stress

.PHONY: bench.baseline
bench.baseline: ## Run bench.full and save as dated baseline report
	mkdir -p reports/benchmarks
	$(PY) scripts/benchmark.py --mcts-sims 50000 --pool-workers $(N_CORES) --pool-duration 60 --mode full 2>&1 | tee reports/benchmarks/$$(date +%Y-%m-%d)_baseline.log

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
	$(PY) scripts/train.py --checkpoint $(CHECKPOINT_BOOTSTRAP)

.PHONY: train.full.dashboard
train.full.dashboard: ## Standard training with web dashboard (start dashboard separately first)
	$(PY) scripts/train.py --checkpoint $(CHECKPOINT_BOOTSTRAP) \
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
	$(PY) scripts/train.py --checkpoint $(CHECKPOINT_LATEST)

.PHONY: train.resume.dashboard
train.resume.dashboard: ## Resume training from latest checkpoint with web dashboard
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	$(PY) scripts/train.py --checkpoint $(CHECKPOINT_LATEST) \
	    --web-dashboard --web-dashboard-url $(DASHBOARD_URL)

.PHONY: plot.train.latest
plot.train.latest: ## Plot latest training log
	$(PY) scripts/plot_training.py --latest

.PHONY: plot.train
plot.train: ## Plot a specific training log (LOG=logs/xxxx.jsonl)
	@test -n "$(LOG)" || (echo "Usage: make plot.train LOG=logs/<file>.jsonl" && exit 1)
	$(PY) scripts/plot_training.py --log-file "$(LOG)"

.PHONY: plot.sealbot.latest
plot.sealbot.latest: ## Plot latest SealBot eval log
	$(PY) scripts/plot_sealbot_eval.py --latest

.PHONY: plot.sealbot.all
plot.sealbot.all: ## Plot aggregated SealBot eval trend
	$(PY) scripts/plot_sealbot_eval.py --all

.PHONY: eval.sealbot.latest
eval.sealbot.latest: ## Evaluate latest checkpoint vs SealBot
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	$(PY) scripts/eval_vs_sealbot.py --checkpoint $(CHECKPOINT_LATEST) --n-games $(SEALBOT_N) --time-limit $(SEALBOT_TIME) --model-sims $(SEALBOT_SIMS)

.PHONY: eval.sealbot
eval.sealbot: ## Evaluate specific checkpoint vs SealBot (CKPT=...)
	@test -n "$(CKPT)" || (echo "Usage: make eval.sealbot CKPT=checkpoints/checkpoint_XXXXXXXX.pt" && exit 1)
	$(PY) scripts/eval_vs_sealbot.py --checkpoint "$(CKPT)" --n-games $(SEALBOT_N) --time-limit $(SEALBOT_TIME) --model-sims $(SEALBOT_SIMS)

.PHONY: eval.sealbot.quick
eval.sealbot.quick: ## Quick SealBot eval (10 games)
	$(MAKE) eval.sealbot.latest SEALBOT_N=10 SEALBOT_TIME=0.01 SEALBOT_SIMS=64

.PHONY: eval.sealbot.full
eval.sealbot.full: ## Full SealBot eval (100 games)
	$(MAKE) eval.sealbot.latest SEALBOT_N=100 SEALBOT_TIME=0.03 SEALBOT_SIMS=128

.PHONY: pretrain
pretrain: ## Bootstrap pretrain (5 epochs, default)
	$(PY) -m hexo_rl.bootstrap.pretrain --epochs 5

.PHONY: pretrain.lite
pretrain.lite: ## Bootstrap pretrain smoke test (100 steps)
	$(PY) -m hexo_rl.bootstrap.pretrain --steps 100

.PHONY: pretrain.full
pretrain.full: ## Full bootstrap pretrain (15 epochs)
	$(PY) -m hexo_rl.bootstrap.pretrain --epochs 15

# ── Corpus generation ────────────────────────────────────────────────────────

CORPUS_DEPTH4_N ?= 2000
CORPUS_DEPTH6_N ?= 1000

.PHONY: corpus.scrape
corpus.scrape: ## Scrape latest human games from [site-redacted] and update manifest
	bash scripts/scrape_daily.sh

.PHONY: corpus.d4
corpus.d4: ## Generate SealBot depth-4 self-play corpus (CORPUS_DEPTH4_N=2000)
	$(PY) -m hexo_rl.bootstrap.generate_corpus --bot sealbot --depth 4 --n-games $(CORPUS_DEPTH4_N) --output data/corpus/bot_games/sealbot_d4

.PHONY: corpus.d6
corpus.d6: ## Generate SealBot depth-6 self-play corpus (CORPUS_DEPTH6_N=1000)
	$(PY) -m hexo_rl.bootstrap.generate_corpus --bot sealbot --depth 6 --n-games $(CORPUS_DEPTH6_N) --output data/corpus/bot_games/sealbot_d6

.PHONY: corpus.all
corpus.all: corpus.d4 corpus.d6 corpus.manifest ## Generate both d4 and d6 corpora

.PHONY: corpus.manifest
corpus.manifest: ## Update data/corpus/manifest.json (scans human + bot dirs)
	$(PY) scripts/update_manifest.py

.PHONY: corpus.analysis
corpus.analysis: corpus.manifest ## Run corpus analysis on human + bot games
	$(PY) -m hexo_rl.bootstrap.corpus_analysis --include-bot-games

.PHONY: corpus.npz
corpus.npz: ## Export corpus to data/bootstrap_corpus.npz for mixed training
	$(PY) scripts/export_corpus_npz.py

