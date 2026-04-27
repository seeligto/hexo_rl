SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PYTHON ?= python3
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

# Dashboard is OFF by default for training — run `make dashboard` in a
# separate terminal to attach. Set DASHBOARD=1 on the train target to force
# the in-process dashboard (not recommended: werkzeug threaded mode produces
# "Session is disconnected" traceback storms under websocket backpressure).
DASHBOARD ?= 0
_NODASH_FLAG = $(if $(filter 1,$(DASHBOARD)),,--no-dashboard)

# Eval parameters
CKPT ?= $(CHECKPOINT_LATEST)
N_GAMES ?= 100
THINK_TIME ?= 0.5
SIMS ?= 128

# Corpus parameters
CORPUS_N ?= 2500
MAX_POSITIONS ?= 50000

N_CORES ?= $(shell $(PY) -c "import os; print(os.cpu_count() or 4)")
BENCH_WORKERS ?= $(shell $(PY) -c "import os; print(max(1, (os.cpu_count() or 4) - 2))")


.PHONY: help
help: ## Show all useful commands
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' Makefile | sort | awk 'BEGIN {FS = ":.*## "; printf "\nUsage: make <target>\n\nTargets:\n"} {printf "  %-30s %s\n", $$1, $$2}'; echo


# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Full first-time setup: env check → deps → engine → artifacts → test
	@PYTHON=$(PYTHON) bash scripts/install.sh

.PHONY: build
build: ## Build/install Rust extension via maturin (LTO + native CPU)
	env -u CONDA_PREFIX VIRTUAL_ENV=$(CURDIR)/.venv $(MATURIN) develop --release -m engine/Cargo.toml

.PHONY: clean
clean: ## Remove all Rust build artifacts and Python caches
	cargo clean
	rm -rf .venv/lib/python*/site-packages/engine .venv/lib/python*/site-packages/engine-*.dist-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete. Run 'make build' to rebuild."

.PHONY: rebuild
rebuild: clean build ## Full clean + optimized rebuild
	@echo "Rebuild complete."


# ── Tests ─────────────────────────────────────────────────────────────────────

.PHONY: test test.rust test.py
test: test.rust test.py ## Run Rust + Python tests (excludes integration marker)

test.rust: ## Run Rust tests (cargo test)
	LD_LIBRARY_PATH=$$($(PY) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$$LD_LIBRARY_PATH cargo test

test.py: ## Run Python tests (excludes slow/integration markers)
	$(PY) -m pytest -q -m "not slow and not integration" tests

.PHONY: test.slow
test.slow: ## Run slow/integration Python tests (~2-5 min)
	$(PY) -m pytest -v -m "integration" tests/test_train_lifecycle.py


# ── Benchmarks ────────────────────────────────────────────────────────────────

.PHONY: bench
bench: ## Phase 4.5 gate (n=5, 90s warmup; compile OFF — matches production training)
	$(PY) scripts/benchmark.py --mcts-sims 50000 --pool-workers $(BENCH_WORKERS) \
		--pool-duration 120 --no-compile

.PHONY: bench.compile
bench.compile: ## Engineering datum: same as bench but with torch.compile ON (NN-isolated only realistic — selfplay path is dispatch-bound, see §124)
	$(PY) scripts/benchmark.py --mcts-sims 50000 --pool-workers $(BENCH_WORKERS) --pool-duration 120

.PHONY: bench.fast
bench.fast: ## Quick benchmark — compile off, n=3, 60s pool (cold-cache friendly)
	$(PY) scripts/benchmark.py --mcts-sims 50000 --pool-workers $(BENCH_WORKERS) \
		--pool-duration 60 --no-compile --n-runs 3 --pool-warmup 30


# ── Sweep harness (knob registry, hardware-agnostic — §126) ───────────────────
# Budget formula: per_cell_s = pool_duration × n_runs + warmup + 60.
# Current registry: 27 cells (n_workers=10, inf_batch=6, wait_ms=3, burst=8).
# sweep: 27 × (90×5+90+60)/60 ≈ 270 min.  sweep.long: 27 × (180×5+90+60)/60 ≈ 472 min.
# The `.long` variant uses 180 s cells per §124/§125 stable methodology.

.PHONY: sweep.detect
sweep.detect: ## Detect host CPU/GPU/VRAM; write reports/sweeps/detected_host.json
	bash scripts/sweep.sh detect

.PHONY: sweep
sweep: ## Full registry sweep (90s cells, ~270 min / 27 cells). SWEEP_ARGS=...
	bash scripts/sweep.sh run --max-minutes 300 $(SWEEP_ARGS)

.PHONY: sweep.long
sweep.long: ## Full sweep with §124/§125 stable methodology (180s cells, ~472 min / 27 cells).
	bash scripts/sweep.sh run --pool-duration 180 --max-minutes 500 $(SWEEP_ARGS)

.PHONY: sweep.fast
sweep.fast: ## Quick sweep — short cells, single knob (KNOB=n_workers default). MAX_MIN= for budget
	bash scripts/sweep.sh run --knobs $(or $(KNOB),n_workers) --pool-duration 30 \
		--n-runs 2 --warmup 15 --max-minutes $(or $(MAX_MIN),30) $(SWEEP_ARGS)

.PHONY: sweep.workers
sweep.workers: ## Sweep n_workers only (ternary, 90s cells). MAX_MIN= for budget
	bash scripts/sweep.sh run --knobs n_workers --max-minutes $(or $(MAX_MIN),60) $(SWEEP_ARGS)

.PHONY: sweep.workers.long
sweep.workers.long: ## n_workers ternary with 180s stable cells. MAX_MIN= for budget
	bash scripts/sweep.sh run --knobs n_workers --pool-duration 180 \
		--max-minutes $(or $(MAX_MIN),120) $(SWEEP_ARGS)

.PHONY: sweep.dryrun
sweep.dryrun: ## Validate harness orchestration with synthetic eval (no bench, no GPU)
	bash scripts/sweep.sh run --dry-run --max-minutes 240 $(SWEEP_ARGS)


# ── Training ──────────────────────────────────────────────────────────────────

.PHONY: train
train: ## Self-play RL from bootstrap checkpoint (VARIANT=..., DASHBOARD=1 to enable in-process dashboard)
	MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(CHECKPOINT_BOOTSTRAP) $(_NODASH_FLAG) $(VARIANT_FLAG)

.PHONY: train.resume
train.resume: ## Resume training from latest checkpoint (VARIANT= supported)
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(CHECKPOINT_LATEST) $(_NODASH_FLAG) $(VARIANT_FLAG)

.PHONY: train.bg
train.bg: ## Self-play RL from bootstrap checkpoint, background (VARIANT= supported)
	@mkdir -p logs
	@nohup env MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(CHECKPOINT_BOOTSTRAP) $(_NODASH_FLAG) $(VARIANT_FLAG) \
		> logs/train_$$(date +%Y%m%d_%H%M%S).log 2>&1 & \
		echo $$! > logs/train.pid; \
		echo "Training started (PID $$(cat logs/train.pid))"
	@echo "Logs: logs/train_*.log"
	@echo "Dashboard: run 'make dashboard' in a separate terminal to attach (http://localhost:5001)"

.PHONY: train.bg.resume
train.bg.resume: ## Resume latest checkpoint, background (VARIANT= supported)
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	@if [ -f logs/train.pid ] && kill -0 $$(cat logs/train.pid) 2>/dev/null; then \
		echo "Refusing to launch: training already running (PID $$(cat logs/train.pid)). Use 'make train.stop' first."; \
		exit 1; \
	fi
	@mkdir -p logs
	@nohup env MALLOC_ARENA_MAX=2 $(PY) scripts/train.py --checkpoint $(CHECKPOINT_LATEST) $(_NODASH_FLAG) $(VARIANT_FLAG) \
		> logs/train_$$(date +%Y%m%d_%H%M%S).log 2>&1 & \
		echo $$! > logs/train.pid; \
		echo "Resumed from $(CHECKPOINT_LATEST) (PID $$(cat logs/train.pid))"
	@echo "Logs: logs/train_*.log"
	@echo "Dashboard: run 'make dashboard' in a separate terminal to attach (http://localhost:5001)"

.PHONY: train.stop
train.stop: ## Stop background training
	@stopped=0; \
	killed_pid=""; \
	if [ -f logs/train.pid ]; then \
		PID=$$(cat logs/train.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID && echo "Stopped PID $$PID (from train.pid)"; \
			stopped=1; \
			killed_pid=$$PID; \
		else \
			echo "Stale train.pid (PID $$PID not running), removing"; \
		fi; \
		rm -f logs/train.pid; \
	fi; \
	pids=$$(pgrep -f '^[^ ]*python[^ ]* .*scripts/train\.py' 2>/dev/null || true); \
	if [ -n "$$killed_pid" ]; then \
		pids=$$(echo "$$pids" | grep -v "^$${killed_pid}$$" || true); \
	fi; \
	if [ -n "$$pids" ]; then \
		echo "$$pids" | xargs -r kill && { echo "Killed train.py pids: $$pids"; stopped=1; } || true; \
	fi; \
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

.PHONY: calib.run
calib.run: ## Graduation-gate calibration run (RUN=R1|R2|R3|R4, DURATION=12600s)
	@test -n "$(RUN)" || (echo "Usage: make calib.run RUN=R1|R2|R3|R4 [DURATION=sec]" && exit 1)
	bash scripts/run_calibration_run.sh $(RUN) $(if $(DURATION),--duration $(DURATION))

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


# ── Probes ────────────────────────────────────────────────────────────────────

.PHONY: probe.bootstrap
probe.bootstrap: ## Probe bootstrap_model.pt; save fixtures/threat_probe_baseline.json + report
	@mkdir -p reports/probes
	$(PY) scripts/probe_threat_logits.py \
		--checkpoint checkpoints/bootstrap_model.pt \
		--write-baseline \
		--output "reports/probes/bootstrap_$(shell date +%Y%m%d_%H%M%S).md"; \
	rc=$$?; [ $$rc -le 1 ] || exit $$rc

.PHONY: probe.latest
probe.latest: ## Threat-logit probe against latest checkpoint; PASS/FAIL step-5k kill criterion
	@test -n "$(CHECKPOINT_LATEST)" || (echo "No checkpoints/checkpoint_*.pt found" && exit 1)
	@mkdir -p reports/probes
	$(PY) scripts/probe_threat_logits.py \
		--checkpoint "$(CHECKPOINT_LATEST)" \
		--baseline-checkpoint checkpoints/bootstrap_model.pt \
		--output "reports/probes/latest_$(shell date +%Y%m%d_%H%M%S).md"

.PHONY: probe.fixtures
probe.fixtures: ## Regenerate fixtures/threat_probe_positions.npz (WARNING: invalidates bootstrap baseline — commit deliberately)
	@echo "WARNING: this replaces the committed fixture set and invalidates the bootstrap baseline."
	@echo "Only run if you intend to rebase the kill criterion. Commit the result explicitly."
	@read -p "Continue? [y/N] " yn; [ "$$yn" = "y" ] || exit 1
	$(PY) scripts/generate_threat_probe_fixtures.py \
		--output fixtures/threat_probe_positions.npz

WINDOWING_CKPT ?= checkpoints/checkpoint_00020496.pt

.PHONY: probe.windowing
probe.windowing: ## Windowing diagnostic probe (Q_cov, Q_anchor, Q_stability) on checkpoint_00020496.pt
	@mkdir -p reports/probes
	$(PY) scripts/probe_windowing.py \
		--checkpoint "$(WINDOWING_CKPT)" \
		--fixture fixtures/windowing_probe_positions.npz \
		--output "reports/probes/windowing_$(shell date +%Y%m%d_%H%M%S).md"


# ── Eval ──────────────────────────────────────────────────────────────────────

.PHONY: eval
eval: ## Evaluate checkpoint vs SealBot (CKPT=latest, N_GAMES=100, THINK_TIME=0.5, SIMS=128)
	@test -n "$(CKPT)" || (echo "No checkpoint found. Pass CKPT= or generate one first." && exit 1)
	$(PY) scripts/eval_vs_sealbot.py --checkpoint "$(CKPT)" --n-games $(N_GAMES) --time-limit $(THINK_TIME) --model-sims $(SIMS)


# ── Corpus ────────────────────────────────────────────────────────────────────

.PHONY: corpus.export
corpus.export: ## Export optimized NPZ corpus for buffer prefill (all positions, no cap)
	$(PY) scripts/export_corpus_npz.py --no-compress

.PHONY: corpus.export.pretrain
corpus.export.pretrain: ## Export human-only Elo-weighted NPZ for pretrain (all positions, no cap)
	$(PY) scripts/export_corpus_npz.py --human-only --no-compress


# ── Dashboard ─────────────────────────────────────────────────────────────────

.PHONY: dashboard
dashboard: ## Serve dashboard without training (VARIANT=gumbel_full by default)
	@$(PY) scripts/serve_dashboard.py \
		--run-dir runs/$(VARIANT) \
		--checkpoint-dir checkpoints/$(VARIANT) \
		--port 5001

.PHONY: dash.open
dash.open: ## Open web dashboard in browser
	@echo "Opening http://localhost:5001"
	@$(PY) -c "import webbrowser; webbrowser.open('http://localhost:5001')" \
		|| echo "Open manually: http://localhost:5001"
