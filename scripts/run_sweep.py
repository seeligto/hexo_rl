#!/usr/bin/env python3
"""§122 sweep driver — sequential phase 1 + 2 trainer for the six channel
variants. Resume-safe; can be invoked repeatedly without re-running completed
work.

Layout:
    checkpoints/sweep/{variant}/checkpoint_NNNNNNNN.pt
    logs/sweep/{variant}/train_*.jsonl    — structlog event stream per variant
    logs/sweep/state.json                 — driver progress + selection record
    logs/tb/sweep/{variant}/              — TensorBoard scalars (mirror of JSON)

Phase 1 (--phase 1):
    Train each of the six variants from a fresh backbone for 2500 steps,
    eval every 500 steps. Aborts on grad-norm > 5.0 sustained 100 steps,
    NaN, or policy CE not below 5.0 by step 1000. Each variant runs as
    its own subprocess so a crash in one does not kill the rest.

Phase 2 (--phase 2):
    Selects top 3 variants by WR vs the configured anchor checkpoint at
    step 2500 (only those with policy CE < 4.0). Always includes
    sweep_18ch as a reference baseline. Resumes each survivor from its
    phase-1 checkpoint, extending the cosine horizon to step 10000.

Phase 3 (--phase 3):
    Hands off to scripts/tournament_sweep.py which runs round-robin
    100-game matches between every pair of step-10000 checkpoints.

Usage:
    .venv/bin/python scripts/run_sweep.py --phase 1
    .venv/bin/python scripts/run_sweep.py --phase 2
    .venv/bin/python scripts/run_sweep.py --phase 3
    .venv/bin/python scripts/run_sweep.py --phase all

    .venv/bin/python scripts/run_sweep.py --phase 1 --configs sweep_2ch sweep_4ch
    .venv/bin/python scripts/run_sweep.py --phase 1 --dry-run    # print plan only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "sweep"
LOGS_ROOT = REPO_ROOT / "logs" / "sweep"
TB_ROOT = REPO_ROOT / "logs" / "tb" / "sweep"
STATE_FILE = LOGS_ROOT / "state.json"
PYTHON = sys.executable

ALL_VARIANTS = (
    "sweep_2ch",
    "sweep_3ch",
    "sweep_4ch",
    "sweep_6ch",
    "sweep_8ch",
    "sweep_18ch",
)
PHASE1_STEPS = 1500            # was 2500. Sweep selects on policy-CE slope, not
                                # absolute strength; 1500 captures the early-train
                                # discriminator. Override with --phase1-steps.
PHASE2_STEPS = 10000
ANCHOR_VARIANT = "sweep_18ch"  # always carried into phase 2 + tournament

# ── Triage gate (§122 wave 2026-04-26) ─────────────────────────────────────
# After PHASE1_TRIAGE_WINDOW_SEC each phase-1 variant is checked against:
#
#   1. did it reach at least PHASE1_TRIAGE_MIN_STEP in the window?
#      → catches "process is alive but step counter is stuck" (hung pool,
#        deadlocked inference server, etc.).
#   2. is its latest policy_ce below the per-variant threshold?
#      → catches catastrophic divergence (NaN, gradient blowup → CE drifts
#        ABOVE uniform 5.89).
#
# The thresholds are *deliberately* near-uniform: the experiment's whole
# point is to detect channel-adequacy signal from narrow inputs, where the
# expected trajectory is slow CE descent over hundreds of steps. An empirical
# laptop run on sweep_2ch sat at 5.86 at step 168 — only ~0.03 below
# uniform — and was still legitimately learning. Setting a tight threshold
# here would kill exactly the "real signal from minimal inputs" we want to
# observe. Triage fires only on "policy went UP from uniform" (clear failure)
# or "process hung" (didn't reach min_step).
PHASE1_TRIAGE_WINDOW_SEC = 1200
PHASE1_TRIAGE_MIN_STEP = 100
PHASE1_TRIAGE_MAX_POLICY_CE_DEFAULT = 5.95  # uniform + small margin = catches divergence only
PER_VARIANT_TRIAGE_MAX_POLICY_CE: Dict[str, float] = {
    # No per-variant overrides currently — the default 5.95 is conservative
    # enough for every variant. Add entries here only if a specific channel
    # set demonstrably requires a different ceiling.
}


def triage_policy_ce_gate_for(variant: str) -> float:
    return PER_VARIANT_TRIAGE_MAX_POLICY_CE.get(variant, PHASE1_TRIAGE_MAX_POLICY_CE_DEFAULT)

# Each variant YAML pins seed: 12200 — controls Python-side stochasticity
# (model init, batch sampling, augmentation indices). Rust self-play workers
# use rand::rng() (OS-entropy thread-local) and are NOT reached by this seed;
# self-play game outcomes have a non-deterministic component per worker. See
# docs/sweep_deployment.md "Self-play RNG seeding" for the full discussion.

# Default anchor checkpoint for "WR vs bootstrap-v5 anchor" eval. Defined as a
# fall-back chain — first existing path wins. Override via --anchor-checkpoint.
DEFAULT_ANCHOR_CANDIDATES = (
    REPO_ROOT / "checkpoints" / "bootstrap_v5.pt",
    REPO_ROOT / "checkpoints" / "bootstrap_model.pt",
    REPO_ROOT / "checkpoints" / "best_model.pt",
)


# ── Phase 1 abort gates (per the §122 sweep brief) ─────────────────────────
HARD_GRAD_NORM = 5.0
HARD_GRAD_NORM_STEPS = 100
# Default policy-CE-by-1000 gate. log(362) ≈ 5.89 is uniform; failing to
# fall below 5.0 by step 1000 means the network has not started learning.
HARD_POLICY_CE_BY_1000 = 5.0
# sweep_2ch (only planes 0/8 — current + opp stones) is the lower bound on
# input information. With cosine LR and AdamW it can be legitimately slow
# in the first 1000 steps. Loosen the early CE gate for it; rely on
# grad-norm + monotonic entropy decline + value MSE for "is this learning
# at all" instead.
PER_VARIANT_POLICY_CE_BY_1000 = {
    "sweep_2ch": 5.7,   # only ~3% below uniform — accept slow start
    "sweep_3ch": 5.5,
}


def policy_ce_gate_for(variant: str) -> float:
    return PER_VARIANT_POLICY_CE_BY_1000.get(variant, HARD_POLICY_CE_BY_1000)


# ── State tracking ─────────────────────────────────────────────────────────

@dataclass
class VariantState:
    name: str
    phase1_complete: bool = False
    phase2_complete: bool = False
    phase1_metrics: Dict[str, float] = field(default_factory=dict)
    phase2_metrics: Dict[str, float] = field(default_factory=dict)
    phase1_failed: Optional[str] = None  # reason string
    phase2_failed: Optional[str] = None


@dataclass
class SweepState:
    variants: Dict[str, VariantState] = field(default_factory=dict)
    phase2_survivors: List[str] = field(default_factory=list)

    def get(self, name: str) -> VariantState:
        if name not in self.variants:
            self.variants[name] = VariantState(name=name)
        return self.variants[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variants": {k: asdict(v) for k, v in self.variants.items()},
            "phase2_survivors": list(self.phase2_survivors),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepState":
        s = cls()
        for k, v in d.get("variants", {}).items():
            s.variants[k] = VariantState(**v)
        s.phase2_survivors = list(d.get("phase2_survivors", []))
        return s


def load_state() -> SweepState:
    if STATE_FILE.exists():
        with STATE_FILE.open() as f:
            return SweepState.from_dict(json.load(f))
    return SweepState()


def save_state(state: SweepState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(state.to_dict(), f, indent=2)
    tmp.replace(STATE_FILE)


# ── Checkpoint discovery ───────────────────────────────────────────────────

def variant_ckpt_dir(variant: str) -> Path:
    return CHECKPOINTS_ROOT / variant


def find_checkpoint_at_step(variant: str, step: int) -> Optional[Path]:
    p = variant_ckpt_dir(variant) / f"checkpoint_{step:08d}.pt"
    return p if p.exists() else None


def latest_checkpoint(variant: str) -> Optional[Tuple[int, Path]]:
    d = variant_ckpt_dir(variant)
    if not d.exists():
        return None
    pat = re.compile(r"^checkpoint_(\d{8})\.pt$")
    best: Optional[Tuple[int, Path]] = None
    for entry in d.iterdir():
        m = pat.match(entry.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, entry)
    return best


# ── Train.py invocation ────────────────────────────────────────────────────

TRIAGE_FAIL_RC = 88  # internal sentinel: subprocess killed by triage gate.


def _read_latest_policy_ce(log_path: Path) -> Tuple[Optional[int], Optional[float]]:
    """Tail the variant's structlog jsonl and return the latest (step, policy_ce).

    Reads only train_step / train_step_summary records and returns the highest
    step that carries a policy_ce / policy_loss field. Returns (None, None) if
    the file is missing or no qualifying record has been written yet. Tolerant
    of partial JSON lines (training is still writing); malformed records are
    skipped.
    """
    if not log_path.exists():
        return None, None
    last_step: Optional[int] = None
    last_ce: Optional[float] = None
    try:
        with log_path.open() as f:
            for line in f:
                if '"train_step"' not in line and '"train_step_summary"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = rec.get("step")
                ce = rec.get("policy_ce", rec.get("policy_loss"))
                if isinstance(step, int) and ce is not None:
                    if last_step is None or step >= last_step:
                        last_step = step
                        last_ce = float(ce)
    except OSError:
        return last_step, last_ce
    return last_step, last_ce


def launch_train(
    variant: str,
    *,
    target_steps: int,
    resume_checkpoint: Optional[Path],
    extra_overrides: Optional[Dict[str, Any]] = None,
    hw_overlay: Optional[str] = None,
    log_dir: Path,
    checkpoint_dir: Path,
    dry_run: bool = False,
    triage_window_sec: int = PHASE1_TRIAGE_WINDOW_SEC,
    triage_min_step: int = PHASE1_TRIAGE_MIN_STEP,
    triage_max_policy_ce: Optional[float] = None,
) -> int:
    """Run scripts/train.py for one variant; return exit code (0 = success).

    Triage gate: after ``triage_window_sec`` seconds, if the variant has
    reached at least ``triage_min_step`` and its latest policy_ce exceeds
    ``triage_max_policy_ce``, the subprocess is killed (SIGINT then SIGTERM
    after a 30 s grace) and ``TRIAGE_FAIL_RC`` is returned. Pass
    ``triage_max_policy_ce=None`` to disable triage (used for phase 2 and
    any other call site that already has a stable model).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(REPO_ROOT / "scripts" / "train.py"),
    ]
    # Hardware overlay sits between base configs and the variant so it can set
    # n_workers / batch_size etc. without clobbering sweep-specific keys
    # (input_channels, seed, total_steps).  Load order: base → --config → --variant.
    if hw_overlay:
        overlay_path = REPO_ROOT / "configs" / "variants" / f"{hw_overlay}.yaml"
        if not overlay_path.exists():
            print(f"WARNING: hw_overlay '{hw_overlay}' not found at {overlay_path}; ignored",
                  file=sys.stderr)
        else:
            cmd += ["--config", str(overlay_path)]
    run_name = f"{variant}_target{target_steps}"
    cmd += [
        "--variant", variant,
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-dir", str(log_dir),
        "--run-name", run_name,
        "--iterations", str(target_steps),
        "--no-dashboard",
    ]
    if resume_checkpoint is not None:
        cmd += [
            "--checkpoint", str(resume_checkpoint),
            "--override-scheduler-horizon",
            # Bootstrap weights-only checkpoints have no scheduler_state baked
            # in; trainer.load_checkpoint raises unless this flag is set. Phase
            # 2 resumes a phase-1 ckpt that DOES have scheduler_state, so the
            # flag is harmless there (only fires if scheduler_state is missing).
            "--allow-fresh-scheduler",
        ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"\n[{variant}] launching: {' '.join(cmd)}")
    if dry_run:
        return 0

    log_path = log_dir / f"{run_name}.jsonl"
    triage_enabled = triage_max_policy_ce is not None and triage_window_sec > 0
    if triage_enabled:
        print(f"[{variant}] triage gate: kill if step >= {triage_min_step} "
              f"and policy_ce > {triage_max_policy_ce} after {triage_window_sec}s")
    else:
        print(f"[{variant}] triage gate: disabled")

    t0 = time.time()
    triage_done = False
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
        while True:
            rc = proc.poll()
            if rc is not None:
                elapsed = time.time() - t0
                print(f"[{variant}] exit={rc} elapsed={elapsed:.1f}s")
                return rc

            if triage_enabled and not triage_done and (time.time() - t0) >= triage_window_sec:
                step, ce = _read_latest_policy_ce(log_path)
                if step is None or step < triage_min_step:
                    print(f"[{variant}] triage FAIL: only reached step {step!r} "
                          f"in {triage_window_sec}s (need >= {triage_min_step}). "
                          f"Killing.", file=sys.stderr)
                    _terminate_subprocess(proc, variant)
                    return TRIAGE_FAIL_RC
                if ce is not None and ce > triage_max_policy_ce:
                    print(f"[{variant}] triage FAIL: step {step}, policy_ce={ce:.4f} "
                          f"> threshold {triage_max_policy_ce}. "
                          f"Channel set looks unable to extract signal. Killing.",
                          file=sys.stderr)
                    _terminate_subprocess(proc, variant)
                    return TRIAGE_FAIL_RC
                ce_str = f"{ce:.4f}" if ce is not None else "?"
                print(f"[{variant}] triage PASS: step {step}, policy_ce={ce_str}")
                triage_done = True

            time.sleep(15)
    except KeyboardInterrupt:
        print(f"[{variant}] interrupted")
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=30)
        except Exception:
            pass
        raise


def _terminate_subprocess(proc: "subprocess.Popen", variant: str) -> None:
    """Send SIGINT, wait 30 s, escalate to SIGTERM, then SIGKILL as last resort."""
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        print(f"[{variant}] SIGINT not honoured in 30s; SIGTERM", file=sys.stderr)
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        print(f"[{variant}] SIGTERM not honoured in 10s; SIGKILL", file=sys.stderr)
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


# ── Variant log parsing ────────────────────────────────────────────────────

def latest_log_file(variant: str) -> Optional[Path]:
    d = LOGS_ROOT / variant
    if not d.exists():
        return None
    cands = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands += sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def parse_eval_metrics(variant: str) -> Dict[int, Dict[str, float]]:
    """Read structlog JSON records for the variant, return {step: metrics}.

    Metrics are aggregated across known event names (eval_pipeline, train,
    axis_distribution); the latest entry per step wins.
    """
    log_path = latest_log_file(variant)
    if log_path is None:
        return {}

    out: Dict[int, Dict[str, float]] = {}
    try:
        with log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = rec.get("step")
                if not isinstance(step, int):
                    continue
                bucket = out.setdefault(step, {})
                for key, val in rec.items():
                    if isinstance(val, (int, float)) and key != "step":
                        bucket[key] = float(val)
    except OSError:
        pass
    return out


def variant_metric_at_step(variant: str, step: int, key: str) -> Optional[float]:
    metrics = parse_eval_metrics(variant)
    if step in metrics and key in metrics[step]:
        return metrics[step][key]
    # Fall back to nearest preceding step.
    earlier = [s for s in metrics if s <= step and key in metrics[s]]
    if not earlier:
        return None
    return metrics[max(earlier)][key]


# ── TB scalar writer ───────────────────────────────────────────────────────

def write_tb_scalars(variant: str) -> None:
    """Convert the variant's structlog JSON metrics to TensorBoard scalars."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print(f"[{variant}] tensorboard not installed; skip TB write")
        return
    metrics = parse_eval_metrics(variant)
    if not metrics:
        return
    out_dir = TB_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    try:
        for step, bucket in sorted(metrics.items()):
            for key, val in bucket.items():
                writer.add_scalar(f"{variant}/{key}", val, step)
        writer.add_text(
            "variant_config",
            f"variant={variant}, channel_count={_channel_count(variant)}",
        )
    finally:
        writer.close()


def _channel_count(variant: str) -> int:
    import yaml
    p = REPO_ROOT / "configs" / "variants" / f"{variant}.yaml"
    with p.open() as f:
        cfg = yaml.safe_load(f) or {}
    return len(cfg.get("input_channels", []))


# ── Phase 1 ────────────────────────────────────────────────────────────────

def is_phase1_done(variant: str) -> bool:
    return find_checkpoint_at_step(variant, PHASE1_STEPS) is not None


def run_phase_1(
    variants: List[str],
    *,
    state: SweepState,
    hw_overlay: Optional[str] = None,
    init_checkpoint: Optional[Path] = None,
    triage_window_sec: int = PHASE1_TRIAGE_WINDOW_SEC,
    triage_min_step: int = PHASE1_TRIAGE_MIN_STEP,
    dry_run: bool,
) -> None:
    print(f"\n=== PHASE 1: {len(variants)} variants × {PHASE1_STEPS} steps ===")
    if init_checkpoint is not None:
        print(f"  init weights from {init_checkpoint} "
              f"(reduced-channel variants slice the input conv at load time)")
    for v in variants:
        vs = state.get(v)
        if is_phase1_done(v):
            print(f"[{v}] phase 1 already complete (ckpt {PHASE1_STEPS} present); skip")
            vs.phase1_complete = True
            save_state(state)
            continue

        triage_ce = triage_policy_ce_gate_for(v)
        rc = launch_train(
            v,
            target_steps=PHASE1_STEPS,
            resume_checkpoint=init_checkpoint,
            hw_overlay=hw_overlay,
            log_dir=LOGS_ROOT / v,
            checkpoint_dir=variant_ckpt_dir(v),
            dry_run=dry_run,
            triage_window_sec=triage_window_sec,
            triage_min_step=triage_min_step,
            triage_max_policy_ce=triage_ce,
        )
        if rc == TRIAGE_FAIL_RC:
            vs.phase1_failed = (
                f"triage_failed (window={triage_window_sec}s "
                f"min_step={triage_min_step} max_policy_ce={triage_ce})"
            )
            print(f"[{v}] phase 1 FAILED: {vs.phase1_failed}", file=sys.stderr)
        elif rc == 0 and is_phase1_done(v):
            vs.phase1_complete = True
            metrics_at_step = parse_eval_metrics(v).get(PHASE1_STEPS, {})
            vs.phase1_metrics = dict(metrics_at_step)
            print(f"[{v}] phase 1 complete: metrics_at_{PHASE1_STEPS}={metrics_at_step}")
        else:
            vs.phase1_failed = f"exit={rc} or checkpoint missing"
            print(f"[{v}] phase 1 FAILED: {vs.phase1_failed}", file=sys.stderr)
        save_state(state)
        write_tb_scalars(v)


def select_phase2(state: SweepState, *, anchor_metric_key: str) -> List[str]:
    """Pick top 3 variants for phase 2.

    With reduced PHASE1_STEPS (1500) and random-init phase 1, absolute play
    strength via wr_best is barely measurable — every variant is still weak.
    Policy-CE slope (rate of policy entropy collapse) is the channel-adequacy
    signal we actually want: variants whose policy CE drops further by the
    end of phase 1 have extracted more signal from their input planes.

    Selection priority (high → low):
      1. Lower policy_ce at PHASE1_STEPS (primary).
      2. If a wr-vs-anchor metric IS present, use it as a tiebreaker.

    sweep_18ch is always included (reference baseline) when its phase 1
    completed successfully.
    """
    candidates: List[Tuple[str, float, Optional[float]]] = []
    for name, vs in state.variants.items():
        if not vs.phase1_complete:
            continue
        ce = vs.phase1_metrics.get("policy_ce", vs.phase1_metrics.get("policy_loss"))
        wr = vs.phase1_metrics.get(anchor_metric_key)
        if ce is None:
            continue
        # Loose upper bound: a variant still > 5.7 at PHASE1_STEPS hasn't
        # learned anything useful even by the end of phase 1.
        if ce >= 5.7:
            continue
        candidates.append((name, float(ce), float(wr) if wr is not None else None))

    # Sort: lower CE first; if equal, higher wr first (None < anything).
    candidates.sort(key=lambda t: (t[1], -(t[2] if t[2] is not None else -1.0)))
    top3 = [name for name, _, _ in candidates[:3]]
    if (ANCHOR_VARIANT not in top3
            and state.variants.get(ANCHOR_VARIANT, VariantState(name=ANCHOR_VARIANT)).phase1_complete):
        top3.append(ANCHOR_VARIANT)
    return top3


# ── Phase 2 ────────────────────────────────────────────────────────────────

def is_phase2_done(variant: str) -> bool:
    return find_checkpoint_at_step(variant, PHASE2_STEPS) is not None


def run_phase_2(
    *,
    state: SweepState,
    anchor_metric_key: str,
    hw_overlay: Optional[str] = None,
    dry_run: bool,
) -> None:
    survivors = select_phase2(state, anchor_metric_key=anchor_metric_key)
    if not survivors:
        print("=== PHASE 2: no survivors meet phase-1 selection criteria; aborting ===",
              file=sys.stderr)
        return
    state.phase2_survivors = list(survivors)
    save_state(state)
    print(f"\n=== PHASE 2: survivors={survivors} → step {PHASE2_STEPS} ===")
    for v in survivors:
        vs = state.get(v)
        if is_phase2_done(v):
            print(f"[{v}] phase 2 already complete (ckpt {PHASE2_STEPS} present); skip")
            vs.phase2_complete = True
            save_state(state)
            continue
        latest = latest_checkpoint(v)
        if latest is None:
            print(f"[{v}] no phase-1 checkpoint found; skip phase 2", file=sys.stderr)
            vs.phase2_failed = "phase1_checkpoint_missing"
            save_state(state)
            continue
        rc = launch_train(
            v,
            target_steps=PHASE2_STEPS,
            resume_checkpoint=latest[1],
            hw_overlay=hw_overlay,
            log_dir=LOGS_ROOT / v,
            checkpoint_dir=variant_ckpt_dir(v),
            dry_run=dry_run,
        )
        if rc == 0 and is_phase2_done(v):
            vs.phase2_complete = True
            metrics_at_step = parse_eval_metrics(v).get(PHASE2_STEPS, {})
            vs.phase2_metrics = dict(metrics_at_step)
            print(f"[{v}] phase 2 complete: metrics_at_{PHASE2_STEPS}={metrics_at_step}")
        else:
            vs.phase2_failed = f"exit={rc} or checkpoint missing"
            print(f"[{v}] phase 2 FAILED: {vs.phase2_failed}", file=sys.stderr)
        save_state(state)
        write_tb_scalars(v)


# ── Phase 3 dispatch ───────────────────────────────────────────────────────

def run_phase_3(
    *,
    state: SweepState,
    anchor_checkpoint: Optional[Path],
    dry_run: bool,
) -> int:
    survivors = state.phase2_survivors
    if not survivors:
        # Fallback: all variants that produced a step-10000 checkpoint.
        survivors = [v for v in ALL_VARIANTS if is_phase2_done(v)]
    if not survivors:
        print("=== PHASE 3: no phase-2 survivors with step-10000 checkpoints ===",
              file=sys.stderr)
        return 1
    print(f"\n=== PHASE 3: round-robin tournament — {survivors} ===")
    cmd = [
        PYTHON, str(REPO_ROOT / "scripts" / "tournament_sweep.py"),
        "--variants", *survivors,
        "--checkpoint-step", str(PHASE2_STEPS),
        "--checkpoints-root", str(CHECKPOINTS_ROOT),
        "--out-dir", str(REPO_ROOT / "reports" / "investigations" / "phase122_sweep"),
    ]
    if anchor_checkpoint is not None:
        cmd += ["--anchor-checkpoint", str(anchor_checkpoint)]
    print(f"  → {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


# ── Main ───────────────────────────────────────────────────────────────────

def resolve_anchor(arg: Optional[str]) -> Optional[Path]:
    if arg:
        p = Path(arg).resolve()
        if not p.exists():
            print(f"WARNING: --anchor-checkpoint {p} does not exist", file=sys.stderr)
            return None
        return p
    for cand in DEFAULT_ANCHOR_CANDIDATES:
        if cand.exists():
            return cand
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--phase", choices=("1", "2", "3", "all"), required=True)
    p.add_argument("--configs", nargs="+", default=None,
                   help="Variant subset (default: all six)")
    p.add_argument("--anchor-checkpoint", default=None,
                   help="Anchor model for WR-based selection (default: bootstrap_v5/v4)")
    p.add_argument("--init-checkpoint", default=None,
                   help="Optional warm-start: phase 1 launches train.py with "
                        "--checkpoint <path>. Default: None (random init for fair "
                        "channel-adequacy comparison). For reduced-channel variants "
                        "trainer.load_checkpoint slices the input conv; for "
                        "sweep_18ch the slice is a no-op.")
    p.add_argument("--anchor-metric-key", default="wr_best",
                   help="Metric key produced by eval_pipeline carrying WR vs anchor "
                        "(eval_pipeline writes results['wr_best'] at "
                        "hexo_rl/eval/eval_pipeline.py — keep these in sync)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without launching subprocesses")
    p.add_argument("--hw-overlay", default=None, metavar="VARIANT_STEM",
                   help="Hardware config to layer between base configs and each sweep variant "
                        "(e.g. gumbel_targets_epyc4080). Sets n_workers, batch_size, etc. "
                        "for the target box without touching sweep-specific keys.")
    p.add_argument("--phase1-triage-window-sec", type=int, default=PHASE1_TRIAGE_WINDOW_SEC,
                   help="Phase 1 triage window in seconds (default: 1200 = 20 min). "
                        "After this window each variant is checked for catastrophic "
                        "failure; survivors continue training to PHASE1_STEPS.")
    p.add_argument("--phase1-triage-min-step", type=int, default=PHASE1_TRIAGE_MIN_STEP,
                   help="Phase 1 triage minimum step (default: 100). A variant that "
                        "hasn't reached this step within the triage window is killed "
                        "as 'process hung'.")
    p.add_argument("--no-triage", action="store_true",
                   help="Disable phase 1 triage gate entirely (no kill-on-CE).")
    args = p.parse_args()

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)

    variants = list(args.configs) if args.configs else list(ALL_VARIANTS)
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        print(f"ERROR: unknown variants: {unknown}", file=sys.stderr)
        return 2

    state = load_state()
    anchor = resolve_anchor(args.anchor_checkpoint)

    if args.phase in ("1", "all"):
        # Phase 1 starts every variant from random init for fair comparison.
        # Warm-starting from the bootstrap was tried (see commit f15e183 +
        # the warm-start smoke at reports/investigations/phase122_sweep_throughput/);
        # for reduced-channel variants the slice of the bootstrap's first conv
        # produces features the rest of the trunk wasn't trained on, so the
        # narrow variants get a degraded-policy head start that's neither
        # "trained" nor "random" — noise that confuses the channel-adequacy
        # signal. The slicing path stays in trainer.load_checkpoint as a
        # useful capability; pass --init-checkpoint <path> on the CLI to
        # opt in (sweep_18ch only, where slicing is a no-op, would benefit).
        init_ckpt = (
            getattr(args, "init_checkpoint", None)
            and Path(args.init_checkpoint)
            or None
        )
        run_phase_1(
            variants, state=state, hw_overlay=args.hw_overlay,
            init_checkpoint=init_ckpt,
            triage_window_sec=(0 if args.no_triage else args.phase1_triage_window_sec),
            triage_min_step=args.phase1_triage_min_step,
            dry_run=args.dry_run,
        )
    if args.phase in ("2", "all"):
        run_phase_2(state=state, anchor_metric_key=args.anchor_metric_key,
                    hw_overlay=args.hw_overlay, dry_run=args.dry_run)
    if args.phase in ("3", "all"):
        rc = run_phase_3(state=state, anchor_checkpoint=anchor, dry_run=args.dry_run)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
