"""
Integration test: full cold-start → prefill → run → SIGINT save → resume lifecycle.

Exercises the complete replay-buffer persistence path end-to-end:

  Phase A  (cold start + run)
    • Starts train.py with a fresh corpus (60 K synthetic positions).
    • Asserts corpus_loaded fires with ≥ 50 K positions *before* selfplay_pool_started.
    • Asserts corpus_prefill_skipped does NOT fire.
    • Runs until ≥ 50 training steps complete, records buffer size N.

  Phase B  (clean shutdown via SIGINT)
    • Sends SIGINT to the running process.
    • Asserts checkpoints/replay_buffer.bin appears within 30 s.
    • Asserts file size > 0.

  Phase C  (resume)
    • Relaunches train.py with --checkpoint <latest> and --iterations 5.
    • Asserts buffer_restored fires with positions ≈ N.
    • Asserts corpus_prefill_skipped fires (buffer-persist path active).
    • Asserts corpus_loaded does NOT fire.
    • Asserts selfplay_pool_started fires (no cold-start delay).

Marked ``slow`` and ``integration`` — excluded from ``make test.py``.
Run with::

    make test.integration

Expected wall-clock time: 2-5 min on target hardware (Ryzen 7 3700x + RTX 3070).
"""
from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
TRAIN_SCRIPT = str(ROOT / "scripts" / "train.py")

# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_jsonl(path: Path) -> list[dict]:
    """Return all valid JSON objects from a structlog JSONL file."""
    if not path.exists():
        return []
    events: list[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return events


def _first_event(events: list[dict], key: str) -> dict | None:
    return next((e for e in events if e.get("event") == key), None)


def _all_events(events: list[dict], key: str) -> list[dict]:
    return [e for e in events if e.get("event") == key]


def _poll_until(
    log_path: Path,
    predicate,
    timeout: float,
    poll_interval: float = 1.0,
):
    """Poll log until predicate(events) returns a truthy value, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate(_parse_jsonl(log_path))
        if result:
            return result
        time.sleep(poll_interval)
    return predicate(_parse_jsonl(log_path))


def _wait_for_file(path: Path, timeout: float) -> bool:
    """Return True if path exists with size > 0 within timeout seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.5)
    return False


# ── Fixture builders ───────────────────────────────────────────────────────────


def _make_corpus(path: Path, n: int = 60_000) -> None:
    """Write a synthetic bootstrap-corpus NPZ with n positions.

    States are all-zero (near-perfect NPZ compression).
    Policies are uniform over 362 actions.
    Outcomes alternate ±1.
    """
    rng = np.random.default_rng(0)
    states = np.zeros((n, 18, 19, 19), dtype=np.float16)
    policies = np.full((n, 362), 1.0 / 362, dtype=np.float32)
    outcomes = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    np.savez_compressed(path, states=states, policies=policies, outcomes=outcomes)


def _write_config(
    path: Path,
    *,
    corpus_path: Path,
    checkpoint_dir: Path,
) -> None:
    """Write a minimal override config that keeps the training loop fast."""
    persist_path = checkpoint_dir / "replay_buffer.bin"
    # Use absolute paths so train.py finds them regardless of cwd.
    path.write_text(
        f"""\
# Lifecycle integration-test overrides.
# Applied on top of base configs by train.py --config.

model:
  res_blocks: 2
  filters: 32

training:
  batch_size: 32
  min_buffer_size: 32
  checkpoint_interval: 20
  eval_interval: 9999999
  buffer_schedule:
    - {{step: 0, capacity: 10000}}
  mixing:
    pretrained_buffer_path: "{corpus_path}"
    pretrain_max_samples: 60000
    decay_steps: 1000000
    min_pretrained_weight: 0.1
    initial_pretrained_weight: 0.8
    buffer_persist: true
    buffer_persist_path: "{persist_path}"

mcts:
  n_simulations: 4
  quiescence_enabled: false

selfplay:
  n_workers: 1
  inference_batch_size: 4
  inference_max_wait_ms: 50.0

eval_pipeline:
  eval_interval: 9999999
"""
    )


# ── Test ───────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
def test_train_lifecycle(tmp_path: Path) -> None:
    """Full cold-start → prefill → run → SIGINT → resume lifecycle."""

    # ── Setup ─────────────────────────────────────────────────────────────────
    corpus_path = tmp_path / "corpus.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    config_path = tmp_path / "lifecycle.yaml"

    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    _make_corpus(corpus_path, n=60_000)
    _write_config(config_path, corpus_path=corpus_path, checkpoint_dir=checkpoint_dir)

    persist_path = checkpoint_dir / "replay_buffer.bin"
    log_path = log_dir / "lifecycle.jsonl"

    base_cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--config", str(config_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-dir", str(log_dir),
        "--no-dashboard",
        "--no-compile",
    ]

    # ── Phase A+B: cold start, run ≥ 50 steps, then SIGINT ───────────────────
    cmd_run = base_cmd + ["--run-name", "lifecycle"]
    proc = subprocess.Popen(
        cmd_run,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # ── Step 1: corpus prefill ≥ 50 K positions ───────────────────────────
        corpus_ev = _poll_until(
            log_path,
            lambda evs: _first_event(evs, "corpus_loaded"),
            timeout=180,
        )
        assert corpus_ev is not None, (
            "corpus_loaded event not found within 180 s — "
            "corpus prefill did not run on cold start"
        )
        positions_loaded = corpus_ev.get("positions", 0)
        assert positions_loaded >= 50_000, (
            f"Expected ≥ 50,000 corpus positions on cold start, got {positions_loaded}"
        )

        # corpus_prefill_skipped must NOT appear (this is a cold start, not a resume)
        events_after_prefill = _parse_jsonl(log_path)
        assert _first_event(events_after_prefill, "corpus_prefill_skipped") is None, (
            "corpus_prefill_skipped fired on cold start — should only appear on resume"
        )

        # corpus_loaded must appear before selfplay_pool_started
        event_sequence = [e.get("event") for e in events_after_prefill]
        corpus_idx = next(
            (i for i, ev in enumerate(event_sequence) if ev == "corpus_loaded"), None
        )
        pool_idx = next(
            (i for i, ev in enumerate(event_sequence) if ev == "selfplay_pool_started"), None
        )
        if corpus_idx is not None and pool_idx is not None:
            assert corpus_idx < pool_idx, (
                f"corpus_loaded (position {corpus_idx}) must precede "
                f"selfplay_pool_started (position {pool_idx}) in the log"
            )

        # ── Step 2: wait for ≥ 50 training steps, record buffer size ─────────
        def _has_50_steps(evs: list[dict]) -> dict | None:
            train_evs = _all_events(evs, "train_step")
            # log_interval=10 by default, so train_step events fire at 10,20,...50
            hits = [e for e in train_evs if e.get("step", 0) >= 50]
            return hits[-1] if hits else None

        step50_ev = _poll_until(log_path, _has_50_steps, timeout=300, poll_interval=2.0)
        assert step50_ev is not None, (
            "No train_step event with step ≥ 50 found within 300 s — "
            "training is too slow or the loop did not start"
        )

        buffer_size_before_shutdown = int(step50_ev.get("buffer_size", 0))
        assert buffer_size_before_shutdown > 0, (
            "buffer_size at step ≥ 50 must be > 0 — self-play positions were not added"
        )

        # ── Step 3: SIGINT → replay_buffer.bin within 30 s ───────────────────
        proc.send_signal(signal.SIGINT)
        assert _wait_for_file(persist_path, timeout=30), (
            f"{persist_path} not found (or empty) within 30 s after SIGINT"
        )

        # Wait for clean process exit (second chance before force-kill)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail("train.py did not exit within 45 s after SIGINT")

    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise

    assert proc.returncode in (0, 1, -2), (
        f"train.py exited with unexpected return code {proc.returncode}"
    )

    # ── Locate latest checkpoint ───────────────────────────────────────────────
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    assert checkpoints, f"No checkpoint_*.pt found in {checkpoint_dir} after Phase A/B"
    latest_ckpt = checkpoints[-1]

    # ── Phase C: resume ────────────────────────────────────────────────────────
    log_path_resume = log_dir / "lifecycle_resume.jsonl"
    cmd_resume = base_cmd + [
        "--run-name", "lifecycle_resume",
        "--checkpoint", str(latest_ckpt),
        "--iterations", "5",
    ]
    result = subprocess.run(
        cmd_resume,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"train.py --checkpoint exited with code {result.returncode}\n"
        f"stderr tail: {result.stderr[-2000:]}"
    )

    events_resume = _parse_jsonl(log_path_resume)

    # ── Step 4a: buffer restored from file ────────────────────────────────────
    restore_ev = _first_event(events_resume, "buffer_restored")
    assert restore_ev is not None, (
        "buffer_restored event not found on resume — "
        "buffer was not loaded from replay_buffer.bin"
    )
    positions_restored = int(restore_ev.get("positions", 0))
    assert positions_restored > 0, (
        "Restored buffer must have > 0 positions"
    )
    # Allow generous tolerance: pool may push extra positions between step-50 log
    # and SIGINT, so restored count can be >= pre-shutdown count; allow small
    # downward tolerance for positions in-flight during the save.
    tolerance = max(100, int(0.10 * buffer_size_before_shutdown))
    assert positions_restored >= buffer_size_before_shutdown - tolerance, (
        f"Restored buffer ({positions_restored}) is much smaller than "
        f"pre-shutdown buffer ({buffer_size_before_shutdown}); "
        f"tolerance was {tolerance}"
    )

    # ── Step 4b: corpus prefill must be skipped on resume ─────────────────────
    assert _first_event(events_resume, "corpus_prefill_skipped") is not None, (
        "corpus_prefill_skipped not found on resume — "
        "corpus was re-loaded unnecessarily instead of using the restored buffer"
    )
    assert _first_event(events_resume, "corpus_loaded") is None, (
        "corpus_loaded fired on resume — corpus prefill was not skipped"
    )

    # ── Step 4c: self-play started immediately (no cold-start warmup wait) ────
    assert _first_event(events_resume, "selfplay_pool_started") is not None, (
        "selfplay_pool_started not found on resume — self-play did not begin"
    )
