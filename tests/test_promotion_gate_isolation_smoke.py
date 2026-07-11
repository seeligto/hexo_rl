"""E1 promotion-gate CUDA isolation — boundary-crossing regression smoke (joins the integration gate).

Drives a REAL ``train.py`` run that CROSSES an eval boundary (the 250k-trigger class geometry: a
small ``eval_interval`` so the boundary falls inside ``--iterations``) with the self-play stall
watchdog ARMED and ``selfplay_stall_timeout_sec`` set EXPLICITLY. This is the exact run2 failure
geometry (eval boundary + armed watchdog). PASS =
  1. the eval boundary is crossed (an ``evaluation_start`` event fires),
  2. ``games_completed`` ADVANCES after the boundary (self-play resumed — no wedge),
  3. the process exits 0 (NOT the watchdog's exit 42 — no stall).

Marked slow + integration → excluded from ``make test.py`` / ``make test``; runs on the GPU via
``.venv/bin/python -m pytest -q -m integration tests/test_promotion_gate_isolation_smoke.py``.
Reuses the closeout smoke's corpus + config helpers (same real launch path).

Design: docs/designs/e1_promotion_gate_cuda_isolation.md §4.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.test_closeout_lifecycle import (
    PYTHON,
    ROOT,
    TRAIN_SCRIPT,
    _make_corpus,
    _parse_jsonl,
    _write_config,
)

# An explicit, non-trivial watchdog timeout for the smoke — long enough that a HEALTHY boundary
# crossing never trips it, short enough that a real wedge fails the smoke fast (not a 45h hang).
_WATCHDOG_TIMEOUT_SEC = 600


@pytest.mark.slow
@pytest.mark.integration
def test_eval_boundary_crosses_with_watchdog_armed(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    config_path = tmp_path / "isolation_smoke.yaml"
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    _make_corpus(corpus_path)
    _write_config(config_path, corpus_path=corpus_path, checkpoint_dir=checkpoint_dir)
    # Arm the self-play stall watchdog with an EXPLICIT timeout (config knob, read by
    # loop._lifecycle_knob → StepCoordinatorConfig.selfplay_stall_timeout_sec). No CLI flag exists.
    with open(config_path, "a") as fh:
        fh.write(f"\nselfplay_stall_timeout_sec: {_WATCHDOG_TIMEOUT_SEC}\n")

    bootstrap = ROOT / "checkpoints" / "bootstrap_model_v6_live2.pt"
    if not bootstrap.exists():
        pytest.skip(f"bootstrap not present: {bootstrap}")

    log_path = log_dir / "isolation_smoke.jsonl"
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--checkpoint", str(bootstrap),
        "--config", str(config_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-dir", str(log_dir),
        "--run-name", "isolation_smoke",
        "--iterations", "16",            # cross MULTIPLE eval_interval=8 boundaries
        "--no-dashboard",
        "--no-compile",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)

    # (3) clean exit — NOT the watchdog's exit 42 (no stall) and NOT any other failure.
    assert proc.returncode == 0, (
        f"train.py exited {proc.returncode} (expected 0; 42 == the self-play stall watchdog fired, "
        f"i.e. the eval boundary WEDGED self-play — the run2 livelock regressed).\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )

    events = _parse_jsonl(log_path)
    kinds = [e.get("event") for e in events]

    # (1) the eval boundary was crossed.
    assert "evaluation_start" in kinds, (
        f"no evaluation_start event — the smoke never crossed an eval boundary; events seen: "
        f"{sorted(set(kinds))}"
    )

    # (2) games_completed ADVANCED after the boundary (self-play resumed — no wedge). Read the
    # games_completed / games_played progression from the iteration events; the LAST value must
    # exceed the value at the FIRST eval boundary.
    def _games(ev: dict) -> int | None:
        for k in ("games_completed", "games_played", "total_games"):
            if isinstance(ev.get(k), int):
                return ev[k]
        return None

    first_eval_idx = next(i for i, e in enumerate(events) if e.get("event") == "evaluation_start")
    games_at_boundary = next(
        (_games(e) for e in events[:first_eval_idx + 1][::-1] if _games(e) is not None), None
    )
    games_final = next((_games(e) for e in events[::-1] if _games(e) is not None), None)
    assert games_final is not None, "no games_completed telemetry emitted"
    if games_at_boundary is not None:
        assert games_final > games_at_boundary, (
            f"games_completed did not advance past the eval boundary "
            f"({games_at_boundary} -> {games_final}) — self-play wedged at the boundary "
            f"(the run2 eval-thread ⊥ self-play GPU-forward livelock)."
        )

    # The watchdog must NOT have fired (a fired watchdog exits 42, already asserted; also confirm no
    # stall event leaked into the stream). The event name is ``selfplay_stall_watchdog``
    # (step_coordinator.py — logged at .error immediately before exit 42).
    assert "selfplay_stall_watchdog" not in kinds, "the stall watchdog fired — a wedge occurred"
