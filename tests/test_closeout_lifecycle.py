"""§D-LOOPFIX W1 / A-CLOSEOUT — the dry-run close-out smoke.

Drives a REAL ``train.py`` to a bounded stop and asserts the lifecycle epilogue:
  • training stopped at exactly N (``iteration_limit_reached`` / session_end step==N),
  • the in-flight eval was DRAINED (not killed at 900 s),
  • a TERMINAL full-battery eval ran on the FINAL checkpoint with stride parity
    IGNORED — best_checkpoint (stride 2) is skipped on the odd final in-run round
    but the terminal eval RUNS it (``terminal_eval_complete`` carries wr_best),
  • the process exited 0.

Cold-starts from the trained bootstrap (the real launch path) so play is decisive.
Marked slow + integration → excluded from ``make test.py`` / ``make test``.

GPU/vast-intended: 19×19 eval games at low sims run to ~80-360 plies and are
minutes each on CPU, so a full terminal-eval completion overruns a CPU budget (the
dispatcher's "CPU-scale; vast only if needed, idle-check"). The close-out LOGIC —
terminal full-battery eval, ignore_stride, promotion, hard-cap backstop, SIGINT
skip, drain-budget math, close_out order — is proven GPU-free by
``tests/training/test_step_coordinator_closeout.py``; this test is the operator's
end-to-end A-CLOSEOUT confirmation on the GPU::

    .venv/bin/python -m pytest -q -m integration tests/test_closeout_lifecycle.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
TRAIN_SCRIPT = str(ROOT / "scripts" / "train.py")


def _parse_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _make_corpus(path: Path, n: int = 3000) -> None:
    # v6_live2 = 4 planes, board 19, 362 policy logits.
    states = np.zeros((n, 4, 19, 19), dtype=np.float16)
    policies = np.full((n, 362), 1.0 / 362, dtype=np.float32)
    rng = np.random.default_rng(0)
    outcomes = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    np.savez_compressed(path, states=states, policies=policies, outcomes=outcomes)


def _write_config(path: Path, *, corpus_path: Path, checkpoint_dir: Path) -> None:
    # Cold-start from the trained bootstrap (the real launch path) so self-play
    # AND eval games are DECISIVE / short — a fresh random-init model plays to
    # ~361-ply board-fill on CPU and the terminal eval can't complete in budget.
    n_workers = max(2, (os.cpu_count() or 4) - 2)
    persist_path = checkpoint_dir / "replay_buffer.bin"
    best_path = checkpoint_dir / "best_model.pt"  # absent → resolve_anchor fresh-inits from the bootstrap
    path.write_text(
        f"""\
# §D-LOOPFIX A-CLOSEOUT smoke overrides (arch inferred from the bootstrap ckpt).
encoding: v6_live2
in_channels: 4

training:
  batch_size: 32
  min_buffer_size: 32
  checkpoint_interval: 8
  eval_interval: 8
  buffer_schedule:
    - {{step: 0, capacity: 10000}}
  mixing:
    pretrained_buffer_path: "{corpus_path}"
    pretrain_max_samples: 3000
    decay_steps: 1000000
    min_pretrained_weight: 0.1
    initial_pretrained_weight: 0.8
    buffer_persist: true
    buffer_persist_path: "{persist_path}"

mcts:
  n_simulations: 4
  quiescence_enabled: false

selfplay:
  n_workers: {n_workers}
  inference_batch_size: 16
  inference_max_wait_ms: 10.0
  random_opening_plies: 0
  playout_cap:
    full_search_prob: 0.0
    fast_prob: 0.0
    fast_sims: 4
    standard_sims: 4

eval_interval: 8
eval_pipeline:
  enabled: true
  eval_interval: 8
  opponents:
    best_checkpoint:
      enabled: true
      stride: 2          # the single (odd) round SKIPS this in-run → terminal must run it
      n_games: 2
      model_sims: 4
      opponent_sims: 4
    random:
      enabled: true
      stride: 1
      n_games: 2
      model_sims: 4
    sealbot:
      enabled: false
    bootstrap_anchor:
      enabled: false
    nnue:
      enabled: false
  gating:
    promotion_winrate: 0.55
    best_model_path: "{best_path}"
    bootstrap_floor:
      enabled: false
"""
    )


@pytest.mark.slow
@pytest.mark.integration
def test_closeout_runs_terminal_full_battery_on_final_checkpoint(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    config_path = tmp_path / "closeout.yaml"
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    _make_corpus(corpus_path)
    _write_config(config_path, corpus_path=corpus_path, checkpoint_dir=checkpoint_dir)

    bootstrap = ROOT / "checkpoints" / "bootstrap_model_v6_live2.pt"
    if not bootstrap.exists():
        pytest.skip(f"bootstrap not present: {bootstrap}")

    log_path = log_dir / "closeout.jsonl"
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--checkpoint", str(bootstrap),
        "--config", str(config_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-dir", str(log_dir),
        "--run-name", "closeout",
        "--iterations", "8",
        "--no-dashboard",
        "--no-compile",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)

    assert proc.returncode == 0, (
        f"train.py exited {proc.returncode} (A-CLOSEOUT requires a clean exit 0).\n"
        f"stderr tail:\n{proc.stderr[-3000:]}"
    )

    events = _parse_jsonl(log_path)
    by_event: dict[str, list[dict]] = {}
    for e in events:
        by_event.setdefault(e.get("event", ""), []).append(e)

    # training stopped at exactly N
    assert by_event.get("iteration_limit_reached"), "training did not reach the iteration limit"

    # an in-run eval round COMPLETED before stop (records the drain-budget measurement)
    assert by_event.get("evaluation_round_complete"), "no in-run eval round completed"

    # the terminal full-battery eval ran on the final checkpoint and COMPLETED
    terminal = by_event.get("terminal_eval_complete", [])
    assert terminal, "terminal_eval_complete never fired — close-out did not run the terminal eval"
    final = terminal[-1]
    assert final.get("completed") is True, f"terminal eval did not complete: {final}"
    assert final.get("step") == 8, f"terminal eval not on the final checkpoint: {final}"

    # ignore_stride: best_checkpoint (stride 2) is skipped on the odd final in-run
    # round but the TERMINAL eval RAN it → wr_best is present.
    assert final.get("wr_best") is not None, (
        f"terminal eval did not run the stride-skipped best_checkpoint opponent "
        f"(ignore_stride failed): {final}"
    )
