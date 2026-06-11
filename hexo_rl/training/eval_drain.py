"""Drain helper for the async eval thread. Extracted from orchestrator.py per §176 P15."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import structlog

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.events import emit_event
from hexo_rl.training.anchor import save_best_model_atomic

log = structlog.get_logger(__name__)


def drain_pending_eval(
    eval_thread: Optional[threading.Thread],
    eval_result: list[Optional[dict[str, Any]]],
    eval_model: Optional[HexTacToeNet],
    best_model: Optional[HexTacToeNet],
    best_model_path: Path,
    best_model_step: Optional[int],
    pool: Any,
    train_step: int,
    run_id: Optional[str] = None,
    encoding: Optional[str] = None,
) -> tuple[Optional[threading.Thread], Optional[int]]:
    """Drain the most recent completed eval: emit event + promote if gated.

    Safe at normal eval ticks and at shutdown — the latter is the whole
    point: without a post-``_run_loop`` drain, a promotion from the final
    eval before Ctrl-C / ``stop_step`` never hits ``best_model.pt`` and
    is silently lost on next restart (D-012).

    Returns ``(new_eval_thread, new_best_model_step)``; callers should
    rebind both. No-op when the thread is still running.
    """
    if eval_thread is None or eval_thread.is_alive():
        return eval_thread, best_model_step
    prev = eval_result[0]
    if prev is None:
        return None, best_model_step
    emit_event({
        "event": "eval_complete",
        "step": prev.get("step", train_step),
        "elo_estimate": prev.get("elo_estimate"),
        "win_rate_vs_sealbot": prev.get("wr_sealbot"),
        "eval_games": prev.get("eval_games", 0),
        "anchor_promoted": prev.get("promoted", False),
        "sealbot_gate_passed": prev.get("sealbot_gate_passed"),
    })
    new_best_step = best_model_step
    if prev.get("promoted"):
        assert eval_model is not None
        assert best_model is not None
        # C1: promote the snapshot that actually passed the gate.
        new_best_step = prev.get("step", train_step)
        promote_anchor(
            eval_model, best_model, best_model_path, pool, new_best_step,
            run_id=run_id, encoding=encoding,
        )
        log.info(
            "best_model_promoted",
            step=train_step,
            eval_step=new_best_step,
            run_id=run_id,
            encoding=encoding,
            path=str(best_model_path),
            graduated=True,
            wr_best=prev.get("wr_best"),
        )
    eval_result[0] = None
    return None, new_best_step


def promote_anchor(
    eval_model: HexTacToeNet,
    best_model: HexTacToeNet,
    best_model_path: Path,
    pool: Any,
    promoted_step: Optional[int],
    *,
    run_id: Optional[str] = None,
    encoding: Optional[str] = None,
    sync_inference: bool = True,
) -> None:
    """Install the gated eval weights as the new anchor: copy → stamped atomic
    save → (optionally) sync self-play inference. Shared by the in-loop drain and
    the §D-LOOPFIX W1 terminal close-out eval so the promotion mechanism (weight
    snapshot, W3 provenance stamp) is identical on both paths.

    ``sync_inference`` (§D-LOOPFIX W1): the in-loop drain syncs the new weights
    into the live self-play inference server and re-tags the replay recorder's
    checkpoint_step (§176 P9 typed forwarder; §D-WALLCAUSATION single weight-sync
    point). The TERMINAL close-out eval passes ``sync_inference=False`` — it runs
    AFTER the pool is stopped (so the terminal eval is UNLOADED), there is no
    self-play left to sync, and the stopped pool has no inference server.
    """
    eval_base = getattr(eval_model, "_orig_mod", eval_model)
    best_model.load_state_dict(eval_base.state_dict())
    best_model.eval()
    # §D-LOOPFIX W3 — stamp the save with the eval step + run_id + encoding so the
    # written anchor is log/filename-distinguishable from the bootstrap (was a bare
    # state_dict → loaders inferred step 0).
    save_best_model_atomic(
        best_model, best_model_path,
        step=promoted_step, run_id=run_id, encoding=encoding,
    )
    if sync_inference:
        pool.sync_inference_weights(eval_base.state_dict())
        if hasattr(pool, "update_checkpoint_step"):
            pool.update_checkpoint_step(promoted_step)
