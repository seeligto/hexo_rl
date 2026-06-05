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
        eval_base = getattr(eval_model, "_orig_mod", eval_model)
        best_model.load_state_dict(eval_base.state_dict())
        best_model.eval()
        save_best_model_atomic(best_model, best_model_path)
        new_best_step = prev.get("step", train_step)
        # §176 P9 — typed forwarder replaces direct ``_inference_server`` reach.
        pool.sync_inference_weights(eval_base.state_dict())
        # §D-WALLCAUSATION: self-play weights just changed to the promoted model,
        # so refresh the replay recorder's checkpoint_step to match.  This is the
        # ONLY weight-sync point — tagging here (not per train step) keeps every
        # recorded game attributed to the weights that actually generated it.
        if hasattr(pool, "update_checkpoint_step"):
            pool.update_checkpoint_step(new_best_step)
        log.info(
            "best_model_promoted",
            step=train_step,
            eval_step=new_best_step,
            path=str(best_model_path),
            graduated=True,
            wr_best=prev.get("wr_best"),
        )
    eval_result[0] = None
    return None, new_best_step
