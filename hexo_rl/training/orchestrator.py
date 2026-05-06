"""Self-play loop side-channel orchestration: eval drain, axis-distribution,
buffer save, pretrain-replay, training-event emission.

Each function is called from training/loop.py at a specific cadence; none
hold loop state. Extracted from training/loop.py per §159 refactor. No
behavior change.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any, Optional

import structlog

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.early_game_probe import (
    EARLY_GAME_ENTROPY_WARN_THRESHOLD,
    EarlyGameProbe,
)
from hexo_rl.monitoring.events import emit_event
from hexo_rl.monitoring.gpu_monitor import GPUMonitor
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
        pool._inference_server.load_state_dict_safe(eval_base.state_dict())
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


def try_save_buffer(
    buffer: Any,
    mixing_cfg: dict[str, Any],
    trigger: str,
    recent_buffer: Optional[Any] = None,
) -> None:
    """Save replay buffer (and optionally recent_buffer) if buffer_persist is enabled."""
    if not mixing_cfg.get("buffer_persist", False):
        return
    bp = Path(mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
    try:
        buffer.save_to_path(str(bp))
        log.info("buffer_saved", path=str(bp), positions=buffer.size, trigger=trigger)
    except Exception as exc:
        log.warning("buffer_save_failed", path=str(bp), error=str(exc))
    if recent_buffer is not None and recent_buffer.size > 0:
        rbp = Path(str(bp) + ".recent")
        try:
            n = recent_buffer.save_to_path(str(rbp))
            log.info("recent_buffer_saved", path=str(rbp), positions=n, trigger=trigger)
        except Exception as exc:
            log.warning("recent_buffer_save_failed", path=str(rbp), error=str(exc))


def replay_pretrain_events(args: argparse.Namespace) -> None:
    """Replay up to 500 pretrain ``training_step`` events into the dashboard on resume."""
    import json
    pretrain_log = Path(args.log_dir) / "pretrain.jsonl"
    if not pretrain_log.exists():
        return
    replay_evs: list[dict] = []
    try:
        with open(pretrain_log) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("event") == "train_step" and d.get("phase") == "pretrain":
                        replay_evs.append({
                            "event": "training_step",
                            "step": d.get("step"),
                            "loss_total": d.get("loss"),
                            "loss_policy": d.get("policy_loss"),
                            "loss_value": d.get("value_loss"),
                            "loss_aux": d.get("aux_opp_reply_loss"),
                            "policy_entropy": d.get("policy_entropy"),
                            "value_accuracy": d.get("value_accuracy"),
                            "lr": d.get("lr"),
                            "grad_norm": d.get("grad_norm"),
                            "corpus_mix": d.get("corpus_mix", {"pretrain": 1.0, "self_play": 0.0}),
                            "phase": "pretrain",
                        })
                except Exception:
                    pass
    except Exception as e:
        log.warning("pretrain_replay_failed", error=str(e))
        return
    if replay_evs:
        log.info("replaying_pretrain_events", count=len(replay_evs[-500:]))
        for ev in replay_evs[-500:]:
            emit_event(ev)


def emit_axis_distribution(
    train_step: int,
    pool: Any,
    config: dict[str, Any],
    baseline: dict[str, float],
    tb_writer: Any,
) -> Optional[float]:
    """Compute and emit selfplay axis-distribution metrics (§axis_dist).

    Samples the last ≤100 completed game move histories from the pool, computes
    per-axis same-color pair fractions, and logs:
      - Absolute fractions to structlog + emit_event
      - Deltas from corpus baseline to TensorBoard (alongside absolutes)
      - Warn/alert thresholds from config.monitors
    """
    from hexo_rl.training.axis_distribution import compute_axis_fractions

    recent_games = pool.recent_move_histories
    if not recent_games:
        return None

    metrics = compute_axis_fractions(recent_games)
    axis_q  = metrics["axis_q"]
    axis_r  = metrics["axis_r"]
    axis_s  = metrics["axis_s"]
    axis_max = metrics["axis_max"]

    mon_cfg = config.get("monitors", {})
    axis_warn  = float(mon_cfg.get("axis_warn",  0.45))
    axis_alert = float(mon_cfg.get("axis_alert", 0.50))
    max_frac = max(axis_q, axis_r, axis_s)

    if max_frac >= axis_alert:
        log.warning(
            "axis_distribution_alert",
            step=train_step,
            axis_max=axis_max,
            max_frac=round(max_frac, 4),
            axis_q=round(axis_q, 4),
            axis_r=round(axis_r, 4),
            axis_s=round(axis_s, 4),
            threshold=axis_alert,
            n_games=len(recent_games),
        )
    elif max_frac >= axis_warn:
        log.warning(
            "axis_distribution_warn",
            step=train_step,
            axis_max=axis_max,
            max_frac=round(max_frac, 4),
            axis_q=round(axis_q, 4),
            axis_r=round(axis_r, 4),
            axis_s=round(axis_s, 4),
            threshold=axis_warn,
            n_games=len(recent_games),
        )
    else:
        log.info(
            "axis_distribution",
            step=train_step,
            axis_max=axis_max,
            axis_q=round(axis_q, 4),
            axis_r=round(axis_r, 4),
            axis_s=round(axis_s, 4),
            n_games=len(recent_games),
        )

    emit_event({
        "event": "axis_distribution",
        "step": train_step,
        "axis_q": axis_q,
        "axis_r": axis_r,
        "axis_s": axis_s,
        "axis_max": axis_max,
        "n_games": len(recent_games),
    })

    if tb_writer is not None:
        tb_metrics: dict[str, float] = {
            "axis_dist/axis_q": axis_q,
            "axis_dist/axis_r": axis_r,
            "axis_dist/axis_s": axis_s,
        }
        for label in ("axis_q", "axis_r", "axis_s"):
            if label in baseline:
                tb_metrics[f"axis_dist_delta/{label}"] = metrics[label] - baseline[label]
        try:
            tb_writer.log_step(train_step, tb_metrics)
        except Exception as _tb_err:
            log.warning("axis_distribution_tb_failed", step=train_step, error=str(_tb_err))

    return axis_q


def emit_training_events(
    train_step: int,
    loss_info: dict[str, float],
    w_pre: float,
    games_played: int,
    last_iter_games: int,
    pool: Any,
    buffer: Any,
    gpu_monitor: GPUMonitor,
    config: dict[str, Any],
    mcts_config: dict[str, Any],
    capacity: int,
    games_per_hour_fn: Any,
    qfire_delta: int,
    early_game_probe: Optional[EarlyGameProbe] = None,
    trainer_model: Optional[Any] = None,
) -> None:
    """Emit ``training_step`` + ``iteration_complete`` events and structlog entry."""
    policy_entropy = float(loss_info.get("policy_entropy", 0.0))
    value_accuracy = float(loss_info.get("value_accuracy", 0.0))
    grad_norm      = float(loss_info.get("grad_norm", float("nan")))
    lr             = float(loss_info.get("lr", 0.0))

    # §115 early-game entropy probe — one forward pass on a fixed 10-position
    # fixture. Rides on log_interval cadence.
    probe_metrics: dict[str, Any] = {}
    if early_game_probe is not None and trainer_model is not None:
        try:
            probe_metrics = early_game_probe.compute(trainer_model)
            if probe_metrics["early_game_entropy_mean"] > EARLY_GAME_ENTROPY_WARN_THRESHOLD:
                log.warning(
                    "early_game_entropy_high",
                    step=train_step,
                    entropy_mean=round(probe_metrics["early_game_entropy_mean"], 4),
                    threshold=EARLY_GAME_ENTROPY_WARN_THRESHOLD,
                    entropy_by_ply=[round(x, 3) for x in probe_metrics["early_game_entropy_by_ply"]],
                )
        except Exception as _egp_err:
            log.warning("early_game_probe_failed", step=train_step, error=str(_egp_err))
            probe_metrics = {}

    training_step_event: dict[str, Any] = {
        "event": "training_step",
        "step": train_step,
        "loss_total":              float(loss_info["loss"]),
        "loss_policy":             float(loss_info["policy_loss"]),
        "loss_value":              float(loss_info["value_loss"]),
        "loss_aux":                float(loss_info.get("opp_reply_loss", 0.0)),
        "loss_ownership":          float(loss_info.get("ownership_loss", 0.0)),
        "loss_threat":             float(loss_info.get("threat_loss", 0.0)),
        "loss_chain":              float(loss_info.get("chain_loss", 0.0)),
        "aux_loss_rows":           int(loss_info.get("aux_loss_rows", 0)),
        "avg_sigma":               float(loss_info.get("avg_sigma", 0.0)),
        "policy_entropy":                  policy_entropy,
        "policy_entropy_pretrain":         float(loss_info.get("policy_entropy_pretrain", float("nan"))),
        "policy_entropy_selfplay":         float(loss_info.get("policy_entropy_selfplay", float("nan"))),
        "selfplay_model_entropy_batch":    float(loss_info.get("selfplay_model_entropy_batch", float("nan"))),  # alias; drop 2026-05-28
        "policy_entropy_recent":           float(loss_info.get("policy_entropy_recent", float("nan"))),
        "policy_entropy_uniform_selfplay": float(loss_info.get("policy_entropy_uniform_selfplay", float("nan"))),
        "policy_target_entropy":   float(loss_info.get("policy_target_entropy", 0.0)),
        # §101 — D-Gumbel / D-Zeroloss split metrics. NaN when the respective
        # subset is empty; renderers must handle NaN + missing keys gracefully.
        "policy_target_entropy_fullsearch":    float(loss_info.get("policy_target_entropy_fullsearch",    float("nan"))),
        "policy_target_entropy_fastsearch":    float(loss_info.get("policy_target_entropy_fastsearch",    float("nan"))),
        "policy_target_kl_uniform_fullsearch": float(loss_info.get("policy_target_kl_uniform_fullsearch", float("nan"))),
        "policy_target_kl_uniform_fastsearch": float(loss_info.get("policy_target_kl_uniform_fastsearch", float("nan"))),
        "frac_fullsearch_in_batch":            float(loss_info.get("frac_fullsearch_in_batch", 0.0)),
        "n_rows_policy_loss":                  int(loss_info.get("n_rows_policy_loss", 0)),
        "n_rows_total":                        int(loss_info.get("n_rows_total", 0)),
        "value_accuracy":          value_accuracy,
        "lr":                      lr,
        "grad_norm":               grad_norm,
        "quiescence_fires_per_step": qfire_delta,
    }
    if probe_metrics:
        training_step_event.update(probe_metrics)
    emit_event(training_step_event)

    gph    = games_per_hour_fn()
    avg_gl = pool.avg_game_length if hasattr(pool, "avg_game_length") else 0.0
    pph    = gph * avg_gl if avg_gl > 0 else 0.0
    _runner = pool._runner

    _buf_sp_pct = round(min(pool.self_play_positions_pushed / max(buffer.size, 1), 1.0), 4)

    emit_event({
        "event": "iteration_complete",
        "step": train_step,
        "games_total":        games_played,
        "games_this_iter":    games_played - last_iter_games,
        "games_per_hour":     round(gph, 1),
        "positions_per_hour": round(pph, 1),
        "avg_game_length":    round(avg_gl, 1),
        "win_rate_p0":        round(float(pool.x_winrate), 4),
        "win_rate_p1":        round(float(pool.o_winrate), 4),
        "draw_rate":          round(float(pool.draws / games_played), 4) if games_played > 0 else 0.0,
        "sims_per_sec":       pool.sims_per_sec or 0.0,
        "buffer_size":        buffer.size,
        "buffer_capacity":    buffer.capacity,
        "corpus_selfplay_frac": round(1.0 - w_pre, 4),
        "batch_fill_pct":     pool.batch_fill_pct,
        "mcts_mean_depth":    float(getattr(_runner, "mcts_mean_depth", 0.0)),
        "mcts_root_concentration": float(getattr(_runner, "mcts_mean_root_concentration", 0.0)),
        # §107 I2 investigation metrics: lifetime-mean per-cluster std-dev of
        # values and top-1 policy disagreement (K≥2 positions only).
        "cluster_value_std_mean":      float(getattr(_runner, "cluster_value_std_mean", 0.0)),
        "cluster_policy_disagreement_mean": float(getattr(_runner, "cluster_policy_disagreement_mean", 0.0)),
        "cluster_variance_sample_count":    int(getattr(_runner, "cluster_variance_sample_count", 0)),
    })

    # Richer summary structlog entry — fires at log_interval cadence alongside
    # the trainer's per-step ``train_step`` log. Kept under a distinct event
    # name to preserve the 1:1 step-to-``train_step`` invariant (Q27 smoke
    # 2026-04-19 root cause: this entry previously emitted under the same
    # ``train_step`` name and duplicated the trainer's per-step emission).
    log.info(
        "train_step_summary",
        step=train_step,
        policy_loss=round(float(loss_info["policy_loss"]), 4),
        value_loss=round(float(loss_info["value_loss"]), 4),
        total_loss=round(float(loss_info["loss"]), 4),
        aux_opp_reply_loss=round(float(loss_info.get("opp_reply_loss", 0.0)), 4),
        avg_sigma=round(float(loss_info.get("avg_sigma", 0.0)), 4),
        policy_entropy=round(policy_entropy, 4),
        policy_entropy_pretrain=round(float(loss_info.get("policy_entropy_pretrain", float("nan"))), 4),
        policy_entropy_selfplay=round(float(loss_info.get("policy_entropy_selfplay", float("nan"))), 4),
        selfplay_model_entropy_batch=round(float(loss_info.get("selfplay_model_entropy_batch", float("nan"))), 4),  # alias; drop 2026-05-28
        policy_entropy_recent=round(float(loss_info.get("policy_entropy_recent", float("nan"))), 4),
        policy_entropy_uniform_selfplay=round(float(loss_info.get("policy_entropy_uniform_selfplay", float("nan"))), 4),
        buffer_size=buffer.size,
        buffer_capacity=buffer.capacity,
        pretrained_weight=round(w_pre, 4),
        selfplay_weight=round(1.0 - w_pre, 4),
        buffer_self_play_pct=_buf_sp_pct,
        games_played=games_played,
        games_per_hour=round(gph, 1),
        sims_per_sec=pool.sims_per_sec,
        x_wins=pool.x_wins,
        o_wins=pool.o_wins,
        draws=pool.draws,
        x_winrate=round(float(pool.x_winrate), 3),
        o_winrate=round(float(pool.o_winrate), 3),
        draw_rate=round(float(pool.draws / games_played), 3) if games_played > 0 else 0.0,
        gpu_util=round(float(gpu_monitor.gpu_util_pct), 1),
        vram_gb=round(float(gpu_monitor.vram_used_gb), 2),
        ownership_loss=round(float(loss_info["ownership_loss"]), 4) if loss_info.get("ownership_loss") is not None else None,
        threat_loss=round(float(loss_info["threat_loss"]), 4) if loss_info.get("threat_loss") is not None else None,
        aux_loss_rows=int(loss_info.get("aux_loss_rows", 0)),
        batch_fill_pct=round(pool.batch_fill_pct, 1),
        inf_forward_count=pool._inference_server._forward_count,
        inf_total_requests=pool._inference_server._total_requests,
        mcts_mean_depth=round(float(getattr(_runner, "mcts_mean_depth", 0.0)), 3),
        mcts_root_concentration=round(float(getattr(_runner, "mcts_mean_root_concentration", 0.0)), 3),
        policy_target_entropy_fullsearch=float(loss_info.get("policy_target_entropy_fullsearch", float("nan"))),
        policy_target_entropy_fastsearch=float(loss_info.get("policy_target_entropy_fastsearch", float("nan"))),
        policy_target_kl_uniform_fullsearch=float(loss_info.get("policy_target_kl_uniform_fullsearch", float("nan"))),
        policy_target_kl_uniform_fastsearch=float(loss_info.get("policy_target_kl_uniform_fastsearch", float("nan"))),
        frac_fullsearch_in_batch=float(loss_info.get("frac_fullsearch_in_batch", 0.0)),
        n_rows_policy_loss=int(loss_info.get("n_rows_policy_loss", 0)),
        n_rows_total=int(loss_info.get("n_rows_total", 0)),
        cluster_value_std_mean=float(getattr(_runner, "cluster_value_std_mean", 0.0)),
        cluster_policy_disagreement_mean=float(getattr(_runner, "cluster_policy_disagreement_mean", 0.0)),
        cluster_variance_sample_count=int(getattr(_runner, "cluster_variance_sample_count", 0)),
        early_game_entropy_mean=round(float(probe_metrics.get("early_game_entropy_mean", float("nan"))), 4)
            if probe_metrics else None,
        early_game_top1_mass_mean=round(float(probe_metrics.get("early_game_top1_mass_mean", float("nan"))), 4)
            if probe_metrics else None,
    )
