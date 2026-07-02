"""Training-loop event emitters: pretrain-replay, axis-distribution, training-step.

Extracted from training/orchestrator.py per §176 P15. No behavior change.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import structlog

from hexo_rl.monitoring.early_game_probe import (
    EARLY_GAME_ENTROPY_WARN_THRESHOLD,
    EarlyGameProbe,
)
from hexo_rl.monitoring.events import emit_event
from hexo_rl.monitoring.gpu_monitor import GPUMonitor

log = structlog.get_logger(__name__)


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
    solver_deltas: Optional[dict[str, Any]] = None,
) -> None:
    """Emit ``training_step`` + ``iteration_complete`` events and structlog entry.

    ``solver_deltas`` (D-WS3V3) carries per-step solver fire-rate fields computed
    from the ``_last_*`` counter deltas in the step coordinator
    (``solver_eligible_per_step`` / ``solver_injected_per_step`` /
    ``solver_fire_rate`` / ``solver_fire_rate_seeded``); merged into the
    ``training_step`` event. ``None`` on an OFF run (no keys emitted).
    """
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
        # B5: aux_loss_rows dropped (== value_rows_selfplay).
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
        # B5: frac_fullsearch_in_batch dropped (== full_search_frac).
        "n_rows_policy_loss":                  int(loss_info.get("n_rows_policy_loss", 0)),
        "n_rows_total":                        int(loss_info.get("n_rows_total", 0)),
        "value_accuracy":          value_accuracy,
        "lr":                      lr,
        "grad_norm":               grad_norm,
        "quiescence_fires_per_step": qfire_delta,
    }
    if probe_metrics:
        training_step_event.update(probe_metrics)
    # D-WS3V3 — per-step solver fire-rate deltas (null-safe fire-rate when
    # eligible==0). Merged only when solver-in-loop is active (deltas supplied).
    if solver_deltas:
        training_step_event.update(solver_deltas)
    emit_event(training_step_event)

    gph    = games_per_hour_fn()
    avg_gl = pool.avg_game_length if hasattr(pool, "avg_game_length") else 0.0
    pph    = gph * avg_gl if avg_gl > 0 else 0.0
    # §176 P9 — typed snapshot replaces ad-hoc ``pool._runner`` reaches.
    rstats = pool.runner_stats()
    istats = pool.inference_stats()

    _buf_sp_pct = round(min(pool.self_play_positions_pushed / max(buffer.size, 1), 1.0), 4)

    # B5 regime-guard: mcts_root_concentration + the §107 I2 cluster trio are
    # PUCT-descent-specific (meaningless under Gumbel-root sampling), so emit
    # them only when gumbel_mcts is OFF. mcts_mean_depth stays unconditional —
    # interior-PUCT descent is Gumbel-valid (audit).
    _puct_regime = not pool.gumbel_mcts
    iteration_complete_event: dict[str, Any] = {
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
        "mcts_mean_depth":    rstats.mcts_mean_depth,
        # D-WS3V3 — cumulative in-run solver fire-rate totals (all 0 on an OFF run).
        "solver_moves_eligible":       rstats.solver_moves_eligible,
        "solver_win_proven":           rstats.solver_win_proven,
        "solver_injected":             rstats.solver_injected,
        "solver_injected_offwindow":   rstats.solver_injected_offwindow,
        "solver_budget_exhausted":     rstats.solver_budget_exhausted,
        "solver_moves_eligible_seeded": rstats.solver_moves_eligible_seeded,
        "solver_injected_seeded":      rstats.solver_injected_seeded,
        "seeded_games_started":        rstats.seeded_games_started,
    }
    if _puct_regime:
        iteration_complete_event["mcts_root_concentration"] = rstats.mcts_mean_root_concentration
        # §107 I2 investigation metrics: lifetime-mean per-cluster std-dev of
        # values and top-1 policy disagreement (K≥2 positions only).
        iteration_complete_event["cluster_value_std_mean"] = rstats.cluster_value_std_mean
        iteration_complete_event["cluster_policy_disagreement_mean"] = rstats.cluster_policy_disagreement_mean
        iteration_complete_event["cluster_variance_sample_count"] = rstats.cluster_variance_sample_count
    emit_event(iteration_complete_event)

    # B5 regime-guard (mirror of iteration_complete): suppress the
    # PUCT-descent-specific root_concentration + I2 cluster trio under Gumbel.
    _puct_log_kwargs: dict[str, Any] = {}
    if _puct_regime:
        _puct_log_kwargs["mcts_root_concentration"] = round(rstats.mcts_mean_root_concentration, 3)
        _puct_log_kwargs["cluster_value_std_mean"] = rstats.cluster_value_std_mean
        _puct_log_kwargs["cluster_policy_disagreement_mean"] = rstats.cluster_policy_disagreement_mean
        _puct_log_kwargs["cluster_variance_sample_count"] = rstats.cluster_variance_sample_count

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
        # B5: aux_loss_rows dropped (== value_rows_selfplay).
        batch_fill_pct=round(pool.batch_fill_pct, 1),
        inf_forward_count=istats.forward_count,
        inf_total_requests=istats.total_requests,
        mcts_mean_depth=round(rstats.mcts_mean_depth, 3),
        policy_target_entropy_fullsearch=float(loss_info.get("policy_target_entropy_fullsearch", float("nan"))),
        policy_target_entropy_fastsearch=float(loss_info.get("policy_target_entropy_fastsearch", float("nan"))),
        policy_target_kl_uniform_fullsearch=float(loss_info.get("policy_target_kl_uniform_fullsearch", float("nan"))),
        policy_target_kl_uniform_fastsearch=float(loss_info.get("policy_target_kl_uniform_fastsearch", float("nan"))),
        # B5: frac_fullsearch_in_batch dropped (== full_search_frac).
        n_rows_policy_loss=int(loss_info.get("n_rows_policy_loss", 0)),
        n_rows_total=int(loss_info.get("n_rows_total", 0)),
        early_game_entropy_mean=round(float(probe_metrics.get("early_game_entropy_mean", float("nan"))), 4)
            if probe_metrics else None,
        early_game_top1_mass_mean=round(float(probe_metrics.get("early_game_top1_mass_mean", float("nan"))), 4)
            if probe_metrics else None,
        **_puct_log_kwargs,
    )
