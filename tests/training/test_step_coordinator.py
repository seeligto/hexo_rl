"""Unit tests for ``StepCoordinator.step()`` (§159a M4).

Covers Q-§159b Section B items 1–14 — the closure-internal decisions
previously unreachable without booting WorkerPool / EvalPipeline /
GPU monitor.  Each test constructs a coordinator with stub collaborators,
a controllable :class:`FakeClock`, and a list-recorder ``event_emitter``,
runs one or more ``step()`` calls, and asserts on the resulting
:class:`StepOutcome` plus side effects on ``shutdown.running``,
``trainer.save_checkpoint``, ``buffer.resize``, etc.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from hexo_rl.training.step_coordinator import (
    DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC,
    StepCoordinator,
    StepCoordinatorConfig,
)


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FakeShutdown:
    def __init__(self) -> None:
        self.running = True
        self.shutdown_save = False


def _make_config(**overrides: Any) -> StepCoordinatorConfig:
    defaults: dict[str, Any] = {
        "eval_interval": 999_999,
        "log_interval": 999_999,
        "checkpoint_interval": 999_999,
        "composition_interval": 999_999,
        "value_probe_interval": 999_999,
        "min_buf_size": 1,
        "capacity": 1000,
        "buffer_schedule": (),
        "training_steps_per_game": 10.0,
        "max_train_burst": 10,
        "batch_size": 16,
        "augment": False,
        "recency_weight": 0.0,
        "mixing_initial_w": 0.8,
        "mixing_min_w": 0.1,
        "mixing_decay_steps": 1_000_000.0,
        "soft_ew_threshold": 0.0,
        "soft_ew_min_pts": 0,
        "hard_gn_threshold": 3.0,
        "hard_gn_min_steps": 5,
        "instrumentation_enabled": False,
        "stop_step": None,
        "final_eval_drain_timeout_sec": 0.0,
    }
    defaults.update(overrides)
    return StepCoordinatorConfig(**defaults)


def _make_trainer(loss_sequence: list[dict[str, float]] | None = None,
                  *, increment: bool = True) -> Mock:
    trainer = Mock()
    trainer.step = 0
    losses = list(loss_sequence or [{"loss": 0.1, "policy_loss": 0.5,
                                     "value_loss": 0.1, "grad_norm": 0.1}])

    def side_effect(*_a: Any, **_kw: Any) -> dict[str, float]:
        if losses:
            loss = losses.pop(0)
        else:
            loss = {"loss": 0.1, "policy_loss": 0.5,
                    "value_loss": 0.1, "grad_norm": 0.1}
        if increment:
            trainer.step += 1
        return loss

    trainer.train_step = Mock(side_effect=side_effect)
    trainer.train_step_from_tensors = Mock(side_effect=side_effect)
    trainer.save_checkpoint = Mock(return_value="/tmp/ckpt")
    trainer.model = Mock()
    trainer.model._orig_mod = trainer.model
    return trainer


def _make_pool(games_completed: int = 0) -> Mock:
    pool = Mock()
    pool.games_completed = games_completed
    pool.n_workers = 1
    # §176 P9 — pool now exposes typed snapshots; provide them on the mock.
    from hexo_rl.selfplay.pool import RunnerStats, InferenceStats
    pool._runner = Mock(mcts_quiescence_fires=0, model_version=0)
    pool._inference_server = Mock()
    _rstats = RunnerStats(
        games_completed=games_completed, positions_generated=0,
        x_wins=0, o_wins=0, draws=0, model_version=0,
        mcts_quiescence_fires=0, mcts_mean_depth=0.0,
        mcts_mean_root_concentration=0.0, cluster_value_std_mean=0.0,
        cluster_policy_disagreement_mean=0.0, cluster_variance_sample_count=0,
        runner_encoding=None,
    )
    _istats = InferenceStats(
        forward_count=0, total_requests=0, encoding_spec=None,
    )
    pool.runner_stats = Mock(return_value=_rstats)
    pool.inference_stats = Mock(return_value=_istats)
    pool.sync_inference_weights = Mock()
    pool.recent_buffer = None
    # §CANARY-VAL stride-5 gate reads this at every eval point; default benign.
    pool.current_stride5_p90 = Mock(return_value=0)
    return pool


def _make_buffer(size: int = 1000, capacity: int = 1000) -> Mock:
    buf = Mock()
    buf.size = size
    buf.capacity = capacity
    buf.resize = Mock()
    buf.save_to_path = Mock()
    return buf


def _make_anchor() -> Mock:
    anchor = Mock()
    anchor.best_model = Mock()
    anchor.best_model._orig_mod = anchor.best_model
    anchor.best_model_step = 0
    anchor.best_model_path = "/tmp/best_model.pt"
    return anchor


def _make_coordinator(
    *,
    trainer: Mock | None = None,
    pool: Mock | None = None,
    buffer: Mock | None = None,
    pretrained_buffer: Any = None,
    eval_pipeline: Any = None,
    eval_model: Any = None,
    shutdown: FakeShutdown | None = None,
    clock: FakeClock | None = None,
    tracemalloc_provider: Any = None,
    config_overrides: dict[str, Any] | None = None,
    event_emitter: Any = None,
    value_probe: Any = None,
    early_game_probe: Any = None,
    axis_baseline: Any = None,
    iterations: int | None = None,
    full_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> StepCoordinator:
    cfg = _make_config(**(config_overrides or {}))
    _subsystems = Mock(
        gpu_monitor=Mock(gpu_util_pct=10.0),
        early_game_probe=early_game_probe,
        value_probe=value_probe,
        axis_baseline=axis_baseline if axis_baseline is not None else {},
        tb_writer=None,
    )
    coord = StepCoordinator(
        trainer=trainer or _make_trainer(),
        buffer=buffer or _make_buffer(),
        pretrained_buffer=pretrained_buffer,
        recent_buffer=None,
        pool=pool or _make_pool(),
        eval_pipeline=eval_pipeline,
        subsystems=_subsystems,
        anchor_state=_make_anchor(),
        shutdown=shutdown or FakeShutdown(),
        eval_model=eval_model if eval_model is not None else Mock(_orig_mod=Mock()),
        bufs=Mock(),
        config=cfg,
        full_config=full_config or {"monitors": {}},
        train_cfg={"batch_size": 16},
        mcts_config={},
        mixing_cfg={"buffer_persist": False},
        batch_size_cfg=16,
        iterations=iterations,
        clock=clock or FakeClock(),
        tracemalloc_provider=tracemalloc_provider,
        event_emitter=event_emitter or Mock(),
        **kwargs,
    )
    # Force closure-equivalent init (closure used `last_train_game_count = 0`
    # hardcoded; coordinator mirrors pool.games_completed which differs in tests
    # that pre-seed pool counts).  Reset here so tests can bump pool.games_completed
    # post-init to drive new_games > 0.
    coord.last_train_game_count = 0
    # Default: bump pool to 1 game so first step() enters the inner loop.
    if coord.pool.games_completed == 0:
        coord.pool.games_completed = 1
    return coord


@pytest.fixture
def patch_orchestrator_helpers():
    """Patch the orchestrator helpers imported into step_coordinator so step()
    doesn't require a real WorkerPool / EvalPipeline / TB writer."""
    with patch("hexo_rl.training.step_coordinator._emit_axis_distribution") as axis, \
         patch("hexo_rl.training.step_coordinator._emit_training_events") as train_ev, \
         patch("hexo_rl.training.step_coordinator._try_save_buffer") as save_buf, \
         patch("hexo_rl.training.step_coordinator._drain_pending_eval") as drain, \
         patch("hexo_rl.training.step_coordinator.assemble_mixed_batch") as mix:
        axis.return_value = None
        drain.return_value = (None, None)  # (eval_thread, best_model_step)
        train_ev.return_value = None
        save_buf.return_value = None
        yield {
            "axis": axis,
            "train_ev": train_ev,
            "save_buf": save_buf,
            "drain": drain,
            "mix": mix,
        }


# ── B#1: hard-abort on sustained gradient norm ──────────────────────────────

def test_hard_abort_grad_norm_streak(patch_orchestrator_helpers):
    """5 consecutive high-GN inner steps trip shutdown.running = False."""
    trainer = _make_trainer(loss_sequence=[
        {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 4.0}
        for _ in range(10)
    ])
    shutdown = FakeShutdown()
    coord = _make_coordinator(
        trainer=trainer,
        shutdown=shutdown,
        config_overrides={"hard_gn_threshold": 3.0, "hard_gn_min_steps": 5},
    )
    out = coord.step()
    assert out.hard_abort_fired is True
    assert shutdown.running is False
    assert coord.consec_high_gn >= 5


def test_hard_abort_resets_on_finite_low(patch_orchestrator_helpers):
    """Single low-GN step between highs resets the counter."""
    trainer = _make_trainer(loss_sequence=[
        {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 4.0},
        {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 4.0},
        {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 4.0},
        {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 0.5},
        {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 4.0},
        {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 4.0},
    ])
    shutdown = FakeShutdown()
    coord = _make_coordinator(
        trainer=trainer,
        shutdown=shutdown,
        config_overrides={"hard_gn_threshold": 3.0, "hard_gn_min_steps": 5,
                          "max_train_burst": 6, "training_steps_per_game": 6.0},
    )
    out = coord.step()
    assert out.hard_abort_fired is False
    assert shutdown.running is True
    # After reset on step 4, only 2 high-GN streak follows
    assert coord.consec_high_gn == 2


# ── B#1c: hard-abort on sustained stride-5 spam (rolling-50 P90) ─────────────

def test_hard_abort_stride5_spam_streak(patch_orchestrator_helpers):
    """3 consecutive eval points with rolling stride5 P90 >= threshold trip
    shutdown.running = False."""
    pool = _make_pool()
    pool.current_stride5_p90 = Mock(return_value=60)  # >= 30 threshold
    shutdown = FakeShutdown()
    coord = _make_coordinator(
        pool=pool,
        shutdown=shutdown,
        config_overrides={"eval_interval": 1, "stride5_p90_threshold": 30.0,
                          "stride5_p90_consec": 3, "max_train_burst": 5,
                          "training_steps_per_game": 5.0},
    )
    out = coord.step()
    assert out.hard_abort_fired is True
    assert shutdown.running is False
    assert coord.consec_high_stride5 >= 3


def test_hard_abort_stride5_resets_on_low(patch_orchestrator_helpers):
    """A single below-threshold eval point resets the consecutive counter, so a
    high/high/low/high/high pattern never reaches 3-in-a-row."""
    pool = _make_pool()
    pool.current_stride5_p90 = Mock(side_effect=[60, 60, 10, 60, 60])
    shutdown = FakeShutdown()
    coord = _make_coordinator(
        pool=pool,
        shutdown=shutdown,
        config_overrides={"eval_interval": 1, "stride5_p90_threshold": 30.0,
                          "stride5_p90_consec": 3, "max_train_burst": 5,
                          "training_steps_per_game": 5.0},
    )
    out = coord.step()
    assert out.hard_abort_fired is False
    assert shutdown.running is True
    assert coord.consec_high_stride5 == 2  # reset at the low, then 2 highs


def test_stride5_gate_off_when_threshold_nonpositive(patch_orchestrator_helpers):
    """threshold <= 0 disables the gate even with extreme rolling P90."""
    pool = _make_pool()
    pool.current_stride5_p90 = Mock(return_value=999)
    shutdown = FakeShutdown()
    coord = _make_coordinator(
        pool=pool,
        shutdown=shutdown,
        config_overrides={"eval_interval": 1, "stride5_p90_threshold": 0.0,
                          "stride5_p90_consec": 3, "max_train_burst": 5,
                          "training_steps_per_game": 5.0},
    )
    out = coord.step()
    assert out.hard_abort_fired is False
    assert shutdown.running is True
    assert coord.consec_high_stride5 == 0


# ── B#2: soft-abort on E-W axis drift ───────────────────────────────────────

def test_soft_abort_ew_window_full_all_above(patch_orchestrator_helpers):
    """Deque full + every entry > threshold → soft_abort_ew_flat fires."""
    # Pre-seed _train_step so eval_interval modulus hits each inner iter
    patch_orchestrator_helpers["axis"].return_value = 0.5  # > 0.4 threshold
    trainer = _make_trainer()
    shutdown = FakeShutdown()
    coord = _make_coordinator(
        trainer=trainer,
        shutdown=shutdown,
        config_overrides={"eval_interval": 1, "soft_ew_threshold": 0.4,
                          "soft_ew_min_pts": 3, "max_train_burst": 5,
                          "training_steps_per_game": 5.0},
    )
    out = coord.step()
    assert out.soft_abort_fired is True
    assert shutdown.running is False
    assert len(coord.ew_history) == 3


def test_soft_abort_ew_window_partial(patch_orchestrator_helpers):
    """Partial window (< min_pts) does not fire soft abort."""
    patch_orchestrator_helpers["axis"].return_value = 0.5
    trainer = _make_trainer()
    shutdown = FakeShutdown()
    coord = _make_coordinator(
        trainer=trainer,
        shutdown=shutdown,
        config_overrides={"eval_interval": 1, "soft_ew_threshold": 0.4,
                          "soft_ew_min_pts": 5, "max_train_burst": 3,
                          "training_steps_per_game": 3.0},
    )
    out = coord.step()
    assert out.soft_abort_fired is False
    assert shutdown.running is True
    # Only 3 inner steps ran, ew_history has 3 entries; min_pts=5 → no fire
    assert len(coord.ew_history) == 3


# ── B#3: buffer growth schedule advancement ─────────────────────────────────

def test_schedule_advances_at_threshold(patch_orchestrator_helpers):
    """Crossing buffer_schedule[idx]['step'] resizes buffer + advances idx."""
    trainer = _make_trainer()
    buffer = _make_buffer(capacity=100)
    schedule = (
        {"step": 0, "capacity": 100},      # idx 0 — already applied (skipped)
        {"step": 2, "capacity": 200},      # idx 1 — fires when train_step >= 2
        {"step": 4, "capacity": 500},      # idx 2 — fires when train_step >= 4
    )
    coord = _make_coordinator(
        trainer=trainer,
        buffer=buffer,
        config_overrides={"buffer_schedule": schedule, "max_train_burst": 5,
                          "training_steps_per_game": 5.0},
    )
    out = coord.step()
    # Both idx 1 (step=2, cap=200) and idx 2 (step=5, cap=500) should have fired
    assert buffer.resize.call_count == 2
    assert coord.schedule_idx == 3
    assert out.buffer_resized == 500  # last fired


def test_schedule_no_double_resize(patch_orchestrator_helpers):
    """Buffer at or above schedule capacity → resize NOT called."""
    trainer = _make_trainer()
    buffer = _make_buffer(capacity=500)  # already at target
    schedule = (
        {"step": 0, "capacity": 100},
        {"step": 2, "capacity": 200},  # 200 < 500 (current), no resize
    )
    coord = _make_coordinator(
        trainer=trainer,
        buffer=buffer,
        config_overrides={"buffer_schedule": schedule, "max_train_burst": 5,
                          "training_steps_per_game": 5.0},
    )
    coord.step()
    # idx advanced past entry but no resize (capacity check guard)
    assert buffer.resize.call_count == 0
    assert coord.schedule_idx == 2


# ── B#4: eval-trigger gating with in-flight thread ──────────────────────────

def test_eval_kicks_off_when_idle(patch_orchestrator_helpers):
    """At eval_interval boundary with no eval thread → spawn new daemon thread."""
    trainer = _make_trainer()
    eval_pipeline = Mock()
    # run_evaluation blocks until the test joins the thread
    eval_pipeline.run_evaluation = Mock(
        side_effect=lambda *a, **kw: {"promoted": False, "step": 1, "wr_best": 0.5},
    )
    coord = _make_coordinator(
        trainer=trainer,
        eval_pipeline=eval_pipeline,
        config_overrides={"eval_interval": 1, "max_train_burst": 1,
                          "training_steps_per_game": 1.0},
    )
    out = coord.step()
    assert out.eval_kicked_off is True
    # Eval thread should be set
    assert coord._eval_thread is not None
    # Wait for thread to finish so test cleans up
    coord._eval_thread.join(timeout=2.0)
    assert eval_pipeline.run_evaluation.call_count == 1


def test_eval_skipped_when_thread_alive(patch_orchestrator_helpers):
    """If previous eval thread is still alive at next eval tick → skip kickoff."""
    trainer = _make_trainer()
    eval_pipeline = Mock()
    eval_pipeline.run_evaluation = Mock(return_value={"promoted": False, "step": 1})
    coord = _make_coordinator(
        trainer=trainer,
        eval_pipeline=eval_pipeline,
        config_overrides={"eval_interval": 1, "max_train_burst": 2,
                          "training_steps_per_game": 2.0},
    )

    # Plant a still-running eval thread; mock drain to leave it alone.
    holding_event = threading.Event()
    def block_until_set() -> None:
        holding_event.wait(timeout=5.0)
    fake_thread = threading.Thread(target=block_until_set, daemon=True)
    fake_thread.start()
    coord._eval_thread = fake_thread
    # drain returns the same thread (still alive, no-op)
    patch_orchestrator_helpers["drain"].return_value = (fake_thread, 0)

    try:
        out = coord.step()
        assert out.eval_skipped_busy is True
        assert out.eval_kicked_off is False
        # Eval pipeline never invoked (thread was alive)
        assert eval_pipeline.run_evaluation.call_count == 0
    finally:
        holding_event.set()
        fake_thread.join(timeout=2.0)


# ── B#5: eval snapshot captures at kickoff, not drain time (R3 / F-016) ─────

def test_eval_snapshot_captures_at_kickoff_not_drain(patch_orchestrator_helpers):
    """The default-arg snapshot in _run_eval captures references AT kickoff.

    Mutating ``coord.eval_model`` or ``coord._best_model_step`` AFTER kickoff
    must NOT propagate into the in-flight eval thread's call to
    ``run_evaluation`` — otherwise F-016 promotion arithmetic breaks.
    """
    trainer = _make_trainer()
    captured: dict[str, Any] = {}
    captured_event = threading.Event()

    def slow_run_eval(model, step, best, *, full_config, best_model_step):
        captured["model"] = model
        captured["best"] = best
        captured["best_model_step"] = best_model_step
        captured_event.set()
        # Block briefly so the test can mutate coord state mid-flight
        time.sleep(0.05)
        return {"promoted": False, "step": step}

    eval_pipeline = Mock()
    eval_pipeline.run_evaluation = Mock(side_effect=slow_run_eval)

    initial_eval_model = Mock(_orig_mod=Mock())
    coord = _make_coordinator(
        trainer=trainer,
        eval_pipeline=eval_pipeline,
        eval_model=initial_eval_model,
        config_overrides={"eval_interval": 1, "max_train_burst": 1,
                          "training_steps_per_game": 1.0},
    )
    # Drain pre-kickoff returns the existing anchor best_model_step (0), not None
    patch_orchestrator_helpers["drain"].return_value = (None, 0)
    coord.step()
    # Wait for the eval thread to capture its bound args
    assert captured_event.wait(timeout=2.0)
    # Mutate coordinator state AFTER kickoff
    coord.eval_model = Mock(_orig_mod=Mock())  # different model
    coord._best_model_step = 9999
    # Wait for eval to finish
    coord._eval_thread.join(timeout=2.0)
    # The eval thread captured the ORIGINAL eval_model + best_model_step at kickoff,
    # not the mutated values.
    assert captured["model"] is initial_eval_model
    assert captured["best_model_step"] == 0  # initial anchor.best_model_step


# ── B#6: steps-budget arithmetic (coordinator wiring) ───────────────────────

def test_steps_budget_clamp_and_round(patch_orchestrator_helpers):
    """coordinator._steps_budget clamps to max_train_burst and rounds new_games."""
    coord = _make_coordinator(
        config_overrides={"training_steps_per_game": 1.5, "max_train_burst": 4},
    )
    assert coord._steps_budget(0) == 1     # max(1, round(0)) = 1
    assert coord._steps_budget(2) == 3     # round(2 * 1.5) = 3
    assert coord._steps_budget(10) == 4    # round(15) clamped to 4


# ── B#7: checkpoint-cadence buffer save skips step 0 ────────────────────────

def test_checkpoint_interval_skips_step_zero(patch_orchestrator_helpers):
    """Even when train_step % checkpoint_interval == 0, train_step=0 is guarded."""
    # Trainer that does NOT increment trainer.step → coord._train_step stays 0
    trainer = _make_trainer(increment=False)
    coord = _make_coordinator(
        trainer=trainer,
        config_overrides={"checkpoint_interval": 1, "max_train_burst": 1,
                          "training_steps_per_game": 1.0},
    )
    coord.step()
    # _try_save_buffer should NOT have been called with checkpoint_interval trigger
    save_calls = patch_orchestrator_helpers["save_buf"].call_args_list
    triggers = [call.args[2] if len(call.args) >= 3 else None for call in save_calls]
    assert "checkpoint_interval" not in triggers


def test_checkpoint_interval_fires_when_step_positive(patch_orchestrator_helpers):
    """train_step > 0 + modulus boundary → buffer save fires."""
    trainer = _make_trainer()  # increments
    coord = _make_coordinator(
        trainer=trainer,
        config_overrides={"checkpoint_interval": 1, "max_train_burst": 1,
                          "training_steps_per_game": 1.0},
    )
    out = coord.step()
    save_calls = patch_orchestrator_helpers["save_buf"].call_args_list
    triggers = [call.args[2] if len(call.args) >= 3 else None for call in save_calls]
    assert "checkpoint_interval" in triggers
    assert out.checkpoint_saved is True


# ── B#8: stop_step early exit inside inner burst ────────────────────────────

def test_stop_step_breaks_inner_burst(patch_orchestrator_helpers):
    """stop_step hits mid-burst → inner loop exits early."""
    trainer = _make_trainer()
    # stop_step=3 → after 3 inner increments, the next iter's check fires break
    coord = _make_coordinator(
        trainer=trainer,
        config_overrides={"stop_step": 3, "max_train_burst": 10,
                          "training_steps_per_game": 10.0},
    )
    out = coord.step()
    # Inner loop should run exactly 3 times (steps 1, 2, 3) — then check at top
    # of 4th iter fires break.
    assert out.steps_run == 3
    assert coord.train_step == 3


# ── B#9: shutdown-signal save-then-break ordering ───────────────────────────

def test_shutdown_signal_save_then_break_order(patch_orchestrator_helpers):
    """shutdown.shutdown_save → save_checkpoint AND save_buffer AND running=False, in order."""
    trainer = _make_trainer()
    shutdown = FakeShutdown()
    shutdown.shutdown_save = True
    coord = _make_coordinator(trainer=trainer, shutdown=shutdown)
    out = coord.step()
    # save_checkpoint called
    assert trainer.save_checkpoint.call_count == 1
    # buffer save called with shutdown_signal trigger
    save_calls = patch_orchestrator_helpers["save_buf"].call_args_list
    assert any(call.args[2] == "shutdown_signal" for call in save_calls)
    # running flipped after the saves
    assert shutdown.running is False
    # No inner steps ran (shutdown took the early branch)
    assert out.steps_run == 0


# ── B#10: tracemalloc cadence swallows exceptions ───────────────────────────

def test_tracemalloc_cadence_swallows_exception(patch_orchestrator_helpers):
    """Exception from tracemalloc.take_snapshot() must NOT propagate from step()."""
    trainer = _make_trainer()
    # Pre-seed trainer.step so train_step=500 mod 500 == 0 fires
    trainer.step = 499
    tm = Mock()
    tm.take_snapshot = Mock(side_effect=RuntimeError("boom"))
    coord = _make_coordinator(
        trainer=trainer,
        tracemalloc_provider=tm,
        config_overrides={"max_train_burst": 1, "training_steps_per_game": 1.0},
    )
    # Should not raise
    coord.step()
    assert tm.take_snapshot.call_count == 1


# ── B#11: instrumentation periodic emits ────────────────────────────────────

def test_instrumentation_periodic_emits(patch_orchestrator_helpers):
    """When instrumentation_enabled and composition_interval boundary hits → events emit."""
    trainer = _make_trainer()
    pool = _make_pool()
    pool.buffer_composition = Mock(return_value={
        "draw_target_fraction": 0.1,
        "colony_terminal_fraction": 0.05,
        "six_terminal_fraction": 0.4,
        "cap_terminal_fraction": 0.45,
    })
    pool.model_version_summary = Mock(return_value={
        "median_range": 1, "p90_range": 2, "spearman_rho_range_vs_draw": 0.0,
    })
    pool.per_worker_draw_rates = Mock(return_value={0: 0.1, 1: 0.2})
    emitter = Mock()
    coord = _make_coordinator(
        trainer=trainer,
        pool=pool,
        event_emitter=emitter,
        config_overrides={"instrumentation_enabled": True,
                          "composition_interval": 1, "value_probe_interval": 999_999,
                          "max_train_burst": 1, "training_steps_per_game": 1.0},
    )
    out = coord.step()
    events = [call.args[0]["event"] for call in emitter.call_args_list]
    assert "buffer_composition" in events
    assert "worker_draw_rate" in events
    assert "model_version_summary" in events
    assert "instrumentation_periodic" in out.instrumentation_emitted


# ── B#12: mcts_pool_overflow soft warning on counter growth ─────────────────

def test_mcts_pool_overflow_warning_on_growth(patch_orchestrator_helpers, monkeypatch):
    """Counter delta > 0 between log-cadence ticks → warning event emitted."""
    trainer = _make_trainer()
    emitter = Mock()
    # Patch engine.mcts_pool_overflow_count to return 5 on first read
    counts = iter([5])
    import engine
    monkeypatch.setattr(engine, "mcts_pool_overflow_count", lambda: next(counts))
    coord = _make_coordinator(
        trainer=trainer,
        event_emitter=emitter,
        config_overrides={"log_interval": 1, "max_train_burst": 1,
                          "training_steps_per_game": 1.0},
    )
    out = coord.step()
    events = [call.args[0]["event"] for call in emitter.call_args_list]
    assert "mcts_pool_overflow" in events
    assert out.pool_overflow_delta == 5


# ── B#13: rolling games-per-hour window (coordinator wiring) ────────────────

def test_games_per_hour_window_pops_old_samples():
    """Coordinator's bound games_per_hour() reads the rolling window correctly."""
    clock = FakeClock(t=0.0)
    pool = _make_pool(games_completed=0)
    coord = _make_coordinator(pool=pool, clock=clock)
    # Pre-seed the rolling window via repeated calls
    coord._games_played = 0
    coord.games_per_hour()
    clock.t = 10.0
    coord._games_played = 100
    coord.games_per_hour()
    clock.t = 70.0
    coord._games_played = 500
    coord.games_per_hour()
    clock.t = 80.0
    coord._games_played = 600
    rate = coord.games_per_hour()
    # After pop, oldest entry (t=0.0) dropped; window has [10,70,80] then [70,80] after pop
    # dt between bounds, dg accordingly
    assert rate > 0


# ── B#14: flush_pending_eval drains promotion on shutdown (D-012 mirror) ────

def test_flush_pending_eval_promotes_when_thread_idle(patch_orchestrator_helpers):
    """flush_pending_eval reads same eval_result instance + promotes anchor."""
    coord = _make_coordinator()
    # Simulate a completed eval that promoted at step 100
    coord._eval_thread = None  # thread already idle
    coord._eval_result[0] = {"promoted": True, "step": 100, "wr_best": 0.6}
    # drain_pending_eval (patched) returns (None, 100) — the new best_model_step
    patch_orchestrator_helpers["drain"].return_value = (None, 100)
    coord.flush_pending_eval()
    # Coordinator's best_model_step rebound from drain's return
    assert coord.best_model_step == 100
    # Drain was called with the coordinator's eval_result instance
    drain_call = patch_orchestrator_helpers["drain"].call_args
    assert drain_call.args[1] is coord._eval_result


def test_flush_pending_eval_swallows_exception(patch_orchestrator_helpers):
    """flush_pending_eval must not propagate drain exceptions (production teardown safety)."""
    coord = _make_coordinator()
    patch_orchestrator_helpers["drain"].side_effect = RuntimeError("boom")
    # Should not raise
    coord.flush_pending_eval()


# ── §EVALGATE-A: final-boundary eval must be drained, not silently dropped ───

def test_flush_drains_inflight_final_boundary_eval():
    """The eval at the FINAL step boundary is kicked off as a daemon thread and is
    drained only at the NEXT boundary.  When the run stops at that boundary
    (``stop_step``), there is no next boundary, so ``flush_pending_eval`` is the only
    chance to consume it — but ``drain_pending_eval`` no-ops on a still-running thread
    (``eval_drain.py``), so flush must JOIN the in-flight eval first.  With the buggy
    default (``final_eval_drain_timeout_sec=0.0``, set in no config) the join was
    skipped and the final eval was silently lost.  The production default must join,
    so the final eval's result is consumed + surfaced (``eval_complete``).

    No web dashboard is wired here (== ``--no-web-dashboard``): the eval fires + drains
    independent of the dashboard.
    """
    started = threading.Event()

    def _run_evaluation(model, step, best, full_config=None, best_model_step=None):
        started.set()
        time.sleep(0.2)  # still mid-eval at the immediate post-loop flush
        return {"promoted": False, "step": int(step), "wr_sealbot": 0.3, "eval_games": 10}

    eval_pipeline = Mock()
    eval_pipeline.run_evaluation = Mock(side_effect=_run_evaluation)

    emitted: list[dict[str, Any]] = []
    with patch("hexo_rl.training.step_coordinator._emit_axis_distribution", return_value=None), \
         patch("hexo_rl.training.step_coordinator._emit_training_events", return_value=None), \
         patch("hexo_rl.training.step_coordinator._try_save_buffer", return_value=None), \
         patch("hexo_rl.training.eval_drain.emit_event", side_effect=emitted.append):
        coord = _make_coordinator(
            eval_pipeline=eval_pipeline,
            iterations=1,
            config_overrides={
                "eval_interval": 1,
                "max_train_burst": 1,
                "stop_step": 1,
                "final_eval_drain_timeout_sec": DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC,
            },
        )
        # step 1 crosses the boundary and kicks off the eval; step 2 hits the
        # iteration-limit gate (O2) and stops the run.
        for _ in range(5):
            if not coord.shutdown.running:
                break
            coord.step()
        assert started.wait(timeout=3.0), "final-boundary eval never kicked off"
        # eval thread is mid-eval here — the final-boundary case the bug drops.
        coord.flush_pending_eval()

    assert any(
        e.get("event") == "eval_complete" and e.get("step") == 1 for e in emitted
    ), "final-boundary eval result was dropped (flush did not join the in-flight eval)"
    assert coord._eval_thread is None, "in-flight eval thread not joined/consumed"


def test_step_aborts_loudly_when_selfplay_producer_dead() -> None:
    """F02: a dead self-play buffer feeder must fail-fast.  step() polls pool
    health every iteration and re-raises loudly instead of silently continuing
    to train on a stale buffer."""
    pool = _make_pool(games_completed=5)
    pool.check_producer_health = Mock(
        side_effect=RuntimeError("self-play buffer feeder died")
    )
    coord = _make_coordinator(pool=pool)
    with pytest.raises(RuntimeError, match="self-play buffer feeder died"):
        coord.step()
    pool.check_producer_health.assert_called()
