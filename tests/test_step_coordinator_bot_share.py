"""§178 step_coordinator n_pre/n_bot/n_self split + refresh-hook cells.

Cells:
  A — bot_buffer=None → identical n_pre/n_self split as pre-T5 (back-compat).
  B — bot_buffer populated, share=0.15, batch_size=256 → n_bot=38; at step 10K
      w_pre≈0.761 → n_pre=166, n_self=52 (design §3 numerics).
  C — n_pretrain=n_pre+n_bot passed to train_step_from_tensors (aux-mask pin).

  T7 refresh hook cells (added in §178 T7 commit):
    Refresh-A — enabled=False → hook does NOT trigger (no warning).
    Refresh-B — enabled=True + promoted + cooldown elapsed → warning emitted,
                _last_bot_refresh_step bumped.
    Refresh-C — cooldown not elapsed → no fire (idempotent within window).
"""
from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from hexo_rl.training.step_coordinator import (
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
        "batch_size": 256,
        "augment": False,
        "recency_weight": 0.0,
        "mixing_initial_w": 0.8,
        "mixing_min_w": 0.1,
        "mixing_decay_steps": 200_000.0,
        "soft_ew_threshold": 0.0,
        "soft_ew_min_pts": 0,
        "hard_gn_threshold": 3.0,
        "hard_gn_min_steps": 5,
        "instrumentation_enabled": False,
        "stop_step": None,
        "final_eval_drain_timeout_sec": 0.0,
        "bot_batch_share": 0.0,
        "bot_corpus_refresh_enabled": False,
        "bot_corpus_refresh_cooldown": 25_000,
    }
    defaults.update(overrides)
    return StepCoordinatorConfig(**defaults)


def _make_trainer() -> Mock:
    trainer = Mock()
    trainer.step = 0
    loss = {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 0.1}

    def side_effect(*_a: Any, **_kw: Any) -> dict[str, float]:
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
    bot_buffer: Any = None,
    eval_pipeline: Any = None,
    eval_model: Any = None,
    shutdown: FakeShutdown | None = None,
    clock: FakeClock | None = None,
    tracemalloc_provider: Any = None,
    config_overrides: dict[str, Any] | None = None,
    event_emitter: Any = None,
    train_step_override: int | None = None,
    **kwargs: Any,
) -> StepCoordinator:
    cfg = _make_config(**(config_overrides or {}))
    _subsystems = Mock(
        gpu_monitor=Mock(gpu_util_pct=10.0),
        early_game_probe=None,
        value_probe=None,
        axis_baseline={},
        tb_writer=None,
    )
    _trainer = trainer or _make_trainer()
    if train_step_override is not None:
        _trainer.step = train_step_override
    coord = StepCoordinator(
        trainer=_trainer,
        buffer=buffer or _make_buffer(),
        pretrained_buffer=pretrained_buffer,
        bot_buffer=bot_buffer,
        recent_buffer=None,
        pool=pool or _make_pool(),
        eval_pipeline=eval_pipeline,
        subsystems=_subsystems,
        anchor_state=_make_anchor(),
        shutdown=shutdown or FakeShutdown(),
        eval_model=eval_model if eval_model is not None else Mock(_orig_mod=Mock()),
        bufs=Mock(),
        config=cfg,
        full_config={"monitors": {}},
        train_cfg={"batch_size": 256},
        mcts_config={},
        mixing_cfg={"buffer_persist": False},
        batch_size_cfg=256,
        iterations=None,
        clock=clock or FakeClock(),
        tracemalloc_provider=tracemalloc_provider or Mock(),
        event_emitter=event_emitter or Mock(),
        **kwargs,
    )
    coord.last_train_game_count = 0
    if coord.pool.games_completed == 0:
        coord.pool.games_completed = 1
    return coord


@pytest.fixture
def patch_helpers():
    with patch("hexo_rl.training.step_coordinator._emit_axis_distribution") as axis, \
         patch("hexo_rl.training.step_coordinator._emit_training_events") as train_ev, \
         patch("hexo_rl.training.step_coordinator._try_save_buffer") as save_buf, \
         patch("hexo_rl.training.step_coordinator._drain_pending_eval") as drain, \
         patch("hexo_rl.training.step_coordinator.assemble_mixed_batch") as mix:
        axis.return_value = None
        drain.return_value = (None, None)
        train_ev.return_value = None
        save_buf.return_value = None
        # Mock batch return — minimal BatchAssemblyResult-like
        mix_batch = Mock()
        mix_batch.states = Mock()
        mix_batch.chain_planes = Mock()
        mix_batch.policies = Mock()
        mix_batch.outcomes = Mock()
        mix_batch.ownership = Mock()
        mix_batch.winning_line = Mock()
        mix_batch.is_full_search = Mock()
        mix_batch.n_recent_actual = 0
        mix.return_value = mix_batch
        yield {"axis": axis, "train_ev": train_ev, "save_buf": save_buf,
               "drain": drain, "mix": mix}


# ── Cell A: bot_buffer=None back-compat regression ────────────────────────────

def test_bot_buffer_none_uses_legacy_split(patch_helpers):
    """bot_buffer=None: n_pre/n_self split = ceil(w_pre*batch_size); n_bot omitted from assemble."""
    pretrained_buffer = _make_buffer(size=100_000)
    coord = _make_coordinator(
        pretrained_buffer=pretrained_buffer,
        bot_buffer=None,
        config_overrides={"max_train_burst": 1, "training_steps_per_game": 1.0,
                          "bot_batch_share": 0.0},
    )
    coord.step()
    # assemble_mixed_batch was called; verify kwargs
    call = patch_helpers["mix"].call_args
    assert call.kwargs["bot_buffer"] is None
    assert call.kwargs["n_bot"] == 0
    # positional n_pre + n_self == batch_size
    n_pre = call.args[3]
    n_self = call.args[4]
    batch_size = call.args[5]
    assert n_pre + n_self == batch_size


# ── Cell B: numerics at step 10K with bot_batch_share=0.15 ───────────────────

def test_bot_buffer_step_10k_numerics(patch_helpers):
    """At step 10K w_pre≈0.761 → n_bot=38, n_pre=166, n_self=52 (design §3 table)."""
    pretrained_buffer = _make_buffer(size=100_000)
    bot_buffer = _make_buffer(size=50_000)
    coord = _make_coordinator(
        pretrained_buffer=pretrained_buffer,
        bot_buffer=bot_buffer,
        train_step_override=10_000,
        config_overrides={"max_train_burst": 1, "training_steps_per_game": 1.0,
                          "bot_batch_share": 0.15, "batch_size": 256,
                          "mixing_initial_w": 0.8, "mixing_min_w": 0.1,
                          "mixing_decay_steps": 200_000.0},
    )
    coord.step()
    call = patch_helpers["mix"].call_args
    n_pre, n_self, batch_size = call.args[3], call.args[4], call.args[5]
    n_bot = call.kwargs["n_bot"]

    # design §3 expectations
    assert n_bot == 38, f"n_bot expected 38 got {n_bot}"
    # w_pre(10000) = max(0.1, 0.8 * exp(-10000/200000)) = 0.8 * exp(-0.05) ≈ 0.7610
    w_pre_expected = max(0.1, 0.8 * math.exp(-10_000 / 200_000))
    n_pre_expected = max(1, math.ceil(w_pre_expected * (256 - 38)))
    assert n_pre == n_pre_expected, (
        f"n_pre expected {n_pre_expected} (w_pre≈{w_pre_expected:.4f}) got {n_pre}"
    )
    # contract numerics: n_pre=166
    assert n_pre == 166
    assert n_self == 256 - n_pre - n_bot
    assert n_self == 52
    assert n_pre + n_bot + n_self == batch_size
    # bot_buffer threaded through
    assert call.kwargs["bot_buffer"] is bot_buffer


# ── Cell C: n_pretrain = n_pre + n_bot threaded to trainer ────────────────────

def test_n_pretrain_extends_through_bot_rows(patch_helpers):
    """train_step_from_tensors must receive n_pretrain = n_pre + n_bot (aux-mask pin)."""
    pretrained_buffer = _make_buffer(size=100_000)
    bot_buffer = _make_buffer(size=50_000)
    trainer = _make_trainer()
    coord = _make_coordinator(
        trainer=trainer,
        pretrained_buffer=pretrained_buffer,
        bot_buffer=bot_buffer,
        train_step_override=10_000,
        config_overrides={"max_train_burst": 1, "training_steps_per_game": 1.0,
                          "bot_batch_share": 0.15, "batch_size": 256,
                          "mixing_decay_steps": 200_000.0},
    )
    coord.step()
    mix_call = patch_helpers["mix"].call_args
    n_pre, n_bot = mix_call.args[3], mix_call.kwargs["n_bot"]
    expected_n_pretrain = n_pre + n_bot
    train_call = trainer.train_step_from_tensors.call_args
    assert train_call.kwargs["n_pretrain"] == expected_n_pretrain, (
        f"n_pretrain must be n_pre+n_bot ({expected_n_pretrain}); "
        f"got {train_call.kwargs['n_pretrain']} — aux mask would leak bot rows into aux loss"
    )
    # Specifically: bot rows are NOT masked out if n_pretrain==n_pre (regression detector)
    assert train_call.kwargs["n_pretrain"] != n_pre, (
        "REGRESSION: n_pretrain==n_pre would feed bot rows (with neutral aux pad) "
        "into aux losses, biasing the spatial heads toward empty/no-threat outputs."
    )


# ── Cell C2: bot_buffer with size==0 falls through ────────────────────────────

def test_bot_buffer_empty_size_falls_to_zero(patch_helpers):
    """bot_buffer.size==0 → n_bot=0 (same path as bot_buffer=None)."""
    pretrained_buffer = _make_buffer(size=100_000)
    bot_buffer = _make_buffer(size=0)
    coord = _make_coordinator(
        pretrained_buffer=pretrained_buffer,
        bot_buffer=bot_buffer,
        config_overrides={"max_train_burst": 1, "training_steps_per_game": 1.0,
                          "bot_batch_share": 0.15},
    )
    coord.step()
    call = patch_helpers["mix"].call_args
    assert call.kwargs["n_bot"] == 0
