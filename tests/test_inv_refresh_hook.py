"""§S181-AUDIT Wave 3 Stage 2A INV pins: bot-corpus refresh hook.

Activates the inert refresh hook wired since §178 per the s179c design
(``docs/designs/s179c_bot_refresh_hook.md``). Addresses L51 (bot-corpus
staleness, predicted Track D C4, confirmed Wave 2 main-run collapse
33%→5% step 20k→40k).

INV pins enforced:

- INV-S179c-1 — refresh never touches human corpus (``pretrained_buffer``)
  or selfplay buffer (``buffer``); identity stable across refresh.
- INV-S179c-2 — refresh tmp path on same filesystem as canonical NPZ;
  config-load time guard rejects cross-FS configurations.
- INV-S179c-3 — hot-reload swap site is post-eval-drain pre-next-iteration;
  swap never occurs inside batch-assembly window (id stable within step).
- INV-S179c-4 — refresh-disabled run is bitwise identical to master
  (no state init beyond None placeholders, no event emit, no Popen).
- INV-S179c-5 — sidecar SHA verify on post-rename swap; mismatch quarantines
  the tmp NPZ and preserves the canonical NPZ untouched.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from hexo_rl.bootstrap.corpus_io import save_corpus
from hexo_rl.training.batch_assembly import (
    BotCorpusSwapError,
    swap_bot_corpus_atomic,
)
from hexo_rl.training.step_coordinator import (
    StepCoordinator,
    StepCoordinatorConfig,
)


# ── Shared fakes (parallel to test_step_coordinator_bot_share.py) ─────────────

class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, sec: float) -> None:
        self.t += sec


class _FakeShutdown:
    def __init__(self) -> None:
        self.running = True
        self.shutdown_save = False


def _make_config(**overrides) -> StepCoordinatorConfig:
    defaults = {
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
        "bot_corpus_refresh_interval_steps": 5_000,
        "bot_corpus_refresh_n_games": 200,
        "bot_corpus_refresh_opponent_model": "ema",
        "bot_corpus_refresh_replace_strategy": "rolling_window",
        "bot_corpus_refresh_max_regens": 20,
        "bot_corpus_refresh_min_wr_delta": 0.0,
        "bot_corpus_refresh_max_plies": 150,
        "bot_corpus_refresh_random_opening_plies": 4,
        "bot_corpus_refresh_think_seconds": 0.5,
        "bot_corpus_refresh_anchor_n_sims": 200,
        "bot_corpus_refresh_anchor_temperature": 0.5,
        "bot_corpus_path": "",
    }
    defaults.update(overrides)
    return StepCoordinatorConfig(**defaults)


def _make_buffer(size: int = 1000, capacity: int = 1000) -> Mock:
    b = Mock()
    b.size = size
    b.capacity = capacity
    b.resize = Mock()
    b.save_to_path = Mock()
    return b


def _make_trainer() -> Mock:
    t = Mock()
    t.step = 0
    loss = {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 0.1}

    def side_effect(*a, **kw):
        t.step += 1
        return loss
    t.train_step = Mock(side_effect=side_effect)
    t.train_step_from_tensors = Mock(side_effect=side_effect)
    t.save_checkpoint = Mock(return_value="/tmp/ckpt")
    t.model = Mock()
    t.model._orig_mod = t.model
    return t


def _make_pool() -> Mock:
    pool = Mock()
    pool.games_completed = 1
    pool.n_workers = 1
    from hexo_rl.selfplay.pool import RunnerStats, InferenceStats
    pool._runner = Mock(mcts_quiescence_fires=0, model_version=0)
    pool._inference_server = Mock()
    rs = RunnerStats(
        games_completed=1, positions_generated=0,
        x_wins=0, o_wins=0, draws=0, model_version=0,
        mcts_quiescence_fires=0, mcts_mean_depth=0.0,
        mcts_mean_root_concentration=0.0, cluster_value_std_mean=0.0,
        cluster_policy_disagreement_mean=0.0, cluster_variance_sample_count=0,
        runner_encoding=None,
    )
    pool.runner_stats = Mock(return_value=rs)
    pool.inference_stats = Mock(return_value=InferenceStats(
        forward_count=0, total_requests=0, encoding_spec=None,
    ))
    pool.sync_inference_weights = Mock()
    pool.recent_buffer = None
    return pool


def _make_anchor() -> Mock:
    a = Mock()
    a.best_model = Mock()
    a.best_model._orig_mod = a.best_model
    a.best_model_step = 0
    a.best_model_path = "/tmp/best_model.pt"
    return a


def _make_coordinator(
    *,
    bot_buffer=None,
    pretrained_buffer=None,
    config_overrides=None,
    train_step_override=None,
    logger=None,
    event_emitter=None,
    bot_corpus_path: str = "",
) -> StepCoordinator:
    cfg = _make_config(
        bot_corpus_path=bot_corpus_path,
        **(config_overrides or {}),
    )
    subsystems = Mock(
        gpu_monitor=Mock(gpu_util_pct=10.0),
        early_game_probe=None, value_probe=None,
        axis_baseline={}, tb_writer=None,
    )
    trainer = _make_trainer()
    if train_step_override is not None:
        trainer.step = train_step_override
    coord = StepCoordinator(
        trainer=trainer,
        buffer=_make_buffer(),
        pretrained_buffer=pretrained_buffer,
        bot_buffer=bot_buffer,
        recent_buffer=None,
        pool=_make_pool(),
        eval_pipeline=None,
        subsystems=subsystems,
        anchor_state=_make_anchor(),
        shutdown=_FakeShutdown(),
        eval_model=Mock(_orig_mod=Mock()),
        bufs=Mock(),
        config=cfg,
        full_config={"monitors": {}},
        train_cfg={"batch_size": 256},
        mcts_config={},
        mixing_cfg={"buffer_persist": False, "bot_corpus_path": bot_corpus_path},
        batch_size_cfg=256,
        iterations=None,
        clock=_FakeClock(),
        tracemalloc_provider=Mock(),
        event_emitter=event_emitter or Mock(),
        logger=logger or Mock(),
    )
    return coord


# ── INV-S179c-1: refresh hook never touches human/selfplay buffers ────────────

def test_inv_s179c_1_refresh_does_not_touch_human_or_selfplay_buffer():
    """The refresh code path must hold ``pretrained_buffer`` and ``buffer``
    identity constant. Only ``bot_buffer`` may be reassigned.
    """
    pre_buf = _make_buffer(size=10_000, capacity=10_000)
    bot_buf = _make_buffer(size=5_000, capacity=5_000)
    coord = _make_coordinator(
        pretrained_buffer=pre_buf,
        bot_buffer=bot_buf,
    )
    # Capture identities pre-refresh.
    id_pretrain = id(coord.pretrained_buffer)
    id_selfplay = id(coord.buffer)

    # Invoke the swap+hot-reload helper directly with mock helpers so it
    # exercises the full code path WITHOUT requiring a real subprocess /
    # NPZ. The helper itself must NEVER touch pretrained / self-play.
    fake_canonical = Path("/tmp/fake_canonical.npz")
    coord._refresh_tmp_npz_path = Path("/tmp/fake_canonical.npz.NEW.tmp.npz")
    with patch(
        "hexo_rl.training.batch_assembly.load_bot_corpus_buffer",
        return_value=_make_buffer(size=6_000, capacity=6_000),
    ), patch(
        "hexo_rl.training.batch_assembly.swap_bot_corpus_atomic",
        return_value=("oldsha", "newsha"),
    ):
        coord._swap_and_hot_reload_bot_corpus(fake_canonical)

    assert id(coord.pretrained_buffer) == id_pretrain, (
        "INV-S179c-1 VIOLATION: pretrained_buffer reassigned by refresh hook"
    )
    assert id(coord.buffer) == id_selfplay, (
        "INV-S179c-1 VIOLATION: self-play buffer reassigned by refresh hook"
    )
    # bot_buffer IS expected to change (that's the whole point).
    assert coord.bot_buffer is not bot_buf


# ── INV-S179c-2: tmp path on same FS as canonical (parent-dir guard) ──────────

def test_inv_s179c_2_resolve_path_fails_when_parent_missing(tmp_path):
    """Resolve canonical path raises when the parent dir does not exist.

    The atomic-swap rename requires same-FS (POSIX guarantee). The parent-
    dir guard is the config-load-time defence: caller must place the
    canonical NPZ where the runtime can host the tmp file next to it.
    """
    bogus_canonical = tmp_path / "nonexistent_parent_dir" / "fake.npz"
    coord = _make_coordinator(bot_corpus_path=str(bogus_canonical))
    with pytest.raises(RuntimeError, match="parent dir missing"):
        coord._resolve_canonical_bot_path()


def test_inv_s179c_2_resolve_path_succeeds_when_parent_exists(tmp_path):
    """When the parent dir exists, resolution returns the canonical path."""
    canonical = tmp_path / "subdir" / "fake.npz"
    canonical.parent.mkdir()
    coord = _make_coordinator(bot_corpus_path=str(canonical))
    resolved = coord._resolve_canonical_bot_path()
    assert resolved is not None
    assert resolved.parent == tmp_path / "subdir"


# ── INV-S179c-3: swap site is post-eval-drain, pre-next-iteration ─────────────

def test_inv_s179c_3_bot_buffer_identity_stable_within_one_step(tmp_path):
    """``id(self.bot_buffer)`` is stable throughout a single ``step()`` call.

    The refresh swap fires inside the eval-drain block AFTER the
    batch-assembly read. Within one outer step, ``self.bot_buffer`` MUST
    NOT be reassigned by a concurrent refresh tick. The test monkey-
    patches assemble_mixed_batch to record id(self.bot_buffer) and
    verifies it never changes between sample_batch start and end.
    """
    pre_buf = _make_buffer(size=10_000, capacity=10_000)
    bot_buf = _make_buffer(size=5_000, capacity=5_000)
    coord = _make_coordinator(
        pretrained_buffer=pre_buf,
        bot_buffer=bot_buf,
        config_overrides={
            "max_train_burst": 1,
            "training_steps_per_game": 1.0,
            "bot_batch_share": 0.15,
            "bot_corpus_refresh_enabled": True,
            "bot_corpus_refresh_interval_steps": 5_000,
        },
        train_step_override=0,
        bot_corpus_path=str(tmp_path / "fake.npz"),
    )
    coord.last_train_game_count = 0

    observed_ids: list[int] = []

    def record_id(*args, **kwargs):
        observed_ids.append(id(coord.bot_buffer))
        mix_result = Mock()
        mix_result.states = Mock()
        mix_result.chain_planes = Mock()
        mix_result.policies = Mock()
        mix_result.outcomes = Mock()
        mix_result.ownership = Mock()
        mix_result.winning_line = Mock()
        mix_result.is_full_search = Mock()
        mix_result.n_recent_actual = 0
        return mix_result

    with patch(
        "hexo_rl.training.step_coordinator.assemble_mixed_batch",
        side_effect=record_id,
    ), patch(
        "hexo_rl.training.step_coordinator._emit_axis_distribution",
        return_value=None,
    ), patch(
        "hexo_rl.training.step_coordinator._emit_training_events",
    ), patch(
        "hexo_rl.training.step_coordinator._try_save_buffer",
    ), patch(
        "hexo_rl.training.step_coordinator._drain_pending_eval",
        return_value=(None, None),
    ):
        coord.step()
    # Identity must be stable: all sampling within this step saw the same buffer.
    assert observed_ids, "assemble_mixed_batch was never invoked in step()"
    assert len(set(observed_ids)) == 1, (
        f"INV-S179c-3 VIOLATION: bot_buffer identity changed within step(): {observed_ids}"
    )


# ── INV-S179c-4: refresh-disabled is bitwise identical to master ──────────────

def test_inv_s179c_4_disabled_init_state_is_minimal():
    """With ``enabled: false`` the refresh state init is purely placeholders.

    No subprocess module imports, no torch.save, no Popen. State variables
    initialised to None/0/False so the disabled path is zero-overhead.
    """
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        config_overrides={"bot_corpus_refresh_enabled": False},
    )
    assert coord._refresh_proc is None
    assert coord._refresh_started_step == 0
    assert coord._refresh_target_anchor_sha is None
    assert coord._refresh_ema_snapshot_path is None
    assert coord._refresh_tmp_npz_path is None
    assert coord._n_refreshes_so_far == 0
    assert coord._force_bot_refresh is False


def test_inv_s179c_4_disabled_step_emits_no_refresh_events():
    """``enabled: false`` step path emits zero ``bot_corpus_*`` events.

    The dashboard event bus must see NO refresh-related events when the
    hook is disabled — proof the activation surface is fully bypassed.
    """
    event_emitter = Mock()
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        config_overrides={
            "bot_corpus_refresh_enabled": False,
            "max_train_burst": 1,
            "training_steps_per_game": 1.0,
        },
        event_emitter=event_emitter,
    )
    coord.last_train_game_count = 0
    with patch(
        "hexo_rl.training.step_coordinator.assemble_mixed_batch",
        return_value=Mock(
            states=Mock(), chain_planes=Mock(), policies=Mock(),
            outcomes=Mock(), ownership=Mock(), winning_line=Mock(),
            is_full_search=Mock(), n_recent_actual=0,
        ),
    ), patch(
        "hexo_rl.training.step_coordinator._emit_axis_distribution",
        return_value=None,
    ), patch(
        "hexo_rl.training.step_coordinator._emit_training_events",
    ), patch(
        "hexo_rl.training.step_coordinator._try_save_buffer",
    ), patch(
        "hexo_rl.training.step_coordinator._drain_pending_eval",
        return_value=(None, None),
    ):
        coord.step()
    bot_events = [
        c for c in event_emitter.call_args_list
        if c.args and isinstance(c.args[0], dict)
        and c.args[0].get("event", "").startswith("bot_corpus_")
    ]
    assert not bot_events, (
        f"INV-S179c-4 VIOLATION: disabled refresh hook emitted events: {bot_events}"
    )


# ── INV-S179c-5: sha verify on post-rename swap ───────────────────────────────

def _write_fixture_npz(path: Path, n: int = 4) -> None:
    """Write a minimal valid sidecar'd NPZ at ``path`` for swap tests."""
    states = np.zeros((n, 8, 19, 19), dtype=np.float16)
    policies = np.zeros((n, 362), dtype=np.float32)
    outcomes = np.zeros((n,), dtype=np.float32)
    save_corpus(
        path,
        arrays={"states": states, "policies": policies, "outcomes": outcomes},
        encoding_name="v6",
        source_manifest="test_inv_refresh_hook fixture",
    )


def test_inv_s179c_5_swap_quarantines_on_sha_mismatch(tmp_path):
    """When the tmp NPZ's actual sha differs from the sidecar's declared sha,
    the swap MUST reject (quarantine .corrupt-<ts>) and preserve canonical.
    """
    canonical = tmp_path / "canonical.npz"
    tmp = tmp_path / "canonical.npz.NEW.tmp.npz"
    _write_fixture_npz(canonical, n=2)
    canonical_pre_sha = canonical.read_bytes()
    _write_fixture_npz(tmp, n=4)

    # Tamper the sidecar's declared sha (force mismatch).
    sidecar = tmp.with_name(tmp.name + ".metadata.json")
    meta = json.loads(sidecar.read_text())
    meta["sha256"] = "deadbeef" * 8
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(BotCorpusSwapError, match="sha mismatch"):
        swap_bot_corpus_atomic(canonical, tmp)

    # Canonical retained byte-for-byte; .bak was NOT created (no rotate fired).
    assert canonical.exists()
    assert canonical.read_bytes() == canonical_pre_sha
    assert not canonical.with_suffix(".npz.bak").exists()

    # Quarantine file exists with corrupt suffix (NPZ + sidecar both moved).
    npz_corrupts = [
        p for p in tmp_path.glob("canonical.npz.NEW.tmp.npz.corrupt-*")
        if not p.name.endswith(".metadata.json")
    ]
    assert len(npz_corrupts) == 1, (
        f"expected one quarantined NPZ; found {npz_corrupts}"
    )
    # And the tmp itself is gone (renamed into the quarantine).
    assert not tmp.exists()


def test_inv_s179c_5_swap_commits_on_valid_sha(tmp_path):
    """Happy path: sha matches → canonical replaced + .bak retained."""
    canonical = tmp_path / "canonical.npz"
    tmp = tmp_path / "canonical.npz.NEW.tmp.npz"
    _write_fixture_npz(canonical, n=2)
    _write_fixture_npz(tmp, n=4)
    canonical_pre_bytes = canonical.read_bytes()

    old_sha, new_sha = swap_bot_corpus_atomic(canonical, tmp)
    # Both shas non-empty.
    assert len(old_sha) == 64
    assert len(new_sha) == 64
    assert old_sha != new_sha  # different fixtures
    # canonical now == old tmp content.
    assert canonical.exists()
    assert not tmp.exists()
    # .bak retains pre-swap canonical.
    bak = canonical.with_suffix(".npz.bak")
    assert bak.exists()
    assert bak.read_bytes() == canonical_pre_bytes
    # Sidecar replaced.
    assert canonical.with_name(canonical.name + ".metadata.json").exists()


def test_inv_s179c_5_swap_fails_on_missing_tmp(tmp_path):
    """Missing tmp NPZ → BotCorpusSwapError; canonical untouched."""
    canonical = tmp_path / "canonical.npz"
    _write_fixture_npz(canonical, n=2)
    bogus_tmp = tmp_path / "does_not_exist.npz.NEW.tmp.npz"
    pre_bytes = canonical.read_bytes()
    with pytest.raises(BotCorpusSwapError, match="tmp NPZ missing"):
        swap_bot_corpus_atomic(canonical, bogus_tmp)
    assert canonical.read_bytes() == pre_bytes


def test_inv_s179c_5_swap_fails_on_missing_sidecar(tmp_path):
    """Tmp NPZ present but sidecar missing → BotCorpusSwapError."""
    canonical = tmp_path / "canonical.npz"
    tmp = tmp_path / "canonical.npz.NEW.tmp.npz"
    _write_fixture_npz(canonical, n=2)
    _write_fixture_npz(tmp, n=4)
    sidecar = tmp.with_name(tmp.name + ".metadata.json")
    sidecar.unlink()
    pre_bytes = canonical.read_bytes()
    with pytest.raises(BotCorpusSwapError, match="sidecar missing"):
        swap_bot_corpus_atomic(canonical, tmp)
    assert canonical.read_bytes() == pre_bytes


# ── Failure-path: subprocess non-zero exit ────────────────────────────────────

def test_subprocess_nonzero_exit_logged_no_state_corruption(tmp_path):
    """Subprocess returns rc != 0 → log + clear refresh state; canonical
    untouched; n_refreshes_so_far NOT incremented (retry possible)."""
    canonical = tmp_path / "canonical.npz"
    _write_fixture_npz(canonical, n=2)
    pre_bytes = canonical.read_bytes()

    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        config_overrides={
            "bot_corpus_refresh_enabled": True,
            "bot_corpus_refresh_interval_steps": 5_000,
        },
        bot_corpus_path=str(canonical),
    )
    # Simulate an in-flight subprocess that already exited non-zero.
    fake_proc = Mock()
    fake_proc.poll = Mock(return_value=2)  # rc=2
    fake_proc.stderr = None
    coord._refresh_proc = fake_proc
    coord._refresh_started_step = 10_000
    coord._refresh_ema_snapshot_path = tmp_path / "snapshot.pt"
    coord._refresh_tmp_npz_path = tmp_path / "canonical.npz.NEW.tmp.npz"
    coord._n_refreshes_so_far = 0
    coord._train_step = 15_000

    coord._tick_bot_refresh()

    # Refresh state cleared.
    assert coord._refresh_proc is None
    assert coord._refresh_target_anchor_sha is None
    # n_refreshes NOT incremented on failure.
    assert coord._n_refreshes_so_far == 0
    # Canonical NPZ untouched.
    assert canonical.read_bytes() == pre_bytes


# ── Failure-path: max_regens cap enforced ─────────────────────────────────────

def test_max_regens_cap_blocks_further_fires(tmp_path):
    """Once ``n_refreshes_so_far >= max_regens`` no further launches fire."""
    canonical = tmp_path / "canonical.npz"
    _write_fixture_npz(canonical, n=2)
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        config_overrides={
            "bot_corpus_refresh_enabled": True,
            "bot_corpus_refresh_interval_steps": 100,
            "bot_corpus_refresh_max_regens": 3,
        },
        bot_corpus_path=str(canonical),
        train_step_override=100_000,
    )
    coord._n_refreshes_so_far = 3  # cap reached
    with patch.object(coord, "_launch_refresh_subprocess") as launch_mock:
        coord._tick_bot_refresh()
        launch_mock.assert_not_called()


# ── Force-trigger sentinel cell ───────────────────────────────────────────────

def test_force_refresh_sentinel_overrides_interval_gate(tmp_path):
    """Operator-drop sentinel file overrides the interval gate."""
    canonical = tmp_path / "canonical.npz"
    _write_fixture_npz(canonical, n=2)
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        config_overrides={
            "bot_corpus_refresh_enabled": True,
            "bot_corpus_refresh_interval_steps": 50_000,
        },
        bot_corpus_path=str(canonical),
        train_step_override=100,  # well inside the interval window
    )
    coord._force_bot_refresh = True
    with patch.object(coord, "_launch_refresh_subprocess") as launch_mock:
        coord._tick_bot_refresh()
        launch_mock.assert_called_once()


def test_force_sentinel_consumed_on_poll(tmp_path):
    """Sentinel file is unlinked + flag set when polled."""
    sentinel = StepCoordinator._FORCE_REFRESH_SENTINEL
    # Best-effort: skip if cannot write to /tmp.
    try:
        sentinel.write_text("touch")
    except OSError:
        pytest.skip("cannot write to /tmp force-refresh sentinel path")

    canonical = tmp_path / "canonical.npz"
    _write_fixture_npz(canonical, n=2)
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        config_overrides={"bot_corpus_refresh_enabled": True},
        bot_corpus_path=str(canonical),
    )
    assert coord._force_bot_refresh is False
    coord._poll_force_refresh_sentinel()
    assert coord._force_bot_refresh is True
    assert not sentinel.exists(), "sentinel should be unlinked after consume"
