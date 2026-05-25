"""§S181-AUDIT Wave 3 Stage 2A — bot-corpus refresh hook behaviour tests.

Lives alongside other ``hexo_rl/training/tests/`` files (test_ema.py
template). Covers the refresh-hook code paths NOT exercised by the
INV pins in ``tests/test_inv_refresh_hook.py``:

  - subprocess command composition (TC4 contract: --anchor receives EMA
    snapshot path, NOT bootstrap model path);
  - EMA-vs-raw snapshot selection driven by
    ``cfg.bot_corpus_refresh_opponent_model``;
  - state transitions across a full launch → poll → swap → reset cycle
    when the subprocess "completes" via mock.

Each test uses heavy mocking to avoid real Popen / torch.save / NPZ I/O
(those paths are covered by the INV pins via integration-style fixtures).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# Local re-imports to keep this test self-contained (helpers in the parent
# tests/ tree are not on the package import path).

from tests.test_inv_refresh_hook import (  # noqa: E402
    _make_buffer,
    _make_coordinator,
)


def test_subprocess_command_routes_anchor_to_ema_snapshot(tmp_path):
    """``_build_refresh_subprocess_command`` puts the EMA snapshot path on
    the --anchor flag, not the original bootstrap model path.
    """
    canonical = tmp_path / "canonical.npz"
    canonical.touch()
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        bot_corpus_path=str(canonical),
        config_overrides={"bot_corpus_refresh_enabled": True},
    )
    anchor_snapshot = tmp_path / "checkpoints" / "refresh_ema_snapshot.pt"
    anchor_snapshot.parent.mkdir()
    anchor_snapshot.touch()
    tmp_npz = canonical.with_name(canonical.name + ".NEW.tmp.npz")
    argv = coord._build_refresh_subprocess_command(canonical, tmp_npz, anchor_snapshot)
    assert "--anchor" in argv
    anchor_idx = argv.index("--anchor")
    assert argv[anchor_idx + 1] == str(anchor_snapshot), (
        "subprocess --anchor must point to the EMA snapshot, not the bootstrap"
    )
    assert "--out" in argv
    out_idx = argv.index("--out")
    assert argv[out_idx + 1] == str(tmp_npz), (
        "subprocess --out must point to the .NEW.tmp.npz path the coordinator"
        " will atomically swap"
    )
    # Other passthrough args wired from config defaults.
    for flag in ("--n-games", "--max-plies", "--random-opening-plies",
                 "--think-seconds", "--anchor-n-sims", "--anchor-temperature"):
        assert flag in argv, f"missing CLI flag {flag} in subprocess command"


def test_opponent_model_ema_calls_inference_state_dict(tmp_path):
    """When ``opponent_model: ema`` is configured, the snapshot saves the
    EMA-routed state dict (via ``trainer.inference_state_dict()``)."""
    canonical = tmp_path / "data" / "canonical.npz"
    canonical.parent.mkdir()
    # Pre-create the snapshot dir so the sha-read step after torch.save mock
    # can open a real (empty) file.
    snapshot_dir = tmp_path / "checkpoints"
    snapshot_dir.mkdir()
    snapshot_file = snapshot_dir / "refresh_ema_snapshot.pt"
    snapshot_file.write_bytes(b"")  # empty stand-in for the saved .pt
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        bot_corpus_path=str(canonical),
        config_overrides={
            "bot_corpus_refresh_enabled": True,
            "bot_corpus_refresh_opponent_model": "ema",
        },
    )
    coord.trainer.inference_state_dict = Mock(return_value={"fake_param": 1.0})
    with patch("torch.save") as save_mock:
        snapshot_path, sha = coord._save_refresh_anchor_snapshot(canonical)
    coord.trainer.inference_state_dict.assert_called_once()
    save_mock.assert_called_once()
    saved_payload = save_mock.call_args.args[0]
    assert saved_payload == {"fake_param": 1.0}
    assert snapshot_path.name == "refresh_ema_snapshot.pt"
    # SHA of empty file is well-known; still verifies the read path.
    assert sha == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_opponent_model_raw_uses_model_state_dict(tmp_path):
    """When ``opponent_model: raw`` is configured, the snapshot saves the
    raw model state_dict, bypassing the EMA accessor."""
    canonical = tmp_path / "data" / "canonical.npz"
    canonical.parent.mkdir()
    snapshot_dir = tmp_path / "checkpoints"
    snapshot_dir.mkdir()
    snapshot_file = snapshot_dir / "refresh_ema_snapshot.pt"
    snapshot_file.write_bytes(b"")
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        bot_corpus_path=str(canonical),
        config_overrides={
            "bot_corpus_refresh_enabled": True,
            "bot_corpus_refresh_opponent_model": "raw",
        },
    )
    coord.trainer.inference_state_dict = Mock(return_value={"ema_routed": 1.0})
    coord.trainer.model.state_dict = Mock(return_value={"raw": 1.0})
    coord.trainer.model._orig_mod = coord.trainer.model
    with patch("torch.save") as save_mock:
        coord._save_refresh_anchor_snapshot(canonical)
    coord.trainer.inference_state_dict.assert_not_called()
    coord.trainer.model.state_dict.assert_called_once()
    saved_payload = save_mock.call_args.args[0]
    assert saved_payload == {"raw": 1.0}


def test_full_cycle_launch_poll_complete_increments_counters(tmp_path):
    """Mock subprocess that "completes" with rc=0 → counters increment +
    state cleared + n_refreshes bumped."""
    canonical = tmp_path / "data" / "canonical.npz"
    canonical.parent.mkdir()
    canonical.touch()
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        bot_corpus_path=str(canonical),
        config_overrides={
            "bot_corpus_refresh_enabled": True,
            "bot_corpus_refresh_interval_steps": 5_000,
        },
        train_step_override=20_000,
    )
    # Simulate an in-flight proc that already completed successfully.
    fake_proc = Mock()
    fake_proc.poll = Mock(return_value=0)
    fake_proc.stderr = None
    coord._refresh_proc = fake_proc
    coord._refresh_started_step = 15_000
    coord._refresh_target_anchor_sha = "ab12cd34" * 8
    coord._refresh_ema_snapshot_path = tmp_path / "fake_snapshot.pt"
    coord._refresh_ema_snapshot_path.touch()
    coord._refresh_tmp_npz_path = tmp_path / "data" / "canonical.npz.NEW.tmp.npz"
    coord._n_refreshes_so_far = 0
    coord._train_step = 20_000

    # Mock swap + reload helpers so we exercise the orchestration logic
    # without actual NPZ I/O.
    with patch.object(
        coord, "_swap_and_hot_reload_bot_corpus",
    ) as swap_mock:
        coord._tick_bot_refresh()
        swap_mock.assert_called_once()
    # State cleared, counters bumped.
    assert coord._refresh_proc is None
    assert coord._refresh_target_anchor_sha is None
    assert coord._refresh_ema_snapshot_path is None
    assert coord._n_refreshes_so_far == 1
    assert coord._last_bot_refresh_step == 20_000
    # Force flag cleared.
    assert coord._force_bot_refresh is False


def test_subprocess_still_running_does_not_swap(tmp_path):
    """Mock subprocess where ``poll() is None`` → no swap, state unchanged."""
    canonical = tmp_path / "data" / "canonical.npz"
    canonical.parent.mkdir()
    coord = _make_coordinator(
        bot_buffer=_make_buffer(size=5_000),
        bot_corpus_path=str(canonical),
        config_overrides={"bot_corpus_refresh_enabled": True},
    )
    fake_proc = Mock()
    fake_proc.poll = Mock(return_value=None)  # still running
    coord._refresh_proc = fake_proc
    coord._n_refreshes_so_far = 0
    with patch.object(coord, "_swap_and_hot_reload_bot_corpus") as swap_mock:
        coord._tick_bot_refresh()
        swap_mock.assert_not_called()
    assert coord._refresh_proc is fake_proc  # still tracked
    assert coord._n_refreshes_so_far == 0
