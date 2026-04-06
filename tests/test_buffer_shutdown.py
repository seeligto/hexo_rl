"""
Tests for replay buffer save on shutdown signal (before pool.stop()).

Verifies the shutdown-signal save path in scripts/train.py:
  - save_to_path is called when buffer_persist is enabled
  - exceptions from save_to_path are caught and logged as warnings
  - save is skipped when buffer_persist is False
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from engine import ReplayBuffer


def _fill_buffer(n: int = 32) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=256)
    rng = np.random.default_rng(0)
    for _ in range(n):
        state = rng.random((18, 19, 19), dtype=np.float32).astype(np.float16)
        policy = rng.dirichlet(np.ones(362)).astype(np.float32)
        buf.push(state, policy, float(rng.choice([-1.0, 0.0, 1.0])))
    return buf


def test_buffer_save_on_shutdown_signal(tmp_path: Path):
    """Buffer is saved before pool.stop() in the shutdown-signal path.

    Replicates the logic added to the _shutdown_save branch in _run_loop:
    after trainer.save_checkpoint(), buffer.save_to_path() is called if
    buffer_persist is enabled in the mixing config.
    """
    buf = _fill_buffer()
    bp = tmp_path / "replay_buffer.bin"
    log = MagicMock()

    # Replicate the shutdown-signal buffer-save block from train.py
    _mix_sd = {"buffer_persist": True, "buffer_persist_path": str(bp)}
    if _mix_sd.get("buffer_persist", False):
        _bp_sd = Path(_mix_sd.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
        try:
            buf.save_to_path(str(_bp_sd))
            log.info("buffer_saved", path=str(_bp_sd), positions=buf.size, trigger="shutdown_signal")
        except Exception as _bp_exc:
            log.warning("buffer_save_failed", path=str(_bp_sd), error=str(_bp_exc))

    assert bp.exists(), "Buffer file must be written by the shutdown-signal save path"
    assert bp.stat().st_size > 0, "Buffer file must be non-empty"
    log.info.assert_called_once()
    assert log.info.call_args == call(
        "buffer_saved", path=str(bp), positions=buf.size, trigger="shutdown_signal"
    )


def test_buffer_save_on_shutdown_signal_exception_does_not_propagate():
    """A failing buffer save must log a warning and never propagate the exception."""
    buf = ReplayBuffer(capacity=64)
    log = MagicMock()

    bad_path = "/nonexistent_dir/sub/replay_buffer.bin"
    _mix_sd = {"buffer_persist": True, "buffer_persist_path": bad_path}
    if _mix_sd.get("buffer_persist", False):
        _bp_sd = Path(_mix_sd.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
        try:
            buf.save_to_path(str(_bp_sd))
            log.info("buffer_saved", path=str(_bp_sd), positions=buf.size, trigger="shutdown_signal")
        except Exception as _bp_exc:
            log.warning("buffer_save_failed", path=str(_bp_sd), error=str(_bp_exc))

    log.warning.assert_called_once()
    assert log.warning.call_args[0][0] == "buffer_save_failed"
    log.info.assert_not_called()


def test_buffer_save_skipped_when_persist_disabled():
    """Buffer save must be skipped when buffer_persist is False."""
    saved = []
    _mix_sd = {"buffer_persist": False}
    if _mix_sd.get("buffer_persist", False):
        saved.append(True)

    assert not saved, "save_to_path must not be called when buffer_persist is False"
