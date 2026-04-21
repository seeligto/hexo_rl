"""Verify `training.augment` config key reaches `sample_batch` end-to-end.

Guards the plumbing path added for Q33-C: `configs/training.yaml` →
`loop.augment_cfg` → `assemble_mixed_batch(augment=…)` →
`buffer.sample_batch(n, augment)`. Regression here silently re-pins
augmentation to True and breaks the Q37 discriminator.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from hexo_rl.training.batch_assembly import (
    BatchBuffers,
    _sample_selfplay,
    allocate_batch_buffers,
    assemble_mixed_batch,
)


def _mk_sample_return(n: int) -> tuple:
    """Return a 7-tuple matching `ReplayBuffer.sample_batch` signature."""
    states       = np.zeros((n, 18, 19, 19), dtype=np.float16)
    chain_planes = np.zeros((n, 6, 19, 19),  dtype=np.float16)
    policies     = np.zeros((n, 362),        dtype=np.float32)
    outcomes     = np.zeros(n,               dtype=np.float32)
    ownership    = np.ones((n, 19, 19),      dtype=np.uint8)
    winning_line = np.zeros((n, 19, 19),     dtype=np.uint8)
    is_full_search = np.ones(n, dtype=np.uint8)
    return states, chain_planes, policies, outcomes, ownership, winning_line, is_full_search


@pytest.mark.parametrize("augment", [True, False])
def test_assemble_mixed_batch_forwards_augment_flag(augment: bool) -> None:
    """Both corpus + self-play `sample_batch` calls receive the requested flag."""
    batch_size = 8
    n_pre      = 4
    n_self     = 4

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_sample_return(n_pre))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_sample_return(n_self))

    bufs = allocate_batch_buffers(batch_size, 362)

    assemble_mixed_batch(
        pretrained_buffer=pretrained,
        buffer=selfplay,
        recent_buffer=None,
        n_pre=n_pre,
        n_self=n_self,
        batch_size=batch_size,
        batch_size_cfg=batch_size,
        recency_weight=0.0,
        bufs=bufs,
        train_step=0,
        augment=augment,
    )

    pretrained.sample_batch.assert_called_once_with(n_pre, augment)
    selfplay.sample_batch.assert_called_once_with(max(1, n_self), augment)


@pytest.mark.parametrize("augment", [True, False])
def test_sample_selfplay_forwards_augment_flag(augment: bool) -> None:
    """`_sample_selfplay` uniform fallback path respects the flag."""
    n_self = 4
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_sample_return(n_self))

    _sample_selfplay(
        buffer=selfplay,
        recent_buffer=None,
        n_self=n_self,
        recency_weight=0.0,
        augment=augment,
    )

    selfplay.sample_batch.assert_called_once_with(max(1, n_self), augment)


def test_assemble_mixed_batch_default_augment_is_true() -> None:
    """Default preserves legacy behaviour (augment=True)."""
    n_pre = 2
    n_self = 2
    batch_size = 4

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_sample_return(n_pre))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_sample_return(n_self))

    bufs = allocate_batch_buffers(batch_size, 362)

    assemble_mixed_batch(
        pretrained_buffer=pretrained,
        buffer=selfplay,
        recent_buffer=None,
        n_pre=n_pre,
        n_self=n_self,
        batch_size=batch_size,
        batch_size_cfg=batch_size,
        recency_weight=0.0,
        bufs=bufs,
        train_step=0,
    )

    pretrained.sample_batch.assert_called_once_with(n_pre, True)
    selfplay.sample_batch.assert_called_once_with(max(1, n_self), True)


def test_training_yaml_declares_augment_key() -> None:
    """Production config ships with augment: true — required key per loop.py."""
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    assert "augment" in cfg, "training.yaml must declare `augment` (required key)"
    assert cfg["augment"] is True, "production default must preserve augmentation"
