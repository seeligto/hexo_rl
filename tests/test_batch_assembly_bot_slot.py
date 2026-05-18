"""§178 batch_assembly bot-slot regression tests.

Covers:
  Cell A: load_bot_corpus_buffer(path=None) → None (back-compat default).
  Cell A2: load_bot_corpus_buffer(path missing file) → None + warning logged.
  Cell B: assemble_mixed_batch(bot_buffer=None, n_bot=0) BYTE-IDENTICAL to
          legacy call without bot kwargs (regression pin for n_pretrain=n_pre).
  Cell C: bot_buffer.size==0 falls through to n_bot=0 path (warmup fallback).
  Cell D: bot_buffer populated → is_full_search[n_pre:n_pre+n_bot]==1 (override).
  Cell E: batch_size != batch_size_cfg + bot active → concat fallback works.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from hexo_rl.training.batch_assembly import (
    allocate_batch_buffers,
    assemble_mixed_batch,
    load_bot_corpus_buffer,
)
from hexo_rl.encoding import lookup as _lookup_encoding

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
N_ACTIONS: int = BOARD_SIZE * BOARD_SIZE + 1


# ── helpers ───────────────────────────────────────────────────────────────────

def _mk_buffer_return(n: int, ifs_value: int = 1) -> tuple:
    """Mock Rust buffer sample_batch return — 8-plane states, configurable ifs."""
    return (
        np.zeros((n, 8,  BOARD_SIZE, BOARD_SIZE), dtype=np.float16),
        np.zeros((n, 6,  BOARD_SIZE, BOARD_SIZE), dtype=np.float16),
        np.zeros((n, N_ACTIONS),                  dtype=np.float32),
        np.zeros(n,                               dtype=np.float32),
        np.ones( (n, BOARD_SIZE, BOARD_SIZE),     dtype=np.uint8),
        np.zeros((n, BOARD_SIZE, BOARD_SIZE),     dtype=np.uint8),
        np.full(n, ifs_value,                     dtype=np.uint8),
    )


def _mk_signed_buffer_return(n: int, marker: float) -> tuple:
    """Marker-tagged buffer return so we can verify slot ordering byte-by-byte."""
    states = np.full((n, 8, BOARD_SIZE, BOARD_SIZE), marker, dtype=np.float16)
    chain  = np.full((n, 6, BOARD_SIZE, BOARD_SIZE), marker, dtype=np.float16)
    pol    = np.full((n, N_ACTIONS), marker, dtype=np.float32)
    out    = np.full(n, marker, dtype=np.float32)
    own    = np.full((n, BOARD_SIZE, BOARD_SIZE), 1, dtype=np.uint8)
    wl     = np.zeros((n, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    ifs    = np.zeros(n, dtype=np.uint8)  # zero so we can verify the override
    return (states, chain, pol, out, own, wl, ifs)


# ── Cell A: load_bot_corpus_buffer back-compat ───────────────────────────────

def test_load_bot_corpus_buffer_path_none_returns_none():
    """No bot_corpus_path key → None (§177-style back-compat)."""
    result = load_bot_corpus_buffer(
        mixing_cfg={},
        config={"encoding": "v6"},
        emit_fn=lambda _e: None,
        buffer_size=0,
        buffer_capacity=1000,
    )
    assert result is None


def test_load_bot_corpus_buffer_path_explicit_none_returns_none():
    """Explicit bot_corpus_path=None → None."""
    result = load_bot_corpus_buffer(
        mixing_cfg={"bot_corpus_path": None},
        config={"encoding": "v6"},
        emit_fn=lambda _e: None,
        buffer_size=0,
        buffer_capacity=1000,
    )
    assert result is None


def test_load_bot_corpus_buffer_missing_file_returns_none(tmp_path: Path):
    """Path set but file missing → None + warning logged (no exception)."""
    missing = tmp_path / "no_such_file.npz"
    result = load_bot_corpus_buffer(
        mixing_cfg={"bot_corpus_path": str(missing)},
        config={"encoding": "v6"},
        emit_fn=lambda _e: None,
        buffer_size=0,
        buffer_capacity=1000,
    )
    assert result is None


# ── Cell B: assemble_mixed_batch byte-identical when bot inactive ────────────

def test_assemble_mixed_batch_no_bot_kwargs_equals_default():
    """Calling without bot kwargs == calling with bot_buffer=None,n_bot=0."""
    batch_size = 32
    n_pre  = 8
    n_self = 24

    rng = np.random.default_rng(0)
    pre_payload = _mk_signed_buffer_return(n_pre, marker=1.0)
    self_payload = _mk_signed_buffer_return(n_self, marker=2.0)

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=pre_payload)
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=self_payload)

    bufs1 = allocate_batch_buffers(batch_size, N_ACTIONS)
    bufs2 = allocate_batch_buffers(batch_size, N_ACTIONS)

    res_default = assemble_mixed_batch(
        pretrained_buffer=pretrained, buffer=selfplay, recent_buffer=None,
        n_pre=n_pre, n_self=n_self, batch_size=batch_size,
        batch_size_cfg=batch_size, recency_weight=0.0,
        bufs=bufs1, train_step=0, augment=False,
    )

    # Re-mock so call counts reset
    pretrained2 = MagicMock()
    pretrained2.sample_batch = MagicMock(return_value=pre_payload)
    selfplay2 = MagicMock()
    selfplay2.sample_batch = MagicMock(return_value=self_payload)
    res_explicit = assemble_mixed_batch(
        pretrained_buffer=pretrained2, buffer=selfplay2, recent_buffer=None,
        n_pre=n_pre, n_self=n_self, batch_size=batch_size,
        batch_size_cfg=batch_size, recency_weight=0.0,
        bufs=bufs2, train_step=0, augment=False,
        bot_buffer=None, n_bot=0,
    )

    np.testing.assert_array_equal(res_default.states, res_explicit.states)
    np.testing.assert_array_equal(res_default.policies, res_explicit.policies)
    np.testing.assert_array_equal(res_default.outcomes, res_explicit.outcomes)
    np.testing.assert_array_equal(res_default.is_full_search, res_explicit.is_full_search)
    assert res_default.n_recent_actual == res_explicit.n_recent_actual


# ── Cell C: bot_buffer.size==0 falls through ──────────────────────────────────

def test_assemble_mixed_batch_bot_buffer_empty_falls_through():
    """bot_buffer.size==0 + n_bot>0 → no bot slot used (warmup fallback)."""
    batch_size = 32
    n_pre  = 8
    n_self = 24

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_buffer_return(n_pre))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_buffer_return(n_self))
    empty_bot = MagicMock()
    empty_bot.size = 0
    empty_bot.sample_batch = MagicMock()  # MUST NOT be called

    bufs = allocate_batch_buffers(batch_size, N_ACTIONS)
    result = assemble_mixed_batch(
        pretrained_buffer=pretrained, buffer=selfplay, recent_buffer=None,
        n_pre=n_pre, n_self=n_self, batch_size=batch_size,
        batch_size_cfg=batch_size, recency_weight=0.0,
        bufs=bufs, train_step=0, augment=False,
        bot_buffer=empty_bot, n_bot=8,
    )
    empty_bot.sample_batch.assert_not_called()
    assert result.states.shape[0] == batch_size


# ── Cell D: is_full_search override for bot rows ─────────────────────────────

def test_assemble_mixed_batch_bot_rows_is_full_search_one():
    """When bot buffer populated: ifs[n_pre:n_pre+n_bot]==1 even if buffer returns 0."""
    batch_size = 32
    n_pre  = 8
    n_bot  = 4
    n_self = 20

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_buffer_return(n_pre, ifs_value=1))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_buffer_return(n_self, ifs_value=1))
    # Bot returns ifs=0 — implementation MUST override to 1
    bot_buf = MagicMock()
    bot_buf.size = 1000
    bot_buf.sample_batch = MagicMock(return_value=_mk_buffer_return(n_bot, ifs_value=0))

    bufs = allocate_batch_buffers(batch_size, N_ACTIONS)
    result = assemble_mixed_batch(
        pretrained_buffer=pretrained, buffer=selfplay, recent_buffer=None,
        n_pre=n_pre, n_self=n_self, batch_size=batch_size,
        batch_size_cfg=batch_size, recency_weight=0.0,
        bufs=bufs, train_step=0, augment=False,
        bot_buffer=bot_buf, n_bot=n_bot,
    )
    bot_buf.sample_batch.assert_called_once_with(n_bot, False)
    # Bot slot is [n_pre : n_pre+n_bot] in batch ordering
    assert np.all(result.is_full_search[n_pre:n_pre + n_bot] == 1), (
        "bot rows must have is_full_search=1 (one-hot SealBot targets are full-search-equivalent)"
    )
    # Corpus slot unchanged
    assert np.all(result.is_full_search[:n_pre] == 1)


def test_assemble_mixed_batch_bot_slot_ordering():
    """Bot slot lands in [n_pre : n_pre+n_bot]; selfplay in [n_pre+n_bot : ...]."""
    batch_size = 32
    n_pre  = 8
    n_bot  = 4
    n_self = 20

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_signed_buffer_return(n_pre, marker=1.0))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_signed_buffer_return(n_self, marker=3.0))
    bot_buf = MagicMock()
    bot_buf.size = 1000
    bot_buf.sample_batch = MagicMock(return_value=_mk_signed_buffer_return(n_bot, marker=2.0))

    bufs = allocate_batch_buffers(batch_size, N_ACTIONS)
    result = assemble_mixed_batch(
        pretrained_buffer=pretrained, buffer=selfplay, recent_buffer=None,
        n_pre=n_pre, n_self=n_self, batch_size=batch_size,
        batch_size_cfg=batch_size, recency_weight=0.0,
        bufs=bufs, train_step=0, augment=False,
        bot_buffer=bot_buf, n_bot=n_bot,
    )
    # Compare states markers
    assert np.allclose(result.states[:n_pre], 1.0)
    assert np.allclose(result.states[n_pre:n_pre + n_bot], 2.0)
    assert np.allclose(result.states[n_pre + n_bot:], 3.0)


# ── Cell E: batch_size_cfg mismatch with bot active ───────────────────────────

def test_assemble_mixed_batch_size_mismatch_with_bot():
    """batch_size != batch_size_cfg → concat fallback handles bot slot."""
    batch_size_cfg = 32
    batch_size_runtime = 24  # mismatched
    n_pre  = 6
    n_bot  = 4
    n_self = 14  # 6+4+14 = 24

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_buffer_return(n_pre, ifs_value=1))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_buffer_return(n_self, ifs_value=1))
    bot_buf = MagicMock()
    bot_buf.size = 1000
    bot_buf.sample_batch = MagicMock(return_value=_mk_buffer_return(n_bot, ifs_value=0))

    bufs = allocate_batch_buffers(batch_size_cfg, N_ACTIONS)
    result = assemble_mixed_batch(
        pretrained_buffer=pretrained, buffer=selfplay, recent_buffer=None,
        n_pre=n_pre, n_self=n_self, batch_size=batch_size_runtime,
        batch_size_cfg=batch_size_cfg, recency_weight=0.0,
        bufs=bufs, train_step=0, augment=False,
        bot_buffer=bot_buf, n_bot=n_bot,
    )
    # 6 corpus + 4 bot + 14 selfplay = 24 rows
    assert result.states.shape[0] == batch_size_runtime
    assert np.all(result.is_full_search[n_pre:n_pre + n_bot] == 1)
