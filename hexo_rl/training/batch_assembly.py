"""Batch assembly: buffer allocation, corpus loading, mixed-batch construction.

Canonical home for the three buffer-touching concerns that train.py previously
owned inline:

  allocate_batch_buffers  — one-time pre-allocation at startup (no malloc cycling
                            during the training loop).
  load_pretrained_buffer  — load corpus NPZ into a Rust ReplayBuffer with neutral
                            aux padding; returns None if corpus absent.
  assemble_mixed_batch    — combine pretrain + self-play (+ optional recent) rows
                            into one batch array, reusing pre-allocated memory in
                            steady-state.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# ── Pre-allocated batch buffers ───────────────────────────────────────────────

@dataclass
class BatchBuffers:
    """Pre-allocated arrays shared across training steps.

    One instance is created at startup via :func:`allocate_batch_buffers` and
    reused each step so the training loop never triggers large malloc/free cycles.
    ``warmup_active`` is flipped to False the first time all sources return
    the expected row count (a side-effect of :func:`assemble_mixed_batch`).
    """
    states: np.ndarray          # (B, 18, 19, 19) float16
    chain_planes: np.ndarray    # (B, 6, 19, 19)  float16
    policies: np.ndarray        # (B, N_ACTIONS) float32
    outcomes: np.ndarray        # (B,) float32
    ownership: np.ndarray       # (B, 19, 19) uint8
    winning_line: np.ndarray    # (B, 19, 19) uint8
    is_full_search: np.ndarray  # (B,) uint8 — 1=full-search, 0=quick-search
    warmup_active: bool = field(default=True)


def allocate_batch_buffers(batch_size: int, n_actions: int) -> BatchBuffers:
    """Allocate shared batch arrays once at startup.

    Args:
        batch_size: Expected batch size from training config.
        n_actions:  Number of policy logits (N_ACTIONS constant).

    Returns:
        A :class:`BatchBuffers` instance with all arrays zeroed.
    """
    return BatchBuffers(
        states=np.empty((batch_size, 18, 19, 19), dtype=np.float16),
        chain_planes=np.empty((batch_size, 6, 19, 19), dtype=np.float16),
        policies=np.empty((batch_size, n_actions), dtype=np.float32),
        outcomes=np.empty(batch_size, dtype=np.float32),
        ownership=np.empty((batch_size, 19, 19), dtype=np.uint8),
        winning_line=np.empty((batch_size, 19, 19), dtype=np.uint8),
        is_full_search=np.ones(batch_size, dtype=np.uint8),  # default full-search
    )


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_pretrained_buffer(
    mixing_cfg: dict[str, Any],
    config: dict[str, Any],
    emit_fn: Callable[[dict[str, Any]], None],
    buffer_size: int,
    buffer_capacity: int,
) -> Optional[Any]:  # returns ReplayBuffer | None
    """Load corpus NPZ into a Rust ReplayBuffer with neutral aux padding.

    Corpus rows carry no per-row aux (the NPZ predates the A1 aux alignment
    feature).  We pad ownership=1 ("empty", decodes to 0.0) and winning_line=0
    so the Rust buffer is well-formed.  The ``n_pretrain`` row-slice in
    ``Trainer._train_on_batch`` then masks these rows out of aux losses.

    Args:
        mixing_cfg:       ``config["training"]["mixing"]`` sub-dict.
        config:           Full merged config (for seed, etc.).
        emit_fn:          ``emit_event`` callable for dashboard events.
        buffer_size:      Current self-play buffer size (for dashboard event).
        buffer_capacity:  Self-play buffer capacity (for dashboard event).

    Returns:
        A populated ``ReplayBuffer`` if the corpus file exists, else ``None``.
    """
    from engine import ReplayBuffer  # local import — engine only available post-build

    pretrained_path = mixing_cfg.get("pretrained_buffer_path")
    if not pretrained_path:
        return None
    if not Path(pretrained_path).exists():
        log.warning(
            "corpus_npz_not_found",
            path=pretrained_path,
            msg=(
                "No corpus NPZ found — skipping pretrained mixing. "
                "Buffer will fill from self-play only. Run 'make corpus.npz' to generate."
            ),
        )
        return None

    log.info(
        "loading_corpus_npz",
        path=pretrained_path,
        msg="copying corpus into Rust pretrained_buffer — may take minutes for large corpora",
    )
    t0 = time.time()
    data = np.load(pretrained_path, mmap_mode="r")
    pre_states   = data["states"]    # (T, 18 or 24, 19, 19) float16
    pre_policies = data["policies"]  # (T, 362) float32
    pre_outcomes = data["outcomes"]  # (T,) float32
    T = len(pre_outcomes)

    # Handle old 24-plane corpus: extract chain from planes 18:24, trim states to :18.
    if pre_states.shape[1] == 24:
        log.info("corpus_24plane_compat", msg="24-plane NPZ detected — extracting chain planes, trimming state to 18 planes")
        pre_chain  = np.array(pre_states[:, 18:24], dtype=np.float16)  # (T, 6, 19, 19)
        pre_states = np.array(pre_states[:, :18],   dtype=np.float16)  # (T, 18, 19, 19)
    else:
        # §102.a: compute chain planes from stone planes at NPZ load so corpus
        # rows carry real chain targets rather than zeros. Route through float32
        # division (mirrors Rust `encode_chain_planes`: `(int as f32) / 6.0f32`
        # → cast to f16) to preserve byte-exactness with the self-play
        # replay-buffer storage. The F2 guard pins the underlying int8 planes
        # against Rust; this path pins the final /6 f16 values.
        from hexo_rl.env.game_state import _compute_chain_planes
        pre_chain = np.empty((T, 6, 19, 19), dtype=np.float16)
        if T > 0:
            cur_all = np.asarray(pre_states[:, 0], dtype=np.float32)
            opp_all = np.asarray(pre_states[:, 8], dtype=np.float32)
            for k in range(T):
                planes_f32 = _compute_chain_planes(cur_all[k], opp_all[k]).astype(np.float32) / 6.0
                pre_chain[k] = planes_f32.astype(np.float16)

    max_pre = int(mixing_cfg.get("pretrain_max_samples", 0))
    if max_pre and T > max_pre:
        _seed = int(config.get("seed", 42))
        _rng  = np.random.default_rng(_seed)
        idx   = np.sort(_rng.choice(T, size=max_pre, replace=False))
        log.info("corpus_capped", original=T, kept=max_pre)
        pre_states   = pre_states[idx]
        pre_chain    = pre_chain[idx]
        pre_policies = pre_policies[idx]
        pre_outcomes = pre_outcomes[idx]
        T = max_pre

    file_mb = os.path.getsize(pretrained_path) / 1e6
    log.info("corpus_prefill", positions=T, file_mb=round(file_mb, 1))
    if T > 100_000:
        log.warning(
            "corpus_prefill_oversized",
            positions=T,
            msg=(
                "NPZ has >100K positions — cold-start only needs 50K. "
                "Run 'make corpus.npz' to regenerate with the optimized pipeline."
            ),
        )

    est_ram_gb = T * 14_448 / (1024 ** 3)
    if est_ram_gb > 2.0:
        log.warning(
            "corpus_prefill_high_ram",
            path=pretrained_path,
            n_positions=T,
            estimated_ram_gb=round(est_ram_gb, 1),
            msg="push_game allocates full corpus in RAM — training starts after this completes",
        )

    pretrained_buffer = ReplayBuffer(capacity=T)
    # Neutral aux: ownership=1 ("empty" → 0.0 after decode), winning_line=0.
    pre_own = np.ones((T, 361), dtype=np.uint8)
    pre_wl  = np.zeros((T, 361), dtype=np.uint8)
    pretrained_buffer.push_game(pre_states, pre_chain, pre_policies, pre_outcomes, pre_own, pre_wl)
    del pre_states, pre_chain, pre_policies, pre_outcomes, pre_own, pre_wl
    del data

    log.info("corpus_loaded", positions=T, seconds=f"{time.time() - t0:.1f}")
    emit_fn({"event": "system_stats", "buffer_size": buffer_size, "buffer_capacity": buffer_capacity})
    return pretrained_buffer


# ── Recent-buffer augmentation ───────────────────────────────────────────────

def _augment_recent_rows(
    s_r: np.ndarray,
    c_r: np.ndarray,
    p_r: np.ndarray,
    own_r_flat: np.ndarray,
    wl_r_flat: np.ndarray,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply 12-fold hex augmentation to a batch of RecentBuffer rows.

    RecentBuffer.sample() does not go through the Rust sample_batch kernel, so
    augmentation must be applied in Python. This mirrors make_augmented_collate
    in pretrain.py: Rust apply_symmetries_batch for stone planes, recompute
    chain planes from augmented stones, numpy scatter for policy/aux targets.

    With augment=False returns inputs unchanged (no copies).
    """
    if not augment:
        return s_r, c_r, p_r, own_r_flat, wl_r_flat

    import engine as _engine
    from hexo_rl.env.game_state import _compute_chain_planes
    from hexo_rl.augment.luts import get_policy_scatters

    n = len(s_r)
    scatters = get_policy_scatters()
    sym_indices = np.random.randint(0, 12, size=n)

    states_f32 = s_r.astype(np.float32)
    states_f32 = _engine.apply_symmetries_batch(states_f32, sym_indices.tolist())
    s_r = states_f32.astype(np.float16)

    c_r_aug = np.empty_like(c_r)
    for i in range(n):
        c_r_aug[i] = (
            _compute_chain_planes(states_f32[i, 0], states_f32[i, 8]).astype(np.float32) / 6.0
        ).astype(np.float16)

    scattered_p   = np.empty_like(p_r)
    scattered_own = np.empty_like(own_r_flat)
    scattered_wl  = np.empty_like(wl_r_flat)
    for i in range(n):
        lut = scatters[int(sym_indices[i])]
        scattered_p[i]   = p_r[i][lut]
        scattered_own[i] = own_r_flat[i][lut[:361]]
        scattered_wl[i]  = wl_r_flat[i][lut[:361]]

    return s_r, c_r_aug, scattered_p, scattered_own, scattered_wl


# ── Mixed-batch assembly ──────────────────────────────────────────────────────

def assemble_mixed_batch(
    pretrained_buffer: Any,           # ReplayBuffer
    buffer: Any,                      # ReplayBuffer (self-play)
    recent_buffer: Optional[Any],     # RecentBuffer | None
    n_pre: int,
    n_self: int,
    batch_size: int,
    batch_size_cfg: int,
    recency_weight: float,
    bufs: BatchBuffers,
    train_step: int,
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble one mixed batch from pretrain + self-play (+ optional recent) buffers.

    During warm-up (buffers partially filled), falls back to ``np.concatenate``
    which allocates.  Once all sources return the full requested row count,
    switches to in-place ``np.copyto`` into ``bufs`` and clears
    ``bufs.warmup_active``.

    Args:
        pretrained_buffer: Corpus Rust ReplayBuffer.
        buffer:            Self-play Rust ReplayBuffer.
        recent_buffer:     Optional Python RecentBuffer for recency weighting.
        n_pre:             Corpus rows to sample.
        n_self:            Self-play rows to sample.
        batch_size:        Total batch size this step (should equal n_pre + n_self).
        batch_size_cfg:    Pre-allocated buffer batch size; if they differ we fall back
                           to concat to avoid out-of-bounds writes.
        recency_weight:    Fraction of self-play rows taken from recent_buffer.
        bufs:              Pre-allocated batch arrays (modified in-place when steady-state).
        train_step:        Current step index (for log messages only).
        augment:           Apply 12-fold hex symmetry augmentation during Rust sample_batch.
                           Default True preserves production behaviour; set False only for
                           diagnostic runs (see CLAUDE.md § Testing conventions).

    Returns:
        Seven arrays ``(states, chain_planes, policies, outcomes, ownership, winning_line,
        is_full_search)`` — views into ``bufs`` in steady-state, freshly allocated during
        warm-up.  Corpus positions always have ``is_full_search=1`` (apply full policy loss).
    """
    s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre = pretrained_buffer.sample_batch(n_pre, augment)

    if batch_size != batch_size_cfg:
        # Edge case: runtime batch size diverged from pre-allocated shape.
        if train_step > 100:
            log.warning("mixed_batch_size_mismatch", batch_size=batch_size, expected=batch_size_cfg)
        s_self, c_self, p_self, o_self, own_self, wl_self, ifs_self = _sample_selfplay(
            buffer, recent_buffer, n_self, recency_weight, augment
        )
        return (
            np.concatenate([s_pre, s_self], axis=0),
            np.concatenate([c_pre, c_self], axis=0),
            np.concatenate([p_pre, p_self], axis=0),
            np.concatenate([o_pre, o_self], axis=0),
            np.concatenate([own_pre, own_self], axis=0),
            np.concatenate([wl_pre, wl_self], axis=0),
            np.concatenate([ifs_pre, ifs_self], axis=0),
        )

    # ── Normal path: try in-place fill into bufs ──────────────────────────────
    use_recent = (
        recent_buffer is not None
        and recent_buffer.size > 0
        and recency_weight > 0.0
        and n_self > 1
    )

    if use_recent:
        n_recent   = max(1, int(round(n_self * recency_weight)))
        n_uniform  = n_self - n_recent
        s_r, c_r, p_r, o_r, own_r_flat, wl_r_flat, ifs_r = recent_buffer.sample(n_recent)
        s_r, c_r, p_r, own_r_flat, wl_r_flat = _augment_recent_rows(
            s_r, c_r, p_r, own_r_flat, wl_r_flat, augment
        )
        own_r = own_r_flat.reshape(-1, 19, 19)
        wl_r  = wl_r_flat.reshape(-1, 19, 19)
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u = buffer.sample_batch(max(1, n_uniform), augment)
        pieces    = [(s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre),
                     (s_r,   c_r,   p_r,   o_r,   own_r,   wl_r,   ifs_r),
                     (s_u,   c_u,   p_u,   o_u,   own_u,   wl_u,   ifs_u)]
        n_avail   = n_pre + len(s_r) + len(s_u)
    else:
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u = buffer.sample_batch(max(1, n_self), augment)
        pieces  = [(s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre),
                   (s_u,   c_u,   p_u,   o_u,   own_u,   wl_u,   ifs_u)]
        n_avail = n_pre + len(s_u)

    if n_avail < batch_size:
        # Warm-up: one or more sources returned fewer rows than requested.
        return (
            np.concatenate([p[0] for p in pieces], axis=0),
            np.concatenate([p[1] for p in pieces], axis=0),
            np.concatenate([p[2] for p in pieces], axis=0),
            np.concatenate([p[3] for p in pieces], axis=0),
            np.concatenate([p[4] for p in pieces], axis=0),
            np.concatenate([p[5] for p in pieces], axis=0),
            np.concatenate([p[6] for p in pieces], axis=0),
        )

    # Steady-state: in-place copy, no heap allocation.
    if bufs.warmup_active:
        log.info("buffer_warmup_ended", step=train_step, n_available=n_avail, batch_size=batch_size)
        bufs.warmup_active = False

    offset = 0
    for s, c, p, o, own, wl, ifs in pieces:
        n = len(s)
        np.copyto(bufs.states[offset:offset + n],          s)
        np.copyto(bufs.chain_planes[offset:offset + n],    c)
        np.copyto(bufs.policies[offset:offset + n],        p)
        np.copyto(bufs.outcomes[offset:offset + n],        o)
        np.copyto(bufs.ownership[offset:offset + n],       own)
        np.copyto(bufs.winning_line[offset:offset + n],    wl)
        np.copyto(bufs.is_full_search[offset:offset + n],  ifs)
        offset += n

    return (bufs.states, bufs.chain_planes, bufs.policies, bufs.outcomes,
            bufs.ownership, bufs.winning_line, bufs.is_full_search)


def _sample_selfplay(
    buffer: Any,
    recent_buffer: Optional[Any],
    n_self: int,
    recency_weight: float,
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample self-play rows, blending recent + uniform when recency_weight > 0."""
    if (recent_buffer is not None and recent_buffer.size > 0
            and recency_weight > 0.0 and n_self > 1):
        n_r = max(1, int(round(n_self * recency_weight)))
        n_u = n_self - n_r
        s_r, c_r, p_r, o_r, own_r_flat, wl_r_flat, ifs_r = recent_buffer.sample(n_r)
        s_r, c_r, p_r, own_r_flat, wl_r_flat = _augment_recent_rows(
            s_r, c_r, p_r, own_r_flat, wl_r_flat, augment
        )
        own_r = own_r_flat.reshape(-1, 19, 19)
        wl_r  = wl_r_flat.reshape(-1, 19, 19)
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u = buffer.sample_batch(max(1, n_u), augment)
        return (
            np.concatenate([s_r, s_u], axis=0),
            np.concatenate([c_r, c_u], axis=0),
            np.concatenate([p_r, p_u], axis=0),
            np.concatenate([o_r, o_u], axis=0),
            np.concatenate([own_r, own_u], axis=0),
            np.concatenate([wl_r, wl_u], axis=0),
            np.concatenate([ifs_r, ifs_u], axis=0),
        )
    return buffer.sample_batch(max(1, n_self), augment)
