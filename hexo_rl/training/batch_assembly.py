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

from hexo_rl.encoding import lookup as _lookup_encoding, normalize_encoding_name as _normalize_encoding_name
from hexo_rl.encoding.resolvers import opp_stone_slot

# Module-level hoist (registry lookup at import time, not per-batch).
_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
BUFFER_CHANNELS: int = _V6.n_planes
NUM_CELLS: int = _V6.n_cells

log = structlog.get_logger(__name__)


# ── Mixed-batch result ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BatchAssemblyResult:
    """Result of :func:`assemble_mixed_batch`.

    Eight arrays (views into pre-allocated ``BatchBuffers`` in steady state,
    freshly allocated during warm-up / size-mismatch fallback) plus the actual
    number of rows drawn from ``recent_buffer`` so callers can slice
    ``[corpus | recent | uniform_self]``.

    §S181-AUDIT Wave 4 4B-impl-3 added ``position_indices`` for the ply-index
    aux head. None when ply_index_weight == 0 (head disabled) to avoid the
    extra sample_batch_with_pos call.

    Frozen for safety; attribute access is perf-parity with tuple unpack.
    """
    states: np.ndarray
    chain_planes: np.ndarray
    policies: np.ndarray
    outcomes: np.ndarray
    ownership: np.ndarray
    winning_line: np.ndarray
    is_full_search: np.ndarray
    n_recent_actual: int
    position_indices: Optional[np.ndarray] = None


# ── Pre-allocated batch buffers ───────────────────────────────────────────────

@dataclass
class BatchBuffers:
    """Pre-allocated arrays shared across training steps.

    One instance is created at startup via :func:`allocate_batch_buffers` and
    reused each step so the training loop never triggers large malloc/free cycles.
    ``warmup_active`` is flipped to False the first time all sources return
    the expected row count (a side-effect of :func:`assemble_mixed_batch`).

    Spatial shapes are encoding-derived (not v6-fixed): pass ``trunk_size``
    to :func:`allocate_batch_buffers` to size for v6w25 (25) / v8 (25);
    the default ``trunk_size=BOARD_SIZE`` keeps v6 behaviour for legacy callers.
    """
    states: np.ndarray          # (B, 8, T, T) float16 (T = trunk_size)
    chain_planes: np.ndarray    # (B, 6, T, T) float16
    policies: np.ndarray        # (B, N_ACTIONS) float32
    outcomes: np.ndarray        # (B,) float32
    ownership: np.ndarray       # (B, T, T) uint8
    winning_line: np.ndarray    # (B, T, T) uint8
    is_full_search: np.ndarray  # (B,) uint8 — 1=full-search, 0=quick-search
    warmup_active: bool = field(default=True)


def allocate_batch_buffers(
    batch_size: int,
    n_actions: int,
    trunk_size: int = BOARD_SIZE,
    aux_stride: Optional[int] = None,
    n_planes: int = BUFFER_CHANNELS,
) -> BatchBuffers:
    """Allocate shared batch arrays once at startup.

    Args:
        batch_size: Expected batch size from training config.
        n_actions:  Number of policy logits (N_ACTIONS constant).
        trunk_size: Spatial dim for state/chain/aux planes (v6=BOARD_SIZE,
                    v6w25=25, v8=25). Default BOARD_SIZE = v6, preserves
                    backward-compat for callers that have not migrated.
        aux_stride: Flat length per aux plane (`trunk_size**2` unless
                    overridden). Currently unused by the pre-allocated
                    arrays (aux is shaped as `(B, T, T)`); reserved for
                    future flat-aux variants.
        n_planes:   State-plane count for the active encoding (v6 → 8,
                    v6tp → 10 incl. turn-phase 16/17). Default BUFFER_CHANNELS
                    (8) preserves byte-identical behavior for v6-family callers.

    Returns:
        A :class:`BatchBuffers` instance with all arrays empty/ones.
    """
    if aux_stride is None:
        aux_stride = trunk_size * trunk_size  # noqa: F841 — reserved hook
    return BatchBuffers(
        states=np.empty((batch_size, n_planes, trunk_size, trunk_size), dtype=np.float16),
        chain_planes=np.empty((batch_size, 6, trunk_size, trunk_size), dtype=np.float16),
        policies=np.empty((batch_size, n_actions), dtype=np.float32),
        outcomes=np.empty(batch_size, dtype=np.float32),
        ownership=np.empty((batch_size, trunk_size, trunk_size), dtype=np.uint8),
        winning_line=np.empty((batch_size, trunk_size, trunk_size), dtype=np.uint8),
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
    board_size = config.get("board_size", BOARD_SIZE)
    pre_states   = data["states"]    # (T, 8, board_size, board_size) float16
    pre_policies = data["policies"]  # (T, policy_logit_count) float32
    pre_outcomes = data["outcomes"]  # (T,) float32
    T = len(pre_outcomes)

    # Validate against the ACTIVE encoding's plane count, not the v6
    # BUFFER_CHANNELS constant (8). v6tp keeps turn-phase planes 16/17 →
    # 10 planes; resolves to 8 for the v6 family (byte-identical check).
    _spec = _lookup_encoding(_normalize_encoding_name(config.get("encoding")))
    _expected_planes = _spec.n_planes
    if pre_states.shape[1] != _expected_planes:
        raise ValueError(
            f"corpus '{pretrained_path}': states.shape[1]={pre_states.shape[1]}, "
            f"expected {_expected_planes} (encoding "
            f"{_normalize_encoding_name(config.get('encoding'))!r}). Regenerate "
            f"with 'scripts/export_corpus_npz.py --encoding <name>'."
        )

    # §102.a: compute chain planes from stone planes. cur t0 is always slot 0;
    # opp t0 (source plane 8) is slot 4 for the v6 family but slot 1 for
    # v6_live2 — derive from the registry, never hardcode 4.
    from hexo_rl.env.game_state import _compute_chain_planes
    _opp_slot = opp_stone_slot(_spec)
    pre_chain = np.empty((T, 6, board_size, board_size), dtype=np.float16)
    if T > 0:
        cur_all = np.asarray(pre_states[:, 0], dtype=np.float32)
        opp_all = np.asarray(pre_states[:, _opp_slot], dtype=np.float32)
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

    _enc = _normalize_encoding_name(config.get("encoding"))
    pretrained_buffer = ReplayBuffer(capacity=T, encoding=_enc)
    # Neutral aux: ownership=1 ("empty" → 0.0 after decode), winning_line=0.
    n_cells = board_size * board_size
    pre_own = np.ones((T, n_cells), dtype=np.uint8)
    pre_wl  = np.zeros((T, n_cells), dtype=np.uint8)
    pretrained_buffer.push_game(pre_states, pre_chain, pre_policies, pre_outcomes, pre_own, pre_wl)
    del pre_states, pre_chain, pre_policies, pre_outcomes, pre_own, pre_wl
    del data

    log.info("corpus_loaded", positions=T, seconds=f"{time.time() - t0:.1f}")
    emit_fn({"event": "system_stats", "buffer_size": buffer_size, "buffer_capacity": buffer_capacity})
    return pretrained_buffer


# ── Bot-corpus loading (§178) ────────────────────────────────────────────────

def load_bot_corpus_buffer(
    mixing_cfg: dict[str, Any],
    config: dict[str, Any],
    emit_fn: Callable[[dict[str, Any]], None],
    buffer_size: int,
    buffer_capacity: int,
) -> Optional[Any]:  # returns ReplayBuffer | None
    """Load §178 bot-game NPZ (SealBot vs anchor) into a Rust ReplayBuffer.

    Parallel to :func:`load_pretrained_buffer` — same NPZ schema, same neutral
    aux padding (ownership=1, winning_line=0). The aux-decode mask at
    :func:`hexo_rl.training.aux_decode.mask_aux_rows` excludes bot rows from
    aux losses via the `n_pretrain` slice (caller must pass
    ``n_pretrain = n_pre + n_bot``).

    Bot games have one-hot policy targets (SealBot's chosen move). Per-row
    ``is_full_search`` returned by ``sample_batch`` should be overridden to 1
    by the caller (sharp targets, NOT quick-search-decimated).

    Args:
        mixing_cfg:       ``config["training"]["mixing"]`` sub-dict — reads
                          ``"bot_corpus_path"``.
        config:           Full merged config (for seed, encoding name).
        emit_fn:          ``emit_event`` callable for dashboard events.
        buffer_size:      Current self-play buffer size (for dashboard event).
        buffer_capacity:  Self-play buffer capacity (for dashboard event).

    Returns:
        Populated ``ReplayBuffer`` if path absent OR file missing → ``None``
        (back-compat: §177-style runs without bot corpus silently skip).
    """
    from engine import ReplayBuffer  # local import — engine only available post-build

    bot_path = mixing_cfg.get("bot_corpus_path")
    if not bot_path:
        return None
    if not Path(bot_path).exists():
        log.warning(
            "bot_corpus_npz_not_found",
            path=bot_path,
            msg=(
                "No bot-corpus NPZ at configured path — skipping bot-corpus slot. "
                "Run 'make corpus.bot' to generate (§178 design §4.2)."
            ),
        )
        return None

    log.info(
        "loading_bot_corpus_npz",
        path=bot_path,
        msg="copying §178 bot corpus into Rust ReplayBuffer",
    )
    t0 = time.time()
    data = np.load(bot_path, mmap_mode="r")
    board_size = config.get("board_size", BOARD_SIZE)
    bot_states   = data["states"]    # (T, 8, board_size, board_size) float16
    bot_policies = data["policies"]  # (T, policy_logit_count) float32 (one-hot)
    bot_outcomes = data["outcomes"]  # (T,) float32 ∈ {-1, +1}
    T = len(bot_outcomes)

    # Validate against the ACTIVE encoding's plane count (see load_pretrained
    # _buffer); resolves to 8 for v6, 10 for v6tp.
    _spec = _lookup_encoding(_normalize_encoding_name(config.get("encoding")))
    _expected_planes = _spec.n_planes
    if bot_states.shape[1] != _expected_planes:
        raise ValueError(
            f"bot corpus '{bot_path}': states.shape[1]={bot_states.shape[1]}, "
            f"expected {_expected_planes} (encoding "
            f"{_normalize_encoding_name(config.get('encoding'))!r}). Regenerate "
            f"with 'scripts/generate_bot_corpus.py'."
        )

    # §178 chain planes from stone planes. opp t0 (source 8) slot is encoding-
    # derived (4 for v6 family, 1 for v6_live2); cur t0 is always slot 0.
    from hexo_rl.env.game_state import _compute_chain_planes
    _opp_slot = opp_stone_slot(_spec)
    bot_chain = np.empty((T, 6, board_size, board_size), dtype=np.float16)
    if T > 0:
        cur_all = np.asarray(bot_states[:, 0], dtype=np.float32)
        opp_all = np.asarray(bot_states[:, _opp_slot], dtype=np.float32)
        for k in range(T):
            planes_f32 = _compute_chain_planes(cur_all[k], opp_all[k]).astype(np.float32) / 6.0
            bot_chain[k] = planes_f32.astype(np.float16)

    file_mb = os.path.getsize(bot_path) / 1e6
    log.info("bot_corpus_prefill", positions=T, file_mb=round(file_mb, 1))

    _enc = _normalize_encoding_name(config.get("encoding"))
    bot_buffer = ReplayBuffer(capacity=max(T, 1), encoding=_enc)
    # Neutral aux: ownership=1 ("empty"→0.0 after decode), winning_line=0.
    # Bot rows masked out of aux loss via n_pretrain = n_pre + n_bot slice.
    n_cells = board_size * board_size
    bot_own = np.ones((T, n_cells), dtype=np.uint8)
    bot_wl  = np.zeros((T, n_cells), dtype=np.uint8)
    bot_buffer.push_game(bot_states, bot_chain, bot_policies, bot_outcomes, bot_own, bot_wl)
    del bot_states, bot_chain, bot_policies, bot_outcomes, bot_own, bot_wl
    del data

    log.info("bot_corpus_loaded", positions=T, seconds=f"{time.time() - t0:.1f}")
    emit_fn({"event": "system_stats", "buffer_size": buffer_size, "buffer_capacity": buffer_capacity})
    return bot_buffer


# ── §S181-AUDIT Wave 3 Stage 2A — atomic bot-corpus NPZ swap ─────────────────

class BotCorpusSwapError(Exception):
    """Raised when atomic swap of a refreshed bot-corpus NPZ fails the
    integrity check (sha mismatch, missing sidecar, missing tmp file).
    The canonical NPZ is preserved untouched in every failure path.
    """


def swap_bot_corpus_atomic(
    canonical_path: str | Path,
    tmp_path: str | Path,
) -> tuple[str, str]:
    """Atomically swap a freshly-written NPZ into the canonical bot-corpus slot.

    Mirrors the ``anchor.py:save_best_model_atomic`` write-rename-verify
    pattern (§S179c §3.1 — coordinator owns the swap, not the subprocess).
    Sequence:

        1. Validate ``tmp_path`` and its sidecar exist.
        2. Read sidecar's declared sha256.
        3. Recompute sha256 of ``tmp_path``; reject on mismatch
           (quarantine as ``.corrupt-<ts>``).
        4. ``os.rename(canonical_path, canonical_path + ".bak")`` if canonical exists.
        5. ``os.rename(tmp_path, canonical_path)`` — POSIX-atomic.
        6. ``os.rename(<tmp sidecar>, <canonical sidecar>)``.
        7. Return ``(old_sha, new_sha)`` for forensic logging.

    Returns:
        ``(old_canonical_sha, new_canonical_sha)``. Empty string on the
        ``old_canonical_sha`` slot when no canonical NPZ existed at swap time.

    Raises:
        ``BotCorpusSwapError`` on missing tmp/sidecar or sha mismatch. The
        canonical NPZ is untouched in every failure path; the bad tmp is
        renamed aside as ``.corrupt-<ts>``.
    """
    # Local imports (cold path; keeps module import-time work minimal).
    import os
    import time as _time
    from hexo_rl.bootstrap.corpus_io import (
        compute_npz_sha256 as _sha256_of_npz,
        _sidecar_path as _sidecar_for,
    )

    canonical = Path(canonical_path)
    tmp = Path(tmp_path)

    if not tmp.exists():
        raise BotCorpusSwapError(
            f"refresh tmp NPZ missing: {tmp} — subprocess may have crashed "
            f"between save and rename"
        )

    tmp_sidecar = _sidecar_for(tmp)
    if not tmp_sidecar.exists():
        raise BotCorpusSwapError(
            f"refresh tmp sidecar missing: {tmp_sidecar} — refusing swap; "
            f"canonical NPZ retained"
        )

    # Sidecar declares sha; verify on-disk content matches.
    import json as _json
    try:
        meta = _json.loads(tmp_sidecar.read_text())
    except (OSError, _json.JSONDecodeError) as exc:
        raise BotCorpusSwapError(
            f"refresh tmp sidecar parse failed: {tmp_sidecar} ({exc})"
        ) from exc
    declared_sha = meta.get("sha256")
    if not isinstance(declared_sha, str) or not declared_sha:
        raise BotCorpusSwapError(
            f"refresh tmp sidecar missing 'sha256': {tmp_sidecar}"
        )

    actual_sha = _sha256_of_npz(tmp)
    if actual_sha != declared_sha:
        # Quarantine — do NOT clobber the canonical NPZ. INV-S179c-2 fail-safe.
        ts = _time.strftime("%Y%m%dT%H%M%S")
        corrupt = tmp.with_suffix(tmp.suffix + f".corrupt-{ts}")
        tmp.replace(corrupt)
        try:
            tmp_sidecar.replace(corrupt.with_name(corrupt.name + ".metadata.json"))
        except OSError:
            pass
        raise BotCorpusSwapError(
            f"refresh tmp sha mismatch (declared {declared_sha[:12]}…, actual "
            f"{actual_sha[:12]}…) — quarantined to {corrupt}; canonical retained"
        )

    # Compute old canonical sha for forensic log (empty string if no canonical
    # existed yet — refresh-from-empty-slot path).
    old_sha = _sha256_of_npz(canonical) if canonical.exists() else ""
    canonical_sidecar = _sidecar_for(canonical)

    # Step 4 — rotate canonical → .bak (one-cycle backup per s179c §3.3).
    if canonical.exists():
        bak = canonical.with_suffix(canonical.suffix + ".bak")
        canonical.replace(bak)
    if canonical_sidecar.exists():
        sidecar_bak = canonical_sidecar.with_suffix(canonical_sidecar.suffix + ".bak")
        canonical_sidecar.replace(sidecar_bak)

    # Step 5 — atomic rename of NPZ.
    tmp.replace(canonical)
    # Step 6 — sidecar second; if it fails, we leave the canonical NPZ as
    # un-sidecar'd (load path will warn + still function per corpus_io.load_corpus
    # back-compat warning). Mitigation per s179c §8 risk #7.
    try:
        tmp_sidecar.replace(canonical_sidecar)
    except OSError as exc:
        log.warning(
            "bot_corpus_swap_sidecar_rename_failed",
            sidecar_src=str(tmp_sidecar),
            sidecar_dst=str(canonical_sidecar),
            error=str(exc),
            msg="canonical NPZ swap committed but sidecar rename failed; load path will warn",
        )

    return old_sha, actual_sha


# ── Recent-buffer augmentation ───────────────────────────────────────────────

def _augment_recent_rows(
    s_r: np.ndarray,
    c_r: np.ndarray,
    p_r: np.ndarray,
    own_r_flat: np.ndarray,
    wl_r_flat: np.ndarray,
    augment: bool,
    opp_slot: int = 4,
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
    board_size = int(s_r.shape[-1])
    n_cells = board_size * board_size
    has_pass = p_r.shape[1] == n_cells + 1
    scatters = get_policy_scatters(board_size, has_pass=has_pass)
    sym_indices = np.random.randint(0, 12, size=n)

    states_f32 = s_r.astype(np.float32)
    if board_size == BOARD_SIZE:
        # v6 fast path — Rust kernel is hardcoded to 19×19.
        states_f32 = _engine.apply_symmetries_batch(states_f32, sym_indices.tolist())
        s_r = states_f32.astype(np.float16)
    else:
        # v6w25 / v8 pure-numpy scatter (Rust apply_symmetries_batch is v6-only).
        C = states_f32.shape[1]
        spatial = n_cells
        states_flat = states_f32.reshape(n, C, spatial)
        augmented = np.empty_like(states_flat)
        policy_aug = np.empty_like(p_r)
        for sym in range(12):
            mask_idx = np.where(sym_indices == sym)[0]
            if mask_idx.size == 0:
                continue
            sc = scatters[sym]
            state_scatter = sc[:spatial] if has_pass else sc
            augmented[mask_idx] = states_flat[mask_idx][:, :, state_scatter]
            policy_aug[mask_idx] = p_r[mask_idx][:, sc]
        states_f32 = augmented.reshape(n, C, board_size, board_size)
        s_r = states_f32.astype(np.float16)
        # chain_planes recomputed below (same for both paths)

    c_r_aug = np.empty_like(c_r)
    for i in range(n):
        c_r_aug[i] = (
            _compute_chain_planes(states_f32[i, 0], states_f32[i, opp_slot]).astype(np.float32) / 6.0
        ).astype(np.float16)

    scattered_p   = np.empty_like(p_r)
    scattered_own = np.empty_like(own_r_flat)
    scattered_wl  = np.empty_like(wl_r_flat)
    for i in range(n):
        lut = scatters[int(sym_indices[i])]
        scattered_p[i]   = p_r[i][lut]
        scattered_own[i] = own_r_flat[i][lut[:n_cells]]
        scattered_wl[i]  = wl_r_flat[i][lut[:n_cells]]

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
    *,
    bot_buffer: Optional[Any] = None,  # ReplayBuffer | None — §178 bot-corpus slot
    n_bot: int = 0,                    # §178 bot rows count
) -> BatchAssemblyResult:
    """Assemble one mixed batch from pretrain + bot + self-play (+ optional recent) buffers.

    During warm-up (buffers partially filled), falls back to ``np.concatenate``
    which allocates.  Once all sources return the full requested row count,
    switches to in-place ``np.copyto`` into ``bufs`` and clears
    ``bufs.warmup_active``.

    §178 bot slot (when ``bot_buffer is not None and n_bot > 0``): bot rows
    inserted between corpus and recent/uniform slots so the aux-mask
    ``[n_pretrain:]`` slice (caller passes ``n_pretrain = n_pre + n_bot``)
    excludes BOTH corpus AND bot rows from aux losses. Bot rows have one-hot
    SealBot targets → ``is_full_search=1`` enforced (overrides whatever
    ``sample_batch`` returned).

    Args:
        pretrained_buffer: Corpus Rust ReplayBuffer.
        buffer:            Self-play Rust ReplayBuffer.
        recent_buffer:     Optional Python RecentBuffer for recency weighting.
        n_pre:             Corpus rows to sample.
        n_self:            Self-play rows to sample.
        batch_size:        Total batch size this step (should equal n_pre + n_bot + n_self).
        batch_size_cfg:    Pre-allocated buffer batch size; if they differ we fall back
                           to concat to avoid out-of-bounds writes.
        recency_weight:    Fraction of self-play rows taken from recent_buffer.
        bufs:              Pre-allocated batch arrays (modified in-place when steady-state).
        train_step:        Current step index (for log messages only).
        augment:           Apply 12-fold hex symmetry augmentation during Rust sample_batch.
                           Default True preserves production behaviour; set False only for
                           diagnostic runs (see CLAUDE.md § Testing conventions).
        bot_buffer:        §178 Optional Rust ReplayBuffer carrying SealBot-vs-anchor games.
                           When ``None`` or ``n_bot == 0`` the slot is omitted (back-compat
                           with §177 and earlier).
        n_bot:             §178 Number of bot rows to sample (0 = no bot slot).

    Returns:
        :class:`BatchAssemblyResult` — seven arrays
        ``(states, chain_planes, policies, outcomes, ownership, winning_line,
        is_full_search)`` (views into ``bufs`` in steady-state, freshly allocated
        during warm-up) plus ``n_recent_actual`` (int): actual number of rows drawn
        from recent_buffer (0 when recent_buffer absent or empty). Batch order is
        ``[corpus | bot | recent | uniform_self]`` when bot active, else
        ``[corpus | recent | uniform_self]`` (back-compat). Caller uses
        ``n_pre + n_bot + n_recent_actual`` to slice. Corpus AND bot positions
        always have ``is_full_search=1``.
    """
    s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre, pos_pre = pretrained_buffer.sample_batch_with_pos(n_pre, augment)

    # §178 bot slot — optional fourth piece (gates on size>0 like steady-state fallback).
    use_bot = bot_buffer is not None and n_bot > 0 and bot_buffer.size > 0
    if use_bot:
        s_b, c_b, p_b, o_b, own_b, wl_b, _ifs_b, pos_b = bot_buffer.sample_batch_with_pos(n_bot, augment)
        # Override is_full_search → 1 for all bot rows (one-hot SealBot targets are
        # full-search-equivalent; design §4.1).
        ifs_b = np.ones(len(s_b), dtype=np.uint8)
    else:
        n_bot = 0  # collapse to 0 so downstream slicing arithmetic is correct
        s_b = c_b = p_b = o_b = own_b = wl_b = ifs_b = pos_b = None

    if batch_size != batch_size_cfg:
        # Edge case: runtime batch size diverged from pre-allocated shape.
        if train_step > 100:
            log.warning("mixed_batch_size_mismatch", batch_size=batch_size, expected=batch_size_cfg)
        s_self, c_self, p_self, o_self, own_self, wl_self, ifs_self, pos_self = _sample_selfplay(
            buffer, recent_buffer, n_self, recency_weight, augment
        )
        # n_recent unknown in this mismatch path — report 0 (caller disables 3-way split).
        if use_bot:
            return BatchAssemblyResult(
                states=np.concatenate([s_pre, s_b, s_self], axis=0),
                chain_planes=np.concatenate([c_pre, c_b, c_self], axis=0),
                policies=np.concatenate([p_pre, p_b, p_self], axis=0),
                outcomes=np.concatenate([o_pre, o_b, o_self], axis=0),
                ownership=np.concatenate([own_pre, own_b, own_self], axis=0),
                winning_line=np.concatenate([wl_pre, wl_b, wl_self], axis=0),
                is_full_search=np.concatenate([ifs_pre, ifs_b, ifs_self], axis=0),
                n_recent_actual=0,
                position_indices=np.concatenate([pos_pre, pos_b, pos_self], axis=0),
            )
        return BatchAssemblyResult(
            states=np.concatenate([s_pre, s_self], axis=0),
            chain_planes=np.concatenate([c_pre, c_self], axis=0),
            policies=np.concatenate([p_pre, p_self], axis=0),
            outcomes=np.concatenate([o_pre, o_self], axis=0),
            ownership=np.concatenate([own_pre, own_self], axis=0),
            winning_line=np.concatenate([wl_pre, wl_self], axis=0),
            is_full_search=np.concatenate([ifs_pre, ifs_self], axis=0),
            n_recent_actual=0,
            position_indices=np.concatenate([pos_pre, pos_self], axis=0),
        )

    # ── Normal path: try in-place fill into bufs ──────────────────────────────
    use_recent = (
        recent_buffer is not None
        and recent_buffer.size > 0
        and recency_weight > 0.0
        and n_self > 1
    )

    bot_piece: Optional[tuple[Any, ...]] = None
    if use_bot:
        bot_piece = (s_b, c_b, p_b, o_b, own_b, wl_b, ifs_b, pos_b)

    n_recent_actual = 0
    if use_recent:
        n_recent_req  = max(1, int(round(n_self * recency_weight)))
        n_uniform     = n_self - n_recent_req
        s_r, c_r, p_r, o_r, own_r_flat, wl_r_flat, ifs_r = recent_buffer.sample(n_recent_req)
        s_r, c_r, p_r, own_r_flat, wl_r_flat = _augment_recent_rows(
            s_r, c_r, p_r, own_r_flat, wl_r_flat, augment, opp_stone_slot(buffer.encoding)
        )
        _bs = int(s_r.shape[-1])
        own_r = own_r_flat.reshape(-1, _bs, _bs)
        wl_r  = wl_r_flat.reshape(-1, _bs, _bs)
        # Recent buffer lacks per-row position_indices — fill zeros (ply_index head
        # masking can ignore these rows if needed; default behaviour treats them as
        # "early-game" target).
        pos_r = np.zeros(len(s_r), dtype=np.uint16)
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u = buffer.sample_batch_with_pos(max(1, n_uniform), augment)
        n_recent_actual = len(s_r)
        pieces    = [(s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre, pos_pre)]
        if bot_piece is not None:
            pieces.append(bot_piece)
        pieces.extend([(s_r, c_r, p_r, o_r, own_r, wl_r, ifs_r, pos_r),
                       (s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u)])
        n_avail = n_pre + (n_bot if use_bot else 0) + len(s_r) + len(s_u)
    else:
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u = buffer.sample_batch_with_pos(max(1, n_self), augment)
        pieces = [(s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre, pos_pre)]
        if bot_piece is not None:
            pieces.append(bot_piece)
        pieces.append((s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u))
        n_avail = n_pre + (n_bot if use_bot else 0) + len(s_u)

    if n_avail < batch_size:
        # Warm-up: one or more sources returned fewer rows than requested.
        return BatchAssemblyResult(
            states=np.concatenate([p[0] for p in pieces], axis=0),
            chain_planes=np.concatenate([p[1] for p in pieces], axis=0),
            policies=np.concatenate([p[2] for p in pieces], axis=0),
            outcomes=np.concatenate([p[3] for p in pieces], axis=0),
            ownership=np.concatenate([p[4] for p in pieces], axis=0),
            winning_line=np.concatenate([p[5] for p in pieces], axis=0),
            is_full_search=np.concatenate([p[6] for p in pieces], axis=0),
            n_recent_actual=n_recent_actual,
            position_indices=np.concatenate([p[7] for p in pieces], axis=0),
        )

    # Steady-state: in-place copy, no heap allocation.
    if bufs.warmup_active:
        log.info("buffer_warmup_ended", step=train_step, n_available=n_avail, batch_size=batch_size)
        bufs.warmup_active = False

    # Concatenate position_indices for return (small u16 array; not in bufs).
    out_pos = np.concatenate([p[7] for p in pieces], axis=0)

    offset = 0
    for s, c, p, o, own, wl, ifs, _pos in pieces:
        n = len(s)
        np.copyto(bufs.states[offset:offset + n],          s)
        np.copyto(bufs.chain_planes[offset:offset + n],    c)
        np.copyto(bufs.policies[offset:offset + n],        p)
        np.copyto(bufs.outcomes[offset:offset + n],        o)
        np.copyto(bufs.ownership[offset:offset + n],       own)
        np.copyto(bufs.winning_line[offset:offset + n],    wl)
        np.copyto(bufs.is_full_search[offset:offset + n],  ifs)
        offset += n

    return BatchAssemblyResult(
        states=bufs.states,
        chain_planes=bufs.chain_planes,
        policies=bufs.policies,
        outcomes=bufs.outcomes,
        ownership=bufs.ownership,
        winning_line=bufs.winning_line,
        is_full_search=bufs.is_full_search,
        n_recent_actual=n_recent_actual,
        position_indices=out_pos,
    )


def _sample_selfplay(
    buffer: Any,
    recent_buffer: Optional[Any],
    n_self: int,
    recency_weight: float,
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample self-play rows, blending recent + uniform when recency_weight > 0.

    §S181-AUDIT Wave 4 4B-impl-3 — 8-tuple now (added position_indices); recent
    buffer rows fill zeros since they lack per-row ply tracking.
    """
    if (recent_buffer is not None and recent_buffer.size > 0
            and recency_weight > 0.0 and n_self > 1):
        n_r = max(1, int(round(n_self * recency_weight)))
        n_u = n_self - n_r
        s_r, c_r, p_r, o_r, own_r_flat, wl_r_flat, ifs_r = recent_buffer.sample(n_r)
        s_r, c_r, p_r, own_r_flat, wl_r_flat = _augment_recent_rows(
            s_r, c_r, p_r, own_r_flat, wl_r_flat, augment, opp_stone_slot(buffer.encoding)
        )
        _bs = int(s_r.shape[-1])
        own_r = own_r_flat.reshape(-1, _bs, _bs)
        wl_r  = wl_r_flat.reshape(-1, _bs, _bs)
        pos_r = np.zeros(len(s_r), dtype=np.uint16)
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u = buffer.sample_batch_with_pos(max(1, n_u), augment)
        return (
            np.concatenate([s_r, s_u], axis=0),
            np.concatenate([c_r, c_u], axis=0),
            np.concatenate([p_r, p_u], axis=0),
            np.concatenate([o_r, o_u], axis=0),
            np.concatenate([own_r, own_u], axis=0),
            np.concatenate([wl_r, wl_u], axis=0),
            np.concatenate([ifs_r, ifs_u], axis=0),
            np.concatenate([pos_r, pos_u], axis=0),
        )
    return buffer.sample_batch_with_pos(max(1, n_self), augment)
