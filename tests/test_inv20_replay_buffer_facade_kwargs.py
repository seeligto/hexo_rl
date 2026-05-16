"""INV20 Python pin — `ReplayBuffer` push API facade kwarg surface preserved
across the cycle 3 Wave 7 Batch B P79 refactor (NON-BREAKING contract).

The Rust pin at
`engine/tests/inv20_replay_buffer_push_config_field_shape.rs` (3 cargo tests)
locks the 3 internal config struct shapes at compile time. This Python pin
locks the user-visible PyO3 facade contract:

  1. Parameter names + order + per-default presence on `ReplayBuffer.push`,
     `push_game`, and `push_many` match the documented pre-refactor surface
     byte-for-byte.
  2. Known-input push calls produce the expected buffer state (forward-only
     behavioral pin — catches accidental drift in `weight_for(game_length)`,
     `is_full_search` byte cast, or ring-buffer accounting through the new
     config-struct delegation).

Renumbered from PREP §B's proposed `INV19` because Wave 6.5 took INV18 +
INV18b (i32::midpoint revert pins) and Wave 7 Batch A took INV19
(SelfPlayRunnerConfig builder pin).
"""
from __future__ import annotations

import inspect

import numpy as np

from engine import ReplayBuffer


# ── Expected facade signatures (post-Wave-7 Batch B, NON-BREAKING) ────────────

# Parameter (name, has_default) tuples in declaration order. Defaults values are
# spot-checked where introspectable; PyO3 0.28 reports negative-int defaults as
# `Ellipsis`, so `game_id`'s default of `-1` is only verified by behavior in
# Test 2, not by introspected value.
EXPECTED_PUSH_PARAMS = [
    ("self",           False),
    ("state",          False),
    ("chain_planes",   False),
    ("policy",         False),
    ("outcome",        False),
    ("ownership",      False),
    ("winning_line",   False),
    ("game_id",        True),   # default −1 (shown as Ellipsis via PyO3)
    ("game_length",    True),   # default 0
    ("is_full_search", True),   # default True
]

EXPECTED_PUSH_GAME_PARAMS = [
    ("self",           False),
    ("states",         False),
    ("chain_planes",   False),
    ("policies",       False),
    ("outcomes",       False),
    ("ownership",      False),
    ("winning_line",   False),
    ("game_id",        True),   # default −1
    ("game_length",    True),   # default 0
    ("is_full_search", True),   # default None
]

EXPECTED_PUSH_MANY_PARAMS = [
    ("self",           False),
    ("states",         False),
    ("chain_planes",   False),
    ("policies",       False),
    ("outcomes",       False),
    ("ownership",      False),
    ("winning_line",   False),
    ("game_lengths",   False),
    ("is_full_search", False),
]


def _check_signature(method, expected) -> None:
    sig = inspect.signature(method)
    actual = [(name, p.default is not inspect._empty) for name, p in sig.parameters.items()]
    assert actual == expected, (
        f"{method.__qualname__} parameter shape drifted from INV20 pin:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )


# ── Test 1 — facade kwarg surface preserved ──────────────────────────────────

def test_push_facade_kwargs_signatures_preserved():
    """INV20 — the 3 push facade methods retain pre-refactor parameter names + defaults."""
    _check_signature(ReplayBuffer.push,      EXPECTED_PUSH_PARAMS)
    _check_signature(ReplayBuffer.push_game, EXPECTED_PUSH_GAME_PARAMS)
    _check_signature(ReplayBuffer.push_many, EXPECTED_PUSH_MANY_PARAMS)

    # Spot-check introspectable defaults (PyO3 0.28 reports i64 −1 as Ellipsis;
    # those defaults are pinned by behavior in Test 2 below, not by value here).
    push_params = inspect.signature(ReplayBuffer.push).parameters
    assert push_params["game_length"].default == 0
    assert push_params["is_full_search"].default is True

    push_game_params = inspect.signature(ReplayBuffer.push_game).parameters
    assert push_game_params["game_length"].default == 0
    assert push_game_params["is_full_search"].default is None


# ── Test 2 — facade byte-identical buffer state (forward-only pin) ───────────

def test_push_facade_byte_identical_buffer_state():
    """INV20 — known-input push calls produce known buffer state.

    Forward-only behavioral pin: pins outcome / is_full_search / size side
    effects of `push`, `push_game`, and `push_many` against accidental drift
    introduced by Wave 7 Batch B's config-struct delegation. Uses homogeneous
    inputs per facade method so the (weighted, replacement) `sample_batch`
    can verify the per-row aggregate without depending on which slot a sample
    lands on.
    """
    # ── push (single position) ─────────────────────────────────────────────
    # Push 8 rows all with the same (outcome=0.5, is_full_search=False).
    buf = ReplayBuffer(capacity=8)
    state = np.zeros((8, 19, 19), dtype=np.float16); state[0, 0, 0] = 1.0
    chain = np.zeros((6, 19, 19), dtype=np.float16)
    policy = np.zeros(362, dtype=np.float32); policy[0] = 1.0
    own = np.ones(361, dtype=np.uint8)
    wl  = np.zeros(361, dtype=np.uint8)
    for _ in range(8):
        buf.push(state, chain, policy, 0.5, own, wl,
                 game_id=42, game_length=10, is_full_search=False)
    assert buf.size == 8
    _, _, _, outcomes, _, _, ifs = buf.sample_batch(16, augment=False)
    assert outcomes.shape == (16,)
    assert ifs.shape == (16,)
    np.testing.assert_allclose(outcomes, 0.5, atol=1e-3,
        err_msg="push outcome must round-trip exactly")
    assert (ifs == 0).all(), \
        "push is_full_search=False must store as 0u8 in every row"

    # Same shape with the default (is_full_search=True, game_length=0).
    buf2 = ReplayBuffer(capacity=8)
    for _ in range(8):
        buf2.push(state, chain, policy, -1.0, own, wl)  # all defaults
    assert buf2.size == 8
    _, _, _, outcomes2, _, _, ifs2 = buf2.sample_batch(16, augment=False)
    np.testing.assert_allclose(outcomes2, -1.0, atol=1e-3)
    assert (ifs2 == 1).all(), \
        "push default is_full_search=True must store as 1u8 in every row"

    # ── push_game (T=4, shared metadata + per-row is_full_search) ─────────
    buf3 = ReplayBuffer(capacity=8)
    t = 4
    states_b = np.zeros((t, 8, 19, 19), dtype=np.float16)
    chain_b  = np.zeros((t, 6, 19, 19), dtype=np.float16)
    pol_b    = np.zeros((t, 362), dtype=np.float32); pol_b[:, 0] = 1.0
    out_b    = np.full(t, 0.25, dtype=np.float32)
    own_b    = np.ones((t, 361), dtype=np.uint8)
    wl_b     = np.zeros((t, 361), dtype=np.uint8)
    ifs_b    = np.zeros(t, dtype=np.uint8)  # all rows is_full_search=False
    buf3.push_game(states_b, chain_b, pol_b, out_b, own_b, wl_b,
                   game_id=7, game_length=20, is_full_search=ifs_b)
    assert buf3.size == t
    _, _, _, outcomes3, _, _, ifs3 = buf3.sample_batch(16, augment=False)
    np.testing.assert_allclose(outcomes3, 0.25, atol=1e-3,
        err_msg="push_game shared outcome must round-trip")
    assert (ifs3 == 0).all(), \
        "push_game is_full_search=zeros must store as 0u8"

    # ── push_game with is_full_search=None default → all rows full-search ─
    buf4 = ReplayBuffer(capacity=8)
    buf4.push_game(states_b, chain_b, pol_b, out_b, own_b, wl_b,
                   game_id=8, game_length=20)
    assert buf4.size == t
    _, _, _, _outc4, _, _, ifs4 = buf4.sample_batch(16, augment=False)
    assert (ifs4 == 1).all(), \
        "push_game default is_full_search=None must default to 1u8"

    # ── push_many (N=4, per-row metadata) ──────────────────────────────────
    buf5 = ReplayBuffer(capacity=8)
    n = 4
    states_n = np.zeros((n, 8, 19, 19), dtype=np.float16)
    chain_n  = np.zeros((n, 6, 19, 19), dtype=np.float16)
    pol_n    = np.zeros((n, 362), dtype=np.float32); pol_n[:, 0] = 1.0
    out_n    = np.full(n, 0.75, dtype=np.float32)
    own_n    = np.ones((n, 361), dtype=np.uint8)
    wl_n     = np.zeros((n, 361), dtype=np.uint8)
    gls_n    = np.full(n, 10, dtype=np.uint16)
    ifs_n    = np.zeros(n, dtype=np.uint8)
    buf5.push_many(states_n, chain_n, pol_n, out_n, own_n, wl_n, gls_n, ifs_n)
    assert buf5.size == n
    _, _, _, outcomes5, _, _, ifs5 = buf5.sample_batch(16, augment=False)
    np.testing.assert_allclose(outcomes5, 0.75, atol=1e-3,
        err_msg="push_many homogeneous outcome must round-trip")
    assert (ifs5 == 0).all(), \
        "push_many is_full_search=zeros must store as 0u8"
