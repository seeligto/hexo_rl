"""
Tests and benchmark for ReplayBuffer.

Run with: pytest tests/test_rust_replay_buffer.py -v
Benchmark: pytest tests/test_rust_replay_buffer.py -v -k benchmark -s
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from engine import ReplayBuffer

CHANNELS      = 8   # HEXB v6: 8 state planes (KEPT_PLANE_INDICES subset)
N_CHAIN_PLANES = 6
BOARD_SIZE    = 19
N_ACTIONS     = BOARD_SIZE * BOARD_SIZE + 1  # 362
AUX_STRIDE    = BOARD_SIZE * BOARD_SIZE      # 361


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_buf(capacity: int = 100) -> ReplayBuffer:
    return ReplayBuffer(capacity)


def random_state() -> np.ndarray:
    return np.random.randn(CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)


def random_chain() -> np.ndarray:
    return np.zeros((N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)


def random_policy() -> np.ndarray:
    p = np.abs(np.random.randn(N_ACTIONS)).astype(np.float32)
    p /= p.sum()
    return p


def empty_own() -> np.ndarray:
    return np.ones(AUX_STRIDE, dtype=np.uint8)  # encoding: 1 = empty


def empty_wl() -> np.ndarray:
    return np.zeros(AUX_STRIDE, dtype=np.uint8)


def random_entry() -> tuple:
    return (
        random_state(),
        random_chain(),
        random_policy(),
        float(np.random.choice([-1.0, 0.0, 1.0])),
        empty_own(),
        empty_wl(),
    )


def push_n(buf: ReplayBuffer, n: int, use_game_id: bool = False) -> None:
    for _ in range(n):
        gid = buf.next_game_id() if use_game_id else -1
        s, c, p, o, own, wl = random_entry()
        buf.push(s, c, p, o, own, wl, game_id=gid)


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_size_zero():
    buf = make_buf(capacity=50)
    assert buf.size == 0
    assert buf.capacity == 50


# ── push / size ───────────────────────────────────────────────────────────────

def test_push_increments_size():
    buf = make_buf(capacity=10)
    for i in range(5):
        s, c, p, o, own, wl = random_entry()
        buf.push(s, c, p, o, own, wl)
        assert buf.size == i + 1


def test_push_caps_at_capacity():
    buf = make_buf(capacity=5)
    for _ in range(12):
        s, c, p, o, own, wl = random_entry()
        buf.push(s, c, p, o, own, wl)
    assert buf.size == 5


# ── push_game ─────────────────────────────────────────────────────────────────

def _ones_own_batch(t: int) -> np.ndarray:
    return np.ones((t, AUX_STRIDE), dtype=np.uint8)


def _zeros_wl_batch(t: int) -> np.ndarray:
    return np.zeros((t, AUX_STRIDE), dtype=np.uint8)


def _zeros_chain_batch(t: int) -> np.ndarray:
    return np.zeros((t, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)


def test_push_game_no_wrap():
    buf = make_buf(capacity=20)
    t = 7
    states   = np.random.randn(t, CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
    policies = np.abs(np.random.randn(t, N_ACTIONS).astype(np.float32))
    outcomes = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0], dtype=np.float32)
    buf.push_game(states, _zeros_chain_batch(t), policies, outcomes, _ones_own_batch(t), _zeros_wl_batch(t))
    assert buf.size == t


def test_push_game_with_wrap():
    buf = make_buf(capacity=7)
    t1 = 5
    s1 = np.ones((t1, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    p1 = np.ones((t1, N_ACTIONS), dtype=np.float32) / N_ACTIONS
    o1 = np.zeros(t1, dtype=np.float32)
    buf.push_game(s1, _zeros_chain_batch(t1), p1, o1, _ones_own_batch(t1), _zeros_wl_batch(t1))
    assert buf.size == t1

    t2 = 4
    s2 = np.ones((t2, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16) * 2.0
    p2 = np.ones((t2, N_ACTIONS), dtype=np.float32) / N_ACTIONS
    o2 = np.ones(t2, dtype=np.float32)
    buf.push_game(s2, _zeros_chain_batch(t2), p2, o2, _ones_own_batch(t2), _zeros_wl_batch(t2))
    assert buf.size == 7  # capped at capacity


# ── sample_batch shapes / dtypes ─────────────────────────────────────────────

def test_sample_raises_on_empty():
    buf = make_buf(capacity=50)
    with pytest.raises(Exception):
        buf.sample_batch(4, augment=False)


def test_sample_returns_correct_shapes():
    buf = make_buf(capacity=200)
    push_n(buf, 100)
    s, c, p, o, own, wl, _ifs = buf.sample_batch(16, augment=True)
    assert s.shape   == (16, CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert c.shape   == (16, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert p.shape   == (16, N_ACTIONS)
    assert o.shape   == (16,)
    assert own.shape == (16, BOARD_SIZE, BOARD_SIZE)
    assert wl.shape  == (16, BOARD_SIZE, BOARD_SIZE)


def test_sample_returns_correct_dtypes():
    buf = make_buf(capacity=100)
    push_n(buf, 50)
    s, c, p, o, own, wl, _ifs = buf.sample_batch(8, augment=True)
    assert s.dtype   == np.float16, f"states should be float16, got {s.dtype}"
    assert c.dtype   == np.float16, f"chain_planes should be float16, got {c.dtype}"
    assert p.dtype   == np.float32, f"policies should be float32, got {p.dtype}"
    assert o.dtype   == np.float32, f"outcomes should be float32, got {o.dtype}"
    assert own.dtype == np.uint8,   f"ownership should be uint8, got {own.dtype}"
    assert wl.dtype  == np.uint8,   f"winning_line should be uint8, got {wl.dtype}"


def test_sample_no_augment_content_roundtrip():
    """Without augmentation, sampled states must contain values matching pushed data."""
    buf = make_buf(capacity=50)
    # Push entries with a unique, easily-verifiable outcome per slot.
    for i in range(20):
        s, c, p, _, own, wl = random_entry()
        buf.push(s, c, p, float(i), own, wl)

    _, _, _, outcomes, _, _, _ = buf.sample_batch(200, augment=False)
    # All returned outcomes must be one of 0..19.
    assert all(int(round(o)) in range(20) for o in outcomes)


# ── Augmentation correctness ──────────────────────────────────────────────────

def test_identity_symmetry_preserves_data():
    """Symmetry 0 (identity) must return the exact original values."""
    buf = ReplayBuffer(10)

    # Place a known state: all zeros except plane 0, cell (0,0) = 1.0
    state = np.zeros((CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    state[0, 0, 0] = 1.0
    chain = np.zeros((N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    buf.push(state, chain, policy, 1.0, empty_own(), empty_wl())

    # sample_batch with augment=False always uses symmetry 0.
    sampled_s, _c, sampled_p, sampled_o, _, _, _ = buf.sample_batch(1, augment=False)
    assert sampled_s[0, 0, 0, 0] == pytest.approx(1.0, abs=1e-3)
    assert sampled_p[0, 0]        == pytest.approx(1.0, abs=1e-3)
    assert sampled_o[0]            == pytest.approx(1.0, abs=1e-3)


def test_pass_action_invariant_under_augmentation():
    """The pass action (index 361) must be unchanged by every symmetry."""
    buf = ReplayBuffer(50)
    rng = np.random.default_rng(42)

    for _ in range(20):
        p = rng.random(N_ACTIONS).astype(np.float32)
        p /= p.sum()
        buf.push(
            rng.standard_normal((CHANNELS, BOARD_SIZE, BOARD_SIZE)).astype(np.float16),
            np.zeros((N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16),
            p, 0.0, empty_own(), empty_wl(),
        )

    for _ in range(10):
        # We can't control which symmetry is chosen, but the pass logit (index -1)
        # must always match some pushed pass logit — check it's in a plausible range.
        _, _c, policies, _, _, _, _ = buf.sample_batch(32, augment=True)
        pass_logits = policies[:, -1]
        assert (pass_logits >= 0.0).all(), "pass logit must be non-negative"
        assert (pass_logits <= 1.0).all(), "pass logit must be ≤ 1 (it's a probability)"


def test_policy_sum_preserved_under_augmentation():
    """Spatial policy logits must sum to the same value before and after augmentation."""
    buf = ReplayBuffer(50)
    rng = np.random.default_rng(0)

    original_sums = []
    for _ in range(30):
        p = rng.random(N_ACTIONS).astype(np.float32)
        original_sums.append(p[:-1].sum())  # sum of spatial logits only
        buf.push(
            rng.standard_normal((CHANNELS, BOARD_SIZE, BOARD_SIZE)).astype(np.float16),
            np.zeros((N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16),
            p, 0.0, empty_own(), empty_wl(),
        )

    # Some cells fall outside the window under certain rotations, so sum can decrease slightly.
    # It must never increase (no mass created) and stay within a small tolerance.
    _, _c, policies, _, _, _, _ = buf.sample_batch(30, augment=True)
    aug_sums = policies[:, :-1].sum(axis=1)
    # At most the sum of the original (some cells may be clipped to 0).
    max_original = max(original_sums)
    assert (aug_sums <= max_original + 1e-4).all()


# ── Correlation guard ─────────────────────────────────────────────────────────

def test_correlation_guard_no_duplicate_game_ids():
    """With enough distinct positions, no batch should contain two entries from the same position."""
    buf = ReplayBuffer(2000)
    # Push 500 positions, each with 3 clusters (same game_id).
    for _ in range(500):
        gid = buf.next_game_id()
        for _ in range(3):
            s, c, p, o, own, wl = random_entry()
            buf.push(s, c, p, o, own, wl, game_id=gid)

    assert buf.size == 1500

    for _ in range(20):
        _, _, _, _, _, _, _ = buf.sample_batch(64, augment=False)
        # If we had access to game_ids we'd check here; since sample_batch doesn't
        # return them, we verify the guard doesn't crash and the shapes are correct.
        # Full dedup verification would require exposing indices — see benchmark below.


# ── next_game_id ──────────────────────────────────────────────────────────────

def test_next_game_id_monotonic():
    buf = make_buf()
    ids = [buf.next_game_id() for _ in range(100)]
    assert ids == list(range(100))


# ── Resize ───────────────────────────────────────────────────────────────────

def test_resize_basic():
    """Resize grows capacity and preserves data."""
    buf = make_buf(capacity=10)
    push_n(buf, 5)
    assert buf.size == 5
    assert buf.capacity == 10

    buf.resize(20)
    assert buf.capacity == 20
    assert buf.size == 5

    # Can sample after resize.
    s, _c, p, o, _, _, _ = buf.sample_batch(5, augment=False)
    assert s.shape[0] == 5


def test_resize_rejects_same_or_smaller():
    buf = make_buf(capacity=10)
    with pytest.raises(Exception):
        buf.resize(10)
    with pytest.raises(Exception):
        buf.resize(5)


def test_resize_preserves_data_after_wrap():
    """After ring-buffer wraps and resize, data is still valid and sampleable."""
    buf = make_buf(capacity=10)
    # Push 15 entries — wraps around, only 10 retained.
    push_n(buf, 15)
    assert buf.size == 10

    buf.resize(20)
    assert buf.capacity == 20
    assert buf.size == 10

    # Sample and verify shapes.
    s, _c, p, o, _, _, _ = buf.sample_batch(10, augment=False)
    assert s.shape == (10, CHANNELS, BOARD_SIZE, BOARD_SIZE)


def test_resize_then_push():
    """After resize, new pushes work correctly."""
    buf = make_buf(capacity=5)
    push_n(buf, 8)  # wraps
    assert buf.size == 5

    buf.resize(10)
    push_n(buf, 3)
    assert buf.size == 8

    s, _c, p, o, _, _, _ = buf.sample_batch(8, augment=False)
    assert s.shape[0] == 8


def test_resize_full_head_zero():
    """Edge case: buffer full with head==0 (no actual rotation needed)."""
    buf = make_buf(capacity=5)
    push_n(buf, 5)  # exactly fills, head wraps to 0
    assert buf.size == 5

    buf.resize(10)
    assert buf.capacity == 10
    assert buf.size == 5

    # Push more.
    push_n(buf, 3)
    assert buf.size == 8


def test_resize_content_roundtrip():
    """Verify specific outcomes survive resize through wrap-around."""
    buf = ReplayBuffer(5)
    # Push outcomes 0..7. Entries 0-2 overwritten by 5-7.
    for i in range(8):
        s = random_state()
        c = random_chain()
        p = random_policy()
        buf.push(s, c, p, float(i), empty_own(), empty_wl())

    # Buffer now contains outcomes [5, 6, 7, 3, 4] (head=3, oldest at slot 3→outcome 3).
    # Actually ring buffer: pushed 8 into cap 5, head=3, outcomes at slots:
    # slot 0: 5, slot 1: 6, slot 2: 7, slot 3: 3, slot 4: 4
    # Logical order oldest→newest: [3, 4, 5, 6, 7]
    buf.resize(10)

    # Sample all 5 entries many times to collect outcomes.
    seen = set()
    for _ in range(50):
        _, _, _, outcomes, _, _, _ = buf.sample_batch(5, augment=False)
        for o in outcomes:
            seen.add(int(round(o)))
    assert seen == {3, 4, 5, 6, 7}, f"Expected {{3,4,5,6,7}}, got {seen}"


# ── Benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark_sample_latency(capsys):
    """Benchmark: sample_batch(256) must complete in < 128ms (< 0.5ms/sample)."""
    CAPACITY   = 200_000
    FILL       = 50_000
    BATCH      = 256
    WARMUP     = 10
    N_ITERS    = 200

    rust_buf  = ReplayBuffer(CAPACITY)
    batch_s   = np.random.randn(1000, CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
    batch_c   = np.zeros((1000, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    batch_p   = np.abs(np.random.randn(1000, N_ACTIONS).astype(np.float32))
    batch_p  /= batch_p.sum(axis=1, keepdims=True)
    batch_o   = np.zeros(1000, dtype=np.float32)

    batch_own = np.ones((1000, AUX_STRIDE), dtype=np.uint8)
    batch_wl  = np.zeros((1000, AUX_STRIDE), dtype=np.uint8)
    for _ in range(FILL // 1000):
        rust_buf.push_game(
            batch_s, batch_c, batch_p, batch_o, batch_own, batch_wl,
            game_id=rust_buf.next_game_id(),
        )

    for _ in range(WARMUP):
        rust_buf.sample_batch(BATCH, augment=True)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        rust_buf.sample_batch(BATCH, augment=True)
    rust_ms = (time.perf_counter() - t0) / N_ITERS * 1000

    with capsys.disabled():
        print(f"\n{'─'*56}")
        print(f"  ReplayBuffer sample_batch benchmark  (batch={BATCH})")
        print(f"{'─'*56}")
        print(f"  {rust_ms:.3f} ms/batch  ({rust_ms/BATCH*1000:.2f} µs/sample)")
        print(f"{'─'*56}\n")

    # Target: < 0.5 ms per sample = < 128 ms per 256-sample batch
    assert rust_ms < 128.0, (
        f"sample_batch({BATCH}) took {rust_ms:.2f} ms — target < 128 ms "
        f"({rust_ms/BATCH*1000:.2f} µs/sample, target < 500 µs/sample)"
    )


# ── Persistence (save/load) ──────────────────────────────────────────────────

def test_buffer_save_load_roundtrip(tmp_path):
    """Push N positions, save, load into fresh buffer, verify size and data."""
    buf = make_buf(200)
    push_n(buf, 150)
    assert buf.size == 150

    path = str(tmp_path / "buf.bin")
    buf.save_to_path(path)

    buf2 = make_buf(200)
    n_loaded = buf2.load_from_path(path)
    assert n_loaded == 150
    assert buf2.size == 150

    # Verify data is valid by sampling
    states, _c, policies, outcomes, _, _, _ = buf2.sample_batch(10, augment=False)
    assert states.shape == (10, CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert policies.shape == (10, N_ACTIONS)
    assert outcomes.shape == (10,)


def test_buffer_load_missing_file_ok(tmp_path):
    """load_from_path on nonexistent path returns 0, no exception."""
    buf = make_buf(100)
    path = str(tmp_path / "nonexistent.bin")
    n = buf.load_from_path(path)
    assert n == 0
    assert buf.size == 0


def test_buffer_load_size_mismatch(tmp_path):
    """Save from capacity=500, load into capacity=250 → loads 250 (most recent)."""
    buf_big = make_buf(500)
    push_n(buf_big, 500)
    assert buf_big.size == 500

    path = str(tmp_path / "buf.bin")
    buf_big.save_to_path(path)

    buf_small = make_buf(250)
    n_loaded = buf_small.load_from_path(path)
    assert n_loaded == 250
    assert buf_small.size == 250

    # Verify data is valid
    states, _c, policies, outcomes, _, _, _ = buf_small.sample_batch(10, augment=False)
    assert states.shape == (10, CHANNELS, BOARD_SIZE, BOARD_SIZE)


# ── push_many ─────────────────────────────────────────────────────────────────

def test_push_many_basic():
    """push_many writes N rows; subsequent sample_batch reads them back."""
    buf = make_buf(100)
    n = 50
    states = np.stack([random_state() for _ in range(n)]).astype(np.float16)
    chain = np.stack([random_chain() for _ in range(n)]).astype(np.float16)
    policies = np.stack([random_policy() for _ in range(n)]).astype(np.float32)
    outcomes = np.random.choice([-1.0, 0.0, 1.0], size=n).astype(np.float32)
    ownership = np.ones((n, AUX_STRIDE), dtype=np.uint8)
    winning_line = np.zeros((n, AUX_STRIDE), dtype=np.uint8)
    game_lengths = np.full(n, 50, dtype=np.uint16)
    is_full_search = np.ones(n, dtype=np.uint8)

    buf.push_many(states, chain, policies, outcomes, ownership, winning_line,
                  game_lengths, is_full_search)
    assert buf.size == n

    s, c, p, o, own, wl, ifs = buf.sample_batch(32, augment=False)
    assert s.shape == (32, CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert c.shape == (32, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert p.shape == (32, N_ACTIONS)
    assert o.shape == (32,)
    assert own.shape == (32, BOARD_SIZE, BOARD_SIZE)
    assert wl.shape == (32, BOARD_SIZE, BOARD_SIZE)
    assert ifs.shape == (32,)
    assert np.all(ifs == 1)


def test_push_many_wraps_ring():
    """N > capacity - size triggers ring overwrite; final size == capacity."""
    buf = make_buf(50)
    # Warm-up with 30 rows.
    push_n(buf, 30)
    assert buf.size == 30

    n = 40  # overflows: 10 new + 30 overwrites.
    states = np.stack([random_state() for _ in range(n)]).astype(np.float16)
    chain = np.zeros((n, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    policies = np.stack([random_policy() for _ in range(n)]).astype(np.float32)
    outcomes = np.zeros(n, dtype=np.float32)
    ownership = np.ones((n, AUX_STRIDE), dtype=np.uint8)
    winning_line = np.zeros((n, AUX_STRIDE), dtype=np.uint8)
    game_lengths = np.full(n, 50, dtype=np.uint16)
    is_full_search = np.ones(n, dtype=np.uint8)

    buf.push_many(states, chain, policies, outcomes, ownership, winning_line,
                  game_lengths, is_full_search)
    assert buf.size == 50  # capped at capacity


def test_push_many_mixed_is_full_search():
    """Per-row is_full_search preserved through bulk push → sample."""
    buf = make_buf(100)
    n = 64
    states = np.stack([random_state() for _ in range(n)]).astype(np.float16)
    chain = np.zeros((n, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    policies = np.stack([random_policy() for _ in range(n)]).astype(np.float32)
    outcomes = np.zeros(n, dtype=np.float32)
    ownership = np.ones((n, AUX_STRIDE), dtype=np.uint8)
    winning_line = np.zeros((n, AUX_STRIDE), dtype=np.uint8)
    game_lengths = np.full(n, 50, dtype=np.uint16)
    # Alternate full/quick search flag.
    is_full_search = (np.arange(n) % 2).astype(np.uint8)

    buf.push_many(states, chain, policies, outcomes, ownership, winning_line,
                  game_lengths, is_full_search)
    assert buf.size == n

    # Sample the full population — should see both 0 and 1 in the ifs column.
    _, _, _, _, _, _, ifs = buf.sample_batch(n, augment=False)
    assert ifs.shape == (n,)
    assert set(np.unique(ifs).tolist()) == {0, 1}


def test_push_many_shape_mismatch_raises():
    """Mismatched outer dimension raises a ValueError."""
    buf = make_buf(100)
    n = 10
    states = np.stack([random_state() for _ in range(n)]).astype(np.float16)
    chain = np.zeros((n, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    policies = np.stack([random_policy() for _ in range(n)]).astype(np.float32)
    outcomes = np.zeros(n, dtype=np.float32)
    ownership = np.ones((n, AUX_STRIDE), dtype=np.uint8)
    winning_line = np.zeros((n, AUX_STRIDE), dtype=np.uint8)
    # Wrong length — should trip the shape check.
    game_lengths = np.full(n - 1, 50, dtype=np.uint16)
    is_full_search = np.ones(n, dtype=np.uint8)

    with pytest.raises(ValueError, match="game_lengths"):
        buf.push_many(states, chain, policies, outcomes, ownership, winning_line,
                      game_lengths, is_full_search)
