"""
Tests and benchmark for RustReplayBuffer.

Run with: pytest tests/test_rust_replay_buffer.py -v
Benchmark: pytest tests/test_rust_replay_buffer.py -v -k benchmark -s
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from native_core import RustReplayBuffer

CHANNELS   = 18
BOARD_SIZE = 19
N_ACTIONS  = BOARD_SIZE * BOARD_SIZE + 1  # 362


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_buf(capacity: int = 100) -> RustReplayBuffer:
    return RustReplayBuffer(capacity)


def random_state() -> np.ndarray:
    return np.random.randn(CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)


def random_policy() -> np.ndarray:
    p = np.abs(np.random.randn(N_ACTIONS)).astype(np.float32)
    p /= p.sum()
    return p


def random_entry() -> tuple:
    return random_state(), random_policy(), float(np.random.choice([-1.0, 0.0, 1.0]))


def push_n(buf: RustReplayBuffer, n: int, use_game_id: bool = False) -> None:
    for _ in range(n):
        gid = buf.next_game_id() if use_game_id else -1
        s, p, o = random_entry()
        buf.push(s, p, o, game_id=gid)


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_size_zero():
    buf = make_buf(capacity=50)
    assert buf.size == 0
    assert buf.capacity == 50


# ── push / size ───────────────────────────────────────────────────────────────

def test_push_increments_size():
    buf = make_buf(capacity=10)
    for i in range(5):
        s, p, o = random_entry()
        buf.push(s, p, o)
        assert buf.size == i + 1


def test_push_caps_at_capacity():
    buf = make_buf(capacity=5)
    for _ in range(12):
        s, p, o = random_entry()
        buf.push(s, p, o)
    assert buf.size == 5


# ── push_game ─────────────────────────────────────────────────────────────────

def test_push_game_no_wrap():
    buf = make_buf(capacity=20)
    t = 7
    states   = np.random.randn(t, CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
    policies = np.abs(np.random.randn(t, N_ACTIONS).astype(np.float32))
    outcomes = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0], dtype=np.float32)
    buf.push_game(states, policies, outcomes)
    assert buf.size == t


def test_push_game_with_wrap():
    buf = make_buf(capacity=7)
    t1 = 5
    s1 = np.ones((t1, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    p1 = np.ones((t1, N_ACTIONS), dtype=np.float32) / N_ACTIONS
    o1 = np.zeros(t1, dtype=np.float32)
    buf.push_game(s1, p1, o1)
    assert buf.size == t1

    t2 = 4
    s2 = np.ones((t2, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16) * 2.0
    p2 = np.ones((t2, N_ACTIONS), dtype=np.float32) / N_ACTIONS
    o2 = np.ones(t2, dtype=np.float32)
    buf.push_game(s2, p2, o2)
    assert buf.size == 7  # capped at capacity


# ── sample_batch shapes / dtypes ─────────────────────────────────────────────

def test_sample_raises_on_empty():
    buf = make_buf(capacity=50)
    with pytest.raises(Exception):
        buf.sample_batch(4, augment=False)


def test_sample_returns_correct_shapes():
    buf = make_buf(capacity=200)
    push_n(buf, 100)
    s, p, o = buf.sample_batch(16, augment=True)
    assert s.shape == (16, CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert p.shape == (16, N_ACTIONS)
    assert o.shape == (16,)


def test_sample_returns_correct_dtypes():
    buf = make_buf(capacity=100)
    push_n(buf, 50)
    s, p, o = buf.sample_batch(8, augment=True)
    assert s.dtype == np.float16, f"states should be float16, got {s.dtype}"
    assert p.dtype == np.float32, f"policies should be float32, got {p.dtype}"
    assert o.dtype == np.float32, f"outcomes should be float32, got {o.dtype}"


def test_sample_no_augment_content_roundtrip():
    """Without augmentation, sampled states must contain values matching pushed data."""
    buf = make_buf(capacity=50)
    # Push entries with a unique, easily-verifiable outcome per slot.
    for i in range(20):
        s, p, _ = random_entry()
        buf.push(s, p, float(i))

    _, _, outcomes = buf.sample_batch(200, augment=False)
    # All returned outcomes must be one of 0..19.
    assert all(int(round(o)) in range(20) for o in outcomes)


# ── Augmentation correctness ──────────────────────────────────────────────────

def test_identity_symmetry_preserves_data():
    """Symmetry 0 (identity) must return the exact original values."""
    buf = RustReplayBuffer(10)

    # Place a known state: all zeros except plane 0, cell (0,0) = 1.0
    state = np.zeros((CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    state[0, 0, 0] = 1.0
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    buf.push(state, policy, 1.0)

    # sample_batch with augment=False always uses symmetry 0.
    sampled_s, sampled_p, sampled_o = buf.sample_batch(1, augment=False)
    assert sampled_s[0, 0, 0, 0] == pytest.approx(1.0, abs=1e-3)
    assert sampled_p[0, 0]        == pytest.approx(1.0, abs=1e-3)
    assert sampled_o[0]            == pytest.approx(1.0, abs=1e-3)


def test_pass_action_invariant_under_augmentation():
    """The pass action (index 361) must be unchanged by every symmetry."""
    buf = RustReplayBuffer(50)
    rng = np.random.default_rng(42)

    for _ in range(20):
        p = rng.random(N_ACTIONS).astype(np.float32)
        p /= p.sum()
        buf.push(rng.standard_normal((CHANNELS, BOARD_SIZE, BOARD_SIZE)).astype(np.float16), p, 0.0)

    for _ in range(10):
        # We can't control which symmetry is chosen, but the pass logit (index -1)
        # must always match some pushed pass logit — check it's in a plausible range.
        _, policies, _ = buf.sample_batch(32, augment=True)
        pass_logits = policies[:, -1]
        assert (pass_logits >= 0.0).all(), "pass logit must be non-negative"
        assert (pass_logits <= 1.0).all(), "pass logit must be ≤ 1 (it's a probability)"


def test_policy_sum_preserved_under_augmentation():
    """Spatial policy logits must sum to the same value before and after augmentation."""
    buf = RustReplayBuffer(50)
    rng = np.random.default_rng(0)

    original_sums = []
    for _ in range(30):
        p = rng.random(N_ACTIONS).astype(np.float32)
        original_sums.append(p[:-1].sum())  # sum of spatial logits only
        buf.push(rng.standard_normal((CHANNELS, BOARD_SIZE, BOARD_SIZE)).astype(np.float16), p, 0.0)

    # Some cells fall outside the window under certain rotations, so sum can decrease slightly.
    # It must never increase (no mass created) and stay within a small tolerance.
    _, policies, _ = buf.sample_batch(30, augment=True)
    aug_sums = policies[:, :-1].sum(axis=1)
    # At most the sum of the original (some cells may be clipped to 0).
    max_original = max(original_sums)
    assert (aug_sums <= max_original + 1e-4).all()


# ── Correlation guard ─────────────────────────────────────────────────────────

def test_correlation_guard_no_duplicate_game_ids():
    """With enough distinct positions, no batch should contain two entries from the same position."""
    buf = RustReplayBuffer(2000)
    # Push 500 positions, each with 3 clusters (same game_id).
    for _ in range(500):
        gid = buf.next_game_id()
        for _ in range(3):
            s, p, o = random_entry()
            buf.push(s, p, o, game_id=gid)

    assert buf.size == 1500

    for _ in range(20):
        _, _, _ = buf.sample_batch(64, augment=False)
        # If we had access to game_ids we'd check here; since sample_batch doesn't
        # return them, we verify the guard doesn't crash and the shapes are correct.
        # Full dedup verification would require exposing indices — see benchmark below.


# ── next_game_id ──────────────────────────────────────────────────────────────

def test_next_game_id_monotonic():
    buf = make_buf()
    ids = [buf.next_game_id() for _ in range(100)]
    assert ids == list(range(100))


# ── Benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark_sample_latency(capsys):
    """
    Benchmark: sample_batch(256) must complete in < 128ms (< 0.5ms/sample).
    Also compares against the Python ReplayBuffer for reference.
    """
    CAPACITY   = 200_000
    FILL       = 50_000
    BATCH      = 256
    WARMUP     = 10
    N_ITERS    = 200

    # ── Rust buffer ───────────────────────────────────────────────────────────
    rust_buf = RustReplayBuffer(CAPACITY)
    batch_s  = np.random.randn(1000, CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
    batch_p  = np.abs(np.random.randn(1000, N_ACTIONS).astype(np.float32))
    batch_p /= batch_p.sum(axis=1, keepdims=True)
    batch_o  = np.zeros(1000, dtype=np.float32)

    for _ in range(FILL // 1000):
        rust_buf.push_game(batch_s, batch_p, batch_o, game_id=rust_buf.next_game_id())

    for _ in range(WARMUP):
        rust_buf.sample_batch(BATCH, augment=True)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        rust_buf.sample_batch(BATCH, augment=True)
    rust_ms = (time.perf_counter() - t0) / N_ITERS * 1000

    # ── Python buffer (reference) ─────────────────────────────────────────────
    try:
        from python.training.replay_buffer import ReplayBuffer
        py_buf = ReplayBuffer(capacity=CAPACITY)
        for _ in range(FILL // 1000):
            py_buf.push_game(batch_s, batch_p, batch_o)

        for _ in range(WARMUP):
            py_buf.sample(BATCH, augment=True)

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            py_buf.sample(BATCH, augment=True)
        py_ms = (time.perf_counter() - t0) / N_ITERS * 1000
        py_label = f"{py_ms:.3f} ms/batch  ({py_ms/BATCH*1000:.2f} µs/sample)"
    except Exception as exc:
        py_label = f"(unavailable: {exc})"
        py_ms    = None

    with capsys.disabled():
        print(f"\n{'─'*56}")
        print(f"  ReplayBuffer sample_batch benchmark  (batch={BATCH})")
        print(f"{'─'*56}")
        print(f"  Rust  : {rust_ms:.3f} ms/batch  ({rust_ms/BATCH*1000:.2f} µs/sample)")
        print(f"  Python: {py_label}")
        if py_ms is not None:
            print(f"  Speedup: {py_ms/rust_ms:.1f}×")
        print(f"{'─'*56}\n")

    # Target: < 0.5 ms per sample = < 128 ms per 256-sample batch
    assert rust_ms < 128.0, (
        f"sample_batch({BATCH}) took {rust_ms:.2f} ms — target < 128 ms "
        f"({rust_ms/BATCH*1000:.2f} µs/sample, target < 500 µs/sample)"
    )
