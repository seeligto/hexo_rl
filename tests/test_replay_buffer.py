"""
Phase 1 tests for ReplayBuffer.

Run with: .venv/bin/pytest tests/test_replay_buffer.py -v
"""
import numpy as np
import pytest
from python.training.replay_buffer import ReplayBuffer


def make_buf(capacity: int = 100, channels: int = 18, size: int = 9) -> ReplayBuffer:
    return ReplayBuffer(capacity=capacity, board_channels=channels, board_size=size)


def random_entry(channels: int = 18, size: int = 9) -> tuple:
    n_actions = size * size + 1
    state   = np.random.randn(channels, size, size).astype(np.float16)
    policy  = np.random.dirichlet(np.ones(n_actions)).astype(np.float32)
    outcome = float(np.random.choice([-1.0, 0.0, 1.0]))
    return state, policy, outcome


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_size_is_zero():
    buf = make_buf()
    assert buf.size == 0
    assert len(buf) == 0


def test_pre_allocated_shapes():
    buf = make_buf(capacity=50, channels=18, size=9)
    assert buf.states.shape   == (50, 18, 9, 9)
    assert buf.policies.shape == (50, 9*9+1)
    assert buf.outcomes.shape == (50,)


def test_pre_allocated_dtypes():
    buf = make_buf()
    assert buf.states.dtype   == np.float16
    assert buf.policies.dtype == np.float32
    assert buf.outcomes.dtype == np.float32


# ── Push / size ───────────────────────────────────────────────────────────────

def test_push_increments_size():
    buf = make_buf(capacity=10)
    for i in range(5):
        buf.push(*random_entry())
        assert buf.size == i + 1


def test_push_caps_at_capacity():
    buf = make_buf(capacity=5)
    for _ in range(10):
        buf.push(*random_entry())
    assert buf.size == 5


def test_push_stores_correct_values():
    buf = make_buf(capacity=10)
    state, policy, outcome = random_entry()
    buf.push(state, policy, outcome)
    np.testing.assert_array_equal(buf.states[0], state)
    np.testing.assert_array_equal(buf.policies[0], policy)
    assert buf.outcomes[0] == outcome


# ── Wrap-around ───────────────────────────────────────────────────────────────

def test_push_wraps_ptr_correctly():
    buf = make_buf(capacity=3)
    entries = [random_entry() for _ in range(5)]
    for e in entries:
        buf.push(*e)
    # After 5 pushes into capacity-3: ptr = 5 % 3 = 2, size = 3
    assert buf._ptr  == 2
    assert buf._size == 3
    # Slot 0 should contain entry[3], slot 1 = entry[4], slot 2 = entry[2]
    np.testing.assert_array_equal(buf.states[0], entries[3][0])
    np.testing.assert_array_equal(buf.states[1], entries[4][0])
    np.testing.assert_array_equal(buf.states[2], entries[2][0])


def test_push_game_no_wrap():
    buf = make_buf(capacity=20, channels=2, size=3)
    t = 5
    states   = np.random.randn(t, 2, 3, 3).astype(np.float16)
    policies = np.random.rand(t, 10).astype(np.float32)
    outcomes = np.array([1.0, -1.0, 0.0, 1.0, -1.0], dtype=np.float32)
    buf.push_game(states, policies, outcomes)
    assert buf.size == t
    np.testing.assert_array_equal(buf.states[:t],   states)
    np.testing.assert_array_equal(buf.policies[:t], policies)
    np.testing.assert_array_equal(buf.outcomes[:t], outcomes)


def test_push_game_with_wrap():
    cap = 7
    buf = make_buf(capacity=cap, channels=2, size=3)
    # Fill 5 entries
    states1   = np.ones((5, 2, 3, 3), dtype=np.float16) * 1.0
    policies1 = np.ones((5, 10),      dtype=np.float32) * 0.1
    outcomes1 = np.zeros(5,           dtype=np.float32)
    buf.push_game(states1, policies1, outcomes1)
    assert buf._ptr == 5

    # Push 4 more — wraps: 2 in [5,6], 2 in [0,1]
    states2   = np.ones((4, 2, 3, 3), dtype=np.float16) * 2.0
    policies2 = np.ones((4, 10),      dtype=np.float32) * 0.2
    outcomes2 = np.ones(4,            dtype=np.float32)
    buf.push_game(states2, policies2, outcomes2)

    assert buf._ptr  == 2          # (5+4) % 7
    assert buf._size == cap        # capped at capacity

    # Slots 5,6,0,1 should contain states2
    np.testing.assert_array_equal(buf.states[5], states2[0])
    np.testing.assert_array_equal(buf.states[6], states2[1])
    np.testing.assert_array_equal(buf.states[0], states2[2])
    np.testing.assert_array_equal(buf.states[1], states2[3])


# ── Sample ────────────────────────────────────────────────────────────────────

def test_sample_returns_correct_shapes():
    buf = make_buf(capacity=50, channels=18, size=9)
    for _ in range(20):
        buf.push(*random_entry())
    states, policies, outcomes = buf.sample(8)
    assert states.shape   == (8, 18, 9, 9)
    assert policies.shape == (8, 9*9+1)
    assert outcomes.shape == (8,)


def test_sample_returns_correct_dtypes():
    buf = make_buf(capacity=50)
    for _ in range(10):
        buf.push(*random_entry())
    states, policies, outcomes = buf.sample(4)
    assert states.dtype   == np.float16
    assert policies.dtype == np.float32
    assert outcomes.dtype == np.float32


def test_sample_raises_on_empty():
    buf = make_buf()
    with pytest.raises(ValueError, match="empty"):
        buf.sample(1)


def test_sample_content_is_valid():
    """Sampled entries should all come from what was pushed."""
    buf = make_buf(capacity=10)
    pushed_outcomes = set()
    for i in range(5):
        outcome = float(i)  # use index as unique outcome
        buf.push(*random_entry()[:2], outcome)
        pushed_outcomes.add(outcome)
    _, _, outcomes = buf.sample(100)
    for o in outcomes:
        assert o in pushed_outcomes, f"sampled outcome {o} not in pushed set"
