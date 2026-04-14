"""Smoke test: game-length weight schedule is wired end-to-end via PyO3."""

import numpy as np
from engine import ReplayBuffer


def test_weight_schedule_changes_sampling():
    buf = ReplayBuffer(capacity=200)
    buf.set_weight_schedule([10, 25], [0.15, 0.50], 1.0)

    state = np.zeros((24, 19, 19), dtype=np.float16)
    policy = np.zeros(362, dtype=np.float32)
    own = np.ones(361, dtype=np.uint8)
    wl  = np.zeros(361, dtype=np.uint8)

    # Push 100 short-game positions (game_length=5, weight=0.15)
    for _ in range(100):
        buf.push(state, policy, 1.0, own, wl, game_length=5)
    # Push 100 long-game positions (game_length=60, weight=1.0)
    for _ in range(100):
        buf.push(state, policy, -1.0, own, wl, game_length=60)

    # Sample 2000 times and count by outcome
    short_count = sum(
        1 for _ in range(2000)
        for (_, _, o, _, _) in [buf.sample_batch(1, False)]
        if float(o[0]) > 0
    )
    # Short-game positions (weight 0.15) should appear much less than long (weight 1.0)
    assert short_count < 600, f"short sampled {short_count}/2000, expected <600 (~13%)"


def test_push_game_length_assigns_different_weights():
    buf = ReplayBuffer(capacity=10)
    buf.set_weight_schedule([10, 25], [0.15, 0.50], 1.0)

    state = np.zeros((24, 19, 19), dtype=np.float16)
    policy = np.zeros(362, dtype=np.float32)
    own = np.ones(361, dtype=np.uint8)
    wl  = np.zeros(361, dtype=np.uint8)

    buf.push(state, policy, 1.0, own, wl, game_length=5)
    buf.push(state, policy, 1.0, own, wl, game_length=60)

    _, _, hist = buf.get_buffer_stats()
    # game_length=5 → weight 0.15 → bucket 0; game_length=60 → weight 1.0 → bucket 2
    assert hist[0] == 1, f"expected 1 in low bucket, got {hist[0]}"
    assert hist[2] == 1, f"expected 1 in high bucket, got {hist[2]}"
