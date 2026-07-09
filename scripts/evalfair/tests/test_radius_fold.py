"""T-RADIUS: radius_from_checkpoint folds correctly vs run2 schedule."""
from __future__ import annotations

import pytest


# run2 schedule: {0:4, 200000:5, 400000:6, 600000:8}
@pytest.mark.parametrize("step,expected_radius", [
    (50000,   4),
    (199999,  4),
    (200000,  5),
    (400000,  6),
    (600000,  8),
    (0,       4),
    (399999,  5),
])
def test_radius_fold(step, expected_radius):
    """radius_from_checkpoint resolves correctly against run2 schedule."""
    from scripts.evalfair.core import radius_from_checkpoint

    # Construct a minimal fake checkpoint with the run2 schedule
    schedule = [
        {"step": 0,      "radius": 4},
        {"step": 200000, "radius": 5},
        {"step": 400000, "radius": 6},
        {"step": 600000, "radius": 8},
    ]
    ck = {
        "step": step,
        "config": {
            "selfplay": {
                "legal_move_radius_schedule": schedule,
            }
        }
    }
    assert radius_from_checkpoint(ck) == expected_radius, (
        f"step={step}: expected {expected_radius}, got {radius_from_checkpoint(ck)}"
    )
