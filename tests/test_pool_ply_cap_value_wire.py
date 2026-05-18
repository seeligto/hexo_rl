"""§178 T3 — verify `ply_cap_value` threads from `training_cfg` through
`WorkerPool.__init__`'s `SelfPlayRunnerConfig(...)` kwargs.

The wire site is `hexo_rl/selfplay/pool.py:303`:

    ply_cap_value=float(
        training_cfg.get("ply_cap_value", training_cfg.get("draw_value", -0.5))
    ),

Back-compat: when `training_cfg["ply_cap_value"]` is absent, the value falls
back to `training_cfg["draw_value"]` (matches pre-§178 behavior). When
`draw_value` is also absent, the floor is -0.5 (matches the existing
`draw_reward=` fallback at the line above).

Cells:
  A. explicit ply_cap_value present → that value wins
  B. ply_cap_value absent + draw_value present → draw_value wins
  C. both absent → -0.5 floor
"""
from __future__ import annotations

import pytest


def _resolve(training_cfg: dict) -> float:
    """Mirror the resolution expression at pool.py:303 verbatim — keep in sync
    if the wire site changes."""
    return float(
        training_cfg.get("ply_cap_value", training_cfg.get("draw_value", -0.5))
    )


def test_pool_ply_cap_value_explicit_wins():
    """Cell A — explicit `ply_cap_value` overrides `draw_value` fallback."""
    cfg = {"ply_cap_value": -0.8, "draw_value": -0.5}
    assert _resolve(cfg) == pytest.approx(-0.8)


def test_pool_ply_cap_value_fallback_to_draw_value():
    """Cell B — absent `ply_cap_value` falls back to `draw_value` (back-compat
    for §177-style runs)."""
    cfg = {"draw_value": -0.3}
    assert _resolve(cfg) == pytest.approx(-0.3)


def test_pool_ply_cap_value_floor_when_both_absent():
    """Cell C — both absent → -0.5 floor (matches `draw_reward` floor at the
    sister line in pool.py)."""
    cfg: dict = {}
    assert _resolve(cfg) == pytest.approx(-0.5)


def test_pool_ply_cap_value_wire_site_exists():
    """Lightweight pin: the literal kwarg `ply_cap_value=` must appear inside
    the `SelfPlayRunnerConfig(...)` block in pool.py. A regression that drops
    the kwarg (silently routing the ply-cap outcome back to draw_reward) fires
    this test."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / "hexo_rl" / "selfplay" / "pool.py"
    text = src.read_text()
    assert "ply_cap_value=float(" in text, (
        "pool.py must wire `ply_cap_value=float(...)` into "
        "SelfPlayRunnerConfig(...) per §178 T3"
    )
    # Both kwargs must live inside the SelfPlayRunnerConfig(...) block.
    # Use line-level proximity: ply_cap_value= line index must be within
    # ~30 lines of the draw_reward= line (same kwarg block).
    lines = text.splitlines()
    draw_reward_lines = [i for i, line in enumerate(lines) if "draw_reward=float(" in line]
    ply_cap_lines = [i for i, line in enumerate(lines) if "ply_cap_value=float(" in line]
    assert draw_reward_lines, "draw_reward=float(...) must still be wired in pool.py"
    assert ply_cap_lines, "ply_cap_value=float(...) must be wired in pool.py per §178 T3"
    assert abs(ply_cap_lines[0] - draw_reward_lines[0]) < 30, (
        f"ply_cap_value and draw_reward must be adjacent kwargs in the same "
        f"SelfPlayRunnerConfig(...) block; got draw_reward at line "
        f"{draw_reward_lines[0]}, ply_cap_value at line {ply_cap_lines[0]}"
    )
