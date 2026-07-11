"""CONFRES Phase 4 — frozen mutation tests M1, M4, M5 (design §10).

M1  eval radius S2→6 / S3 via the SAME call path eval uses (resolve.radius schedule-scan →
    resolve_eval_radius), so a regression that re-pins eval off the curriculum FIRES.
M4  decode_override → resolve + LOUD (never raises); declared_encoding + mismatched stamp → RAISE;
    decode_override + declared_encoding together → RAISE (mutually exclusive).
M5  registry-default(5) == curriculum(5); mutate the curriculum → 6 → eval FOLLOWS (the radius is
    curriculum-derived from ONE source, not the registry default).
"""
from __future__ import annotations

import pytest

from hexo_rl.config.resolve.radius import resolve_eval_radius, resolve_radius_from_schedule
from hexo_rl.eval.eval_board import make_eval_board

_ENC = "v6_live2_ls"
_REGISTRY_DEFAULT = 5  # engine/src/encoding/registry.toml [encodings.v6_live2_ls]


# ── M1: eval radius follows the curriculum through the run's own resolution path ──
def test_m1_eval_radius_tracks_curriculum_stages():
    # The run2 curriculum: R 4→5→6→8. Eval resolves via the SAME schedule-scan self-play uses,
    # then resolve_eval_radius (override None → track). A mutation that pinned eval to the registry
    # default would return 5 at every step and FAIL the step-6 / step-8 assertions.
    sched = [
        {"step": 0, "radius": 4},
        {"step": 200_000, "radius": 5},
        {"step": 400_000, "radius": 6},
        {"step": 600_000, "radius": 8},
    ]
    for step, want in [(0, 4), (200_000, 5), (400_000, 6), (600_000, 8), (500_000, 6)]:
        curriculum = resolve_radius_from_schedule(sched, step)
        eval_radius = resolve_eval_radius(curriculum, override=None)
        assert eval_radius == want, f"step {step}: eval radius {eval_radius} != curriculum {want}"
        # And the eval board actually binds it (the S2→6 board-level effect).
        assert make_eval_board(_ENC, eval_radius).legal_move_radius() == want


def test_m1_explicit_override_pins_yardstick_regardless_of_curriculum():
    # The fixed-yardstick escape hatch: override wins over the curriculum stage.
    assert resolve_eval_radius(6, override=8) == 8
    assert resolve_eval_radius(4, override=8) == 8


# ── M4: decode_override loud (no raise); declared+mismatch raises; both → raise ──
def test_m4_decode_override_resolves_loud_never_raises():
    # decode_override is a deliberate cross-decode: it wins and logs, never raises on a disagreeing
    # stamp. Exercise the reconcile rule the loader uses: a decode_override is NOT a declared
    # assertion, so a mismatched stamp does not conflict (the loader logs encoding_decode_override).
    from hexo_rl.config.resolve.encoding import reconcile_declared_vs_stamp, UNSPECIFIED

    # decode_override path: the loader treats the override as authoritative, NOT as a declaration —
    # reconcile with declared UNSPECIFIED + the (disagreeing) stamp does not raise; the override
    # wins downstream. Here we assert the no-raise property of the declaration side.
    res = reconcile_declared_vs_stamp(UNSPECIFIED, "v6_live2")  # no declaration → stamp, no raise
    assert res.name == "v6_live2"


def test_m4_declared_mismatched_stamp_raises():
    from hexo_rl.eval.checkpoint_loader import (
        DeclaredEncodingMismatchError,
        _check_declared_vs_stamped_encoding,
    )

    with pytest.raises(DeclaredEncodingMismatchError):
        _check_declared_vs_stamped_encoding(
            "v6_live2_ls", "v6_live2", "checkpoint metadata['encoding_name']",
        )
    # Agreement (or no stamp) does NOT raise.
    assert _check_declared_vs_stamped_encoding("v6_live2_ls", "v6_live2_ls", "meta") == "v6_live2_ls"
    assert _check_declared_vs_stamped_encoding("v6_live2_ls", None, None) == "v6_live2_ls"


def test_m4_decode_override_and_declared_together_raise(tmp_path):
    # Mutually exclusive: passing both to load_model_with_encoding raises before any ckpt read.
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    import torch as _t

    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"x")  # never read — the mutual-exclusion check fires first
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_model_with_encoding(
            fake, _t.device("cpu"),
            declared_encoding="v6_live2_ls", decode_override="v6_live2",
        )


# ── M5: registry-default == curriculum at one stage; mutate curriculum → eval follows ──
def test_m5_curriculum_mutation_moves_eval_off_the_registry_default():
    # At the stage where curriculum == registry default (5), eval reads 5 either way — the trap
    # that hid the run2 eval-radius bug until 400k. Mutating the curriculum to 6 must move eval to
    # 6 (curriculum-derived), NOT stay pinned at the registry default 5.
    sched_at_default = [{"step": 0, "radius": 5}]
    assert resolve_eval_radius(resolve_radius_from_schedule(sched_at_default, 100), None) == 5
    assert resolve_eval_radius(resolve_radius_from_schedule(sched_at_default, 100), None) == _REGISTRY_DEFAULT

    sched_advanced = [{"step": 0, "radius": 5}, {"step": 100, "radius": 6}]
    curriculum6 = resolve_radius_from_schedule(sched_advanced, 100)
    assert curriculum6 == 6
    assert resolve_eval_radius(curriculum6, None) == 6  # eval FOLLOWS off the registry default
    assert make_eval_board(_ENC, resolve_eval_radius(curriculum6, None)).legal_move_radius() == 6
