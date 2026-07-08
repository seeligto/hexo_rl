"""D-SHRIMP S4b fix — eval boards follow the run's legal_move_radius curriculum.

Self-play threads the §174 ``legal_move_radius_schedule`` into every game, but the
in-loop eval + promotion gate historically bound the registry's FIXED radius
(``v6_live2_ls`` = 5) and could not follow the curriculum (4→5→6→8). These tests pin:

1. ``make_eval_board`` applies the curriculum-current radius via the engine's
   ``override_legal_move_radius`` bypass (the registry guard forbids the plain setter).
2. ``resolve_eval_radius`` keeps eval CONNECTED to self-play by the SAME source (the
   resolved curriculum radius) by default, with an explicit override for a fixed yardstick.
"""
from __future__ import annotations

import pytest

from hexo_rl.eval.eval_board import make_eval_board, resolve_eval_radius

_ENC = "v6_live2_ls"
_REGISTRY_DEFAULT = 5  # engine/src/encoding/registry.toml [encodings.v6_live2_ls]


class TestMakeEvalBoard:
    def test_none_keeps_registry_default(self) -> None:
        assert make_eval_board(_ENC).legal_move_radius() == _REGISTRY_DEFAULT
        assert make_eval_board(_ENC, None).legal_move_radius() == _REGISTRY_DEFAULT

    @pytest.mark.parametrize("radius", [4, 5, 6, 8])
    def test_override_applies_curriculum_radius(self, radius: int) -> None:
        # Overriding is legal AFTER with_encoding_name (bypass), unlike the guarded setter.
        assert make_eval_board(_ENC, radius).legal_move_radius() == radius

    def test_override_changes_legal_envelope(self) -> None:
        # Behavioural check: a wider radius exposes strictly more legal moves once a
        # stone is on the board (the legal set is the per-stone radius dilation).
        b4 = make_eval_board(_ENC, 4)
        b8 = make_eval_board(_ENC, 8)
        b4.apply_move(0, 0)
        b8.apply_move(0, 0)
        assert b8.legal_move_count() > b4.legal_move_count()


class TestResolveEvalRadius:
    def test_default_tracks_curriculum_single_source(self) -> None:
        # No override → eval radius IS the self-play curriculum radius (connected).
        assert resolve_eval_radius(4, None) == 4
        assert resolve_eval_radius(8, None) == 8

    def test_none_curriculum_stays_none(self) -> None:
        # Non-curriculum runs (no schedule) → None → leaves the registry default.
        assert resolve_eval_radius(None, None) is None

    def test_explicit_override_pins_fixed_yardstick(self) -> None:
        # Configurable escape hatch: pin a fixed eval radius regardless of curriculum.
        assert resolve_eval_radius(4, 5) == 5
        assert resolve_eval_radius(8, 5) == 5

    def test_override_zero_is_honoured_not_falsy_dropped(self) -> None:
        # 0 is a real radius value, not "unset" — must not collapse to the curriculum.
        assert resolve_eval_radius(4, 0) == 0


class TestEvaluatorThreading:
    """The Evaluator ctor carries the radius so every opponent board it builds inherits it."""

    def _evaluator(self, radius):
        import torch

        from hexo_rl.eval.evaluator import Evaluator

        cfg = {"encoding": _ENC, "evaluation": {}}
        return Evaluator(object(), torch.device("cpu"), cfg, legal_move_radius=radius)

    def test_stores_radius(self) -> None:
        assert self._evaluator(6).legal_move_radius == 6

    def test_defaults_to_none(self) -> None:
        import torch

        from hexo_rl.eval.evaluator import Evaluator

        ev = Evaluator(object(), torch.device("cpu"), {"encoding": _ENC})
        assert ev.legal_move_radius is None
