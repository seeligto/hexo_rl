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


# ── D-WATCHGUARD WP3(A) — live-run boundary guard ────────────────────────────
#
# The tests above pin the seam functions in isolation. They do NOT pin the thing
# a live run actually depends on: that the COMPOSED config (base yamls + variant)
# driven through the REAL StepCoordinator resolvers yields an eval board whose
# radius equals the self-play curriculum radius at every stage boundary.
#
# tests/test_radius_curriculum.py:47 re-implements ``_resolve_radius`` for its
# unit tests. A re-implementation cannot catch divergence between itself and the
# method the eval loop calls — so this guard drives the real method objects and
# reads the radius back off a real engine Board.
#
# run2_mw_fresh crosses 200000 (r=4→5) and 400000 (r=5→6). At r=5 the curriculum
# radius coincides with the v6_live2_ls registry default, so a regression that
# re-pinned eval to the registry would be INVISIBLE for the whole 200k–400k
# stage and would silently reappear at 400000. Hence the boundary cases.

_LIVE_VARIANT = "configs/variants/run2_mw_fresh.yaml"
_BASE_CONFIGS = (  # mirrors the path list in scripts/train.py
    "configs/model.yaml", "configs/training.yaml",
    "configs/selfplay.yaml", "configs/game_replay.yaml",
    "configs/monitoring.yaml", "configs/monitors.yaml",
)


class TestComposedConfigDrivesEvalRadius:
    """Real config compose → real StepCoordinator resolvers → real engine Board."""

    @staticmethod
    def _eval_board_radius_at(cfg: dict, step: int) -> int:
        from hexo_rl.training.step_coordinator import StepCoordinator

        class _Shim:
            """Carries only the attributes the two real methods read off ``self``."""

        shim = _Shim()
        shim.full_config = cfg
        # Unbound REAL methods — not a copy of their logic.
        shim._current_radius = StepCoordinator._resolve_radius(shim, step)
        eval_radius = StepCoordinator._resolve_eval_radius(shim)
        return make_eval_board(_ENC, eval_radius).legal_move_radius()

    @pytest.fixture(scope="class")
    def cfg(self) -> dict:
        from hexo_rl.utils.config import load_config

        return load_config(*_BASE_CONFIGS, _LIVE_VARIANT)

    def test_no_fixed_yardstick_override_pins_eval(self, cfg: dict) -> None:
        # If someone sets evaluation.legal_move_radius, eval STOPS tracking the
        # curriculum. That is a legal choice, but it must be a deliberate one.
        eval_cfg = cfg.get("evaluation", cfg.get("eval", {}))
        override = eval_cfg.get("legal_move_radius") if isinstance(eval_cfg, dict) else None
        assert override is None, (
            "run2_mw_fresh pins a fixed eval yardstick; eval no longer tracks the "
            "§174 curriculum. Intentional? Then update this guard."
        )

    def test_eval_radius_equals_curriculum_at_every_scheduled_stage(self, cfg: dict) -> None:
        schedule = cfg["selfplay"]["legal_move_radius_schedule"]
        assert schedule, "run2_mw_fresh lost its legal_move_radius_schedule"
        for entry in schedule:
            step, radius = int(entry["step"]), int(entry["radius"])
            assert self._eval_board_radius_at(cfg, step) == radius, (
                f"eval board radius != curriculum radius at step {step}"
            )

    def test_eval_radius_holds_just_below_each_boundary(self, cfg: dict) -> None:
        schedule = cfg["selfplay"]["legal_move_radius_schedule"]
        for prev, nxt in zip(schedule, schedule[1:]):
            step = int(nxt["step"]) - 1
            assert self._eval_board_radius_at(cfg, step) == int(prev["radius"]), (
                f"eval board radius jumped early at step {step}"
            )

    def test_400k_boundary_advances_eval_off_the_registry_default(self, cfg: dict) -> None:
        # The canary. r=5 == _REGISTRY_DEFAULT, so 200k–400k cannot discriminate
        # "tracks the curriculum" from "pinned to the registry". 400k can.
        assert self._eval_board_radius_at(cfg, 399_999) == _REGISTRY_DEFAULT
        assert self._eval_board_radius_at(cfg, 400_000) == 6
