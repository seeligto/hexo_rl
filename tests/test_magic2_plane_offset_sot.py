"""MAGIC-2 — plane-offset single-source-of-truth pin (Python boundary).

Audit finding MAGIC-2 (`codebase_consistency_audit_2026-06-02.md`): the
my/opp/moves-remaining/ply-parity source-plane offsets (0/8/16/17) lived as bare
literals in three independent representations (Rust match arms, registry
kept_plane_indices, Python tensor subscripts). The v6_live2 H-PLANE arc exists
to prevent exactly this pretrain<->selfplay plane drift.

The Rust side is pinned by
`engine/src/board/state/encode.rs::channel_select_matches_registry_kept_set_v6_live2`
(encoder writes <-> registry kept-set, kept == named core.rs consts). This file
pins the *Python* boundary: the registry kept-set the corpus tensor build and the
resolvers slot helpers depend on equals the four named source-plane offsets.
A registry edit or a resolvers-const drift trips these (`injected drift -> RED`).
"""
from hexo_rl.encoding import lookup
from hexo_rl.encoding.resolvers import (
    _CUR_STONE_SRC_PLANE,
    _OPP_STONE_SRC_PLANE,
    _MOVES_REMAINING_SRC_PLANE,
    _PLY_PARITY_SRC_PLANE,
    _TURN_PHASE_SRC_PLANES,
)


def test_v6_live2_kept_set_is_the_named_source_plane_offsets():
    """v6_live2 keeps exactly [my, opp, moves_remaining, ply_parity] in that
    order — pinned to the named resolvers constants, not bare literals."""
    spec = lookup("v6_live2")
    assert tuple(spec.kept_plane_indices) == (
        _CUR_STONE_SRC_PLANE,
        _OPP_STONE_SRC_PLANE,
        _MOVES_REMAINING_SRC_PLANE,
        _PLY_PARITY_SRC_PLANE,
    ), (
        "v6_live2 kept_plane_indices drifted from the named source-plane offsets "
        f"(my/opp/mr/ply); registry={tuple(spec.kept_plane_indices)}"
    )


def test_turn_phase_src_planes_derived_from_named_singletons():
    """The turn-phase set is one SoT: built from the named singletons, not a
    second bare {16, 17} literal."""
    assert _TURN_PHASE_SRC_PLANES == frozenset(
        {_MOVES_REMAINING_SRC_PLANE, _PLY_PARITY_SRC_PLANE}
    )
    # The four offsets are distinct (no aliasing collision in the kept set).
    offsets = {
        _CUR_STONE_SRC_PLANE,
        _OPP_STONE_SRC_PLANE,
        _MOVES_REMAINING_SRC_PLANE,
        _PLY_PARITY_SRC_PLANE,
    }
    assert len(offsets) == 4


def test_game_state_imports_named_offsets_not_bare_literals():
    """The corpus tensor build addresses planes by the named offsets imported
    from resolvers (the absolute source-plane SoT)."""
    from hexo_rl.env import game_state
    assert game_state.MY_STONE_PLANE == _CUR_STONE_SRC_PLANE
    assert game_state.MOVES_REMAINING_PLANE == _MOVES_REMAINING_SRC_PLANE
    assert game_state.PLY_PARITY_PLANE == _PLY_PARITY_SRC_PLANE
