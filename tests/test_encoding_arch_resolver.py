"""§P5-CT de-hardcoding sweep — `resolve_arch` resolver + parametrized INV.

ONE registry-derived resolver maps an encoding NAME → arch facts, so consumer
sites stop hardcoding plane counts / kept-index lists / opponent slots. This
test pins `resolve_arch` output against the registry SoT for EVERY registered
encoding (count-agnostic — iterates the registry, never a literal 9/10).

Companion `test_encoding_no_positional_plane_slice.py` adds the second INV
facet (no live-path bare `[:, <int>]` plane index) — the L65 lesson.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup, resolve_arch
from hexo_rl.encoding.registry import _load as _load_registry

# Source-plane semantics fixed by the v6 wire format (game_state.to_tensor):
#   0      → current-player stone, t0
#   8      → opponent stone, t0
#   1,2,3  → current-player history t-1..t-3
#   9,10,11→ opponent history t-1..t-3
#   16,17  → turn-phase scalars (moves_remaining==2, ply parity)
_CUR_SRC = 0
_OPP_SRC = 8
_HISTORY_SRC = frozenset({1, 2, 3, 9, 10, 11})
_TURN_PHASE_SRC = frozenset({16, 17})

# resolve_arch is a CNN dense-plane arch resolver — graph encodings carry no
# planes (kept_plane_indices == []), so restrict the SoT-parity INV to grid.
_ALL_ENCODINGS = sorted(n for n in _load_registry() if lookup(n).representation == "grid")


@pytest.mark.parametrize("name", _ALL_ENCODINGS)
def test_resolve_arch_matches_registry(name):
    """resolve_arch output equals the registry spec on every field, for every
    registered encoding. This is the SoT-parity INV pin."""
    spec = lookup(name)
    kept = list(spec.kept_plane_indices)
    arch = resolve_arch(name)

    assert arch.name == spec.name
    assert arch.in_channels == spec.n_planes
    assert arch.kept_indices == tuple(kept)
    assert arch.k_max == spec.k_max
    assert arch.policy_logit_count == spec.policy_logit_count
    assert arch.cur_stone_slot == kept.index(_CUR_SRC)
    assert arch.opp_stone_slot == kept.index(_OPP_SRC)

    exp_history = tuple(i for i, src in enumerate(kept) if src in _HISTORY_SRC)
    exp_turn = tuple(i for i, src in enumerate(kept) if src in _TURN_PHASE_SRC)
    assert arch.history_planes == exp_history
    assert arch.turn_phase_planes == exp_turn


@pytest.mark.parametrize("name", _ALL_ENCODINGS)
def test_resolve_arch_slots_well_formed(name):
    """All derived slot indices are in range and the four slot classes are
    pairwise disjoint (a slot is never both a stone and a history plane)."""
    arch = resolve_arch(name)
    n = arch.in_channels
    all_slots = (
        [arch.cur_stone_slot, arch.opp_stone_slot]
        + list(arch.history_planes)
        + list(arch.turn_phase_planes)
    )
    for s in all_slots:
        assert 0 <= s < n, f"{name}: slot {s} out of range [0,{n})"
    assert len(all_slots) == len(set(all_slots)), f"{name}: overlapping slots"


def test_resolve_arch_known_values():
    """Spot-check the three encodings the §P5-CT arc turned on."""
    v6 = resolve_arch("v6")
    assert v6.in_channels == 8
    assert v6.cur_stone_slot == 0 and v6.opp_stone_slot == 4
    assert v6.history_planes == (1, 2, 3, 5, 6, 7)
    assert v6.turn_phase_planes == ()

    tp = resolve_arch("v6tp")
    assert tp.in_channels == 10
    assert tp.opp_stone_slot == 4
    assert tp.turn_phase_planes == (8, 9)

    live2 = resolve_arch("v6_live2")
    assert live2.in_channels == 4
    assert live2.cur_stone_slot == 0 and live2.opp_stone_slot == 1
    assert live2.history_planes == ()
    assert live2.turn_phase_planes == (2, 3)


def test_resolve_arch_normalizes_dict_form():
    """resolve_arch funnels through normalize_encoding_name, so the dict form
    Trainer._propagate_encoding_into_config produces also resolves."""
    assert resolve_arch({"version": "v6_live2"}).in_channels == 4


def test_resolve_arch_is_frozen():
    """ArchSpec is immutable — callers cannot mutate a shared resolver result."""
    arch = resolve_arch("v6")
    with pytest.raises((AttributeError, TypeError)):
        arch.in_channels = 99  # type: ignore[misc]
