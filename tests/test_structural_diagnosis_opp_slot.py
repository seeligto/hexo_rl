"""§P5-CT — NEW L65-class finds beyond the ledger: structural_diagnosis probes
hardcoded the v6 opponent kept-slot 4.

position_classifier.classify_state read `state[4]` for the opponent, and
a3_h_bank sliced `states[:, 4]` — both crash / misread on a 4-plane v6_live2
array (opp at slot 1). Route the opponent slot via the registry (plane count →
encoding → opp_stone_slot).
"""
from __future__ import annotations

import numpy as np
import pytest

from hexo_rl.encoding import lookup
from hexo_rl.encoding.resolvers import opp_stone_slot
from scripts.structural_diagnosis.track_a.position_classifier import classify_state
from scripts.structural_diagnosis.track_a.a3_h_bank import _slots_for_plane_count


@pytest.mark.parametrize("encoding", ["v6", "v6tp", "v6_live2"])
def test_slots_for_plane_count_match_registry(encoding):
    spec = lookup(encoding)
    cur_slot, opp_slot = _slots_for_plane_count(spec.n_planes)
    assert cur_slot == 0
    assert opp_slot == opp_stone_slot(spec)


def test_classify_state_opp_slot_param_4plane():
    """A 4-plane v6_live2 state must classify via opp_slot=1 without IndexError."""
    spec = lookup("v6_live2")
    state = np.zeros((spec.n_planes, 19, 19), dtype=np.float16)
    # plant a tight opponent blob at the registry-derived opp slot
    opp_slot = opp_stone_slot(spec)
    state[opp_slot, 9:12, 9:12] = 1.0
    # default opp_slot=4 would IndexError on a 4-plane state; pass the real slot
    cls = classify_state(state, opp_slot=opp_slot, min_stones=5)
    assert cls in ("colony", "extension", "neither")


def test_classify_state_default_is_v6_backcompat():
    """Default opp_slot stays 4 (v6) — existing 8-plane callers unchanged."""
    state = np.zeros((8, 19, 19), dtype=np.float16)
    state[4, 9:12, 9:12] = 1.0  # opp blob at v6 slot 4
    assert classify_state(state, min_stones=5) == "colony"
