"""GNN-integration WP-5a — HEXG training-data path integration (RUST half).

Drives the REAL vertical slice on real self-play positions
(`reports/probes/gnn_integration/wpa_positions.json`):

    push_graph_position → sample_graph_batch (rebuild-at-native-builder + D6 aug)
      → collate_graph_batch (full 18-assertion set incl. the ADV-7 argmax leg on
        the emitted targets) → GnnNet.forward_batch → ragged_policy_ce +
        binned_value_loss → backward

plus the persist cross-format LOUD-FAIL (a HEXG file handed to the dense loader)
and the TRUE ADV-7 desync (a target-argmax poisoned onto a non-legal cell must
raise `AugRoundTripMismatch`). Part-3 of the run4 integration smoke gate (§9).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine import HexgBuffer, ReplayBuffer  # noqa: E402
from hexo_rl.selfplay.graph_collate import (  # noqa: E402
    AugRoundTripMismatch,
    collate_graph_batch,
    stone_mask_from_batch,
)

ENC = "gnn_axis_v1"
_WPA = Path("reports/probes/gnn_integration/wpa_positions.json")

# hex neighbor offsets (axial) — a stone's neighbors are always within radius 6,
# hence legal if empty, so they are guaranteed legal-node coords in the rebuild.
_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def _load_positions(n: int):
    if not _WPA.exists():
        pytest.skip(f"{_WPA} not present (WP-A frozen position set)")
    data = json.loads(_WPA.read_text())
    return data["positions"][:n]


def _empty_neighbor(stones):
    """One empty hex-neighbor of SOME stone — distance 1 from a stone and empty,
    so guaranteed legal (radius 6) in the rebuilt graph."""
    occ = {(q, r) for q, r, _ in stones}
    for q0, r0, _ in stones:
        for dq, dr in _NEIGHBORS:
            c = (q0 + dq, r0 + dr)
            if c not in occ:
                return c
    raise AssertionError("no empty neighbor — degenerate position")


def _push_positions(buf: HexgBuffer, positions, game_id_base=0):
    for i, p in enumerate(positions):
        stones = [(int(q), int(r), int(pl)) for q, r, pl in p["stones"]]
        # Put a clear visit peak on a legal neighbor, spread the rest on two more.
        nq, nr = _empty_neighbor(stones)
        visits = [(int(nq), int(nr), 1.0)]
        buf.push_graph_position(
            stones,
            visits,
            int(p["current_player"]),
            int(p["moves_remaining"]),
            int(p.get("ply", 0)) & 0xFFFF,
            True,  # is_full_search
            1.0 if i % 2 == 0 else -1.0,  # outcome
            True,  # value_valid
            30,  # game_length
            game_id_base + i,
        )


def _collated(buf: HexgBuffer, batch_size: int, augment: bool):
    wire, tg = buf.sample_graph_batch(batch_size, augment)
    batch = collate_graph_batch(
        wire,
        expected_version=1,
        device="cpu",
        semantic="full",
        target_argmax_cells=tg.target_argmax_cells,
    )
    return wire, tg, batch


# ─────────────────────────────────────────────────────────────────────────────


def test_push_sample_collate_roundtrip_native_builder():
    buf = HexgBuffer(64, ENC)
    _push_positions(buf, _load_positions(32))
    assert buf.size == 32

    wire, tg, batch = _collated(buf, 16, augment=False)
    assert wire.n_graphs == 16
    # F7: the sampled wire is native by construction (rebuild calls build_axis_graph).
    assert int(wire.builder_impl) == 1
    assert int(wire.contract_version) == 1
    # policy target: one segment per graph, each summing to ~1 (no off-window drop).
    pt = np.asarray(tg.policy_target)
    lo = np.asarray(wire.legal_offsets)
    assert pt.shape[0] == int(lo[-1])
    for g in range(16):
        seg = pt[lo[g]:lo[g + 1]]
        assert abs(float(seg.sum()) - 1.0) < 1e-4, f"graph {g} target must sum to 1"
    assert batch.n_graphs == 16


def test_forward_and_losses_finite_on_real_positions():
    from hexo_rl.model.gnn_net import GnnNet
    from hexo_rl.training.binned_value import binned_value_loss
    from hexo_rl.training.losses import ragged_policy_ce

    buf = HexgBuffer(64, ENC)
    _push_positions(buf, _load_positions(48))
    wire, tg, batch = _collated(buf, 24, augment=True)

    net = GnnNet()
    stone_mask = stone_mask_from_batch(batch)
    policy_logits, value, bin_logits = net.forward_batch(
        batch.x, batch.edge_index, batch.edge_attr, batch.legal_mask,
        stone_mask, batch.node_offsets,
    )
    assert torch.isfinite(policy_logits).all(), "policy logits must be finite"
    assert torch.isfinite(value).all(), "value must be finite"
    assert torch.isfinite(bin_logits).all(), "bin logits must be finite"

    policy_target = torch.tensor(np.asarray(tg.policy_target), dtype=torch.float32)
    outcomes = torch.tensor(np.asarray(tg.outcomes), dtype=torch.float32)
    value_mask = torch.tensor(np.asarray(tg.value_valid), dtype=torch.uint8)
    ifs = torch.tensor(np.asarray(tg.is_full_search), dtype=torch.uint8)

    l_pol = ragged_policy_ce(policy_logits, policy_target, batch.legal_offsets, ifs)
    l_val = binned_value_loss(bin_logits, outcomes, value_mask=value_mask)
    loss = l_pol + l_val
    assert torch.isfinite(loss), f"loss must be finite (pol={l_pol}, val={l_val})"

    # backward — every grad finite (the full training-data path is grad-capable).
    loss.backward()
    n_grad = 0
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "all grads must be finite"
            n_grad += 1
    assert n_grad > 0, "backward must populate gradients"


def test_augmented_sample_target_is_coherent():
    """The single-call emission keeps graph + target synced, so the collate's
    ADV-7 argmax leg PASSES on every augmented draw (target argmax is always a
    legal node of the emitted graph). Repeat to hit many D6 elements."""
    buf = HexgBuffer(32, ENC)
    _push_positions(buf, _load_positions(8))
    for _ in range(20):
        # collate with target_argmax_cells runs the AugRoundTrip argmax canary;
        # a raise here would mean the sample path desynced graph vs target.
        _collated(buf, 8, augment=True)


def test_adv7_desync_target_argmax_raises():
    """TRUE ADV-7: manufacture the desync the single-call path forbids — a target
    whose argmax cell is NOT a legal node of the emitted graph. The collate's
    AugRoundTrip argmax leg must raise `AugRoundTripMismatch`."""
    buf = HexgBuffer(32, ENC)
    _push_positions(buf, _load_positions(8))
    wire, tg, _ = _collated(buf, 8, augment=False)  # first collate must pass
    # Poison graph 0's argmax onto a STONE cell (occupied — never a legal node).
    coords = np.asarray(wire.node_coords).reshape(-1, 2)
    n_stones = np.asarray(wire.n_stones)
    node_off = np.asarray(wire.node_offsets)
    stone_cell = tuple(int(v) for v in coords[node_off[0]])  # first stone of graph 0
    poisoned = list(tg.target_argmax_cells)
    poisoned[0] = stone_cell
    assert n_stones[0] >= 1
    with pytest.raises(AugRoundTripMismatch):
        collate_graph_batch(
            wire, expected_version=1, device="cpu", semantic="full",
            target_argmax_cells=poisoned,
        )


def test_hexg_file_rejected_by_dense_loader():
    """Cross-format LOUD-FAIL: a HEXG file handed to the dense `ReplayBuffer`
    loader must reject on the HEXB magic check (never silently mis-parse)."""
    buf = HexgBuffer(16, ENC)
    _push_positions(buf, _load_positions(4))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.hexg")
        buf.save_to_path(path)
        # Round-trips into a fresh HEXG buffer.
        buf2 = HexgBuffer(16, ENC)
        assert buf2.load_from_path(path) == 4
        # But the dense loader must LOUD-FAIL on the foreign magic.
        dense = ReplayBuffer(16, "v6")
        with pytest.raises(Exception):
            dense.load_from_path(path)


def test_grid_encoding_rejected():
    with pytest.raises(Exception):
        HexgBuffer(16, "v6")


# ── WP5b commit-A red-team fix-pass (ADV-A / ADV-B / ADV-D) ─────────────────
# graduated from `reports/probes/gnn_integration/WP5b_commitA_redteam.md`
# #1 (RAGGED-PAYLOAD end-to-end) — the two write-seam gaps + the one HELD
# structural guard, all driven via the real PyO3 `push_graph_position` entry
# point (not the pure Rust helper).


@pytest.mark.parametrize("bad_outcome", [float("nan"), float("inf"), float("-inf")])
def test_adv_a_push_rejects_non_finite_outcome(bad_outcome):
    """ADV-A: a non-finite `outcome` must be rejected LOUD at push time — before
    the fix, this sailed through untouched (outcome lives in a separate
    `GraphTargets` object the 18-assertion collate contract never inspects) and
    poisoned `binned_value_loss` with NaN on any value_valid=1 row."""
    buf = HexgBuffer(16, ENC)
    positions = _load_positions(1)
    stones = [(int(q), int(r), int(pl)) for q, r, pl in positions[0]["stones"]]
    nq, nr = _empty_neighbor(stones)
    with pytest.raises(Exception):
        buf.push_graph_position(
            stones,
            [(int(nq), int(nr), 1.0)],
            int(positions[0]["current_player"]),
            int(positions[0]["moves_remaining"]),
            0,
            True,
            bad_outcome,  # outcome
            True,
            30,
            0,
        )
    assert buf.size == 0, "a rejected push must not mutate the buffer"


@pytest.mark.parametrize("bad_player", [0, 2, 5, -3])
def test_adv_b_push_rejects_stone_player_out_of_range(bad_player):
    """ADV-B: a stone-list player field outside {+1, -1} must be rejected LOUD
    at push time — before the fix, `sample_graph_batch` silently rebuilt a
    structurally-valid-but-wrong-feature graph (collate check #14 recomputes
    `src_player` from the corrupt node identity, so it passed silently)."""
    buf = HexgBuffer(16, ENC)
    positions = _load_positions(1)
    stones = [(int(q), int(r), int(pl)) for q, r, pl in positions[0]["stones"]]
    nq, nr = _empty_neighbor(stones)  # occupied-cell set unaffected by the player tamper below
    stones[0] = (stones[0][0], stones[0][1], bad_player)
    with pytest.raises(Exception):
        buf.push_graph_position(
            stones,
            [(int(nq), int(nr), 1.0)],
            int(positions[0]["current_player"]),
            int(positions[0]["moves_remaining"]),
            0,
            True,
            1.0,
            True,
            30,
            0,
        )
    assert buf.size == 0, "a rejected push must not mutate the buffer"


def test_adv_d_illegal_visit_coord_raises_through_push_sample_roundtrip():
    """ADV-D (HELD in the red-team, pinned here through the REAL push→sample
    chain via PyO3): a visit mass poisoned onto an OCCUPIED (illegal) cell must
    raise a loud mass-drop error at `sample_graph_batch`, not silently
    under-weight the CE target. WP-5a already unit-tests the pure
    `mass_drop_check` helper and the Rust-internal `push_record_impl` →
    `sample_graph_batch_impl` chain (`hexg/tests.rs::
    sample_rejects_illegal_cell_visit_mass_drop`); this pins the same
    behavior through the actual Python entry points self-play/training use."""
    buf = HexgBuffer(16, ENC)
    stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
    # 0.9 mass on the OCCUPIED (0,0) cell (illegal), 0.1 on a legal neighbor.
    visits = [(0, 0, 0.9), (2, 0, 0.1)]
    buf.push_graph_position(stones, visits, 1, 2, 5, True, 1.0, True, 30, 7)
    assert buf.size == 1
    with pytest.raises(Exception):
        buf.sample_graph_batch(1, False)
