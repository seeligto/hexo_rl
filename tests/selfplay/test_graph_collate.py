"""WP-3 step 5 — contract tests for the single graph-wire resolver.

Drives `collate_graph_batch` (`hexo_rl/selfplay/graph_collate.py`) against a
REAL block-diagonal wire from the Rust producer
(`InferenceBatcher.next_graph_batch`), then EXECUTES the ragged-contract's nine
adversarial payloads (ADV-1..9, `gnn_ragged_contract_v1.md` §4.2) — each must
die with its NAMED error. This is the contract-test spec made real; the smoke
gate's part-3 depends on it.
"""
from __future__ import annotations

import copy
import dataclasses
import time

import numpy as np
import pytest

from engine import InferenceBatcher
from hexo_rl.encoding import lookup
from hexo_rl.selfplay import graph_collate as gc
from hexo_rl.selfplay.graph_collate import (
    AugRoundTripMismatch,
    BatchCountMismatch,
    DtypeMismatch,
    EdgeAttrGeometryMismatch,
    EdgeCrossesGraphBoundary,
    EmptyLegalSet,
    GatherNotLegalNode,
    GraphContractVersionMismatch,
    NodeCountChecksum,
    NonNativeSampleBuilder,
    OffsetsNonMonotonic,
    ScatterGatherCrossesGraph,
    ScatterSlotAliasing,
    ScatterSlotCanonicalMismatch,
    collate_graph_batch,
    graph_wire_from_rust,
)


# ---------------------------------------------------------------------------
# Build a REAL multi-graph wire from the Rust producer, copy into a mutable
# payload (module-scope: built once). The mock graphs are a fixed MIXED spread
# board (two far clusters → in-window + off-window legal nodes), so the wire
# exercises the −1 sentinel path and the cross-graph checks (B ≥ 2).
# ---------------------------------------------------------------------------
def _build_real_payload(n_graphs: int = 6):
    batcher = InferenceBatcher(encoding_spec=lookup("gnn_axis_v1"))
    try:
        batcher.spawn_mock_graph_games(n_graphs)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not batcher.has_pending_graph_requests():
            time.sleep(0.01)
        # let every mock thread enqueue before we pop a single fused batch.
        time.sleep(0.3)
        ids, gw = batcher.next_graph_batch(batch_size=n_graphs, max_wait_ms=2000)
        payload = graph_wire_from_rust(gw)
        assert payload.n_graphs >= 2, f"need B>=2 for cross-graph ADV, got {payload.n_graphs}"
        return ids, payload
    finally:
        # wake the blocked mock threads (Err on close) so none linger.
        batcher.close()


@pytest.fixture(scope="module")
def real_payload():
    _ids, payload = _build_real_payload()
    return payload


def _clone(p):
    """Deep-copy a payload (arrays copied) so a mutation can't leak across tests."""
    return dataclasses.replace(
        p,
        node_feat=p.node_feat.copy(),
        node_coords=p.node_coords.copy(),
        edge_index=p.edge_index.copy(),
        edge_attr=p.edge_attr.copy(),
        node_offsets=p.node_offsets.copy(),
        edge_offsets=p.edge_offsets.copy(),
        legal_offsets=p.legal_offsets.copy(),
        legal_node_gather=p.legal_node_gather.copy(),
        policy_dst_slot=p.policy_dst_slot.copy(),
        n_nodes_checksum=p.n_nodes_checksum.copy(),
        n_stones=p.n_stones.copy(),
        window_center=p.window_center.copy(),
        current_player=p.current_player.copy(),
    )


# ---------------------------------------------------------------------------
# Parity / round-trip gate — a real wire collates clean (full semantic layer).
# ---------------------------------------------------------------------------
def test_real_wire_collates_clean_full_semantic(real_payload):
    batch = collate_graph_batch(real_payload, expected_version=1, device="cpu", semantic="full")
    B = real_payload.n_graphs
    N = real_payload.node_feat.size // 11
    E = real_payload.edge_attr.size // 5
    Lg = real_payload.legal_node_gather.size
    assert batch.x.shape == (N, 11)
    assert batch.edge_index.shape == (2, E)
    assert batch.edge_attr.shape == (E, 5)
    assert str(batch.edge_index.dtype) == "torch.int64"
    assert batch.legal_mask.shape == (N,)
    assert int(batch.legal_mask.sum().item()) == Lg, "one legal node per legal-gather row"
    assert batch.legal_offsets.shape == (B + 1,)
    assert batch.n_graphs == B
    # off-window sentinel is CARRIED, never dropped (option (b)).
    assert int((real_payload.policy_dst_slot == -1).sum()) > 0, "fixture must have off-window nodes"


def test_off_window_sentinel_survives_collate(real_payload):
    # The −1 sentinel rows must reach the GraphBatch untouched (no dense drop).
    batch = collate_graph_batch(real_payload, device="cpu", semantic="full")
    slot = batch.policy_dst_slot.cpu().numpy()
    assert (slot == -1).sum() == (real_payload.policy_dst_slot == -1).sum()


# ---------------------------------------------------------------------------
# Handshakes
# ---------------------------------------------------------------------------
def test_contract_version_mismatch(real_payload):
    p = _clone(real_payload)
    p.contract_version = 2
    with pytest.raises(GraphContractVersionMismatch):
        collate_graph_batch(p, expected_version=1, device="cpu")


def test_non_native_builder_handshake(real_payload):
    p = _clone(real_payload)
    p.builder_impl = 2
    with pytest.raises(NonNativeSampleBuilder):
        collate_graph_batch(p, device="cpu")
    # test-only escape hatch: oracle builder accepted under the flag.
    collate_graph_batch(p, device="cpu", allow_oracle_builder=True, semantic="off")


# ---------------------------------------------------------------------------
# ADV-1..9 — each parses structurally but corrupts silently under a naive impl.
# ---------------------------------------------------------------------------
def test_adv_1a_dropped_last_node(real_payload):
    p = _clone(real_payload)
    N = p.node_feat.size // 11
    p.node_offsets[-1] = N - 1  # drops last node, still monotonic
    with pytest.raises(OffsetsNonMonotonic):
        collate_graph_batch(p, device="cpu")


def test_adv_1b_interior_off_by_one(real_payload):
    p = _clone(real_payload)
    p.node_offsets[1] += 1  # interior, endpoints + monotonicity intact
    with pytest.raises(NodeCountChecksum):
        collate_graph_batch(p, device="cpu")


def test_adv_2a_gather_crosses_graph(real_payload):
    p = _clone(real_payload)
    # legal node 0 belongs to graph 0; point its gather into graph 1.
    p.legal_node_gather[0] = int(p.node_offsets[1])
    with pytest.raises(ScatterGatherCrossesGraph):
        collate_graph_batch(p, device="cpu")


def test_adv_2b_slot_aliasing(real_payload):
    p = _clone(real_payload)
    g0_end = int(p.legal_offsets[1])
    in_win = [i for i in range(g0_end) if p.policy_dst_slot[i] != -1]
    assert len(in_win) >= 2, "graph 0 must have >=2 in-window legal nodes"
    p.policy_dst_slot[in_win[1]] = p.policy_dst_slot[in_win[0]]  # collide two moves
    with pytest.raises(ScatterSlotAliasing):
        collate_graph_batch(p, device="cpu")


def test_adv_3_edge_crosses_graph(real_payload):
    p = _clone(real_payload)
    E = p.edge_attr.size // 5
    ei = p.edge_index.reshape(2, E).copy()
    ei[1, 0] = int(p.node_offsets[1])  # edge 0 (graph 0) → dst in graph 1
    p.edge_index = ei.reshape(-1)
    with pytest.raises(EdgeCrossesGraphBoundary):
        collate_graph_batch(p, device="cpu")


def test_adv_4_edge_index_wrong_dtype(real_payload):
    p = _clone(real_payload)
    p.edge_index = p.edge_index.astype(np.uint16)  # the u16-wrap trap
    with pytest.raises(DtypeMismatch):
        collate_graph_batch(p, device="cpu")


def test_adv_7_slot_map_unrotated(real_payload):
    # graph rotated (window centre moved) but the slot-map left unrotated →
    # the slot-map leg of ADV-7 fires (ScatterSlotCanonicalMismatch).
    p = _clone(real_payload)
    p.window_center[0] = p.window_center[0] + 1  # shift graph 0's centre q
    with pytest.raises(ScatterSlotCanonicalMismatch):
        collate_graph_batch(p, device="cpu", semantic="full")


def test_adv_8_edge_attr_permuted(real_payload):
    p = _clone(real_payload)
    p.edge_attr[3] = -p.edge_attr[3]  # flip first real edge's signed_dist
    with pytest.raises(EdgeAttrGeometryMismatch):
        collate_graph_batch(p, device="cpu", semantic="full")


def test_adv_9_gather_at_stone_node(real_payload):
    p = _clone(real_payload)
    # legal node 0 (graph 0) → point its gather at graph 0's first STONE row.
    p.legal_node_gather[0] = int(p.node_offsets[0])
    with pytest.raises(GatherNotLegalNode):
        collate_graph_batch(p, device="cpu", semantic="full")


# ---------------------------------------------------------------------------
# Remaining named errors (coverage of the 18 not hit by ADV-1..9)
# ---------------------------------------------------------------------------
def test_empty_legal_set():
    # A structurally-valid single-graph payload with ZERO legal nodes (1 stone +
    # 1 dummy) — reaches the EmptyLegalSet check (E=0, Lg=0 skip the earlier
    # edge/gather checks) and must die there.
    p = gc.GraphWirePayload(
        contract_version=1,
        builder_impl=1,
        n_graphs=1,
        node_feat=np.zeros(2 * 11, dtype=np.float32),
        node_coords=np.zeros(2 * 2, dtype=np.int32),
        edge_index=np.zeros(0, dtype=np.int64),
        edge_attr=np.zeros(0, dtype=np.float32),
        node_offsets=np.array([0, 2], dtype=np.int64),
        edge_offsets=np.array([0, 0], dtype=np.int64),
        legal_offsets=np.array([0, 0], dtype=np.int64),
        legal_node_gather=np.zeros(0, dtype=np.int64),
        policy_dst_slot=np.zeros(0, dtype=np.int32),
        n_nodes_checksum=np.array([2], dtype=np.uint32),
        n_stones=np.array([1], dtype=np.uint16),
        window_center=np.array([0, 0], dtype=np.int32),
        current_player=np.array([1], dtype=np.int8),
    )
    with pytest.raises(EmptyLegalSet):
        collate_graph_batch(p, device="cpu")


def test_batch_count_mismatch(real_payload):
    p = _clone(real_payload)
    p.current_player = p.current_player[:-1].copy()  # drop one graph's player
    with pytest.raises(BatchCountMismatch):
        collate_graph_batch(p, device="cpu")


def test_aug_round_trip_mismatch(real_payload):
    # trainer-path canary: a target-argmax cell that is not a legal node of its
    # graph = graph/target desync (silent label poisoning).
    p = _clone(real_payload)
    B = p.n_graphs
    targets = [None] * B
    targets[0] = (99999, 99999)  # not a legal node
    with pytest.raises(AugRoundTripMismatch):
        collate_graph_batch(p, device="cpu", semantic="full", target_argmax_cells=targets)


# ---------------------------------------------------------------------------
# Full step-3 + step-5 integration: real wire → collate → segmented-softmax →
# submit_graph_inference_results → the assembled LegalSetPolicy wakes the
# blocked mock worker (completed_graph_games increments).
# ---------------------------------------------------------------------------
@pytest.mark.timeout(30)
def test_wire_round_trips_to_assemble_and_completes():
    batcher = InferenceBatcher(encoding_spec=lookup("gnn_axis_v1"))
    try:
        n = 4
        batcher.spawn_mock_graph_games(n)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not batcher.has_pending_graph_requests():
            time.sleep(0.01)
        time.sleep(0.3)
        ids, gw = batcher.next_graph_batch(batch_size=n, max_wait_ms=2000)
        assert len(ids) == n

        # collate is clean on the real wire.
        payload = graph_wire_from_rust(gw)
        collate_graph_batch(payload, device="cpu", semantic="full")

        # per-graph uniform probs (stand-in for the segmented softmax output).
        lo = payload.legal_offsets.astype(np.int64)
        Lg = int(lo[-1])
        probs = np.zeros(Lg, dtype=np.float32)
        for g in range(len(ids)):
            s, e = int(lo[g]), int(lo[g + 1])
            probs[s:e] = 1.0 / float(e - s)
        values = np.zeros(len(ids), dtype=np.float32)
        batcher.submit_graph_inference_results(ids, probs, lo, values)

        done = time.monotonic() + 5.0
        while time.monotonic() < done and batcher.completed_graph_games() < n:
            time.sleep(0.01)
        assert batcher.completed_graph_games() == n
    finally:
        batcher.close()


def test_grid_batcher_rejects_graph_methods():
    b = InferenceBatcher(feature_len=8 * 19 * 19, policy_len=362)
    with pytest.raises(ValueError, match="RepresentationMismatch"):
        b.next_graph_batch(4, 5)
    with pytest.raises(ValueError, match="RepresentationMismatch"):
        b.spawn_mock_graph_games(1)


def test_check_graph_request_seam_obligations():
    b = InferenceBatcher(encoding_spec=lookup("gnn_axis_v1"))
    good = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
    b.check_graph_request(good, 1, 2)  # no raise
    # current_player out of range
    with pytest.raises(ValueError, match="current_player"):
        b.check_graph_request(good, 2, 2)
    # moves_remaining out of range (narrowing-cast guard)
    with pytest.raises(ValueError, match="moves_remaining"):
        b.check_graph_request(good, 1, 256)
    # coord overflow (WP1 Attack-2)
    with pytest.raises(ValueError, match="coord"):
        b.check_graph_request([(2**31 - 1, 0, 1)], 1, 2)
    # bad player
    with pytest.raises(ValueError, match="player"):
        b.check_graph_request([(0, 0, 5)], 1, 2)
