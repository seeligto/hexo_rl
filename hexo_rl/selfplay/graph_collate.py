"""The SINGLE wire reader for the GNN ragged-payload contract v1 (WP-3 step 5).

`collate_graph_batch` is the one-and-only consumer of the block-diagonal graph
wire emitted by the Rust `InferenceBatcher.next_graph_batch` (`GraphWire`,
`engine/src/inference_bridge.rs`). It lives in a module imported by BOTH the
self-play hot path (`inference_server.py`, WP-3 step 6) AND the promotion-gate
CUDA subprocess (`eval_pipeline.py`) — eval reads self-play's seam
(`gnn_ragged_contract_v1.md` §2.3 / seam design §6). It is import-safe with no
parent-process state.

It (1) asserts `contract_version == 1`; (2) asserts the native-builder handshake
(`builder_impl == 1`) on any training/self-play path unless
`HEXO_ALLOW_ORACLE_BUILDER=1`; (3) runs the 18-assertion set (13 structural,
always full; 4 semantic/geometric, full on the trainer path / canary on the hot
path); (4) builds block-diagonal torch tensors. There is NO silent fixed-width
fallback anywhere — every mismatch raises a NAMED error (the F1 silent-corruption
class this contract exists to kill, D-FORENSIC F1).

The OUTPUT is NOT a dense-`[B,362]` scatter (WP-3 option (b), seam design §1):
`collate_graph_batch` produces only the INPUT `GraphBatch`; the InferenceServer
segment-softmaxes the per-legal-node logits and returns ragged probs consumed
Rust-side by `assemble_ls_from_gnn_probs`. `policy_dst_slot` (incl. the −1
off-window sentinel) travels on the wire but is NOT dropped or dense-scattered
here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# The 3 win axes in axial coords (mirrors hexo_graph::WIN_AXES /
# strix_v1_graph.WIN_AXES). Load-bearing for the edge-geometry recompute.
WIN_AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))

# Contract-fixed schema widths (single-sourced against hexo_graph constants;
# callers pass spec.node_feat_dim / spec.edge_feat_dim from the registry).
_NODE_FEAT_DIM = 11  # fallback default for spec.node_feat_dim (registry gnn_axis_v1)
_EDGE_FEAT_DIM = 5  # fallback default for spec.edge_feat_dim (registry gnn_axis_v1)
_OFF_WINDOW_SLOT = -1
_BUILDER_IMPL_NATIVE = 1


# ---------------------------------------------------------------------------
# The 18 named errors (contract §2.5). All subclass ValueError so existing
# loud-fail call sites (die-loud → submit_inference_failure) catch uniformly.
# ---------------------------------------------------------------------------
class GraphContractError(ValueError):
    """Base for every graph-wire contract violation."""


# --- startup handshake -----------------------------------------------------
class GraphContractVersionMismatch(GraphContractError):
    pass


class NonNativeSampleBuilder(GraphContractError):
    pass


# --- structural layer (13) -------------------------------------------------
class NodeFeatDimMismatch(GraphContractError):
    pass


class EdgeAttrDimMismatch(GraphContractError):
    pass


class DtypeMismatch(GraphContractError):
    pass


class BatchCountMismatch(GraphContractError):
    pass


class OffsetsNonMonotonic(GraphContractError):
    pass


class NodeCountChecksum(GraphContractError):
    pass


class EdgeIndexOutOfBounds(GraphContractError):
    pass


class EdgeCrossesGraphBoundary(GraphContractError):
    pass


class ScatterGatherCrossesGraph(GraphContractError):
    pass


class ScatterSlotOutOfBounds(GraphContractError):
    pass


class ScatterSlotAliasing(GraphContractError):
    pass


class EmptyLegalSet(GraphContractError):
    pass


# --- semantic / geometric layer (4) ---------------------------------------
class EdgeAttrGeometryMismatch(GraphContractError):
    pass


class GatherNotLegalNode(GraphContractError):
    pass


class ScatterSlotCanonicalMismatch(GraphContractError):
    pass


class AugRoundTripMismatch(GraphContractError):
    pass


# ---------------------------------------------------------------------------
# Payload + output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class GraphWirePayload:
    """Pure-Python mirror of the Rust `GraphWire` pyclass. The resolver reads
    the SAME duck-typed attribute surface from either the Rust wire (numpy via
    PyO3 getters) or this dataclass (tests / the oracle-built payload). Every
    array is flat 1-D numpy with the contract dtype (§2.1)."""

    contract_version: int
    builder_impl: int
    n_graphs: int
    node_feat: np.ndarray
    node_coords: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_offsets: np.ndarray
    edge_offsets: np.ndarray
    legal_offsets: np.ndarray
    legal_node_gather: np.ndarray
    policy_dst_slot: np.ndarray
    n_nodes_checksum: np.ndarray
    n_stones: np.ndarray
    window_center: np.ndarray
    current_player: np.ndarray


@dataclass
class GraphBatch:
    """Collated block-diagonal torch tensors. `x`/`edge_index`/`edge_attr`/
    `legal_mask` feed `GnnNet.forward_batch`; the remaining fields support the
    ragged OUTPUT assemble (seam design §3.4) and stay on device."""

    x: Any  # torch.Tensor (N, 11) float
    edge_index: Any  # (2, E) int64
    edge_attr: Any  # (E, 5) float
    legal_mask: Any  # (N,) bool
    legal_offsets: Any  # (B+1,) int64
    legal_node_gather: Any  # (Lg,) int64 (global rows)
    policy_dst_slot: Any  # (Lg,) int64 (per-graph slot; -1 off-window)
    node_offsets: Any  # (B+1,) int64
    node_coords: Any  # (N, 2) int64
    window_center: Any  # (B, 2) int64
    current_player: Any  # (B,) int8→int64
    n_stones: Any  # (B,) int64
    n_graphs: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canary state (hot-path semantic cadence). Trainer runs semantic="full";
# self-play runs semantic="canary" — first batch after reset + every Nth.
# ---------------------------------------------------------------------------
_CANARY_STATE = {"count": 0}


def reset_semantic_canary() -> None:
    """Reset the canary counter — call after every process start / weight swap
    so the FIRST batch runs the full geometric layer (seam design §6.1)."""
    _CANARY_STATE["count"] = 0


def _canary_should_run(period: int) -> bool:
    n = _CANARY_STATE["count"]
    _CANARY_STATE["count"] = n + 1
    return (n == 0) or (period > 0 and n % period == 0)


# ---------------------------------------------------------------------------
# Geometry helper — window_flat_idx_at_geom (byte-parity with the Rust builder
# core.rs / hexo_graph::window_flat_idx). Vectorized over coord arrays.
# ---------------------------------------------------------------------------
def _canonical_slot_vec(
    q: np.ndarray, r: np.ndarray, cq: np.ndarray, cr: np.ndarray, trunk: int
) -> np.ndarray:
    half = (trunk - 1) // 2
    wq = q - cq + half
    wr = r - cr + half
    inside = (wq >= 0) & (wq < trunk) & (wr >= 0) & (wr < trunk)
    slot = np.where(inside, wq * trunk + wr, _OFF_WINDOW_SLOT)
    return slot.astype(np.int64)


def _graph_of(offsets: np.ndarray, count: int) -> np.ndarray:
    """Map each element index [0,count) to its graph via CSR `offsets`."""
    return np.searchsorted(offsets, np.arange(count), side="right") - 1


# ---------------------------------------------------------------------------
# Rust GraphWire → GraphWirePayload adapter (step-6 wiring reads the pyclass).
# ---------------------------------------------------------------------------
def graph_wire_from_rust(gw: Any) -> GraphWirePayload:
    """Read the Rust `engine.GraphWire` pyclass getters into a payload. The
    resolver is duck-typed, so this is only needed where a caller wants an
    explicit dataclass; `collate_graph_batch` accepts the pyclass directly."""
    return GraphWirePayload(
        contract_version=int(gw.contract_version),
        builder_impl=int(gw.builder_impl),
        n_graphs=int(gw.n_graphs),
        node_feat=np.asarray(gw.node_feat),
        node_coords=np.asarray(gw.node_coords),
        edge_index=np.asarray(gw.edge_index),
        edge_attr=np.asarray(gw.edge_attr),
        node_offsets=np.asarray(gw.node_offsets),
        edge_offsets=np.asarray(gw.edge_offsets),
        legal_offsets=np.asarray(gw.legal_offsets),
        legal_node_gather=np.asarray(gw.legal_node_gather),
        policy_dst_slot=np.asarray(gw.policy_dst_slot),
        n_nodes_checksum=np.asarray(gw.n_nodes_checksum),
        n_stones=np.asarray(gw.n_stones),
        window_center=np.asarray(gw.window_center),
        current_player=np.asarray(gw.current_player),
    )


# ---------------------------------------------------------------------------
# The single resolver
# ---------------------------------------------------------------------------
def collate_graph_batch(
    wire: Any,
    expected_version: int = 1,
    *,
    trunk_size: int = 19,  # caller passes spec.trunk_size (registry)
    win_length: int = 6,
    node_feat_dim: int = _NODE_FEAT_DIM,
    edge_feat_dim: int = _EDGE_FEAT_DIM,
    device: Optional[str] = None,
    semantic: str = "full",
    canary_period: int = 64,
    allow_oracle_builder: bool = False,
    target_argmax_cells: Optional[Sequence[Optional[tuple[int, int]]]] = None,
) -> GraphBatch:
    """Validate + collate one block-diagonal graph wire → `GraphBatch`.

    `semantic`: "full" (trainer — every batch), "canary" (hot path — first +
    every Nth), or "off". The structural layer (13) always runs full. Raises a
    NAMED `GraphContractError` on any mismatch — never a silent fallback.
    """
    import torch  # deferred: keeps this module import-safe in torch-free envs

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- resolver step 1: contract-version handshake (§2.3) ---
    cv = int(wire.contract_version)
    if cv != expected_version:
        raise GraphContractVersionMismatch(
            f"contract_version {cv} != expected {expected_version}"
        )

    # --- resolver step 2: native-builder handshake (F7) ---
    allow_oracle = allow_oracle_builder or os.environ.get("HEXO_ALLOW_ORACLE_BUILDER") == "1"
    builder_impl = int(wire.builder_impl)
    if builder_impl != _BUILDER_IMPL_NATIVE and not allow_oracle:
        raise NonNativeSampleBuilder(
            f"builder_impl {builder_impl} != {_BUILDER_IMPL_NATIVE} (native); "
            "the 26x Python-builder sample-path trap is refused "
            "(set HEXO_ALLOW_ORACLE_BUILDER=1 only for parity tests/CI)"
        )

    # Pull flat arrays (numpy view of either the pyclass getters or the payload).
    node_feat = np.asarray(wire.node_feat)
    node_coords = np.asarray(wire.node_coords)
    edge_index = np.asarray(wire.edge_index)
    edge_attr = np.asarray(wire.edge_attr)
    node_offsets = np.asarray(wire.node_offsets)
    edge_offsets = np.asarray(wire.edge_offsets)
    legal_offsets = np.asarray(wire.legal_offsets)
    legal_node_gather = np.asarray(wire.legal_node_gather)
    policy_dst_slot = np.asarray(wire.policy_dst_slot)
    n_nodes_checksum = np.asarray(wire.n_nodes_checksum)
    n_stones = np.asarray(wire.n_stones)
    window_center = np.asarray(wire.window_center)
    current_player = np.asarray(wire.current_player)
    B = int(wire.n_graphs)

    # --- resolver step 3a: STRUCTURAL layer (13) — always full ---
    _check_structural(
        node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
        legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
        window_center, current_player, B, node_feat_dim, edge_feat_dim,
    )

    # --- resolver step 3b: SEMANTIC/GEOMETRIC layer (4) — mode-gated ---
    run_semantic = semantic == "full" or (semantic == "canary" and _canary_should_run(canary_period))
    if run_semantic:
        _check_semantic(
            node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
            legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
            window_center, current_player, B, trunk_size, win_length, node_feat_dim,
            edge_feat_dim, target_argmax_cells,
        )

    # --- resolver step 4: block-diagonal torch tensors (edge_index already global) ---
    N = node_feat.size // node_feat_dim
    E = edge_attr.size // edge_feat_dim
    x = torch.from_numpy(np.ascontiguousarray(node_feat, dtype=np.float32)).reshape(N, node_feat_dim).to(device)
    ei = torch.from_numpy(np.ascontiguousarray(edge_index, dtype=np.int64)).reshape(2, E).to(device)
    ea = torch.from_numpy(np.ascontiguousarray(edge_attr, dtype=np.float32)).reshape(E, edge_feat_dim).to(device)
    legal_mask_np = np.zeros(N, dtype=bool)
    legal_mask_np[legal_node_gather.astype(np.int64)] = True
    legal_mask = torch.from_numpy(legal_mask_np).to(device)

    return GraphBatch(
        x=x,
        edge_index=ei,
        edge_attr=ea,
        legal_mask=legal_mask,
        legal_offsets=torch.from_numpy(np.ascontiguousarray(legal_offsets, dtype=np.int64)).to(device),
        legal_node_gather=torch.from_numpy(np.ascontiguousarray(legal_node_gather, dtype=np.int64)).to(device),
        policy_dst_slot=torch.from_numpy(np.ascontiguousarray(policy_dst_slot, dtype=np.int64)).to(device),
        node_offsets=torch.from_numpy(np.ascontiguousarray(node_offsets, dtype=np.int64)).to(device),
        node_coords=torch.from_numpy(np.ascontiguousarray(node_coords, dtype=np.int64)).reshape(N, 2).to(device),
        window_center=torch.from_numpy(np.ascontiguousarray(window_center, dtype=np.int64)).reshape(B, 2).to(device),
        current_player=torch.from_numpy(np.ascontiguousarray(current_player, dtype=np.int64)).to(device),
        n_stones=torch.from_numpy(np.ascontiguousarray(n_stones, dtype=np.int64)).to(device),
        n_graphs=B,
        device=device,
    )


# ---------------------------------------------------------------------------
# Structural layer (13) — index in-range / unique / monotonic / typed.
# ---------------------------------------------------------------------------
def _check_structural(
    node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
    legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
    window_center, current_player, B, node_feat_dim, edge_feat_dim,
) -> None:
    # 1. NodeFeatDimMismatch
    if node_feat.size % node_feat_dim != 0:
        raise NodeFeatDimMismatch(f"len(node_feat)={node_feat.size} not divisible by {node_feat_dim}")
    N = node_feat.size // node_feat_dim
    if node_coords.size != 2 * N:
        raise NodeFeatDimMismatch(f"len(node_coords)={node_coords.size} != 2N={2 * N}")

    # 2. EdgeAttrDimMismatch
    if edge_attr.size % edge_feat_dim != 0:
        raise EdgeAttrDimMismatch(f"len(edge_attr)={edge_attr.size} not divisible by {edge_feat_dim}")
    E = edge_attr.size // edge_feat_dim
    if edge_index.size != 2 * E:
        raise EdgeAttrDimMismatch(f"len(edge_index)={edge_index.size} != 2E={2 * E}")

    # 3. DtypeMismatch — indices must be i64 (the u16-wrap ADV-4 defense).
    _require_dtype(node_feat, np.float32, "node_feat")
    _require_dtype(edge_attr, np.float32, "edge_attr")
    _require_dtype(node_coords, np.int32, "node_coords")
    _require_dtype(edge_index, np.int64, "edge_index")
    _require_dtype(node_offsets, np.int64, "node_offsets")
    _require_dtype(edge_offsets, np.int64, "edge_offsets")
    _require_dtype(legal_offsets, np.int64, "legal_offsets")
    _require_dtype(legal_node_gather, np.int64, "legal_node_gather")
    _require_dtype(policy_dst_slot, np.int32, "policy_dst_slot")
    _require_dtype(n_nodes_checksum, np.uint32, "n_nodes_checksum")
    _require_dtype(n_stones, np.uint16, "n_stones")
    _require_dtype(window_center, np.int32, "window_center")
    _require_dtype(current_player, np.int8, "current_player")

    # 4. BatchCountMismatch
    for name, arr, want in (
        ("node_offsets", node_offsets, B + 1),
        ("edge_offsets", edge_offsets, B + 1),
        ("legal_offsets", legal_offsets, B + 1),
        ("n_nodes_checksum", n_nodes_checksum, B),
        ("n_stones", n_stones, B),
        ("current_player", current_player, B),
        ("window_center", window_center, 2 * B),
    ):
        if arr.size != want:
            raise BatchCountMismatch(f"len({name})={arr.size} != {want} (B={B})")

    Lg = legal_node_gather.size
    if policy_dst_slot.size != Lg:
        raise BatchCountMismatch(f"len(policy_dst_slot)={policy_dst_slot.size} != Lg={Lg}")

    # 5. OffsetsNonMonotonic — non-decreasing, [0]=0, [B]=total.
    for name, off, total in (
        ("node_offsets", node_offsets, N),
        ("edge_offsets", edge_offsets, E),
        ("legal_offsets", legal_offsets, Lg),
    ):
        if off[0] != 0:
            raise OffsetsNonMonotonic(f"{name}[0]={off[0]} != 0")
        if off[-1] != total:
            raise OffsetsNonMonotonic(f"{name}[B]={off[-1]} != total {total}")
        if np.any(np.diff(off) < 0):
            raise OffsetsNonMonotonic(f"{name} not non-decreasing")

    # 6. NodeCountChecksum — per-graph count == checksum; n_stones+1 <= checksum.
    per_graph_nodes = np.diff(node_offsets)
    if not np.array_equal(per_graph_nodes, n_nodes_checksum.astype(np.int64)):
        raise NodeCountChecksum("per-graph node count != n_nodes_checksum")
    if np.any(n_stones.astype(np.int64) + 1 > n_nodes_checksum.astype(np.int64)):
        raise NodeCountChecksum("n_stones + 1 > n_nodes_checksum for some graph")

    # 7. EdgeIndexOutOfBounds
    if E > 0 and (edge_index.min() < 0 or edge_index.max() >= N):
        raise EdgeIndexOutOfBounds(f"edge_index out of [0,{N})")

    # 8. EdgeCrossesGraphBoundary
    if E > 0:
        ei2 = edge_index.reshape(2, E)
        node_graph = _graph_of(node_offsets, N)
        edge_graph = _graph_of(edge_offsets, E)
        src_g = node_graph[ei2[0]]
        dst_g = node_graph[ei2[1]]
        if np.any(src_g != edge_graph) or np.any(dst_g != edge_graph):
            raise EdgeCrossesGraphBoundary("an edge endpoint is outside its own graph's node range")

    # 9. ScatterGatherCrossesGraph
    if Lg > 0:
        node_graph = _graph_of(node_offsets, N)
        legal_graph = _graph_of(legal_offsets, Lg)
        gather_g = node_graph[legal_node_gather]
        if np.any(gather_g != legal_graph):
            raise ScatterGatherCrossesGraph("legal_node_gather points into another graph")

    # 10. ScatterSlotOutOfBounds — slot >= 362 or (negative and != -1).
    bad = (policy_dst_slot >= 362) | ((policy_dst_slot < 0) & (policy_dst_slot != _OFF_WINDOW_SLOT))
    if np.any(bad):
        raise ScatterSlotOutOfBounds("policy_dst_slot out of [0,362) and not the -1 sentinel")

    # 11. ScatterSlotAliasing — within one graph, two legal nodes share a slot.
    if Lg > 0:
        legal_graph = _graph_of(legal_offsets, Lg)
        in_win = policy_dst_slot != _OFF_WINDOW_SLOT
        keys = legal_graph.astype(np.int64) * 400 + policy_dst_slot.astype(np.int64)
        keys = keys[in_win]
        if keys.size != np.unique(keys).size:
            raise ScatterSlotAliasing("two legal nodes in one graph map to the same slot")

    # 12. EmptyLegalSet
    if np.any(np.diff(legal_offsets) == 0):
        raise EmptyLegalSet("a graph has an empty legal set")


def _require_dtype(arr: np.ndarray, want, name: str) -> None:
    if arr.dtype != np.dtype(want):
        raise DtypeMismatch(f"{name} dtype {arr.dtype} != {np.dtype(want)}")


# ---------------------------------------------------------------------------
# Semantic / geometric layer (4) — points at the geometrically-correct thing.
# ---------------------------------------------------------------------------
def _check_semantic(
    node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
    legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
    window_center, current_player, B, trunk_size, win_length, node_feat_dim,
    edge_feat_dim, target_argmax_cells,
) -> None:
    N = node_feat.size // node_feat_dim
    E = edge_attr.size // edge_feat_dim
    Lg = legal_node_gather.size
    coords = node_coords.reshape(N, 2).astype(np.int64)
    nf = node_feat.reshape(N, node_feat_dim)
    node_graph = _graph_of(node_offsets, N)
    # dummy node of each graph = last row of the graph.
    dummy_of_graph = node_offsets[1:] - 1
    node_is_dummy = np.zeros(N, dtype=bool)
    node_is_dummy[dummy_of_graph] = True
    cp = current_player.astype(np.int64)  # per-graph +1/-1

    # 14. EdgeAttrGeometryMismatch — recompute attrs from coords + player id.
    if E > 0:
        ei2 = edge_index.reshape(2, E)
        ea = edge_attr.reshape(E, edge_feat_dim)
        src = ei2[0]
        dst = ei2[1]
        touches_dummy = node_is_dummy[src] | node_is_dummy[dst]
        # dummy edges must be all-zero.
        if np.any(np.abs(ea[touches_dummy]) > 1e-6):
            raise EdgeAttrGeometryMismatch("a dummy edge has non-zero attrs")
        real = ~touches_dummy
        if np.any(real):
            eas = ea[real]
            s = src[real]
            d = dst[real]
            onehot = eas[:, :3]
            # exactly one of the 3 axis one-hots is 1.0, rest 0.0.
            axis = np.argmax(onehot, axis=1)
            onehot_ok = np.all((onehot == (np.arange(3)[None, :] == axis[:, None]).astype(np.float32)), axis=1)
            if not np.all(onehot_ok):
                raise EdgeAttrGeometryMismatch("edge axis one-hot is not a clean one-hot")
            dist = eas[:, 3]  # plane-literal-ok: edge_attr col 3 = signed_dist (contract §2.1, not a state plane)
            di = np.rint(dist).astype(np.int64)
            if np.any(dist != di.astype(dist.dtype)):
                raise EdgeAttrGeometryMismatch("signed_dist is non-integral")
            axis_vec = np.array(WIN_AXES, dtype=np.int64)  # (3,2)
            av = axis_vec[axis]  # (M,2)
            delta = coords[d] - coords[s]
            expect = di[:, None] * av
            if np.any(delta != expect) or np.any(di == 0) or np.any(np.abs(di) > (win_length - 1)):
                raise EdgeAttrGeometryMismatch("edge delta != signed_dist * axis_vec (rows misaligned/scrambled)")
            # src_player: relative own/opp cols × current_player[g], 0 for empty.
            g_of = node_graph[s]
            own = nf[s, 0]
            opp = nf[s, 1]
            expect_sp = (own - opp) * cp[g_of].astype(np.float32)
            if np.any(np.abs(eas[:, 4] - expect_sp) > 1e-6):  # plane-literal-ok: edge_attr col 4 = src_player (contract §2.1)
                raise EdgeAttrGeometryMismatch("edge src_player != node stone identity")

    # 15. GatherNotLegalNode — gather in the legal subrange (not stone/dummy).
    if Lg > 0:
        legal_graph = _graph_of(legal_offsets, Lg)
        lo = node_offsets[legal_graph] + n_stones.astype(np.int64)[legal_graph]
        hi = node_offsets[legal_graph + 1] - 1  # dummy row excluded
        if np.any(legal_node_gather < lo) or np.any(legal_node_gather >= hi):
            raise GatherNotLegalNode("legal_node_gather points at a stone or dummy node")
        # per-graph legal count == checksum - n_stones - 1.
        per_graph_legal = np.diff(legal_offsets)
        expect_legal = n_nodes_checksum.astype(np.int64) - n_stones.astype(np.int64) - 1
        if not np.array_equal(per_graph_legal, expect_legal):
            raise GatherNotLegalNode("per-graph legal count != checksum - n_stones - 1")

    # 16. ScatterSlotCanonicalMismatch — slot == canonical slot of the gathered coord.
    if Lg > 0:
        legal_graph = _graph_of(legal_offsets, Lg)
        gcoord = coords[legal_node_gather]
        wc = window_center.reshape(B, 2).astype(np.int64)
        cq = wc[legal_graph, 0]
        cr = wc[legal_graph, 1]
        canon = _canonical_slot_vec(gcoord[:, 0], gcoord[:, 1], cq, cr, trunk_size)  # plane-literal-ok: node_coords cols 0/1 = (q,r) axial (contract §2.1)
        if not np.array_equal(canon, policy_dst_slot.astype(np.int64)):
            raise ScatterSlotCanonicalMismatch(
                "policy_dst_slot != canonical window slot of the gathered (rotated) coord"
            )

    # 17. AugRoundTripMismatch — runtime canary (trainer path with a target):
    # the target-argmax cell must map to a legal node whose slot equals the
    # canonical slot of that cell's (rotated) coord. Skipped on inference (no
    # target).
    if target_argmax_cells is not None:
        if len(target_argmax_cells) != B:
            raise AugRoundTripMismatch(f"target_argmax_cells len {len(target_argmax_cells)} != B {B}")
        legal_graph = _graph_of(legal_offsets, Lg) if Lg > 0 else np.array([], dtype=np.int64)
        gcoord = coords[legal_node_gather] if Lg > 0 else np.zeros((0, 2), dtype=np.int64)
        for g in range(B):
            cell = target_argmax_cells[g]
            if cell is None:
                continue
            sel = np.where(legal_graph == g)[0]
            match = [i for i in sel if tuple(gcoord[i]) == tuple(cell)]
            if not match:
                raise AugRoundTripMismatch(
                    f"graph {g}: target cell {cell} is not a legal node (graph/target desync)"
                )


__all__ = [
    "collate_graph_batch",
    "graph_wire_from_rust",
    "reset_semantic_canary",
    "GraphWirePayload",
    "GraphBatch",
    "WIN_AXES",
    # errors
    "GraphContractError",
    "GraphContractVersionMismatch",
    "NonNativeSampleBuilder",
    "NodeFeatDimMismatch",
    "EdgeAttrDimMismatch",
    "DtypeMismatch",
    "BatchCountMismatch",
    "OffsetsNonMonotonic",
    "NodeCountChecksum",
    "EdgeIndexOutOfBounds",
    "EdgeCrossesGraphBoundary",
    "ScatterGatherCrossesGraph",
    "ScatterSlotOutOfBounds",
    "ScatterSlotAliasing",
    "EmptyLegalSet",
    "EdgeAttrGeometryMismatch",
    "GatherNotLegalNode",
    "ScatterSlotCanonicalMismatch",
    "AugRoundTripMismatch",
]
