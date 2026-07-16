"""S7 F8 — `LocalInferenceEngine.infer_batch` graph branch.

Closes S7 round-2 Finding F8 (reports/probes/gnn_integration/S7_smoke_gate.md
"Re-run after blocker fixes"): `hexo_rl/selfplay/inference.py:82`'s
`infer_batch` was dense-only (`self.model.in_channels` unconditional read) —
the offline searched-eval path (`gumbel_search_py.py::_infer_and_expand` ->
`engine.infer_batch(leaves)`, the path `mantis_pull_eval.py` stage-2 drives)
crashed on the FIRST head move for any graph checkpoint, before a single
game played (`AttributeError: 'GnnNet' object has no attribute
'in_channels'`).

Fix reuses the WP-3 production graph inference seam
(`InferenceBatcher.submit_graphs_and_wait` -> the background `InferenceServer`
graph loop -> `collate_graph_batch` -> `GnnNet.forward_batch` -> segment-
softmax -> `assemble_ls_from_gnn_probs`) — the SAME machinery
`tests/selfplay/test_gnn_seam_smoke.py` already validates end-to-end for
self-play, now reachable from the synchronous `LocalInferenceEngine.infer_batch`
API every eval/deploy caller (`ModelPlayer`, `deploy_strength_eval`,
`gumbel_search_py`) already uses.

CPU-only here (byte-parity / determinism, matching `test_gnn_seam_smoke.py`'s
own convention) — the seam is representation, not device, dependent; a
separate CUDA leg is unnecessary duplication of already-CUDA-gated coverage
elsewhere (`test_gnn_train_step.py`'s fp16 parametrize exercises the graph
forward path on real CUDA).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from engine import Board
from hexo_rl.encoding import lookup
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board
from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine

_SPEC = lookup("gnn_axis_v1")


def _graph_engine() -> LocalInferenceEngine:
    net = GnnNet().to(torch.device("cpu")).eval()
    return LocalInferenceEngine(net, torch.device("cpu"), encoding_spec=_SPEC)


def test_infer_batch_graph_branch_no_attributeerror_and_correct_shape():
    """The exact S7 F8 crash site: infer_batch on a GnnNet must not raise, and
    must return a policy vector of the registry's policy_logit_count."""
    engine = _graph_engine()
    try:
        board = Board()
        policies, values = engine.infer_batch([board])
        assert len(policies) == 1
        assert len(values) == 1
        assert len(policies[0]) == _SPEC.policy_logit_count
        assert all(np.isfinite(policies[0]))
        assert np.isfinite(values[0])
        # No off-window drop expected for THIS encoding at ply 0 (whole-board,
        # OQ-6) — dense mass alone should already be ~1, not near-zero.
        assert abs(sum(policies[0]) - 1.0) < 1e-3
    finally:
        engine.close()


def test_infer_batch_graph_branch_produces_a_legal_argmax_move():
    """The argmax cell of the returned policy must be an actually-legal move
    — proves the returned vector is coordinate-aligned with the board's own
    action space, not just finite/well-shaped."""
    engine = _graph_engine()
    try:
        board = Board()
        policies, _values = engine.infer_batch([board])
        best_idx = int(np.argmax(policies[0]))
        legal_flat = {board.to_flat(q, r) for q, r in board.legal_moves()}
        assert best_idx in legal_flat
    finally:
        engine.close()


def test_infer_batch_graph_empty_boards_no_op():
    """Byte-identical early return (`if not boards: return [], []`) — the
    graph branch must never construct/query the batcher for an empty batch."""
    engine = _graph_engine()
    try:
        assert engine.infer_batch([]) == ([], [])
    finally:
        engine.close()


def test_run_gumbel_on_board_graph_plays_a_legal_searched_move():
    """End-to-end proof: a REAL Gumbel search (the exact
    `gumbel_search_py.run_gumbel_on_board` -> `_infer_and_expand` ->
    `engine.infer_batch` path `mantis_pull_eval.py` stage-2 drives) on a
    graph checkpoint completes and plays a legal move — the live-proof
    criterion S7's own gate re-run names ('a graph ckpt must play searched
    games end-to-end')."""
    engine = _graph_engine()
    try:
        board = Board()
        out = run_gumbel_on_board(engine, board, n_sims=8, m=4, gumbel_scale=0.0)
        assert out["played_move"] is not None
        assert out["played_move"] in set(board.legal_moves())
        assert np.isfinite(out["improved_policy"]).all()
    finally:
        engine.close()


def test_infer_batch_per_cluster_graph_raises_not_attributeerror():
    """Sweep-completeness sibling (S7 F5b/F7/F8 fix-class): the no-drop
    legal-set decode has no graph analogue (OQ-6) — must die loud with a
    named, actionable error, never the bare `.in_channels` AttributeError."""
    engine = _graph_engine()
    try:
        with pytest.raises(NotImplementedError, match="infer_batch_per_cluster"):
            engine.infer_batch_per_cluster([Board()])
    finally:
        engine.close()


def test_graph_engine_close_stops_server_thread_idempotent():
    engine = _graph_engine()
    assert engine._graph_server is not None
    server_thread = engine._graph_server
    engine.infer_batch([Board()])  # warm the thread up first
    engine.close()
    assert engine._graph_server is None
    assert not server_thread.is_alive()
    engine.close()  # idempotent — no raise on double-close


def test_dense_engine_unaffected_no_graph_server_byte_identical():
    """Dense (grid) construction must not spin up any graph machinery — the
    hoisted branch stays a true no-op for every existing dense caller."""
    net = HexTacToeNet(board_size=19, res_blocks=1, filters=8, in_channels=8)
    engine = LocalInferenceEngine(net, torch.device("cpu"), encoding_spec=lookup("v6"))
    try:
        assert engine._is_graph is False
        assert engine._graph_server is None
        assert engine._graph_batcher is None
        policies, values = engine.infer_batch([Board()])
        assert len(policies) == 1 and len(policies[0]) == lookup("v6").policy_logit_count
    finally:
        engine.close()  # no-op for a dense engine — must not raise
