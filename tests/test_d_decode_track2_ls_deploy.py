"""§D-DECODE Track 2 — parity tests for the Rust multi-window legal-set deploy head.

The structural claim (D-DECODE, committed ebdeb77): the single-window g=0 Gumbel-SH
deploy head CANNOT REPRESENT off-window saving moves (`board.to_flat()` == usize::MAX →
no policy logit / no child), while the multi-window NO-DROP legal-set action space (the
space the net trains under in self-play) CAN. Track 2 productionizes that fix:
  * Rust `MCTSTree.expand_and_backup_ls` (PyO3) — the load-bearing wiring.
  * Python `LocalInferenceEngine.infer_batch_per_cluster` — RAW per-cluster outputs.
  * `run_gumbel_on_board(..., legal_set=True)` / `DeployHeadBot(..., legal_set=True)`.

These tests are WEIGHT-FREE: the off-window representability is a property of the ACTION
SPACE, not the net weights (a random-init net still produces a legal-set policy; the ls
path retains off-window covered cells regardless of logit values). So a fresh net on CPU
suffices and the asymmetry is deterministic. The firmed ORACLE (the 0/50-vs-12/50
position-level block rate, weight-dependent) is reproduced separately by the harness
`scripts/d_decode/firm_block_positions_track2.py` against the real checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from engine import Board, MCTSTree
from hexo_rl.encoding import lookup, normalize_encoding_name as _norm
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine

ENC = _norm("v6_live2_ls")


@pytest.fixture(scope="module")
def spec():
    return lookup(ENC)


@pytest.fixture(scope="module")
def engine(spec):
    torch.manual_seed(0)
    model = HexTacToeNet(encoding=ENC).eval()
    assert model.in_channels == len(spec.kept_plane_indices)
    return LocalInferenceEngine(model, torch.device("cpu"), encoding_spec=spec)


def _spread_board() -> Board:
    """Deterministic board with stones marched outward so the centroid 19-window
    cannot cover every radius-5 legal cell → off-window legal cells exist."""
    b = Board.with_encoding_name(ENC)
    seq = [(0, 0)]
    for step in range(1, 14):
        seq.append((step * 2, 0))
        seq.append((step * 2, 1))
    for (q, r) in seq:
        if not b.check_win() and (q, r) in b.legal_moves():
            b.apply_move(q, r)
    return b


def _off_window_legal(board: Board, n_actions: int):
    return {
        (int(q), int(r))
        for (q, r) in board.legal_moves()
        if board.to_flat(q, r) >= n_actions - 1
    }


def _root_children_coords(engine, board, spec, legal_set: bool):
    """Expand the root once via the ls or dense path; return the child coord set."""
    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    if legal_set:
        pols, vals, leaf_k = engine.infer_batch_per_cluster(leaves)
        tree.expand_and_backup_ls(
            pols, vals, leaf_k,
            int(spec.policy_logit_count),
            bool(spec.has_pass_slot),
            int(spec.cluster_window_size),
        )
    else:
        pols, vals = engine.infer_batch(leaves)
        tree.expand_and_backup(pols, vals)
    return {(int(q), int(r)) for ((q, r), *_rest) in tree.get_root_children_info()}


def test_offwindow_legal_cells_exist(engine, spec):
    """Precondition: the spread board genuinely has off-window legal cells (else the
    asymmetry test is vacuous)."""
    board = _spread_board()
    off = _off_window_legal(board, spec.policy_logit_count)
    assert len(off) > 0, "fixture board has no off-window legal cells"


def test_ls_root_represents_offwindow_singlewindow_cannot(engine, spec):
    """Load-bearing structural parity: the multi-window legal-set root expansion
    INCLUDES off-window cells as children; the single-window dense expansion includes
    NONE (no logit slot — `to_flat` == usize::MAX). This is the whole D-DECODE fix."""
    board = _spread_board()
    off = _off_window_legal(board, spec.policy_logit_count)

    ls_coords = _root_children_coords(engine, board, spec, legal_set=True)
    sw_coords = _root_children_coords(engine, board, spec, legal_set=False)

    ls_off = ls_coords & off
    sw_off = sw_coords & off

    assert len(ls_off) > 0, "ls root expansion did NOT represent any off-window cell"
    assert len(sw_off) == 0, (
        f"single-window root expansion represented off-window cells {sorted(sw_off)[:5]} "
        "— it structurally cannot (no logit slot); the dense path is broken"
    )


def test_expand_and_backup_ls_k_alignment_guard(engine, spec):
    """The Rust K-alignment / center-order contract fires loudly on a mismatch
    (sum(leaf_k) != len(policies)) rather than silently corrupting the tree."""
    board = _spread_board()
    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    pols, vals, leaf_k = engine.infer_batch_per_cluster(leaves)
    # Corrupt the K vector → must raise.
    bad_leaf_k = [leaf_k[0] + 1] if leaf_k else [1]
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls(
            pols, vals, bad_leaf_k,
            int(spec.policy_logit_count),
            bool(spec.has_pass_slot),
            int(spec.cluster_window_size),
        )


def test_gumbel_ls_searches_offwindow_candidates(engine, spec):
    """SH capability: the g=0 Gumbel-SH deploy head over the legal-set action space CAN
    search (and therefore CAN win on) off-window cells — at least one off-window root
    child accrues visits during sequential halving. The single-window run never does
    (no off-window child exists to visit). Played move is always a legal cell."""
    board = _spread_board()
    off = _off_window_legal(board, spec.policy_logit_count)
    legal = {(int(q), int(r)) for (q, r) in board.legal_moves()}

    out_ls = run_gumbel_on_board(
        engine, board, n_sims=150, m=16, c_visit=50.0, c_scale=1.0, c_puct=1.5,
        dirichlet=False, gumbel_scale=0.0, legal_set=True,
        rng=np.random.default_rng(0),
    )
    assert out_ls["played_move"] is not None
    assert (int(out_ls["played_move"][0]), int(out_ls["played_move"][1])) in legal

    # The witness dict (`_collect_witnesses`) exposes per-coord root-child visits.
    # An off-window cell with visits > 0 proves SH actually searched it as a candidate
    # → the winner CAN be an off-window block.
    visited_off = {c for c, v in out_ls["child_visits"].items() if v > 0 and c in off}
    assert len(visited_off) > 0, (
        "no off-window candidate was searched by the ls Gumbel-SH head — the legal-set "
        "candidate set never reached an off-window cell"
    )
