"""Smoke test: multi-cluster colony threat detection.

Verifies that when stones form multiple distant clusters, the Rust core
returns multiple cluster views and the model can process them.

Migrated from scripts/smoke_test_clusters.py (completed milestone check).
"""

import pytest
import torch
import numpy as np
from python.model.network import HexTacToeNet
from python.env.game_state import GameState
from native_core import Board


@pytest.fixture
def two_cluster_board():
    """Board with P1 colony at q=25 and P2 threat at center."""
    board = Board()

    # Turn 0: P1 (1 move)
    board.apply_move(25, 0)

    # Turn 1: P2 (2 moves)
    board.apply_move(0, 0)
    board.apply_move(1, 0)

    # Turn 2: P1 (2 moves)
    board.apply_move(25, 1)
    board.apply_move(25, 2)

    # Turn 3: P2 (2 moves)
    board.apply_move(2, 0)
    board.apply_move(3, 0)

    # Turn 4: P1 (2 moves)
    board.apply_move(25, 3)
    board.apply_move(25, 4)

    # Turn 5: P2 (2 moves) - 5-in-a-row threat + filler
    board.apply_move(4, 0)
    board.apply_move(-5, 5)

    return board


def test_multiple_clusters_detected(two_cluster_board):
    """Rust core should detect at least 2 separate clusters."""
    views, centers = two_cluster_board.get_cluster_views()
    assert len(centers) >= 2, f"Expected at least 2 clusters, got {len(centers)}"


def test_model_processes_multi_cluster_board(two_cluster_board):
    """Model forward pass should succeed on a multi-cluster board."""
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, res_blocks=4, filters=32)
    model.to(device)
    model.eval()

    state = GameState.from_board(two_cluster_board)
    tensors, centers = state.to_tensor()
    # Use first cluster view for forward pass
    batch = torch.from_numpy(tensors[0:1]).float().to(device)

    with torch.no_grad():
        log_policy, value, value_logit = model(batch)

    assert log_policy.shape[0] == 1
    assert log_policy.shape[1] >= 361  # 361 cells + optional pass
    assert value.shape == (1, 1)
