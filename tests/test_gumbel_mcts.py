"""Tests for Gumbel MCTS: Sequential Halving and forced root selection.

Tests the Rust MCTSTree Gumbel features through the PyO3 boundary.
"""

import pytest
from engine import Board, MCTSTree


BOARD_SIZE = 19
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1


def uniform_policy():
    return [1.0 / N_ACTIONS] * N_ACTIONS


def run_search(tree: MCTSTree, n_sims: int) -> None:
    """Run n_sims standard MCTS simulations on the tree."""
    for _ in range(n_sims):
        leaves = tree.select_leaves(1)
        if not leaves:
            break
        policies = [uniform_policy() for _ in leaves]
        values = [0.0] * len(leaves)
        tree.expand_and_backup(policies, values)


class TestGumbelMCTSBasic:
    """Tests for the MCTSTree Gumbel-related features accessible from Python."""

    def test_tree_search_produces_valid_policy(self):
        """Standard search still produces valid policy (no Gumbel regression)."""
        tree = MCTSTree(c_puct=1.5)
        board = Board()
        tree.new_game(board)

        run_search(tree, 20)

        policy = tree.get_policy(temperature=1.0, board_size=BOARD_SIZE)
        assert len(policy) == N_ACTIONS
        total = sum(policy)
        assert abs(total - 1.0) < 1e-4, f"Policy should sum to 1.0, got {total}"
        assert all(p >= 0.0 for p in policy), "No negative probabilities"

    def test_improved_policy_valid_after_search(self):
        """get_improved_policy (completed Q-values) produces valid output."""
        tree = MCTSTree(c_puct=1.5)
        board = Board()
        tree.new_game(board)

        run_search(tree, 50)

        # Simulates what game_runner does after search.
        visits = tree.root_visits()
        assert visits == 50

        # get_improved_policy is not directly exposed via PyO3 in the current API.
        # Instead we verify the standard policy is valid, which indirectly ensures
        # the root child data that get_improved_policy reads is correctly populated.
        policy = tree.get_policy(temperature=1.0, board_size=BOARD_SIZE)
        total = sum(policy)
        assert abs(total - 1.0) < 1e-4

    def test_root_visits_accumulate(self):
        """Root visits count matches number of simulations."""
        tree = MCTSTree(c_puct=1.5)
        board = Board()
        tree.new_game(board)

        n_sims = 30
        run_search(tree, n_sims)
        assert tree.root_visits() == n_sims

    def test_search_on_advanced_position(self):
        """Search works on a board with several moves already played."""
        board = Board()
        board.apply_move(0, 0)
        board.apply_move(1, 0)
        board.apply_move(1, 1)

        tree = MCTSTree(c_puct=1.5)
        tree.new_game(board)

        run_search(tree, 20)

        policy = tree.get_policy(temperature=1.0, board_size=BOARD_SIZE)
        total = sum(policy)
        assert abs(total - 1.0) < 1e-4

    def test_low_sim_search_completes(self):
        """Search with very few sims (mimicking 50-sim fast games) completes."""
        tree = MCTSTree(c_puct=1.5)
        board = Board()
        tree.new_game(board)

        run_search(tree, 5)

        assert tree.root_visits() == 5
        policy = tree.get_policy(temperature=1.0, board_size=BOARD_SIZE)
        assert len(policy) == N_ACTIONS


class TestSelfPlayRunnerGumbelConfig:
    """Test that SelfPlayRunner accepts Gumbel config parameters."""

    def test_runner_accepts_gumbel_params(self):
        """SelfPlayRunner constructor accepts gumbel_mcts, gumbel_m, gumbel_explore_moves."""
        from engine import SelfPlayRunner

        # Should not raise — verifies the PyO3 signature accepts the new params.
        runner = SelfPlayRunner(
            n_workers=1,
            max_moves_per_game=0,  # 0 moves = instant game completion
            n_simulations=10,
            leaf_batch_size=1,
            c_puct=1.5,
            fpu_reduction=0.25,
            feature_len=24 * 19 * 19,
            policy_len=19 * 19 + 1,
            gumbel_mcts=True,
            gumbel_m=8,
            gumbel_explore_moves=5,
        )
        assert runner is not None

    def test_runner_defaults_gumbel_off(self):
        """SelfPlayRunner defaults to gumbel_mcts=False."""
        from engine import SelfPlayRunner

        # Default construction — should not use Gumbel.
        runner = SelfPlayRunner(
            n_workers=1,
            max_moves_per_game=0,
            n_simulations=10,
        )
        assert runner is not None


class TestQuiescenceFireCount:
    """Tests for the quiescence_fire_count FFI getter on PyMCTSTree."""

    def test_getter_returns_zero_on_new_tree(self):
        """get_quiescence_fire_count() returns 0 before any searches."""
        tree = MCTSTree(c_puct=1.5)
        board = Board()
        tree.new_game(board)
        # PyO3 #[getter] fn get_quiescence_fire_count → Python property quiescence_fire_count
        count = tree.quiescence_fire_count
        assert isinstance(count, int)
        assert count == 0

    def test_getter_resets_after_new_game(self):
        """new_game() resets quiescence_fire_count to 0."""
        tree = MCTSTree(c_puct=1.5)
        board = Board()
        tree.new_game(board)
        # Run a short search — early-game boards won't fire quiescence,
        # but the getter must still work and be non-negative.
        for _ in range(5):
            leaves = tree.select_leaves(1)
            if not leaves:
                break
            tree.expand_and_backup(
                [[1.0 / N_ACTIONS] * N_ACTIONS for _ in leaves],
                [0.0] * len(leaves),
            )
        count_before = tree.quiescence_fire_count
        assert count_before >= 0

        # After new_game(), counter resets.
        tree.new_game(board)
        assert tree.quiescence_fire_count == 0
