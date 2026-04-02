"""
Single-process self-play worker (Phase 2).

Plays one game at a time using MCTS + the current network. Each game
position is recorded with its MCTS policy and pushed to the replay buffer
at game end (once the outcome is known).

Phase 2 additions over Phase 1:
  - Dirichlet noise at root (self-play only, disabled for evaluation).
  - Temperature scheduling by ply (tau=1.0 early, tau=0.1 late, tau=0 eval).
  - Both controlled by config and a `use_dirichlet` flag.

Usage (via scripts/train.py):
    worker = SelfPlayWorker(model, config, device)
    n_positions = worker.play_game(replay_buffer)
    n_positions = worker.play_game(replay_buffer, use_dirichlet=False)  # eval
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from engine import Board, MCTSTree
from hexo_rl.env.game_state import GameState
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.selfplay.policy_projection import project_global_policy_to_local
from hexo_rl.selfplay.tensor_buffer import TensorBuffer
from engine import ReplayBuffer
from hexo_rl.selfplay.utils import BOARD_SIZE, N_ACTIONS, get_temperature  # noqa: F401 (re-exported)

# Backward-compat: callers that do `from hexo_rl.selfplay.worker import get_temperature`
# continue to work without changes.
__all__ = ["SelfPlayWorker", "get_temperature"]


class SelfPlayWorker:
    """Plays self-play games and pushes data to a ReplayBuffer.

    Args:
        model:   Trained (or random) HexTacToeNet.
        config:  Config dict.  Used keys:
                     n_simulations, c_puct, temperature_threshold_ply,
                     board_size (must match network input).
        device:  torch.device.
    """

    def __init__(
        self,
        model:  HexTacToeNet,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device

        mcts_cfg = config.get("mcts", config)
        self.n_sims          = int(mcts_cfg.get("n_simulations", config.get("n_simulations", 50)))
        self.c_puct          = float(mcts_cfg.get("c_puct", 1.5))
        self.temp_threshold  = int(mcts_cfg.get("temperature_threshold_ply", 30))
        self.dirichlet_alpha = float(mcts_cfg.get("dirichlet_alpha", 0.3))
        self.dirichlet_eps   = float(mcts_cfg.get("epsilon", 0.25))

        self._engine = LocalInferenceEngine(model, device)
        self.tree    = MCTSTree(c_puct=self.c_puct)
        self._buf    = TensorBuffer()

        # Keep a direct reference for callers that accessed worker.model
        self.model = model

    # ── Inference (thin delegation) ────────────────────────────────────────────

    @torch.no_grad()
    def _infer(self, board: Board) -> Tuple[List[float], float]:
        return self._engine.infer(board)

    @torch.no_grad()
    def _infer_batch(self, boards: List[Board]) -> Tuple[List[List[float]], List[float]]:
        return self._engine.infer_batch(boards)

    # ── MCTS search ────────────────────────────────────────────────────────────

    def _run_mcts(
        self,
        board: Board,
        use_dirichlet: bool = True,
        temperature: Optional[float] = None,
    ) -> List[float]:
        return self._run_mcts_with_sims(
            board, n_sims=self.n_sims,
            use_dirichlet=use_dirichlet, temperature=temperature,
        )

    def _run_mcts_with_sims(
        self,
        board: Board,
        n_sims: int,
        use_dirichlet: bool = True,
        temperature: Optional[float] = None,
        batch_size: int = 8,
    ) -> List[float]:
        """Run n_sims MCTS simulations from `board` using batched inference."""
        self.tree.new_game(board)
        # Dirichlet noise only at the start of a full compound turn, not at
        # intermediate plies (second stone of a 2-stone turn).  Ply 0 is P1's
        # single opening stone — that IS a full turn, so noise applies there.
        is_intermediate_ply = board.moves_remaining == 1 and board.ply > 0
        dirichlet_applied  = is_intermediate_ply  # skip noise if mid-turn
        effective_batch    = max(1, int(batch_size))

        sims_done = 0
        while sims_done < n_sims:
            current_batch = min(effective_batch, n_sims - sims_done)
            try:
                leaves = self.tree.select_leaves(current_batch)
            except BaseException as exc:
                # Native MCTS can occasionally panic during batched leaf
                # reconstruction. Recover by restarting at root in batch_size=1 mode.
                if current_batch > 1 and "cell already occupied" in str(exc):
                    self.tree.new_game(board)
                    dirichlet_applied = False
                    effective_batch   = 1
                    leaves = self.tree.select_leaves(1)
                else:
                    raise

            if not leaves:
                break

            policies, values = self._engine.infer_batch(leaves)
            self.tree.expand_and_backup(policies, values)
            sims_done += current_batch

            # Apply Dirichlet noise to root priors after the first expansion.
            if use_dirichlet and not dirichlet_applied:
                n_ch = self.tree.root_n_children()
                if n_ch > 0:
                    noise = np.random.dirichlet(
                        [self.dirichlet_alpha] * n_ch
                    ).tolist()
                    self.tree.apply_dirichlet_to_root(noise, self.dirichlet_eps)
                    dirichlet_applied = True

        if temperature is None:
            temperature = get_temperature(
                ply=int(board.ply),
                mode="evaluation" if not use_dirichlet else "training",
                config=self.config,
            )

        return self.tree.get_policy(temperature=temperature, board_size=BOARD_SIZE)

    # ── Action sampling ────────────────────────────────────────────────────────

    @staticmethod
    def _sample_action(
        policy: List[float],
        legal_moves: List[Tuple[int, int]],
        board: Board,
    ) -> Tuple[int, int]:
        """Sample a move from the MCTS policy, restricted to legal moves.

        Falls back to uniform sampling if MCTS assigns zero probability to all
        legal moves (degenerate case).
        """
        legal_flat = [board.to_flat(q, r) for q, r in legal_moves]
        probs = np.array(
            [policy[i] if i < N_ACTIONS else 0.0 for i in legal_flat],
            dtype=np.float64,
        )
        total = probs.sum()
        if total < 1e-9:
            probs = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            probs /= total
        idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[idx]

    # ── Game loop ──────────────────────────────────────────────────────────────

    def play_game(
        self, buffer: "ReplayBuffer", use_dirichlet: bool = True
    ) -> Tuple[int, Optional[int]]:
        """Play one complete game and push all positions to `buffer`.

        Args:
            buffer:        Target replay buffer.
            use_dirichlet: True for self-play training games.  False for eval.

        Returns:
            (number of positions added, winner as 1/-1 or None for draw).
        """
        rust_board = Board()
        state      = GameState.from_board(rust_board)
        self._buf.reset()

        records: List[Tuple[np.ndarray, np.ndarray, int]] = []

        # Playout cap randomization: 90% fast search, 10% deep search.
        fast_sims = np.random.randint(15, 26)

        while True:
            if rust_board.check_win() or rust_board.legal_move_count() == 0:
                break

            current_n_sims = self.n_sims
            if use_dirichlet and np.random.random() < 0.9:
                current_n_sims = fast_sims

            mcts_policy = self._run_mcts_with_sims(
                rust_board, n_sims=current_n_sims, use_dirichlet=use_dirichlet,
            )

            # _buf.assemble reuses its array in-place; copy each slice before storing.
            full_tensor, centers = self._buf.assemble(state)
            global_policy = np.array(mcts_policy, dtype=np.float32)
            for k, center in enumerate(centers):
                state_tensor = full_tensor[k].copy()
                policy_arr   = project_global_policy_to_local(
                    rust_board, center, global_policy, board_size=BOARD_SIZE,
                )
                records.append((state_tensor, policy_arr, state.current_player))

            legal = rust_board.legal_moves()
            if not legal:
                break
            q, r  = self._sample_action(mcts_policy, legal, rust_board)
            state = state.apply_move(rust_board, q, r)

        winner = rust_board.winner()

        for state_tensor, policy_arr, player in records:
            if winner is None:
                outcome = 0.0
            else:
                outcome = 1.0 if player == winner else -1.0
            buffer.push(state_tensor, policy_arr, outcome)

        return len(records), winner
