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

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from native_core import Board, MCTSTree
from python.env.game_state import GameState
from python.model.network import HexTacToeNet
from python.training.replay_buffer import ReplayBuffer

# Board is always 19×19 (BOARD_SIZE from native_core)
_BOARD_SIZE: int = 19
_N_ACTIONS:  int = _BOARD_SIZE * _BOARD_SIZE + 1  # 362


def get_temperature(ply: int, mode: str, config: Dict[str, Any]) -> float:
    """Return the MCTS sampling temperature for the current game state.

    Args:
        ply:    Total half-moves played so far (board.ply).
        mode:   "training"   → tau=1.0 for first N plies, tau=0.1 after.
                "evaluation" → tau=0.0 (argmax, deterministic).
                "bootstrap"  → tau=0.5 (moderate, for minimax corpus games).
        config: Config dict.  Reads ``temperature_threshold_ply`` from the
                ``mcts`` sub-dict if present, else top-level, else default 30.

    Returns:
        Sampling temperature as a float.
    """
    if mode == "evaluation":
        return 0.0
    if mode == "bootstrap":
        return 0.5
    # Training mode: ply-based schedule.
    mcts_cfg = config.get("mcts", config)
    threshold = int(mcts_cfg.get("temperature_threshold_ply",
                                  config.get("temperature_threshold_ply", 30)))
    return 1.0 if ply < threshold else 0.1


def _flat_to_coords(flat: int) -> Tuple[int, int]:
    """Convert a flat board index back to (q, r) axial coordinates."""
    half = (_BOARD_SIZE - 1) // 2   # 9
    q = flat // _BOARD_SIZE - half
    r = flat %  _BOARD_SIZE - half
    return q, r


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
        self.model  = model
        self.config = config
        self.device = device

        mcts_cfg = config.get("mcts", config)
        self.n_sims         = int(mcts_cfg.get("n_simulations", config.get("n_simulations", 50)))
        self.c_puct         = float(mcts_cfg.get("c_puct", 1.5))
        self.temp_threshold = int(mcts_cfg.get("temperature_threshold_ply", 30))
        self.dirichlet_alpha = float(mcts_cfg.get("dirichlet_alpha", 0.3))
        self.dirichlet_eps   = float(mcts_cfg.get("epsilon", 0.25))

        self.tree = MCTSTree(c_puct=self.c_puct)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer(self, board: Board) -> Tuple[List[float], float]:
        """Run network inference on a single board.

        Returns (policy_probs, value) where policy_probs is a list of
        length _N_ACTIONS and value ∈ [-1, 1].

        Uses GameState.from_board with no history context (leaf nodes in MCTS
        don't carry full game history). The 8 history planes are zero except
        for the current board at position 7.
        """
        state = GameState.from_board(board)
        tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(self.device)

        local_tensor = tensor[:, :18, :, :]
        global_tensor = tensor[:, 18:, :, :]

        self.model.eval()
        log_policy, value = self.model(local_tensor.float(), global_tensor.float())

        policy_probs = log_policy.exp().squeeze(0).cpu().numpy().tolist()
        v = value.squeeze().item()
        return policy_probs, v

    # ── MCTS search ───────────────────────────────────────────────────────────

    def _run_mcts(
        self,
        board: Board,
        use_dirichlet: bool = True,
        temperature: Optional[float] = None,
    ) -> List[float]:
        """Run self.n_sims MCTS simulations from `board`.

        Args:
            board:          Current board state (root of search).
            use_dirichlet:  If True, mix Dirichlet noise into root priors after
                            the first expansion (self-play mode).  Set False for
                            evaluation games.
            temperature:    Override sampling temperature.  If None, uses the
                            ply-based schedule (tau=1.0 early, tau=0.1 late).

        Returns:
            Visit-count policy vector (length _N_ACTIONS).
        """
        self.tree.new_game(board)
        dirichlet_applied = False

        for sim_idx in range(self.n_sims):
            leaves = self.tree.select_leaves(1)
            if not leaves:
                break
            policy, value = self._infer(leaves[0])
            self.tree.expand_and_backup([policy], [value])

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

        return self.tree.get_policy(temperature=temperature, board_size=_BOARD_SIZE)

    # ── Action sampling ───────────────────────────────────────────────────────

    @staticmethod
    def _sample_action(policy: List[float], legal_moves: List[Tuple[int, int]], board: "Board") -> Tuple[int, int]:
        """Sample a move from the MCTS policy, restricted to legal moves.

        If MCTS assigns zero probability to all legal moves (degenerate case),
        falls back to uniform sampling.
        """
        n_actions = _BOARD_SIZE * _BOARD_SIZE + 1

        # Map legal moves to window-relative flat indices (matches MCTS policy vector)
        legal_flat = [board.to_flat(q, r) for q, r in legal_moves]

        # Extract and normalise probabilities over legal moves
        probs = np.array([policy[i] if i < n_actions else 0.0 for i in legal_flat],
                         dtype=np.float64)
        total = probs.sum()
        if total < 1e-9:
            probs = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            probs /= total

        idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[idx]

    # ── Game loop ─────────────────────────────────────────────────────────────

    def play_game(self, buffer: ReplayBuffer, use_dirichlet: bool = True) -> Tuple[int, Optional[int]]:
        """Play one complete game and push all positions to `buffer`.

        Args:
            buffer:        Target replay buffer.
            use_dirichlet: True for self-play training games (adds exploration
                           noise at root). False for evaluation games.

        Returns:
            Tuple of (number of positions added, winner).
            Winner is 1, -1, or None (draw).
        """
        rust_board = Board()
        state      = GameState.from_board(rust_board)

        # Collect (tensor, mcts_policy) for each step; outcome unknown until game end.
        records: List[Tuple[np.ndarray, np.ndarray, int]] = []
        # Each record: (state_tensor, mcts_policy, player_at_this_move)

        # Playout cap randomization (only for self-play)
        # 90% fast search, 10% deep search
        fast_sims = np.random.randint(15, 26) # 15-25 sims

        while True:
            # ── Win / draw check ──
            if rust_board.check_win() or rust_board.legal_move_count() == 0:
                break

            # ── Playout cap randomization ──
            current_n_sims = self.n_sims
            if use_dirichlet: # self-play
                if np.random.random() < 0.9:
                    current_n_sims = fast_sims
            
            # ── MCTS search ──
            mcts_policy = self._run_mcts_with_sims(rust_board, n_sims=current_n_sims, use_dirichlet=use_dirichlet)

            # ── Record position ──
            state_tensor = state.to_tensor()           # (18, 19, 19) float16
            policy_arr   = np.array(mcts_policy, dtype=np.float32)
            records.append((state_tensor, policy_arr, state.current_player))

            # ── Sample and apply move ──
            legal = rust_board.legal_moves()
            if not legal:
                break
            q, r = self._sample_action(mcts_policy, legal, rust_board)
            state = state.apply_move(rust_board, q, r)

        # ── Determine outcome ──
        # check_win() is True if the player who JUST moved won.
        # current_player at game end is the player who will move NEXT (the loser).
        winner = rust_board.winner()  # 1, -1, or None (draw)

        # Push to buffer: outcome from each player's own perspective.
        for state_tensor, policy_arr, player in records:
            if winner is None:
                outcome = 0.0
            else:
                outcome = 1.0 if player == winner else -1.0
            buffer.push(state_tensor, policy_arr, outcome)

        return len(records), winner

    def _run_mcts_with_sims(
        self,
        board: Board,
        n_sims: int,
        use_dirichlet: bool = True,
        temperature: Optional[float] = None,
    ) -> List[float]:
        """Run n_sims MCTS simulations from `board`."""
        self.tree.new_game(board)
        dirichlet_applied = False

        for _ in range(n_sims):
            leaves = self.tree.select_leaves(1)
            if not leaves:
                break
            policy, value = self._infer(leaves[0])
            self.tree.expand_and_backup([policy], [value])

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

        return self.tree.get_policy(temperature=temperature, board_size=_BOARD_SIZE)
