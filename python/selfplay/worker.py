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
        policies, values = self._infer_batch([board])
        return policies[0], values[0]

    @torch.no_grad()
    def _infer_batch(self, boards: List[Board]) -> Tuple[List[List[float]], List[float]]:
        if not boards:
            return [], []
            
        all_tensors = []
        board_info = [] # (K, centers)
        
        for board in boards:
            state = GameState.from_board(board)
            tensor, centers = state.to_tensor(board)
            all_tensors.append(torch.from_numpy(tensor))
            board_info.append((len(centers), centers))
            
        # Concatenate all clusters into one large batch for the network
        batch_tensor = torch.cat(all_tensors, dim=0).to(self.device)
        
        self.model.eval()
        # Use float16 for inference speedup if on CUDA
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            log_policy, value = self.model(batch_tensor.float())
        
        policies_np = log_policy.exp().cpu().float().numpy() # (TotalK, 362)
        values_np = value.squeeze(-1).cpu().float().numpy()  # (TotalK,)
        
        results_p = []
        results_v = []
        
        curr = 0
        for i, board in enumerate(boards):
            K, centers = board_info[i]
            
            # Extract sub-batch for this board
            board_policies = policies_np[curr:curr+K]
            board_values = values_np[curr:curr+K]
            curr += K
            
            # Aggregate value (pessimistic)
            v = float(board_values.min())
            
            # Map local cluster policies to global policy vector
            n_actions = _BOARD_SIZE * _BOARD_SIZE + 1
            global_policy = np.zeros(n_actions, dtype=np.float64)
            legal = board.legal_moves()
            
            for q, r in legal:
                mcts_idx = board.to_flat(q, r)
                if mcts_idx >= n_actions - 1:
                    continue
                    
                max_prob = 0.0
                for k, (cq, cr) in enumerate(centers):
                    wq = q - cq + (_BOARD_SIZE - 1) // 2
                    wr = r - cr + (_BOARD_SIZE - 1) // 2
                    if 0 <= wq < _BOARD_SIZE and 0 <= wr < _BOARD_SIZE:
                        local_idx = wq * _BOARD_SIZE + wr
                        if board_policies[k, local_idx] > max_prob:
                            max_prob = board_policies[k, local_idx]
                
                global_policy[mcts_idx] = max_prob
                
            total = global_policy.sum()
            if total > 1e-9:
                global_policy /= total
            else:
                global_policy.fill(1.0 / n_actions)
                
            results_p.append(global_policy.tolist())
            results_v.append(v)
            
        return results_p, results_v

    # ── MCTS search ───────────────────────────────────────────────────────────

    def _run_mcts(
        self,
        board: Board,
        use_dirichlet: bool = True,
        temperature: Optional[float] = None,
    ) -> List[float]:
        return self._run_mcts_with_sims(board, n_sims=self.n_sims, use_dirichlet=use_dirichlet, temperature=temperature)

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
            state_tensor = state.to_tensor(rust_board)[0][0] # just store one for replay for now           # (18, 19, 19) float16
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
        batch_size: int = 8,
    ) -> List[float]:
        """Run n_sims MCTS simulations from `board` using batched inference."""
        self.tree.new_game(board)
        dirichlet_applied = False

        # Run simulations in batches
        for sim_idx in range(0, n_sims, batch_size):
            current_batch = min(batch_size, n_sims - sim_idx)
            leaves = self.tree.select_leaves(current_batch)
            if not leaves:
                break
            
            # Batch inference on all selected leaves
            policies, values = self._infer_batch(leaves)
            self.tree.expand_and_backup(policies, values)

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
