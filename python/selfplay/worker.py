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
from python.env.game_state import GameState, HISTORY_LEN
from python.model.network import HexTacToeNet
from python.selfplay.policy_projection import project_global_policy_to_local
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

        # Rolling tensor buffer for the game loop (avoids per-step np.zeros + history rebuild).
        # Reset at the start of each play_game call.
        self._game_buf: Optional[np.ndarray] = None  # shape (K, 18, 19, 19) float16
        self._game_K: int = 0

    # ── Rolling tensor buffer ─────────────────────────────────────────────────

    def _assemble_tensor(
        self, state: GameState
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Assemble the (K, 18, 19, 19) float16 tensor using a pre-allocated rolling buffer.

        On first call (or when K changes), allocates and fills from scratch.
        On subsequent calls with the same K, performs an in-place circular shift
        of the history planes and writes only the new current planes 0 and 8,
        avoiding both np.zeros allocation and the 7-step history loop.
        """
        K = len(state.views)
        centers = state.centers

        if self._game_buf is None or K != self._game_K:
            # Full rebuild: new game or cluster count changed.
            self._game_K = K
            buf = np.zeros((K, 18, _BOARD_SIZE, _BOARD_SIZE), dtype=np.float16)
            history = state.move_history
            for k in range(K):
                for t in range(1, HISTORY_LEN):
                    if t > len(history):
                        break
                    prior = history[-t]
                    if k < len(prior.views):
                        buf[k, t]     = prior.views[k][0]
                        buf[k, 8 + t] = prior.views[k][1]
            self._game_buf = buf
        else:
            # Circular shift: push all history planes one step older.
            # planes 1..7 ← 0..6  (my-stones history)
            # planes 9..15 ← 8..14 (opp-stones history)
            buf = self._game_buf
            buf[:, 1:8, :, :] = buf[:, 0:7, :, :]
            buf[:, 9:16, :, :] = buf[:, 8:15, :, :]

        # Write current-timestep planes (always overwritten).
        for k in range(K):
            buf[k, 0] = state.views[k][0]
            buf[k, 8] = state.views[k][1]

        # Scalar planes (broadcast across K and spatial dims).
        buf[:, 16, :, :] = np.float16(0.0 if state.moves_remaining == 1 else 1.0)
        buf[:, 17, :, :] = np.float16(float(state.ply % 2))

        return buf, centers

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
            tensor, centers = state.to_tensor()  # uses cached self.views, no second get_cluster_views
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
        # Reset rolling buffer for new game.
        self._game_buf = None
        self._game_K = 0

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
            # _assemble_tensor reuses the rolling buffer in-place, so we must
            # copy each cluster slice before storing (buffer is overwritten next step).
            full_tensor, centers = self._assemble_tensor(state)
            global_policy = np.array(mcts_policy, dtype=np.float32)
            for k, center in enumerate(centers):
                state_tensor = full_tensor[k].copy()
                policy_arr = project_global_policy_to_local(
                    rust_board,
                    center,
                    global_policy,
                    board_size=_BOARD_SIZE,
                )
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
        effective_batch_size = max(1, int(batch_size))

        # Run simulations in batches.
        sims_done = 0
        while sims_done < n_sims:
            current_batch = min(effective_batch_size, n_sims - sims_done)
            try:
                leaves = self.tree.select_leaves(current_batch)
            except BaseException as exc:
                # Native MCTS can occasionally panic during batched leaf
                # reconstruction. Recover by restarting search at root and
                # continuing in conservative batch_size=1 mode.
                if current_batch > 1 and "cell already occupied" in str(exc):
                    self.tree.new_game(board)
                    dirichlet_applied = False
                    effective_batch_size = 1
                    leaves = self.tree.select_leaves(1)
                else:
                    raise
            if not leaves:
                break
            
            # Batch inference on all selected leaves
            policies, values = self._infer_batch(leaves)
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

        return self.tree.get_policy(temperature=temperature, board_size=_BOARD_SIZE)
