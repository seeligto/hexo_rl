"""Local inference engine: batches boards through the network and returns
global policy vectors + min-pooled scalar values.

§172 A4.2 — registry-driven encoding plumbing. `LocalInferenceEngine` is
the Python-side fallback inference path used by `OurModelBot` (eval/bot
play) and any callers that do not go through the Rust `SelfPlayRunner`.
It is multi-window aware via `state.to_tensor()` which returns a list of
cluster centers; for single-window encodings (v6, v7full) the centers
loop runs with K=1 and degenerates to the trivial mapping.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from engine import Board
from hexo_rl.encoding import EncodingSpec, lookup
from hexo_rl.env.game_state import GameState
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.utils.constants import KEPT_PLANE_INDICES


class LocalInferenceEngine:
    """Wraps a HexTacToeNet and handles the full inference pipeline:

    1. Build (K, C, board_size, board_size) tensors for a batch of boards
       (C = model.in_channels).
    2. Run a single forward pass.
    3. Map per-cluster local policy outputs → one global policy vector per board.
    4. Aggregate per-cluster values via min-pooling.

    Registry-driven (§172 A4.2): caller passes `encoding_spec`; defaults to
    v6 for backward compat with legacy eval / bot call sites.
    """

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        encoding_spec: Optional[EncodingSpec] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.encoding_spec: EncodingSpec = (
            encoding_spec if encoding_spec is not None else lookup("v6")
        )

    @torch.inference_mode()
    def infer(self, board: Board) -> Tuple[List[float], float]:
        """Single-board convenience wrapper around ``infer_batch``."""
        policies, values = self.infer_batch([board])
        return policies[0], values[0]

    @torch.inference_mode()
    def infer_batch(self, boards: List[Board]) -> Tuple[List[List[float]], List[float]]:
        """Run inference on a list of boards.

        Returns:
            policies: List of global policy vectors (length spec.policy_logit_count each).
            values:   List of scalar values, one per board (min-pooled over clusters).
        """
        if not boards:
            return [], []

        spec = self.encoding_spec
        board_size = spec.board_size
        n_actions = spec.policy_logit_count
        half = (board_size - 1) // 2

        all_tensors = []
        board_info: List[Tuple[int, List[Tuple[int, int]]]] = []

        for board in boards:
            state = GameState.from_board(board)
            tensor, centers = state.to_tensor()
            if tensor.shape[1] != self.model.in_channels:
                tensor = tensor[:, KEPT_PLANE_INDICES]
            all_tensors.append(torch.from_numpy(tensor))
            board_info.append((len(centers), centers))

        # Single batched forward pass over all clusters from all boards.
        batch_tensor = torch.cat(all_tensors, dim=0).to(self.device)

        self.model.eval()
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type in ("cuda", "mps"))):
            log_policy, value, _v_logit = self.model(batch_tensor.float())

        policies_np = log_policy.exp().cpu().float().numpy()  # (TotalK, n_actions)
        values_np   = value.squeeze(-1).cpu().float().numpy()  # (TotalK,)

        results_p: List[List[float]] = []
        results_v: List[float] = []

        cursor = 0
        for i, board in enumerate(boards):
            K, centers = board_info[i]
            board_policies = policies_np[cursor:cursor + K]
            board_values   = values_np[cursor:cursor + K]
            cursor += K

            # Min-pool over clusters: treat the worst window as the board value.
            v = float(board_values.min())

            # Map each legal move to the highest probability across all windows.
            global_policy = np.zeros(n_actions, dtype=np.float64)
            for q, r in board.legal_moves():
                mcts_idx = board.to_flat(q, r)
                if mcts_idx >= n_actions - 1:
                    continue
                max_prob = 0.0
                for k, (cq, cr) in enumerate(centers):
                    wq = q - cq + half
                    wr = r - cr + half
                    if 0 <= wq < board_size and 0 <= wr < board_size:
                        local_idx = wq * board_size + wr
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
