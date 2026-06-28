"""D-DECODE Workflow-2 — MULTI-WINDOW no-drop Gumbel-SH deploy bot.

The DECISIVE-test bot. Marries:
  * KClusterMCTSBot's MULTI-WINDOW no-drop action space — rust_board.get_cluster_views()
    -> K views; value pool = min across K; policy pool = scatter-max across K onto the
    FULL legal-move set (off-window cells representable throughout the tree). (Inherited
    verbatim from KClusterMCTSBot: _expand, _aggregate_priors, _puct_select, _Node.)
  * g=0 Gumbel Sequential-Halving ROOT decoding — candidate set = top-m=16 by aggregated
    log_prior (g=0 zeroes the Gumbel noise so candidate select = argsort(-prior)), SH
    visit allocation, PUCT-interior descent below the forced root child, completed-Q
    sigma, SH winner = argmax over survivors of (log_prior + sigma * completed_q).

The SH math MIRRORS hexo_rl/eval/gumbel_search_py.run_gumbel_on_board (the single-window
Rust-tree reference) but runs over the multi-window Python _Node tree instead of the
v6-locked Rust MCTSTree. Deploy knobs: m=16, n_sims=150, c_visit=50, c_scale=1, c_puct=1.5.

This is INFERENCE-ONLY investigation code (no retrain, no engine rebuild). It is the
multi-window analog of DeployHeadBot: same g=0 deterministic SH winner, but the candidate
set + interior descent live in the no-drop multi-window action space rather than the
single-window Rust tree that has no logit for off-window cells.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot, _Node, _puct_select

Action = Tuple[int, int]


class MultiWindowGumbelSHBot(KClusterMCTSBot):
    """Multi-window no-drop Gumbel-SH deploy head (g=0 deterministic).

    Inherits the no-drop multi-window machinery from ``KClusterMCTSBot`` (``_expand``
    builds children over the FULL legal set via scatter-max priors + min value pool;
    off-window cells get a prior and a child node). Overrides ``get_move`` to decode the
    played move via g=0 Gumbel Sequential-Halving over the root candidate set instead of
    the visit-count argmax.

    Args mirror ``KClusterMCTSBot`` plus the Gumbel-SH knobs:
        gumbel_m: candidate cap m (deploy 16).
        c_visit, c_scale: completed-Q sigma scale (deploy 50.0 / 1.0).
        gumbel_scale: root Gumbel(0,1) scale. 0.0 = deterministic deploy head (the
            canonical strength-eval / deploy convention; mctx gumbel_scale=0).
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_sims: int = 150,
        c_puct: float = 1.5,
        gumbel_m: int = 16,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        gumbel_scale: float = 0.0,
        fpu_q: float = 0.0,
        kept_plane_indices: Optional[list[int]] = None,
        seed: int = 0,
    ) -> None:
        # temperature is unused (SH winner, not visit softmax) — pass 0.0 to the base.
        super().__init__(
            model, device, n_sims=n_sims, c_puct=c_puct, temperature=0.0,
            fpu_q=fpu_q, kept_plane_indices=kept_plane_indices,
        )
        self.gumbel_m = int(gumbel_m)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self.gumbel_scale = float(gumbel_scale)
        self._rng = np.random.default_rng(seed)

    def name(self) -> str:
        return f"mw_gumbel_sh(m{self.gumbel_m},n{self.n_sims})"

    # ── one forced-root-child simulation (PUCT-interior descent below the candidate) ──
    def _simulate_forced(self, root: _Node, root_board, forced_action: Action) -> None:
        """One MCTS sim whose FIRST edge is forced to ``forced_action`` (a root
        candidate), then PUCT-descends below it. Mirrors the Rust ``forced_root_child``
        single-window descent, but in the multi-window Python tree.

        Backup is byte-identical to ``KClusterMCTSBot._simulate``: ``value`` is the
        leaf-current-player perspective; walking up, flip iff ``path[i-1].child_flips``.
        """
        sim_board = root_board.clone()
        # Forced first edge: root -> candidate child.
        sim_board.apply_move(*forced_action)
        node = root.children[forced_action]
        path: List[_Node] = [root, node]

        # PUCT-interior descent below the forced child.
        while node.expanded() and not node.is_terminal and node.children:
            action = _puct_select(node, self.c_puct, self.fpu_q)
            sim_board.apply_move(*action)
            node = node.children[action]
            path.append(node)

        if node.is_terminal:
            value = node.terminal_value
        else:
            value = self._expand(node, sim_board)

        for i in range(len(path) - 1, -1, -1):
            n = path[i]
            n.visits += 1
            n.value_sum += value
            if i > 0 and path[i - 1].child_flips:
                value = -value

    # ── Gumbel-SH helpers (parity with run_gumbel_on_board) ──────────────────────────
    @staticmethod
    def _root_q_hat(root: _Node, child: _Node) -> float:
        """Root-perspective completed-Q of a root child, clamped to [-1, 1].

        Mirrors ``_puct_select``'s Q sign convention: root-frame child Q =
        (-1 if root.child_flips else 1) * child.value(); 0.0 for an unvisited child
        (run_gumbel_on_board uses q_hat=0.0 when visits==0)."""
        if child.visits == 0:
            return 0.0
        sign = -1.0 if root.child_flips else 1.0
        q = sign * child.value()
        return max(-1.0, min(1.0, q))

    def _score_candidates(
        self,
        root: _Node,
        candidates: List[Action],
        gumbels: Dict[Action, float],
        log_priors: Dict[Action, float],
        max_n: int,
    ) -> List[Tuple[Action, float]]:
        out: List[Tuple[Action, float]] = []
        for a in candidates:
            child = root.children[a]
            q_hat = self._root_q_hat(root, child)
            sigma = (self.c_visit + max_n) * self.c_scale * q_hat
            out.append((a, float(gumbels[a] + log_priors[a] + sigma)))
        return out

    @torch.no_grad()
    def get_move(self, state, rust_board) -> Action:
        # ── Root expansion (one batched K-cluster forward) ──────────────────────────
        root = _Node(prior=0.0)
        self._expand(root, rust_board)
        if root.is_terminal:
            raise RuntimeError("MultiWindowGumbelSHBot: root is terminal; no move")

        actions = list(root.children.keys())
        n_children = len(actions)
        if n_children == 0:
            legal = rust_board.legal_moves()
            return (int(legal[0][0]), int(legal[0][1]))

        # log-priors over the FULL (multi-window) legal set.
        log_priors: Dict[Action, float] = {
            a: float(math.log(max(root.children[a].prior, 1e-8))) for a in actions
        }
        # Gumbel(0,1) root noise * scale. gumbel_scale=0.0 -> all zero (deterministic).
        if self.gumbel_scale != 0.0:
            u = self._rng.uniform(1e-10, 1.0 - 1e-7, size=n_children)
            g = self.gumbel_scale * (-np.log(-np.log(u)))
        else:
            g = np.zeros(n_children)
        gumbels: Dict[Action, float] = {a: float(g[i]) for i, a in enumerate(actions)}

        # ── candidate set = top-eff_m by (gumbel + log_prior) ───────────────────────
        eff_m = min(int(self.gumbel_m), int(self.n_sims), n_children)  # effective_m parity
        order = sorted(actions, key=lambda a: -(gumbels[a] + log_priors[a]))
        candidates: List[Action] = order[:eff_m]

        # Sanity asserts (per the prompt): candidate count == min(16, n_legal).
        assert len(candidates) == min(self.gumbel_m, self.n_sims, n_children), (
            f"candidate count {len(candidates)} != min(m,n_sims,n_legal)"
        )

        num_phases = 1 if eff_m <= 1 else math.ceil(math.log2(eff_m))
        budget = max(0, int(self.n_sims) - 1)  # root expansion ate one forward
        sims_used = 0

        def _max_n_all() -> int:
            return max((c.visits for c in root.children.values()), default=0)

        for phase in range(num_phases):
            if sims_used >= budget:
                break
            remaining = budget - sims_used
            remaining_phases = num_phases - phase
            sims_per = max(1, remaining // (remaining_phases * len(candidates)))
            for a in candidates:
                if sims_used >= budget:
                    break
                n_to_run = min(sims_per, budget - sims_used)
                for _ in range(n_to_run):
                    self._simulate_forced(root, rust_board, a)
                    sims_used += 1
            if len(candidates) <= 1:  # break at BOTTOM (inner.rs:802 parity)
                break
            scored = self._score_candidates(root, candidates, gumbels, log_priors, _max_n_all())
            scored.sort(key=lambda t: t[1], reverse=True)
            keep = (len(scored) + 1) // 2  # div_ceil(2) parity
            candidates = [a for a, _ in scored[:keep]]

        # ── SH winner = argmax score over surviving candidates ──────────────────────
        final = self._score_candidates(root, candidates, gumbels, log_priors, _max_n_all())
        winner = max(final, key=lambda t: t[1])[0] if final else candidates[0]
        return (int(winner[0]), int(winner[1]))


__all__ = ["MultiWindowGumbelSHBot"]
