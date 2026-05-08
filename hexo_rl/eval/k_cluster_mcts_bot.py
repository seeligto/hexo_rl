"""KClusterMCTSBot — Python PUCT MCTS for v6 / v6w25 (K-cluster) models.

Sister to ``V8MCTSBot``. Drives MCTS for K-cluster encodings whose action
space (362 for v6 / 626 for v6w25) and per-position cluster count K make
the v6-locked Rust ``MCTSTree`` (BOARD_SIZE=19, N_ACTIONS=362) unusable
for v6w25 and any future K-cluster window > 19. v7full / v6 still has
the option of the Rust MCTS path; this Python bot is wired in for
v6w25 + matched-MCTS comparison work (§169 A1/A2/A3).

NN forward
----------
For each leaf: ``rust_board.get_cluster_views()`` → ``K`` views of shape
``(2, S, S)``; ``GameState.from_board`` + ``to_tensor`` lifts that to
``(K, 18, S, S)`` (history planes 1-7 / 9-15 zero — same convention as
``V6ArgmaxBot``); slice to ``KEPT_PLANE_INDICES`` for 8-plane v6 wire
format. The K planes are batched into one model forward (single GPU call
per leaf) so per-leaf wall is bounded by K rather than K × per-position
forward latency.

Aggregation (matches engine ``records::aggregate_policy`` and
``worker_loop.rs:299-401`` semantics)
- value pool: ``min`` across K clusters (negamax-conservative).
- policy pool: scatter-max across K clusters (per-legal-move max prob),
  renormalised over the legal set.

PUCT (matches engine ``mcts/mod.rs``)
- ``Q + c_puct · prior · √N(s) / (1 + N(s,a))``.
- Q from parent's perspective (sign flipped at turn boundaries — HTTT
  plays 2 stones per turn, so child shares parent's perspective when
  ``parent.moves_remaining == 2``; flips when ``moves_remaining == 1``).
- Virtual loss: OFF (sequential simulations — see // TODO below).
- FPU: classical Q=0 for unvisited (configurable via ``fpu_q``).
- Dirichlet noise: OFF (eval-only).

Sequential leaf scheduling
- // TODO(P2): leaf-batched descent with virtual loss. K-batched per
  leaf already amortises K × NN forwards into one GPU call (typical
  K=2-5, so ~3× over a strictly-sequential per-cluster forward), but
  inter-leaf batching (B=8-32) would cut wall by another ~3-5× and is
  needed if MCTS-N evals run > 12 hr at the chosen N.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env.game_state import GameState
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.utils.constants import KEPT_PLANE_INDICES
from hexo_rl.utils.global_crop import compute_global_crop_from_board


Action = Tuple[int, int]

SUPPORTED_POOLS = ("min_max", "pma", "pma_global")


class _Node:
    __slots__ = (
        "prior",
        "visits",
        "value_sum",
        "children",
        "is_terminal",
        "terminal_value",
        "child_flips",
    )

    def __init__(self, prior: float = 0.0) -> None:
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[Action, "_Node"] = {}
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0
        # Whether children's value perspective is opposite this node's. Set at
        # expand time from sim_board.moves_remaining; constant for all children
        # (HTTT turn structure makes it a node-level property).
        self.child_flips: bool = False

    def expanded(self) -> bool:
        return bool(self.children) or self.is_terminal

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _puct_select(node: _Node, c_puct: float, fpu_q: float) -> Action:
    """Pick the child action by max Q + U (PUCT). Q in parent's frame."""
    sign = -1.0 if node.child_flips else 1.0
    total = max(1, node.visits)
    sqrt_total = math.sqrt(total)
    best_score = -math.inf
    best_action: Optional[Action] = None
    for action, child in node.children.items():
        if child.visits == 0:
            q = fpu_q
        else:
            q = sign * child.value()
        u = c_puct * child.prior * sqrt_total / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
    assert best_action is not None
    return best_action


def _aggregate_priors(
    log_policy_K: np.ndarray,  # (K, S*S+1)
    centers: List[Tuple[int, int]],
    legal_moves: List[Action],
    view_size: int,
) -> List[float]:
    """Scatter-max across K cluster log-policies onto the legal-move set.

    Mirrors engine ``records::aggregate_policy`` (max over clusters of
    per-cluster softmax-prob, renormalise over legal). Done in prob space
    so different clusters' log-prob shifts compose correctly.
    """
    K = log_policy_K.shape[0]
    assert K == len(centers), "centers and policy K must match"
    view_half = (view_size - 1) // 2
    # Per-cluster softmax: shift by max, exp, normalise.
    probs_K = np.empty_like(log_policy_K, dtype=np.float64)
    for k in range(K):
        lp = log_policy_K[k].astype(np.float64)
        lp -= lp.max()
        e = np.exp(lp)
        s = e.sum()
        if s < 1e-12:
            probs_K[k] = 1.0 / lp.shape[0]
        else:
            probs_K[k] = e / s

    n_legal = len(legal_moves)
    max_probs = np.zeros(n_legal, dtype=np.float64)
    for i, (q, r) in enumerate(legal_moves):
        best = 0.0
        for k, (cq, cr) in enumerate(centers):
            wq = q - cq + view_half
            wr = r - cr + view_half
            if 0 <= wq < view_size and 0 <= wr < view_size:
                p = probs_K[k][wq * view_size + wr]
                if p > best:
                    best = p
        max_probs[i] = best

    s = max_probs.sum()
    if s < 1e-12:
        return [1.0 / n_legal] * n_legal
    return (max_probs / s).tolist()


def _aggregate_priors_pma(
    log_policy_agg: np.ndarray,            # (1, S*S+1)
    canonical_center: Tuple[int, int],
    legal_moves: List[Action],
    view_size: int,
) -> List[float]:
    """Read the PMA-aggregated single-cluster policy onto the legal-move set.

    The PMA pool returns one log-policy in cluster-0's frame. Each legal
    move (q, r) maps onto cluster 0 at (wq, wr) = (q-cq+half, r-cr+half).
    Moves whose mapping falls outside the [0, view_size) window are assigned
    a small floor (1e-6 then renormalised) — this is rare for the centermost
    cluster but possible at board edges. ``replacing scatter-max``: the bot
    no longer takes max-prob across K clusters; the aggregation lives inside
    PMA's attention pool.
    """
    view_half = (view_size - 1) // 2
    cq, cr = canonical_center
    lp = log_policy_agg[0].astype(np.float64)
    lp -= lp.max()
    e = np.exp(lp)
    s = e.sum()
    probs = e / s if s > 1e-12 else np.full_like(lp, 1.0 / lp.shape[0])

    n_legal = len(legal_moves)
    raw = np.zeros(n_legal, dtype=np.float64)
    for i, (q, r) in enumerate(legal_moves):
        wq = q - cq + view_half
        wr = r - cr + view_half
        if 0 <= wq < view_size and 0 <= wr < view_size:
            raw[i] = probs[wq * view_size + wr]
        else:
            raw[i] = 1e-6
    s = raw.sum()
    if s < 1e-12:
        return [1.0 / n_legal] * n_legal
    return (raw / s).tolist()


class KClusterMCTSBot(BotProtocol):
    """PUCT MCTS over a v6 / v6w25 K-cluster NN policy + value head.

    Args:
        model: ``HexTacToeNet`` with encoding ∈ {'v6', 'v6w25'}.
        device: torch device.
        n_sims: simulations per move.
        c_puct: PUCT exploration constant (engine default 1.5).
        temperature: 0.0 = argmax visit count; >0 = visit-count softmax.
        pool_type: cluster-pool aggregation. 'min_max' (value min, policy
            scatter-max), 'pma' (§169 A2 — Set-Transformer pool over the K
            cluster tokens), or 'pma_global' (§169 A3 — pma + global summary
            token built from a 32×32 cur/opp/canvas-mask crop).
        fpu_q: First-Play-Urgency Q for unvisited children. Default 0.0
            (classical FPU; matches V8MCTSBot). Engine uses 0.25-decayed
            dynamic FPU — set ``fpu_q`` to taste.
    """

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        n_sims: int = 64,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        pool_type: Optional[str] = None,
        fpu_q: float = 0.0,
    ) -> None:
        encoding = getattr(model, "encoding", None)
        if encoding not in ("v6", "v6w25"):
            raise ValueError(
                f"KClusterMCTSBot requires a v6/v6w25 model; got encoding={encoding!r}"
            )
        # Default pool_type to model.pool_type if the caller didn't specify;
        # this lets the eval dispatcher stay encoding-only and pool_type
        # ride along on the checkpoint config.
        if pool_type is None:
            pool_type = getattr(model, "pool_type", "min_max")
        if pool_type not in SUPPORTED_POOLS:
            raise ValueError(
                f"unknown pool_type={pool_type!r}; supported: {SUPPORTED_POOLS}"
            )
        # Cross-check: if the model says 'pma' the bot must also be 'pma' so
        # the K-aggregation goes through model.aggregated_forward_K. A mismatch
        # would silently apply scatter-max on top of PMA's already-aggregated
        # output — surface it loudly.
        model_pool = getattr(model, "pool_type", "min_max")
        if model_pool != pool_type:
            raise ValueError(
                f"KClusterMCTSBot pool_type={pool_type!r} disagrees with "
                f"model.pool_type={model_pool!r}; the bot must match the "
                f"model so the K-aggregation site is consistent."
            )
        self.model = model.eval()
        self.device = device
        self.n_sims = int(n_sims)
        self.c_puct = float(c_puct)
        self.temperature = float(temperature)
        self.pool_type = pool_type
        self.fpu_q = float(fpu_q)
        self._slice_to_8 = self.model.in_channels == 8

    def reset(self) -> None:
        # K-cluster v6/v6w25 models read no Python history (planes 1-7 /
        # 9-15 stay zero in V6ArgmaxBot's path); nothing to clear.
        return

    def name(self) -> str:
        return f"k_cluster_mcts{self.n_sims}"

    @torch.no_grad()
    def _forward_K(
        self, tensor_K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single batched forward over K cluster windows.

        ``tensor_K`` shape: ``(K, C, S, S)`` where C is the model's
        ``in_channels`` (8 for v6/v6w25 wire format).
        Returns ``(log_policy_K, values_K)`` as numpy.
        """
        x = torch.from_numpy(tensor_K).float().to(self.device)
        log_policy, value, _v_logit = self.model(x)
        return (
            log_policy.cpu().numpy(),
            value.cpu().numpy().reshape(-1),
        )

    def _expand(self, node: _Node, sim_board) -> float:
        """Expand ``node`` from a leaf state via batched K-cluster NN forward.

        Returns the value (from current player's perspective).
        """
        if sim_board.check_win():
            # Last move was by the OTHER player; current player just lost.
            node.is_terminal = True
            node.terminal_value = -1.0
            return -1.0

        legal_moves = list(sim_board.legal_moves())
        if not legal_moves:
            node.is_terminal = True
            node.terminal_value = 0.0
            return 0.0

        node.child_flips = (int(sim_board.moves_remaining) == 1)

        state = GameState.from_board(sim_board)
        tensor, centers = state.to_tensor()  # (K, 18, S, S) float16
        if self._slice_to_8:
            tensor = tensor[:, KEPT_PLANE_INDICES]
        K, _, view_h, view_w = tensor.shape
        assert view_h == view_w, "KClusterMCTSBot expects square cluster window"
        view_size = view_h

        if self.pool_type in ("pma", "pma_global"):
            # PMA pool: model aggregates K cluster tokens internally and
            # returns a single (1, n_actions) policy + (1, 1) value. The
            # aggregated policy is emitted in cluster-0's frame (the
            # centermost cluster — the model's natural spatial reference
            # under v6w25 training where target_k was the move's cluster).
            #
            # pma_global also feeds a (3, 32, 32) global summary crop
            # computed from the live board's stones in the current-player's
            # frame. The crop's canvas-mask plane carries the active-bbox
            # indicator so the GlobalTokenEncoder's KataGo gpool ignores
            # padding cells (T2 §E.1 pitfall 2 fix).
            x = torch.from_numpy(tensor).float().to(self.device)
            agg_kwargs: Dict = {}
            if self.pool_type == "pma_global":
                gc_np = compute_global_crop_from_board(sim_board)   # (3, 32, 32) f16
                agg_kwargs["global_crop"] = (
                    torch.from_numpy(gc_np).float().to(self.device)
                )
            log_p_agg, value_agg, _ = self.model.aggregated_forward_K(x, **agg_kwargs)
            value = float(value_agg.cpu().numpy().reshape(-1)[0])
            priors = _aggregate_priors_pma(
                log_p_agg.cpu().numpy(),               # (1, n_actions)
                centers[0],                             # cluster-0 frame
                legal_moves,
                view_size,
            )
        else:
            log_p_K, values_K = self._forward_K(tensor)
            # Pool — value: min across K (negamax-conservative); policy:
            # scatter-max across K, renormalised over legal.
            value = float(values_K.min())
            priors = _aggregate_priors(
                log_p_K, list(centers), legal_moves, view_size
            )

        for action, p in zip(legal_moves, priors):
            node.children[action] = _Node(prior=p)
        return value

    def _simulate(self, root: _Node, root_board) -> None:
        """One MCTS simulation: descend → expand leaf → backup with HTTT
        turn-aware sign flipping."""
        node = root
        path: List[_Node] = [node]
        # sim_board: clone the live board so this sim mutates freely.
        sim_board = root_board.clone()

        while node.expanded() and not node.is_terminal and node.children:
            action = _puct_select(node, self.c_puct, self.fpu_q)
            sim_board.apply_move(*action)
            node = node.children[action]
            path.append(node)

        if node.is_terminal:
            value = node.terminal_value
        else:
            value = self._expand(node, sim_board)

        # Backup. ``value`` is from the leaf-current-player's perspective.
        # Walking from path[i+1] up to path[i]: flip iff path[i].child_flips
        # (i.e. moving across that edge changed the player-to-move).
        for i in range(len(path) - 1, -1, -1):
            n = path[i]
            n.visits += 1
            n.value_sum += value
            if i > 0 and path[i - 1].child_flips:
                value = -value

    @torch.no_grad()
    def get_move(self, state: GameState, rust_board) -> Action:
        # Build root by expanding it once.
        root = _Node(prior=0.0)
        self._expand(root, rust_board)
        if root.is_terminal:
            raise RuntimeError("KClusterMCTSBot: root is terminal; no moves to make")

        for _ in range(self.n_sims):
            self._simulate(root, rust_board)

        actions = list(root.children.keys())
        visits = np.array(
            [root.children[a].visits for a in actions], dtype=np.float64
        )
        if self.temperature == 0.0:
            idx = int(visits.argmax())
        else:
            t = max(self.temperature, 1e-6)
            scaled = visits ** (1.0 / t)
            s = scaled.sum()
            if s < 1e-12:
                probs = np.ones(len(actions)) / len(actions)
            else:
                probs = scaled / s
            idx = int(np.random.choice(len(actions), p=probs))
        return actions[idx]
