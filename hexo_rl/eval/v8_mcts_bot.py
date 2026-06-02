"""V8MCTSBot — Python MCTS for v8 models.

The Rust engine's MCTSTree hardcodes BOARD_SIZE=19 / N_ACTIONS=362 (v6
wire format) so cannot drive v8 search. This module implements a small
sequential PUCT MCTS in Python on top of a cloned PyBoard per sim and
the v8 NN encoder. Slow vs Rust (~5ms per NN forward × n_sims per move)
but correct — sufficient for offline SealBot eval and any v8 ablation
work. A Rust v8 MCTS port is a separate sprint.

Memory note: the tree is a dict-of-dicts keyed on (q,r) actions so that
re-visiting a node during backup is O(d). The board state at each tree
node is reconstructed by replaying the action chain on a cloned root
board — no per-node Board allocation. History deques for the v8 encoder
are extended along the tree-descent path.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.bootstrap.dataset_v8 import (
    BOARD_SIZE_V8,
    HALF_V8,
    HISTORY_LEN_V8,
    encode_position_v8,
)
from hexo_rl.env import GameState
from hexo_rl.model.network import HexTacToeNet


Action = Tuple[int, int]


class _Node:
    __slots__ = (
        "prior", "visits", "value_sum", "children", "is_terminal",
        "terminal_value", "child_flips",
    )

    def __init__(self, prior: float = 0.0) -> None:
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[Action, "_Node"] = {}
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0
        # SCATTER-1: True iff moving from this node to a child crosses a turn
        # boundary (this node's moves_remaining == 1 ⇒ the turn-final stone
        # flips the player). Set at expand; gates the backup sign flip below.
        self.child_flips: bool = False

    def expanded(self) -> bool:
        return bool(self.children) or self.is_terminal

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _puct_select(node: _Node, c_puct: float) -> Action:
    """Pick the child action by max Q + U (PUCT)."""
    total = max(1, node.visits)
    sqrt_total = math.sqrt(total)
    best_score = -math.inf
    best_action: Optional[Action] = None
    for action, child in node.children.items():
        if child.visits == 0:
            q = 0.0
        else:
            q = child.value()
        u = c_puct * child.prior * sqrt_total / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
    assert best_action is not None
    return best_action


def _backup_turn_aware(path: List["_Node"], leaf_value: float) -> None:
    """Backup ``leaf_value`` (from the leaf side-to-move's perspective) up
    ``path``, flipping perspective ONLY across a turn boundary
    (``parent.child_flips``).

    SCATTER-1: HeXO places 2 stones per turn, so the player-to-move changes
    only at turn boundaries — NOT on every tree edge. The previous code flipped
    the sign on every backup step, corrupting Q for intra-turn edges. Mirrors
    ``KClusterMCTSBot._simulate``'s backup.
    """
    value = leaf_value
    for i in range(len(path) - 1, -1, -1):
        n = path[i]
        n.visits += 1
        n.value_sum += value
        if i > 0 and path[i - 1].child_flips:
            value = -value


def _legal_priors_from_logp(
    log_policy: np.ndarray,
    legal_moves: List[Action],
    cq: int,
    cr: int,
) -> List[float]:
    """Project the v8 (625,) log-policy onto the legal-move set, normalize."""
    scores = np.full(len(legal_moves), -1e30, dtype=np.float64)
    for i, (q, r) in enumerate(legal_moves):
        wq = q - cq + HALF_V8
        wr = r - cr + HALF_V8
        if 0 <= wq < BOARD_SIZE_V8 and 0 <= wr < BOARD_SIZE_V8:
            scores[i] = float(log_policy[wq * BOARD_SIZE_V8 + wr])
    shifted = scores - scores.max()
    probs = np.exp(shifted)
    s = probs.sum()
    if s < 1e-12:
        return [1.0 / len(legal_moves)] * len(legal_moves)
    probs /= s
    return probs.tolist()


class V8MCTSBot(BotProtocol):
    """Sequential PUCT MCTS over a v8 NN policy + value head.

    Args:
        model: v8 HexTacToeNet (encoding == "v8").
        device: torch device.
        n_sims: simulations per move.
        c_puct: PUCT exploration constant.
        temperature: 0.0 = argmax visit count; >0 = softmax sample.
    """

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        n_sims: int = 128,
        c_puct: float = 1.5,
        temperature: float = 0.0,
    ) -> None:
        if getattr(model, "encoding", None) != "v8":
            raise ValueError(
                f"V8MCTSBot requires a v8 model; got encoding={getattr(model, 'encoding', None)!r}"
            )
        self.model = model.eval()
        self.device = device
        self.n_sims = int(n_sims)
        self.c_puct = float(c_puct)
        self.temperature = float(temperature)
        self._history: Deque[Tuple[List[Action], List[Action]]] = deque(
            maxlen=HISTORY_LEN_V8 - 1
        )
        self._last_ply: Optional[int] = None

    def reset(self) -> None:
        self._history.clear()
        self._last_ply = None

    def name(self) -> str:
        return f"v8_mcts{self.n_sims}"

    @torch.no_grad()
    def _evaluate(
        self,
        board_stones: List[Tuple[int, int, int]],
        cur_player: int,
        history: Deque[Tuple[List[Action], List[Action]]],
        ply: int,
        moves_rem: int,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Run one v8 NN forward; return (log_policy_625, value_scalar, centroid)."""
        tensor, (cq, cr), _ = encode_position_v8(
            board_stones, cur_player, history, ply, moves_rem,
            canvas_realness=getattr(self.model, "canvas_realness", False),
        )
        x = torch.from_numpy(tensor).unsqueeze(0).float().to(self.device)
        log_policy, value, _v_logit = self.model(x)
        return (
            log_policy.squeeze(0).cpu().numpy(),
            float(value.item()),
            (cq, cr),
        )

    def _expand(
        self,
        node: _Node,
        rust_board,  # PyBoard at this node's state
        history: Deque[Tuple[List[Action], List[Action]]],
    ) -> float:
        """Expand `node` from a leaf state via NN forward.

        Returns the value (from current player's perspective).
        """
        if rust_board.check_win():
            # CF-1 terminal sign from the side-to-move (SCATTER-1): a first-stone
            # win keeps the winner to move (moves_remaining==1 ⇒ +1.0); a
            # turn-final win flips to the loser (moves_remaining==2 ⇒ -1.0). Route
            # through the engine SoT, never a hardcoded -1.0.
            tv = rust_board.terminal_value_to_move()
            node.is_terminal = True
            node.terminal_value = tv
            return tv

        legal_moves = list(rust_board.legal_moves())
        if not legal_moves:
            node.is_terminal = True
            node.terminal_value = 0.0
            return 0.0

        cur_player = int(rust_board.current_player)
        ply = int(rust_board.ply)
        moves_rem = int(rust_board.moves_remaining)
        board_stones = list(rust_board.get_stones())
        log_p, value, (cq, cr) = self._evaluate(
            board_stones, cur_player, history, ply, moves_rem
        )
        priors = _legal_priors_from_logp(log_p, legal_moves, cq, cr)
        # SCATTER-1: moving from this node crosses a turn boundary iff this is
        # the turn-final stone (moves_remaining == 1). Gates the backup flip.
        node.child_flips = (moves_rem == 1)
        for action, p in zip(legal_moves, priors):
            node.children[action] = _Node(prior=p)
        return value

    def _simulate(self, root: _Node, root_board) -> None:
        """One MCTS simulation: descend → expand leaf → backup."""
        node = root
        path: List[_Node] = [node]
        action_path: List[Action] = []
        # Per-sim history copy (extended along descent to keep the v8 encoder
        # consistent with what the model saw at training time).
        history = deque(self._history, maxlen=HISTORY_LEN_V8 - 1)
        # board: clone root board so this sim can mutate freely.
        sim_board = root_board.clone()

        # Descend.
        while node.expanded() and not node.is_terminal and node.children:
            action = _puct_select(node, self.c_puct)
            # Snapshot pre-move state for history (matches V8ArgmaxBot
            # convention: history[-1] is "ply T-1 stones").
            cur_player = int(sim_board.current_player)
            stones = list(sim_board.get_stones())
            cur_xy = [(q, r) for (q, r, p) in stones if p == cur_player]
            opp_xy = [(q, r) for (q, r, p) in stones if p != cur_player]
            history.append((cur_xy, opp_xy))
            sim_board.apply_move(*action)
            node = node.children[action]
            path.append(node)
            action_path.append(action)

        # Expand or use terminal value.
        if node.is_terminal:
            value = node.terminal_value
        else:
            value = self._expand(node, sim_board, history)

        # Backup. `value` is from the leaf side-to-move's perspective; flip
        # ONLY across a turn boundary (parent.child_flips), not every step —
        # HeXO places 2 stones per turn (SCATTER-1).
        _backup_turn_aware(path, value)

    @torch.no_grad()
    def get_move(self, state: GameState, rust_board) -> Action:
        if self._last_ply is not None and rust_board.ply <= self._last_ply:
            self.reset()

        # Build root by expanding it once.
        root = _Node(prior=0.0)
        self._expand(root, rust_board, self._history)
        if root.is_terminal:
            raise RuntimeError("V8MCTSBot: root is terminal; no moves to make")

        # Run sims.
        for _ in range(self.n_sims):
            self._simulate(root, rust_board)

        # Pick action by visit count.
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
        chosen = actions[idx]

        # Update bot history with pre-move snapshot of the live board (NOT
        # the clone) so the next get_move sees the right history.
        cur_player = int(rust_board.current_player)
        stones = list(rust_board.get_stones())
        cur_xy = [(q, r) for (q, r, p) in stones if p == cur_player]
        opp_xy = [(q, r) for (q, r, p) in stones if p != cur_player]
        self._history.append((cur_xy, opp_xy))
        self._last_ply = int(rust_board.ply)
        return chosen
