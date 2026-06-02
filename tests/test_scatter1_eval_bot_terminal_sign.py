"""SCATTER-1 — Python eval-bot terminal value uses the engine CF-1 sign.

Audit finding A3 / SCATTER-1: KClusterMCTSBot and V8MCTSBot hardcoded
`node.terminal_value = -1.0` at every check_win leaf (the pre-CF-1
"current player just lost" assumption). That is WRONG for a turn-first-stone
win: `apply_move` flips the player only on a turn-final stone, so a win on the
first stone of a 2-stone turn leaves the winner still to move
(`moves_remaining == 1`) and the correct leaf value is +1.0. Both bots must
route through the engine-owned CF-1 sign (`Board.terminal_value_to_move()`),
the single SoT that mirrors `mcts/backup.rs`.

The fix is eval-only and conservative (the old bug under-valued first-stone
winning lines → biased measured SealBot WR downward), so it cannot false-PASS a
gate; it makes standalone eval accurate.
"""
from __future__ import annotations

from collections import deque

import torch

from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot, _Node as KNode
from hexo_rl.eval.v8_mcts_bot import V8MCTSBot, _Node as VNode
from hexo_rl.model.network import HexTacToeNet

DEVICE = torch.device("cpu")


class _FakeFirstStoneWin:
    """A won terminal reached on the first stone of a 2-stone turn: the winner
    is still to move (moves_remaining == 1), so the CF-1 leaf value is +1.0."""

    moves_remaining = 1

    def check_win(self) -> bool:
        return True

    def terminal_value_to_move(self) -> float:
        # Mirrors the real engine method (mr==1 ⇒ +1.0). Hard-coding the engine
        # contract here keeps the test independent of the .so while still
        # pinning that the bot RETURNS this value rather than a constant -1.0.
        return 1.0


def _tiny_v6_model() -> HexTacToeNet:
    torch.manual_seed(0)
    return HexTacToeNet(board_size=19, in_channels=8, filters=8,
                        res_blocks=2, encoding="v6").eval()


def _tiny_v8_model() -> HexTacToeNet:
    torch.manual_seed(0)
    return HexTacToeNet(board_size=25, in_channels=11, filters=8,
                        res_blocks=2, encoding="v8").eval()


def test_k_cluster_terminal_value_uses_cf1_sign():
    bot = KClusterMCTSBot(_tiny_v6_model(), DEVICE, n_sims=1)
    node = KNode()
    v = bot._expand(node, _FakeFirstStoneWin())
    assert v == 1.0, f"first-stone win must score +1.0 (CF-1), got {v}"
    assert node.terminal_value == 1.0
    assert node.is_terminal


def test_v8_terminal_value_uses_cf1_sign():
    bot = V8MCTSBot(_tiny_v8_model(), DEVICE, n_sims=1)
    node = VNode()
    v = bot._expand(node, _FakeFirstStoneWin(), deque())
    assert v == 1.0, f"first-stone win must score +1.0 (CF-1), got {v}"
    assert node.terminal_value == 1.0
    assert node.is_terminal


def test_v8_backup_flips_only_across_turn_boundaries():
    """V8 used to flip perspective on EVERY backup step (audit: double-wrong for
    HeXO's 2-stone turns). The turn-aware backup flips ONLY across a turn
    boundary (parent.child_flips), matching KClusterMCTSBot."""
    from hexo_rl.eval.v8_mcts_bot import _backup_turn_aware

    root, c1, c2 = VNode(), VNode(), VNode()
    root.child_flips = False   # root → c1: same turn (root.moves_remaining == 2)
    c1.child_flips = True       # c1 → c2: crosses a turn boundary (c1.moves_remaining == 1)
    # leaf value +1.0 from c2's side-to-move (a win for c2's player).
    _backup_turn_aware([root, c1, c2], 1.0)

    assert c2.value_sum == 1.0                  # c2's own perspective
    assert c1.value_sum == -1.0                 # opponent of c2 (turn boundary)
    assert root.value_sum == -1.0               # same turn as c1 → same sign
    # The OLD unconditional per-step flip would give root.value_sum == +1.0
    # (flipping the non-boundary root→c1 edge) — the bug this fixes.
    assert all(n.visits == 1 for n in (root, c1, c2))
