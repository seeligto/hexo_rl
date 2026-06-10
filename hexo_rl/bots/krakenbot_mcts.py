"""KrakenBotMCTSBot — skeleton wrapper for KrakenBot's MCTS+ResNet bot.

**STATUS: UNAVAILABLE** as of Wave A1 (2026-05-14).

KrakenBot's MCTSBot needs a trained ResNet checkpoint loaded from
`vendor/bots/krakenbot/training/{resnet_results,mcts_results}/best.pt`,
but the upstream repo gitignores `.pt` files (`vendor/bots/krakenbot/.gitignore:8`)
and never published a mirror (no HF / S3 URL in README/requirements/pyproject).

This wrapper raises `FileNotFoundError` at construction with the
operator-actionable message; downstream tourney harness skips it.

If weights become available later, set `model_path=<your.pt>` at
construction and (separately) compile the Cython MCTS modules:

```
cd vendor/bots/krakenbot && \\
  /home/timmy/Work/hexo_rl/.venv/bin/python setup_puct.py build_ext --inplace
```

Wave A1 report: docs/archive/reports/s176_a1_kraken_smoke.md (section (d) MCTSBot row).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import structlog

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState

log = structlog.get_logger()

_KRAKENBOT_ROOT = Path(__file__).parents[2] / "vendor" / "bots" / "krakenbot"
assert _KRAKENBOT_ROOT.exists(), f"KrakenBot not found at {_KRAKENBOT_ROOT}"

# Default model paths (relative to _KRAKENBOT_ROOT). Match the hardcoded
# upstream defaults in mcts_bot.py: training/resnet_results/best.pt is
# preferred; training/mcts_results/best.pt is the fallback.
_DEFAULT_RESNET = _KRAKENBOT_ROOT / "training" / "resnet_results" / "best.pt"
_DEFAULT_SELFPLAY = _KRAKENBOT_ROOT / "training" / "mcts_results" / "best.pt"


class KrakenBotMCTSBot(BotProtocol):
    """BotProtocol wrapper for KrakenBot MCTSBot (PUCT + ResNet).

    Args:
        n_sims: MCTS simulation budget per call.
        model_path: Override checkpoint path. None tries
                    `training/resnet_results/best.pt` then
                    `training/mcts_results/best.pt`.

    Raises:
        FileNotFoundError: if no checkpoint is found at the resolved path.
            Operator must supply weights (see module docstring).
    """

    def __init__(
        self,
        n_sims: int = 200,
        model_path: str | None = None,
    ) -> None:
        if model_path is None:
            if _DEFAULT_RESNET.exists():
                model_path = str(_DEFAULT_RESNET)
            elif _DEFAULT_SELFPLAY.exists():
                model_path = str(_DEFAULT_SELFPLAY)
            else:
                model_path = str(_DEFAULT_RESNET)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"KrakenBot MCTSBot weights missing at {model_path}. "
                "See docs/archive/reports/s176_a1_kraken_smoke.md for status. "
                "Weights are not in the submodule (gitignored upstream); "
                "operator must provide them or run from a checkpoint dir."
            )

        if str(_KRAKENBOT_ROOT) not in sys.path:
            sys.path.insert(0, str(_KRAKENBOT_ROOT))

        from mcts_bot import MCTSBot as _KMCTSBot  # type: ignore[import]
        from game import Player as _KPlayer        # type: ignore[import]

        self._KPlayer = _KPlayer
        self._n_sims = int(n_sims)
        self._model_path = model_path
        self._bot = _KMCTSBot(
            time_limit=None,
            n_sims=self._n_sims,
            model_path=model_path,
            device="cpu",
        )
        self._pending_move: tuple[int, int] | None = None

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        # Lazy import _MockGame (defined alongside the random wrapper) to keep
        # this file self-contained — we still parallel the structure here.
        from hexo_rl.bots.krakenbot_random import _MockGame

        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            return move

        if state.moves_remaining > 1:
            self._pending_move = None

        board_dict: dict = {}
        for q, r, p in rust_board.get_stones():
            board_dict[(q, r)] = (
                self._KPlayer.A if p == 1 else self._KPlayer.B
            )

        kp = self._KPlayer.A if state.current_player == 1 else self._KPlayer.B
        game = _MockGame(
            board=board_dict,
            current_player=kp,
            moves_left_in_turn=state.moves_remaining,
            move_count=len(board_dict),
        )

        result = self._bot.get_move(game)

        if not result:
            log.warning("krakenbot_mcts_no_moves_found", ply=state.ply)
            legal = rust_board.legal_moves()
            if not legal:
                raise RuntimeError("No legal moves available on board")
            return random.choice(legal)

        # Compound: cache second stone.
        if len(result) >= 2 and state.moves_remaining > 1:
            q2, r2 = result[1]
            if rust_board.get(q2, r2) == 0:
                self._pending_move = (q2, r2)
            else:
                log.warning("krakenbot_mcts_pending_illegal", q=q2, r=r2)

        q1, r1 = result[0]
        return (q1, r1)

    def reset(self) -> None:
        self._pending_move = None

    def name(self) -> str:
        return f"KrakenBotMCTS(n={self._n_sims})"
