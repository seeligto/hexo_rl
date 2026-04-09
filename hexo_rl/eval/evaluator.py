"""Evaluator — plays games between a model and opponents via BotProtocol.

All opponents are injected via dependency injection; the evaluator never
imports concrete bot implementations directly.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from engine import Board, MCTSTree
from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.colony_detection import is_colony_win
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.selfplay.utils import BOARD_SIZE, N_ACTIONS, get_temperature

try:
    import structlog
    log = structlog.get_logger()
except ImportError:
    class _StdLoggerAdapter:
        def __init__(self) -> None:
            self._log = logging.getLogger("evaluator")

        def info(self, event: str, **kwargs: Any) -> None:
            if kwargs:
                self._log.info("%s %s", event, kwargs)
            else:
                self._log.info("%s", event)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _StdLoggerAdapter()


@dataclass
class EvalResult:
    """Result from a set of evaluation games."""
    win_rate: float
    win_count: int
    n_games: int
    colony_wins: int


class ModelPlayer(BotProtocol):
    """Lightweight MCTS-based model player for evaluation.

    Uses LocalInferenceEngine + MCTSTree directly, avoiding the full
    SelfPlayWorker with its replay buffer integration.
    """

    def __init__(
        self,
        model: HexTacToeNet,
        config: Dict[str, Any],
        device: torch.device,
        n_sims: int = 100,
        temperature: float = 0.0,
    ) -> None:
        self._engine = LocalInferenceEngine(model, device)
        self._tree = MCTSTree(float(config.get("mcts", config).get("c_puct", 1.5)))
        self._n_sims = n_sims
        self._config = config
        self._temperature = temperature

    def get_move(self, state: GameState, rust_board: object) -> Tuple[int, int]:
        board = rust_board
        self._tree.new_game(board)
        batch_size = 8
        sims_done = 0

        while sims_done < self._n_sims:
            current_batch = min(batch_size, self._n_sims - sims_done)
            leaves = self._tree.select_leaves(current_batch)
            if not leaves:
                break
            policies, values = self._engine.infer_batch(leaves)
            self._tree.expand_and_backup(policies, values)
            sims_done += current_batch

        policy = self._tree.get_policy(temperature=self._temperature, board_size=BOARD_SIZE)

        legal_moves = board.legal_moves()
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
        if self._temperature == 0.0:
            idx = int(np.argmax(probs))
            return legal_moves[idx]
        # Temperature > 0: sample from the distribution.
        idx = int(np.random.choice(len(legal_moves), p=probs))
        return legal_moves[idx]

    def name(self) -> str:
        return "model_player"


class Evaluator:
    """Benchmarking agent against baseline bots via dependency injection."""

    def __init__(self, model: HexTacToeNet, device: torch.device, config: Dict[str, Any]):
        self.model = getattr(model, "_orig_mod", model)
        self.device = device
        self.config = config
        eval_cfg = config.get("evaluation", config.get("eval", {}))
        self.random_model_sims = int(eval_cfg.get("random_model_sims", 100))
        self.sealbot_model_sims = int(eval_cfg.get("sealbot_model_sims", 200))
        self.progress_every = max(1, int(eval_cfg.get("progress_every", 1)))
        self.colony_centroid_threshold = float(
            eval_cfg.get("colony_centroid_threshold", 6.0)
        )

    def _log_progress(self, phase: str, idx: int, total: int, start_time: float, win_count: int) -> None:
        if idx % self.progress_every != 0 and idx != total:
            return
        elapsed = max(1e-6, time.time() - start_time)
        games_done = idx
        sec_per_game = elapsed / games_done
        eta_sec = sec_per_game * (total - games_done)
        log.info(
            "evaluation_game_progress",
            phase=phase,
            game=idx,
            total_games=total,
            elapsed_sec=round(elapsed, 2),
            sec_per_game=round(sec_per_game, 2),
            eta_sec=round(eta_sec, 2),
            partial_winrate=round(win_count / games_done, 3),
        )

    def evaluate(
        self,
        opponent: BotProtocol,
        n_games: int,
        model_sims: int,
        phase: str = "eval",
    ) -> EvalResult:
        """Play n_games against an opponent, return EvalResult.

        Args:
            opponent: Any BotProtocol implementation.
            n_games: Number of games to play.
            model_sims: MCTS simulations per move for the model.
            phase: Label for logging.
        """
        model_player = ModelPlayer(self.model, self.config, self.device, n_sims=model_sims)
        win_count = 0
        colony_wins = 0
        t0 = time.time()

        log.info("evaluation_games_start", phase=phase, n_games=n_games, model_sims=model_sims)

        for i in range(n_games):
            board = Board()
            state = GameState.from_board(board)
            model_player_side = 1 if i % 2 == 0 else -1

            while not board.check_win() and board.legal_move_count() > 0:
                if board.current_player == model_player_side:
                    q, r = model_player.get_move(state, board)
                else:
                    q, r = opponent.get_move(state, board)

                state = state.apply_move(board, q, r)

            if board.winner() == model_player_side:
                win_count += 1
                if is_colony_win(board.get_stones(), model_player_side, self.colony_centroid_threshold):
                    colony_wins += 1
            self._log_progress(phase, i + 1, n_games, t0, win_count)

        wr = win_count / n_games
        log.info(
            "evaluation_games_complete",
            phase=phase,
            n_games=n_games,
            model_sims=model_sims,
            winrate=wr,
            colony_wins=colony_wins,
            elapsed_sec=round(time.time() - t0, 2),
        )
        return EvalResult(win_rate=wr, win_count=win_count, n_games=n_games, colony_wins=colony_wins)

    def evaluate_vs_random(self, n_games: int = 20, model_sims: int | None = None, random_bot: Optional[BotProtocol] = None) -> EvalResult:
        """Play n_games against a random bot. Accepts bot via DI or creates default."""
        if random_bot is None:
            from hexo_rl.bootstrap.bots.random_bot import RandomBot
            random_bot = RandomBot()
        sims = self.random_model_sims if model_sims is None else int(model_sims)
        return self.evaluate(random_bot, n_games, sims, phase="random")

    def evaluate_vs_sealbot(
        self,
        n_games: int = 10,
        time_limit: float = 0.05,
        model_sims: int | None = None,
        sealbot: Optional[BotProtocol] = None,
    ) -> EvalResult:
        """Play n_games against SealBot. Accepts bot via DI or creates default."""
        if sealbot is None:
            from hexo_rl.bootstrap.bots.sealbot_bot import SealBotBot
            sealbot = SealBotBot(time_limit=time_limit)
        sims = self.sealbot_model_sims if model_sims is None else int(model_sims)
        return self.evaluate(sealbot, n_games, sims, phase="sealbot")

    def evaluate_vs_model(
        self,
        opponent_model: HexTacToeNet,
        n_games: int = 20,
        model_sims: int | None = None,
        opponent_sims: int | None = None,
    ) -> EvalResult:
        """Play n_games against another model and return EvalResult."""
        current_sims = self.sealbot_model_sims if model_sims is None else int(model_sims)
        other_sims = self.sealbot_model_sims if opponent_sims is None else int(opponent_sims)

        opponent_player = ModelPlayer(
            getattr(opponent_model, "_orig_mod", opponent_model),
            self.config, self.device, n_sims=other_sims,
        )
        return self.evaluate(opponent_player, n_games, current_sims, phase="best_arena")
