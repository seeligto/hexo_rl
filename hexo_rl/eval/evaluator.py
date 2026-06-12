"""Evaluator — plays games between a model and opponents via BotProtocol.

All opponents are injected via dependency injection; the evaluator never
imports concrete bot implementations directly.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from engine import Board, MCTSTree
from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.colony_detection import is_colony_win
from hexo_rl.eval.defaults import (
    DEFAULT_C_PUCT,
    DEFAULT_COLONY_CENTROID_THRESHOLD,
    DEFAULT_EVAL_SEED_BASE,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_EVALUATOR_RANDOM_MODEL_SIMS,
    DEFAULT_EVALUATOR_SEALBOT_MODEL_SIMS,
)
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding import normalize_encoding_name as _normalize_encoding_name

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
    draw_count: int = 0


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
        # Derive board geometry from encoding registry (§173 eval-fix).
        # §175: config["encoding"] may be str OR {"version": ...} dict after
        # Trainer._propagate_encoding_into_config — normalize either form.
        encoding_name = _normalize_encoding_name(config.get("encoding") if config else None)
        spec = _lookup_encoding(encoding_name)
        # Pass the resolved spec so the engine slices the 18-plane wire tensor
        # to THIS encoding's kept planes (v6tp keeps 16/17 → 10 planes); the
        # default would slice to v6's 8 and mismatch a 10-channel model.
        self._engine = LocalInferenceEngine(model, device, encoding_spec=spec)
        self._tree = MCTSTree(float(config.get("mcts", config).get("c_puct", DEFAULT_C_PUCT)))
        self._n_sims = n_sims
        self._config = config
        self._temperature = temperature
        self.board_size = spec.board_size
        self.n_actions = spec.policy_logit_count

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

        policy = self._tree.get_policy(temperature=self._temperature, board_size=self.board_size)

        legal_moves = board.legal_moves()
        legal_flat = [board.to_flat(q, r) for q, r in legal_moves]
        probs = np.array(
            [policy[i] if i < self.n_actions else 0.0 for i in legal_flat],
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
        self.random_model_sims = int(eval_cfg.get("random_model_sims", DEFAULT_EVALUATOR_RANDOM_MODEL_SIMS))
        self.sealbot_model_sims = int(eval_cfg.get("sealbot_model_sims", DEFAULT_EVALUATOR_SEALBOT_MODEL_SIMS))
        self.progress_every = max(1, int(eval_cfg.get("progress_every", 1)))
        self.colony_centroid_threshold = float(
            eval_cfg.get("colony_centroid_threshold", DEFAULT_COLONY_CENTROID_THRESHOLD)
        )
        self._eval_temperature = float(eval_cfg.get("eval_temperature", DEFAULT_EVAL_TEMPERATURE))
        self._eval_random_opening_plies = int(eval_cfg.get("eval_random_opening_plies", 0))
        self._eval_seed_base = int(eval_cfg.get("eval_seed_base", DEFAULT_EVAL_SEED_BASE))

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
        model_player = ModelPlayer(
            self.model, self.config, self.device,
            n_sims=model_sims,
            temperature=self._eval_temperature,
        )
        win_count = 0
        draw_count = 0
        colony_wins = 0
        t0 = time.time()

        log.info("evaluation_games_start", phase=phase, n_games=n_games, model_sims=model_sims)

        for i in range(n_games):
            np.random.seed(self._eval_seed_base + i)
            random.seed(self._eval_seed_base + i)
            encoding_name = _normalize_encoding_name(self.config.get("encoding"))
            board = Board.with_encoding_name(encoding_name)
            state = GameState.from_board(board)
            model_player_side = 1 if i % 2 == 0 else -1
            ply = 0

            while not board.check_win() and board.legal_move_count() > 0:
                if ply < self._eval_random_opening_plies:
                    q, r = random.choice(board.legal_moves())
                elif board.current_player == model_player_side:
                    q, r = model_player.get_move(state, board)
                else:
                    q, r = opponent.get_move(state, board)

                state = state.apply_move(board, q, r)
                ply += 1

            winner = board.winner()
            if winner == model_player_side:
                win_count += 1
                if is_colony_win(board.get_stones(), model_player_side, self.colony_centroid_threshold):
                    colony_wins += 1
            elif winner is None:
                draw_count += 1
            self._log_progress(phase, i + 1, n_games, t0, win_count)

        wr = (win_count + 0.5 * draw_count) / n_games
        log.info(
            "evaluation_games_complete",
            phase=phase,
            n_games=n_games,
            model_sims=model_sims,
            winrate=wr,
            win_count=win_count,
            draw_count=draw_count,
            colony_wins=colony_wins,
            elapsed_sec=round(time.time() - t0, 2),
        )
        return EvalResult(
            win_rate=wr, win_count=win_count, n_games=n_games,
            colony_wins=colony_wins, draw_count=draw_count,
        )

    def evaluate_batched(
        self,
        opponent_factory: Any,
        n_games: int,
        model_sims: int,
        phase: str = "eval",
    ) -> EvalResult:
        """Cross-game batched eval (D-EVALFOUND C3) — same EvalResult as ``evaluate``
        but plays ``n_games`` concurrently with the model's MCTS batched across games
        (GPU 50% → high). ``opponent_factory`` builds a FRESH opponent per game (each
        concurrent game needs its own). Uses per-game RNG seeded ``eval_seed_base+i``
        and color i%2 — the serial path's schedule (G5 behavior-neutral target).

        Aggregate-equivalent to ``evaluate`` (G4), NOT byte-identical to the old global-
        RNG serial transcripts (the per-game-RNG fix is the one behavior change)."""
        from hexo_rl.eval.eval_batcher import batched_evaluate

        log.info("evaluation_games_start", phase=phase, n_games=n_games,
                 model_sims=model_sims, batched=True)
        t0 = time.time()
        res = batched_evaluate(
            self.model, self.config, self.device, opponent_factory, n_games, model_sims,
            temperature=self._eval_temperature, seed_base=self._eval_seed_base,
            opening_plies=self._eval_random_opening_plies,
            colony_centroid_threshold=self.colony_centroid_threshold,
        )
        log.info("evaluation_games_complete", phase=phase, n_games=n_games,
                 model_sims=model_sims, winrate=res.win_rate, win_count=res.win_count,
                 draw_count=res.draw_count, colony_wins=res.colony_wins,
                 elapsed_sec=round(time.time() - t0, 2), batched=True)
        return res

    def evaluate_vs_random(self, n_games: int = 20, model_sims: int | None = None, random_bot: Optional[BotProtocol] = None) -> EvalResult:
        """Play n_games against a random bot. Accepts bot via DI or creates default."""
        if random_bot is None:
            from hexo_rl.bots.random_bot import RandomBot
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
            from hexo_rl.bots.sealbot_bot import SealBotBot
            sealbot = SealBotBot(time_limit=time_limit)
        sims = self.sealbot_model_sims if model_sims is None else int(model_sims)
        return self.evaluate(sealbot, n_games, sims, phase="sealbot")

    def evaluate_vs_nnue(
        self,
        n_games: int = 100,
        time_per_stone_ms: int = 500,
        model_sims: int | None = None,
        nnue: Optional[BotProtocol] = None,
    ) -> EvalResult:
        """Play n_games against the Hammerhead minimax+NNUE bot — the §P6
        second ladder opponent (eval-only). Accepts the bot via DI or creates
        a default. ``time_per_stone_ms`` is Hammerhead's per-stone search
        budget (mirrors SealBot's think-time knob); the model plays at the
        same ``sealbot_model_sims`` MCTS budget so the two ladder rungs are
        directly comparable. The NnueBot import is lazy so the heavyweight
        engine never loads on the self-play/training path."""
        if nnue is None:
            from hexo_rl.bots.nnue_bot import NnueBot
            nnue = NnueBot(time_per_stone_ms=time_per_stone_ms)
        sims = self.sealbot_model_sims if model_sims is None else int(model_sims)
        return self.evaluate(nnue, n_games, sims, phase="nnue")

    def evaluate_vs_offwindow_adversary(
        self,
        n_games: int = 100,
        model_sims: int | None = None,
        arm: str = "exploit",
        opening_plies: int = 6,
    ) -> dict[str, Any]:
        """EXPLOITABILITY metric (D-EXPLOIT) — the off-window forced-win rate of the
        off-window adversary vs THIS model's own MCTS (genuine resistance). Returns the
        summary dict; ``off_window_forced_win_rate`` is the monitored exploitability
        trend, NOT a promotion gate. The adversary import is lazy (eval-path only).
        Mirrors the standalone ``scripts/exploit_probe.py`` measurement."""
        from hexo_rl.eval.offwindow_probe import run_adversary_games
        encoding = _normalize_encoding_name(self.config.get("encoding"))
        spec = _lookup_encoding(encoding)
        model_player = ModelPlayer(
            self.model, self.config, self.device, n_sims=int(model_sims or self.sealbot_model_sims),
            temperature=0.0,
        )
        summary, _recs = run_adversary_games(
            model_player, encoding, spec, arm, int(n_games), int(model_sims or self.sealbot_model_sims),
            opening_plies=int(opening_plies), seed_base=self._eval_seed_base,
        )
        return summary

    def evaluate_vs_argmax_sealbot(
        self,
        n_games: int = 20,
        time_limit: float = 0.5,
        sealbot: Optional[BotProtocol] = None,
    ) -> EvalResult:
        """Argmax-only model vs SealBot — n_sims=1 collapses MCTS to prior-argmax.

        §170 P4 P1 DRIFT detector arm. With n_sims=1 the PUCT loop visits
        exactly one root child (the unvisited action with the largest prior),
        so ``best_action`` returns approximately ``argmax(policy_head)``.
        Compared against full-MCTS WR (e.g. bootstrap_anchor MCTS-128) this
        isolates policy-head behaviour from value-head behaviour — divergence
        between the two is the §170 value-drift signature.
        """
        if sealbot is None:
            from hexo_rl.bots.sealbot_bot import SealBotBot
            sealbot = SealBotBot(time_limit=time_limit)
        return self.evaluate(sealbot, n_games, model_sims=1, phase="argmax_n")

    def evaluate_vs_model(
        self,
        opponent_model: HexTacToeNet,
        n_games: int = 20,
        model_sims: int | None = None,
        opponent_sims: int | None = None,
        opponent_encoding: str | None = None,
    ) -> EvalResult:
        """Play n_games against another model and return EvalResult.

        ``opponent_encoding`` (the opponent's own ``spec.name``, e.g. a
        cross-encoding ``bootstrap_anchor``'s) drives the opponent ModelPlayer's
        input-plane slice + board geometry.  Without it (F07) the opponent's
        18-plane wire tensor is sliced to the CURRENT model's encoding, feeding a
        cross-encoding anchor the wrong planes and corrupting ``wr_bootstrap_anchor``
        / ``wr_best`` — the promotion gates.  Note: derive it from the spec the
        loader returned, NOT ``opponent_model.encoding`` — ``_build_model_from_spec``
        stamps a hardcoded ``"v6"``/``"v8"`` there, so a v6w25 anchor mis-reports.
        ``None`` ⇒ same encoding as this model (the same-encoding champion case).
        """
        current_sims = self.sealbot_model_sims if model_sims is None else int(model_sims)
        other_sims = self.sealbot_model_sims if opponent_sims is None else int(opponent_sims)

        if opponent_encoding is not None:
            opponent_config = dict(self.config)
            opponent_config["encoding"] = _normalize_encoding_name(opponent_encoding)
        else:
            opponent_config = self.config
        opponent_player = ModelPlayer(
            getattr(opponent_model, "_orig_mod", opponent_model),
            opponent_config, self.device, n_sims=other_sims,
            temperature=self._eval_temperature,
        )
        return self.evaluate(opponent_player, n_games, current_sims, phase="best_arena")
