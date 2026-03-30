import logging
import torch
import time
from typing import Dict, Any
from native_core import Board  # type: ignore[attr-defined]
from python.env.game_state import GameState
from python.bootstrap.bots.random_bot import RandomBot
from python.bootstrap.bots.ramora_bot import RamoraBot
from python.model.network import HexTacToeNet
from python.selfplay.worker import SelfPlayWorker

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

class Evaluator:
    """Benchmarking agent against baseline bots."""
    
    def __init__(self, model: HexTacToeNet, device: torch.device, config: Dict[str, Any]):
        # Unwrap compiled model to avoid PyTorch torch.compile deadlocks 
        # when running concurrently with the background WorkerPool inference thread
        self.model = getattr(model, "_orig_mod", model)
        self.device = device
        self.config = config
        eval_cfg = config.get("evaluation", config.get("eval", {}))
        self.random_model_sims = int(eval_cfg.get("random_model_sims", 100))
        self.ramora_model_sims = int(eval_cfg.get("ramora_model_sims", 200))
        self.progress_every = max(1, int(eval_cfg.get("progress_every", 1)))
        # Use a worker instance for MCTS search during evaluation
        self.worker = SelfPlayWorker(model, config, device)

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
        
    def evaluate_vs_random(self, n_games: int = 20, model_sims: int | None = None) -> float:
        """Play n_games against RandomBot, return win rate."""
        random_bot = RandomBot()
        win_count = 0
        sims = self.random_model_sims if model_sims is None else int(model_sims)
        t0 = time.time()

        log.info("evaluation_games_start", phase="random", n_games=n_games, model_sims=sims)
        
        for i in range(n_games):
            board = Board()
            state = GameState.from_board(board)
            # Alternate who starts
            model_player = 1 if i % 2 == 0 else -1
            
            while not board.check_win() and board.legal_move_count() > 0:
                if board.current_player == model_player:
                    # Deterministic play for evaluation (temperature=0)
                    policy = self.worker._run_mcts_with_sims(board, n_sims=sims, use_dirichlet=False, temperature=0.0)
                    q, r = self.worker._sample_action(policy, board.legal_moves(), board)
                else:
                    q, r = random_bot.get_move(state, board)
                
                state = state.apply_move(board, q, r)
                
            if board.winner() == model_player:
                win_count += 1
            self._log_progress("random", i + 1, n_games, t0, win_count)

        wr = win_count / n_games
        log.info("evaluation_games_complete", phase="random", n_games=n_games, model_sims=sims, winrate=wr, elapsed_sec=round(time.time() - t0, 2))
        return wr

    def evaluate_vs_ramora(
        self,
        n_games: int = 10,
        time_limit: float = 0.05,
        model_sims: int | None = None,
    ) -> float:
        """Play n_games against RamoraBot, return win rate."""
        ramora_bot = RamoraBot(time_limit=time_limit)
        win_count = 0
        sims = self.ramora_model_sims if model_sims is None else int(model_sims)
        t0 = time.time()

        log.info(
            "evaluation_games_start",
            phase="ramora",
            n_games=n_games,
            model_sims=sims,
            ramora_time_limit=time_limit,
        )
        
        for i in range(n_games):
            board = Board()
            state = GameState.from_board(board)
            model_player = 1 if i % 2 == 0 else -1
            
            while not board.check_win() and board.legal_move_count() > 0:
                if board.current_player == model_player:
                    # Stronger search for benchmarking
                    policy = self.worker._run_mcts_with_sims(board, n_sims=sims, use_dirichlet=False, temperature=0.0)
                    q, r = self.worker._sample_action(policy, board.legal_moves(), board)
                else:
                    q, r = ramora_bot.get_move(state, board)
                
                state = state.apply_move(board, q, r)
                
            if board.winner() == model_player:
                win_count += 1
            self._log_progress("ramora", i + 1, n_games, t0, win_count)

        wr = win_count / n_games
        log.info(
            "evaluation_games_complete",
            phase="ramora",
            n_games=n_games,
            model_sims=sims,
            ramora_time_limit=time_limit,
            winrate=wr,
            elapsed_sec=round(time.time() - t0, 2),
        )
        return wr

    def evaluate_vs_model(
        self,
        opponent_model: HexTacToeNet,
        n_games: int = 20,
        model_sims: int | None = None,
        opponent_sims: int | None = None,
    ) -> float:
        """Play n_games against another model and return current-model win rate."""
        current_sims = self.ramora_model_sims if model_sims is None else int(model_sims)
        other_sims = self.ramora_model_sims if opponent_sims is None else int(opponent_sims)

        # Keep the opponent worker separate so each side has its own tree state.
        opponent_worker = SelfPlayWorker(opponent_model, self.config, self.device)
        win_count = 0
        t0 = time.time()

        log.info(
            "evaluation_games_start",
            phase="best_arena",
            n_games=n_games,
            model_sims=current_sims,
            opponent_sims=other_sims,
        )

        for i in range(n_games):
            board = Board()
            state = GameState.from_board(board)
            model_player = 1 if i % 2 == 0 else -1

            while not board.check_win() and board.legal_move_count() > 0:
                if board.current_player == model_player:
                    policy = self.worker._run_mcts_with_sims(
                        board,
                        n_sims=current_sims,
                        use_dirichlet=False,
                        temperature=0.0,
                    )
                    q, r = self.worker._sample_action(policy, board.legal_moves(), board)
                else:
                    policy = opponent_worker._run_mcts_with_sims(
                        board,
                        n_sims=other_sims,
                        use_dirichlet=False,
                        temperature=0.0,
                    )
                    q, r = opponent_worker._sample_action(policy, board.legal_moves(), board)

                state = state.apply_move(board, q, r)

            if board.winner() == model_player:
                win_count += 1
            self._log_progress("best_arena", i + 1, n_games, t0, win_count)

        wr = win_count / n_games
        log.info(
            "evaluation_games_complete",
            phase="best_arena",
            n_games=n_games,
            model_sims=current_sims,
            opponent_sims=other_sims,
            winrate=wr,
            elapsed_sec=round(time.time() - t0, 2),
        )
        return wr
