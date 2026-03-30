
import structlog
import torch
from typing import Dict, Any
from native_core import Board  # type: ignore[attr-defined]
from python.env.game_state import GameState
from python.bootstrap.bots.random_bot import RandomBot
from python.bootstrap.bots.ramora_bot import RamoraBot
from python.model.network import HexTacToeNet
from python.selfplay.worker import SelfPlayWorker

log = structlog.get_logger()

class Evaluator:
    """Benchmarking agent against baseline bots."""
    
    def __init__(self, model: HexTacToeNet, device: torch.device, config: Dict[str, Any]):
        self.model = model
        self.device = device
        self.config = config
        eval_cfg = config.get("evaluation", config.get("eval", {}))
        self.random_model_sims = int(eval_cfg.get("random_model_sims", 100))
        self.ramora_model_sims = int(eval_cfg.get("ramora_model_sims", 200))
        # Use a worker instance for MCTS search during evaluation
        self.worker = SelfPlayWorker(model, config, device)
        
    def evaluate_vs_random(self, n_games: int = 20, model_sims: int | None = None) -> float:
        """Play n_games against RandomBot, return win rate."""
        random_bot = RandomBot()
        win_count = 0
        sims = self.random_model_sims if model_sims is None else int(model_sims)
        
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
                
        return win_count / n_games

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
                
        return win_count / n_games

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

        return win_count / n_games
