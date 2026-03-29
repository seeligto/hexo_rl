import os
import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import structlog
from tqdm import tqdm

from python.model.network import HexTacToeNet
from python.env.game_state import GameState
from native_core import Board
from python.bootstrap.scraper import scrape_hexo_did
from python.bootstrap.bots.ramora_bot import RamoraBot
from python.bootstrap.bots.random_bot import RandomBot
from python.bootstrap.generate_corpus import load_cached_bot_games, RAW_HUMAN_DIR

log = structlog.get_logger()

class BootstrapTrainer:
    def __init__(self, model: HexTacToeNet, config: Dict[str, Any], device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config.get('lr', 0.002)),
            weight_decay=float(config.get('weight_decay', 0.0001))
        )
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=(self.device.type == "cuda"))

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        
        pbar = tqdm(loader, desc="Training")
        for states, policies, outcomes in pbar:
            states = states.to(self.device).float()
            policies = policies.to(self.device)
            outcomes = outcomes.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
                log_policy, value = self.model(states)
                
                policy_loss = -(policies * log_policy).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(value.squeeze(1), outcomes)
                loss = policy_loss + value_loss
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        return total_loss / len(loader), total_policy_loss / len(loader), total_value_loss / len(loader)

def convert_to_dataset(games: List[List[Tuple[int, int]]]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    log.info("converting_to_dataset", n_games=len(games))
    dataset = []
    
    for moves in tqdm(games, desc="Converting"):
        board = Board()
        state = GameState.from_board(board)
        
        history = []
        
        for q, r in moves:
            tensor, centers = state.to_tensor(board)
            K = len(centers)
            
            target_k = -1
            target_local_idx = -1
            
            for k, (cq, cr) in enumerate(centers):
                wq = q - cq + 9
                wr = r - cr + 9
                if 0 <= wq < 19 and 0 <= wr < 19:
                    target_k = k
                    target_local_idx = wq * 19 + wr
                    break
                    
            if target_k != -1:
                policy = np.zeros(19*19 + 1, dtype=np.float32)
                policy[target_local_idx] = 1.0
                history.append((tensor[target_k], policy, board.current_player))
            
            try:
                state = state.apply_move(board, q, r)
            except Exception:
                break
                
        winner = board.winner()
        outcome = 0.0
        if winner is not None:
            outcome = float(winner)
            
        for s_t, p, player in history:
            val = 1.0 if player == outcome else (-1.0 if outcome != 0.0 else 0.0)
            dataset.append((s_t, p, val))
            
    return dataset

class BootstrapDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        s, p, v = self.data[idx]
        return torch.from_numpy(s), torch.from_numpy(p), torch.tensor(v, dtype=torch.float32)

def pretrain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true", help="Use cached parsed dataset if available")
    parser.add_argument("--force-regenerate", action="store_true", help="Force dataset generation")
    parser.add_argument("--epochs", type=int, default=7, help="Number of training epochs")
    args = parser.parse_args()

    config = {
        'lr': 0.002,
        'weight_decay': 0.0001,
        'batch_size': 512,
        'epochs': args.epochs,
        'board_size': 19,
        'res_blocks': 10,
        'filters': 128
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cache_dir = Path("data/corpus")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "parsed_dataset.pkl"

    if args.use_cache and cache_file.exists() and not args.force_regenerate:
        log.info("loading_cached_dataset", path=str(cache_file))
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    else:
        bot_games = load_cached_bot_games()
        log.info("loaded_bot_games", count=len(bot_games))
        
        human_games = []
        for p in RAW_HUMAN_DIR.glob("*.json"):
            try:
                with open(p, 'r') as f:
                    game_details = json.load(f)
                    if 'moves' in game_details:
                        moves = [(m['x'], m['y']) for m in game_details['moves']]
                        human_games.append(moves)
            except:
                continue
        log.info("loaded_human_games", count=len(human_games))
        
        all_games = human_games + bot_games
        if not all_games:
            log.warning("no_games_found", hint="Run generate_corpus.py first")
            return
            
        data = convert_to_dataset(all_games)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        log.info("saved_cached_dataset", path=str(cache_file))

    dataset = BootstrapDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    model = HexTacToeNet(board_size=19, res_blocks=10, filters=128)
    trainer = BootstrapTrainer(model, config, device)
    
    for epoch in range(1, config['epochs'] + 1):
        loss, p_loss, v_loss = trainer.train_epoch(loader)
        log.info("epoch_complete", epoch=epoch, loss=loss, p_loss=p_loss, v_loss=v_loss)
        
    save_path = Path("checkpoints/bootstrap_model.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("model_saved", path=str(save_path))

    # 7. Validate
    validate(model, device)


def validate(model: HexTacToeNet, device: torch.device):
    from python.selfplay.worker import SelfPlayWorker
    
    # Simple wrapper to use the model in MCTS
    config = {
        'mcts': {
            'n_simulations': 100, # Faster eval
            'c_puct': 1.5,
            'temperature_threshold_ply': 30,
            'dirichlet_alpha': 0.3,
            'epsilon': 0.25
        }
    }
    worker = SelfPlayWorker(model, config, device)
    
    # 1. vs RandomBot
    log.info("validating_vs_random")
    random_bot = RandomBot()
    win_count = 0
    n_games = 20
    
    for i in range(n_games):
        board = Board()
        state = GameState.from_board(board)
        # Randomize who starts
        model_player = 1 if i % 2 == 0 else -1
        
        while not board.check_win() and board.legal_move_count() > 0:
            if board.current_player == model_player:
                # Use MCTS with model
                policy = worker._run_mcts_with_sims(board, n_sims=100, use_dirichlet=False, temperature=0.0)
                q, r = worker._sample_action(policy, board.legal_moves(), board)
            else:
                q, r = random_bot.get_move(state, board)
            
            state = state.apply_move(board, q, r)
            
        if board.winner() == model_player:
            win_count += 1
            
    random_win_rate = win_count / n_games
    log.info("validation_random_result", win_rate=random_win_rate)
    
    # 2. vs RamoraBot(depth=3)
    log.info("validating_vs_ramora_d3")
    ramora_bot = RamoraBot(time_limit=0.05) # depth 3 approx
    win_count = 0
    n_games = 10
    
    for i in range(n_games):
        board = Board()
        state = GameState.from_board(board)
        model_player = 1 if i % 2 == 0 else -1
        
        while not board.check_win() and board.legal_move_count() > 0:
            if board.current_player == model_player:
                policy = worker._run_mcts_with_sims(board, n_sims=400, use_dirichlet=False, temperature=0.0)
                q, r = worker._sample_action(policy, board.legal_moves(), board)
            else:
                q, r = ramora_bot.get_move(state, board)
            
            state = state.apply_move(board, q, r)
            
        if board.winner() == model_player:
            win_count += 1
            
    ramora_win_rate = win_count / n_games
    log.info("validation_ramora_result", win_rate=ramora_win_rate)
    
    if random_win_rate >= 0.90 and ramora_win_rate > 0.05:
        log.info("validation_passed")
    else:
        log.warning("validation_failed")

if __name__ == "__main__":
    pretrain()
