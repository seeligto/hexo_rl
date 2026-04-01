import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any
import structlog
from tqdm import tqdm

from python.model.network import HexTacToeNet
from python.bootstrap.dataset import BootstrapDataset, convert_to_dataset
from python.bootstrap.bots.sealbot_bot import SealBotBot
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
            states   = states.to(self.device).float()
            policies = policies.to(self.device)
            outcomes = outcomes.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
                log_policy, value, v_logit = self.model(states)
                policy_loss = -(policies * log_policy).sum(dim=1).mean()
                value_target = (outcomes + 1.0) / 2.0
                value_loss  = nn.functional.binary_cross_entropy_with_logits(
                    v_logit.squeeze(1), value_target
                )
                loss        = policy_loss + value_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss        += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n = len(loader)
        return total_loss / n, total_policy_loss / n, total_value_loss / n


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

    cache_dir  = Path("data/corpus")
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
            except Exception:
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
    loader  = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model   = HexTacToeNet(board_size=19, res_blocks=12, filters=128)
    trainer = BootstrapTrainer(model, config, device)

    for epoch in range(1, config['epochs'] + 1):
        loss, p_loss, v_loss = trainer.train_epoch(loader)
        log.info("epoch_complete", epoch=epoch, loss=loss, p_loss=p_loss, v_loss=v_loss)

    save_path = Path("checkpoints/bootstrap_model.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("model_saved", path=str(save_path))

    validate(model, device)


def validate(model: HexTacToeNet, device: torch.device) -> None:
    from native_core import Board
    from python.env.game_state import GameState
    from python.selfplay.worker import SelfPlayWorker

    config = {
        'mcts': {
            'n_simulations': 100,
            'c_puct': 1.5,
            'temperature_threshold_ply': 30,
            'dirichlet_alpha': 0.3,
            'epsilon': 0.25,
        }
    }
    worker = SelfPlayWorker(model, config, device)

    # 1. vs RandomBot
    log.info("validating_vs_random")
    random_bot  = RandomBot()
    win_count   = 0
    n_games     = 20
    for i in range(n_games):
        board        = Board()
        state        = GameState.from_board(board)
        model_player = 1 if i % 2 == 0 else -1
        while not board.check_win() and board.legal_move_count() > 0:
            if board.current_player == model_player:
                policy = worker._run_mcts_with_sims(board, n_sims=100, use_dirichlet=False, temperature=0.0)
                q, r   = worker._sample_action(policy, board.legal_moves(), board)
            else:
                q, r = random_bot.get_move(state, board)
            state = state.apply_move(board, q, r)
        if board.winner() == model_player:
            win_count += 1
    random_win_rate = win_count / n_games
    log.info("validation_random_result", win_rate=random_win_rate)

    # 2. vs SealBot (shallow time budget ≈ depth-3)
    log.info("validating_vs_sealbot")
    sealbot = SealBotBot(time_limit=0.05)
    win_count  = 0
    n_games    = 10
    for i in range(n_games):
        board        = Board()
        state        = GameState.from_board(board)
        model_player = 1 if i % 2 == 0 else -1
        while not board.check_win() and board.legal_move_count() > 0:
            if board.current_player == model_player:
                policy = worker._run_mcts_with_sims(board, n_sims=400, use_dirichlet=False, temperature=0.0)
                q, r   = worker._sample_action(policy, board.legal_moves(), board)
            else:
                q, r = sealbot.get_move(state, board)
            state = state.apply_move(board, q, r)
        if board.winner() == model_player:
            win_count += 1
    sealbot_win_rate = win_count / n_games
    log.info("validation_sealbot_result", win_rate=sealbot_win_rate)

    if random_win_rate >= 0.90 and sealbot_win_rate > 0.05:
        log.info("validation_passed")
    else:
        log.warning("validation_failed")


if __name__ == "__main__":
    pretrain()
