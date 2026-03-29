
import os
import json
import hashlib
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import structlog
from tqdm import tqdm

from native_core import Board
from python.env.game_state import GameState
from python.bootstrap.bots.ramora_bot import RamoraBot
from python.bootstrap.scraper import scrape_hexo_did, deduplicate_games

log = structlog.get_logger()

# Storage paths
CORPUS_DIR = Path("data/corpus")
RAW_HUMAN_DIR = CORPUS_DIR / "raw_human"
GENERATED_BOT_DIR = CORPUS_DIR / "generated_bot"

def get_game_hash(moves: List[Tuple[int, int]]) -> str:
    game_str = ",".join([f"{q},{r}" for q, r in moves])
    return hashlib.sha256(game_str.encode()).hexdigest()

def save_bot_game(moves: List[Tuple[int, int]], winner: Optional[int]):
    GENERATED_BOT_DIR.mkdir(parents=True, exist_ok=True)
    game_hash = get_game_hash(moves)
    cache_file = GENERATED_BOT_DIR / f"{game_hash}.json"
    
    data = {
        "moves": [{"x": q, "y": r} for q, r in moves],
        "winner": winner,
        "generated_at": time.time()
    }
    
    with open(cache_file, 'w') as f:
        json.dump(data, f)

def load_cached_bot_games() -> List[List[Tuple[int, int]]]:
    all_moves = []
    if not GENERATED_BOT_DIR.exists():
        return []
        
    for p in GENERATED_BOT_DIR.glob("*.json"):
        try:
            with open(p, 'r') as f:
                data = json.load(f)
                moves = [(m['x'], m['y']) for m in data['moves']]
                all_moves.append(moves)
        except:
            continue
    return all_moves

def generate_bot_games(n_target: int, depth_mix: Dict[int, float], force_regenerate: bool = False) -> List[List[Tuple[int, int]]]:
    if force_regenerate:
        log.info("clearing_bot_cache")
        if GENERATED_BOT_DIR.exists():
            for p in GENERATED_BOT_DIR.glob("*.json"):
                p.unlink()
                
    cached_games = load_cached_bot_games()
    log.info("loaded_bot_cache", count=len(cached_games))
    
    if len(cached_games) >= n_target:
        log.info("target_already_met", target=n_target)
        return cached_games[:n_target]
        
    n_needed = n_target - len(cached_games)
    log.info("generating_more_bot_games", count=n_needed)
    
    all_games = cached_games
    depth_to_time = {3: 0.05, 5: 0.2}
    
    for i in tqdm(range(n_needed), desc="Bot Games"):
        r = os.urandom(1)[0] / 255.0
        cumulative = 0
        chosen_depth = 3
        for depth, prob in depth_mix.items():
            cumulative += prob
            if r <= cumulative:
                chosen_depth = depth
                break
        
        bot = RamoraBot(time_limit=depth_to_time.get(chosen_depth, 0.05))
        board = Board()
        state = GameState.from_board(board)
        moves = []
        
        while not board.check_win() and board.legal_move_count() > 0:
            try:
                q, r = bot.get_move(state, board)
                moves.append((q, r))
                state = state.apply_move(board, q, r)
            except Exception as e:
                log.error("generation_error", error=str(e))
                break
                
        winner = board.winner()
        save_bot_game(moves, winner)
        all_games.append(moves)
        
    return all_games

def main():
    parser = argparse.ArgumentParser(description="Generate and manage game corpus for pretraining.")
    parser.add_argument("--bot-games", type=int, default=50, help="Target number of bot games.")
    parser.add_argument("--human-pages", type=int, default=10, help="Number of human game pages to scrape.")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use existing cached games.")
    parser.add_argument("--force-regenerate", action="store_true", help="Delete cache and start fresh.")
    
    args = parser.parse_args()
    
    if args.force_regenerate:
        log.info("force_regenerate_enabled")
        
    # 1. Human games
    log.info("collecting_human_games")
    human_games = scrape_hexo_did(
        max_pages=args.human_pages, 
        use_cache=args.use_cache and not args.force_regenerate
    )
    
    # 2. Bot games
    log.info("collecting_bot_games")
    bot_games = generate_bot_games(
        n_target=args.bot_games,
        depth_mix={3: 0.7, 5: 0.3},
        force_regenerate=args.force_regenerate
    )
    
    total = len(human_games) + len(bot_games)
    log.info("corpus_collection_complete", 
             human=len(human_games), 
             bot=len(bot_games), 
             total=total)

if __name__ == "__main__":
    main()
