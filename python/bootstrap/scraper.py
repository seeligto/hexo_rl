import requests
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import structlog
from pathlib import Path
from abc import ABC, abstractmethod

log = structlog.get_logger()

# Default storage for raw human games
RAW_HUMAN_DIR = Path("data/corpus/raw_human")

class GameArchiveScraper(ABC):
    def __init__(self, base_url: str, api_prefix: str = "/api", storage_dir: Path = RAW_HUMAN_DIR):
        self.base_url = base_url.rstrip('/')
        self.api_prefix = api_prefix.rstrip('/')
        self.headers = {
            'User-Agent': 'HexTacToe-AZ-Scraper/0.1.0'
        }
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_games_list(self, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def fetch_game_details(self, game_id: str) -> Optional[Dict[str, Any]]:
        pass

    def get_cached_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        cache_file = self.storage_dir / f"{game_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                log.error("cache_read_error", game_id=game_id, error=str(e))
        return None

    def save_to_cache(self, game_id: str, data: Dict[str, Any]) -> None:
        cache_file = self.storage_dir / f"{game_id}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            log.error("cache_write_error", game_id=game_id, error=str(e))

class HexoDidScraper(GameArchiveScraper):
    def __init__(self, storage_dir: Path = RAW_HUMAN_DIR):
        super().__init__("https://[site-redacted]", storage_dir=storage_dir)

    def fetch_games_list(self, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        url = f"{self.base_url}{self.api_prefix}/finished-games"
        params = {'page': page, 'limit': limit}
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return resp.json().get('games', [])
        except Exception as e:
            log.error("fetch_error", url=url, error=str(e))
        return []

    def fetch_game_details(self, game_id: str) -> Optional[Dict[str, Any]]:
        # Check cache first
        cached = self.get_cached_game(game_id)
        if cached:
            return cached

        url = f"{self.base_url}{self.api_prefix}/finished-games/{game_id}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.save_to_cache(game_id, data)
                return data
        except Exception as e:
            log.error("fetch_error", url=url, error=str(e))
        return None

class HexoComScraper(GameArchiveScraper):
    def __init__(self, storage_dir: Path = RAW_HUMAN_DIR):
        super().__init__("https://[site-redacted]", storage_dir=storage_dir)

    def fetch_games_list(self, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        # TODO: Implement once public API for listing is confirmed
        log.warning("not_implemented", source="[site-redacted]")
        return []

    def fetch_game_details(self, game_id: str) -> Optional[Dict[str, Any]]:
        # Check cache first
        cached = self.get_cached_game(game_id)
        if cached:
            return cached

        url = f"{self.base_url}{self.api_prefix}/games/{game_id}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.save_to_cache(game_id, data)
                return data
        except Exception as e:
            log.error("fetch_error", url=url, error=str(e))
        return None

def deduplicate_games(games: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    seen = set()
    unique_games = []
    for game in games:
        # Create a hash of the move sequence
        game_str = ",".join([f"{q},{r}" for q, r in game])
        game_hash = hashlib.sha256(game_str.encode()).hexdigest()
        if game_hash not in seen:
            seen.add(game_hash)
            unique_games.append(game)
    return unique_games

def scrape_hexo_did(max_pages: int = 10, rated_only: bool = True, use_cache: bool = True) -> List[List[Tuple[int, int]]]:
    scraper = HexoDidScraper()
    all_game_moves = []
    
    # If using cache, we can also just load everything from storage_dir first
    if use_cache:
        cached_count = 0
        for p in scraper.storage_dir.glob("*.json"):
            try:
                with open(p, 'r') as f:
                    game_details = json.load(f)
                    if rated_only and not game_details.get('gameOptions', {}).get('rated'):
                        continue
                    if 'moves' in game_details:
                        moves = [(m['x'], m['y']) for m in game_details['moves']]
                        all_game_moves.append(moves)
                        cached_count += 1
            except:
                continue
        if cached_count > 0:
            log.info("loaded_from_cache", count=cached_count)

    # Then scrape new pages if needed
    for page in range(1, max_pages + 1):
        log.info("scraping_page", page=page)
        games = scraper.fetch_games_list(page=page)
        if not games:
            break
            
        for game in games:
            if rated_only and not game.get('gameOptions', {}).get('rated'):
                continue
            
            # Check move count
            if game.get('moveCount', 0) < 20:
                continue
                
            game_details = scraper.fetch_game_details(game['id'])
            if game_details and 'moves' in game_details:
                moves = [(move['x'], move['y']) for move in game_details['moves']]
                all_game_moves.append(moves)
            
            time.sleep(0.1) # Be polite
            
        time.sleep(0.5) # Be polite
        
    log.info("scraping_complete", raw_count=len(all_game_moves))
    unique_moves = deduplicate_games(all_game_moves)
    log.info("deduplication_complete", unique_count=len(unique_moves))
    return unique_moves

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Smoke test
    games = scrape_hexo_did(max_pages=args.pages, use_cache=not args.no_cache)
    print(f"Scraped {len(games)} unique games.")
