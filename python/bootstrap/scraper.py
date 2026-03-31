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

# ~480 most recent games accessible unauthenticated; full archive requires
# WolverinDEV export — see memory note
UNAUTHENTICATED_GAME_LIMIT = 480


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
    def fetch_games_list(self, page: int = 1, page_size: int = 20, base_timestamp: Optional[int] = None) -> List[Dict[str, Any]]:
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
        super().__init__("https://hexo.did.science", storage_dir=storage_dir)

    def fetch_games_list(self, page: int = 1, page_size: int = 20, base_timestamp: Optional[int] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}{self.api_prefix}/finished-games"
        params: Dict[str, Any] = {'page': page, 'pageSize': page_size}
        if base_timestamp is not None:
            params['before'] = base_timestamp
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return resp.json().get('games', [])
        except Exception as e:
            log.error("fetch_error", url=url, error=str(e))
        return []

    def fetch_game_details(self, game_id: str) -> Optional[Dict[str, Any]]:
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
        super().__init__("https://he-xo.com", storage_dir=storage_dir)

    def fetch_games_list(self, page: int = 1, page_size: int = 20, base_timestamp: Optional[int] = None) -> List[Dict[str, Any]]:
        # TODO: Implement once public API for listing is confirmed
        log.warning("not_implemented", source="he-xo.com")
        return []

    def fetch_game_details(self, game_id: str) -> Optional[Dict[str, Any]]:
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


def _passes_filter(game: Dict[str, Any]) -> bool:
    """Filter using listing-endpoint summary fields before fetching full records."""
    if not game.get('gameOptions', {}).get('rated'):
        return False
    if game.get('moveCount', 0) < 20:
        return False
    if game.get('gameResult', {}).get('reason') != 'six-in-a-row':
        return False
    return True


def deduplicate_games(games: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    seen = set()
    unique_games = []
    for game in games:
        game_str = ",".join([f"{q},{r}" for q, r in game])
        game_hash = hashlib.sha256(game_str.encode()).hexdigest()
        if game_hash not in seen:
            seen.add(game_hash)
            unique_games.append(game)
    return unique_games


def scrape_hexo_did(
    max_pages: int = 10,
    page_size: int = 20,
    use_cache: bool = True,
) -> List[List[Tuple[int, int]]]:
    scraper = HexoDidScraper()
    all_game_moves = []

    if use_cache:
        cached_count = 0
        for p in scraper.storage_dir.glob("*.json"):
            try:
                with open(p, 'r') as f:
                    game_details = json.load(f)
                    if not game_details.get('gameOptions', {}).get('rated'):
                        continue
                    if 'moves' in game_details:
                        # site (x,y) == native_core (q,r) — pointy-top axial, no conversion
                        moves = [(m['x'], m['y']) for m in game_details['moves']]
                        all_game_moves.append(moves)
                        cached_count += 1
            except Exception:
                continue
        if cached_count > 0:
            log.info("loaded_from_cache", count=cached_count)

    # Fix baseTimestamp at scrape-run start to prevent pagination drift from new games added mid-run
    import time as _time
    base_timestamp = int(_time.time() * 1000)
    log.info("scrape_start", base_timestamp=base_timestamp, max_pages=max_pages, page_size=page_size)

    fetched_ids: List[str] = []

    for page in range(1, max_pages + 1):
        # Hard stop before hitting the unauthenticated API wall
        games_seen_so_far = (page - 1) * page_size
        if games_seen_so_far >= UNAUTHENTICATED_GAME_LIMIT:
            log.warning(
                "approaching_unauthenticated_api_limit",
                games_seen=games_seen_so_far,
                limit=UNAUTHENTICATED_GAME_LIMIT,
            )
            break

        log.info("scraping_page", page=page)
        games = scraper.fetch_games_list(page=page, page_size=page_size, base_timestamp=base_timestamp)
        if not games:
            log.info("empty_page_stop", page=page)
            break

        for game in games:
            if not _passes_filter(game):
                continue

            game_id = game['id']
            game_details = scraper.fetch_game_details(game_id)
            if game_details and 'moves' in game_details:
                # site (x,y) == native_core (q,r) — pointy-top axial, no conversion
                moves = [(move['x'], move['y']) for move in game_details['moves']]
                all_game_moves.append(moves)
                fetched_ids.append(game_id)

            time.sleep(0.1)  # Be polite

        time.sleep(0.5)  # Be polite

    log.info("scraping_complete", raw_count=len(all_game_moves))
    unique_moves = deduplicate_games(all_game_moves)
    log.info("deduplication_complete", unique_count=len(unique_moves))
    return unique_moves, fetched_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    games, ids = scrape_hexo_did(
        max_pages=args.pages,
        page_size=args.page_size,
        use_cache=not args.no_cache,
    )
    print(f"Scraped {len(games)} unique games.")
    if ids:
        print("First 3 filtered game IDs:")
        for gid in ids[:3]:
            print(f"  {gid}")
