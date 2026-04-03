import requests
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
from pathlib import Path
from abc import ABC, abstractmethod

import yaml

log = structlog.get_logger()

# Default storage for raw human games
RAW_HUMAN_DIR = Path("data/corpus/raw_human")

# 500 most recent games accessible unauthenticated (confirmed from source:
# apiQueryService.ts line 68-70 — page * pageSize >= 500 returns 401)
UNAUTHENTICATED_GAME_LIMIT = 500


def _load_scraper_config() -> Dict[str, Any]:
    """Load scraper defaults from configs/corpus.yaml."""
    config_path = Path("configs/corpus.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("scraper", {})
    return {}


class GameArchiveScraper(ABC):
    def __init__(self, base_url: str, api_prefix: str = "/api", storage_dir: Path = RAW_HUMAN_DIR):
        self.base_url = base_url.rstrip('/')
        self.api_prefix = api_prefix.rstrip('/')
        self.headers = {
            'User-Agent': 'HexTacToe-AZ-Scraper/0.2.0'
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
            params['baseTimestamp'] = base_timestamp
        try:
            log.debug("http_request", url=url, params=params)
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
            log.debug("http_request", url=url)
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.save_to_cache(game_id, data)
                return data
        except Exception as e:
            log.error("fetch_error", url=url, error=str(e))
        return None

    def fetch_leaderboard(self) -> List[Dict[str, Any]]:
        """Fetch the current leaderboard from /api/leaderboard."""
        url = f"{self.base_url}{self.api_prefix}/leaderboard"
        try:
            log.debug("http_request", url=url)
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            log.error("fetch_leaderboard_error", url=url, error=str(e))
        return []

    def fetch_profile_games(self, profile_id: str) -> List[Dict[str, Any]]:
        """Fetch up to 10 recent games for a player profile.

        This endpoint is independent of the 500-game public finished-games
        cap, so it can surface games outside that window.
        """
        url = f"{self.base_url}{self.api_prefix}/profiles/{profile_id}/games"
        try:
            log.debug("http_request", url=url)
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Response may be a list or a dict with a 'games' key
                if isinstance(data, list):
                    return data
                return data.get('games', [])
        except Exception as e:
            log.error("fetch_profile_games_error", profile_id=profile_id, error=str(e))
        return []


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


def _extract_player_elos(game_details: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Extract player Elo values from a full game record.

    Returns (player_black_elo, player_white_elo) where black=players[0],
    white=players[1] per the API schema.  Returns None for missing/null values.
    """
    players = game_details.get('players', [])
    elo_black = None
    elo_white = None
    if len(players) >= 1:
        elo_val = players[0].get('elo')
        if elo_val is not None:
            elo_black = int(elo_val)
    if len(players) >= 2:
        elo_val = players[1].get('elo')
        if elo_val is not None:
            elo_white = int(elo_val)
    return elo_black, elo_white


def _enrich_with_elo(game_details: Dict[str, Any]) -> Dict[str, Any]:
    """Add player_black_elo and player_white_elo fields to a game record."""
    elo_black, elo_white = _extract_player_elos(game_details)
    game_details['player_black_elo'] = elo_black
    game_details['player_white_elo'] = elo_white
    return game_details


def _passes_elo_filter(game_details: Dict[str, Any], min_elo: int) -> bool:
    """Return True if both players meet the minimum Elo threshold.

    Games with missing Elo (unrated/null) are excluded when min_elo > 0.
    """
    if min_elo <= 0:
        return True
    elo_black = game_details.get('player_black_elo')
    elo_white = game_details.get('player_white_elo')
    if elo_black is None or elo_white is None:
        return False
    return elo_black >= min_elo and elo_white >= min_elo


def _get_player_profile_ids(game_details: Dict[str, Any]) -> Set[str]:
    """Extract profileId values from a game record's players list."""
    ids: Set[str] = set()
    for player in game_details.get('players', []):
        pid = player.get('profileId')
        if pid:
            ids.add(pid)
    return ids


def _passes_top_players_filter(game_details: Dict[str, Any], top_player_ids: Set[str]) -> bool:
    """Return True if at least one player is in the top-player set."""
    if not top_player_ids:
        return True
    game_pids = _get_player_profile_ids(game_details)
    return bool(game_pids & top_player_ids)


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
    min_elo: int = 0,
    top_players_only: bool = False,
    top_n: int = 20,
    req_delay: float = 1.0,
) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
    scraper = HexoDidScraper()
    all_game_moves: List[List[Tuple[int, int]]] = []

    if use_cache:
        cached_count = 0
        for p in scraper.storage_dir.glob("*.json"):
            try:
                with open(p, 'r') as f:
                    game_details = json.load(f)
                    if not game_details.get('gameOptions', {}).get('rated'):
                        continue
                    if 'moves' in game_details:
                        # site (x,y) == engine (q,r) — pointy-top axial, no conversion
                        moves = [(m['x'], m['y']) for m in game_details['moves']]
                        all_game_moves.append(moves)
                        cached_count += 1
            except Exception:
                continue
        if cached_count > 0:
            log.info("loaded_from_cache", count=cached_count)

    fetched_ids: List[str] = []
    empty_streak = 0

    # --- Standard paginated scrape (up to 500-game public window) ---
    import time as _time
    base_timestamp = int(_time.time() * 1000)
    log.info("scrape_start", base_timestamp=base_timestamp, max_pages=max_pages,
             page_size=page_size, min_elo=min_elo, top_players_only=top_players_only)

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

        page_fetched = 0
        for game in games:
            if not _passes_filter(game):
                continue

            game_id = game['id']
            game_details = scraper.fetch_game_details(game_id)
            if not game_details or 'moves' not in game_details:
                time.sleep(req_delay)
                continue

            # Enrich with Elo fields
            _enrich_with_elo(game_details)

            # Apply post-fetch filters
            if not _passes_elo_filter(game_details, min_elo):
                time.sleep(req_delay)
                continue

            # Re-save with Elo enrichment
            scraper.save_to_cache(game_id, game_details)

            moves = [(move['x'], move['y']) for move in game_details['moves']]
            all_game_moves.append(moves)
            fetched_ids.append(game_id)
            page_fetched += 1

            time.sleep(req_delay)

        if page_fetched == 0:
            empty_streak += 1
            if empty_streak >= 3:
                log.info("no_qualifying_games_stop", consecutive_empty_pages=empty_streak)
                break
        else:
            empty_streak = 0

    # --- Top-player profile scrape (reaches beyond the 500-game window) ---
    if top_players_only:
        leaderboard = scraper.fetch_leaderboard()
        top_profile_ids = []
        for entry in leaderboard[:top_n]:
            pid = entry.get('profileId')
            if pid:
                top_profile_ids.append(pid)
        log.info("top_players_resolved", count=len(top_profile_ids), top_n=top_n)
        time.sleep(req_delay)

        for profile_id in top_profile_ids:
            profile_games = scraper.fetch_profile_games(profile_id)
            log.info("profile_games_fetched", profile_id=profile_id, count=len(profile_games))
            time.sleep(req_delay)

            for game in profile_games:
                if not _passes_filter(game):
                    continue

                game_id = game.get('id')
                if not game_id:
                    continue

                game_details = scraper.fetch_game_details(game_id)
                if not game_details or 'moves' not in game_details:
                    time.sleep(req_delay)
                    continue

                _enrich_with_elo(game_details)

                if not _passes_elo_filter(game_details, min_elo):
                    time.sleep(req_delay)
                    continue

                scraper.save_to_cache(game_id, game_details)

                moves = [(move['x'], move['y']) for move in game_details['moves']]
                all_game_moves.append(moves)
                fetched_ids.append(game_id)

                time.sleep(req_delay)

    log.info("scraping_complete", raw_count=len(all_game_moves))
    unique_moves = deduplicate_games(all_game_moves)
    log.info("deduplication_complete", unique_count=len(unique_moves))
    return unique_moves, fetched_ids


if __name__ == "__main__":
    import argparse

    # Load config defaults
    scraper_cfg = _load_scraper_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--min-elo", type=int,
                        default=scraper_cfg.get("min_elo", 0),
                        help="Skip games where either player's Elo is below threshold")
    parser.add_argument("--top-players-only", action="store_true",
                        default=scraper_cfg.get("top_players_only", False),
                        help="Only keep games with at least one top-N leaderboard player")
    parser.add_argument("--top-n", type=int,
                        default=scraper_cfg.get("top_n", 20),
                        help="Number of top leaderboard players to include (default 20)")
    parser.add_argument("--req-delay", type=float,
                        default=scraper_cfg.get("req_delay", 1.0),
                        help="Seconds between API requests (default 1.0)")
    args = parser.parse_args()

    games, ids = scrape_hexo_did(
        max_pages=args.pages,
        page_size=args.page_size,
        use_cache=not args.no_cache,
        min_elo=args.min_elo,
        top_players_only=args.top_players_only,
        top_n=args.top_n,
        req_delay=args.req_delay,
    )
    print(f"Scraped {len(games)} unique games.")
    if ids:
        print("First 3 filtered game IDs:")
        for gid in ids[:3]:
            print(f"  {gid}")
