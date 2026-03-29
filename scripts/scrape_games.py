
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.bootstrap.scraper import HexoDidScraper, HexoComScraper, deduplicate_games

def main():
    parser = argparse.ArgumentParser(description="Scrape game archives for Hex Tac Toe.")
    parser.add_argument("--source", type=str, choices=["hexo_did", "hexo_com", "all"], default="hexo_did",
                        help="Which source to scrape from.")
    parser.add_argument("--pages", type=int, default=10, help="Number of pages to scrape.")
    parser.add_argument("--output", type=str, default="data/scraped_games.json", help="Where to save the games.")
    parser.add_argument("--rated-only", action="store_true", default=True, help="Only scrape rated games.")
    
    args = parser.parse_args()
    
    scrapers = []
    if args.source == "hexo_did" or args.source == "all":
        scrapers.append(HexoDidScraper())
    if args.source == "hexo_com" or args.source == "all":
        scrapers.append(HexoComScraper())
        
    all_game_moves = []
    
    for scraper in scrapers:
        print(f"Scraping from {scraper.base_url}...")
        for page in range(1, args.pages + 1):
            print(f"  Page {page}/{args.pages}...")
            games = scraper.fetch_games_list(page=page)
            if not games:
                break
                
            for game in games:
                if args.rated_only and not game.get('gameOptions', {}).get('rated'):
                    continue
                
                if game.get('moveCount', 0) < 20:
                    continue
                    
                details = scraper.fetch_game_details(game['id'])
                if details and 'moves' in details:
                    moves = [(m['x'], m['y']) for m in details['moves']]
                    all_game_moves.append(moves)
                
                time.sleep(0.1)
            time.sleep(0.5)
            
    unique_games = deduplicate_games(all_game_moves)
    print(f"Total unique games scraped: {len(unique_games)}")
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([[[q, r] for q, r in g] for g in unique_games], f)
        
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    import json
    import time
    main()
