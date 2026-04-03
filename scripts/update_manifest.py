"""Update data/corpus/manifest.json by scanning files on disk.

Safe to run while scrapers or generators are active — reads are atomic
at the directory level and the final write is an atomic rename.

Usage:
    python scripts/update_manifest.py
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_HUMAN_DIR = REPO_ROOT / "data" / "corpus" / "raw_human"
BOT_GAMES_DIR = REPO_ROOT / "data" / "corpus" / "bot_games"
INJECTED_DIR = REPO_ROOT / "data" / "corpus" / "injected"
MANIFEST_PATH = REPO_ROOT / "data" / "corpus" / "manifest.json"


def _count_json(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.json")))


def _count_json_recursive(directory: Path) -> dict[str, int]:
    """Count JSON files per subdirectory."""
    if not directory.exists():
        return {}
    counts: dict[str, int] = {}
    for sub in sorted(directory.iterdir()):
        if sub.is_dir():
            n = len(list(sub.glob("*.json")))
            if n > 0:
                counts[sub.name] = n
    return counts


def _human_date_range() -> tuple[str, str]:
    """Extract oldest/newest created_at from human game files."""
    oldest, newest = "", ""
    dates: list[str] = []
    for p in RAW_HUMAN_DIR.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            d = data.get("created_at", "")
            if d:
                dates.append(d)
        except Exception:
            continue
    if dates:
        dates.sort()
        oldest, newest = dates[0], dates[-1]
    return oldest, newest


def elo_band_key(elo: int | None) -> str:
    """Assign an Elo value to a band key. Returns 'unrated' for None."""
    if elo is None:
        return "unrated"
    if elo < 1000:
        return "sub_1000"
    if elo < 1200:
        return "1000_1200"
    if elo < 1400:
        return "1200_1400"
    return "1400_plus"


def _compute_elo_bands() -> dict[str, int]:
    """Scan human game files and bucket by max(player_black_elo, player_white_elo)."""
    bands: dict[str, int] = {
        "sub_1000": 0,
        "1000_1200": 0,
        "1200_1400": 0,
        "1400_plus": 0,
        "unrated": 0,
    }
    for p in RAW_HUMAN_DIR.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            continue
        elo_b = data.get("player_black_elo")
        elo_w = data.get("player_white_elo")
        if elo_b is None and elo_w is None:
            bands["unrated"] += 1
        else:
            vals = [v for v in (elo_b, elo_w) if v is not None]
            max_elo = max(vals) if vals else None
            bands[elo_band_key(max_elo)] += 1
    return bands


def main() -> None:
    human_count = _count_json(RAW_HUMAN_DIR)
    bot_breakdown = _count_json_recursive(BOT_GAMES_DIR)
    bot_total = sum(bot_breakdown.values())
    injected_count = _count_json(INJECTED_DIR)
    oldest, newest = _human_date_range()
    elo_bands = _compute_elo_bands()

    manifest = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "human_games": human_count,
        "bot_games": bot_total,
        "bot_breakdown": bot_breakdown,
        "injected_games": injected_count,
        "total_games": human_count + bot_total + injected_count,
        "human_date_range": {"oldest": oldest, "newest": newest},
        "elo_bands": elo_bands,
        "filter": {"rated": True, "min_moves": 20, "reason": "six-in-a-row"},
    }

    # Atomic write: write to temp file then rename
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=MANIFEST_PATH.parent, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(manifest, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.rename(MANIFEST_PATH)

    total = human_count + bot_total + injected_count
    elo_summary = ", ".join(f"{k}={v}" for k, v in elo_bands.items() if v > 0)
    print(
        f"Manifest updated: {human_count} human, {bot_total} bot "
        f"({', '.join(f'{k}={v}' for k, v in bot_breakdown.items())}), "
        f"{injected_count} injected, {total} total"
    )
    if elo_summary:
        print(f"Elo bands: {elo_summary}")


if __name__ == "__main__":
    main()
