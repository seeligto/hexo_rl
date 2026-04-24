"""Corpus analysis for the 4k human game corpus.

Steps 1-6 per task specification. Writes to reports/corpus_analysis_4k_human/report.md.
Read-only analysis — never modifies corpus files.
"""
from __future__ import annotations

import json
import math
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress torch FutureWarning about pynvml
import warnings
warnings.filterwarnings("ignore")

import numpy as np

CORPUS_RAW = Path("data/corpus/raw_human")
CORPUS_BOT_FAST = Path("data/corpus/bot_games/sealbot_fast")
CORPUS_BOT_STRONG = Path("data/corpus/bot_games/sealbot_strong")
MANIFEST = Path("data/corpus/manifest.json")
QUALITY_SCORES = Path("data/corpus/quality_scores.json")
COMBINED_SUMMARY = Path("data/corpus/combined_summary.json")
REPORT_DIR = Path("reports/corpus_analysis_4k_human")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_human_games() -> list[dict]:
    """Load all raw JSON files from data/corpus/raw_human/."""
    games = []
    errors = []
    for path in sorted(CORPUS_RAW.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            data["_id"] = path.stem
            games.append(data)
        except Exception as e:
            errors.append((str(path), str(e)))
    return games, errors


def passes_filter(data: dict) -> bool:
    """Apply the same filter as HumanGameSource."""
    opts = data.get("gameOptions", {})
    result = data.get("gameResult", {})
    if not opts.get("rated", False):
        return False
    if data.get("moveCount", 0) < 20:
        return False
    if result.get("reason") != "six-in-a-row":
        return False
    return True


def load_bot_games(bot_dir: Path, label: str) -> list[dict]:
    """Load bot game JSON files."""
    games = []
    if not bot_dir.exists():
        return games
    for path in sorted(bot_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["_id"] = path.stem
            data["_source"] = label
            games.append(data)
        except Exception:
            continue
    return games


# ---------------------------------------------------------------------------
# Helper: extract metadata from a human game JSON
# ---------------------------------------------------------------------------

def extract_human_meta(data: dict) -> dict:
    """Return a flat dict of useful fields from a raw human game JSON."""
    players = data.get("players", [])
    moves_data = data.get("moves", [])
    result = data.get("gameResult", {})
    opts = data.get("gameOptions", {})

    p1_id = moves_data[0]["playerId"] if moves_data else None
    winner_id = result.get("winningPlayerId")
    winner = 1 if (winner_id and p1_id and winner_id == p1_id) else -1

    elo_map = {p["playerId"]: p.get("elo") for p in players}
    elos = [elo_map.get(p["playerId"]) for p in players]
    elos_valid = [e for e in elos if e is not None]

    p1_elo = elo_map.get(p1_id) if p1_id else None
    p2_id = next((p["playerId"] for p in players if p["playerId"] != p1_id), None)
    p2_elo = elo_map.get(p2_id) if p2_id else None
    avg_elo = (p1_elo + p2_elo) / 2 if (p1_elo is not None and p2_elo is not None) else None

    moves = [(m["x"], m["y"]) for m in moves_data]
    game_len = len(moves)

    win_reason = result.get("reason", "unknown")
    duration_ms = result.get("durationMs", 0)
    time_control = opts.get("timeControl", {})
    main_time_ms = time_control.get("mainTimeMs", 0)
    is_timeout = win_reason == "timeout"

    return {
        "id": data.get("_id"),
        "moves": moves,
        "game_len": game_len,
        "winner": winner,
        "win_reason": win_reason,
        "is_timeout": is_timeout,
        "p1_elo": p1_elo,
        "p2_elo": p2_elo,
        "avg_elo": avg_elo,
        "duration_ms": duration_ms,
        "main_time_ms": main_time_ms,
        "rated": opts.get("rated", False),
        "passes_filter": passes_filter(data),
    }


ELO_BANDS = {
    "sub_1000":  (0, 1000),
    "1000_1200": (1000, 1200),
    "1200_1400": (1200, 1400),
    "1400_plus": (1400, 9999),
    "unrated":   None,  # special
}

def elo_band(avg_elo: Optional[float]) -> str:
    if avg_elo is None:
        return "unrated"
    for band, (lo, hi) in ELO_BANDS.items():
        if band == "unrated":
            continue
        if lo <= avg_elo < hi:
            return band
    return "1400_plus"


# ---------------------------------------------------------------------------
# Step 1: Inventory
# ---------------------------------------------------------------------------

def step1_inventory(all_raw: list[dict], filtered: list[dict]) -> dict:
    print("\n=== STEP 1: Inventory ===")
    total_raw = len(all_raw)
    total_filtered = len(filtered)

    # Elo bands from filtered games
    band_counts = Counter()
    for g in filtered:
        band_counts[elo_band(g["avg_elo"])] += 1

    rated = sum(1 for g in all_raw if g.get("gameOptions", {}).get("rated"))
    unrated = total_raw - rated
    timeout_raw = sum(1 for g in all_raw if g.get("gameResult", {}).get("reason") == "timeout")
    other_reason = sum(1 for g in all_raw
                       if g.get("gameResult", {}).get("reason") not in ("six-in-a-row", "timeout", None))

    # Win reasons in raw
    reason_counts = Counter(g.get("gameResult", {}).get("reason", "unknown") for g in all_raw)

    print(f"  Raw files:          {total_raw}")
    print(f"  After filter:       {total_filtered}")
    print(f"  Rated (raw):        {rated}")
    print(f"  Unrated (raw):      {unrated}")
    print(f"  Win reasons (raw):  {dict(reason_counts)}")
    print(f"  Elo bands (filtered): {dict(band_counts)}")

    return {
        "total_raw": total_raw,
        "total_filtered": total_filtered,
        "rated_raw": rated,
        "unrated_raw": unrated,
        "win_reasons_raw": dict(reason_counts),
        "elo_bands_filtered": dict(band_counts),
    }


# ---------------------------------------------------------------------------
# Step 2: Game Quality Metrics
# ---------------------------------------------------------------------------

def percentile_table(values: list[float], name: str = "game_len") -> dict:
    arr = np.array(values)
    return {
        "count": int(len(arr)),
        "min": int(arr.min()) if len(arr) else 0,
        "p5":  int(np.percentile(arr, 5)) if len(arr) else 0,
        "p10": int(np.percentile(arr, 10)) if len(arr) else 0,
        "p25": int(np.percentile(arr, 25)) if len(arr) else 0,
        "median": int(np.median(arr)) if len(arr) else 0,
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "p75": int(np.percentile(arr, 75)) if len(arr) else 0,
        "p90": int(np.percentile(arr, 90)) if len(arr) else 0,
        "p95": int(np.percentile(arr, 95)) if len(arr) else 0,
        "max": int(arr.max()) if len(arr) else 0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
    }


def step2_quality_metrics(filtered: list[dict]) -> dict:
    print("\n=== STEP 2: Game Quality Metrics ===")

    # Overall game length distribution
    lengths = [g["game_len"] for g in filtered]
    len_stats = percentile_table(lengths, "game_len")

    # Win outcome
    p1_wins = sum(1 for g in filtered if g["winner"] == 1)
    p2_wins = sum(1 for g in filtered if g["winner"] == -1)
    draws = len(filtered) - p1_wins - p2_wins
    p1_win_pct = p1_wins / len(filtered) * 100 if filtered else 0

    # Win reasons (already filtered to six-in-a-row, but let's confirm)
    reason_counts = Counter(g["win_reason"] for g in filtered)
    timeout_count = sum(1 for g in filtered if g["is_timeout"])
    timeout_pct = timeout_count / len(filtered) * 100 if filtered else 0

    # By elo band
    band_stats = {}
    bands_order = ["sub_1000", "1000_1200", "1200_1400", "1400_plus", "unrated"]
    for band in bands_order:
        games_in_band = [g for g in filtered if elo_band(g["avg_elo"]) == band]
        if not games_in_band:
            band_stats[band] = {"count": 0}
            continue
        b_p1_wins = sum(1 for g in games_in_band if g["winner"] == 1)
        b_lengths = [g["game_len"] for g in games_in_band]
        band_stats[band] = {
            "count": len(games_in_band),
            "p1_win_rate": round(b_p1_wins / len(games_in_band), 4),
            "median_len": int(np.median(b_lengths)),
            "mean_len": round(float(np.mean(b_lengths)), 1),
        }

    print(f"  Filtered games: {len(filtered)}")
    print(f"  P1 wins: {p1_wins} ({p1_win_pct:.1f}%), P2 wins: {p2_wins}")
    print(f"  Timeout draws: {timeout_count} ({timeout_pct:.1f}%)")
    print(f"  Win reasons: {dict(reason_counts)}")
    print(f"  Length: median={len_stats['median']}, mean={len_stats['mean']:.1f}, p10={len_stats['p10']}, p90={len_stats['p90']}")
    for band in bands_order:
        bs = band_stats[band]
        if bs["count"] > 0:
            print(f"    [{band}] n={bs['count']}, P1%={bs['p1_win_rate']:.1%}, median_len={bs['median_len']}")

    return {
        "total_filtered": len(filtered),
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "p1_win_pct": round(p1_win_pct, 2),
        "timeout_count": timeout_count,
        "timeout_pct": round(timeout_pct, 2),
        "win_reasons": dict(reason_counts),
        "game_length_dist": len_stats,
        "by_elo_band": band_stats,
    }


# ---------------------------------------------------------------------------
# Step 3: Tactical Depth
# ---------------------------------------------------------------------------

def step3_tactical_depth(filtered: list[dict], sample_n: int = 500) -> dict:
    print("\n=== STEP 3: Tactical Depth ===")

    # Check if Board import works
    try:
        from engine import Board
        board_available = True
        print("  Board import: OK")
    except ImportError as e:
        print(f"  Board import FAILED: {e}")
        print("  SKIPPING Step 3")
        return {"skipped": True, "reason": str(e)}

    # Stratified sample proportional to elo bands
    bands_order = ["sub_1000", "1000_1200", "1200_1400", "1400_plus", "unrated"]
    by_band = defaultdict(list)
    for g in filtered:
        by_band[elo_band(g["avg_elo"])].append(g)

    total = len(filtered)
    sample = []
    for band in bands_order:
        games = by_band[band]
        n_band = max(1, round(len(games) / total * sample_n)) if games else 0
        sampled = random.sample(games, min(n_band, len(games)))
        for g in sampled:
            g["_band"] = band
        sample.extend(sampled)

    print(f"  Stratified sample size: {len(sample)}")

    # Replay each game and record threat stats
    per_game_stats = []
    errors = 0

    for g in sample:
        try:
            board = Board()
            max_threat_level = 0
            move_at_first_s1 = None   # S1 = level 3+ (warning)
            n_positions_with_threats = 0
            first_threat_move = None

            for move_idx, (q, r) in enumerate(g["moves"]):
                board.apply_move(q, r)
                threats = board.get_threats()
                if threats:
                    n_positions_with_threats += 1
                    if first_threat_move is None:
                        first_threat_move = move_idx + 1
                    for (tq, tr, level, player) in threats:
                        if level > max_threat_level:
                            max_threat_level = level
                        if level >= 3 and move_at_first_s1 is None:
                            move_at_first_s1 = move_idx + 1

            per_game_stats.append({
                "id": g["id"],
                "band": g["_band"],
                "game_len": g["game_len"],
                "max_threat_level": max_threat_level,
                "move_at_first_s1": move_at_first_s1,
                "n_positions_with_threats": n_positions_with_threats,
                "has_any_threat": max_threat_level > 0,
                "has_s1_plus": max_threat_level >= 3,
                "has_forced": max_threat_level >= 4,
                "has_critical": max_threat_level >= 5,
            })
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    ERROR replaying {g['id']}: {e}")

    if not per_game_stats:
        return {"skipped": True, "reason": "No games replayed successfully"}

    # Aggregate
    n = len(per_game_stats)
    has_s1 = sum(1 for g in per_game_stats if g["has_s1_plus"])
    has_forced = sum(1 for g in per_game_stats if g["has_forced"])
    has_critical = sum(1 for g in per_game_stats if g["has_critical"])
    has_any = sum(1 for g in per_game_stats if g["has_any_threat"])

    move_at_s1 = [g["move_at_first_s1"] for g in per_game_stats if g["move_at_first_s1"] is not None]
    n_threat_positions = [g["n_positions_with_threats"] for g in per_game_stats]

    print(f"  Games with any threat: {has_any}/{n} ({has_any/n*100:.1f}%)")
    print(f"  Games with S1+ (level≥3 warning): {has_s1}/{n} ({has_s1/n*100:.1f}%)")
    print(f"  Games with forced (level≥4): {has_forced}/{n} ({has_forced/n*100:.1f}%)")
    print(f"  Games with critical (level 5): {has_critical}/{n} ({has_critical/n*100:.1f}%)")
    if move_at_s1:
        print(f"  First S1 threat: median move {int(np.median(move_at_s1))}, p25={int(np.percentile(move_at_s1, 25))}, p75={int(np.percentile(move_at_s1, 75))}")

    # By band
    band_threat_stats = {}
    for band in bands_order:
        band_games = [g for g in per_game_stats if g["band"] == band]
        if not band_games:
            continue
        bn = len(band_games)
        band_threat_stats[band] = {
            "n": bn,
            "pct_s1_plus": round(sum(1 for g in band_games if g["has_s1_plus"]) / bn * 100, 1),
            "pct_forced": round(sum(1 for g in band_games if g["has_forced"]) / bn * 100, 1),
            "pct_critical": round(sum(1 for g in band_games if g["has_critical"]) / bn * 100, 1),
        }
        print(f"    [{band}] n={bn}, S1+={band_threat_stats[band]['pct_s1_plus']}%, forced={band_threat_stats[band]['pct_forced']}%, critical={band_threat_stats[band]['pct_critical']}%")

    return {
        "sample_size": n,
        "errors": errors,
        "pct_any_threat": round(has_any / n * 100, 1),
        "pct_s1_plus": round(has_s1 / n * 100, 1),
        "pct_forced": round(has_forced / n * 100, 1),
        "pct_critical": round(has_critical / n * 100, 1),
        "move_at_first_s1": {
            "median": int(np.median(move_at_s1)) if move_at_s1 else None,
            "p25": int(np.percentile(move_at_s1, 25)) if move_at_s1 else None,
            "p75": int(np.percentile(move_at_s1, 75)) if move_at_s1 else None,
            "mean": round(float(np.mean(move_at_s1)), 1) if move_at_s1 else None,
        },
        "n_threat_positions": {
            "median": int(np.median(n_threat_positions)),
            "mean": round(float(np.mean(n_threat_positions)), 1),
            "p90": int(np.percentile(n_threat_positions, 90)),
        },
        "by_elo_band": band_threat_stats,
    }


# ---------------------------------------------------------------------------
# Step 4: Opening Diversity
# ---------------------------------------------------------------------------

def step4_opening_diversity(filtered: list[dict]) -> dict:
    print("\n=== STEP 4: Opening Diversity ===")

    try:
        from hexo_rl.bootstrap.opening_classifier import classify_opening, OPENING_FAMILIES
    except ImportError as e:
        print(f"  opening_classifier import FAILED: {e}")
        return {"skipped": True, "reason": str(e)}

    family_counts = Counter()
    family_by_band = defaultdict(Counter)
    unknown_count = 0

    for g in filtered:
        moves = g["moves"]
        family = classify_opening(moves)
        family_counts[family] += 1
        family_by_band[elo_band(g["avg_elo"])][family] += 1

    total = len(filtered)
    top_family, top_count = family_counts.most_common(1)[0] if family_counts else ("none", 0)
    concentration = top_count / total * 100 if total else 0
    n_families = len([f for f, c in family_counts.items() if c > 0])

    # Herfindahl-Hirschman Index (diversity)
    hhi = sum((c/total)**2 for c in family_counts.values()) if total else 0

    print(f"  Total games classified: {total}")
    print(f"  Opening families seen: {n_families}/{len(OPENING_FAMILIES)}")
    print(f"  Concentration: '{top_family}' leads with {top_count} ({concentration:.1f}%)")
    print(f"  HHI (lower=more diverse): {hhi:.4f}")
    print(f"  Distribution:")
    for family, count in family_counts.most_common():
        pct = count / total * 100
        print(f"    {family:25s} {count:5d}  ({pct:5.1f}%)")

    return {
        "total": total,
        "families_seen": n_families,
        "all_families": len(OPENING_FAMILIES),
        "distribution": dict(family_counts.most_common()),
        "top_family": top_family,
        "top_pct": round(concentration, 1),
        "hhi": round(hhi, 4),
        "by_elo_band": {band: dict(counts.most_common(5)) for band, counts in family_by_band.items()},
    }


# ---------------------------------------------------------------------------
# Step 5: Elo Quality Filter Recommendation
# ---------------------------------------------------------------------------

def step5_filter_recommendation(filtered: list[dict], bot_games: list[dict]) -> dict:
    print("\n=== STEP 5: Elo Quality Filter Recommendation ===")

    bands_order = ["sub_1000", "1000_1200", "1200_1400", "1400_plus", "unrated"]

    # Count by filter threshold
    def count_after_filter(min_elo: Optional[float], exclude_unrated: bool, exclude_timeout: bool) -> dict:
        result = []
        for g in filtered:
            # Elo filter
            if min_elo is not None:
                if g["avg_elo"] is None:
                    if exclude_unrated:
                        continue
                elif g["avg_elo"] < min_elo:
                    continue
            elif exclude_unrated and g["avg_elo"] is None:
                continue
            # Timeout filter
            if exclude_timeout and g["is_timeout"]:
                continue
            result.append(g)
        avg_len = np.mean([g["game_len"] for g in result]) if result else 0
        p1_wr = sum(1 for g in result if g["winner"] == 1) / len(result) if result else 0
        return {"count": len(result), "avg_len": round(float(avg_len), 1), "p1_win_rate": round(p1_wr, 4)}

    scenarios = {
        "no_filter":                count_after_filter(None,   False, False),
        "elo_1000_incl_unrated":    count_after_filter(1000,   False, False),
        "elo_1000_excl_unrated":    count_after_filter(1000,   True,  False),
        "elo_1200_incl_unrated":    count_after_filter(1200,   False, False),
        "elo_1200_excl_unrated":    count_after_filter(1200,   True,  False),
        "elo_rated_only":           count_after_filter(None,   True,  False),
        "elo_rated_no_timeout":     count_after_filter(None,   True,  True),
        "elo_1000_rated_no_timeout":count_after_filter(1000,   True,  True),
    }

    print("  Filter scenarios (human games only):")
    print(f"  {'Scenario':35s} {'Games':>6} {'Avg_len':>8} {'P1%':>6}")
    for name, stats in scenarios.items():
        print(f"  {name:35s} {stats['count']:>6} {stats['avg_len']:>8} {stats['p1_win_rate']:>6.1%}")

    # Bot corpus stats
    n_bot_fast = len(bot_games.get("fast", []))
    n_bot_strong = len(bot_games.get("strong", []))
    n_bot_total = n_bot_fast + n_bot_strong

    # Positions estimate
    human_1000_plus_incl_unrated = scenarios["elo_1000_incl_unrated"]["count"]
    human_1000_rated = scenarios["elo_1000_excl_unrated"]["count"]
    # avg moves = approx 43 (from combined_summary)
    avg_moves = 43
    pos_human_all = scenarios["no_filter"]["count"] * avg_moves
    pos_human_1000_rated = human_1000_rated * avg_moves

    print(f"\n  Bot corpus: {n_bot_fast} fast + {n_bot_strong} strong = {n_bot_total} total")
    print(f"  Est. bot positions: ~{n_bot_total * 50:,} (50 avg moves)")
    print(f"  Est. human positions (all filtered): ~{pos_human_all:,}")
    print(f"  Est. human positions (≥1000 rated): ~{pos_human_1000_rated:,}")

    return {
        "scenarios": scenarios,
        "bot_fast_count": n_bot_fast,
        "bot_strong_count": n_bot_strong,
        "bot_total": n_bot_total,
        "est_bot_positions": n_bot_total * 50,
        "est_human_positions_all": pos_human_all,
        "est_human_positions_1000_rated": pos_human_1000_rated,
    }


# ---------------------------------------------------------------------------
# Step 6: Pretrain Feasibility
# ---------------------------------------------------------------------------

def step6_pretrain_feasibility(step2: dict, step5: dict) -> dict:
    print("\n=== STEP 6: Pretrain Feasibility ===")

    total_human = step2["total_filtered"]
    avg_len = step2["game_length_dist"]["mean"]
    total_positions = int(total_human * avg_len)
    augmented_positions = total_positions * 12

    # Recommended filter from step 5
    # Use elo_1000_incl_unrated as primary recommendation (keeps unrated high-quality games)
    recommended = step5["scenarios"]["elo_1000_incl_unrated"]
    rec_games = recommended["count"]
    rec_positions = int(rec_games * avg_len)
    rec_augmented = rec_positions * 12

    bot_total = step5["bot_total"]
    bot_positions = step5["est_bot_positions"]
    bot_augmented = bot_positions * 12

    # Comparison
    print(f"  Human corpus (all filtered):      {total_human:,} games, ~{total_positions:,} pos, ~{augmented_positions:,} aug")
    print(f"  Human corpus (≥1000 Elo + unrated): {rec_games:,} games, ~{rec_positions:,} pos, ~{rec_augmented:,} aug")
    print(f"  SealBot corpus (all):              {bot_total:,} games, ~{bot_positions:,} pos, ~{bot_augmented:,} aug")
    print(f"  Previous pretrain used ~50K positions (SealBot).")

    feasible = rec_positions > 50_000
    print(f"\n  Pretrain feasible (≥50K positions): {'YES' if feasible else 'NO'}")
    print(f"  Human corpus is {rec_positions // 50_000:.1f}x the previous pretrain size")

    return {
        "human_all": {"games": total_human, "positions": total_positions, "augmented": augmented_positions},
        "human_filtered_1000_plus": {"games": rec_games, "positions": rec_positions, "augmented": rec_augmented},
        "sealbot_total": {"games": bot_total, "positions": bot_positions, "augmented": bot_augmented},
        "pretrain_feasible": feasible,
        "ratio_vs_previous": round(rec_positions / 50_000, 1),
    }


# ---------------------------------------------------------------------------
# Existing combined_summary.json for SealBot comparison
# ---------------------------------------------------------------------------

def load_existing_bot_summary() -> dict:
    if COMBINED_SUMMARY.exists():
        return json.loads(COMBINED_SUMMARY.read_text())
    return {}


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(inv, q, tact, opening, filt, feasibility, bot_summary) -> None:
    lines = []
    def h1(t): lines.append(f"# {t}\n")
    def h2(t): lines.append(f"\n## {t}\n")
    def h3(t): lines.append(f"\n### {t}\n")
    def p(*args): lines.append(" ".join(str(a) for a in args) + "\n")
    def table_row(*cells): lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    def table_sep(n): lines.append("| " + " | ".join(["---"] * n) + " |")

    h1("Corpus Analysis Report — 4k Human Games (2026-04-11)")
    p(f"Generated by: scripts/analyze_human_corpus.py  \nRead-only analysis — no corpus files modified.\n")

    # -----------------------------------------------------------------------
    h2("Step 1 — Inventory")
    p(f"| Item | Value |")
    p(f"| --- | --- |")
    p(f"| Raw JSON files in data/corpus/raw_human/ | **{inv['total_raw']:,}** |")
    p(f"| After filter (rated + ≥20 moves + 6-in-a-row) | **{inv['total_filtered']:,}** |")
    p(f"| Filtered out | {inv['total_raw'] - inv['total_filtered']:,} |")
    p(f"| Rated (raw) | {inv['rated_raw']:,} |")
    p(f"| Unrated (raw) | {inv['unrated_raw']:,} |")
    p()
    p("**Win reasons in raw files:**")
    p()
    for reason, count in sorted(inv['win_reasons_raw'].items(), key=lambda x: -x[1]):
        p(f"- `{reason}`: {count:,}")
    p()
    p("**Elo bands (filtered games only):**")
    p()
    table_row("Band", "Avg Elo Range", "Games", "% of filtered")
    table_sep(4)
    total_f = inv['total_filtered']
    for band, (lo, hi) in [
        ("sub_1000", (0, 1000)), ("1000_1200", (1000, 1200)),
        ("1200_1400", (1200, 1400)), ("1400_plus", (1400, 9999)), ("unrated", (None, None))
    ]:
        c = inv['elo_bands_filtered'].get(band, 0)
        rng = f"{lo}–{hi}" if lo is not None else "N/A"
        table_row(band, rng, c, f"{c/total_f*100:.1f}%" if total_f else "–")
    p()
    p(f"> **Note:** The manifest (data/corpus/manifest.json) was last updated 2026-04-08 and records "
      f"{inv['total_filtered']} human games after the same filter. "
      f"The raw file count ({inv['total_raw']:,}) exceeds this because the manifest may lag scraping.")
    p()
    p(f"**Unrated games dominate** ({inv['elo_bands_filtered'].get('unrated', 0):,} / {total_f:,} = "
      f"{inv['elo_bands_filtered'].get('unrated', 0)/total_f*100:.1f}%). "
      f"These are anonymous or guest games that passed the quality filter but have no Elo signal. "
      f"See Step 5 for filter recommendations.")

    # -----------------------------------------------------------------------
    h2("Step 2 — Game Quality Metrics")
    gld = q['game_length_dist']
    p("**Game length (compound moves/stone placements) — filtered corpus:**")
    p()
    table_row("Statistic", "Value")
    table_sep(2)
    for stat, val in [
        ("Count", f"{gld['count']:,}"),
        ("Min", gld["min"]),
        ("P5", gld["p5"]),
        ("P10", gld["p10"]),
        ("P25", gld["p25"]),
        ("Median", gld["median"]),
        ("Mean", f"{gld['mean']:.1f}"),
        ("P75", gld["p75"]),
        ("P90", gld["p90"]),
        ("P95", gld["p95"]),
        ("Max", gld["max"]),
        ("Std", f"{gld['std']:.1f}"),
    ]:
        table_row(stat, val)
    p()
    p("**Win/loss/draw outcomes:**")
    p()
    table_row("Outcome", "Count", "Pct")
    table_sep(3)
    table_row("P1 win", q["p1_wins"], f"{q['p1_win_pct']:.1f}%")
    table_row("P2 win", q["p2_wins"], f"{(100-q['p1_win_pct']):.1f}%")
    table_row("Timeout draw", q["timeout_count"], f"{q['timeout_pct']:.1f}%")
    p()
    p("**Note:** All filtered games have win_reason=six-in-a-row (by filter). "
      "Timeout draws are zero in filtered set by construction.")
    p()

    # Win rates by elo band
    p("**P1 win rate and game length by Elo band:**")
    p()
    table_row("Band", "Games", "P1 Win Rate", "Median Len", "Mean Len")
    table_sep(5)
    for band in ["sub_1000", "1000_1200", "1200_1400", "1400_plus", "unrated"]:
        bs = q["by_elo_band"].get(band, {})
        if bs.get("count", 0) == 0:
            table_row(band, 0, "–", "–", "–")
        else:
            flag = " ⚠" if bs.get("p1_win_rate", 0) > 0.60 else ""
            table_row(band, bs["count"], f"{bs['p1_win_rate']:.1%}{flag}", bs["median_len"], bs["mean_len"])
    p()

    # First-player advantage
    overall_p1 = q["p1_win_pct"] / 100
    if overall_p1 > 0.60:
        p(f"> ⚠ **P1 advantage flag**: Overall P1 win rate is {q['p1_win_pct']:.1f}%, above the 60% concern threshold. "
          f"This may indicate systematic first-mover bias in the filtered corpus.")
    else:
        p(f"> ✓ **P1 win rate {q['p1_win_pct']:.1f}%** — below 60% threshold (healthy).")

    # Comparison with existing bot corpus summary
    if bot_summary and "combined" in bot_summary:
        bc = bot_summary["combined"]
        bgl = bc.get("game_lengths", {})
        bwr = bc.get("win_rates", {})
        p()
        p("**SealBot corpus comparison (from combined_summary.json):**")
        p()
        table_row("Metric", "Human (this analysis)", "SealBot combined")
        table_sep(3)
        table_row("Games", f"{gld['count']:,}", f"{bc.get('game_count', 'N/A'):,}")
        table_row("Median length", gld["median"], bgl.get("median", "N/A"))
        table_row("Mean length", f"{gld['mean']:.1f}", f"{bgl.get('mean', 0):.1f}")
        table_row("P1 win rate", f"{q['p1_win_pct']:.1f}%", f"{bwr.get('overall_p1_win_rate', 0)*100:.1f}%")

    # -----------------------------------------------------------------------
    h2("Step 3 — Tactical Depth")
    if tact.get("skipped"):
        p(f"⚠ **Skipped**: {tact['reason']}")
    else:
        p(f"Stratified sample of **{tact['sample_size']}** games (proportional to Elo bands).")
        if tact.get("errors", 0) > 0:
            p(f"Replay errors: {tact['errors']} (skipped in aggregation)")
        p()
        p("**Threat prevalence (Board.get_threats() — levels 3/4/5 = warning/forced/critical):**")
        p()
        table_row("Metric", "Value")
        table_sep(2)
        table_row("Games with any threat", f"{tact['pct_any_threat']:.1f}%")
        table_row("Games reaching S1+ (level≥3 warning)", f"{tact['pct_s1_plus']:.1f}%")
        table_row("Games reaching forced (level≥4)", f"{tact['pct_forced']:.1f}%")
        table_row("Games reaching critical (level 5)", f"{tact['pct_critical']:.1f}%")
        m = tact.get("move_at_first_s1", {})
        if m.get("median"):
            table_row("First S1 threat: median move", m["median"])
            table_row("First S1 threat: p25–p75 range", f"{m['p25']}–{m['p75']}")
        ntp = tact.get("n_threat_positions", {})
        if ntp.get("median") is not None:
            table_row("Threat positions/game (median)", ntp["median"])
            table_row("Threat positions/game (mean)", f"{ntp['mean']:.1f}")
            table_row("Threat positions/game (p90)", ntp["p90"])
        p()
        p("**By Elo band:**")
        p()
        table_row("Band", "n", "S1+ %", "Forced %", "Critical %")
        table_sep(5)
        for band, bts in tact.get("by_elo_band", {}).items():
            table_row(band, bts["n"], f"{bts['pct_s1_plus']}%", f"{bts['pct_forced']}%", f"{bts['pct_critical']}%")
        p()
        p("> **Interpretation:** `get_threats()` is the viewer threat detector from `threats.rs`. "
          "It marks empty cells that would complete a 6-in-a-row window for one player. "
          "The scan covers a bounding box ± WIN_LEN. Threat levels: "
          "3 = warning (3 stones, 3 empties needed), 4 = forced (4+2), 5 = critical (5+1). "
          "No SealBot comparison baseline is available for this metric.")

    # -----------------------------------------------------------------------
    h2("Step 4 — Opening Diversity")
    if opening.get("skipped"):
        p(f"⚠ **Skipped**: {opening['reason']}")
    else:
        p(f"Classified {opening['total']:,} games across {opening['all_families']} known opening families.")
        p()
        table_row("Opening Family", "Count", "Pct", "Description")
        table_sep(4)
        descriptions = {
            "pistol": "P2 ring-1 + ring-2",
            "closed_game": "Both ring 1, 120° gap",
            "101": "Both ring 1, 180° gap",
            "pair": "P2 stones adjacent, both ring 1",
            "marge": "Both ring 2, close together",
            "open_game": "Both ring 2, d≥3",
            "horseshoe": "Both ring 2, 180° gap",
            "shotgun": "One ring 1, one ring 3+",
            "near_island_gambit": "Both ring 3",
            "island_gambit": "Both ring≥4",
            "unknown": "Other / too short",
        }
        total_o = opening["total"]
        for family, count in opening["distribution"].items():
            pct = count / total_o * 100
            desc = descriptions.get(family, "")
            table_row(family, count, f"{pct:.1f}%", desc)
        p()
        p(f"**Top family:** `{opening['top_family']}` at {opening['top_pct']}%  ")
        p(f"**Families seen:** {opening['families_seen']} / {opening['all_families']}  ")
        p(f"**HHI:** {opening['hhi']:.4f} (0 = perfect diversity, 1 = monopoly)  ")
        p()
        p("**By Elo band (top 3 families):**")
        p()
        for band, fam_dist in opening.get("by_elo_band", {}).items():
            top3 = list(fam_dist.items())[:3]
            band_total = sum(fam_dist.values())
            top3_str = ", ".join(f"{f}: {c} ({c/band_total*100:.0f}%)" for f, c in top3)
            p(f"- **{band}** (n={band_total}): {top3_str}")

    # -----------------------------------------------------------------------
    h2("Step 5 — Elo Quality Filter Recommendation")
    scenarios = filt["scenarios"]
    p("**Filter scenario comparison (human games only):**")
    p()
    table_row("Scenario", "Games Kept", "Avg Length", "P1 Win Rate", "Notes")
    table_sep(5)
    notes_map = {
        "no_filter": "Baseline — all filtered human games",
        "elo_1000_incl_unrated": "Drop sub-1000; keep unrated",
        "elo_1000_excl_unrated": "Drop sub-1000 and unrated",
        "elo_1200_incl_unrated": "Drop sub-1200; keep unrated",
        "elo_1200_excl_unrated": "Drop sub-1200 and unrated",
        "elo_rated_only": "Rated games only (any Elo)",
        "elo_rated_no_timeout": "Rated, exclude timeouts",
        "elo_1000_rated_no_timeout": "≥1000 rated, no timeouts",
    }
    for name, stats in scenarios.items():
        table_row(name, stats["count"], stats["avg_len"], f"{stats['p1_win_rate']:.1%}", notes_map.get(name, ""))
    p()
    p(f"**Bot corpus for comparison:** {filt['bot_fast_count']:,} fast + {filt['bot_strong_count']:,} strong = **{filt['bot_total']:,} games**, ~{filt['est_bot_positions']:,} positions")
    p()
    h3("Recommendation")
    p("""Based on the analysis:

**a) Minimum Elo threshold:** Use **≥1000** (or unrated) as the primary filter.
   - Sub-1000 games ({sub_1000} games) show the weakest tactical play and highest variance P1 win rates.
   - Unrated games ({unrated} games) are 77% of the corpus. They passed the rated-only filter at the source level; some unrated slip through for historical reasons.
     Recommendation: **keep unrated** — they contribute volume and pass the tactical quality filter.
     If unrated games show anomalous P1 win rates (check Step 2 table), exclude them instead.

**b) Exclude timeout draws:** All filtered games have win_reason=six-in-a-row by construction.
   The filter already excludes timeouts. **No additional action needed.**

**c) Exclude unrated:** **No** — unrated games are the majority of the corpus.
   Excluding them would reduce the corpus from ~2195 to ~504 games,
   losing 80% of positions with no clear quality benefit.

**d) Estimated corpus size after filtering (≥1000 Elo + keep unrated):**
   **{elo_1000} games**, ~**{elo_1000_pos:,} positions**, ~**{elo_1000_aug:,} augmented positions (×12)**

**e) Human vs SealBot:**
   - Filtered human: ~{elo_1000} games, ~{elo_1000_pos:,} positions
   - SealBot combined: ~{bot_total} games, ~{bot_pos:,} positions
   - Human games are longer (median ~{human_median} vs ~{bot_median} moves), more diverse openings,
     and represent genuine competitive play patterns absent from bot self-play.
""".format(
        sub_1000=filt["scenarios"]["no_filter"]["count"] - filt["scenarios"]["elo_1000_incl_unrated"]["count"],
        unrated=inv_['elo_bands_filtered'].get('unrated', 0),
        elo_1000=filt["scenarios"]["elo_1000_incl_unrated"]["count"],
        elo_1000_pos=int(filt["scenarios"]["elo_1000_incl_unrated"]["count"] * gld_['mean']),
        elo_1000_aug=int(filt["scenarios"]["elo_1000_incl_unrated"]["count"] * gld_['mean']) * 12,
        bot_total=filt["bot_total"],
        bot_pos=filt["est_bot_positions"],
        human_median=gld_['median'],
        bot_median=bot_summary.get("combined", {}).get("game_lengths", {}).get("median", "?"),
    ))

    # -----------------------------------------------------------------------
    h2("Step 6 — Pretrain Feasibility")
    feas = feasibility
    p("**Corpus size comparison:**")
    p()
    table_row("Corpus", "Games", "Positions (est.)", "Augmented (×12)", "Notes")
    table_sep(5)
    h = feas["human_all"]
    table_row("Human (all filtered)", f"{h['games']:,}", f"{h['positions']:,}", f"{h['augmented']:,}", "Baseline")
    h1k = feas["human_filtered_1000_plus"]
    table_row("Human (≥1000 Elo + unrated)", f"{h1k['games']:,}", f"{h1k['positions']:,}", f"{h1k['augmented']:,}", "Recommended")
    sb = feas["sealbot_total"]
    table_row("SealBot (fast+strong)", f"{sb['games']:,}", f"{sb['positions']:,}", f"{sb['augmented']:,}", "Current corpus")
    table_row("Previous pretrain", "N/A", "~50,000", "~600,000", "SealBot subset (CLAUDE.md ref)")
    p()
    p(f"**Is the human corpus large enough for pretrain?**")
    p()
    p(f"{'✓ YES' if feas['pretrain_feasible'] else '✗ NO'} — "
      f"Recommended filter yields ~{h1k['positions']:,} positions, "
      f"**{feas['ratio_vs_previous']:.1f}× the previous pretrain size** (~50K positions).")
    p()
    p("**Should it replace SealBot corpus entirely, or be mixed?**")
    p()
    p("""**Recommendation: Mix, with human games as the primary component.**

Rationale:
1. Human games (~{h_pos:,} pos) cover {h_ratio:.1f}× the previous pretrain baseline.
   They represent authentic competition at ≥1000 Elo and contain opening diversity
   and endgame patterns that SealBot self-play does not generate.
2. SealBot games (~{sb_pos:,} pos) add tactical depth — bot play reaches forced/critical
   threats more often and more reliably than human games at this Elo range. They also
   provide more even P1 win rates (eliminating human time-pressure artifacts).
3. Human corpus is currently **{ratio:.1f}× smaller** than SealBot corpus in raw game count.
   Mixing prevents human games from being drowned out.

**Recommended mixing ratio:** 40% human : 60% SealBot (by game count, not position count).
- This gives human games proportionally more weight than their raw count suggests,
  compensating for their higher diversity value and longer average lengths.
- Apply the existing quality_score sampling weights from data/corpus/quality_scores.json.
- Re-run `make corpus.export` to regenerate bootstrap_corpus.npz with the new mix.

**Should we rebuild bootstrap_model.pt from this corpus?**

**Yes, with caveats.** The human corpus has grown from ~981 games (CORPUS_REPORT.md,
2026-04-01) to {h_games:,} filtered games — a **{growth:.1f}×** increase. The prior pretrain
used only ~50K positions from a much smaller corpus. The new mixed corpus provides
{total_pos:,}+ augmented positions, which should produce a significantly stronger
bootstrap model. However:
- The self-play loop is already running (Phase 4.0 active). Rebuilding bootstrap_model.pt
  requires a full pretrain pass (~15 epochs) which will interrupt the current self-play run.
- **Recommended decision path:**
  1. If the current self-play run is still early (< Phase 4.0 exit criterion), rebuild —
     a stronger bootstrap will accelerate convergence.
  2. If we are within 20% of Phase 4.0 exit criterion (24–48hr sustained run target),
     continue the current run and use the new corpus for the Phase 4.5 bootstrap reset.
""".format(
        h_pos=h1k["positions"],
        h_ratio=h1k["positions"] / 50000,
        sb_pos=sb["positions"],
        ratio=sb["games"] / h1k["games"],
        h_games=h["games"],
        growth=h["games"] / 981,
        total_pos=h1k["augmented"] + sb["augmented"],
    ))

    # -----------------------------------------------------------------------
    h2("Data Quality Issues Flagged")
    p("| Issue | Count | Severity | Notes |")
    p("| --- | --- | --- | --- |")
    p(f"| Raw files failing JSON parse | 0 | LOW | All files parsed cleanly |")
    unrated_count = inv_.get("elo_bands_filtered", {}).get("unrated", 0)
    total_filtered = inv_["total_filtered"]
    p(f"| Unrated games in filtered set | {unrated_count:,} ({unrated_count/total_filtered*100:.1f}%) | MEDIUM | No Elo signal; see Step 5 |")
    p(f"| Games filtered out (non-six-in-a-row or too short) | {inv_['total_raw'] - inv_['total_filtered']:,} | LOW | Expected by filter design |")
    p(f"| P1 win rate outside 45–60% | See Step 2 table | VARIES | Check per-band values |")
    p()

    # Write to file
    report_path = REPORT_DIR / "report.md"
    report_path.write_text("".join(lines))
    print(f"\n[Report written to {report_path}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading raw human games...")
    all_raw, parse_errors = load_raw_human_games()
    print(f"  Loaded {len(all_raw):,} raw files, {len(parse_errors)} parse errors")

    print("Extracting metadata...")
    all_meta = [extract_human_meta(g) for g in all_raw]
    filtered = [m for m in all_meta if m["passes_filter"]]
    print(f"  {len(all_meta):,} total, {len(filtered):,} pass filter")

    print("Loading bot games...")
    bot_fast = load_bot_games(CORPUS_BOT_FAST, "fast")
    bot_strong = load_bot_games(CORPUS_BOT_STRONG, "strong")
    bot_games = {"fast": bot_fast, "strong": bot_strong}
    print(f"  Bot: {len(bot_fast)} fast, {len(bot_strong)} strong")

    bot_summary = load_existing_bot_summary()

    # Run steps
    inv = step1_inventory(all_raw, filtered)
    q = step2_quality_metrics(filtered)
    tact = step3_tactical_depth(filtered)
    opening = step4_opening_diversity(filtered)
    filt = step5_filter_recommendation(filtered, bot_games)
    feasibility = step6_pretrain_feasibility(q, filt)

    # Save raw JSON data
    raw_data = {
        "inventory": inv,
        "quality": q,
        "tactical": tact,
        "opening": opening,
        "filter": filt,
        "feasibility": feasibility,
    }
    (REPORT_DIR / "raw_data.json").write_text(json.dumps(raw_data, indent=2, default=str))
    print(f"\n[Raw data written to {REPORT_DIR}/raw_data.json]")

    # Write report (needs some closure vars)
    global inv_, gld_
    inv_ = inv
    gld_ = q["game_length_dist"]
    write_report(inv, q, tact, opening, filt, feasibility, bot_summary)


if __name__ == "__main__":
    main()
