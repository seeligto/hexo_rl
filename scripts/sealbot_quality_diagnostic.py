#!/usr/bin/env python3
"""SealBot Corpus Quality Diagnostic — read-only analysis.

Determines whether poor SealBot play quality stems from:
(A) sparse random openings, (B) SealBot search defect, or (C) both.

Output: reports/corpus_diagnostics/sealbot_quality_YYYYMMDD.md
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

# Project paths
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.bootstrap.bots.sealbot_bot import SealBotBot

# ─── Hex formation detection (pure Python, no Rust FormationDetector) ───

# The 3 axial directions on a hex grid
HEX_DIRS = [(1, 0), (0, 1), (1, -1)]


def _count_line(stones_set: set, q: int, r: int, dq: int, dr: int, player_stones: set) -> tuple[int, bool, bool]:
    """Count consecutive same-player stones starting from (q,r) in direction (dq,dr).
    Returns (count, open_start, open_end) where open means the end cell is empty."""
    count = 0
    cq, cr = q, r
    while (cq, cr) in player_stones:
        count += 1
        cq += dq
        cr += dr
    open_end = (cq, cr) not in stones_set  # empty cell beyond the line
    return count, open_end


def count_formations(board_dict: dict[tuple[int, int], int]) -> dict[str, int]:
    """Count open-three and open-four formations for each player.

    Returns dict with keys: 'p1_open3', 'p1_open4', 'p2_open3', 'p2_open4'.
    """
    all_stones = set(board_dict.keys())
    p1_stones = {pos for pos, p in board_dict.items() if p == 1}
    p2_stones = {pos for pos, p in board_dict.items() if p == -1}

    result = {'p1_open3': 0, 'p1_open4': 0, 'p2_open3': 0, 'p2_open4': 0}
    counted = set()  # avoid double-counting same formation

    for player_label, player_stones in [('p1', p1_stones), ('p2', p2_stones)]:
        for (q, r) in player_stones:
            for dq, dr in HEX_DIRS:
                # Only start counting from the beginning of a line
                prev = (q - dq, r - dr)
                if prev in player_stones:
                    continue  # not the start of this line

                # Walk forward
                count = 0
                cq, cr = q, r
                while (cq, cr) in player_stones:
                    count += 1
                    cq += dq
                    cr += dr

                if count < 3:
                    continue

                # Check open ends
                start_open = (q - dq, r - dr) not in all_stones
                end_open = (cq, cr) not in all_stones

                # Formation key for dedup
                fkey = (q, r, dq, dr, player_label, count)
                if fkey in counted:
                    continue
                counted.add(fkey)

                if count == 3 and start_open and end_open:
                    result[f'{player_label}_open3'] += 1
                elif count == 4 and (start_open or end_open):
                    result[f'{player_label}_open4'] += 1
                elif count >= 4 and start_open and end_open:
                    result[f'{player_label}_open4'] += 1

    return result


def has_open_four(board_dict: dict[tuple[int, int], int], player: int) -> list[tuple]:
    """Find all open-four formations for given player. Returns list of (start_q, start_r, dq, dr)."""
    all_stones = set(board_dict.keys())
    player_stones = {pos for pos, p in board_dict.items() if p == player}
    threats = []

    for (q, r) in player_stones:
        for dq, dr in HEX_DIRS:
            prev = (q - dq, r - dr)
            if prev in player_stones:
                continue

            count = 0
            cq, cr = q, r
            while (cq, cr) in player_stones:
                count += 1
                cq += dq
                cr += dr

            if count >= 4:
                start_open = (q - dq, r - dr) not in all_stones
                end_open = (cq, cr) not in all_stones
                if start_open or end_open:
                    threats.append((q, r, dq, dr, count))

    return threats


def board_to_dict(board: Board) -> dict[tuple[int, int], int]:
    """Convert Rust Board to Python dict."""
    return {(q, r): p for q, r, p in board.get_stones()}


def replay_moves_to_board(moves: list[tuple[int, int]], up_to: Optional[int] = None) -> tuple[Board, GameState]:
    """Replay a move sequence onto a fresh board."""
    board = Board()
    state = GameState.from_board(board)
    limit = up_to if up_to is not None else len(moves)
    for i, (q, r) in enumerate(moves):
        if i >= limit:
            break
        if board.check_win():
            break
        state = state.apply_move(board, q, r)
    return board, state


# ─── Step 1: Characterize bot corpus threat density ───

def step1_corpus_threat_density(report_lines: list[str]) -> None:
    report_lines.append("## Step 1: Bot Corpus Threat Density at First SealBot Move\n")

    fast_dir = PROJECT / "data" / "corpus" / "bot_games" / "sealbot_fast"
    strong_dir = PROJECT / "data" / "corpus" / "bot_games" / "sealbot_strong"

    all_files = []
    for d, label in [(fast_dir, "fast"), (strong_dir, "strong")]:
        if d.exists():
            files = sorted(d.glob("*.json"))
            for f in files:
                all_files.append((f, label))

    rng = random.Random(42)
    rng.shuffle(all_files)
    sample = all_files[:50]

    rows = []
    for filepath, depth_label in sample:
        with open(filepath) as f:
            data = json.load(f)
        moves = [(m["x"], m["y"]) for m in data["moves"]]

        # The opening injection is n_random_opening moves (3 for d4, 3 for d6)
        # After those, SealBot plays. So at first SealBot move, board has n_random stones.
        n_random = 3  # both d4 and d6 use 3 random opening moves

        board, state = replay_moves_to_board(moves, up_to=n_random)
        bd = board_to_dict(board)
        formations = count_formations(bd)
        n_stones = len(bd)
        _, centers = board.get_cluster_views()
        k = len(centers)

        total_formations = sum(formations.values())
        rows.append({
            'file': filepath.name, 'depth': depth_label,
            'stones': n_stones, 'formations': total_formations,
            'open3': formations['p1_open3'] + formations['p2_open3'],
            'open4': formations['p1_open4'] + formations['p2_open4'],
            'K': k,
        })

    report_lines.append(f"Sampled {len(rows)} games (mix of d4/d6).\n")
    report_lines.append("### Summary Statistics at First SealBot Move\n")
    report_lines.append("| Metric | Mean | Min | Max |")
    report_lines.append("|--------|------|-----|-----|")

    for metric in ['stones', 'formations', 'open3', 'open4', 'K']:
        vals = [r[metric] for r in rows]
        mean = sum(vals) / len(vals) if vals else 0
        report_lines.append(f"| {metric} | {mean:.1f} | {min(vals)} | {max(vals)} |")

    # K distribution
    k_dist = defaultdict(int)
    for r in rows:
        k_dist[r['K']] += 1
    report_lines.append(f"\n**K distribution:** {dict(sorted(k_dist.items()))}\n")

    # Formation distribution
    form_dist = defaultdict(int)
    for r in rows:
        form_dist[r['formations']] += 1
    report_lines.append(f"**Formation count distribution:** {dict(sorted(form_dist.items()))}\n")

    zero_formation_pct = sum(1 for r in rows if r['formations'] == 0) / len(rows) * 100
    report_lines.append(f"**Games with ZERO formations at first SealBot move:** {zero_formation_pct:.0f}%\n")
    report_lines.append("")


# ─── Step 2: Test SealBot on complex human positions ───

def step2_sealbot_on_complex(report_lines: list[str]) -> None:
    report_lines.append("## Step 2: SealBot Threat Detection on Complex Human Positions\n")

    human_dir = PROJECT / "data" / "corpus" / "raw_human"
    if not human_dir.exists():
        report_lines.append("**ERROR:** No human games found.\n")
        return

    # Find 5 longest human games
    human_files = sorted(human_dir.glob("*.json"))
    game_lengths = []
    for f in human_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            n_moves = len(data.get("moves", []))
            game_lengths.append((n_moves, f, data))
        except Exception:
            continue

    game_lengths.sort(reverse=True)
    top5 = game_lengths[:5]

    report_lines.append(f"Selected 5 longest human games (move counts: {[g[0] for g in top5]}).\n")
    report_lines.append("For each game: replay to move 20, then let SealBot d6 play 10 moves (5 compound turns).\n")

    sealbot = SealBotBot(time_limit=1.0, max_depth=6)

    total_threats = 0
    total_blocked = 0
    total_moves_checked = 0

    for game_idx, (n_moves, filepath, data) in enumerate(top5):
        report_lines.append(f"### Game {game_idx+1}: {filepath.name} ({n_moves} moves)\n")

        # Parse moves from human game format
        moves_raw = data.get("moves", [])
        moves = [(m["x"], m["y"]) for m in moves_raw]

        # Replay to move 20
        board, state = replay_moves_to_board(moves, up_to=20)
        bd = board_to_dict(board)
        n_stones_at_20 = len(bd)
        formations_at_20 = count_formations(bd)
        report_lines.append(f"Position at move 20: {n_stones_at_20} stones, "
                          f"formations: {formations_at_20}\n")

        # Let SealBot play 10 more moves
        move_log = []
        for move_i in range(10):
            if board.check_win():
                move_log.append(f"  [move {move_i+1}] Game over (win detected)")
                break

            bd_before = board_to_dict(board)
            current_p = state.current_player
            opponent = -current_p

            # Check for opponent open-four threats
            opp_threats = has_open_four(bd_before, opponent)
            threat_present = len(opp_threats) > 0

            try:
                q, r = sealbot.get_move(state, board)
                state = state.apply_move(board, q, r)
            except Exception as e:
                move_log.append(f"  [move {move_i+1}] SealBot error: {e}")
                break

            # Check if SealBot blocked any threat
            blocked = False
            if threat_present:
                total_threats += 1
                # A move blocks if it's placed at an extension point of the open-four
                for tq, tr, dq, dr, cnt in opp_threats:
                    # Check if SealBot placed at either end of the line
                    end_q = tq + dq * cnt
                    end_r = tr + dr * cnt
                    start_q = tq - dq
                    start_r = tr - dr
                    if (q, r) == (end_q, end_r) or (q, r) == (start_q, start_r):
                        blocked = True
                        break
                if blocked:
                    total_blocked += 1

            total_moves_checked += 1
            status = ""
            if threat_present:
                status = f" | open-four threat: YES | blocked: {'YES' if blocked else 'NO'}"
            else:
                status = f" | open-four threat: NO"
            move_log.append(f"  [move {move_i+1}] SealBot({current_p}) plays ({q},{r}){status}")

        report_lines.append("```")
        report_lines.extend(move_log)
        report_lines.append("```\n")

    # Aggregate
    if total_threats > 0:
        block_rate = total_blocked / total_threats * 100
        report_lines.append(f"### Aggregate: {total_blocked}/{total_threats} open-four threats blocked "
                          f"({block_rate:.0f}%) across {total_moves_checked} moves checked\n")
    else:
        report_lines.append(f"### Aggregate: 0 open-four threats encountered in {total_moves_checked} moves\n")
    report_lines.append("")


# ─── Step 3: Head-to-head on seeded vs random positions ───

def step3_head_to_head(report_lines: list[str]) -> None:
    report_lines.append("## Step 3: SealBot d6 vs SealBot d6 — Complex vs Random Opening\n")

    human_dir = PROJECT / "data" / "corpus" / "raw_human"
    human_files = sorted(human_dir.glob("*.json"))

    # Find a game with at least 25 moves
    target_game = None
    for f in human_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            if len(data.get("moves", [])) >= 30:
                target_game = data
                target_file = f.name
                break
        except Exception:
            continue

    if target_game is None:
        report_lines.append("**ERROR:** No human game with 25+ moves found.\n")
        return

    def play_out_sealbot(board: Board, state: GameState, max_plies: int = 200, label: str = "") -> dict:
        """Play SealBot d6 vs SealBot d6 to completion."""
        sealbot = SealBotBot(time_limit=1.0, max_depth=6)
        move_list = []
        start_time = time.monotonic()

        for _ in range(max_plies):
            if board.check_win():
                break
            if board.legal_move_count() == 0:
                break
            try:
                q, r = sealbot.get_move(state, board)
                state = state.apply_move(board, q, r)
                move_list.append((q, r))
            except Exception as e:
                break

        elapsed = time.monotonic() - start_time
        bd = board_to_dict(board)
        _, centers = board.get_cluster_views()
        winner = board.winner()

        # Check for uncontested 6-in-a-row: was there a line of 5+ that wasn't blocked?
        formations = count_formations(bd)

        return {
            'moves': len(move_list),
            'winner': winner,
            'K': len(centers),
            'stones': len(bd),
            'formations': formations,
            'elapsed': elapsed,
        }

    # Game A: from human game at move 25
    report_lines.append(f"### Game A: Starting from human game {target_file} at move 25\n")
    moves_raw = target_game.get("moves", [])
    moves = [(m["x"], m["y"]) for m in moves_raw]
    board_a, state_a = replay_moves_to_board(moves, up_to=25)
    bd_a_start = board_to_dict(board_a)
    report_lines.append(f"Position at move 25: {len(bd_a_start)} stones, "
                       f"formations: {count_formations(bd_a_start)}\n")
    result_a = play_out_sealbot(board_a, state_a, label="complex")

    # Game B: from random 4-stone opening
    report_lines.append(f"### Game B: Starting from random 4-stone opening\n")
    rng = random.Random(12345)
    board_b = Board()
    state_b = GameState.from_board(board_b)
    # Place 4 random stones scattered
    placed = 0
    while placed < 4:
        q = rng.randint(-3, 3)
        r = rng.randint(-3, 3)
        if board_b.get(q, r) == 0:
            state_b = state_b.apply_move(board_b, q, r)
            placed += 1
    bd_b_start = board_to_dict(board_b)
    report_lines.append(f"Random opening: {len(bd_b_start)} stones, "
                       f"formations: {count_formations(bd_b_start)}\n")
    result_b = play_out_sealbot(board_b, state_b, label="random")

    # Side-by-side comparison
    report_lines.append("### Side-by-Side Comparison\n")
    report_lines.append("| Metric | Game A (complex) | Game B (random) |")
    report_lines.append("|--------|-----------------|-----------------|")
    report_lines.append(f"| Starting stones | {len(bd_a_start)} | {len(bd_b_start)} |")
    report_lines.append(f"| Moves played by SealBot | {result_a['moves']} | {result_b['moves']} |")
    report_lines.append(f"| Winner | {result_a['winner']} | {result_b['winner']} |")
    report_lines.append(f"| Final K (clusters) | {result_a['K']} | {result_b['K']} |")
    report_lines.append(f"| Final stones | {result_a['stones']} | {result_b['stones']} |")
    report_lines.append(f"| Final formations | {result_a['formations']} | {result_b['formations']} |")
    report_lines.append(f"| Time (s) | {result_a['elapsed']:.1f} | {result_b['elapsed']:.1f} |")
    report_lines.append("")


# ─── Step 4: Analyze the random opening generator ───

def step4_opening_generator(report_lines: list[str]) -> None:
    report_lines.append("## Step 4: Random Opening Generator Analysis\n")

    gen_path = PROJECT / "hexo_rl" / "bootstrap" / "generate_corpus.py"
    with open(gen_path) as f:
        source = f.read()

    # Extract the relevant function
    report_lines.append("### Relevant code from `hexo_rl/bootstrap/generate_corpus.py`\n")
    report_lines.append("#### `_play_one_game()` — opening injection logic:\n")
    report_lines.append("```python")

    # Extract just the random opening section
    lines = source.split('\n')
    in_func = False
    brace_depth = 0
    for i, line in enumerate(lines):
        if 'def _play_one_game' in line:
            in_func = True
        if in_func:
            report_lines.append(line)
            if 'while not board.check_win' in line:
                # Include a few more lines for context
                for j in range(i+1, min(i+5, len(lines))):
                    report_lines.append(lines[j])
                break

    report_lines.append("```\n")

    # Extract the n_random_opening default logic
    report_lines.append("#### CLI default for `n_random_opening`:\n")
    report_lines.append("```python")
    in_section = False
    for line in lines:
        if 'random_opening' in line.lower() and 'auto' in line.lower():
            in_section = True
        if in_section:
            report_lines.append(line)
            if 'n_random' in line and '=' in line and 'args' not in line:
                pass
            if line.strip() == '' and in_section:
                break
    # Fallback: just print the section we know about
    report_lines.append("```\n")

    # Analysis
    report_lines.append("### Verdict\n")
    report_lines.append("""The random opening generator:
1. Places exactly **3 random moves** before SealBot takes over (for both d4 and d6)
2. Moves are selected uniformly from `board.legal_moves()` — which on an empty board means the full margin around existing stones
3. **No minimum formation density is enforced** — there is no check that the random opening creates any threats
4. **No clustering guarantee** — the 3 random stones can land anywhere, potentially creating 2-3 separate clusters with no tactical interaction
5. The seed is `rng_seed + game_idx`, so with seed=42 the openings are deterministic per game index

**Is there a guarantee of threat density before SealBot starts playing?** **NO.** With only 3 scattered random stones, the probability of creating even a single open-three is very low. SealBot inherits positions with essentially zero tactical content.\n""")


# ─── Main ───

def main() -> None:
    report_lines: list[str] = []
    report_lines.append(f"# SealBot Corpus Quality Diagnostic — {date.today().isoformat()}\n")
    report_lines.append("Objective: determine whether poor SealBot play quality is caused by "
                       "(A) sparse random openings, (B) SealBot search defect, or (C) both.\n")

    print("Running Step 1: Corpus threat density...")
    step1_corpus_threat_density(report_lines)

    print("Running Step 2: SealBot on complex human positions (this may take a few minutes)...")
    step2_sealbot_on_complex(report_lines)

    print("Running Step 3: Head-to-head comparison...")
    step3_head_to_head(report_lines)

    print("Running Step 4: Opening generator analysis...")
    step4_opening_generator(report_lines)

    # ─── Conclusion ───
    report_lines.append("## Conclusion\n")
    report_lines.append("""Based on the evidence above:

**Step 1** shows that at the point SealBot begins playing, the board has only 3 stones
with (likely) zero formations. The opening is genuinely sparse — there is nothing for
SealBot to respond to tactically.

**Step 2** tests whether SealBot can defend when real threats exist on complex positions.
The open-four blocking rate on human game positions directly answers whether SealBot has
a search defect independent of opening quality.

**Step 3** compares SealBot's play quality when starting from a complex position vs a
random sparse position, showing whether the same engine produces different quality games
based on starting conditions.

**Step 4** confirms that the opening generator has no threat density guarantee — 3 random
stones on an infinite board almost never create tactical structure.

See the data tables above for the specific numbers that determine the final verdict.\n""")

    # Write report
    out_path = PROJECT / "reports" / "corpus_diagnostics" / f"sealbot_quality_{date.today().strftime('%Y%m%d')}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
