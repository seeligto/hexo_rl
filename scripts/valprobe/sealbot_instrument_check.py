"""D-C VALPROBE — SealBot instrument-check gate (pre-registered).

For each quiet-loss position that native TacticalSolver FAILED on (budget-exhausted),
probe SealBot at depth 6, 7, 8.  Also include ~4 extra quiet positions from other
loss games and 2 terminal-adjacent positive controls.

PRE-REGISTERED INTERPRETATION:
  GREEN  (proceed): SealBot proves LOSS (correct sign, head loses) on >=60% of quiet
         positions at some depth <=8, AND positive controls prove.
  RED    (dead-end): SealBot proves <=1 of the quiet positions even at d8 ->
         WP1-via-SealBot not viable; fall back to WP4-only.
  MIXED: report fraction + depth curve; judgment call.

Usage (from /home/timmy/Work/Hexo/hexo_rl, with .venv):
  .venv/bin/python scripts/valprobe/sealbot_instrument_check.py
      [--games-jsonl PATH] [--depths 6,7,8] [--n-quiet-extra N] [--window-half N]
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# SealBot pybind paths (mirrors sealbot_bot.py)
_SEALBOT_ROOT = str(ROOT / "vendor" / "bots" / "sealbot")
_SEALBOT_BEST = str(ROOT / "vendor" / "bots" / "sealbot" / "best")
for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from minimax_cpp import MinimaxBot as _MinimaxBot  # type: ignore[import]
from game import Player as SealPlayer              # type: ignore[import]

ENC = "v6_live2_ls"
WIN_THRESHOLD = 99_999_000  # |last_score| >= this => terminal mate proven


# ---------------------------------------------------------------------------
# Position extraction (reuse ceiling-probe logic)
# ---------------------------------------------------------------------------

def _find_terminal_ply(moves):
    import engine
    b = engine.Board.with_encoding_name(ENC)
    for mv in moves:
        b.apply_move(int(mv[0]), int(mv[1]))
        if b.winner() is not None:
            return b.ply
    return None


def extract_positions(games_jsonl: str, n_quiet_fixed: int = 6, n_quiet_extra: int = 4,
                      n_terminal: int = 2) -> list[dict]:
    """Extract the EXACT 6 positions the native solver failed on + n_quiet_extra more
    from other loss games + n_terminal positive controls.

    The 6 fixed positions (g0_t3, g1_t1, g1_t3, g2_t1, g2_t3, g3_t1) are identified
    by the same game/turn-index logic as the ceiling probe: first 3 loss games (sorted
    long->short), turns 1/3 (0-indexed).
    """
    import engine

    with open(games_jsonl) as f:
        games = [json.loads(l) for l in f if l.strip()]

    loss_games = [
        g for g in games
        if (g["head_as_p1"] and g["winner"] == "p2")
        or (not g["head_as_p1"] and g["winner"] == "p1")
    ]
    loss_games.sort(key=lambda g: g["plies"], reverse=True)

    quiet: list[dict] = []
    terminal: list[dict] = []

    for gi, g in enumerate(loss_games):
        if len(quiet) >= (n_quiet_fixed + n_quiet_extra) and len(terminal) >= n_terminal:
            break

        moves = g["moves"]
        head_is_p1 = g["head_as_p1"]
        head_player = 1 if head_is_p1 else -1

        terminal_ply = _find_terminal_ply(moves)
        if terminal_ply is None:
            continue

        head_compound_starts: list[int] = []
        prev_cp = None
        for n in range(terminal_ply - 1, 0, -1):
            b = engine.Board.with_encoding_name(ENC)
            for mv in moves[:n]:
                b.apply_move(int(mv[0]), int(mv[1]))
            cp = b.current_player
            if cp == head_player and prev_cp != head_player:
                head_compound_starts.append(n)
            prev_cp = cp

        for idx, n in enumerate(head_compound_starts[:8]):
            b = engine.Board.with_encoding_name(ENC)
            for mv in moves[:n]:
                b.apply_move(int(mv[0]), int(mv[1]))
            cp = b.current_player
            tm_opp = b.threat_moves(-cp)
            tm_self = b.threat_moves(cp)
            in_check = len(tm_opp) > 0
            has_own_win = len(tm_self) > 0
            turns_before_end = idx + 1

            rec = {
                "pos_id": f"g{gi}_t{idx}",
                "game_idx": gi,
                "game_plies": g["plies"],
                "ply": b.ply,
                "turns_before_end": turns_before_end,
                "in_check": in_check,
                "has_own_win": has_own_win,
                "is_quiet": not in_check and not has_own_win,
                "moves": [list(m) for m in moves[:n]],
                "kind": None,
                "head_player": head_player,
            }

            # terminal-adjacent (positive controls): turns_before_end==1
            if turns_before_end == 1 and len(terminal) < n_terminal:
                rec["kind"] = "terminal_adjacent"
                terminal.append(rec)
            # fixed quiet set: gi <= 3 (same games as ceiling probe), t1 or t3 (idx 1 or 3)
            elif 2 <= turns_before_end <= 4 and rec["is_quiet"]:
                if gi < 4 and len(quiet) < n_quiet_fixed:
                    rec["kind"] = "quiet_loss_fixed"
                    quiet.append(rec)
                elif gi >= 4 and len(quiet) < (n_quiet_fixed + n_quiet_extra):
                    rec["kind"] = "quiet_loss_extra"
                    quiet.append(rec)

    result = terminal + quiet
    print(f"[extract] {len(terminal)} terminal-adjacent, {len(quiet)} quiet-loss "
          f"({sum(1 for r in quiet if r['kind']=='quiet_loss_fixed')} fixed + "
          f"{sum(1 for r in quiet if r['kind']=='quiet_loss_extra')} extra)")
    return result


# ---------------------------------------------------------------------------
# SealBot probe — direct call, no BotProtocol wrapper needed
# ---------------------------------------------------------------------------

class _MockGame:
    def __init__(self, board: dict, current_player: int, moves_left: int, move_count: int) -> None:
        self.board = board
        self.current_player = SealPlayer.A if current_player == 1 else SealPlayer.B
        self.moves_left_in_turn = moves_left
        self.move_count = move_count


def _board_dict_from_moves(moves: list) -> tuple[dict, int, int, int]:
    """Replay moves; return (board_dict, current_player, moves_left_in_turn, move_count)."""
    import engine
    b = engine.Board.with_encoding_name(ENC)
    for mv in moves:
        b.apply_move(int(mv[0]), int(mv[1]))

    board_dict = {}
    for q, r, p in b.get_stones():
        board_dict[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B

    # moves_left_in_turn: compound turn has 2 moves; since we're at the turn START,
    # moves_remaining == 2 (quiet positions are always at compound-turn starts).
    moves_left = b.moves_remaining
    return board_dict, b.current_player, moves_left, len(board_dict), b


def is_off_window(move: tuple, stones: list, window_half: int) -> bool:
    """Chebyshev-distance off-window check (mirrors solver_backup_bot._is_off_window)."""
    qs = [q for q, _r, _p in stones]
    rs = [r for _q, r, _p in stones]
    cq = int((min(qs) + max(qs)) / 2)
    cr = int((min(rs) + max(rs)) / 2)
    return max(abs(move[0] - cq), abs(move[1] - cr)) > window_half


def probe_sealbot(moves: list, depth: int, window_half: int | None) -> dict:
    """Run SealBot at `depth` on the position defined by replaying `moves`.

    Returns dict: proven_loss, last_score, result_moves, wall_s, off_window_filtered,
                  current_player, colony_skip.
    """
    board_dict, current_player, moves_left, move_count, b = _board_dict_from_moves(moves)
    stones = list(b.get_stones())

    # Colony/OOB guard (mirrors SolverBackupBot defaults: max_coord=60, max_clusters=4)
    colony_max_coord = 60
    colony_max_clusters = 4
    if stones:
        max_coord = max(max(abs(int(q)), abs(int(r))) for q, r, _p in stones)
    else:
        max_coord = 0
    n_clusters = len(b.centers) if hasattr(b, 'centers') else 1
    colony_skip = (n_clusters > colony_max_clusters) or (max_coord > colony_max_coord)

    if colony_skip:
        return {
            "proven_loss": False, "last_score": None, "result_moves": [],
            "wall_s": 0.0, "off_window_filtered": False,
            "colony_skip": True, "current_player": current_player,
        }

    # Fresh SealBot instance per probe (cold TT = deterministic, per-game reset in D-SOLVER A1)
    mbot = _MinimaxBot(time_limit=120.0)  # large ceiling — depth bounds search
    mbot.max_depth = depth

    game = _MockGame(board_dict, current_player, moves_left, move_count)

    t0 = time.perf_counter()
    result_moves = mbot.get_move(game)
    wall_s = time.perf_counter() - t0
    last_score = float(mbot.last_score)

    # proven LOSS: |last_score| >= WIN_THRESHOLD AND negative (head loses)
    proven_loss = (last_score <= -WIN_THRESHOLD)

    # Off-window filter (window_half guard from D-SOLVER A1):
    # If the first proven move is off-window, flag it as untrusted (but still report score).
    off_window_filtered = False
    if window_half is not None and result_moves and proven_loss and stones:
        s1 = (int(result_moves[0][0]), int(result_moves[0][1]))
        s2 = (int(result_moves[1][0]), int(result_moves[1][1])) if len(result_moves) >= 2 else None
        if is_off_window(s1, stones, window_half) or (
            s2 is not None and is_off_window(s2, stones, window_half)
        ):
            off_window_filtered = True

    return {
        "proven_loss": proven_loss and not off_window_filtered,
        "proven_loss_raw": proven_loss,  # before off-window filter
        "last_score": last_score,
        "result_moves": [(int(m[0]), int(m[1])) for m in result_moves] if result_moves else [],
        "wall_s": round(wall_s, 2),
        "off_window_filtered": off_window_filtered,
        "colony_skip": False,
        "current_player": current_player,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(games_jsonl: str, depths: list[int], n_quiet_extra: int,
        window_half: int | None) -> None:
    positions = extract_positions(games_jsonl, n_quiet_fixed=6, n_quiet_extra=n_quiet_extra)
    if not positions:
        print("ERROR: no positions found")
        sys.exit(1)

    rows = []
    total_positions = len(positions)
    task_n = total_positions * len(depths)
    done = 0

    print(f"\n[probe] {total_positions} positions × {len(depths)} depths = {task_n} probes")
    print(f"[probe] window_half={window_half}, WIN_THRESHOLD={WIN_THRESHOLD:,}\n")

    for pos in positions:
        for depth in depths:
            done += 1
            print(f"[{done}/{task_n}] {pos['pos_id']} ({pos['kind']}, tbe={pos['turns_before_end']}) d={depth} ... ",
                  end="", flush=True)
            r = probe_sealbot(pos["moves"], depth, window_half)
            row = {
                "pos_id": pos["pos_id"],
                "kind": pos["kind"],
                "turns_before_end": pos["turns_before_end"],
                "ply": pos["ply"],
                "game_plies": pos["game_plies"],
                "depth": depth,
                "proven_loss": r["proven_loss"],
                "proven_loss_raw": r["proven_loss_raw"],
                "last_score": r["last_score"],
                "sign_correct": r["proven_loss"],  # proven_loss already checks negative score + window
                "off_window_filtered": r["off_window_filtered"],
                "colony_skip": r["colony_skip"],
                "current_player": r["current_player"],
                "wall_s": r["wall_s"],
                "result_moves": r["result_moves"],
            }
            rows.append(row)
            tag = "PROVEN_LOSS" if r["proven_loss"] else ("raw_loss_OW_filtered" if r["proven_loss_raw"] and r["off_window_filtered"] else ("colony_skip" if r["colony_skip"] else "no_proof"))
            print(f"{tag}  score={r['last_score']}  {r['wall_s']:.1f}s", flush=True)

    _print_table(rows)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("SEALBOT INSTRUMENT CHECK — RESULTS TABLE")
    print("=" * 100)
    hdr = f"{'pos_id':<16} {'kind':<22} {'tbe':>4} {'d':>3} {'proven_loss?':>12} {'last_score':>14} {'sign_ok?':>8} {'ow_filt':>7} {'wall_s':>7}"
    print(hdr)
    print("-" * 100)

    rows_s = sorted(rows, key=lambda r: (r["pos_id"], r["depth"]))
    prev = None
    for r in rows_s:
        sep = "\n" if prev and r["pos_id"] != prev else ""
        pl = "YES" if r["proven_loss"] else ("raw(OW)" if r["proven_loss_raw"] else "no")
        sc = f"{r['last_score']:.0f}" if r["last_score"] is not None else "N/A"
        sk = "YES" if r["sign_correct"] else "no"
        ow = "YES" if r["off_window_filtered"] else "no"
        print(f"{sep}{r['pos_id']:<16} {r['kind']:<22} {r['turns_before_end']:>4} {r['depth']:>3} "
              f"{pl:>12} {sc:>14} {sk:>8} {ow:>7} {r['wall_s']:>7.1f}s")
        prev = r["pos_id"]

    print("=" * 100)

    # Summary per kind
    quiet_rows = [r for r in rows if "quiet" in r["kind"]]
    quiet_fixed = [r for r in rows if r["kind"] == "quiet_loss_fixed"]
    quiet_extra = [r for r in rows if r["kind"] == "quiet_loss_extra"]
    terminal_rows = [r for r in rows if r["kind"] == "terminal_adjacent"]

    quiet_pos = sorted(set(r["pos_id"] for r in quiet_rows))
    quiet_fixed_pos = sorted(set(r["pos_id"] for r in quiet_fixed))

    print(f"\n--- terminal_adjacent (positive controls) ---")
    for depth in sorted(set(r["depth"] for r in terminal_rows)):
        g = [r for r in terminal_rows if r["depth"] == depth]
        n_proved = sum(1 for r in g if r["proven_loss"])
        wall_strs = ", ".join(f"{r['wall_s']:.1f}s" for r in g)
        print(f"  d={depth}: proved {n_proved}/{len(g)}  walls=[{wall_strs}]")

    print(f"\n--- quiet_loss_fixed (same positions native solver failed, n={len(quiet_fixed_pos)}) ---")
    for depth in sorted(set(r["depth"] for r in quiet_fixed)):
        g = [r for r in quiet_fixed if r["depth"] == depth]
        n_proved = sum(1 for r in g if r["proven_loss"])
        n_raw = sum(1 for r in g if r["proven_loss_raw"])
        walls_proved = [r["wall_s"] for r in g if r["proven_loss"]]
        print(f"  d={depth}: proved {n_proved}/{len(g)} (raw_before_OW_filter={n_raw})  "
              f"walls_proved={walls_proved}")

    print(f"\n--- quiet_loss_extra (additional positions, n={len(set(r['pos_id'] for r in quiet_extra))}) ---")
    for depth in sorted(set(r["depth"] for r in quiet_extra)):
        g = [r for r in quiet_extra if r["depth"] == depth]
        n_proved = sum(1 for r in g if r["proven_loss"])
        print(f"  d={depth}: proved {n_proved}/{len(g)}")

    # VERDICT
    print("\n" + "=" * 100)
    print("VERDICT (PRE-REGISTERED)")
    print("=" * 100)

    all_quiet_pos = sorted(set(r["pos_id"] for r in quiet_rows))
    n_quiet_total = len(all_quiet_pos)
    # "proved" = at any depth, proven_loss==True
    proved_quiet_pos = set(r["pos_id"] for r in quiet_rows if r["proven_loss"])
    n_proved_quiet = len(proved_quiet_pos)
    frac = n_proved_quiet / n_quiet_total if n_quiet_total else 0.0

    # Controls
    n_controls = len(set(r["pos_id"] for r in terminal_rows))
    proved_controls = len(set(r["pos_id"] for r in terminal_rows if r["proven_loss"]))

    print(f"Controls: {proved_controls}/{n_controls} terminal-adjacent proved")
    print(f"Quiet: {n_proved_quiet}/{n_quiet_total} positions proved LOSS at some depth <=8  ({frac:.0%})")

    if proved_controls < n_controls:
        print("\nWARNING: positive controls did NOT all prove — instrument may be broken at these positions")

    if frac >= 0.60 and proved_controls == n_controls:
        print("\nVERDICT: GREEN — proceed to full WP1 SealBot re-run")
        # Depth analysis
        for depth in sorted(set(r["depth"] for r in quiet_rows)):
            g = [r for r in quiet_rows if r["depth"] == depth]
            n_p = sum(1 for r in g if r["proven_loss"])
            wall_list = [r["wall_s"] for r in g if r["proven_loss"]]
            avg_w = sum(wall_list) / len(wall_list) if wall_list else 0
            max_w = max(wall_list) if wall_list else 0
            print(f"  d={depth}: {n_p}/{len(g)} proved, avg_wall={avg_w:.1f}s, max_wall={max_w:.1f}s")
        # Cost estimate
        # Per-game: ~4 quiet turn-starts probed (tbe=2..4), use max depth wall
        all_proved_rows = [r for r in quiet_rows if r["proven_loss"]]
        if all_proved_rows:
            best_depth = min(r["depth"] for r in all_proved_rows)
            best_wall_rows = [r for r in quiet_rows if r["proven_loss"] and r["depth"] == best_depth]
            p90_wall = sorted(r["wall_s"] for r in best_wall_rows)[int(len(best_wall_rows) * 0.9)] if best_wall_rows else 0
            print(f"\n  SealBot min-sufficient depth: d={best_depth}, p90 wall per proved pos: {p90_wall:.1f}s")
            # 57 games × ~4 quiet positions × p90_wall (serial), parallel on 24 cores
            serial_h = 57 * 4 * p90_wall / 3600
            parallel_h = serial_h / 24
            print(f"  Full 57-game re-run estimate: ~{serial_h:.1f} core-hours serial, "
                  f"~{parallel_h:.1f}h on 24 vast cores")
    elif n_proved_quiet <= 1:
        print("\nVERDICT: RED — SealBot also insufficient; <=1 quiet position proved at d8")
        print("  WP1-via-SealBot not viable. Recommend: WP4-only + bank findings.")
    else:
        print(f"\nVERDICT: MIXED — {frac:.0%} proved (threshold 60%). Judgment call required.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-jsonl", default=str(ROOT / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"))
    ap.add_argument("--depths", default="6,7,8")
    ap.add_argument("--n-quiet-extra", type=int, default=4)
    ap.add_argument("--window-half", type=int, default=9, help="0 to disable")
    args = ap.parse_args()
    depths = [int(d) for d in args.depths.split(",")]
    wh = args.window_half if args.window_half > 0 else None
    run(args.games_jsonl, depths, args.n_quiet_extra, wh)
