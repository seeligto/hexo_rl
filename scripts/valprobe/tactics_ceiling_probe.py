"""D-C VALPROBE WP1 ceiling discriminator — single run to decide whether the native
TacticalSolver (neighbor_dist=2, widened) can ever provide ground-truth T_provable.

PRE-REGISTERED INTERPRETATION (written before any run):
  CEILING-CONFIRMED: across tested 2+-turns-before-terminal quiet-loss positions,
      0 prove LOSS at any budget up to 5M nodes, AND searches are budget-EXHAUSTED
      (nodes > budget) → native route to T_provable is architecturally dead for WP1.
  LEVER-ALIVE: >=1 position proves LOSS (result==-1) within <=2M nodes → native
      route is budget-bound, viable with adequate offline budget → report
      proven-fraction + nodes/wall to judge tractability for 57-game WP1.

Usage (on vast, from /workspace/hexo_rl, after maturin rebuild):
  .venv/bin/python scripts/valprobe/tactics_ceiling_probe.py
      [--games-jsonl PATH] [--n-quiet N] [--n-terminal N] [--workers N]

Output: table to stdout + reports/valprobe/tactics_ceiling_probe_results.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure project root on path (works whether invoked from root or scripts/)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

ENC = "v6_live2_ls"
BUDGETS = [1_000_000, 2_000_000, 5_000_000]
DEPTH = 14


# ---------------------------------------------------------------------------
# Position extraction
# ---------------------------------------------------------------------------

def _replay(moves, enc=ENC):
    import engine
    b = engine.Board.with_encoding_name(enc)
    for mv in moves:
        b.apply_move(int(mv[0]), int(mv[1]))
    return b


def _find_terminal_ply(moves):
    import engine
    b = engine.Board.with_encoding_name(ENC)
    for mv in moves:
        b.apply_move(int(mv[0]), int(mv[1]))
        if b.winner() is not None:
            return b.ply
    return None


def extract_positions(games_jsonl: str, n_quiet: int = 6, n_terminal: int = 2) -> list[dict]:
    """Extract quiet-loss probe positions + positive controls from games.jsonl.

    Quiet-loss: head to move, NOT in check (opp has no immediate threat),
    head has no own immediate win, 2–4 compound turns before game ends.
    Terminal-adjacent: head to move within 1 compound turn of terminal (positive controls).
    """
    import engine

    with open(games_jsonl) as f:
        games = [json.loads(l) for l in f if l.strip()]

    loss_games = [
        g for g in games
        if (g["head_as_p1"] and g["winner"] == "p2")
        or (not g["head_as_p1"] and g["winner"] == "p1")
    ]
    # Sort long→short: longer games have more quiet developmental positions
    loss_games.sort(key=lambda g: g["plies"], reverse=True)

    quiet: list[dict] = []
    terminal: list[dict] = []

    for gi, g in enumerate(loss_games):
        if len(quiet) >= n_quiet and len(terminal) >= n_terminal:
            break

        moves = g["moves"]
        head_is_p1 = g["head_as_p1"]
        head_player = 1 if head_is_p1 else -1

        terminal_ply = _find_terminal_ply(moves)
        if terminal_ply is None:
            continue

        # Walk backwards to find each compound-turn START for head
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
            turns_before_end = idx + 1  # 1 = last head compound turn before terminal

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
                "kind": None,  # set below
            }

            if turns_before_end == 1 and len(terminal) < n_terminal:
                rec["kind"] = "terminal_adjacent"
                terminal.append(rec)
            elif 2 <= turns_before_end <= 4 and rec["is_quiet"] and len(quiet) < n_quiet:
                rec["kind"] = "quiet_loss"
                quiet.append(rec)

    result = terminal + quiet
    print(f"[extract] {len(terminal)} terminal-adjacent, {len(quiet)} quiet-loss positions found")
    return result


# ---------------------------------------------------------------------------
# Solver worker — runs in a subprocess to avoid GIL / memory sharing
# ---------------------------------------------------------------------------

def _prove_at_budget(args: tuple) -> dict:
    """Worker: prove a position at a single budget. Runs in subprocess."""
    pos_id, moves, budget, depth = args
    import engine, time

    b = engine.Board.with_encoding_name(ENC)
    for mv in moves:
        b.apply_move(int(mv[0]), int(mv[1]))

    solver = engine.TacticalSolver(window_half=None, cand_cap=20, neighbor_dist=2)

    t0 = time.perf_counter()
    result, line, nodes = solver.prove(b, depth, budget)
    wall_s = time.perf_counter() - t0

    # Infer exhaustion: budget.tick() increments nodes BEFORE checking, so
    # nodes > budget means exhausted (the cap+1-th tick set exhausted=true).
    exhausted = nodes > budget

    return {
        "pos_id": pos_id,
        "budget": budget,
        "result": result,
        "nodes": nodes,
        "exhausted": exhausted,
        "wall_s": round(wall_s, 3),
        "line": line[:4] if line else [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(games_jsonl: str, n_quiet: int, n_terminal: int, workers: int, out_dir: str) -> None:
    positions = extract_positions(games_jsonl, n_quiet=n_quiet, n_terminal=n_terminal)
    if not positions:
        print("ERROR: no positions found — check games.jsonl path")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tactics_ceiling_probe_results.jsonl")

    # Build work items: (pos_id, moves, budget, depth)
    # Run each budget sequentially per position for early-stop analysis,
    # but parallelize ACROSS positions × budgets for throughput.
    work = []
    for pos in positions:
        for budget in BUDGETS:
            work.append((pos["pos_id"], pos["moves"], budget, DEPTH))

    print(f"[run] {len(positions)} positions × {len(BUDGETS)} budgets = {len(work)} tasks, workers={workers}")

    # Collect per-pos metadata (not repeated in work tuples)
    pos_meta = {p["pos_id"]: {k: v for k, v in p.items() if k != "moves"} for p in positions}

    rows: list[dict] = []
    t_total = time.perf_counter()
    with open(out_path, "w") as out_f:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_prove_at_budget, item): item for item in work}
            for n_done, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                # Merge position metadata
                meta = pos_meta[row["pos_id"]]
                row.update({k: meta[k] for k in ("kind", "turns_before_end", "ply", "game_plies", "in_check", "has_own_win")})
                rows.append(row)
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                verdict = {1: "WIN", -1: "LOSS", 0: "UNKNOWN"}.get(row["result"], "?")
                ex_flag = "EXHAUSTED" if row["exhausted"] else "COMPLETED"
                print(
                    f"  [{n_done}/{len(work)}] {row['pos_id']} budget={row['budget']//1_000_000}M "
                    f"-> {verdict} nodes={row['nodes']:,} {ex_flag} wall={row['wall_s']:.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - t_total
    _print_table(rows, elapsed)


def _print_table(rows: list[dict], elapsed: float) -> None:
    print("\n" + "=" * 80)
    print("TACTICS CEILING PROBE — RESULTS TABLE")
    print("=" * 80)

    # Header
    print(f"{'pos_id':<18} {'kind':<18} {'t_b_e':>5} {'budget':>8} {'result':>7} "
          f"{'nodes':>10} {'exhausted':>9} {'wall_s':>7}")
    print("-" * 80)

    # Sort by pos_id then budget
    rows_sorted = sorted(rows, key=lambda r: (r["pos_id"], r["budget"]))
    prev_pos = None
    for r in rows_sorted:
        sep = "" if r["pos_id"] == prev_pos else "\n" if prev_pos else ""
        verdict = {1: "WIN", -1: "LOSS", 0: "UNKNOWN"}.get(r["result"], "?")
        ex_flag = "YES" if r["exhausted"] else "no"
        print(f"{sep}{r['pos_id']:<18} {r['kind']:<18} {r['turns_before_end']:>5} "
              f"{r['budget']//1_000_000:>5}M  {verdict:>7} {r['nodes']:>10,} "
              f"{ex_flag:>9} {r['wall_s']:>7.1f}s")
        prev_pos = r["pos_id"]

    print("=" * 80)

    # Summary statistics
    quiet = [r for r in rows if r["kind"] == "quiet_loss"]
    terminal = [r for r in rows if r["kind"] == "terminal_adjacent"]

    print(f"\nTotal wall: {elapsed:.1f}s")
    print(f"Positions: {len(set(r['pos_id'] for r in rows))} ({len(set(r['pos_id'] for r in quiet))} quiet, "
          f"{len(set(r['pos_id'] for r in terminal))} terminal-adjacent)")

    for label, group in [("terminal_adjacent", terminal), ("quiet_loss", quiet)]:
        print(f"\n--- {label} ---")
        by_budget = {}
        for r in group:
            by_budget.setdefault(r["budget"], []).append(r)
        for budget, grp in sorted(by_budget.items()):
            n_proved = sum(1 for r in grp if r["result"] == -1)
            n_exhausted = sum(1 for r in grp if r["exhausted"])
            n_completed = len(grp) - n_exhausted
            nodes_list = sorted(r["nodes"] for r in grp)
            med_nodes = nodes_list[len(nodes_list) // 2]
            print(f"  budget={budget//1_000_000}M: proved={n_proved}/{len(grp)} "
                  f"exhausted={n_exhausted}/{len(grp)} completed={n_completed}/{len(grp)} "
                  f"med_nodes={med_nodes:,}")

    # Quiet positions: any proven?
    quiet_proved = [r for r in quiet if r["result"] == -1]
    quiet_exhausted_5m = [r for r in quiet if r["budget"] == 5_000_000 and r["exhausted"]]
    quiet_positions = set(r["pos_id"] for r in quiet)
    quiet_proved_positions = set(r["pos_id"] for r in quiet_proved)

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if quiet_proved:
        # LEVER-ALIVE
        best = min(quiet_proved, key=lambda r: r["nodes"])
        print(f"LEVER-ALIVE")
        print(f"  {len(quiet_proved_positions)}/{len(quiet_positions)} quiet positions proved LOSS")
        print(f"  Min nodes to prove: {best['nodes']:,} (pos={best['pos_id']}, budget={best['budget']//1_000_000}M)")
        # Estimate tractability for 57 games
        wall_per_pos = max(r["wall_s"] for r in quiet_proved)
        n_positions_per_game = 4  # ~4 quiet positions per game
        total_wall_57 = wall_per_pos * n_positions_per_game * 57
        print(f"  Tractability estimate (57 games × ~{n_positions_per_game} positions): "
              f"~{total_wall_57/3600:.1f} core-hours (serial); on 24 cores: ~{total_wall_57/24/3600:.1f}h")
    else:
        all_5m_exhausted = all(r["exhausted"] for r in quiet_exhausted_5m)
        if all_5m_exhausted and len(quiet_exhausted_5m) > 0:
            print(f"CEILING-CONFIRMED")
            print(f"  0/{len(quiet_positions)} quiet positions proved LOSS at any budget (1M/2M/5M nodes)")
            print(f"  All 5M-budget searches EXHAUSTED (nodes > budget) — not completed")
            print(f"  Native solver architecturally cannot provide T_provable for WP1 quiet-loss positions")
        else:
            print(f"CEILING-CONFIRMED (partial — some 5M searches completed without proof)")
            print(f"  0/{len(quiet_positions)} quiet positions proved LOSS")
            print(f"  {len(quiet_exhausted_5m)}/{len(set(r['pos_id'] for r in quiet if r['budget']==5_000_000))} exhausted at 5M")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--games-jsonl",
        default=str(ROOT / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"),
        help="Path to games.jsonl (loss games from 248k retro-slope eval)",
    )
    ap.add_argument("--n-quiet", type=int, default=6, help="Number of quiet-loss probe positions")
    ap.add_argument("--n-terminal", type=int, default=2, help="Number of terminal-adjacent positive controls")
    ap.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 4)),
                    help="Parallel workers (default: min(24, cpu_count))")
    ap.add_argument("--out-dir", default=str(ROOT / "reports/valprobe"), help="Output directory")
    args = ap.parse_args()

    run(
        games_jsonl=args.games_jsonl,
        n_quiet=args.n_quiet,
        n_terminal=args.n_terminal,
        workers=args.workers,
        out_dir=args.out_dir,
    )
