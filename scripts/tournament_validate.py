#!/usr/bin/env python
"""§176 Wave B — round-robin tourney CLI.

Plays a comma-separated list of bots in a round-robin, producing:
  - per_game.jsonl   — one JSON object per game (terminal metrics)
  - summary.md       — markdown summary with BT ratings + H2H matrix + latency
  - ratings.csv      — BT MLE Elo-scaled, anchored at first bot
  - h2h_matrix.csv   — head-to-head win counts
  - colony_table.csv — per-bot colony-fraction means

Critical (§17 GIL regression — Wave A4 constraint #1): single game at a time,
NEVER concurrent bot pools. Master prompt allowance for single-game-at-a-time
in-process is honoured; under no circumstance run two games in parallel.

Critical (§176 Wave A1 finding): MinimaxBot carries a Zobrist TT + history
across get_move() calls and across games. Each game gets a FRESH instance
constructed via BOT_REGISTRY[name](); never reuse instances across games.

Halt-on-wall: --max-wall-seconds dumps partial JSONL on overrun and exits 0.

Usage:
  .venv/bin/python scripts/tournament_validate.py \\
    --bots randombot,sealbot,our_v6_mcts128 \\
    --n_games 5 \\
    --max_plies 200 \\
    --output reports/s176_b_smoke/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Path injection for repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine import Board                                        # noqa: E402
from hexo_rl.bootstrap.bot_protocol import BotProtocol          # noqa: E402
from hexo_rl.env.game_state import GameState                    # noqa: E402

# Imports of bots are lazy inside BOT_REGISTRY closures so missing
# weights/binaries do not crash the harness at import time.


# ────────────────────────────────────────────────────────────────────────────
# Bot registry
# ────────────────────────────────────────────────────────────────────────────

def _build_our_model_config(n_sims: int) -> Dict[str, Any]:
    """Build a minimal config dict for OurModelBot. Mirrors selfplay.yaml
    keys SelfPlayWorker reads (n_simulations, c_puct, …) and model.yaml.
    Inference is single-threaded (single-game tourney, no pool).
    """
    return {
        "mcts": {
            "n_simulations": int(n_sims),
            "c_puct": 1.5,
            "fpu_reduction": 0.25,
            "quiescence_enabled": True,
            "dirichlet_alpha": 0.05,
            "epsilon": 0.0,
            "dirichlet_enabled": False,
            "temperature_threshold_ply": 30,
        },
        "model": {
            "board_size": 19,
            "in_channels": 8,
            "res_blocks": 12,
            "filters": 128,
            "se_reduction_ratio": 4,
            "encoding": "v6",
        },
        "encoding": "v6",
    }


BOT_REGISTRY: Dict[str, Callable[[], BotProtocol]] = {}


def _register_default_bots(model_ckpt: str, our_n_sims: int) -> None:
    """Populate BOT_REGISTRY with closures (lazy bot instantiation).

    Note: each closure constructs a FRESH instance per call. Tourney loop
    invokes the closure once per game to avoid TT carryover (§176 Wave A1).
    """

    def make_randombot() -> BotProtocol:
        from hexo_rl.bots.random_bot import RandomBot
        return RandomBot()

    def make_sealbot() -> BotProtocol:
        from hexo_rl.bots.sealbot_bot import SealBotBot
        return SealBotBot(time_limit=0.5)

    def make_our_v6_mcts() -> BotProtocol:
        from hexo_rl.bots.our_model_bot import OurModelBot
        return OurModelBot(
            checkpoint_path=model_ckpt,
            config=_build_our_model_config(our_n_sims),
            temperature=0.0,
        )

    def make_our_v6_argmax() -> BotProtocol:
        from hexo_rl.bots.our_model_bot import OurModelBot
        # argmax = MCTS n_sims=1 (one prior expansion → arg-max policy head).
        # Acceptable proxy; native argmax-only would require a separate eval path.
        return OurModelBot(
            checkpoint_path=model_ckpt,
            config=_build_our_model_config(1),
            temperature=0.0,
        )

    BOT_REGISTRY.clear()
    BOT_REGISTRY.update({
        "randombot":             make_randombot,
        "sealbot":               make_sealbot,
        "our_v6_mcts128":        make_our_v6_mcts,
        "our_v6_argmax":         make_our_v6_argmax,
    })


# ────────────────────────────────────────────────────────────────────────────
# Game loop
# ────────────────────────────────────────────────────────────────────────────

_HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def _hex_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def _connected_components(stones: set[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    visited: set[Tuple[int, int]] = set()
    comps: List[List[Tuple[int, int]]] = []
    for start in stones:
        if start in visited:
            continue
        comp: List[Tuple[int, int]] = []
        queue = deque([start])
        visited.add(start)
        while queue:
            q, r = queue.popleft()
            comp.append((q, r))
            for dq, dr in _HEX_DIRS:
                nb = (q + dq, r + dr)
                if nb in stones and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comps.append(comp)
    return comps


def _pairwise_stats(stones: List[Tuple[int, int]]) -> Tuple[float, int]:
    """Return (mean_pairwise_hex_dist, max_pairwise_hex_dist). 0/0 on len<2."""
    if len(stones) < 2:
        return 0.0, 0
    total = 0
    n = 0
    mx = 0
    for i in range(len(stones)):
        for j in range(i + 1, len(stones)):
            d = _hex_distance(stones[i], stones[j])
            total += d
            mx = max(mx, d)
            n += 1
    return (total / n if n else 0.0), mx


@dataclass
class GameRecord:
    pair: Tuple[str, str]
    game_idx: int
    p1: str
    p2: str
    winner: str  # "p1" | "p2" | "draw"
    winner_bot: str
    plies: int
    n_stones_p1: int
    n_stones_p2: int
    colony_fraction_winner: float
    n_components_winner: int
    mean_pairwise_hex_dist_winner: float
    max_pairwise_hex_dist_winner: int
    terminal_threat_count: int
    winner_completed_via_threat: bool
    p1_total_wall_s: float
    p2_total_wall_s: float
    p1_max_move_s: float
    p2_max_move_s: float

    def to_jsonl(self) -> str:
        return json.dumps({
            "pair": list(self.pair),
            "game_idx": self.game_idx,
            "p1": self.p1,
            "p2": self.p2,
            "winner": self.winner,
            "winner_bot": self.winner_bot,
            "plies": self.plies,
            "n_stones_p1": self.n_stones_p1,
            "n_stones_p2": self.n_stones_p2,
            "colony_fraction_winner": self.colony_fraction_winner,
            "n_components_winner": self.n_components_winner,
            "mean_pairwise_hex_dist_winner": self.mean_pairwise_hex_dist_winner,
            "max_pairwise_hex_dist_winner": self.max_pairwise_hex_dist_winner,
            "terminal_threat_count": self.terminal_threat_count,
            "winner_completed_via_threat": self.winner_completed_via_threat,
            "p1_total_wall_s": self.p1_total_wall_s,
            "p2_total_wall_s": self.p2_total_wall_s,
            "p1_max_move_s": self.p1_max_move_s,
            "p2_max_move_s": self.p2_max_move_s,
        })


def _terminal_metrics(board: Board) -> Tuple[float, int, float, int, int]:
    """Return (colony_fraction_winner, n_components_winner, mean_pw, max_pw, threats)
    measured on the WINNER's stones at game end. For a draw, returns all zeros."""
    winner_int = board.winner()
    if winner_int is None:
        return 0.0, 0, 0.0, 0, len(board.get_threats())

    stones_all = board.get_stones()
    winner_stones = [(q, r) for q, r, p in stones_all if p == winner_int]
    if not winner_stones:
        return 0.0, 0, 0.0, 0, len(board.get_threats())

    comps = _connected_components(set(winner_stones))
    n_comps = len(comps)
    max_comp = max(len(c) for c in comps) if comps else 0
    # colony_fraction here = fraction of winner's stones NOT in the largest comp.
    colony_frac = (
        1.0 - (max_comp / len(winner_stones)) if winner_stones else 0.0
    )
    mean_pw, max_pw = _pairwise_stats(winner_stones)
    threats = len(board.get_threats())
    return colony_frac, n_comps, mean_pw, max_pw, threats


def _play_one_game(
    p1_factory: Callable[[], BotProtocol],
    p2_factory: Callable[[], BotProtocol],
    p1_name: str,
    p2_name: str,
    pair_label: Tuple[str, str],
    game_idx: int,
    max_plies: int,
) -> GameRecord:
    """Play one full game with FRESH bot instances. Returns terminal record."""
    bot_p1 = p1_factory()
    bot_p2 = p2_factory()

    b = Board()
    state = GameState.from_board(b)

    p1_total = 0.0
    p2_total = 0.0
    p1_max = 0.0
    p2_max = 0.0

    last_mover = 0  # 1 or -1
    ply = 0
    while ply < max_plies and not b.check_win() and b.legal_move_count() > 0:
        if b.current_player == 1:
            t0 = time.perf_counter()
            q, r = bot_p1.get_move(state, b)
            dt = time.perf_counter() - t0
            p1_total += dt
            p1_max = max(p1_max, dt)
            last_mover = 1
        else:
            t0 = time.perf_counter()
            q, r = bot_p2.get_move(state, b)
            dt = time.perf_counter() - t0
            p2_total += dt
            p2_max = max(p2_max, dt)
            last_mover = -1

        state = state.apply_move(b, q, r)
        ply += 1

    # Terminal outcome
    if b.check_win():
        winner_int = b.winner()
        if winner_int == 1:
            winner_label = "p1"
            winner_bot = p1_name
        elif winner_int == -1:
            winner_label = "p2"
            winner_bot = p2_name
        else:
            winner_label = "draw"
            winner_bot = ""
    else:
        winner_label = "draw"
        winner_bot = ""

    stones_all = b.get_stones()
    n_p1 = sum(1 for _, _, p in stones_all if p == 1)
    n_p2 = sum(1 for _, _, p in stones_all if p == -1)

    colony_frac, n_comps, mean_pw, max_pw, threats = _terminal_metrics(b)

    # winner_completed_via_threat: did winner's final move sit at a threat
    # level >=1 cell? Cheap proxy — get_threats() runs on the terminal board,
    # so we check whether the move history's last winner stone is at a threat
    # location pre-final. Best-effort; not exact since threats() runs post-move.
    # For smoke we report False unless terminal_threat_count > 0 AND last_mover
    # matches winner.
    if winner_label == "p1" and last_mover == 1 and threats > 0:
        winner_via_threat = True
    elif winner_label == "p2" and last_mover == -1 and threats > 0:
        winner_via_threat = True
    else:
        winner_via_threat = False

    return GameRecord(
        pair=pair_label,
        game_idx=game_idx,
        p1=p1_name,
        p2=p2_name,
        winner=winner_label,
        winner_bot=winner_bot,
        plies=ply,
        n_stones_p1=n_p1,
        n_stones_p2=n_p2,
        colony_fraction_winner=round(colony_frac, 4),
        n_components_winner=n_comps,
        mean_pairwise_hex_dist_winner=round(mean_pw, 3),
        max_pairwise_hex_dist_winner=max_pw,
        terminal_threat_count=threats,
        winner_completed_via_threat=winner_via_threat,
        p1_total_wall_s=round(p1_total, 4),
        p2_total_wall_s=round(p2_total, 4),
        p1_max_move_s=round(p1_max, 4),
        p2_max_move_s=round(p2_max, 4),
    )


# ────────────────────────────────────────────────────────────────────────────
# Bradley-Terry MLE — reuses hexo_rl.eval.bradley_terry
# ────────────────────────────────────────────────────────────────────────────

def _compute_bt_ratings(
    games: List[GameRecord], bot_names: List[str]
) -> Dict[str, Tuple[float, float, float]]:
    """Bradley-Terry MLE Elo-scaled ratings. Draws split 0.5/0.5.
    Anchor: bot_names[0].
    """
    from hexo_rl.eval.bradley_terry import compute_ratings

    # Map bot name → integer id.
    name_to_id = {n: i for i, n in enumerate(bot_names)}
    id_to_name = {i: n for n, i in name_to_id.items()}

    # Aggregate pairwise wins (draws split 0.5).
    wins: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    for g in games:
        a, b = name_to_id[g.p1], name_to_id[g.p2]
        # Canonical (smaller_id, larger_id) pair key.
        key = (min(a, b), max(a, b))
        flip = a > b
        if g.winner == "p1":
            wins[key][1 if flip else 0] += 1.0
        elif g.winner == "p2":
            wins[key][0 if flip else 1] += 1.0
        else:
            wins[key][0] += 0.5
            wins[key][1] += 0.5

    pairwise = [
        (a, b, wins_a, wins_b)
        for (a, b), (wins_a, wins_b) in wins.items()
    ]

    anchor_id = name_to_id[bot_names[0]]
    raw = compute_ratings(pairwise, anchor_id=anchor_id)

    out: Dict[str, Tuple[float, float, float]] = {}
    for pid, rating in raw.items():
        out[id_to_name[pid]] = rating
    return out


# ────────────────────────────────────────────────────────────────────────────
# Output
# ────────────────────────────────────────────────────────────────────────────

def _write_outputs(
    output_dir: Path,
    games: List[GameRecord],
    bot_names: List[str],
    n_games_per_pair: int,
    total_wall_s: float,
    halted: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # per_game.jsonl
    jsonl_path = output_dir / "per_game.jsonl"
    with jsonl_path.open("w") as f:
        for g in games:
            f.write(g.to_jsonl() + "\n")

    # H2H matrix + ratings
    n = len(bot_names)
    h2h = [[0 for _ in range(n)] for _ in range(n)]
    for g in games:
        i, j = bot_names.index(g.p1), bot_names.index(g.p2)
        if g.winner == "p1":
            h2h[i][j] += 1
        elif g.winner == "p2":
            h2h[j][i] += 1
        else:
            # draws not counted in H2H wins; show in summary separately
            pass

    h2h_path = output_dir / "h2h_matrix.csv"
    with h2h_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["winner\\loser"] + bot_names)
        for i, row in enumerate(h2h):
            w.writerow([bot_names[i]] + row)

    # BT ratings
    try:
        ratings = _compute_bt_ratings(games, bot_names)
    except Exception as exc:
        print(f"[warn] BT ratings failed: {exc}", file=sys.stderr)
        ratings = {n: (0.0, 0.0, 0.0) for n in bot_names}

    ratings_path = output_dir / "ratings.csv"
    with ratings_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bot", "elo", "ci_lo", "ci_hi"])
        for name in bot_names:
            r, lo, hi = ratings.get(name, (0.0, 0.0, 0.0))
            w.writerow([name, r, lo, hi])

    # colony_table.csv — per-bot colony-fraction mean when they won
    per_bot_wins: Dict[str, List[float]] = defaultdict(list)
    for g in games:
        if g.winner_bot:
            per_bot_wins[g.winner_bot].append(g.colony_fraction_winner)

    colony_path = output_dir / "colony_table.csv"
    with colony_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bot", "n_wins", "colony_fraction_mean"])
        for name in bot_names:
            wins_list = per_bot_wins.get(name, [])
            mean = sum(wins_list) / len(wins_list) if wins_list else 0.0
            w.writerow([name, len(wins_list), round(mean, 4)])

    # summary.md
    summary_path = output_dir / "summary.md"
    lines: List[str] = []
    lines.append("# Tourney summary (§176 Wave B smoke)")
    lines.append("")
    lines.append(f"- Bots: {', '.join(bot_names)} ({len(bot_names)})")
    lines.append(f"- Games per pair: {n_games_per_pair}")
    lines.append(f"- Total games: {len(games)}")
    lines.append(f"- Wall: {total_wall_s:.1f} s")
    lines.append(f"- Halted (max-wall-seconds): {halted}")
    lines.append("")
    lines.append("## BT ratings (Elo, anchor=" + bot_names[0] + ")")
    lines.append("")
    lines.append("| Bot | Elo | CI lo | CI hi |")
    lines.append("|---|---:|---:|---:|")
    ranked = sorted(
        bot_names, key=lambda n: ratings.get(n, (0.0, 0.0, 0.0))[0], reverse=True
    )
    for name in ranked:
        r, lo, hi = ratings.get(name, (0.0, 0.0, 0.0))
        lines.append(f"| {name} | {r:.1f} | {lo:.1f} | {hi:.1f} |")
    lines.append("")
    lines.append("## Head-to-head (rows = winner, cols = loser)")
    lines.append("")
    lines.append("| | " + " | ".join(bot_names) + " |")
    lines.append("|---|" + "|".join(["---"] * len(bot_names)) + "|")
    for i, row in enumerate(h2h):
        lines.append("| " + bot_names[i] + " | " + " | ".join(str(x) for x in row) + " |")
    lines.append("")
    lines.append("## Colony fractions (winner-side)")
    lines.append("")
    lines.append("| Bot | Wins | Colony frac mean |")
    lines.append("|---|---:|---:|")
    for name in bot_names:
        wins_list = per_bot_wins.get(name, [])
        mean = sum(wins_list) / len(wins_list) if wins_list else 0.0
        lines.append(f"| {name} | {len(wins_list)} | {mean:.3f} |")
    lines.append("")
    lines.append("## Latency (mean per move, seconds)")
    lines.append("")
    lines.append("| Bot | total_wall_s | max_move_s |")
    lines.append("|---|---:|---:|")
    per_bot_lat_total: Dict[str, List[float]] = defaultdict(list)
    per_bot_lat_max: Dict[str, List[float]] = defaultdict(list)
    for g in games:
        per_bot_lat_total[g.p1].append(g.p1_total_wall_s)
        per_bot_lat_total[g.p2].append(g.p2_total_wall_s)
        per_bot_lat_max[g.p1].append(g.p1_max_move_s)
        per_bot_lat_max[g.p2].append(g.p2_max_move_s)
    for name in bot_names:
        tot = per_bot_lat_total.get(name, [])
        mx = per_bot_lat_max.get(name, [])
        tot_mean = sum(tot) / len(tot) if tot else 0.0
        mx_mean = sum(mx) / len(mx) if mx else 0.0
        lines.append(f"| {name} | {tot_mean:.3f} | {mx_mean:.3f} |")
    lines.append("")

    summary_path.write_text("\n".join(lines))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="§176 Wave B tourney harness.")
    parser.add_argument(
        "--bots",
        type=str,
        required=True,
        help="Comma-separated bot names (see BOT_REGISTRY).",
    )
    parser.add_argument("--n_games", type=int, default=5)
    parser.add_argument("--max_plies", type=int, default=200)
    parser.add_argument(
        "--random_opening_plies", type=int, default=0,
        help="Unused for now; reserved for Wave C. Pre-pin in CLI.",
    )
    parser.add_argument(
        "--max-wall-seconds", type=int, default=0,
        help="Hard halt budget; 0 = unbounded. On overrun, dumps partial JSONL.",
    )
    parser.add_argument(
        "--our_model_ckpt",
        type=str,
        default="checkpoints/bootstrap_model_v6.pt",
        help="Checkpoint path for our_v6_* bots.",
    )
    parser.add_argument(
        "--our_n_sims", type=int, default=128,
        help="MCTS sims for our_v6_mcts128. Drop to 64 for fast smoke.",
    )
    parser.add_argument(
        "--our_ckpts", type=str, default="",
        help="Comma-separated list of additional checkpoint paths. Pair with --our_labels.",
    )
    parser.add_argument(
        "--our_labels", type=str, default="",
        help="Comma-separated labels matching --our_ckpts. Each label registers a "
        "BOT_REGISTRY entry with OurModelBot @ MCTS=--our_n_sims using that ckpt.",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    _register_default_bots(args.our_model_ckpt, args.our_n_sims)

    if args.our_ckpts:
        from hexo_rl.bots.our_model_bot import OurModelBot
        ckpts = [c.strip() for c in args.our_ckpts.split(",") if c.strip()]
        labels = [l.strip() for l in args.our_labels.split(",") if l.strip()]
        if len(ckpts) != len(labels):
            print(
                f"[err] --our_ckpts ({len(ckpts)}) and --our_labels ({len(labels)}) "
                f"must have same length",
                file=sys.stderr,
            )
            return 1
        for ckpt, label in zip(ckpts, labels):
            def _make(ckpt_path=ckpt, n_sims=args.our_n_sims):
                return OurModelBot(
                    checkpoint_path=ckpt_path,
                    config=_build_our_model_config(n_sims),
                    temperature=0.0,
                )
            BOT_REGISTRY[label] = _make
            print(f"[info] registered bot '{label}' @ MCTS-{args.our_n_sims} ckpt={ckpt}",
                  file=sys.stderr)

    bot_names = [s.strip() for s in args.bots.split(",") if s.strip()]
    for name in bot_names:
        if name not in BOT_REGISTRY:
            print(
                f"[err] unknown bot '{name}'. Known: {sorted(BOT_REGISTRY)}",
                file=sys.stderr,
            )
            return 1

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all pairs (i < j).
    pairs: List[Tuple[str, str]] = []
    for i in range(len(bot_names)):
        for j in range(i + 1, len(bot_names)):
            pairs.append((bot_names[i], bot_names[j]))

    print(
        f"[info] tourney: {len(bot_names)} bots, {len(pairs)} pairs, "
        f"{args.n_games} games/pair = {len(pairs) * args.n_games} games"
    )
    print(f"[info] output: {output_dir}")

    games: List[GameRecord] = []
    t_start = time.perf_counter()
    halted = False

    for pair in pairs:
        for gi in range(args.n_games):
            elapsed = time.perf_counter() - t_start
            if args.max_wall_seconds > 0 and elapsed > args.max_wall_seconds:
                print(
                    f"[halt] max-wall-seconds={args.max_wall_seconds}s exceeded "
                    f"(elapsed={elapsed:.1f}s). Dumping partial JSONL.",
                    file=sys.stderr,
                )
                halted = True
                break

            # Alternate P1/P2 every other game for side balance.
            if gi % 2 == 0:
                p1_name, p2_name = pair[0], pair[1]
            else:
                p1_name, p2_name = pair[1], pair[0]

            p1_factory = BOT_REGISTRY[p1_name]
            p2_factory = BOT_REGISTRY[p2_name]

            t_game_0 = time.perf_counter()
            try:
                record = _play_one_game(
                    p1_factory, p2_factory,
                    p1_name, p2_name,
                    pair_label=pair,
                    game_idx=gi,
                    max_plies=args.max_plies,
                )
            except Exception as exc:
                print(
                    f"[err] game {pair} #{gi} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue
            t_game = time.perf_counter() - t_game_0
            print(
                f"[game] {pair[0]} vs {pair[1]} #{gi}: "
                f"winner={record.winner_bot or 'draw'} plies={record.plies} "
                f"wall={t_game:.1f}s"
            )
            games.append(record)

        if halted:
            break

    total_wall = time.perf_counter() - t_start
    print(f"[info] all done. {len(games)} games in {total_wall:.1f}s.")

    _write_outputs(output_dir, games, bot_names, args.n_games, total_wall, halted)
    print(f"[info] outputs: {output_dir}/per_game.jsonl, summary.md, ratings.csv, h2h_matrix.csv, colony_table.csv")

    return 0 if not halted else 0  # halted still returns 0 — partial JSONL is intentional


if __name__ == "__main__":
    sys.exit(main())
