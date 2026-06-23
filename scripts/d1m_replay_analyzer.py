#!/usr/bin/env python3
"""d1m_replay_analyzer.py — INTERIM (Python replay) golong kill-gate read.

Reconstructs per-game board STRUCTURE (longest_line + n_components) from the
self-play replay JSONL the D-1M Gumbel run persists, so the golong
fragmentation kill-gate can be read NOW — the live engine does NOT emit
n_components / longest_line per game (the monitor shows them "n/a — not
instrumented"). This is the INTERIM Python stand-in.

  *** ALL OUTPUTS LABELLED: INTERIM (Python replay) — superseded by S7 engine emit. ***

Why interim, why pin: a planned S7 Rust engine emit will produce these per game
on the hot path. To avoid a measurement DISCONTINUITY (CLAUDE.md "Verify the
measurement unit before building a frame on it"), every definition below is
pinned to the engine's EXISTING line/adjacency conventions so an interim Python
number is numerically comparable to the future Rust emit.

================================ PINNED DEFINITIONS ============================

Data source (SELF-PLAY traces, NOT eval games):
  hexo_rl/monitoring/game_recorder.py:64-93 (GameRecorder.maybe_record) writes
  one JSON object per sampled self-play game:
    moves           : [[q, r], ...]  AXIAL coords in PLAY ORDER (negatives ok —
                      theoretically-infinite board; NOT [row,col] indices)
    outcome         : "x_win" | "o_win" | "draw"   (x_win=P1, o_win=P2)
    game_length     : int PLIES (stone count, not compound moves)
    timestamp       : ISO-8601 UTC
    checkpoint_step : int training step
  Wiring: hexo_rl/selfplay/pool.py:798-802 (self._recorder.maybe_record(
  moves=move_history, winner_code=winner, game_length=plies)).

Ply -> player rule (HTTT: P1 opens 1 stone, then both alternate 2/turn):
  P1 | P2 P2 | P1 P1 | P2 P2 | ...   move 0 = P1; [1,2]=P2; [3,4]=P1; ...
  PINNED verbatim to hexo_rl/selfplay/instrumentation.py:103 (the canonical
  colony_extension_fraction owner), reused here so attribution is byte-identical
  to the live colony metric:
    own_is_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
  Engine ground truth for the turn structure: engine/src/board/state/core.rs
  :107-109,199-200,527-532 (moves_remaining 1 on ply0, then 2/turn, flip at 0).

Hex distance (used by BOTH adjacency below):
  axial_distance(a,b) = max(|dq|, |dr|, |dq+dr|)
  hexo_rl/utils/coordinates.py:89 — mirrors engine hex_distance
  (engine/src/board/state/core.rs:50, the (|dq|+|dr|+|ds|)/2 identity form).

longest_line  (per the relevant player):
  Longest straight run of one player's consecutive stones along ANY of the three
  hex axes. PINNED to the engine win-line scan:
    - WIN_LENGTH = 6 (engine/src/board/moves.rs:44).  6-in-a-row wins.
    - HEX_AXES = [(1,0), (0,1), (1,-1)] (engine/src/board/state/core.rs:54).
      Three positive axis directions; each walked BOTH ways (=6 directions).
    - count_in_line logic (engine/src/board/moves.rs:222-233):
      run = 1 + count_direction(+dq,+dr) + count_direction(-dq,-dr); max over axes.
    We reproduce count_in_line exactly (consecutive same-colour cells), per stone,
    taking the max => the player's longest line (capped naturally at WIN_LENGTH=6
    on a won game, the engine never extends past a 6-win).
  longest_line_fraction — DENOMINATOR DECISION (documented):
    The engine has NO longest_line_fraction today; colony_extension_fraction
    normalizes over a player's CLASSIFIED STONE COUNT (instrumentation.py:107-114).
    To stay consistent with THAT normalization convention we report
      longest_line_fraction = longest_line / max(1, player_stone_count)
    i.e. fraction of a player's stones that lie on its single longest line —
    a structural-coherence ratio that FALLS as stones fragment off the main run
    (the golong "longest_line down" signal). We ALSO surface the raw
    longest_line (engine count_in_line unit, 0..6+) and longest_line / WIN_LENGTH
    so a reviewer can pick whichever the S7 emit ends up choosing.
    >>> RISK: the S7 Rust emit's denominator is not yet specified. If S7 picks
        /WIN_LENGTH instead of /player_stone_count the absolute scale shifts
        (both reported here; the TREND direction is identical either way). <<<

n_components (per the relevant player):
  Connected components of a player's stones under the SAME connectivity the
  engine uses for clustering: BFS where an edge exists iff
  axial_distance(a,b) <= CLUSTER_THRESHOLD.
    - PINNED to get_clusters (engine/src/board/moves.rs:478-520): BFS/flood-fill,
      edge iff hex_distance <= self.cluster_threshold.
    - DEFAULT_CLUSTER_THRESHOLD = 5 (engine/src/board/moves.rs:70).  Default used.
  WHICH PLAYER: the engine get_clusters is COLOR-BLIND (clusters ALL stones of
  both colors, for NN windowing). The golong kill-gate tracked the WINNER'S
  structure fragmenting, so we report n_components PER-PLAYER and headline the
  WINNER's value (draws: report both, headline max).
    >>> RISK: definitional fork vs engine get_clusters. The live engine
        get_clusters is color-blind; we use the SAME adjacency+threshold but
        split by player to match the golong "winner fragments" semantics. If S7
        emits the color-blind both-colors component count instead, the absolute
        n_components differs (one merged graph vs two per-color graphs). We emit
        BOTH the per-player (winner) value and the color-blind all-stones value
        so the future comparison is unambiguous.   Threshold (5) is the lever:
        document it; S7 must use the same threshold or the count is incomparable.

colony_extension_fraction (carried for the co-occurrence read, recomputed here):
  PINNED verbatim to instrumentation.py:86-114 — a stone is a "colony
  extension" iff its min axial_distance to ANY opponent stone > 6
  (_COLONY_EXT_HEX_DIST=6); fraction over both players' classified stones.
  Recomputed from replay moves so the kill-gate triplet is self-contained.

forced_win_conversion:
  NOT present in the replay JSONL (it's a monitor-side forced_win_trend event,
  read by scripts/d1m_monitor.py from the d1m log, NOT the replay file). We
  CANNOT compute it from replays. Left as an explicit TODO hook; this analyzer
  surfaces the OTHER two legs of the triplet (n_components UP + longest_line
  DOWN) which ARE computable here, and prints where to read the third leg.

distinct-game fraction (OPTIONAL SALVAGE, INTERIM):
  distinct(hash(moves)) / total over the sample — the §D-ARGMAX effective-n
  measure the LIVE monitor cannot compute (no game-id in its log). We can here
  by hashing the move sequence. Byte-identical move sequences are deduped.

Board geometry sourced from the encoding registry (NOT hardcoded):
  registry encoding v6_live2_ls board_size=19 (engine/src/encoding/registry.toml).
  board_size only bounds the NN window; replay coords are GLOBAL axial and may
  exceed it — we do NOT clip, structure is computed in unbounded axial space
  (matches the engine, which stores cells in a HashMap, not a fixed array).

===============================================================================

Usage:
  .venv/bin/python scripts/d1m_replay_analyzer.py                 # remote D-1M, sample 200
  .venv/bin/python scripts/d1m_replay_analyzer.py --local FILE    # offline dev file
  .venv/bin/python scripts/d1m_replay_analyzer.py --sample 500    # cap N games
  .venv/bin/python scripts/d1m_replay_analyzer.py --validate      # print ONE fully-worked game
  .venv/bin/python scripts/d1m_replay_analyzer.py --per-game      # per-game table
  .venv/bin/python scripts/d1m_replay_analyzer.py --bin 30000     # checkpoint_step bin width

Env (match d1m_monitor.py):
  D1M_HOST (default vast)  D1M_REPO (default /workspace/hexo_rl)
  Remote replay path default: <repo>/logs/replays/games_<UTC-today>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

INTERIM_LABEL = "INTERIM (Python replay) — superseded by S7 engine emit."

# ---------------------------------------------------------------------------
# PINNED engine constants (sourced, not invented — see header for file:line).
# ---------------------------------------------------------------------------
# engine/src/board/state/core.rs:54 — three positive hex axes (walked both ways).
HEX_AXES = [(1, 0), (0, 1), (1, -1)]
# engine/src/board/moves.rs:44
WIN_LENGTH = 6
# engine/src/board/moves.rs:70 (DEFAULT_CLUSTER_THRESHOLD)
DEFAULT_CLUSTER_THRESHOLD = 5
# hexo_rl/selfplay/instrumentation.py:19 (_COLONY_EXT_HEX_DIST)
COLONY_EXT_HEX_DIST = 6


def board_size_from_registry(name: str = "v6_live2_ls") -> int:
    """Read board_size from the canonical encoding registry (no hardcode).

    Falls back to a registry default lookup; raises if the encoding is missing
    so a typo can't silently substitute a wrong geometry.
    """
    try:
        from hexo_rl.encoding import lookup  # type: ignore
        enc = lookup(name)
        bs = getattr(enc, "board_size", None)
        if bs is None and isinstance(enc, dict):
            bs = enc.get("board_size")
        if bs:
            return int(bs)
    except Exception:
        pass
    # Minimal TOML fallback so the script runs even without the package import.
    reg = os.path.join(os.path.dirname(__file__), "..", "engine", "src",
                       "encoding", "registry.toml")
    reg = os.path.abspath(reg)
    cur = None
    try:
        with open(reg) as f:
            for ln in f:
                s = ln.strip()
                if s.startswith("[encodings."):
                    cur = s[len("[encodings."):].rstrip("]")
                elif cur == name and s.startswith("board_size"):
                    return int(s.split("=", 1)[1].strip())
    except Exception:
        pass
    raise SystemExit("could not resolve board_size for encoding %r" % name)


# ---------------------------------------------------------------------------
# Pinned geometry helpers (mirror engine / instrumentation exactly).
# ---------------------------------------------------------------------------

def axial_distance(a, b):
    """max(|dq|,|dr|,|dq+dr|) — hexo_rl/utils/coordinates.py:89 == engine hex_distance."""
    dq = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    ds = abs((a[0] + a[1]) - (b[0] + b[1]))
    return max(dq, dr, ds)


def split_players(moves):
    """Return (p1_stones, p2_stones) using the PINNED ply->player rule.

    Verbatim from instrumentation.py:103 — keeps attribution byte-identical to
    the live colony_extension metric. P1 opens 1, then both alternate 2/turn.
    """
    p1, p2 = [], []
    for ply, (q, r) in enumerate(moves):
        own_is_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
        (p1 if own_is_p1 else p2).append((q, r))
    return p1, p2


def longest_line(stones):
    """Longest straight consecutive run of `stones` along any HEX_AXES axis.

    Reproduces engine count_in_line (moves.rs:222-233): for each stone, for each
    axis, run = 1 + walk(+dir) + walk(-dir); take the global max. The engine's
    last-move fast path is an optimization of the same quantity; scanning every
    stone is the exhaustive equivalent and matches find_winning_line's fallback.
    """
    if not stones:
        return 0
    cells = set(stones)
    best = 0

    def walk(q, r, dq, dr):
        n = 0
        while True:
            q += dq
            r += dr
            if (q, r) not in cells:
                break
            n += 1
        return n

    for (q, r) in cells:
        for (dq, dr) in HEX_AXES:
            run = 1 + walk(q, r, dq, dr) + walk(q, r, -dq, -dr)
            if run > best:
                best = run
    return best


def n_components(stones, threshold=DEFAULT_CLUSTER_THRESHOLD):
    """Connected components of `stones` under axial_distance <= threshold.

    Same connectivity as engine get_clusters (moves.rs:478-520): BFS flood-fill,
    edge iff hex_distance <= cluster_threshold (default 5). Applied per-player
    here (golong winner-structure semantics); the color-blind all-stones variant
    is computed separately by the caller for the S7-comparison risk note.
    """
    pts = list(set(stones))
    n = len(pts)
    if n == 0:
        return 0
    seen = [False] * n
    comps = 0
    for i in range(n):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            c = stack.pop()
            for j in range(n):
                if not seen[j] and axial_distance(pts[c], pts[j]) <= threshold:
                    seen[j] = True
                    stack.append(j)
    return comps


def colony_extension(moves):
    """(ext_count, total) — VERBATIM instrumentation.py:86-114.

    A stone is a colony extension iff min axial_distance to ANY opponent stone
    > COLONY_EXT_HEX_DIST (6). Fraction over both players' classified stones.
    """
    if not moves:
        return (0, 0)
    p1, p2 = split_players(moves)
    ext = total = 0
    for own, opp in ((p1, p2), (p2, p1)):
        if not opp:
            continue
        for s in own:
            total += 1
            if min(axial_distance(s, o) for o in opp) > COLONY_EXT_HEX_DIST:
                ext += 1
    return (ext, total)


# ---------------------------------------------------------------------------
# Per-game analysis.
# ---------------------------------------------------------------------------
WINNER_BY_OUTCOME = {"x_win": "p1", "o_win": "p2", "draw": None}


def analyze_game(rec, threshold=DEFAULT_CLUSTER_THRESHOLD):
    """Compute the per-game structure dict. `rec` is a parsed replay record."""
    moves = [tuple(m) for m in rec.get("moves", [])]
    outcome = rec.get("outcome")
    p1, p2 = split_players(moves)
    winner = WINNER_BY_OUTCOME.get(outcome)

    ll_p1, ll_p2 = longest_line(p1), longest_line(p2)
    nc_p1, nc_p2 = n_components(p1, threshold), n_components(p2, threshold)
    nc_blind = n_components(p1 + p2, threshold)  # color-blind (engine get_clusters style)

    # Headline = the relevant player's structure: winner if decisive, else max.
    if winner == "p1":
        ll, nc, stones = ll_p1, nc_p1, len(p1)
    elif winner == "p2":
        ll, nc, stones = ll_p2, nc_p2, len(p2)
    else:  # draw — report the more-structured side (max line / most components)
        if ll_p1 >= ll_p2:
            ll, stones = ll_p1, len(p1)
        else:
            ll, stones = ll_p2, len(p2)
        nc = max(nc_p1, nc_p2)

    ext_count, ext_total = colony_extension(moves)
    return {
        "outcome": outcome,
        "winner": winner or "draw",
        "checkpoint_step": rec.get("checkpoint_step"),
        "game_length_plies": rec.get("game_length"),
        "n_stones_p1": len(p1),
        "n_stones_p2": len(p2),
        # headline (relevant-player) structure:
        "longest_line": ll,
        "longest_line_fraction": ll / max(1, stones),     # /player_stone_count (colony norm)
        "longest_line_frac_of_win": ll / WIN_LENGTH,       # /WIN_LENGTH (alt denom)
        "n_components": nc,
        # both-player detail + color-blind variant (for S7 comparison):
        "longest_line_p1": ll_p1, "longest_line_p2": ll_p2,
        "n_components_p1": nc_p1, "n_components_p2": nc_p2,
        "n_components_colorblind": nc_blind,
        # colony co-signal:
        "colony_ext_count": ext_count,
        "colony_ext_total": ext_total,
        "colony_extension_fraction": (ext_count / ext_total) if ext_total else 0.0,
        # effective-n salvage:
        "moves_hash": hash_moves(moves),
    }


def hash_moves(moves):
    import hashlib
    h = hashlib.sha1(json.dumps([list(m) for m in moves]).encode()).hexdigest()
    return h[:12]


# ---------------------------------------------------------------------------
# Data loading (SELF-PLAY replay only; sampling with NO silent truncation).
# ---------------------------------------------------------------------------

def default_remote_path(repo):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return "%s/logs/replays/games_%s.jsonl" % (repo, today)


def load_lines_remote(host, path, timeout=90):
    cmd = ["ssh", host, "cat '%s'" % path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit("ssh timeout (>%ds) reading %s:%s" % (timeout, host, path))
    if p.returncode != 0:
        msg = [m for m in (p.stderr or "").splitlines()
               if "vast.ai" not in m and "Have fun" not in m]
        raise SystemExit("ssh rc=%d reading %s:%s: %s" % (
            p.returncode, host, path, " ".join(msg[-3:]) or "?"))
    return p.stdout.splitlines()


def load_lines_local(path):
    try:
        with open(path) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        raise SystemExit("local file not found: %s" % path)


def parse_records(lines):
    recs = []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("{"):
            continue
        try:
            d = json.loads(s)
            if "moves" in d:          # self-play replay record (NOT an eval game)
                recs.append(d)
        except Exception:
            pass
    return recs


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def fnum(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def co_occurrence_read(games, bin_w):
    """Bin per-game structure by checkpoint_step; surface the kill-gate trend.

    Kill-gate = forced_win_conversion DOWN  AND  n_components UP  AND
    longest_line DOWN co-occurring. forced_win_conversion is NOT in the replay
    file (monitor-side); we surface the two computable legs + colony co-signal.
    """
    bins = defaultdict(lambda: {"ll": [], "nc": [], "col": [], "n": 0})
    for g in games:
        cs = g.get("checkpoint_step")
        b = (cs // bin_w) * bin_w if isinstance(cs, int) else "?"
        e = bins[b]
        e["ll"].append(g["longest_line_fraction"])
        e["nc"].append(g["n_components"])
        e["col"].append(g["colony_extension_fraction"])
        e["n"] += 1
    rows = []
    for b in sorted(bins, key=lambda x: (x == "?", x)):
        e = bins[b]
        rows.append({
            "bin": b,
            "n": e["n"],
            "longest_line_frac_mean": sum(e["ll"]) / len(e["ll"]),
            "n_components_mean": sum(e["nc"]) / len(e["nc"]),
            "colony_ext_frac_mean": sum(e["col"]) / len(e["col"]),
        })
    return rows


def print_validate(rec, threshold):
    """Print ONE fully-worked game so a reviewer can hand-check the script."""
    moves = [tuple(m) for m in rec.get("moves", [])]
    p1, p2 = split_players(moves)
    print("=" * 78)
    print("VALIDATE — one fully-worked game   [%s]" % INTERIM_LABEL)
    print("=" * 78)
    print("outcome=%s  checkpoint_step=%s  game_length(plies)=%s" % (
        rec.get("outcome"), rec.get("checkpoint_step"), rec.get("game_length")))
    print("\nmoves (ply: [q,r] -> player), PINNED rule "
          "own_is_p1=(ply==0) or (((ply-1)//2)%2==1):")
    for ply, (q, r) in enumerate(moves):
        own_is_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
        print("  ply %3d: (%3d,%3d) -> %s" % (ply, q, r, "P1" if own_is_p1 else "P2"))
    print("\nP1 stones (%d): %s" % (len(p1), sorted(p1)))
    print("P2 stones (%d): %s" % (len(p2), sorted(p2)))

    print("\n--- longest_line (engine count_in_line; HEX_AXES=%s, WIN_LENGTH=%d) ---"
          % (HEX_AXES, WIN_LENGTH))
    for tag, st in (("P1", p1), ("P2", p2)):
        ll = longest_line(st)
        print("  %s longest_line = %d   (/stones=%d -> frac %.3f ; /WIN_LENGTH -> %.3f)"
              % (tag, ll, len(st), ll / max(1, len(st)), ll / WIN_LENGTH))

    print("\n--- n_components (axial_distance <= threshold=%d, engine get_clusters) ---"
          % threshold)
    for tag, st in (("P1", p1), ("P2", p2)):
        print("  %s n_components = %d" % (tag, n_components(st, threshold)))
    print("  COLOR-BLIND (all stones, engine get_clusters style) = %d"
          % n_components(p1 + p2, threshold))

    ec, et = colony_extension(moves)
    print("\n--- colony_extension (instrumentation.py verbatim, hex_dist>%d) ---"
          % COLONY_EXT_HEX_DIST)
    print("  ext_count=%d  total=%d  fraction=%.3f"
          % (ec, et, (ec / et) if et else 0.0))

    g = analyze_game(rec, threshold)
    print("\n--- HEADLINE per-game record (winner=%s) ---" % g["winner"])
    for k in ("longest_line", "longest_line_fraction", "longest_line_frac_of_win",
              "n_components", "n_components_colorblind", "colony_extension_fraction",
              "moves_hash"):
        print("  %-26s %s" % (k, g[k]))
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="INTERIM Python replay golong kill-gate read (n_components + longest_line)")
    ap.add_argument("--local", metavar="FILE",
                    help="read a local replay JSONL instead of remote vast")
    ap.add_argument("--remote-path", metavar="PATH",
                    help="explicit remote replay path (default: today's games_<date>.jsonl)")
    ap.add_argument("--sample", type=int, default=200,
                    help="cap N games analyzed (no silent truncation; prints sampled X of Y)")
    ap.add_argument("--bin", type=int, default=30000,
                    help="checkpoint_step bin width for the co-occurrence read")
    ap.add_argument("--threshold", type=int, default=DEFAULT_CLUSTER_THRESHOLD,
                    help="cluster adjacency threshold (engine default %d)" % DEFAULT_CLUSTER_THRESHOLD)
    ap.add_argument("--validate", action="store_true",
                    help="print ONE fully-worked game for hand-check, then exit")
    ap.add_argument("--per-game", action="store_true",
                    help="print the per-game table")
    ap.add_argument("--encoding", default="v6_live2_ls",
                    help="registry encoding for board geometry (default v6_live2_ls)")
    args = ap.parse_args()

    host = os.environ.get("D1M_HOST", "vast")
    repo = os.environ.get("D1M_REPO", "/workspace/hexo_rl")
    bs = board_size_from_registry(args.encoding)

    # ---- load (SELF-PLAY replay only) ----
    if args.local:
        src = args.local
        lines = load_lines_local(args.local)
    else:
        src = "%s:%s" % (host, args.remote_path or default_remote_path(repo))
        rp = args.remote_path or default_remote_path(repo)
        lines = load_lines_remote(host, rp)
    recs = parse_records(lines)
    total = len(recs)
    if total == 0:
        raise SystemExit("no self-play replay records in %s" % src)

    # sample cap — explicit, no silent truncation
    sampled = recs[:args.sample]
    print("# %s" % INTERIM_LABEL)
    print("# source (self-play traces): %s" % src)
    print("# board_size=%d (registry encoding=%s)  WIN_LENGTH=%d  HEX_AXES=%s  cluster_threshold=%d"
          % (bs, args.encoding, WIN_LENGTH, HEX_AXES, args.threshold))
    print("# sampled %d of %d games" % (len(sampled), total))

    if args.validate:
        # pick the first DECISIVE game (clearer hand-check than a 150-ply draw)
        pick = next((r for r in sampled if r.get("outcome") in ("x_win", "o_win")), sampled[0])
        print_validate(pick, args.threshold)
        return

    games = [analyze_game(r, args.threshold) for r in sampled]

    # ---- per-game table ----
    if args.per_game:
        print("\nPER-GAME  [%s]" % INTERIM_LABEL)
        hdr = ("step", "outcome", "win", "ll", "ll_frac", "ll/6", "n_comp",
               "nc_blind", "colony", "hash")
        print("  " + "  ".join("%-9s" % h for h in hdr))
        for g in games:
            print("  " + "  ".join("%-9s" % x for x in (
                g["checkpoint_step"], g["outcome"], g["winner"],
                g["longest_line"], fnum(g["longest_line_fraction"]),
                fnum(g["longest_line_frac_of_win"]),
                g["n_components"], g["n_components_colorblind"],
                fnum(g["colony_extension_fraction"]), g["moves_hash"])))

    # ---- aggregate ----
    n = len(games)
    mean = lambda k: sum(g[k] for g in games) / n
    print("\nAGGREGATE over %d sampled games  [%s]" % (n, INTERIM_LABEL))
    print("  longest_line (mean)            %s" % fnum(mean("longest_line"), 2))
    print("  longest_line_fraction (mean)   %s   (/player_stone_count)" % fnum(mean("longest_line_fraction")))
    print("  longest_line_frac_of_win (mean)%s   (/WIN_LENGTH=6)" % fnum(mean("longest_line_frac_of_win")))
    print("  n_components winner (mean)     %s" % fnum(mean("n_components"), 2))
    print("  n_components colorblind (mean) %s" % fnum(mean("n_components_colorblind"), 2))
    print("  colony_extension_fraction(mean)%s" % fnum(mean("colony_extension_fraction")))

    # ---- effective-n salvage (distinct-game fraction; §D-ARGMAX) ----
    hashes = [g["moves_hash"] for g in games]
    distinct = len(set(hashes))
    print("\nEFFECTIVE-N (distinct-game fraction)  [%s]" % INTERIM_LABEL)
    print("  distinct move-sequences = %d / %d  (fraction %.3f)"
          % (distinct, n, distinct / n))
    print("  -> the §D-ARGMAX measure the live monitor CANNOT compute (no game-id");
    print("     in its log). Low fraction = deterministic regime, CI over raw")
    print("     game count is over-confident by ~sqrt(copies).")

    # ---- co-occurrence kill-gate read ----
    rows = co_occurrence_read(games, args.bin)
    print("\nGOLONG KILL-GATE co-occurrence read (by checkpoint_step bin=%d)  [%s]"
          % (args.bin, INTERIM_LABEL))
    print("  kill-gate = forced_win_conversion DOWN  AND  n_components UP  AND  longest_line DOWN")
    print("  %-10s %-5s %-22s %-20s %-22s" % (
        "bin", "n", "longest_line_frac_mean", "n_components_mean", "colony_ext_frac_mean"))
    for r in rows:
        print("  %-10s %-5d %-22s %-20s %-22s" % (
            r["bin"], r["n"], fnum(r["longest_line_frac_mean"]),
            fnum(r["n_components_mean"], 2), fnum(r["colony_ext_frac_mean"])))
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        ll_dn = b["longest_line_frac_mean"] < a["longest_line_frac_mean"]
        nc_up = b["n_components_mean"] > a["n_components_mean"]
        col_up = b["colony_ext_frac_mean"] > a["colony_ext_frac_mean"]
        legs = []
        legs.append("longest_line %s" % ("DOWN (kill-leg)" if ll_dn else "up/flat"))
        legs.append("n_components %s" % ("UP (kill-leg)" if nc_up else "down/flat"))
        legs.append("colony %s" % ("UP (co-signal)" if col_up else "down/flat"))
        print("  first->last bin: " + " | ".join(legs))
        if ll_dn and nc_up:
            print("  >>> BOTH computable kill-legs fire. Check forced_win_conversion")
            print("      (scripts/d1m_monitor.py forced_win_trend) to complete the triplet.")
        else:
            print("  >>> kill-gate NOT triggered on the computable legs (good).")
    print("\n  NOTE: forced_win_conversion is NOT in the replay JSONL — read it from")
    print("        the d1m log via scripts/d1m_monitor.py (forced_win_trend event).")
    print("        [TODO hook: join on checkpoint_step to complete the triplet.]")
    print("\n# %s" % INTERIM_LABEL)


if __name__ == "__main__":
    main()
