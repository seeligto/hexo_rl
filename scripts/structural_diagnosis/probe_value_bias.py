#!/usr/bin/env python3
"""§S181 IMPL-S181T1 — bootstrap value/policy-head colony-bias probe.

Standalone diagnostic. INSPECTION-ONLY. No training, no hot-path, no config.

What it does
------------
1. Corpus colony/extension statistics from raw_human game JSONs +
   data/bootstrap_corpus.npz position tensors, bucketed by Elo band.
2. Builds a canonical position bank: N colony positions (compact stone
   clusters) + N extension positions (long thin chains), v6 8-plane format.
3. Forwards two models on the bank — the trained anchor
   `bootstrap_model_v6.pt` and a fresh-init bias floor — and records
   value-head + policy-head outputs.
4. Emits a JSON sidecar (probe_value_bias_results.json) for the audit md.

Why it imports HexTacToeNet
---------------------------
Reconstructing the 12-block SE-ResNet trunk standalone would be ~400 LOC of
error-prone duplication and could silently diverge from the trained
architecture, invalidating the probe. The model *class* is imported; all
probe LOGIC (position construction, bias metrics, CIs) is standalone here.
No selfplay / training / MCTS code paths are touched.

Run:  .venv/bin/python scripts/structural_diagnosis/probe_value_bias.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Model class only — see module docstring for the import-boundary rationale.
from hexo_rl.model.network import HexTacToeNet  # noqa: E402

BOARD = 19
N_PLANES = 8
CENTER = BOARD // 2  # 9
RNG = np.random.default_rng(20260522)

# Elo bands — matches data/corpus/manifest.json banding.
ELO_BANDS = [
    ("sub_1000", 0, 1000),
    ("1000_1200", 1000, 1200),
    ("1200_1400", 1200, 1400),
    ("1400_plus", 1400, 10_000),
]


# ---------------------------------------------------------------------------
# Geometry — hex axial coords. Game JSONs store (x, y) axial offsets from
# centre. 6 axial neighbour directions.
# ---------------------------------------------------------------------------
HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def hex_dist(a, b):
    """Axial hex distance."""
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (abs(dx) + abs(dy) + abs(dx + dy)) // 2


def max_chain_len(coords):
    """Longest straight contiguous run of co-linear adjacent stones (no gaps).

    coords: set of (x,y) axial. Returns length of longest run along any of
    the 3 hex axes (each axis = a direction + its negation).
    """
    if not coords:
        return 0
    cset = set(coords)
    best = 1
    axes = [(1, 0), (0, 1), (1, -1)]
    for c in cset:
        for d in axes:
            # only count runs starting at a chain head (no predecessor)
            prev = (c[0] - d[0], c[1] - d[1])
            if prev in cset:
                continue
            n = 1
            cur = c
            while (cur[0] + d[0], cur[1] + d[1]) in cset:
                cur = (cur[0] + d[0], cur[1] + d[1])
                n += 1
            best = max(best, n)
    return best


def mean_pairwise_dist(coords):
    """Mean pairwise hex distance over a stone set. Compactness metric."""
    cs = list(coords)
    if len(cs) < 2:
        return 0.0
    tot = 0
    cnt = 0
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            tot += hex_dist(cs[i], cs[j])
            cnt += 1
    return tot / cnt


# ---------------------------------------------------------------------------
# PART 1 — corpus colony/extension statistics from raw_human JSONs
# ---------------------------------------------------------------------------
def corpus_stats():
    raw_dir = ROOT / "data/corpus/raw_human"
    qs_path = ROOT / "data/corpus/quality_scores.json"
    quality = json.loads(qs_path.read_text()) if qs_path.exists() else {}

    band_acc = {b[0]: defaultdict(list) for b in ELO_BANDS}
    band_acc["_all"] = defaultdict(list)
    n_games = 0
    n_skipped = 0

    files = sorted(raw_dir.glob("*.json"))
    for fp in files:
        try:
            g = json.loads(fp.read_text())
        except Exception:
            n_skipped += 1
            continue
        players = g.get("players", [])
        if len(players) != 2:
            n_skipped += 1
            continue
        elos = [p.get("elo") for p in players]
        if any(e is None for e in elos):
            n_skipped += 1
            continue
        avg_elo = sum(elos) / 2.0
        moves = g.get("moves", [])
        if len(moves) < 12:
            n_skipped += 1
            continue

        # Reconstruct per-player final stone sets.
        # Move 1 = single opener; thereafter players alternate placing 2.
        # playerIds map to seat; first move's playerId == player 1.
        by_player = defaultdict(set)
        for m in moves:
            pid = m.get("playerId")
            x, y = m.get("x"), m.get("y")
            if pid is None or x is None or y is None:
                continue
            by_player[pid].add((int(x), int(y)))

        result = g.get("gameResult", {})
        winner_pid = result.get("winningPlayerId")
        first_pid = moves[0].get("playerId")

        # winner stone set = the colony/extension subject (the line that won)
        win_set = by_player.get(winner_pid, set())
        all_stones = set()
        for s in by_player.values():
            all_stones |= s

        glen = len(moves)
        mcl = max_chain_len(win_set)
        mpd = mean_pairwise_dist(win_set)
        p1_win = 1.0 if winner_pid == first_pid else 0.0

        band = None
        for name, lo, hi in ELO_BANDS:
            if lo <= avg_elo < hi:
                band = name
                break
        rec = {
            "glen": glen,
            "mcl": mcl,
            "mpd": mpd,
            "p1_win": p1_win,
            "n_win_stones": len(win_set),
            "n_all_stones": len(all_stones),
        }
        for tgt in (band_acc["_all"], band_acc.get(band, None)):
            if tgt is None:
                continue
            for k, v in rec.items():
                tgt[k].append(v)
        n_games += 1

    def summarize(acc):
        if not acc.get("glen"):
            return None
        glen = np.array(acc["glen"], float)
        mcl = np.array(acc["mcl"], float)
        mpd = np.array(acc["mpd"], float)
        p1 = np.array(acc["p1_win"], float)
        nws = np.array(acc["n_win_stones"], float)
        # operational colony/extension classification on the winning line:
        #   extension  := max straight chain >= 5 (winning lines are 6-runs;
        #                 >=5 captures the near-complete winning extension)
        #   colony     := max chain <= 3 AND mean pairwise dist <= 2.6
        #   (positions that win MUST contain a 6-chain, so the winning-line
        #    classification is dominated by extension; the colony fraction
        #    here measures how *compact* the surrounding cluster is — see md)
        ext = (mcl >= 5)
        compact = (mpd <= 2.6)
        return {
            "n_games": int(len(glen)),
            "median_game_len": float(np.median(glen)),
            "mean_game_len": float(glen.mean()),
            "mean_max_chain": float(mcl.mean()),
            "mean_pairwise_dist_winline": float(mpd.mean()),
            "mean_win_stones": float(nws.mean()),
            "p1_win_rate": float(p1.mean()),
            "p1_win_ci95": wilson_ci(int(p1.sum()), len(p1)),
            "frac_winline_extension_ge5": float(ext.mean()),
            "frac_winline_compact_mpd_le2p6": float(compact.mean()),
        }

    out = {"_skipped": n_skipped, "_total_games": n_games}
    for name in [b[0] for b in ELO_BANDS] + ["_all"]:
        out[name] = summarize(band_acc[name])
    return out


# ---------------------------------------------------------------------------
# PART 1b — position-level colony/extension fraction from corpus.npz
# ---------------------------------------------------------------------------
def corpus_position_stats(sample_n=40_000):
    npz = ROOT / "data/bootstrap_corpus.npz"
    if not npz.exists():
        return {"error": f"{npz} absent"}
    d = np.load(npz, mmap_mode="r")
    states = d["states"]
    weights = d["weights"]
    n_total = states.shape[0]
    idx = np.sort(RNG.choice(n_total, size=min(sample_n, n_total), replace=False))

    # plane 0 = current-player most-recent stones; plane 4 = opponent.
    ext_cnt = 0
    colony_cnt = 0
    neither = 0
    n_used = 0
    chain_hist = defaultdict(int)
    w_ext, w_col = [], []
    for i in idx:
        st = np.asarray(states[i], dtype=np.float32)
        for plane in (0, 4):  # both players
            ys, xs = np.nonzero(st[plane] > 0.5)
            coords = set(zip(xs.tolist(), ys.tolist()))
            if len(coords) < 4:
                continue
            n_used += 1
            mcl = max_chain_len(coords)
            mpd = mean_pairwise_dist(coords)
            chain_hist[min(mcl, 8)] += 1
            is_ext = mcl >= 4
            is_col = (mpd <= 2.2) and (mcl <= 3)
            if is_ext:
                ext_cnt += 1
                w_ext.append(float(weights[i]))
            elif is_col:
                colony_cnt += 1
                w_col.append(float(weights[i]))
            else:
                neither += 1
    return {
        "sampled_positions": int(len(idx)),
        "stone_sets_classified": n_used,
        "frac_extension_ge4": ext_cnt / max(n_used, 1),
        "frac_colony_compact": colony_cnt / max(n_used, 1),
        "frac_neither": neither / max(n_used, 1),
        "mean_weight_extension": float(np.mean(w_ext)) if w_ext else None,
        "mean_weight_colony": float(np.mean(w_col)) if w_col else None,
        "max_chain_histogram": {str(k): chain_hist[k] for k in sorted(chain_hist)},
        "definitions": {
            "extension": "max straight chain >= 4 (no gaps) in a player's stone set",
            "colony": "mean pairwise hex dist <= 2.2 AND max chain <= 3",
        },
    }


# ---------------------------------------------------------------------------
# PART 2/3 — canonical position bank + model probes
# ---------------------------------------------------------------------------
def _empty_state():
    return np.zeros((N_PLANES, BOARD, BOARD), dtype=np.float32)


def _place(state, cur_coords, opp_coords):
    """Write stones into v6 8-plane tensor.

    plane 0..3 = current player history (t0..t-3), most-recent-first.
    plane 4..7 = opponent history.
    A static position has no history depth; replicate the t0 occupancy
    into all 4 history planes (a position that has *been* there all along).
    This matches how a quiescent mid-cluster looks: all stones present
    across the recent window.
    """
    for (x, y) in cur_coords:
        gx, gy = x + CENTER, y + CENTER
        if 0 <= gx < BOARD and 0 <= gy < BOARD:
            for p in range(0, 4):
                state[p, gy, gx] = 1.0
    for (x, y) in opp_coords:
        gx, gy = x + CENTER, y + CENTER
        if 0 <= gx < BOARD and 0 <= gy < BOARD:
            for p in range(4, 8):
                state[p, gy, gx] = 1.0


def build_colony_position(size, color="cur", jitter=0):
    """Compact hex blob of `size` stones for `color`; opponent gets a small
    scattered set elsewhere. Returns (state, meta)."""
    # grow a compact blob from origin via BFS over hex neighbours
    blob = [(jitter, 0)]
    frontier = list(blob)
    seen = set(blob)
    while len(blob) < size and frontier:
        c = frontier.pop(0)
        for d in HEX_DIRS:
            nc = (c[0] + d[0], c[1] + d[1])
            if nc not in seen:
                seen.add(nc)
                blob.append(nc)
                frontier.append(nc)
                if len(blob) >= size:
                    break
    blob = blob[:size]
    # opponent: a few stones far from the blob (neutral filler)
    opp = [(8, -4), (8, -5), (7, -3)]
    cur, oppc = (blob, opp) if color == "cur" else (opp, blob)
    st = _empty_state()
    _place(st, cur, oppc)
    return st, {
        "kind": "colony", "size": size, "color": color,
        "max_chain": max_chain_len(set(blob)),
        "mean_pair_dist": mean_pairwise_dist(set(blob)),
    }


def build_extension_position(length, color="cur", jitter=0):
    """Straight thin chain of `length` stones along a hex axis."""
    line = [(i, jitter) for i in range(length)]
    opp = [(8, -4), (8, -5), (7, -3)]
    cur, oppc = (line, opp) if color == "cur" else (opp, line)
    st = _empty_state()
    _place(st, cur, oppc)
    return st, {
        "kind": "extension", "length": length, "color": color,
        "max_chain": max_chain_len(set(line)),
        "mean_pair_dist": mean_pairwise_dist(set(line)),
    }


def build_bank(n_each=50):
    """N colony + N extension positions, matched on stone count.

    Matching: a colony of S stones is compared to an extension of S stones.
    Stone counts span 4..9 (the regime where chain/compactness diverge and
    below the 6-in-row win that ends the game). Both colors covered.
    """
    colony, extension = [], []
    sizes = [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]  # weighted toward mid
    i = 0
    while len(colony) < n_each:
        s = sizes[i % len(sizes)]
        color = "cur" if (i % 2 == 0) else "opp"
        jit = (i % 3) - 1
        st, meta = build_colony_position(s, color, jit)
        meta["idx"] = i
        colony.append((st, meta))
        i += 1
    i = 0
    while len(extension) < n_each:
        s = sizes[i % len(sizes)]
        color = "cur" if (i % 2 == 0) else "opp"
        jit = (i % 3) - 1
        st, meta = build_extension_position(s, color, jit)
        meta["idx"] = i
        extension.append((st, meta))
        i += 1
    return colony, extension


# ---------------------------------------------------------------------------
# PART 2c — discriminating near-win / open-N sub-probe
# ---------------------------------------------------------------------------
# A sharper hand-built test than the matched 50+50 bank. Four positions,
# all current-player perspective, same opponent filler as the main bank:
#
#   open 5-in-row : a straight co-linear chain of 5 cur stones whose BOTH
#                   axis endpoints are empty -> either endpoint completes a
#                   6-in-row. This is the canonical "one move from win"
#                   extension. Reuses build_extension_position(5): line at
#                   (0,0)..(4,0); the two win-completing cells are (-1,0)
#                   and (5,0).
#   compact 5-blob: the equivalent-stone-count compact hex blob (5 stones,
#                   BFS-grown). Reuses build_colony_position(5).
#   open 4-in-row : straight co-linear chain of 4 cur stones (no embedded
#                   win; one step short of the open-5). build_extension_position(4).
#   compact 4-blob: 4-stone compact hex blob. build_colony_position(4).
#
# Construction choice (documented per task brief): the audit §2 text names
# the 4 positions but not their exact coords. The open-5 argmax cells the
# audit cites — (-1,0) and (5,0) — uniquely fix the open-5 as the straight
# line (0,0)..(4,0) along axis (1,0). The remaining three are reconstructed
# from the same two canonical builders (straight axis chain / BFS hex blob)
# already used for the main bank, with color="cur", jitter=0, so the
# sub-probe shares the main bank's exact geometry conventions. No new
# position primitives are introduced.
SUBPROBE_SPEC = [
    ("open_5_in_row", "extension", 5),
    ("compact_5_blob", "colony", 5),
    ("open_4_in_row", "extension", 4),
    ("compact_4_blob", "colony", 4),
]


def build_subprobe_positions():
    """Build the 4 hand-built near-win / open-N discriminating positions.

    Returns list of (name, state, meta) — all current-player, jitter 0.
    """
    out = []
    for name, kind, n in SUBPROBE_SPEC:
        if kind == "extension":
            st, meta = build_extension_position(n, color="cur", jitter=0)
        else:
            st, meta = build_colony_position(n, color="cur", jitter=0)
        meta = dict(meta)
        meta["subprobe_name"] = name
        out.append((name, st, meta))
    return out


def argmax_cell(idx):
    """Map a 362-action policy index to (x, y) centre-relative axial coords.
    Returns 'pass' for the pass slot (361)."""
    if idx == 361:
        return "pass"
    return [idx % BOARD - CENTER, idx // BOARD - CENTER]


def probe_subprobe(model, positions):
    """Forward `model` on the 4 sub-probe positions; record value + argmax."""
    res = []
    with torch.no_grad():
        for name, st, meta in positions:
            t = torch.from_numpy(st[None]).float()
            out = model(t)
            log_policy, value, v_logit = out[0], out[1], out[2]
            lp = log_policy[0].numpy()
            top5 = np.argsort(lp)[::-1][:5].tolist()
            argmax = int(top5[0])
            res.append({
                "name": name,
                "meta": meta,
                "value": float(value[0].item()),
                "v_logit": float(v_logit[0].item()),
                "argmax": argmax,
                "argmax_cell": argmax_cell(argmax),
                "argmax_is_pass": argmax == 361,
                "top5": top5,
                "top5_cells": [argmax_cell(a) for a in top5],
                "top5_logp": [float(lp[a]) for a in top5],
            })
    return res


def load_anchor():
    p = ROOT / "checkpoints/bootstrap_model_v6.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = HexTacToeNet(encoding="v6")
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


def build_fresh_init():
    """v6-architecture fresh-init bias floor.

    Task names archive_quarantine/bootstrap_model_random_init_v6w25.pt — but
    that file is v6w25 architecture (626 actions, 25x25). It is NOT
    arch-compatible with the v6 anchor and cannot be a same-arch floor.
    A fresh HexTacToeNet(encoding='v6') with a fixed seed IS the correct
    same-architecture floor and is what we use.
    """
    torch.manual_seed(20260522)
    model = HexTacToeNet(encoding="v6")
    model.eval()
    return model


def probe_model(model, bank, tag):
    colony, extension = bank
    res = {"colony": [], "extension": []}
    with torch.no_grad():
        for label, items in (("colony", colony), ("extension", extension)):
            for st, meta in items:
                t = torch.from_numpy(st[None]).float()
                out = model(t)
                # forward returns (log_policy, value, v_logit, *extras)
                log_policy, value, v_logit = out[0], out[1], out[2]
                lp = log_policy[0].numpy()
                val = float(value[0].item())
                vlog = float(v_logit[0].item())
                top5 = np.argsort(lp)[::-1][:5].tolist()
                argmax = int(top5[0])
                res[label].append({
                    "meta": meta,
                    "value": val,
                    "v_logit": vlog,
                    "argmax": argmax,
                    "argmax_is_pass": argmax == 361,
                    "top5": top5,
                    "top5_logp": [float(lp[a]) for a in top5],
                })
    return res


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, (c - h) / d), min(1.0, (c + h) / d)]


def mean_ci(xs, z=1.96):
    a = np.asarray(xs, float)
    n = len(a)
    if n == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": m, "sd": float(a.std(ddof=1)) if n > 1 else 0.0,
            "ci95": [m - z * se, m + z * se], "n": n}


def welch(xs, ys):
    """Welch's t-test on two independent samples. Returns t, df, two-sided p."""
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": None, "df": None, "p_approx": None}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": None, "df": None, "p_approx": None}
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # normal-approx two-sided p (df large enough here)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"t": float(t), "df": float(df), "p_approx": float(p)}


def analyze_probe(res, tag):
    cv = [r["value"] for r in res["colony"]]
    ev = [r["value"] for r in res["extension"]]
    c_pass = sum(r["argmax_is_pass"] for r in res["colony"])
    e_pass = sum(r["argmax_is_pass"] for r in res["extension"])

    # policy: does argmax on a position extend the player's own line?
    # for colony positions, "extends colony" = argmax cell is hex-adjacent
    # to the existing cluster (compaction) vs far (reach-out).
    def adjacency_frac(items):
        adj = 0
        tot = 0
        for r in items:
            am = r["argmax"]
            if am == 361:
                continue
            ax, ay = am % BOARD - CENTER, am // BOARD - CENTER
            meta = r["meta"]
            # recover own coords from meta kind
            if meta["kind"] == "colony":
                # rebuild blob deterministically same as builder
                st, _ = build_colony_position(meta["size"], meta["color"],
                                              meta["idx"] % 3 - 1)
            else:
                st, _ = build_extension_position(meta["length"], meta["color"],
                                                 meta["idx"] % 3 - 1)
            plane = 0 if meta["color"] == "cur" else 4
            ys, xs = np.nonzero(st[plane] > 0.5)
            own = set(zip((xs - CENTER).tolist(), (ys - CENTER).tolist()))
            tot += 1
            if any(hex_dist((ax, ay), o) <= 1 for o in own):
                adj += 1
        return adj, tot

    c_adj, c_tot = adjacency_frac(res["colony"])
    e_adj, e_tot = adjacency_frac(res["extension"])

    # Sharper extension probe: for a straight thin chain the threat-correct
    # move is a co-linear endpoint extension (grows the run toward 6). A
    # colony-biased policy instead attaches off-axis (thickens the blob).
    # Count, over extension positions, how often argmax = the co-linear
    # endpoint cell vs an off-axis adjacent cell.
    def extension_endpoint_frac(items):
        endpoint = 0
        offaxis = 0
        other = 0
        tot = 0
        for r in items:
            meta = r["meta"]
            if meta["kind"] != "extension":
                continue
            am = r["argmax"]
            if am == 361:
                continue
            ax, ay = am % BOARD - CENTER, am // BOARD - CENTER
            length = meta["length"]
            jit = meta["idx"] % 3 - 1
            line = [(i, jit) for i in range(length)]
            lset = set(line)
            # the two co-linear endpoint extension cells (axis = (1,0))
            ends = {(line[0][0] - 1, jit), (line[-1][0] + 1, jit)}
            tot += 1
            if (ax, ay) in ends:
                endpoint += 1
            elif any(hex_dist((ax, ay), o) <= 1 for o in lset):
                offaxis += 1
            else:
                other += 1
        return {"endpoint": endpoint, "offaxis": offaxis,
                "other": other, "total": tot}

    # Sharper colony probe: for a compact blob, does argmax break OUT
    # (a cell far from the blob centroid — reaching toward an extension
    # line) or stay IN (thicken the blob)? out = max-hex-dist to any stone
    # >= 2; in = adjacent (dist 1).
    def colony_breakout_frac(items):
        out = 0
        inside = 0
        tot = 0
        for r in items:
            meta = r["meta"]
            if meta["kind"] != "colony":
                continue
            am = r["argmax"]
            if am == 361:
                continue
            ax, ay = am % BOARD - CENTER, am // BOARD - CENTER
            st, _ = build_colony_position(meta["size"], meta["color"],
                                          meta["idx"] % 3 - 1)
            plane = 0 if meta["color"] == "cur" else 4
            ys, xs = np.nonzero(st[plane] > 0.5)
            own = set(zip((xs - CENTER).tolist(), (ys - CENTER).tolist()))
            mind = min(hex_dist((ax, ay), o) for o in own)
            tot += 1
            if mind >= 2:
                out += 1
            else:
                inside += 1
        return {"breakout": out, "stay_compact": inside, "total": tot}

    ext_ep = extension_endpoint_frac(res["extension"])
    col_bo = colony_breakout_frac(res["colony"])

    return {
        "tag": tag,
        "value_colony": mean_ci(cv),
        "value_extension": mean_ci(ev),
        "value_delta_colony_minus_extension": (
            float(np.mean(cv) - np.mean(ev))),
        "value_welch": welch(cv, ev),
        "argmax_pass_colony": [c_pass, len(res["colony"])],
        "argmax_pass_extension": [e_pass, len(res["extension"])],
        "argmax_adjacent_to_own_colony": [c_adj, c_tot,
                                          wilson_ci(c_adj, c_tot)],
        "argmax_adjacent_to_own_extension": [e_adj, e_tot,
                                             wilson_ci(e_adj, e_tot)],
        "extension_endpoint_breakdown": ext_ep,
        "extension_endpoint_frac": (
            ext_ep["endpoint"] / max(ext_ep["total"], 1)),
        "extension_endpoint_ci95": wilson_ci(ext_ep["endpoint"],
                                             ext_ep["total"]),
        "colony_breakout_breakdown": col_bo,
        "colony_breakout_frac": col_bo["breakout"] / max(col_bo["total"], 1),
        "colony_breakout_ci95": wilson_ci(col_bo["breakout"],
                                          col_bo["total"]),
    }


def main():
    print("[1] corpus game-level stats (raw_human JSONs)...", flush=True)
    cstats = corpus_stats()
    print("    games:", cstats["_total_games"], "skipped:", cstats["_skipped"])

    print("[1b] corpus position-level colony/extension fractions...",
          flush=True)
    pstats = corpus_position_stats()

    print("[2] building canonical position bank (50 colony + 50 ext)...",
          flush=True)
    bank = build_bank(50)

    print("[3] loading anchor + fresh-init floor...", flush=True)
    anchor = load_anchor()
    fresh = build_fresh_init()

    print("[4] probing anchor...", flush=True)
    anchor_raw = probe_model(anchor, bank, "anchor")
    print("[4] probing fresh-init floor...", flush=True)
    fresh_raw = probe_model(fresh, bank, "fresh_init")

    anchor_an = analyze_probe(anchor_raw, "anchor")
    fresh_an = analyze_probe(fresh_raw, "fresh_init")

    print("[5] discriminating near-win / open-N sub-probe...", flush=True)
    subprobe_positions = build_subprobe_positions()
    subprobe_anchor = probe_subprobe(anchor, subprobe_positions)
    subprobe_fresh = probe_subprobe(fresh, subprobe_positions)

    out = {
        "meta": {
            "script": "scripts/structural_diagnosis/probe_value_bias.py",
            "anchor": "checkpoints/bootstrap_model_v6.pt",
            "anchor_sha_prefix": "7ab77d2c",
            "encoding": "v6",
            "n_colony": 50, "n_extension": 50,
            "fresh_init_note": (
                "fresh HexTacToeNet(encoding='v6') seed=20260522; the "
                "quarantine random-init file is v6w25 arch — not "
                "arch-compatible with the v6 anchor, cannot be the floor"),
        },
        "corpus_game_stats": cstats,
        "corpus_position_stats": pstats,
        "probe_anchor": anchor_an,
        "probe_fresh_init": fresh_an,
        "subprobe_near_win": {
            "description": (
                "Discriminating hand-built sub-probe — 4 positions, "
                "current-player perspective. open_5_in_row = straight "
                "5-chain (0,0)..(4,0), both axis endpoints (-1,0)/(5,0) "
                "empty -> either completes a 6-in-row. compact_5_blob = "
                "5-stone BFS hex blob. open_4_in_row / compact_4_blob = "
                "the 4-stone analogues. Built by build_subprobe_positions() "
                "from the same straight-axis / BFS-blob primitives as the "
                "main 50+50 bank (color=cur, jitter=0)."),
            "anchor": subprobe_anchor,
            "fresh_init": subprobe_fresh,
        },
    }
    op = ROOT / "scripts/structural_diagnosis/probe_value_bias_results.json"
    op.write_text(json.dumps(out, indent=2))
    print("written", op)

    # console summary
    a = anchor_an
    print("\n=== ANCHOR value-head ===")
    print(f"  colony   value mean {a['value_colony']['mean']:.4f} "
          f"CI {a['value_colony']['ci95']}")
    print(f"  extension value mean {a['value_extension']['mean']:.4f} "
          f"CI {a['value_extension']['ci95']}")
    print(f"  delta (colony-ext) {a['value_delta_colony_minus_extension']:.4f}"
          f"  welch p={a['value_welch']['p_approx']}")
    print(f"  argmax adjacent-to-own: colony {a['argmax_adjacent_to_own_colony']}"
          f"  ext {a['argmax_adjacent_to_own_extension']}")
    print(f"  extension endpoint frac {a['extension_endpoint_frac']:.3f} "
          f"CI {a['extension_endpoint_ci95']}  {a['extension_endpoint_breakdown']}")
    print(f"  colony breakout frac {a['colony_breakout_frac']:.3f} "
          f"CI {a['colony_breakout_ci95']}  {a['colony_breakout_breakdown']}")
    f = fresh_an
    print("=== FRESH-INIT floor ===")
    print(f"  delta (colony-ext) {f['value_delta_colony_minus_extension']:.4f}"
          f"  welch p={f['value_welch']['p_approx']}")

    print("\n=== ANCHOR near-win sub-probe ===")
    for r in subprobe_anchor:
        print(f"  {r['name']:16s} value {r['value']:+.4f}  "
              f"argmax_cell {r['argmax_cell']}  "
              f"top5_cells {r['top5_cells']}")


if __name__ == "__main__":
    main()
