#!/usr/bin/env python3
"""§D-FRAGILITY Phase 1 — value-head discrimination on SPREAD vs COMPACT positions.

DECISIVE A/B/C test. Pre-registered in reports/investigations/fragility_diagnosis_2026-06-07.md.

Does the value head separate WINNING-spread from LOSING-spread positions, or is it
mushy on SPREAD specifically (B), uniformly / fine (A), or fine (C)?

Eval-only. Builds ONE common, matched position pool from the golong-arc self-play
replays (checkpoint_step >= --min-checkpoint), then scores EVERY checkpoint's value
head on the SAME pool. Reuses the shared detectors — no metric copies:
  forced_win_detector: bbox_span, depth1_wins, depth2_wins  (clear-won + spread)
  golong_game_analysis: _bbox, _components                  (regime metrics)
  training.checkpoints.load_inference_model                 (value-head forward)

Output: JSON (per-checkpoint AUC_spread / AUC_compact + clear-won mean value +
value_fc2_weight_abs_max) + a printed summary. Zero geometry literals
(kept_plane_indices / planes from the encoding spec).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from engine import Board
from hexo_rl.encoding import lookup, normalize_encoding_name
from hexo_rl.diagnostics.forced_win_detector import (
    bbox_span, depth1_wins, depth2_wins, is_off_window, window_center, cheb,
)
from hexo_rl.training.checkpoints import load_inference_model

# regime metrics — reuse golong_game_analysis (no copy)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from golong_game_analysis import _bbox, _components  # noqa: E402


def build_pool(files, name, spec, min_ckpt, max_pos, seed):
    """Replay arc games; record one row per turn-start snapshot for the mover.

    Row = (wire_tensor (n_planes,S,S) float32, side, ply, bbox_span, density,
    n_components, clear_won, winner). winner is the engine terminal winner of the
    game (ground truth), None for a draw/non-terminal game.
    """
    kept = list(spec.kept_plane_indices)
    S = spec.trunk_size
    games = []
    for f in files:
        try:
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("game_length", 0) <= 0:
                    continue
                if d.get("checkpoint_step", 0) < min_ckpt:
                    continue
                games.append(d)
        except FileNotFoundError:
            print(f"  (skip missing {f})", file=sys.stderr)
    rng = random.Random(seed)
    rng.shuffle(games)

    rows = []
    for g in games:
        mv = [(int(q), int(r)) for (q, r) in g["moves"]]
        board = Board.with_encoding_name(name)
        # ground-truth terminal winner of THIS game (replay to end on a scratch board)
        scratch = Board.with_encoding_name(name)
        winner = None
        for (q, r) in mv:
            try:
                scratch.apply_move(q, r)
            except Exception:
                break
            if scratch.check_win():
                winner = int(scratch.winner())
                break
        # walk turn-by-turn, snapshot at each turn start
        i, n = 0, len(mv)
        while i < n:
            cp = int(board.current_player)
            snap = board.clone()
            # encode the turn-start position (the mover's decision point)
            flat = np.asarray(snap.to_tensor(), dtype=np.float32).reshape(spec.n_source_planes, S, S)
            wire = flat[kept]
            stones = [(int(q), int(r), int(p)) for (q, r, p) in snap.get_stones()]
            allc = [(q, r) for (q, r, _p) in stones]
            dq, dr, area = _bbox(allc)
            bsp = bbox_span(allc)
            density = (len(allc) / area) if area else 0.0
            comps = 0
            for p in (1, -1):
                comps += len(_components({(q, r) for (q, r, pp) in stones if pp == p}))
            d1 = depth1_wins(snap, cp)
            d2 = depth2_wins(snap, cp)
            win_cells = [tuple(c) for c in d1] + [tuple(c) for pr in d2 for c in pr]
            clear_won = bool(win_cells)
            # is ANY completing cell in-window (convertible), or off-window-only (unconvertible)?
            has_in_window = any(not is_off_window(snap, c, spec) for c in win_cells) if win_cells else False
            rows.append({
                "wire": wire, "side": cp, "ply": int(snap.ply),
                "bbox": int(bsp), "density": float(density), "ncomp": int(comps),
                "clear_won": clear_won, "clear_won_in_window": has_in_window, "winner": winner,
            })
            # advance this turn
            while i < n:
                q, r = mv[i]
                try:
                    board.apply_move(q, r)
                except Exception:
                    i = n
                    break
                i += 1
                if board.check_win():
                    break
                if int(board.current_player) != cp:
                    break
    rng.shuffle(rows)
    if max_pos and len(rows) > max_pos:
        rows = rows[:max_pos]
    return rows


def auc(pos_won, pos_lost):
    """Mann-Whitney AUC = P(value_won > value_lost). NaN if either side empty."""
    nw, nl = len(pos_won), len(pos_lost)
    if nw == 0 or nl == 0:
        return float("nan")
    allv = np.concatenate([pos_won, pos_lost])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_won = ranks[:nw].sum()
    return (r_won - nw * (nw + 1) / 2) / (nw * nl)


def auc_boot(pos_won, pos_lost, nboot, rng):
    if len(pos_won) == 0 or len(pos_lost) == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(nboot):
        w = pos_won[rng.integers(0, len(pos_won), len(pos_won))]
        l = pos_lost[rng.integers(0, len(pos_lost), len(pos_lost))]
        vals.append(auc(w, l))
    vals = np.array(vals)
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--ckpt-dir", default="investigation/fragility_2026-06-07/checkpoints")
    ap.add_argument("--replays", nargs="+",
                    default=["investigation/fragility_2026-06-07/replays/games_2026-06-06.jsonl",
                             "investigation/fragility_2026-06-07/replays/games_2026-06-07.jsonl"])
    ap.add_argument("--min-checkpoint", type=int, default=53000)
    ap.add_argument("--max-pos", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--out", default="investigation/fragility_2026-06-07/value_discrim.json")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[pool] building from {args.replays} (checkpoint_step >= {args.min_checkpoint}) ...")
    rows = build_pool(args.replays, name, spec, args.min_checkpoint, args.max_pos, args.seed)
    print(f"[pool] {len(rows)} positions")
    if not rows:
        print("no positions"); return 1

    X = np.stack([r["wire"] for r in rows]).astype(np.float32)        # (N, P, S, S)
    side = np.array([r["side"] for r in rows])
    winner = np.array([0 if r["winner"] is None else r["winner"] for r in rows])
    has_winner = np.array([r["winner"] is not None for r in rows])
    bbox = np.array([r["bbox"] for r in rows], dtype=np.float64)
    density = np.array([r["density"] for r in rows], dtype=np.float64)
    ncomp = np.array([r["ncomp"] for r in rows], dtype=np.float64)
    clear_won = np.array([r["clear_won"] for r in rows])
    cw_inwin = np.array([r["clear_won_in_window"] for r in rows])
    cw_offonly = clear_won & ~cw_inwin
    ply = np.array([r["ply"] for r in rows])

    # outcome for side-to-move: +1 won, -1 lost, 0 draw/none
    result = np.where(has_winner, np.where(winner == side, 1, -1), 0)
    won_mask = result == 1
    lost_mask = result == -1

    # regime split on the pool's OWN median (run distribution, not arbitrary)
    bbox_med = float(np.median(bbox))
    dens_med = float(np.median(density))
    comp_med = float(np.median(ncomp))
    spread_bbox = bbox > bbox_med            # primary (per mandate)
    spread_dens = density < dens_med         # robustness (low density = spread)
    spread_comp = ncomp > comp_med           # robustness (more comps = spread)

    print(f"[regime] bbox median={bbox_med:.1f}  density median={dens_med:.3f}  ncomp median={comp_med:.1f}")
    print(f"[outcome] won={won_mask.sum()} lost={lost_mask.sum()} draw/none={(result==0).sum()}")
    print(f"[ply] spread(bbox) mean ply={ply[spread_bbox].mean():.1f}  compact mean ply={ply[~spread_bbox].mean():.1f}")
    print(f"[clear-won] total={clear_won.sum()}  in-spread(bbox)={ (clear_won&spread_bbox).sum() }  in-compact={ (clear_won&~spread_bbox).sum() }")
    print(f"[clear-won IN-WINDOW (convertible)] spread={ (cw_inwin&spread_bbox).sum() }  compact={ (cw_inwin&~spread_bbox).sum() }   [OFF-only spread={ (cw_offonly&spread_bbox).sum() }]")

    ckpts = sorted(Path(args.ckpt_dir).glob("checkpoint_*.pt"))
    rng = np.random.default_rng(args.seed)
    out = {"pool_n": len(rows), "bbox_median": bbox_med, "density_median": dens_med,
           "ncomp_median": comp_med, "n_won": int(won_mask.sum()), "n_lost": int(lost_mask.sum()),
           "ply_spread_mean": float(ply[spread_bbox].mean()), "ply_compact_mean": float(ply[~spread_bbox].mean()),
           "checkpoints": []}

    def regime_auc(vals, spread_mask, label):
        sp_w = vals[spread_mask & won_mask]; sp_l = vals[spread_mask & lost_mask]
        co_w = vals[~spread_mask & won_mask]; co_l = vals[~spread_mask & lost_mask]
        a_sp = auc(sp_w, sp_l); a_co = auc(co_w, co_l)
        ci_sp = auc_boot(sp_w, sp_l, args.nboot, rng)
        ci_co = auc_boot(co_w, co_l, args.nboot, rng)
        return {"split": label, "auc_spread": a_sp, "ci_spread": ci_sp,
                "auc_compact": a_co, "ci_compact": ci_co,
                "n_spread_won": int(len(sp_w)), "n_spread_lost": int(len(sp_l)),
                "n_compact_won": int(len(co_w)), "n_compact_lost": int(len(co_l))}

    print(f"\nclear-won mean-value columns: cwSPD_IN = clear-won IN-WINDOW spread (should -> +1 if value healthy);")
    print(f"cwSPD_OFF = clear-won off-window-only spread (~0 is CORRECT, unconvertible); cwCMP_IN = clear-won IN-WINDOW compact.")
    print(f"\n{'ckpt':>8}  {'fc2max':>7}  {'AUC_all':>7}  {'AUC_spd':>7}  {'AUC_cmp':>7}  {'cwSPD_IN':>8}  {'cwSPD_OFF':>9}  {'cwCMP_IN':>8}")
    for cp in ckpts:
        step = int("".join(ch for ch in cp.stem if ch.isdigit()).lstrip("0") or "0")
        sd = torch.load(cp, map_location="cpu", weights_only=False)
        state = sd.get("model_state", sd)
        fc2 = float(state["value_fc2.weight"].abs().max())
        model, mspec, label = load_inference_model(cp, {}, device=device)
        model = model.float().eval()
        vals = np.empty(len(rows), dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, len(rows), 256):
                xb = torch.from_numpy(X[b0:b0 + 256]).to(device)
                out_t = model(xb)
                v = out_t[1].float().detach().cpu().numpy().reshape(-1)
                vals[b0:b0 + len(v)] = v
        a_all = auc(vals[won_mask], vals[lost_mask])
        prim = regime_auc(vals, spread_bbox, "bbox")
        dens = regime_auc(vals, spread_dens, "density")
        comp = regime_auc(vals, spread_comp, "ncomp")
        def mv(mask):
            return float(vals[mask].mean()) if mask.any() else float("nan")
        cw_spd = mv(clear_won & spread_bbox)
        cw_cmp = mv(clear_won & ~spread_bbox)
        # the decisive control: clear-won + IN-WINDOW (convertible) spread should read -> +1
        cw_spd_in = mv(cw_inwin & spread_bbox)
        cw_spd_off = mv(cw_offonly & spread_bbox)
        cw_cmp_in = mv(cw_inwin & ~spread_bbox)
        out["checkpoints"].append({
            "step": step, "value_fc2_abs_max": fc2, "auc_all": a_all,
            "regime": {"bbox": prim, "density": dens, "ncomp": comp},
            "clearwon_spread_meanv": cw_spd, "clearwon_compact_meanv": cw_cmp,
            "clearwon_spread_INWINDOW_meanv": cw_spd_in,
            "clearwon_spread_OFFONLY_meanv": cw_spd_off,
            "clearwon_compact_INWINDOW_meanv": cw_cmp_in,
            "mean_value_won": float(vals[won_mask].mean()), "mean_value_lost": float(vals[lost_mask].mean()),
        })
        print(f"{step:>8}  {fc2:>7.4f}  {a_all:>7.3f}  {prim['auc_spread']:>7.3f}  {prim['auc_compact']:>7.3f}  {cw_spd_in:>7.3f}  {cw_spd_off:>7.3f}  {cw_cmp_in:>7.3f}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
