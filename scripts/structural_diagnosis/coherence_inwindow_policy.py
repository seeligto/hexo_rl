#!/usr/bin/env python3
"""§D-COHERENCE Phase 2 — INWINDOW-arm mechanism confirm: is the in-window finishing
degradation POLICY-side (de-sharpening on the winning move) or VALUE-side?

Phase 1 lit V-INWINDOW: in-window forced-win conversion drops 30k->87.5k (turn-level
d=-0.151 CI-cleared) while off-window share is flat and the in-window forced-win COUNT
per game-side RISES (not survivorship). This script asks WHICH head degrades — and it
re-derives the value-head behavior FROM SCRATCH on this pool (does NOT rely on the
§D-FRAGILITY value-AUC finding).

Method (fixed-pool, mirrors §D-FRAGILITY's design but self-contained): build ONE common
pool of near-terminal snapshots from the golong-arc self-play replays, two disjoint
classes by the engine detector on the turn-start snapshot:
  WIN-OPPTY : side-to-move has >=1 IN-WINDOW immediate forced win (the finishing
              opportunity the policy must convert). win_idx = its in-window win-move
              action indices.
  LOSE      : the OPPONENT has a forced win at the snapshot and the mover has none
              (the side-to-move is under a winning threat).
Then score EVERY checkpoint's POLICY and VALUE heads on the SAME pool:
  POLICY  finishing mass  = softmax prob on the in-window win move(s)  (WIN-OPPTY rows)
  POLICY  top1_hits_win   = argmax lands on a win move                 (WIN-OPPTY rows)
  VALUE   mean on WIN-OPPTY (expect -> +1 if value recognizes the win)
  VALUE   AUC(win>lose)    = P(value(WIN-OPPTY,won) > value(LOSE,lost)) — independent
                            value DISCRIMINATION on finishing-relevant positions.

Read: policy finishing mass FALLS / entropy RISES while value mean+AUC HOLD -> the model
still KNOWS it is winning but the POLICY won't pull the trigger = POLICY/target-shaping
degradation (the unexplored in-window line-coherence lever). If value ALSO sags -> value-
side / both.

EVAL-ONLY. Read-only on banked replays + checkpoints. Zero geometry literals
(policy_logit_count / kept_plane_indices from the spec; off-window via is_off_window).
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
from hexo_rl.diagnostics.forced_win_detector import depth1_wins, depth2_wins, is_off_window
from hexo_rl.encoding import lookup, normalize_encoding_name
from hexo_rl.training.checkpoints import load_inference_model


def _terminal_winner(mv, name):
    b = Board.with_encoding_name(name)
    for (q, r) in mv:
        try:
            b.apply_move(q, r)
        except Exception:
            return None
        if b.check_win():
            return int(b.winner())
    return None


def build_pool(files, name, spec, min_step, max_pos, seed):
    """One row per near-terminal snapshot: WIN-OPPTY (mover in-window forced win) or
    LOSE (opponent forced win, mover none). Records ground-truth terminal winner."""
    kept = list(spec.kept_plane_indices)
    S, P, LOGITS = spec.trunk_size, spec.n_source_planes, int(spec.policy_logit_count)
    games = []
    for f in files:
        try:
            fh = open(f)
        except FileNotFoundError:
            print(f"  (skip missing {f})", file=sys.stderr); continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("game_length", 0) <= 0:
                continue
            if int(d.get("checkpoint_step", 0)) < min_step:
                continue
            games.append(d)
    rng = random.Random(seed)
    rng.shuffle(games)

    win_rows = []
    for g in games:
        mv = [(int(q), int(r)) for (q, r) in g["moves"]]
        winner = _terminal_winner(mv, name)
        src = int(g.get("checkpoint_step", 0))
        board = Board.with_encoding_name(name)
        i, n = 0, len(mv)
        while i < n:
            cp = int(board.current_player)
            snap = board.clone()
            d1 = depth1_wins(snap, cp); d2 = depth2_wins(snap, cp)
            immediate = set(tuple(c) for c in d1) | set(tuple(f) for (f, _s) in d2)
            inwin = [c for c in immediate if not is_off_window(snap, c, spec)]
            if inwin:
                win_idx = sorted({int(snap.to_flat(int(q), int(r))) for (q, r) in inwin})
                win_idx = [k for k in win_idx if 0 <= k < LOGITS]
                if win_idx:
                    flat = np.asarray(snap.to_tensor(), dtype=np.float32).reshape(P, S, S)
                    win_rows.append({"wire": flat[kept], "win_idx": win_idx, "side": cp,
                                     "won": (winner == cp), "src_step": src,
                                     "depth2_only": (not d1)})
            while i < n:
                q, r = mv[i]
                try:
                    board.apply_move(q, r)
                except Exception:
                    i = n; break
                i += 1
                if board.check_win():
                    break
                if int(board.current_player) != cp:
                    break
    rng.shuffle(win_rows)
    if max_pos:
        win_rows = win_rows[:max_pos]
    return win_rows


def _auc(pos, neg):
    nw, nl = len(pos), len(neg)
    if nw == 0 or nl == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[:nw].sum() - nw * (nw + 1) / 2) / (nw * nl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--replays", nargs="+",
                    default=sorted(str(p) for p in
                                   Path("investigation/coherence_2026-06-08/replays").glob("games_2026-06-0*.jsonl")))
    ap.add_argument("--ckpt-dirs", nargs="+",
                    default=["investigation/coherence_2026-06-08/checkpoints",
                             "investigation/fragility_2026-06-07/checkpoints"])
    ap.add_argument("--min-step", type=int, default=30000)
    ap.add_argument("--max-pos", type=int, default=4000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", default="investigation/coherence_2026-06-08/inwindow_policy_value.json")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    LOGITS = int(spec.policy_logit_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[pool] in-window forced-win snapshots from {len(args.replays)} replays (ckpt_step >= {args.min_step}) ...")
    win_rows = build_pool(args.replays, name, spec, args.min_step, args.max_pos, args.seed)
    won_sub = sum(r["won"] for r in win_rows)
    src_steps = np.array([r["src_step"] for r in win_rows])
    print(f"[pool] WIN-OPPTY={len(win_rows)} (terminal-won {won_sub} / not-won {len(win_rows)-won_sub})")
    import collections
    print(f"[pool] by source checkpoint_step: {dict(sorted(collections.Counter(src_steps.tolist()).items()))}")
    if not win_rows:
        print("no win-oppty positions"); return 1

    Xw = np.stack([r["wire"] for r in win_rows]).astype(np.float32)
    win_masks = np.zeros((len(win_rows), LOGITS), dtype=bool)
    for j, r in enumerate(win_rows):
        win_masks[j, r["win_idx"]] = True
    won_mask = np.array([r["won"] for r in win_rows])
    # distributional test: source-bucket membership (do LATER-sourced in-window wins
    # finish worse under a FIXED model? -> harder positions, not head degradation)
    lo_src, hi_src = int(src_steps.min()), int(src_steps.max())
    lo_bucket = src_steps == lo_src
    hi_bucket = src_steps == hi_src

    ckpts = {}
    for dd in args.ckpt_dirs:
        for cp in Path(dd).glob("checkpoint_*.pt"):
            tail = cp.stem.replace("checkpoint_", "")
            if not tail.isdigit():
                continue  # skip _ema / _rewarm
            step = int(tail.lstrip("0") or "0")
            ckpts.setdefault(step, cp)
    steps = sorted(ckpts)
    print(f"[ckpts] {steps}")

    rng = np.random.default_rng(args.seed)

    def vforward(model, X):
        v = np.empty(len(X), dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, len(X), 512):
                xb = torch.from_numpy(X[b0:b0 + 512]).to(device)
                v[b0:b0 + len(xb)] = model(xb)[1].float().cpu().numpy().reshape(-1)
        return v

    out = {"encoding": name, "win_n": len(win_rows), "n_won": int(won_sub),
           "policy_logit_count": LOGITS, "min_step": args.min_step,
           "src_lo": lo_src, "src_hi": hi_src, "checkpoints": []}

    print(f"\n{'='*120}")
    print("§D-COHERENCE Phase 2 — POLICY finishing-mass + VALUE on a FIXED in-window forced-win pool")
    print("VAL AUC = value separates terminal-WON from terminal-NOT-WON within the finishing pool (self-contained)")
    print(f"distributional cols: p_win on positions SOURCED from ckpt-step {lo_src} vs {hi_src} (under the same model)")
    print(f"{'='*120}")
    hdr = (f"{'step':>7} | {'POLICY p_win':>22} | {'top1':>5} | {'entropy':>7} || "
           f"{'VAL mean(win)':>13} | {'VAL AUC(won>not)':>16} || "
           f"{'p_win@src'+str(lo_src//1000)+'k':>12} | {'p_win@src'+str(hi_src//1000)+'k':>12}")
    print(hdr); print("-" * len(hdr))

    for st in steps:
        model, _msp, _lab = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        pwin = np.empty(len(win_rows)); top1 = np.empty(len(win_rows), dtype=bool)
        ent = np.empty(len(win_rows))
        with torch.no_grad():
            for b0 in range(0, len(win_rows), 512):
                xb = torch.from_numpy(Xw[b0:b0 + 512]).to(device)
                p = torch.softmax(model(xb)[0].float(), dim=1).cpu().numpy()
                m = win_masks[b0:b0 + len(p)]
                pwin[b0:b0 + len(p)] = (p * m).sum(axis=1)
                top1[b0:b0 + len(p)] = m[np.arange(len(p)), p.argmax(axis=1)]
                ent[b0:b0 + len(p)] = -(p * np.log(p + 1e-12)).sum(axis=1)
        vwin = vforward(model, Xw)
        # self-contained value discrimination: terminal-WON vs terminal-NOT-WON finishing positions
        vauc = _auc(vwin[won_mask], vwin[~won_mask])

        def bci(vals):
            if len(vals) == 0:
                return (float("nan"), float("nan"))
            bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(args.nboot)]
            return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

        mp, mt, me = float(pwin.mean()), float(top1.mean()), float(ent.mean())
        mvw = float(vwin.mean())
        cpw = bci(pwin)
        pwin_lo = float(pwin[lo_bucket].mean()); pwin_hi = float(pwin[hi_bucket].mean())
        out["checkpoints"].append({
            "step": st, "p_win": mp, "p_win_ci": cpw, "top1_hits_win": mt,
            "policy_entropy": me, "value_mean_win": mvw,
            "value_auc_won_vs_notwon": (None if vauc != vauc else float(vauc)),
            "p_win_src_lo": pwin_lo, "p_win_src_hi": pwin_hi})
        print(f"{st:>7} | {mp:.4f} [{cpw[0]:.4f},{cpw[1]:.4f}] | {mt:.3f} | {me:>7.3f} || "
              f"{mvw:>13.3f} | {vauc:>16.3f} || {pwin_lo:>12.4f} | {pwin_hi:>12.4f}")

    cks = out["checkpoints"]
    if len(cks) >= 2:
        a, b = cks[0], cks[-1]
        print(f"\n{'-'*120}")
        print(f"ARC {a['step']} -> {b['step']} (HEAD degradation test, fixed pool):")
        print(f"  POLICY p_win    {a['p_win']:.4f} -> {b['p_win']:.4f}  (d={b['p_win']-a['p_win']:+.4f})   "
              f"top1 {a['top1_hits_win']:.3f} -> {b['top1_hits_win']:.3f}   entropy {a['policy_entropy']:.3f} -> {b['policy_entropy']:.3f}")
        print(f"  VALUE mean(win) {a['value_mean_win']:.3f} -> {b['value_mean_win']:.3f}   "
              f"AUC(won>not) {a['value_auc_won_vs_notwon']} -> {b['value_auc_won_vs_notwon']}")
        # distributional: average over checkpoints of p_win on lo-src vs hi-src positions
        plo = float(np.mean([c["p_win_src_lo"] for c in cks]))
        phi = float(np.mean([c["p_win_src_hi"] for c in cks]))
        print(f"\nDISTRIBUTIONAL test (avg over all {len(cks)} checkpoints): "
              f"p_win on src-{a['step'] if False else out['src_lo']} positions = {plo:.4f}  "
              f"vs src-{out['src_hi']} positions = {phi:.4f}   (d={phi-plo:+.4f})")
        print("  -> if hi-src << lo-src under the SAME models, the LATER in-window forced wins are intrinsically")
        print("     HARDER to finish (distributional / play-structure), not a policy-head finishing-skill regression.")
        out["arc_delta"] = {"from": a["step"], "to": b["step"],
                            "p_win_delta": b["p_win"] - a["p_win"],
                            "value_mean_win_delta": b["value_mean_win"] - a["value_mean_win"],
                            "dist_p_win_src_lo": plo, "dist_p_win_src_hi": phi}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
