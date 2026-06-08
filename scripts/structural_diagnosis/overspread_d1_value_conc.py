#!/usr/bin/env python3
"""§D-OVERSPREAD D1 — VALUE-DISCRIMINATION CEILING in the CONCENTRATION framing.

DRIVER QUESTION (do NOT trust the banked AUC(won>not)=0.79): the banked value AUC is
OUTCOME discrimination (terminal-WON vs terminal-NOT-WON). It says nothing about whether
the value head can rank a CONCENTRATED won position above a SCATTERED won position. If the
value head is blind to own-force concentration WITHIN won positions, it cannot drive search
away from over-spread -> D1 LIT (value-first fix).

METHOD (extends coherence_inwindow_policy.build_pool + coherence_overspread structure):
  1. Build the §D-COHERENCE fixed in-window forced-win pool (one row per turn-start snapshot
     where the mover has >=1 IN-WINDOW immediate forced win). Same detector wiring.
  2. RESTRICT to terminal-WON rows only (won == True). This holds OUTCOME constant so the
     remaining variance is concentration, NOT outcome.
  3. Label each WON row by the mover's OWN-force concentration on the SAME snapshot, using
     the coherence_overspread metrics (pure board replay, no NN):
         mover_ncomp        (PRIMARY; FEWER components = more concentrated)
         largest_blob_frac  (HIGHER = more concentrated)
         local_support      (HIGHER support around the win = more concentrated)
         turn_redundancy    (count_winning_turns; HIGHER fork redundancy = concentrated)
  4. For EACH of the 11 checkpoints, score the value head on the WON pool and compute
         AUC_conc = P(value(CONCENTRATED won) > value(SCATTERED won))
     under (a) median split and (b) tertile-extreme split on mover_ncomp (primary), plus
     a continuous Spearman-style rank check, and corroborating AUC_conc on the 3 secondary
     labels. Report per-checkpoint + arc trend.

PRE-REGISTERED VERDICT (no threshold moves):
  LIT iff outcome-matched AUC_conc <= 0.60 AND flat/declining over the arc.
  OUT iff value ranks concentration well (AUC_conc clearly > 0.60).

§2 guard: A2 avg-pool was falsified for value DISCRIMINATION-COLLAPSE (aggregation-arch);
§D-FRAGILITY/§D-COHERENCE value AUC = OUTCOME discrimination. NEITHER tested
concentration-ranking-within-won. Transfer NOT pre-excluded -> tested fresh here.

EVAL-ONLY. Read-only on banked replays + checkpoints. Zero geometry literals
(policy_logit_count / kept_plane_indices from spec; off-window via is_off_window; hex
adjacency = HEX_AXES). New untracked script; git diff --stat stays empty for tracked source.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import (
    HEX_AXES, depth1_wins, depth2_wins, is_off_window,
)
from hexo_rl.encoding import lookup, normalize_encoding_name
from hexo_rl.training.checkpoints import load_inference_model

# turn-correct primitive (NOT depth-1) per the locked TURN-vs-PLY correction
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turn_wins import count_winning_turns  # noqa: E402

_NB = HEX_AXES + [(-q, -r) for (q, r) in HEX_AXES]   # 6-neighbour hex adjacency (coherence_overspread)


def _components(cells):
    """Connected-component sizes of an own-stone set under hex adjacency (coherence_overspread)."""
    seen, out = set(), []
    for s in cells:
        if s in seen:
            continue
        q = deque([s]); seen.add(s); sz = 0
        while q:
            c = q.popleft(); sz += 1
            for dq, dr in _NB:
                nx = (c[0] + dq, c[1] + dr)
                if nx in cells and nx not in seen:
                    seen.add(nx); q.append(nx)
        out.append(sz)
    return out


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


def build_won_pool(files, name, spec, min_step, max_pos, seed):
    """In-window forced-win turn-start snapshots, RESTRICTED to terminal-WON. Each row carries
    the wire tensor (for value scoring) + the coherence_overspread concentration labels measured
    on the SAME snapshot."""
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

    rows = []
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
            if inwin and (winner == cp):   # RESTRICT to terminal-WON only
                win_idx = sorted({int(snap.to_flat(int(q), int(r))) for (q, r) in inwin})
                win_idx = [k for k in win_idx if 0 <= k < LOGITS]
                if win_idx:
                    # concentration labels on the SAME snapshot (coherence_overspread metrics)
                    stones = [(int(q), int(r), int(pp)) for (q, r, pp) in snap.get_stones()]
                    mine = {(q, r) for (q, r, pp) in stones if pp == cp}
                    mc = _components(mine) if mine else [0]
                    ncomp = len(mc)
                    largest_frac = (max(mc) / len(mine)) if mine else 0.0
                    supp = sum(((wc[0] + dq, wc[1] + dr) in mine)
                               for wc in inwin for dq in (-1, 0, 1) for dr in (-1, 0, 1))
                    local_support = supp / len(inwin)
                    turn_red = count_winning_turns(snap, cp)   # turn-correct redundancy
                    flat = np.asarray(snap.to_tensor(), dtype=np.float32).reshape(P, S, S)
                    rows.append({
                        "wire": flat[kept], "src_step": src,
                        "mover_ncomp": ncomp, "largest_frac": largest_frac,
                        "local_support": local_support, "turn_red": turn_red,
                        "mover_stones": len(mine),
                    })
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
    rng.shuffle(rows)
    if max_pos:
        rows = rows[:max_pos]
    return rows


def _auc(pos, neg):
    """P(random pos-sample value > random neg-sample value), tie-corrected (Mann-Whitney)."""
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


def _auc_boot(v_conc, v_scat, rng, nboot):
    """Bootstrap CI for AUC over BOTH groups resampled."""
    if len(v_conc) == 0 or len(v_scat) == 0:
        return (float("nan"), float("nan"))
    bs = []
    for _ in range(nboot):
        a = v_conc[rng.integers(0, len(v_conc), len(v_conc))]
        b = v_scat[rng.integers(0, len(v_scat), len(v_scat))]
        bs.append(_auc(a, b))
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


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
    ap.add_argument("--max-pos", type=int, default=6000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/d1_value_conc.json")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[pool] terminal-WON in-window forced-win snapshots from {len(args.replays)} replays "
          f"(ckpt_step >= {args.min_step}) ...")
    rows = build_won_pool(args.replays, name, spec, args.min_step, args.max_pos, args.seed)
    if not rows:
        print("no terminal-WON in-window forced-win positions"); return 1
    src_steps = np.array([r["src_step"] for r in rows])
    ncomp = np.array([r["mover_ncomp"] for r in rows], dtype=np.float64)
    lfrac = np.array([r["largest_frac"] for r in rows], dtype=np.float64)
    lsupp = np.array([r["local_support"] for r in rows], dtype=np.float64)
    tred = np.array([r["turn_red"] for r in rows], dtype=np.float64)
    print(f"[pool] WON rows = {len(rows)}")
    print(f"[pool] by source checkpoint_step: {dict(sorted(collections.Counter(src_steps.tolist()).items()))}")
    print(f"[pool] mover_ncomp: mean={ncomp.mean():.2f} median={np.median(ncomp):.1f} "
          f"[{np.percentile(ncomp,33):.1f}..{np.percentile(ncomp,67):.1f} tertile cuts]")
    print(f"[pool] largest_frac mean={lfrac.mean():.3f}  local_support mean={lsupp.mean():.3f}  "
          f"turn_red mean={tred.mean():.3f}")

    # ---- concentration splits (PRIMARY = mover_ncomp; concentrated = FEWER components) ----
    # median split: concentrated = ncomp < median ; scattered = ncomp > median (drop ==median ties)
    med = np.median(ncomp)
    conc_med = ncomp < med
    scat_med = ncomp > med
    # tertile-extreme split on ncomp
    t_lo = np.percentile(ncomp, 33.333)
    t_hi = np.percentile(ncomp, 66.667)
    conc_ter = ncomp <= t_lo      # fewest components = most concentrated
    scat_ter = ncomp >= t_hi      # most components = most scattered
    print(f"[split] PRIMARY mover_ncomp median={med:.1f}: "
          f"conc(n<med)={conc_med.sum()} scat(n>med)={scat_med.sum()} ties(==med)={(ncomp==med).sum()}")
    print(f"[split] tertile-extreme: conc(n<={t_lo:.1f})={conc_ter.sum()} scat(n>={t_hi:.1f})={scat_ter.sum()}")

    # secondary labels: concentrated direction = HIGH for frac/support/turn_red
    def hi_lo_split(arr):
        m = np.median(arr)
        return (arr > m), (arr < m)   # concentrated = above median, scattered = below

    conc_frac, scat_frac = hi_lo_split(lfrac)
    conc_supp, scat_supp = hi_lo_split(lsupp)
    conc_tred, scat_tred = hi_lo_split(tred)

    X = np.stack([r["wire"] for r in rows]).astype(np.float32)

    ckpts = {}
    for dd in args.ckpt_dirs:
        for cp in Path(dd).glob("checkpoint_*.pt"):
            tail = cp.stem.replace("checkpoint_", "")
            if not tail.isdigit():
                continue
            step = int(tail.lstrip("0") or "0")
            ckpts.setdefault(step, cp)
    steps = sorted(ckpts)
    print(f"[ckpts] {steps}  (n={len(steps)})")

    rng = np.random.default_rng(args.seed)

    def vforward(model, Xa):
        v = np.empty(len(Xa), dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, len(Xa), 512):
                xb = torch.from_numpy(Xa[b0:b0 + 512]).to(device)
                v[b0:b0 + len(xb)] = model(xb)[1].float().cpu().numpy().reshape(-1)
        return v

    def spearman(values, label):
        """rank-correlation of value vs CONCENTRATION (more-concentrated should -> higher value
        if value tracks concentration). label is the concentration scalar already oriented so
        that HIGHER = more concentrated."""
        a = values.argsort(kind="mergesort"); ra = np.empty(len(values)); ra[a] = np.arange(len(values))
        b = label.argsort(kind="mergesort"); rb = np.empty(len(label)); rb[b] = np.arange(len(label))
        ra = (ra - ra.mean()); rb = (rb - rb.mean())
        denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / denom) if denom > 0 else float("nan")

    conc_scalar = -ncomp   # PRIMARY concentration scalar oriented HIGHER = more concentrated

    out = {"encoding": name, "won_n": len(rows), "min_step": args.min_step,
           "split_primary": "mover_ncomp (concentrated = FEWER components)",
           "ncomp_median": float(med), "ncomp_tertiles": [float(t_lo), float(t_hi)],
           "verdict_rule": "LIT iff AUC_conc <= 0.60 AND flat/declining; OUT iff clearly > 0.60",
           "checkpoints": []}

    print(f"\n{'='*128}")
    print("§D-OVERSPREAD D1 — VALUE ranks CONCENTRATED won > SCATTERED won? (outcome held to terminal-WON)")
    print("AUC_conc = P(value(concentrated-won) > value(scattered-won)). PRIMARY split = mover_ncomp.")
    print(f"{'='*128}")
    hdr = (f"{'step':>7} | {'AUC_med(ncomp)':>15} [95%CI] | {'AUC_ter(ncomp)':>15} | "
           f"{'rho(v,conc)':>11} || {'AUC(frac)':>9} | {'AUC(supp)':>9} | {'AUC(tred)':>9} | {'v_mean':>7}")
    print(hdr); print("-" * len(hdr))

    for st in steps:
        model, _msp, _lab = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        v = vforward(model, X)

        auc_med = _auc(v[conc_med], v[scat_med])
        ci_med = _auc_boot(v[conc_med], v[scat_med], rng, args.nboot)
        auc_ter = _auc(v[conc_ter], v[scat_ter])
        rho = spearman(v, conc_scalar)

        auc_frac = _auc(v[conc_frac], v[scat_frac])
        auc_supp = _auc(v[conc_supp], v[scat_supp])
        auc_tred = _auc(v[conc_tred], v[scat_tred])
        vmean = float(v.mean())

        out["checkpoints"].append({
            "step": st,
            "auc_conc_ncomp_median": float(auc_med), "auc_conc_ncomp_median_ci": ci_med,
            "auc_conc_ncomp_tertile": float(auc_ter),
            "rho_value_concentration": rho,
            "auc_conc_largest_frac": float(auc_frac),
            "auc_conc_local_support": float(auc_supp),
            "auc_conc_turn_redundancy": float(auc_tred),
            "value_mean": vmean,
        })
        print(f"{st:>7} | {auc_med:>15.3f} [{ci_med[0]:.3f},{ci_med[1]:.3f}] | {auc_ter:>15.3f} | "
              f"{rho:>11.3f} || {auc_frac:>9.3f} | {auc_supp:>9.3f} | {auc_tred:>9.3f} | {vmean:>7.3f}")

    cks = out["checkpoints"]
    a, b = cks[0], cks[-1]
    med_vals = np.array([c["auc_conc_ncomp_median"] for c in cks])
    ter_vals = np.array([c["auc_conc_ncomp_tertile"] for c in cks])
    # arc trend = OLS slope of AUC_med vs step
    sv = np.array([c["step"] for c in cks], dtype=np.float64)
    slope = float(np.polyfit(sv, med_vals, 1)[0]) * 1000.0   # per 1k steps
    print(f"\n{'-'*128}")
    print(f"ARC {a['step']} -> {b['step']}:")
    print(f"  AUC_conc(ncomp,median): {a['auc_conc_ncomp_median']:.3f} -> {b['auc_conc_ncomp_median']:.3f}  "
          f"(mean over arc {med_vals.mean():.3f}, min {med_vals.min():.3f}, max {med_vals.max():.3f})")
    print(f"  AUC_conc(ncomp,tertile): mean over arc {ter_vals.mean():.3f}  "
          f"[{a['auc_conc_ncomp_tertile']:.3f} -> {b['auc_conc_ncomp_tertile']:.3f}]")
    print(f"  arc slope (AUC_med per 1k steps): {slope:+.5f}")
    out["arc"] = {
        "from": a["step"], "to": b["step"],
        "auc_med_from": a["auc_conc_ncomp_median"], "auc_med_to": b["auc_conc_ncomp_median"],
        "auc_med_mean": float(med_vals.mean()), "auc_med_min": float(med_vals.min()),
        "auc_med_max": float(med_vals.max()),
        "auc_ter_mean": float(ter_vals.mean()),
        "auc_med_slope_per_1k": slope,
    }
    # verdict helper (mechanical; final call in the note)
    mean_auc = float(med_vals.mean())
    lit = (mean_auc <= 0.60) and (slope <= 0.0)
    out["mechanical_flag"] = ("LIT" if lit else ("OUT" if mean_auc > 0.60 else "INCONCLUSIVE"))
    print(f"\n  MECHANICAL FLAG (note makes the final call): mean AUC_conc={mean_auc:.3f}, "
          f"slope={slope:+.5f} -> {out['mechanical_flag']}")
    print("  (LIT iff mean AUC_conc <= 0.60 AND flat/declining; OUT iff clearly > 0.60)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
