#!/usr/bin/env python3
"""RED-TEAM verification of §D-OVERSPREAD D1 (value-discrimination ceiling, concentration).

The committed `overspread_d1_value_conc.py` produces an OUT verdict whose own note concedes the
PRIMARY ncomp AUC rides stone-count (corr 0.75), so the OUT rests on TWO confound-free strands
claimed in PROSE ONLY (NOT in d1_value_conc.json, where auc_conc_turn_redundancy is NaN at every
checkpoint):
  (A) fork(count_winning_turns>=3) won > single(==1) won  AUC 0.613/0.692/0.695 RISING.
  (B) largest_blob_frac AUC 0.62-0.72.
This probe RE-DERIVES, read-only, on the SAME pool builder:
  1. corr(mover_ncomp, mover_stones) + AUC(fewer-stones-won) -> confirm the disclosed confound.
  2. fork(>=3) vs single(==1) AUC per checkpoint -> verify strand (A) actually exists + is >0.60.
  3. PURE concentration: within a narrow mid-stone band, AUC_conc(ncomp) -> is it >0.60 or ~chance?
  4. Is largest_blob_frac ALSO stone-confounded? corr(largest_frac, stones) + within-band AUC(frac).
If BOTH confound-free strands evaporate inside a matched-stone band, the OUT verdict is unsupported
by the artifact and D1 should be INCONCLUSIVE, not OUT.

EVAL-ONLY, read-only. Reuses the committed builder. New untracked script; tracked git-diff clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overspread_d1_value_conc import build_won_pool, _auc, _auc_boot  # noqa: E402
from hexo_rl.encoding import lookup, normalize_encoding_name  # noqa: E402
from hexo_rl.training.checkpoints import load_inference_model  # noqa: E402


def main():
    name = normalize_encoding_name("v6_live2")
    spec = lookup(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replays = sorted(str(p) for p in
                     Path("investigation/coherence_2026-06-08/replays").glob("games_2026-06-0*.jsonl"))
    rows = build_won_pool(replays, name, spec, 30000, 6000, 20260608)
    print(f"WON rows = {len(rows)}")

    ncomp = np.array([r["mover_ncomp"] for r in rows], dtype=np.float64)
    stones = np.array([r["mover_stones"] for r in rows], dtype=np.float64)
    lfrac = np.array([r["largest_frac"] for r in rows], dtype=np.float64)
    tred = np.array([r["turn_red"] for r in rows], dtype=np.float64)
    X = np.stack([r["wire"] for r in rows]).astype(np.float32)

    # ---- (1) confirm disclosed ncomp/stone confound ----
    c = np.corrcoef(ncomp, stones)[0, 1]
    print(f"\n[1] corr(mover_ncomp, mover_stones) = {c:.3f}  (note claims 0.750)")
    print(f"    turn_red distribution: {dict(zip(*np.unique(tred, return_counts=True)))}")
    n_fork = int((tred >= 3).sum()); n_single = int((tred == 1).sum())
    print(f"    fork(>=3) n={n_fork}  single(==1) n={n_single}  (==2) n={int((tred==2).sum())}")

    # narrow mid-stone band (central 50% of the stone distribution)
    lo, hi = np.percentile(stones, 25), np.percentile(stones, 75)
    band = (stones >= lo) & (stones <= hi)
    print(f"    mid-stone band [{lo:.0f},{hi:.0f}]  n={int(band.sum())}")

    ckpts = {}
    for dd in ["investigation/coherence_2026-06-08/checkpoints",
               "investigation/fragility_2026-06-07/checkpoints"]:
        for cp in Path(dd).glob("checkpoint_*.pt"):
            tail = cp.stem.replace("checkpoint_", "")
            if tail.isdigit():
                ckpts.setdefault(int(tail.lstrip("0") or "0"), cp)
    steps = sorted(ckpts)
    rng = np.random.default_rng(20260608)

    def vfwd(model):
        v = np.empty(len(X), dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, len(X), 512):
                xb = torch.from_numpy(X[b0:b0 + 512]).to(device)
                v[b0:b0 + len(xb)] = model(xb)[1].float().cpu().numpy().reshape(-1)
        return v

    # ncomp median split for the within-band check (same direction as committed: conc = fewer)
    med = np.median(ncomp)
    fr_med = np.median(lfrac)

    print(f"\n{'step':>7} | {'AUC_fork(>=3 vs ==1)':>20} [95%CI] | {'AUC_band_ncomp':>14} | "
          f"{'AUC_band_frac':>13} | {'corr_frac_stone':>15}")
    cfr = np.corrcoef(lfrac, stones)[0, 1]
    forkA, bandA, bandFr = [], [], []
    for st in steps:
        model, _s, _l = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        v = vfwd(model)
        # (2) fork-redundancy strand A: value(fork won) > value(single won)
        a_fork = _auc(v[tred >= 3], v[tred == 1])
        ci_fork = _auc_boot(v[tred >= 3], v[tred == 1], rng, 2000)
        # (3) pure concentration within stone band: conc = ncomp<med, scat = ncomp>med
        bm = band & (ncomp < med); bs = band & (ncomp > med)
        a_band = _auc(v[bm], v[bs])
        # (4) largest_frac within band: conc = frac>med, scat = frac<med
        bfm = band & (lfrac > fr_med); bfs = band & (lfrac < fr_med)
        a_bf = _auc(v[bfm], v[bfs])
        forkA.append(a_fork); bandA.append(a_band); bandFr.append(a_bf)
        print(f"{st:>7} | {a_fork:>20.3f} [{ci_fork[0]:.3f},{ci_fork[1]:.3f}] | {a_band:>14.3f} | "
              f"{a_bf:>13.3f} | {cfr:>15.3f}")

    forkA = np.array(forkA); bandA = np.array(bandA); bandFr = np.array(bandFr)
    print(f"\n[A] fork(>=3 vs ==1) AUC: mean={forkA.mean():.3f} min={forkA.min():.3f} "
          f"max={forkA.max():.3f}  30k={forkA[0]:.3f} 90k={forkA[-1]:.3f} "
          f"slope/1k={np.polyfit(steps,forkA,1)[0]*1000:+.5f}")
    print(f"[band] pure-conc ncomp AUC (matched stones): mean={bandA.mean():.3f} "
          f"min={bandA.min():.3f} max={bandA.max():.3f}")
    print(f"[band] largest_frac AUC (matched stones): mean={bandFr.mean():.3f} "
          f"min={bandFr.min():.3f} max={bandFr.max():.3f}  corr(frac,stones)={cfr:.3f}")
    print(f"\nVERDICT GUARD: OUT needs >=1 confound-free strand clearly >0.60.")
    print(f"  fork-redundancy clears: {forkA.min()>0.60} (min {forkA.min():.3f})")
    print(f"  largest_frac in-band clears: {bandFr.mean()>0.60} (mean {bandFr.mean():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
