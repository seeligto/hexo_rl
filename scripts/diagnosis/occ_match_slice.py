#!/usr/bin/env python3
"""§D-VALPROBE — occupancy-stratified fixture slice matched to a reference bank.

Fixture-fix step under the pre-registered FIXTURE-VALID gate: re-slices a large
fresh bank so its occupancy marginal matches the live reference (conditioning
variable only — outcomes/predictions untouched). Strata above the fresh bank's
support ceiling are refilled from the remaining rows and REPORTED — a large
shortfall means the generation path cannot reach the reference regime and the
matched slice is PARTIAL, not a pass (red-team rt1, 2026-06-11: matching
occupancy can degrade the k-axis overlap; always re-run fixture_valid_check on
the output and report both axes).

Run (vast):
  .venv/bin/python scripts/diagnosis/occ_match_slice.py \
    --bank data/selfplay_bank12k_v6_live2_ls_50k.npz \
    --reference data/livetail_bank_e928c854.npz \
    --encoding v6_live2_ls --n 4000 \
    --out data/selfplay_fixture_v6_live2_ls_50k_occmatched.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.encoding import cur_stone_slot, lookup, opp_stone_slot  # noqa: E402

KEYS = ("states", "outcomes", "k_counts", "plies", "game_ids")


def occupancy(states: np.ndarray, cur: int, opp: int) -> np.ndarray:
    return (states[:, cur].astype(np.float32)
            + states[:, opp].astype(np.float32) > 0.5).sum(axis=(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser(description="occupancy-matched fixture slice")
    ap.add_argument("--bank", required=True, help="large fresh bank npz")
    ap.add_argument("--reference", required=True, help="reference bank npz (live regime)")
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--strata", type=int, default=10, help="occupancy quantile strata")
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = lookup(args.encoding)
    cur, opp = cur_stone_slot(spec), opp_stone_slot(spec)
    ref = np.load(args.reference)
    bank = np.load(args.bank)
    occ_ref = occupancy(ref["states"], cur, opp)
    occ_bank = occupancy(bank["states"], cur, opp)

    edges = np.quantile(occ_ref, np.linspace(0, 1, args.strata + 1))
    edges[0], edges[-1] = -1.0, max(edges[-1], occ_bank.max()) + 1.0
    target_per = args.n // args.strata
    rng = np.random.default_rng(args.seed)
    chosen, shortfall, strata_log = [], 0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = np.where((occ_bank > lo) & (occ_bank <= hi))[0]
        take = min(target_per, len(pool))
        strata_log.append({"range": [float(lo), float(hi)],
                           "support": int(len(pool)), "taken": int(take)})
        shortfall += target_per - take
        if take:
            chosen.append(rng.choice(pool, size=take, replace=False))
    chosen = np.concatenate(chosen)
    if shortfall:
        rest = np.setdiff1d(np.arange(len(occ_bank)), chosen)
        chosen = np.concatenate(
            [chosen, rng.choice(rest, size=min(shortfall, len(rest)), replace=False)])
    chosen.sort()

    out = pathlib.Path(args.out)
    np.savez_compressed(out, **{k: bank[k][chosen] for k in KEYS})
    sidecar = {
        "slicer": "scripts/diagnosis/occ_match_slice.py",
        "bank": args.bank, "reference": args.reference,
        "seed": args.seed, "n": int(len(chosen)),
        "shortfall_refilled": int(shortfall),
        "strata": strata_log,
        "occ_mean_matched": float(occupancy(bank["states"][chosen], cur, opp).mean()),
        "occ_mean_reference": float(occ_ref.mean()),
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"[occ-match] n={len(chosen)} shortfall_refilled={shortfall} "
          f"occ_mean={sidecar['occ_mean_matched']:.1f} "
          f"(reference {sidecar['occ_mean_reference']:.1f}) -> {out}")
    if shortfall:
        print(f"[occ-match] PARTIAL: {shortfall}/{args.n} rows refilled outside "
              "their target strata — generation support ceiling; re-run "
              "fixture_valid_check on the output and report both axes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
