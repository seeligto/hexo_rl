#!/usr/bin/env python3
"""D-DECODE Workflow-2 RED-TEAM — COMPUTE CONFOUND check.

RED-TEAM QUESTION: Does multi-window defend because of the multi-window ACTION
SPACE, or just because it gets ~K× more NN evaluations than single-window
deploy (K = average clusters per leaf, v6_live2_ls typical K≈2-3)?

PROTOCOL:
  1. Run single-window DeployHeadBot at MATCHED NN-eval budget:
     n_sims_matched = round(K_avg * 150) where K_avg measured from 10 random
     games of the same length (avg K=2.38 → 357, rounded to 450 to ensure
     we OVER-budget the single-window arm).
  2. Same seeds/axes as stage C (seed_base=7000, same 6-config cycle).
  3. Measure off_window_forced_win_rate.

PREDICTION:
  If action-space is the cause → single-window@matched_sims STILL gets forced
  at ~0.335 (because off-window cells have no logit regardless of sim budget).
  If compute was the confound → single-window@matched_sims defends near 0.0.

N=60 games (10× each 6-config cycle) is enough to discriminate, since the
distinct-game n is 6 configs. A rate of 2/6 = 0.333 vs 0/6 = 0 is the signal.
INFERENCE-ONLY. Writes artifacts to reports/d_decode/redteam2_*.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "d_decode"))

import numpy as np
import torch

from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.bots.offwindow_geom import HEX_AXES
from hexo_rl.encoding import lookup as _lookup, normalize_encoding_name as _norm
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot, _build_engine_for_model, extract_deploy_knobs,
)
from hexo_rl.eval.offwindow_probe import play_game, summarize
from hexo_rl.utils.device import best_device

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"

# K_avg measured from 10 random games of ~35 plies with v6_live2_ls: avg=2.38.
# Matched budget = ceil(2.38 * 150) = 357, rounded UP to 450 to be conservative
# (i.e., we give single-window MORE than the multi-window budget — if it still
# fails, action-space is clearly the cause, not compute).
K_AVG_MEASURED = 2.38
N_SIMS_BASE = 150
N_SIMS_MATCHED = 450  # > K_avg * N_SIMS_BASE; conservative over-budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=60,
                    help="games (default 60 = 10 cycles of 6 distinct configs)")
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--n-sims-matched", type=int, default=N_SIMS_MATCHED)
    ap.add_argument("--out", default="reports/d_decode/redteam2_compute_confound")
    args = ap.parse_args()

    device = best_device()
    model, _spec, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))

    # Override n_sims to matched budget; keep all other knobs identical.
    knobs_matched = dict(knobs)
    knobs_matched["n_sims_full"] = args.n_sims_matched

    engine = _build_engine_for_model(model, ENC, device)
    bot = DeployHeadBot(engine, knobs_matched, label="sw_matched", seed=args.seed_base)

    print(f"[redteam2_compute] COMPUTE CONFOUND CHECK", flush=True)
    print(f"  K_avg_measured={K_AVG_MEASURED}  base_sims={N_SIMS_BASE}", flush=True)
    print(f"  matched_sims={args.n_sims_matched}  (>{K_AVG_MEASURED}*{N_SIMS_BASE}={K_AVG_MEASURED*N_SIMS_BASE:.0f})", flush=True)
    print(f"  ckpt={CKPT}  enc={ENC}  n={args.n_games}  seed_base={args.seed_base}", flush=True)
    print(f"  device={device}", flush=True)
    print(f"  PREDICTION: if action-space is cause -> forced_rate ~0.335 (same as baseline).", flush=True)
    print(f"              if compute is confound  -> forced_rate ~0.0.", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    t0 = time.time()
    with (out.with_suffix(".jsonl")).open("w") as fp:
        for gi in range(args.n_games):
            seed = args.seed_base + gi
            rng = random.Random(seed)
            np.random.seed(seed)
            random.seed(seed)
            adv_side = 1 if gi % 2 == 0 else -1
            axis = HEX_AXES[gi % len(HEX_AXES)]
            adversary = OffWindowAdversaryBot(
                arm="exploit", encoding=ENC, axis=axis, seed=seed
            )
            bot.reset()
            rec = play_game(
                bot, adversary, ENC, adv_side, spec,
                args.n_sims_matched, 150, 0, rng
            )
            rec.update({
                "arm": "exploit", "game": gi, "seed": seed,
                "axis": list(axis), "n_sims": args.n_sims_matched,
            })
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
            recs.append(rec)
            if (gi + 1) % 6 == 0 or (gi + 1) == args.n_games:
                el = time.time() - t0
                ow = sum(x["off_window_win"] for x in recs)
                aw = sum(x["adversary_won"] for x in recs)
                fp_rate = round(ow / len(recs), 3)
                print(
                    f"[redteam2_compute] {gi+1}/{args.n_games} "
                    f"adv_wins={aw} off_window_wins={ow} forced_rate={fp_rate} "
                    f"{el:.0f}s {el/len(recs):.1f}s/game",
                    flush=True,
                )

    s = summarize("exploit", recs)

    # Deduplicate by game config (same D-ARGMAX analysis as reference)
    import collections
    combo_results = collections.defaultdict(list)
    for r in recs:
        key = (tuple(r["axis"]), r["adv_side"])
        combo_results[key].append(r["off_window_win"])
    distinct_forced = sum(
        1 for k, wins in combo_results.items() if any(wins)
    )
    distinct_total = len(combo_results)

    summary = {
        "confound_check": "compute",
        "n_sims_base": N_SIMS_BASE,
        "k_avg_measured": K_AVG_MEASURED,
        "n_sims_matched": args.n_sims_matched,
        "matched_vs_base_ratio": round(args.n_sims_matched / N_SIMS_BASE, 2),
        "n_games": args.n_games,
        "seed_base": args.seed_base,
        "checkpoint": CKPT,
        "encoding": ENC,
        "off_window_forced_rate_raw": s["off_window_forced_win_rate"],
        "distinct_game_configs": distinct_total,
        "distinct_configs_forced": distinct_forced,
        "distinct_forced_rate": round(distinct_forced / distinct_total, 3) if distinct_total else None,
        "reference_baseline_0.335_n_sims": N_SIMS_BASE,
        "multiwindow_0.0_n_sims": N_SIMS_BASE,
        "verdict": (
            "ACTION_SPACE_CAUSE"
            if s["off_window_forced_win_rate"] > 0.2
            else ("COMPUTE_CONFOUND" if s["off_window_forced_win_rate"] < 0.05 else "AMBIGUOUS")
        ),
        "knobs": knobs_matched,
    }
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print("\n[redteam2_compute] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    elapsed = time.time() - t0
    print(
        f"[redteam2_compute] forced_rate={s['off_window_forced_win_rate']:.3f} "
        f"distinct_forced={distinct_forced}/{distinct_total} "
        f"verdict={summary['verdict']} ({elapsed:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
