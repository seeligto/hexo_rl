#!/usr/bin/env python3
"""D-DECODE Workflow-2 STAGE C — off-window DEFENSE of the multi-window Gumbel-SH bot.

Runs the off-window exploit probe (offwindow_probe.play_game) with MultiWindowGumbelSHBot
as DEFENDER, EXPLOIT arm, on the SAME adversary seeds/axes as the deploy 0.335 run and the
kcluster 0.0 run (seed_base=7000, opening_plies=0, max_plies=150) -> paired-on-adversary.

Reports the ABSOLUTE off_window_forced_win_rate (robust to the exploit-probe arm-aliasing
bug, which contaminates only the exploit-control CONTRAST). 3-way reference:
  single-window deploy g=0 Gumbel-SH : 0.335   (reports/d_solver_offwindow/run1)
  multi-window no-drop PUCT kcluster : 0.0     (reports/d_solver_offwindow/ref_kcluster)
  multi-window no-drop Gumbel-SH (THIS)        <- the decisive number

INFERENCE-ONLY. Args: --n-games (default 200), --seed-base 7000, --out.
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
from hexo_rl.eval.offwindow_probe import play_game, summarize
from hexo_rl.utils.device import best_device
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--opening-plies", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=150)
    ap.add_argument("--arm", default="exploit")
    ap.add_argument("--out", default="reports/d_decode/stage_c_offwindow_defense")
    args = ap.parse_args()

    device = best_device()
    model, _spec, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))
    # deploy knobs from the checkpoint config (m=16, n=150, c_visit=50, c_scale=1, c_puct=1.5)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    from hexo_rl.eval.deploy_strength_eval import extract_deploy_knobs
    knobs = extract_deploy_knobs(ck.get("config", {}))
    bot = MultiWindowGumbelSHBot(
        model, device, n_sims=knobs["n_sims_full"], c_puct=float(knobs["c_puct"]),
        gumbel_m=knobs["gumbel_m"], c_visit=knobs["c_visit"], c_scale=knobs["c_scale"],
        gumbel_scale=0.0, kept_plane_indices=list(spec.kept_plane_indices), seed=args.seed_base,
    )
    print(f"[stageC] bot={bot.name()} knobs={knobs} enc={ENC} arm={args.arm} "
          f"n={args.n_games} seed_base={args.seed_base} opening={args.opening_plies} "
          f"device={device}", flush=True)

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
            adversary = OffWindowAdversaryBot(arm=args.arm, encoding=ENC, axis=axis, seed=seed)
            bot.reset()
            rec = play_game(bot, adversary, ENC, adv_side, spec,
                            knobs["n_sims_full"], args.max_plies, args.opening_plies, rng)
            rec.update({"arm": args.arm, "game": gi, "seed": seed, "axis": list(axis)})
            fp.write(json.dumps(rec) + "\n"); fp.flush()
            recs.append(rec)
            if (gi + 1) % 5 == 0 or (gi + 1) == args.n_games:
                el = time.time() - t0
                ow = sum(x["off_window_win"] for x in recs)
                aw = sum(x["adversary_won"] for x in recs)
                fr = sum(x["any_offwindow_forcing_position"] for x in recs)
                print(f"[{args.arm}] {gi+1}/{args.n_games} adv_wins={aw} off_window_wins={ow} "
                      f"forcing_pos={fr} {el:.0f}s {el/len(recs):.1f}s/game", flush=True)

    s = summarize(args.arm, recs)
    summary = {
        "arm": args.arm, "summary": s, "n_games": args.n_games,
        "seed_base": args.seed_base, "opening_plies": args.opening_plies,
        "checkpoint": CKPT, "encoding": ENC, "knobs": knobs,
        "bot": bot.name(),
        "reference_3way": {
            "single_window_deploy_gumbel_sh": 0.335,
            "multi_window_nodrop_puct_kcluster": 0.0,
            "multi_window_nodrop_gumbel_sh_THIS": s["off_window_forced_win_rate"],
        },
    }
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print("\n[stageC] SUMMARY:\n", json.dumps(summary, indent=2), flush=True)
    print(f"[stageC] off_window_forced_win_rate={s['off_window_forced_win_rate']} "
          f"strict={s['strict_off_window_forced_rate']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
