#!/usr/bin/env python3
"""D-DECODE Workflow-2 STAGE D — WR of the multi-window Gumbel-SH deploy head ALONE
vs SealBot-d5 (NO A1 SolverBackupBot — do not double-count the offense lever).

Mirrors scripts/eval/run_a1_solver_backup.py's BASELINE arm (DeployHeadBot ALONE vs
fixed-depth SealBot, color-balanced, fresh SealBot per game, RNG-seeded openings) but
swaps the single-window DeployHeadBot for MultiWindowGumbelSHBot. Same opening_plies=4,
sealbot_depth=5, seed_base=20260627 -> directly comparable to the A1 baseline 0.47.

Reports WR = (wins + 0.5*draws)/n for the multi-window bot. INFERENCE-ONLY.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "d_decode"))

import numpy as np
import torch

from hexo_rl.bots.sealbot_bot import SealBotBot
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import (
    _build_engine_for_model, _play_one_game, extract_deploy_knobs,
)
from hexo_rl.encoding import lookup as _lookup, normalize_encoding_name as _norm
from hexo_rl.utils.device import best_device
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"


def _wr(games, label):
    wins = draws = 0
    for g in games:
        if g["winner"] == "draw":
            draws += 1
        elif (g["winner"] == "p1" and g["p1"] == label) or (g["winner"] == "p2" and g["p2"] == label):
            wins += 1
    n = len(games)
    return (wins + 0.5 * draws) / n if n else 0.0, wins, draws, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=100)
    ap.add_argument("--sealbot-depth", type=int, default=5)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=20260627)
    ap.add_argument("--out", default="reports/d_decode/stage_d_wr_vs_sealbot")
    args = ap.parse_args()

    device = best_device()
    model, _spec, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    engine = _build_engine_for_model(model, ENC, device)  # parity w/ deploy harness build

    bot = MultiWindowGumbelSHBot(
        model, device, n_sims=knobs["n_sims_full"], c_puct=float(knobs["c_puct"]),
        gumbel_m=knobs["gumbel_m"], c_visit=knobs["c_visit"], c_scale=knobs["c_scale"],
        gumbel_scale=0.0, kept_plane_indices=list(spec.kept_plane_indices), seed=args.seed_base,
    )
    cand_label = "mw_gumbel_sh"
    print(f"[stageD] bot={bot.name()} vs SealBot-d{args.sealbot_depth} n={args.n_games} "
          f"opening={args.opening_plies} seed_base={args.seed_base} knobs={knobs} "
          f"device={device}  (baseline single-window deploy WR=0.47)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    games = []
    t0 = time.time()
    with (out.with_suffix(".jsonl")).open("w") as fp:
        for gi in range(args.n_games):
            if hasattr(bot, "reset"):
                bot.reset()
            opp = SealBotBot(time_limit=60.0, max_depth=args.sealbot_depth)  # fixed-depth, cold TT
            seed = args.seed_base + gi
            random.seed(seed)
            np.random.seed(seed % (2**31))
            if gi % 2 == 0:
                p1, p2, l1, l2 = bot, opp, cand_label, "sealbot"
            else:
                p1, p2, l1, l2 = opp, bot, "sealbot", cand_label
            g = _play_one_game(p1, p2, l1, l2, ENC, opening_plies=args.opening_plies, seed=seed)
            g["arm"] = cand_label
            g["seed"] = seed
            fp.write(json.dumps(g) + "\n"); fp.flush()
            games.append(g)
            if (gi + 1) % 5 == 0 or (gi + 1) == args.n_games:
                wr, w, d, n = _wr(games, cand_label)
                el = time.time() - t0
                print(f"[stageD] {gi+1}/{args.n_games} wr={wr:.3f} (w{w}/d{d}/n{n}) "
                      f"{el:.0f}s {el/len(games):.1f}s/game", flush=True)

    wr, w, d, n = _wr(games, cand_label)
    summary = {
        "bot": bot.name(), "wr": round(wr, 4), "wins": w, "draws": d, "n_games": n,
        "sealbot_depth": args.sealbot_depth, "opening_plies": args.opening_plies,
        "seed_base": args.seed_base, "checkpoint": CKPT, "encoding": ENC, "knobs": knobs,
        "baseline_single_window_deploy_wr": 0.47,
        "holds_wr": bool(wr >= 0.47 - 1e-9),
    }
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print("\n[stageD] SUMMARY:\n", json.dumps(summary, indent=2), flush=True)
    print(f"[stageD] WR={wr:.4f} vs baseline 0.47 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
