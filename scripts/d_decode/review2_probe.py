#!/usr/bin/env python3
"""D-DECODE Reviewer-2 independent off-window defense probe.

Reproduces D2's stage_c defense cell with a FRESH seed base (9000) and
n_games=60. Uses MultiWindowGumbelSHBot (the D2 bot) loaded from its
canonical path. Writes artifacts to reports/d_decode/review2_*.

Run:
  .venv/bin/python scripts/d_decode/review2_probe.py \
      --checkpoint checkpoints/checkpoint_00272357.pt \
      --encoding v6_live2_ls \
      --n-games 60 --seed-base 9000 \
      --out reports/d_decode/review2_offwindow.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Import the D2 bot from its canonical path
sys.path.insert(0, str(REPO / "scripts" / "d_decode"))
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot  # noqa: E402

from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot  # noqa: E402
from hexo_rl.bots.offwindow_geom import HEX_AXES  # noqa: E402
from hexo_rl.diagnostics.forced_win_detector import cheb, is_off_window, window_center  # noqa: E402
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name as _norm  # noqa: E402
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
from hexo_rl.eval.offwindow_probe import play_game, summarize  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402


CHECKPOINT = "checkpoints/checkpoint_00272357.pt"
ENCODING = "v6_live2_ls"
GUMBEL_M = 16
N_SIMS = 150
C_PUCT = 1.5
C_VISIT = 50.0
C_SCALE = 1.0
GUMBEL_SCALE = 0.0  # g=0 deterministic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--encoding", default=ENCODING)
    ap.add_argument("--n-games", type=int, default=60)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--out", default="reports/d_decode/review2_offwindow.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = best_device()
    ckpt_path = REPO / args.checkpoint
    print(f"[review2] Loading {ckpt_path} on {device}", flush=True)

    model, _spec, label = load_model_with_encoding(ckpt_path, device)
    # Override encoding to v6_live2_ls (state-dict identical to v6_live2;
    # detection returns v6_live2 architecture family — must override for multi-window).
    label = args.encoding
    try:
        model.encoding = label
    except Exception:
        pass
    spec = _lookup_encoding(_norm(label))

    # Build the MultiWindowGumbelSHBot (D2 bot)
    bot = MultiWindowGumbelSHBot(
        model, device,
        n_sims=N_SIMS, c_puct=C_PUCT,
        gumbel_m=GUMBEL_M, c_visit=C_VISIT, c_scale=C_SCALE,
        gumbel_scale=GUMBEL_SCALE, seed=args.seed_base,
        kept_plane_indices=list(spec.kept_plane_indices),
    )
    print(f"[review2] bot={bot.name()} enc={label} n_games={args.n_games} "
          f"seed_base={args.seed_base} gumbel_scale={GUMBEL_SCALE}", flush=True)

    recs = []
    t0 = time.time()
    with out.open("w") as fp:
        for gi in range(args.n_games):
            seed = args.seed_base + gi
            rng = random.Random(seed)
            np.random.seed(seed)
            random.seed(seed)
            adv_side = 1 if gi % 2 == 0 else -1
            axis = HEX_AXES[gi % len(HEX_AXES)]
            adversary = OffWindowAdversaryBot(arm="exploit", encoding=label, axis=axis, seed=seed)
            rec = play_game(bot, adversary, label, adv_side, spec,
                            N_SIMS, 150, 0, rng)
            rec.update({"arm": "exploit", "game": gi, "seed": seed, "axis": list(axis)})
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
            recs.append(rec)
            if (gi + 1) % 10 == 0 or (gi + 1) == args.n_games:
                el = time.time() - t0
                ow = sum(x["off_window_win"] for x in recs)
                aw = sum(x["adversary_won"] for x in recs)
                print(f"[review2] {gi+1}/{args.n_games}  adv_wins={aw} "
                      f"off_window_wins={ow}  {el:.0f}s  {el/max(1,len(recs)):.1f}s/game",
                      flush=True)

    s = summarize("exploit", recs)
    elapsed = time.time() - t0

    result = {
        "reviewer": "review2_independent",
        "bot": bot.name(),
        "gumbel_scale": GUMBEL_SCALE,
        "gumbel_m": GUMBEL_M,
        "n_sims": N_SIMS,
        "summary": s,
        "n_games": args.n_games,
        "seed_base": args.seed_base,
        "checkpoint": str(ckpt_path),
        "encoding": label,
        "elapsed_s": round(elapsed, 1),
        "knobs": {"gumbel_m": GUMBEL_M, "c_visit": C_VISIT, "c_scale": C_SCALE,
                  "n_sims_full": N_SIMS, "c_puct": C_PUCT},
        "d2_reference_off_window_forced_rate": 0.0,
        "d2_n_games": 200,
        "d2_seed_base": 7000,
    }
    summary_path = out.with_suffix(".json").with_stem(out.stem + "_summary")
    summary_path.write_text(json.dumps(result, indent=2))
    print(f"[review2] DONE  off_window_forced_rate={s['off_window_forced_win_rate']}  "
          f"adversary_win_rate={s['adversary_win_rate']}  "
          f"elapsed={elapsed:.0f}s  wrote {out} + {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
