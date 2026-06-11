#!/usr/bin/env python3
"""§D-WALLCAUSATION Phase A follow-up — the DECISIVE causation instrument.

Regenerate self-play from a frozen checkpoint via the **Rust worker_loop** path
(`WorkerPool` — the same path training uses), which searches OFF-WINDOW cells via
the uniform fallback prior (`backup.rs:112`) and can therefore spread to the
training-self-play regime (median 17 / max 306, §OFFWINDOW §2). This is the path
the ModelPlayer regeneration (`wallcausation_selfplay_gen.py`) structurally CANNOT
reach — ModelPlayer drops off-window at selection (`evaluator.py:113`) so its boards
are capped at the 19×19 window (max_spread≤18) and the off-window wall stays dormant.

Records EVERY game (game_replay.sample_rate=1) to a per-checkpoint jsonl and tags it
with the checkpoint step (the §B recorder fix), then runs the shared forced-win
detector over BOTH mover sides — so off-window incidence + board spread on the REAL
wall regime can be correlated against the archived colony signal (the question
ModelPlayer left INCONCLUSIVE).

Run (per checkpoint):
  .venv/bin/python scripts/diagnosis/wallcausation_rust_regen.py \
     --checkpoint archive/s180b_3knob_fail/ckpts/ckpt_step00050000.pt --step 50000 \
     --n-games 120 --n-workers 4 --variant vast \
     --out reports/investigations/wallcausation_data/rust_s180b_step00050000.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import ReplayBuffer  # noqa: E402
from hexo_rl.diagnostics.forced_win_detector import (  # noqa: E402
    analyze_recorded_game, bbox_span, engine_player_sides,
)
from hexo_rl.encoding import resolve_encoding_for_eval  # noqa: E402
from hexo_rl.model.network import HexTacToeNet  # noqa: E402
from hexo_rl.selfplay.pool import WorkerPool  # noqa: E402
from hexo_rl.utils.config import load_config  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n-games", type=int, default=120)
    ap.add_argument("--n-workers", type=int, default=4, help="game runners (laptop-safe default)")
    ap.add_argument("--variant", default="vast", help="config variant for worker/batch/sims tuning")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wait-sec", type=int, default=1200)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec_dir = out.parent / f"_rustrec_{args.step}"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for stale in rec_dir.glob("games_*.jsonl"):
        stale.unlink()

    cfg = load_config("configs/model.yaml", "configs/training.yaml",
                      "configs/selfplay.yaml", f"configs/variants/{args.variant}.yaml")
    spec = resolve_encoding_for_eval(args.checkpoint, None)
    cfg["encoding"] = spec.name
    for stale_key in ("board_size", "in_channels", "n_planes", "cluster_window_size",
                      "cluster_threshold", "legal_move_radius"):
        cfg.pop(stale_key, None)
    # Record EVERY game with moves, into a clean per-checkpoint dir (the §B fix lets
    # us also carry the step tag via pool.update_checkpoint_step below).
    cfg["game_replay"] = {"output_dir": str(rec_dir), "sample_rate": 1, "enabled": True}

    device = best_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mcfg = cfg.get("model", {})
    model = HexTacToeNet(
        board_size=spec.trunk_size, in_channels=spec.n_planes,
        filters=int(mcfg.get("filters", 128)), res_blocks=int(mcfg.get("res_blocks", 12)),
        encoding=spec.name,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if state is None:
        state = ckpt
    try:
        model.load_state_dict(state, strict=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] partial load: {exc}", flush=True)
    model.eval()

    buffer = ReplayBuffer(capacity=200_000, encoding=spec.name)
    pool = WorkerPool(model, cfg, device, buffer, n_workers=args.n_workers)
    pool.update_checkpoint_step(args.step)   # §B fix — tag records with this step
    print(f"[rust-regen] step={args.step} ckpt={args.checkpoint} enc={spec.name} "
          f"device={device} n_workers={args.n_workers} target={args.n_games}", flush=True)
    pool.start()
    t0 = time.perf_counter()
    try:
        while pool.games_completed < args.n_games:
            time.sleep(2.0)
            if time.perf_counter() - t0 > args.max_wait_sec:
                print("[rust-regen] max-wait hit; stopping early", flush=True)
                break
            g = pool.games_completed
            if g and g % 20 == 0:
                print(f"  {g}/{args.n_games} games", flush=True)
    finally:
        pool.stop()
    elapsed = time.perf_counter() - t0

    # gather recorded games (sample_rate=1 → every game)
    recs = []
    for f in sorted(rec_dir.glob("games_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs = [r for r in recs if r.get("moves")]
    sides = engine_player_sides(spec.name)
    spreads = [bbox_span([(m[0], m[1]) for m in r["moves"]]) for r in recs]
    draws = sum(1 for r in recs if r.get("outcome") == "draw")
    units = fw_units = ow_units = wallcost = 0
    sf = so = sc = 0
    steps_seen = set()
    with out.open("w") as fp:
        for r in recs:
            steps_seen.add(r.get("checkpoint_step"))
            mv = [(m[0], m[1]) for m in r["moves"]]
            winner = {"x_win": 1, "o_win": -1, "draw": None}.get(r.get("outcome"))
            fp.write(json.dumps({"step": args.step, "outcome": r.get("outcome"),
                                 "checkpoint_step": r.get("checkpoint_step"),
                                 "n_ply": len(mv), "spread": bbox_span(mv), "moves": r["moves"]}) + "\n")
            for side in sides:
                units += 1
                s = analyze_recorded_game(mv, r.get("outcome", ""), encoding=spec.name, mover_side=side)
                if s.forced_win_turns > 0:
                    fw_units += 1; sf += s.forced_win_turns; so += s.off_window_forced_turns; sc += s.converted
                    if s.off_window_forced_turns > 0:
                        ow_units += 1
                        if winner != side:
                            wallcost += 1

    summary = {
        "step": args.step, "encoding": spec.name, "n_games": len(recs),
        "elapsed_sec": round(elapsed, 1), "checkpoint_step_tags": sorted(x for x in steps_seen if x is not None),
        "draw_rate": round(draws / len(recs), 4) if recs else None,
        "median_spread": statistics.median(spreads) if spreads else None,
        "max_spread": max(spreads) if spreads else None,
        "p90_spread": int(np.percentile(spreads, 90)) if spreads else None,
        "off_window_incidence": round(ow_units / units, 4) if units else None,
        "off_window_rate": round(so / sf, 4) if sf else None,
        "forced_incidence": round(fw_units / units, 4) if units else None,
        "non_conversion": round(1 - sc / sf, 4) if sf else None,
        "wall_cost_incidence": round(wallcost / units, 4) if units else None,
    }
    print("[rust-regen] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    json.dump(summary, open(str(out) + ".summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
