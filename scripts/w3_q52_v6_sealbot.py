#!/usr/bin/env python3
"""W3 Q52 — v6 bootstrap vs SealBot anchor (150 games, 128 sims, 0.5s think).

Matches §114 protocol exactly. Writes report to reports/audit_2026-04-30/.

Usage:
  .venv/bin/python scripts/w3_q52_v6_sealbot.py
  .venv/bin/python scripts/w3_q52_v6_sealbot.py --dry-run  # 4 games
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

V6_CKPT = REPO_ROOT / "checkpoints" / "bootstrap_model.pt"
OUT_DIR = REPO_ROOT / "reports" / "audit_2026-04-30"

N_GAMES_DEFAULT = 150
SIMS_DEFAULT = 128
TIME_LIMIT_DEFAULT = 0.5

V4_WR = 0.187      # §114 anchor (28/150)
V4_COLONY_FRAC = 0.82  # §114 colony-win fraction


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=N_GAMES_DEFAULT)
    p.add_argument("--sims", type=int, default=SIMS_DEFAULT)
    p.add_argument("--time-limit", type=float, default=TIME_LIMIT_DEFAULT)
    p.add_argument("--dry-run", action="store_true", help="4 games only, verify setup")
    args = p.parse_args()

    n_games = 4 if args.dry_run else args.n_games
    n_sims = args.sims
    time_limit = args.time_limit

    from hexo_rl.eval.evaluator import Evaluator
    from hexo_rl.utils.config import load_config
    from hexo_rl.utils.device import best_device
    from hexo_rl.training.trainer import Trainer

    device = best_device()
    print(f"Device: {device}")
    print(f"Loading v6 from {V6_CKPT}...")

    cfg = load_config(
        str(REPO_ROOT / "configs" / "model.yaml"),
        str(REPO_ROOT / "configs" / "training.yaml"),
        str(REPO_ROOT / "configs" / "selfplay.yaml"),
    )
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["sealbot_model_sims"] = n_sims
    cfg["evaluation"]["eval_temperature"] = 0.5

    trainer = Trainer.load_checkpoint(
        V6_CKPT,
        checkpoint_dir=str(REPO_ROOT / "checkpoints"),
        device=device,
        fallback_config=cfg,
    )
    evaluator = Evaluator(trainer.model, device, cfg)

    print(f"Running {n_games} games vs SealBot (sims={n_sims}, time_limit={time_limit}s)...")
    t0 = time.time()
    result = evaluator.evaluate_vs_sealbot(
        n_games=n_games,
        time_limit=time_limit,
        model_sims=n_sims,
    )
    elapsed = time.time() - t0

    if args.dry_run:
        print(f"\nDRY RUN COMPLETE — wins={result.win_count}/{n_games}  elapsed={elapsed:.0f}s")
        return

    point, lower, upper = wilson_ci(result.win_count, n_games)
    colony_frac = result.colony_wins / result.win_count if result.win_count > 0 else 0.0

    if lower >= 0.14:
        verdict = "PASS"
    elif lower >= 0.10:
        verdict = "WARN"
    else:
        verdict = "BLOCK"

    colony_flag = colony_frac < 0.70 and result.win_count > 0

    print(f"\n=== Q52 RESULT ===")
    print(f"v6 wins: {result.win_count}/{n_games}  ({point:.1%})")
    print(f"Wilson 95% CI: [{lower:.1%}, {upper:.1%}]")
    print(f"Colony wins: {result.colony_wins}/{result.win_count}  ({colony_frac:.1%})")
    print(f"v4 anchor: WR=18.7%, colony={V4_COLONY_FRAC:.0%}")
    if colony_flag:
        print(f"  WARNING: colony fraction {colony_frac:.1%} < 70% (v4 was 82%)")
    print(f"VERDICT: {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "W3_q52_v6_sealbot.md"
    md_content = f"""# W3 Q52 — v6 Bootstrap vs SealBot Anchor

**Date:** 2026-04-30
**Protocol:** {n_games} games, {n_sims} sims, {time_limit}s think (matches §114 protocol)
**v4 anchor:** 18.7% WR (28/150), 82% colony wins (§114)

## Result

| Metric | v6 | v4 anchor |
|---|---|---|
| Wins | {result.win_count} / {n_games} | 28 / 150 |
| Win rate (point) | {point:.1%} | 18.7% |
| Wilson 95% CI lower | {lower:.1%} | — |
| Wilson 95% CI upper | {upper:.1%} | — |
| Colony wins | {result.colony_wins} / {result.win_count} | 23 / 28 |
| Colony-win fraction | {colony_frac:.1%} | 82% |
| Elapsed | {elapsed:.0f}s | — |

{"⚠️ **Colony-win fraction below 70%** — secondary regression flag even if WR holds." if colony_flag else "Colony-win fraction within expected range (≥70% of v4 anchor)."}

## Gate logic

- PASS: lower-CI ≥ 14% (within noise of v4 anchor)
- WARN: lower-CI in [10%, 14%)
- BLOCK: lower-CI < 10%

## Verdict: {verdict}

{"PASS: v6 SealBot WR within noise of v4 anchor. Phase 4.0 sustained run UNBLOCKED on Q52." if verdict == "PASS" else
 "WARN: v6 SealBot lower-CI in [10%, 14%). Proceed with early kill-switch: abort sustained run if SealBot WR does not trend above 25% by step 5000." if verdict == "WARN" else
 "BLOCK: v6 SealBot lower-CI < 10%. Significant regression. Do not launch Phase 4.0 sustained run."}
"""
    md_path.write_text(md_content)
    print(f"Report: {md_path}")

    if verdict == "BLOCK":
        print("\nBLOCK — Q52 failed. Do not launch Phase 4.0 sustained run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
