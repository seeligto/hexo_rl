"""One-shot runner: evaluate a single checkpoint and save result.json.

Usage:
  .venv/bin/python -m scripts.evalfair.run_retro_ckpt \\
    --ckpt checkpoints/run2_retro/checkpoint_00050000.pt \\
    --book-r4 tests/fixtures/opening_books/evalfair_r4_v2.json \\
    --book-r5 tests/fixtures/opening_books/evalfair_r5_v2.json \\
    --out reports/evalfair/retro_slope \\
    --workers 4

Resume-safe: if <out>/<ckpt_stem>/result.json exists, prints the cached result and exits 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

from scripts.evalfair.book import load_book
from scripts.evalfair.core import ArmSpec, extract_deploy_knobs, radius_from_checkpoint, run_arm
from scripts.evalfair.retro_slope import resolve_book_for_radius


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--book-r4", default=None, dest="book_r4")
    ap.add_argument("--book-r5", default=None, dest="book_r5")
    ap.add_argument("--out", default="reports/evalfair/retro_slope")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--expect-encoding", default="v6_live2_ls")
    args = ap.parse_args()

    books_by_radius: Dict[int, dict] = {}
    if args.book_r4:
        b = load_book(Path(args.book_r4))
        books_by_radius[int(b["radius_stage"])] = b
    if args.book_r5:
        b = load_book(Path(args.book_r5))
        books_by_radius[int(b["radius_stage"])] = b
    if not books_by_radius:
        ap.error("Provide --book-r4 and/or --book-r5")

    out = Path(args.out)
    ckpt_path = args.ckpt
    ckpt_out = out / Path(ckpt_path).stem

    result_path = ckpt_out / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        print(
            f"[run_retro_ckpt] CACHED step={existing['ckpt_step']} "
            f"wr={existing['wr']:.3f} CI=[{existing['pair_ci'][0]:.3f},{existing['pair_ci'][1]:.3f}] "
            f"radius={existing.get('radius')} eff_n={existing.get('eff_n')}"
        )
        return

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    resolved_radius = radius_from_checkpoint(ck)
    print(f"[run_retro_ckpt] {Path(ckpt_path).name} radius={resolved_radius}")

    book = resolve_book_for_radius(resolved_radius, books_by_radius, ckpt_path)
    ckpt_out.mkdir(parents=True, exist_ok=True)
    arm = ArmSpec(label="sims150")

    result = run_arm(
        ckpt_path, arm, book,
        out_dir=str(ckpt_out),
        workers=args.workers,
        n_boot=args.n_boot,
        book_seed=book.get("seed", 20260709),
        expect_encoding=args.expect_encoding,
    )
    print(
        f"[run_retro_ckpt] DONE step={result['ckpt_step']} wr={result['wr']:.3f} "
        f"CI=[{result['pair_ci'][0]:.3f},{result['pair_ci'][1]:.3f}] "
        f"radius={result.get('radius')} eff_n={result.get('eff_n')} "
        f"wall={result.get('wall_sec',0):.0f}s"
    )


if __name__ == "__main__":
    main()
