"""evalfair/sims_ladder.py — WP3 driver: sims arms + solver arm on one ckpt.

Builds the driver now; the full WP3 run is a later dispatch.
Arms: n_sims in {75, 150, 300, 600} + solver-backup arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evalfair.book import load_book, FIXTURE_DIR
from scripts.evalfair.core import ArmSpec, run_arm

SIMS_ARMS = [75, 150, 300, 600]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="WP3 sims ladder: run sims arms + solver arm on one ckpt."
    )
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--book", default=None, help="Path to book_v2 JSON")
    ap.add_argument("--out", default="reports/sims_ladder", help="Output directory")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--expect-encoding", default="v6_live2_ls", dest="expect_encoding")
    ap.add_argument("--smoke", action="store_true", help="4 pairs per arm (quick smoke)")
    ap.add_argument(
        "--arms", nargs="+", type=int, default=SIMS_ARMS,
        help=f"Sims values to run (default: {SIMS_ARMS})"
    )
    ap.add_argument("--no-solver", action="store_true", dest="no_solver",
                    help="Skip the solver-backup arm")
    args = ap.parse_args()

    book_path = args.book or str(FIXTURE_DIR / "evalfair_r4_v2.json")
    book = load_book(Path(book_path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_pairs = 4 if args.smoke else None
    book_seed = book.get("seed", 20260709)

    summary = {"ckpt": args.ckpt, "book_id": book.get("book_id"), "arms": []}

    for n_sims in args.arms:
        arm = ArmSpec(label=f"sims{n_sims}", n_sims_override=n_sims)
        arm_out = out / f"sims{n_sims}"
        arm_out.mkdir(parents=True, exist_ok=True)
        result = run_arm(
            args.ckpt, arm, book,
            out_dir=str(arm_out),
            workers=args.workers,
            n_boot=args.n_boot,
            book_seed=book_seed,
            expect_encoding=args.expect_encoding,
            n_pairs=n_pairs,
        )
        print(
            f"[WP3] sims={n_sims}  wr={result['wr']:.3f}"
            f"  CI=[{result['pair_ci'][0]:.3f},{result['pair_ci'][1]:.3f}]"
            f"  wall={result['wall_sec']:.0f}s"
        )
        summary["arms"].append({"arm": f"sims{n_sims}", **result})

    if not args.no_solver:
        arm = ArmSpec(label="solver_backup", solver_backup=True)
        arm_out = out / "solver_backup"
        arm_out.mkdir(parents=True, exist_ok=True)
        result = run_arm(
            args.ckpt, arm, book,
            out_dir=str(arm_out),
            workers=args.workers,
            n_boot=args.n_boot,
            book_seed=book_seed,
            expect_encoding=args.expect_encoding,
            n_pairs=n_pairs,
        )
        print(
            f"[WP3] solver_backup  wr={result['wr']:.3f}"
            f"  CI=[{result['pair_ci'][0]:.3f},{result['pair_ci'][1]:.3f}]"
            f"  counters={result['solver_counters']}"
        )
        summary["arms"].append({"arm": "solver_backup", **result})

    (out / "ladder_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
