"""evalfair/run_eval.py — single ckpt, single arm -> games.jsonl + result.json (WP3 unit, WP5).

Accepts --override-n-sims (WP3 only); stamps deploy_matched=false in that case.
Workers supported via --workers N (default 1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evalfair.book import load_book, FIXTURE_DIR
from scripts.evalfair.core import ArmSpec, run_arm


def main() -> None:
    ap = argparse.ArgumentParser(
        description="evalfair single-ckpt single-arm evaluation -> games.jsonl + result.json"
    )
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument(
        "--book",
        default=None,
        help="Path to book_v2 JSON (default: evalfair_r4_v2.json fixture)",
    )
    ap.add_argument("--out", default="reports/evalfair", help="Output directory")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument(
        "--override-n-sims",
        type=int,
        default=None,
        dest="override_n_sims",
        help="Override n_sims_full (WP3 only; stamps deploy_matched=false)",
    )
    ap.add_argument("--solver-backup", action="store_true", dest="solver_backup",
                    help="Wrap head in SolverBackupBot (WP3.3; stamps deploy_matched=false)")
    ap.add_argument("--expect-encoding", default="v6_live2_ls", dest="expect_encoding")
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="Limit to first N pairs (smoke/test use)")
    args = ap.parse_args()

    # Determine book
    book_path = args.book
    if book_path is None:
        book_path = str(FIXTURE_DIR / "evalfair_r4_v2.json")
    book = load_book(Path(book_path))

    arm = ArmSpec(
        label=f"sims{args.override_n_sims or 'deploy'}",
        n_sims_override=args.override_n_sims,
        solver_backup=args.solver_backup,
    )

    result = run_arm(
        args.ckpt, arm, book,
        out_dir=args.out,
        workers=args.workers,
        n_boot=args.n_boot,
        book_seed=book.get("seed", 20260709),
        expect_encoding=args.expect_encoding,
        n_pairs=args.n_pairs,
    )

    print(
        f"wr={result['wr']:.3f}  CI=[{result['pair_ci'][0]:.3f},{result['pair_ci'][1]:.3f}]"
        f"  n={result['n']}  eff_n={result['eff_n']}"
        f"  deploy_matched={result['deploy_matched']}"
        f"  wall={result['wall_sec']:.0f}s"
    )


if __name__ == "__main__":
    main()
