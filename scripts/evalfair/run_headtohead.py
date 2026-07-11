"""evalfair/run_headtohead.py — WP1 net-vs-net CLI: ckpt A vs ckpt B -> result.json.

Two deploy heads (g=0 deploy-matched), colors swapped, per-side sims. Board radius = book's
radius_stage. Prints WR (head A), pair-CI, eff_n, draw_rate, and a loud DETERMINISM-COLLAPSE
warning when distinct trajectories fall below half the games (widen the book, rerun).

READ A:  --ckpt-a run2_retro/checkpoint_00248000.pt --ckpt-b run2_retro/checkpoint_00050000.pt
         --sims-a 150 --sims-b 150   (run on BOTH r4 and r5 books)
READ B:  --ckpt-a run2_175k.pt --ckpt-b run2_175k.pt --sims-a 150 --sims-b 75
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evalfair.book import FIXTURE_DIR, load_book
from scripts.evalfair.headtohead import run_headtohead


def main() -> None:
    ap = argparse.ArgumentParser(description="evalfair head-vs-head (net-vs-net) evaluation")
    ap.add_argument("--ckpt-a", required=True, dest="ckpt_a", help="Focal head A checkpoint")
    ap.add_argument("--ckpt-b", required=True, dest="ckpt_b", help="Opponent head B checkpoint")
    ap.add_argument("--book", default=None, help="book_v2 JSON (default: evalfair_r4_v2 fixture)")
    ap.add_argument("--out", default="reports/anchorx/hh_run", help="Output directory")
    ap.add_argument("--sims-a", type=int, default=None, dest="sims_a",
                    help="Override head A n_sims_full (default: ckpt's deploy sims)")
    ap.add_argument("--sims-b", type=int, default=None, dest="sims_b",
                    help="Override head B n_sims_full (default: ckpt's deploy sims)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--expect-encoding", default="v6_live2_ls", dest="expect_encoding")
    ap.add_argument("--n-pairs", type=int, default=None, dest="n_pairs",
                    help="Limit to first N pairs (smoke/test use)")
    ap.add_argument("--label-a", default="head_a", dest="label_a")
    ap.add_argument("--label-b", default="head_b", dest="label_b")
    args = ap.parse_args()

    book_path = args.book or str(FIXTURE_DIR / "evalfair_r4_v2.json")
    book = load_book(Path(book_path))

    r = run_headtohead(
        args.ckpt_a, args.ckpt_b, book,
        out_dir=args.out, sims_a=args.sims_a, sims_b=args.sims_b,
        workers=args.workers, n_boot=args.n_boot,
        book_seed=book.get("seed", 20260709),
        expect_encoding=args.expect_encoding, n_pairs=args.n_pairs,
        label_a=args.label_a, label_b=args.label_b,
    )

    print(
        f"[{r['book_id']} r{r['board_radius']}] "
        f"A(step {r['ckpt_a_step']}, sims {r['sims_a']}) vs B(step {r['ckpt_b_step']}, sims {r['sims_b']})\n"
        f"  WR_A={r['wr']:.3f}  pair-CI=[{r['pair_ci'][0]:.3f},{r['pair_ci'][1]:.3f}]"
        f"  n={r['n']} eff_n={r['eff_n']} (distinct {r['distinct_frac']:.2f})"
        f"  draws={r['draw_rate']:.2f}  bad_pairs={r['bad_pairs']}\n"
        f"  off_stage: A={r['off_stage_a']} (native r{r['native_radius_a']}) "
        f"B={r['off_stage_b']} (native r{r['native_radius_b']})  wall={r['wall_sec']:.0f}s"
    )
    if r["determinism_collapse"]:
        print(
            f"  ** DETERMINISM COLLAPSE ** distinct {r['eff_n']}/{r['n']} < 0.5 — "
            f"argmax heads mirror-locked on this book. Widen to a 4-ply book and rerun; "
            f"do NOT trust the pair-CI as-is."
        )


if __name__ == "__main__":
    main()
