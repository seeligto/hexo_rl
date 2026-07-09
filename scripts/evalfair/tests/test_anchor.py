"""T-ANCHOR (integration): 175k on evalfair_r4_v2 reproduces WR 0.594 within CI.

Skipped unless --run-integration is passed; needs GPU + run2_175k.pt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CKPT = Path("/home/timmy/Work/Hexo/hexo_rl/scripts/arena/weights/run2_175k.pt")
BOOK_R4 = Path("/home/timmy/Work/Hexo/hexo_rl/tests/fixtures/opening_books/evalfair_r4_v2.json")


@pytest.mark.integration
def test_anchor_175k_wr_within_ci(tmp_path):
    """175k anchor on evalfair_r4_v2 must reproduce WR 0.594 within CI [0.508, 0.672]."""
    from scripts.evalfair.core import run_arm, ArmSpec
    from scripts.evalfair.book import load_book

    book = load_book(BOOK_R4)
    arm = ArmSpec(label="sims150")
    result = run_arm(
        str(CKPT), arm, book,
        out_dir=str(tmp_path),
        workers=1,
        n_boot=2000,
        book_seed=20260709,
    )
    wr = result["wr"]
    lo, hi = result["pair_ci"]
    # Verdict-2 verified: WR=0.594, CI=[0.508, 0.672]
    assert 0.508 <= wr <= 0.672, f"anchor WR {wr:.3f} outside CI [0.508, 0.672]"
    assert lo >= 0.45, f"CI lower {lo:.3f} too low (anchor regression)"
    assert hi <= 0.75, f"CI upper {hi:.3f} too high"
