"""§S181 Track A — validate position classifier against T3 bank.

Loads the T3 40-position bank, builds each position via Board moves, and
runs `classify_board()`. Reports accuracy vs the bank's stored `pos_class`
field. Used during classifier development; not part of the audit outputs.

  .venv/bin/python scripts/structural_diagnosis/track_a/_validate_classifier.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from engine import Board
from scripts.structural_diagnosis.track_a.position_classifier import (
    classify_board,
    _find_max_open_run,
    _mean_hex_dist_from_centroid,
    HALF,
    TRUNK_SZ,
)
import numpy as np


def main():
    fixture = REPO / "tests" / "fixtures" / "value_spread_bank.json"
    data = json.loads(fixture.read_text())
    positions = data["positions"]

    n_total = 0
    n_correct = 0
    matrix = {"colony": {"colony": 0, "extension": 0, "neither": 0},
              "extension": {"colony": 0, "extension": 0, "neither": 0}}
    details = []
    for spec in positions:
        b = Board()
        ok = True
        for q, r in spec["moves"]:
            try:
                b.apply_move(int(q), int(r))
            except Exception:
                ok = False
                break
        if not ok:
            print(f"  SKIP {spec['name']}: invalid moves")
            continue
        truth = "colony" if spec["pos_class"] == "colony" else "extension"
        pred = classify_board(b)
        # diagnostic numbers
        stones = b.get_stones()
        cp_mask = np.zeros((TRUNK_SZ, TRUNK_SZ), dtype=bool)
        op_mask = np.zeros((TRUNK_SZ, TRUNK_SZ), dtype=bool)
        occ_mask = np.zeros((TRUNK_SZ, TRUNK_SZ), dtype=bool)
        for q, r, p in stones:
            i, j = q + HALF, r + HALF
            if not (0 <= i < TRUNK_SZ and 0 <= j < TRUNK_SZ):
                continue
            occ_mask[i, j] = True
            if p == b.current_player:
                cp_mask[i, j] = True
            else:
                op_mask[i, j] = True
        cp_run = _find_max_open_run(cp_mask, occ_mask)
        op_run = _find_max_open_run(op_mask, occ_mask)
        md = _mean_hex_dist_from_centroid(occ_mask)
        n_in = int(occ_mask.sum())
        n_total += 1
        matrix[truth][pred] += 1
        if pred == truth:
            n_correct += 1
        details.append(dict(name=spec["name"], truth=truth, pred=pred,
                            cp_run=cp_run, op_run=op_run, mean_hex_dist=round(md, 3),
                            n_in_window=n_in, n_total_stones=len(stones)))

    print(f"\n=== T3 bank classifier validation ===")
    print(f"n_total: {n_total}")
    print(f"n_correct: {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%\n")
    print(f"Confusion matrix (rows = truth, cols = pred):")
    print(f"{'':12} {'colony':>10} {'extension':>10} {'neither':>10}")
    for t in ("colony", "extension"):
        row = matrix[t]
        print(f"{t:12} {row['colony']:>10} {row['extension']:>10} {row['neither']:>10}")
    print()

    misclass = [d for d in details if d["pred"] != d["truth"]]
    if misclass:
        print(f"Misclassifications ({len(misclass)}):")
        for d in misclass:
            print(f"  {d}")
    print()

    # Print distribution to calibrate thresholds
    col = [d for d in details if d["truth"] == "colony"]
    ext = [d for d in details if d["truth"] == "extension"]
    print(f"colony mean_hex_dist: min={min(d['mean_hex_dist'] for d in col):.2f}, "
          f"max={max(d['mean_hex_dist'] for d in col):.2f}, "
          f"mean={np.mean([d['mean_hex_dist'] for d in col]):.2f}")
    print(f"extension mean_hex_dist: min={min(d['mean_hex_dist'] for d in ext):.2f}, "
          f"max={max(d['mean_hex_dist'] for d in ext):.2f}, "
          f"mean={np.mean([d['mean_hex_dist'] for d in ext]):.2f}")
    print(f"colony cp_run: {[d['cp_run'] for d in col]}")
    print(f"extension cp_run: {[d['cp_run'] for d in ext]}")
    print(f"extension op_run: {[d['op_run'] for d in ext]}")


if __name__ == "__main__":
    main()
