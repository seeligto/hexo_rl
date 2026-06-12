#!/usr/bin/env python3
"""Generate the cross-language temperature-parity golden fixture.

D-TEMPDECAY (2026-06-12, review HIGH finding): pins the contract that the Python
``get_temperature(mode="training")`` path and the Rust ``compute_move_temperature``
path produce the SAME within-game temperature for the same (ply, threshold,
temp_min) — so the two formulas + the ply→compound-turn clock cannot silently
diverge.

Reference contract (f64), driven by PLY (covers the ply→cm clock + the ply-0
special case + the ply1/ply2 boundary, exactly as both impls compute it):

    cm  = 0 if ply == 0 else (ply + 1) // 2          # == Rust div_ceil(ply, 2)
    tau = max(temp_min, cos(pi/2 * cm / threshold))  if threshold > 0 and cm < threshold
        = temp_min                                   otherwise   (threshold 0 ⇒ OFF)

Both `tests/test_temperature.py::test_*_parity_golden` (Python) and
`engine/tests/temperature_parity_golden.rs` (Rust) read the emitted CSV and assert
their impl reproduces `expected_tau` within 1e-6 (Rust is f32, ~1 ULP below tol).

Regenerate:  python tests/fixtures/gen_temperature_parity_golden.py
"""
from __future__ import annotations

import math
from pathlib import Path

# (threshold_compound, temp_min) — control(OFF) + the 3 probe arms + the §143
# legacy cosine + a wide-window case.
CONFIGS = [(0, 0.5), (12, 0.30), (12, 0.45), (12, 0.20), (15, 0.05), (24, 0.35)]
# plies: ply-0, the ply1/ply2 (same-cm) boundary, mid-decay, at/just-past the
# compound threshold, and deep tail.
PLIES = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 30, 48]


def reference_tau(ply: int, threshold: int, temp_min: float) -> float:
    cm = 0 if ply == 0 else (ply + 1) // 2
    if threshold > 0 and cm < threshold:
        return max(temp_min, math.cos(math.pi / 2 * cm / threshold))
    return temp_min


def main() -> None:
    out = Path(__file__).with_name("temperature_parity_golden.csv")
    rows = ["# ply,threshold_compound,temp_min,expected_tau  (see gen_temperature_parity_golden.py)"]
    for thr, tmin in CONFIGS:
        for ply in PLIES:
            tau = reference_tau(ply, thr, tmin)
            rows.append(f"{ply},{thr},{tmin:.4f},{tau:.10f}")
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows) - 1} rows -> {out}")


if __name__ == "__main__":
    main()
