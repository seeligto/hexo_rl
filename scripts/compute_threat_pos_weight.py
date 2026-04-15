#!/usr/bin/env python
"""Compute the Q19 threat-head BCE pos_weight empirically from a buffer.

The threat head predicts per-cell membership in the 6-cell winning line. The
`winning_line` label is ~1.6% positive (6 of 361 cells per game-end row, all
zeros for draws), which without pos_weight drives BCE logits strongly negative
(see docs/07_PHASE4_SPRINT_LOG.md §91 and docs/06_OPEN_QUESTIONS.md Q19).

This script computes `pos_weight = (1 − p) / p` from the empirical positive
fraction `p` in the replay buffer (HEXB v3) — or falls back to the §91
theoretical value when no buffer is available — and optionally writes the
result to configs/training.yaml:threat_pos_weight.

Usage:
    .venv/bin/python scripts/compute_threat_pos_weight.py             # print only
    .venv/bin/python scripts/compute_threat_pos_weight.py --write     # update yaml
    .venv/bin/python scripts/compute_threat_pos_weight.py --buffer-path <path>
"""
from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path
from typing import Optional

import numpy as np


CONFIG_PATH = Path("configs/training.yaml")
DEFAULT_BUFFER_PATH = Path("checkpoints/replay_buffer.bin")
# §91 field value: 59.0 was set empirically from early buffer samples before the
# draw-rate correction was applied. The stricter theoretical derivation gives
# 6 / (361 × 0.75) ≈ 1.25% positive → (1−p)/p ≈ 79, but the empirical buffer
# sampled during §91 yielded p ≈ 1.67% → (1−p)/p ≈ 59. We keep 59.0 as the
# fallback constant to match the §91 sprint note; run --write against a real
# buffer to get an up-to-date empirical estimate.
_THEORETICAL_POS_WEIGHT = 59.0

# HEXB v3 header constants (little-endian): magic(4) + version(4) + n_planes(4)
# + capacity(8) + size(8) = 28 bytes total.
_HEXB_MAGIC = 0x4845_5842  # "HEXB"
_HEXB_VERSION = 3
_HEXB_HEADER_FMT = "<IIIQQ"  # magic, version, n_planes, capacity, size
_HEXB_HEADER_SIZE = struct.calcsize(_HEXB_HEADER_FMT)  # 28

# Cap allocation to match the bootstrap corpus ceiling (~200K rows) with
# headroom. load_from_path truncates to the most recent rows per loader
# contract, so this is always safe.
_MAX_ALLOC_ROWS = 250_000


def _read_hexb_row_count(path: Path) -> int:
    """Return the saved row count from a HEXB v3 header, or 0 on any error.

    Reads only the 28-byte header — no row data is touched. Validation is
    minimal: wrong magic or wrong version returns 0 (caller falls back to the
    theoretical constant anyway).
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(_HEXB_HEADER_SIZE)
        if len(raw) < _HEXB_HEADER_SIZE:
            return 0
        magic, version, _n_planes, _capacity, size = struct.unpack(_HEXB_HEADER_FMT, raw)
        if magic != _HEXB_MAGIC or version != _HEXB_VERSION:
            return 0
        return int(size)
    except OSError:
        return 0


def from_buffer(buffer_path: Path, sample_n: int = 10_000) -> Optional[float]:
    """Return empirical (1−p)/p from a Rust ReplayBuffer on disk, or None.

    Loads the buffer, samples up to `sample_n` rows with augment=False (we
    only need the scalar positive fraction — augmentation is a no-op under
    mean reduction), and returns the inverse label density.
    """
    if not buffer_path.exists():
        return None
    try:
        from engine import ReplayBuffer
    except ImportError:
        print(f"[warn] engine extension not importable; falling back to theoretical")
        return None

    row_count = _read_hexb_row_count(buffer_path)
    alloc = max(1, min(row_count, sample_n, _MAX_ALLOC_ROWS))
    buf = ReplayBuffer(alloc)
    try:
        n = buf.load_from_path(str(buffer_path))
    except Exception as exc:
        print(f"[warn] failed to load buffer at {buffer_path}: {exc}")
        return None

    if n == 0:
        return None

    batch_size = min(n, sample_n)
    try:
        _, _, _, _, winning_line = buf.sample_batch(batch_size, False)
    except Exception as exc:
        print(f"[warn] failed to sample from buffer: {exc}")
        return None

    wl = np.asarray(winning_line, dtype=np.float32)
    p = float(wl.mean())
    if p <= 0.0:
        print(f"[warn] empirical positive fraction is 0 (all-draw buffer?) — falling back")
        return None

    print(
        f"empirical: sampled {batch_size} rows from {buffer_path} — "
        f"positive fraction p = {p:.6f}"
    )
    return (1.0 - p) / p


def update_config(value: float) -> None:
    text = CONFIG_PATH.read_text()
    pattern = re.compile(r"^(threat_pos_weight:\s*)[\d.]+", re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError(
            f"could not find threat_pos_weight: key in {CONFIG_PATH}"
        )
    new_text = pattern.sub(lambda m: f"{m.group(1)}{value:.1f}", text)
    CONFIG_PATH.write_text(new_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buffer-path",
        type=Path,
        default=DEFAULT_BUFFER_PATH,
        help=f"HEXB v3 replay buffer file to sample from (default: {DEFAULT_BUFFER_PATH})",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=10_000,
        help="rows to sample for the empirical estimate (default: 10000)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="overwrite threat_pos_weight: in configs/training.yaml with the computed value",
    )
    args = parser.parse_args()

    empirical = from_buffer(args.buffer_path, args.sample_n)
    if empirical is not None:
        value = empirical
        print(f"empirical pos_weight = (1-p)/p = {value:.2f}")
    else:
        value = _THEORETICAL_POS_WEIGHT
        print(
            f"no usable buffer at {args.buffer_path} — "
            f"using §91 theoretical pos_weight = {value:.2f}"
        )

    if args.write:
        update_config(value)
        print(f"wrote threat_pos_weight: {value:.1f} to {CONFIG_PATH}")
    else:
        print(f"(dry run — pass --write to update {CONFIG_PATH})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
