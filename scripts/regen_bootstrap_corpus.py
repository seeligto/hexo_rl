#!/usr/bin/env python3
"""§122 sweep helper — regenerate bootstrap corpus sliced to a channel subset.

The canonical corpus at ``data/bootstrap_corpus.npz`` stores 18-plane states.
Sweep variants train models with reduced input planes; this script slices the
canonical corpus along axis 1 to emit ``data/bootstrap_corpus_sweep_{name}.npz``
with shape (T, len(channels), 19, 19). Stone positions are NOT recomputed —
planes 0 (current stones) and 8 (opponent stones) carry the full positional
information already.

The training pipeline (``hexo_rl.training.batch_assembly.load_pretrained_buffer``)
detects sliced corpora and scatters them back to the 18-plane buffer wire
format on load, with non-selected planes zeroed. The model then slices the
selected planes again before the trunk forward; the round-trip is consistent
because the canonical corpus values for non-selected planes are exactly what
the model would have seen anyway.

Smoke test:
    .venv/bin/python scripts/regen_bootstrap_corpus.py --variant sweep_2ch
    # writes data/bootstrap_corpus_sweep_2ch.npz with shape (T, 2, 19, 19)

The regression check at the end of every run reconstructs game #0 from the
sliced tensors and verifies plane-0 (current) and plane-8 (opponent) stone
positions match the canonical corpus byte-for-byte.

Usage:
    .venv/bin/python scripts/regen_bootstrap_corpus.py --variant sweep_2ch
    .venv/bin/python scripts/regen_bootstrap_corpus.py --all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CORPUS = REPO_ROOT / "data" / "bootstrap_corpus.npz"
VARIANTS_DIR = REPO_ROOT / "configs" / "variants"
DATA_DIR = REPO_ROOT / "data"

SWEEP_VARIANTS = (
    "sweep_2ch",
    "sweep_3ch",
    "sweep_4ch",
    "sweep_6ch",
    "sweep_8ch",
    "sweep_18ch",
)


def load_variant_channels(variant: str) -> List[int]:
    """Return the `input_channels` list for a sweep variant YAML."""
    path = VARIANTS_DIR / f"{variant}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"variant config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    channels = cfg.get("input_channels")
    if channels is None:
        raise ValueError(
            f"{path} has no `input_channels` field — required for sweep variants."
        )
    if not isinstance(channels, list):
        raise ValueError(f"{path}: `input_channels` must be a list, got {type(channels).__name__}")
    return [int(c) for c in channels]


def regen_one(variant: str, *, force: bool = False) -> Path:
    """Slice canonical corpus to the variant's channel set and write NPZ."""
    if not CANONICAL_CORPUS.exists():
        raise FileNotFoundError(
            f"canonical corpus missing: {CANONICAL_CORPUS} — run `make corpus.export` first"
        )
    channels = load_variant_channels(variant)
    out_path = DATA_DIR / f"bootstrap_corpus_{variant}.npz"
    if out_path.exists() and not force:
        print(f"[{variant}] already exists at {out_path}; skip (use --force to override)")
        return out_path

    t0 = time.time()
    print(f"[{variant}] loading canonical corpus from {CANONICAL_CORPUS}")
    data = np.load(CANONICAL_CORPUS, mmap_mode="r")
    states_canon = data["states"]    # (T, 18 or 24, 19, 19) float16
    policies = np.array(data["policies"])
    outcomes = np.array(data["outcomes"])
    weights = np.array(data["weights"]) if "weights" in data.files else None

    # Trim 24-plane legacy corpora to 18 (chain planes were moved to a separate
    # sub-buffer post-§97); the sweep operates on the 18-plane wire layout.
    if states_canon.shape[1] == 24:
        print(f"[{variant}] trimming 24-plane corpus to 18 planes")
        states_18 = np.array(states_canon[:, :18], dtype=np.float16)
    elif states_canon.shape[1] == 18:
        states_18 = np.array(states_canon, dtype=np.float16)
    else:
        raise ValueError(
            f"unexpected canonical corpus plane count: {states_canon.shape[1]} "
            f"(expected 18 or 24)"
        )

    for c in channels:
        if c < 0 or c >= 18:
            raise ValueError(
                f"[{variant}] channel {c} out of range [0, 18); fix the variant YAML."
            )

    print(f"[{variant}] slicing to channels {channels}")
    states_sliced = states_18[:, channels, :, :].astype(np.float16)
    print(
        f"[{variant}] sliced shape={states_sliced.shape} "
        f"dtype={states_sliced.dtype} "
        f"size={states_sliced.nbytes / 1e6:.1f} MB"
    )

    save_kwargs = dict(
        states=states_sliced,
        policies=policies,
        outcomes=outcomes,
        input_channels=np.array(channels, dtype=np.int64),
    )
    if weights is not None:
        save_kwargs["weights"] = weights

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **save_kwargs)
    print(f"[{variant}] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ── Regression check: stone positions in game #0 must match canonical. ──
    # Pick the canonical plane indices for current/opponent stones; verify
    # they appear at the same slot in the sliced tensor (channels list maps
    # canonical plane → slice slot).
    if 0 in channels and 8 in channels:
        idx_cur = channels.index(0)
        idx_opp = channels.index(8)
        canon_cur = np.asarray(states_18[0, 0])
        canon_opp = np.asarray(states_18[0, 8])
        sliced_cur = np.asarray(states_sliced[0, idx_cur])
        sliced_opp = np.asarray(states_sliced[0, idx_opp])
        if not np.array_equal(canon_cur, sliced_cur):
            raise RuntimeError(
                f"[{variant}] regression FAIL: game #0 plane 0 (current stones) "
                f"diverges between canonical and sliced corpus."
            )
        if not np.array_equal(canon_opp, sliced_opp):
            raise RuntimeError(
                f"[{variant}] regression FAIL: game #0 plane 8 (opponent stones) "
                f"diverges between canonical and sliced corpus."
            )
        n_cur = int(np.asarray(canon_cur).sum())
        n_opp = int(np.asarray(canon_opp).sum())
        print(
            f"[{variant}] regression OK: game#0 stones match "
            f"(plane0 cur={n_cur}, plane8 opp={n_opp})"
        )
    else:
        # The HexTacToeNet validator already rejects channel lists missing 0/8
        # at model construction. This branch is unreachable in practice;
        # we keep it as a safety net for hand-edited configs.
        print(
            f"[{variant}] WARNING: channels {channels} missing 0 or 8; "
            f"skipping stone-position regression check."
        )

    print(f"[{variant}] done in {time.time() - t0:.1f}s")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", action="append", default=[],
                   help="Variant name (sweep_Nch). Repeatable. Mutually exclusive with --all.")
    p.add_argument("--all", action="store_true",
                   help="Regenerate corpora for all six sweep variants.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing NPZ files.")
    args = p.parse_args()

    if args.all and args.variant:
        print("ERROR: --all and --variant are mutually exclusive", file=sys.stderr)
        return 2
    if not args.all and not args.variant:
        print("ERROR: pass --variant <name> (repeatable) or --all", file=sys.stderr)
        return 2

    variants = list(SWEEP_VARIANTS) if args.all else args.variant
    failed: List[str] = []
    for v in variants:
        try:
            regen_one(v, force=args.force)
        except Exception as exc:
            print(f"[{v}] FAILED: {exc}", file=sys.stderr)
            failed.append(v)

    if failed:
        print(f"\nfailures: {failed}", file=sys.stderr)
        return 1
    print(f"\nall {len(variants)} variants regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
