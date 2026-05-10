#!/usr/bin/env python
"""Stamp legacy `.pt` checkpoints with §172 A2 §8 metadata.

Operator-driven helper — NOT auto-run during tests or commits.

For each `.pt` file under the target directory:
  1. Load the dict (torch.load, weights_only=False — files are trusted).
  2. If `metadata['encoding_name']` already present → 'current' (no-op).
  3. If a `model_state` dict (or raw state-dict) is recoverable, infer
     encoding via `hexo_rl.encoding.compat.infer_encoding_from_state_dict`
     and rewrite the .pt with a stamped metadata block.
  4. Otherwise → 'unresolvable' (logged, never overwritten).

`--dry-run` prints what WOULD happen but writes nothing. Idempotent: a
second run on stamped ckpts is a no-op.

Examples:
    python scripts/backfill_checkpoint_metadata.py --dir checkpoints/ --dry-run
    python scripts/backfill_checkpoint_metadata.py --dir checkpoints/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

# Repo-root sys.path bootstrap — matches scripts/benchmark.py + scripts/train.py.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from hexo_rl.encoding import compat  # noqa: E402
from hexo_rl.encoding.registry import EncodingRegistryError  # noqa: E402
from hexo_rl.training.checkpoints import build_checkpoint_metadata  # noqa: E402


def _extract_state_dict(d: Any) -> Mapping[str, Any] | None:
    """Pull a state-dict out of a loaded ckpt object (full-dict OR raw sd)."""
    if isinstance(d, Mapping):
        if "model_state" in d and isinstance(d["model_state"], Mapping):
            return d["model_state"]
        # Heuristic: if any value looks like a tensor, treat the whole dict
        # as a raw state_dict (e.g. inference_only.pt, best_model.pt).
        for v in d.values():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return d
    return None


def process_checkpoint(path: Path, *, dry_run: bool) -> tuple[str, str | None, str]:
    """Inspect/stamp one .pt. Returns (status, encoding, notes).

    status ∈ {current, backfilled, dry-run-would-backfill, unresolvable, error}.
    """
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # broad on purpose: corrupt zips, arch issues
        return ("error", None, f"load failed: {exc!s}")

    if isinstance(d, Mapping) and isinstance(d.get("metadata"), Mapping) \
            and isinstance(d["metadata"].get("encoding_name"), str):
        return ("current", d["metadata"]["encoding_name"], "")

    sd = _extract_state_dict(d)
    if sd is None:
        return ("unresolvable", None, "no state_dict / model_state recoverable")

    try:
        enc_name = compat.infer_encoding_from_state_dict(sd, str(path))
    except EncodingRegistryError as exc:
        return ("unresolvable", None, f"infer failed: {exc!s}")

    if dry_run:
        return ("dry-run-would-backfill", enc_name, "stamp on next non-dry run")

    # Re-save: full-dict ckpt → set d["metadata"]; raw state-dict → wrap.
    if isinstance(d, Mapping) and "model_state" in d:
        out: dict[str, Any] = dict(d)
        out["metadata"] = build_checkpoint_metadata(encoding_name=enc_name)
    else:
        # Raw state-dict case (best_model.pt / inference_only.pt). Wrap in
        # a full-dict shape so the load path can read metadata uniformly.
        out = {
            "step": None,
            "model_state": dict(sd) if isinstance(sd, Mapping) else sd,
            "optimizer_state": None,
            "scaler_state": None,
            "scheduler_state": None,
            "config": None,
            "metadata": build_checkpoint_metadata(encoding_name=enc_name),
        }
    torch.save(out, path)
    return ("backfilled", enc_name, "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("checkpoints"),
        help="directory to scan recursively for .pt files (default: ./checkpoints)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report only; do not rewrite any files",
    )
    args = parser.parse_args(argv)

    if not args.dir.exists():
        print(f"error: {args.dir} does not exist", file=sys.stderr)
        return 2

    rows: list[tuple[Path, str, str | None, str]] = []
    for p in sorted(args.dir.rglob("*.pt")):
        status, enc, notes = process_checkpoint(p, dry_run=args.dry_run)
        rows.append((p, status, enc, notes))

    # Width-padded report.
    if not rows:
        print(f"no .pt files under {args.dir}")
        return 0

    path_w = max(len(str(r[0])) for r in rows)
    status_w = max(len(r[1]) for r in rows)
    enc_w = max(len(r[2] or "—") for r in rows)
    print(f"{'path':<{path_w}}  {'status':<{status_w}}  {'encoding':<{enc_w}}  notes")
    print("-" * (path_w + status_w + enc_w + 12))
    n_unresolvable = 0
    for path, status, enc, notes in rows:
        print(
            f"{str(path):<{path_w}}  {status:<{status_w}}  "
            f"{(enc or '—'):<{enc_w}}  {notes}"
        )
        if status in ("unresolvable", "error"):
            n_unresolvable += 1

    return 1 if n_unresolvable else 0


if __name__ == "__main__":
    raise SystemExit(main())
