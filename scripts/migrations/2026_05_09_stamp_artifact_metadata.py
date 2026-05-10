#!/usr/bin/env python
"""Stamp §172 A2 §8 metadata onto legacy HeXO artifacts.

Design reference: docs/designs/encoding_registry_design.md §8.

Three subcommands:

  checkpoints   Stamp legacy `.pt` checkpoints with encoding metadata.
                For each .pt file under --dir:
                  1. Load the dict (torch.load, weights_only=False — files are trusted).
                  2. If `metadata['encoding_name']` already present → 'current' (no-op).
                  3. If a model_state dict is recoverable, infer encoding via
                     hexo_rl.encoding.compat.infer_encoding_from_state_dict and
                     rewrite the .pt with a stamped metadata block.
                  4. Otherwise → 'unresolvable' (logged, never overwritten).

  corpora       Backfill `.metadata.json` sidecars for legacy `.npz` corpora.
                §172 Phase A5.2 — one-shot helper to bring pre-registry corpora
                into the sidecar regime defined by design §9.

  model-variant Backfill `model_variant: None` on stamped checkpoints that carry
                metadata (encoding_name present) but lack the model_variant key.
                §172 A10 close-out — brings stamped ckpts to the full 8-key schema.
                Skips ckpts without metadata entirely (A5 backfill responsibility).

All subcommands accept `--dry-run` (report only; write nothing) and are
idempotent (second run on already-stamped artifacts is a no-op).

Operator-driven — NOT auto-run during tests or commits.

Examples:
    python -m scripts.migrations.2026_05_09_stamp_artifact_metadata checkpoints --dir checkpoints/ --dry-run
    python -m scripts.migrations.2026_05_09_stamp_artifact_metadata checkpoints --dir checkpoints/
    python -m scripts.migrations.2026_05_09_stamp_artifact_metadata corpora --dir data/ --dry-run
    python -m scripts.migrations.2026_05_09_stamp_artifact_metadata corpora --dir data/ --manifest manifest.yaml
    python -m scripts.migrations.2026_05_09_stamp_artifact_metadata model-variant --dir checkpoints/ --dry-run
    python -m scripts.migrations.2026_05_09_stamp_artifact_metadata model-variant --dir checkpoints/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Mapping

# Repo-root sys.path bootstrap — matches scripts/benchmark.py + scripts/train.py.
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Subcommand: checkpoints
# ---------------------------------------------------------------------------

def _ckpt_extract_state_dict(d: Any) -> Mapping[str, Any] | None:
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


def _process_checkpoint(path: pathlib.Path, *, dry_run: bool) -> tuple[str, str | None, str]:
    """Inspect/stamp one .pt. Returns (status, encoding, notes).

    status ∈ {current, backfilled, dry-run-would-backfill, unresolvable, error}.
    """
    import torch  # local import — not needed for corpora subcommand
    from hexo_rl.encoding import compat
    from hexo_rl.encoding.registry import EncodingRegistryError
    from hexo_rl.training.checkpoints import build_checkpoint_metadata

    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # broad on purpose: corrupt zips, arch issues
        return ("error", None, f"load failed: {exc!s}")

    if isinstance(d, Mapping) and isinstance(d.get("metadata"), Mapping) \
            and isinstance(d["metadata"].get("encoding_name"), str):
        return ("current", d["metadata"]["encoding_name"], "")

    sd = _ckpt_extract_state_dict(d)
    if sd is None:
        return ("unresolvable", None, "no state_dict / model_state recoverable")

    try:
        enc_name = compat.infer_encoding_from_state_dict(sd, str(path))
    except EncodingRegistryError as exc:
        return ("unresolvable", None, f"infer failed: {exc!s}")

    if dry_run:
        return ("dry-run-would-backfill", enc_name, "stamp on next non-dry run")

    # Re-save: full-dict ckpt → set d["metadata"]; raw state-dict → wrap.
    import torch  # already imported above, but keep pyflakes happy
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


def _cmd_checkpoints(args: argparse.Namespace) -> int:
    import torch  # noqa: F401 — imported here to give clear ImportError if missing
    target = pathlib.Path(args.dir)
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 2

    rows: list[tuple[pathlib.Path, str, str | None, str]] = []
    for p in sorted(target.rglob("*.pt")):
        status, enc, notes = _process_checkpoint(p, dry_run=args.dry_run)
        rows.append((p, status, enc, notes))

    if not rows:
        print(f"no .pt files under {target}")
        return 0

    path_w = max(len(str(r[0])) for r in rows)
    status_w = max(len(r[1]) for r in rows)
    enc_w = max(len(r[2] or "—") for r in rows)
    print(f"{'path':<{path_w}}  {'status':<{status_w}}  {'encoding':<{enc_w}}  notes")
    print("-" * (path_w + status_w + enc_w + 12))
    n_bad = 0
    for path, status, enc, notes in rows:
        print(
            f"{str(path):<{path_w}}  {status:<{status_w}}  "
            f"{(enc or '—'):<{enc_w}}  {notes}"
        )
        if status in ("unresolvable", "error"):
            n_bad += 1

    return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# Subcommand: corpora
# ---------------------------------------------------------------------------

# Filename → encoding inference table. Order matters: longest match first.
_CORPUS_FILENAME_RULES: list[tuple[str, str | None]] = [
    ("bootstrap_corpus_v8_canvas_realness", "v8_canvas_realness"),
    ("bootstrap_corpus_v6w25",              "v6w25"),
    ("bootstrap_corpus_v7full",             "v7full"),
    ("bootstrap_corpus_v7",                 "v7"),
    ("bootstrap_corpus_v8",                 "v8"),
    ("adversarial_corpus_v8",               "v8"),
    ("bootstrap_corpus_sweep_2ch",          None),
    ("bootstrap_corpus_sweep_6ch",          None),
    ("bootstrap_corpus",                    "v6"),  # design §9 default
]


def _infer_corpus_encoding(stem: str) -> str | None:
    for prefix, enc in _CORPUS_FILENAME_RULES:
        if stem == prefix or stem.startswith(prefix + "_") or stem.startswith(prefix + "-"):
            return enc
    return None


def _load_manifest(path: pathlib.Path) -> dict[str, str]:
    """Read `encodings: {<filename>: <encoding>}` mapping from YAML."""
    try:
        import yaml  # local import — only needed when a manifest is supplied
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"manifest YAML requires PyYAML; install via .venv: {exc}"
        )
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"manifest root must be a mapping: {path}")
    encodings = raw.get("encodings", {})
    if not isinstance(encodings, dict):
        raise SystemExit(f"manifest 'encodings' must be a mapping: {path}")
    out: dict[str, str] = {}
    for k, v in encodings.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise SystemExit(
                f"manifest 'encodings' entries must be str:str — got {k!r}:{v!r}"
            )
        out[k] = v
    return out


def _read_n_positions(npz_path: pathlib.Path) -> int:
    """Lazily read leading dim of 'states' (or first array) — mmap to avoid full read."""
    import numpy as np
    with np.load(npz_path, mmap_mode="r") as data:
        if "states" in data.files:
            arr = data["states"]
        else:
            if not data.files:
                raise SystemExit(f"npz contains no arrays: {npz_path}")
            arr = data[data.files[0]]
        return int(arr.shape[0])


def _truncate_sha(s: str) -> str:
    return s[:12] + "…" if len(s) > 13 else s


def _process_corpus(
    npz_path: pathlib.Path,
    *,
    manifest: dict[str, str],
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    from hexo_rl.bootstrap.corpus_io import (
        SCHEMA_VERSION,
        compute_npz_sha256,
        _resolve_git_commit,
        _sidecar_path,
        _utc_now_iso,
    )
    from hexo_rl.encoding import lookup as _registry_lookup
    from hexo_rl.encoding.registry import EncodingRegistryError

    sidecar = _sidecar_path(npz_path)
    name = npz_path.name

    # Resolve encoding: manifest > filename rule.
    if name in manifest:
        encoding = manifest[name]
        source = "manifest"
    else:
        encoding = _infer_corpus_encoding(npz_path.stem)
        source = "filename" if encoding is not None else "none"

    if encoding is None:
        return {
            "path": str(npz_path),
            "status": "no-manifest-required-skip",
            "encoding": None,
            "sha256": None,
            "notes": "filename matches sweep/unknown; supply --manifest to disambiguate",
        }

    # Validate against registry up front.
    try:
        _registry_lookup(encoding)
    except EncodingRegistryError as exc:
        return {
            "path": str(npz_path),
            "status": "error",
            "encoding": encoding,
            "sha256": None,
            "notes": f"unknown encoding ({source}): {exc}",
        }

    if sidecar.exists() and not force:
        return {
            "path": str(npz_path),
            "status": "current",
            "encoding": encoding,
            "sha256": None,
            "notes": f"sidecar already present ({sidecar.name})",
        }

    sha = compute_npz_sha256(npz_path)
    n_positions = _read_n_positions(npz_path)
    metadata = {
        "encoding_name": encoding,
        "sha256": sha,
        "n_positions": n_positions,
        "source_manifest": None,
        "created_at": _utc_now_iso(),
        "created_by_commit": _resolve_git_commit(),
        "schema_version": SCHEMA_VERSION,
        "extra": {"backfilled_from": source},
    }

    if dry_run:
        return {
            "path": str(npz_path),
            "status": "dry-run",
            "encoding": encoding,
            "sha256": _truncate_sha(sha),
            "notes": f"would write {sidecar.name}",
        }

    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(npz_path),
        "status": "rewritten" if force else "backfilled",
        "encoding": encoding,
        "sha256": _truncate_sha(sha),
        "notes": f"wrote {sidecar.name}",
    }


def _format_corpus_report(rows: list[dict[str, Any]]) -> str:
    cols = ("path", "status", "encoding", "sha256", "notes")
    widths = {c: max(len(c), max((len(str(r.get(c) or "")) for r in rows), default=0))
              for c in cols}
    lines = []
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    lines.append(header)
    lines.append("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        lines.append(" | ".join(str(r.get(c) or "").ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _cmd_corpora(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.dir)
    if not root.is_dir():
        print(f"error: --dir {root} is not a directory", file=sys.stderr)
        return 2

    manifest: dict[str, str] = {}
    if args.manifest:
        manifest = _load_manifest(pathlib.Path(args.manifest))

    npz_paths = sorted(p for p in root.rglob("*.npz") if p.is_file())
    if not npz_paths:
        print(f"no .npz files under {root}")
        return 0

    rows: list[dict[str, Any]] = []
    any_error = False
    for p in npz_paths:
        row = _process_corpus(p, manifest=manifest, force=args.force, dry_run=args.dry_run)
        rows.append(row)
        if row["status"] == "error":
            any_error = True

    print(_format_corpus_report(rows))
    return 1 if any_error else 0


# ---------------------------------------------------------------------------
# Subcommand: model-variant
# ---------------------------------------------------------------------------

def _cmd_model_variant(args: argparse.Namespace) -> int:
    """Add 'model_variant: None' to any stamped ckpt missing the field.

    Idempotent — skips ckpts that already have the key OR lack metadata
    entirely (those are the A5 backfill's responsibility, not ours).
    """
    import torch

    root = pathlib.Path(args.dir)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    n_stamped = 0
    n_skipped_already = 0
    n_skipped_no_meta = 0
    n_failed = 0

    for path in sorted(root.rglob("*.pt")):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[ERR] {path}: load failed: {exc}")
            n_failed += 1
            continue

        if not isinstance(ck, dict):
            n_skipped_no_meta += 1
            continue

        meta = ck.get("metadata")
        if not isinstance(meta, dict):
            n_skipped_no_meta += 1
            continue

        # Only act on ckpts that are already stamped (encoding_name present).
        # Un-stamped ckpts are A5's responsibility.
        if not isinstance(meta.get("encoding_name"), str):
            n_skipped_no_meta += 1
            continue

        if "model_variant" in meta:
            n_skipped_already += 1
            continue

        meta["model_variant"] = None
        if args.dry_run:
            print(f"[DRY] would stamp model_variant=None on {path}")
        else:
            torch.save(ck, path)
            print(f"[OK]  stamped model_variant=None on {path}")
        n_stamped += 1

    print(
        f"\nSummary: stamped={n_stamped}, "
        f"skipped_already_has={n_skipped_already}, "
        f"skipped_no_meta={n_skipped_no_meta}, "
        f"failed={n_failed}"
    )
    return 0 if n_failed == 0 else 1


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="2026_05_09_stamp_artifact_metadata",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # -- checkpoints subcommand --
    p_ckpt = sub.add_parser(
        "checkpoints",
        help="Stamp encoding metadata into legacy .pt checkpoint files",
        description=(
            "For each .pt file under --dir, infer encoding from the state-dict "
            "shape and stamp metadata['encoding_name']. Idempotent."
        ),
    )
    p_ckpt.add_argument(
        "--dir",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints"),
        help="directory to scan recursively for .pt files (default: ./checkpoints)",
    )
    p_ckpt.add_argument(
        "--dry-run",
        action="store_true",
        help="report only; do not rewrite any files",
    )

    # -- corpora subcommand --
    p_corp = sub.add_parser(
        "corpora",
        help="Backfill .metadata.json sidecars for legacy .npz corpus files",
        description=(
            "For each .npz under --dir, infer encoding from filename rules or "
            "a --manifest YAML and write a .metadata.json sidecar. Idempotent."
        ),
    )
    p_corp.add_argument(
        "--dir",
        default="data/",
        help="root directory to scan (default: data/)",
    )
    p_corp.add_argument(
        "--manifest",
        default=None,
        help="optional YAML with encodings: {<filename>: <encoding>}",
    )
    p_corp.add_argument(
        "--dry-run",
        action="store_true",
        help="report only; do not write sidecars",
    )
    p_corp.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing sidecars",
    )

    # -- model-variant subcommand --
    p_mv = sub.add_parser(
        "model-variant",
        help="Backfill 'model_variant: None' on stamped ckpts missing the field",
        description=(
            "For each .pt file under --dir that already has metadata "
            "(encoding_name present) but lacks model_variant, stamps "
            "model_variant=None. Idempotent. Skips un-stamped ckpts "
            "(A5 backfill responsibility). §172 A10 close-out."
        ),
    )
    p_mv.add_argument(
        "--dir",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints"),
        help="directory to scan recursively for .pt files (default: ./checkpoints)",
    )
    p_mv.add_argument(
        "--dry-run",
        action="store_true",
        help="report only; do not rewrite any files",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "checkpoints":
        return _cmd_checkpoints(args)
    if args.subcommand == "corpora":
        return _cmd_corpora(args)
    if args.subcommand == "model-variant":
        return _cmd_model_variant(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
