"""Backfill `<path>.metadata.json` sidecars for legacy `.npz` corpora.

§172 Phase A5.2 — one-shot helper to bring pre-registry corpora into the
sidecar regime defined by `docs/designs/encoding_registry_design.md` §9.

Usage:
    python scripts/backfill_corpus_metadata.py [--dir data/]
                                               [--manifest manifest.yaml]
                                               [--dry-run]
                                               [--force]

Filename inference (when no manifest entry):
    bootstrap_corpus_v6w25.npz   -> v6w25
    bootstrap_corpus_v8.npz      -> v8
    bootstrap_corpus_v8_canvas_realness.npz -> v8_canvas_realness
    adversarial_corpus_v8.npz    -> v8
    bootstrap_corpus.npz         -> v6  (default per design §9 backcompat)
    bootstrap_corpus_v7full.npz  -> v7full
    bootstrap_corpus_v7.npz      -> v7
    *_sweep_*.npz                -> SKIP (no canonical encoding; manifest
                                  required to override).

Manifest YAML schema (optional):
    encodings:
      bootstrap_corpus.npz: v6
      bootstrap_corpus_sweep_2ch.npz: v6
      ...

Manifest entries override filename inference. Files matching no rule and
absent from manifest are reported as `no-manifest-required-skip`.

Idempotent: existing sidecar present -> skipped (status `current`)
unless `--force`. Inferred encoding is verified against the registry
before write — unknown names are rejected up-front.

Not auto-run: invoke explicitly. Tests must not call this.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

# Make repo root importable when run as a script.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from hexo_rl.bootstrap.corpus_io import (  # noqa: E402
    SCHEMA_VERSION,
    compute_npz_sha256,
)
from hexo_rl.bootstrap.corpus_io import (  # noqa: E402
    _resolve_git_commit,
    _sidecar_path,
    _utc_now_iso,
)
from hexo_rl.encoding import lookup as _registry_lookup  # noqa: E402
from hexo_rl.encoding.registry import EncodingRegistryError  # noqa: E402


# Filename → encoding inference table. Order matters: longest match
# first (sweeping is the rare exception that requires a manifest).
_FILENAME_RULES: list[tuple[str, str | None]] = [
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


def _infer_encoding(stem: str) -> str | None:
    for prefix, enc in _FILENAME_RULES:
        if stem == prefix or stem.startswith(prefix + "_") or stem.startswith(prefix + "-"):
            return enc
        if stem == prefix:
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
    """Lazily read leading dim of 'states' (or first array) from `npz_path`.

    Mmap so we don't pay full read cost.
    """
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


def _process_one(
    npz_path: pathlib.Path,
    *,
    manifest: dict[str, str],
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    sidecar = _sidecar_path(npz_path)
    name = npz_path.name

    # Resolve encoding: manifest > filename rule.
    if name in manifest:
        encoding = manifest[name]
        source = "manifest"
    else:
        encoding = _infer_encoding(npz_path.stem)
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
        "status": "backfilled" if not (sidecar.exists() and force) else "rewritten",
        "encoding": encoding,
        "sha256": _truncate_sha(sha),
        "notes": f"wrote {sidecar.name}",
    }


def _format_report(rows: list[dict[str, Any]]) -> str:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill .metadata.json sidecars for legacy corpora",
    )
    parser.add_argument("--dir", default="data/",
                        help="root directory to scan (default: data/)")
    parser.add_argument("--manifest", default=None,
                        help="optional YAML w/ encodings: {<filename>: <enc>}")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only; do not write sidecars")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing sidecars")
    args = parser.parse_args(argv)

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
        row = _process_one(p, manifest=manifest, force=args.force, dry_run=args.dry_run)
        rows.append(row)
        if row["status"] == "error":
            any_error = True

    print(_format_report(rows))
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
