"""Corpus metadata sidecar — save/load with sha256 + encoding validation.

Authored §172 Phase A5.2 (2026-05-09). Implements the corpus side of the
encoding registry contract per `docs/designs/encoding_registry_design.md`
§9. Sidecar `<path>.metadata.json` is written next to each `.npz`
(separate file — keeps `.npz` archive bytes immutable for sha-stability).

Public API:
  - `save_corpus(path, *, arrays, encoding_name, ...)` writes npz +
    sidecar.
  - `load_corpus(path, *, expected_encoding=None)` validates sidecar
    sha256 + encoding match; raises `CorpusMetadataError` on mismatch;
    emits `DeprecationWarning` if sidecar absent.
  - `validate_corpus_sidecar(path, *, expected_encoding=None,
    actual_sha=None)` — same sidecar validation as `load_corpus` (schema
    version, sha256, encoding match) WITHOUT the eager full-array load;
    accepts an already-computed `actual_sha` to avoid re-streaming a large
    npz a caller has already hashed (WP0.4 fix-wave FIX-1).
  - `CorpusMetadataError` exception type for all metadata failures.
  - `compute_npz_sha256(path)` exposed for backfill script reuse.

Schema version: 1 (locked at A2). Future bumps: write `schema_version=N`,
load rejects sidecars with `schema_version > <known>`.

Integration with `pretrain.load_corpus` and `dataset_v*.py` deferred to
A4-followup. This module ships standalone alongside the existing
loaders.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import subprocess
import warnings
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
_SIDECAR_SUFFIX = ".metadata.json"
_HASH_CHUNK = 1 << 20  # 1 MiB


class CorpusMetadataError(Exception):
    """Raised on sidecar parse failure, sha mismatch, or encoding mismatch."""


def _sidecar_path(npz_path: pathlib.Path) -> pathlib.Path:
    """Return `<npz_path>.metadata.json` next to the corpus file."""
    return npz_path.with_name(npz_path.name + _SIDECAR_SUFFIX)


def compute_npz_sha256(path: str | pathlib.Path) -> str:
    """Streaming sha256 of the file at `path`. Hex digest."""
    p = pathlib.Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_git_commit() -> str:
    """git rev-parse HEAD, or 'unknown' if we're outside a repo / git absent."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        return out.decode("ascii", errors="replace").strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _utc_now_iso() -> str:
    """ISO 8601 UTC w/ Z suffix; second-resolution to match design example."""
    now = _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0)
    # isoformat() yields '+00:00' suffix; design doc uses 'Z'.
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _infer_n_positions(arrays: dict[str, np.ndarray]) -> int:
    """Leading dim of the first array. Matches existing pretrain.py contract.

    Raises CorpusMetadataError if `arrays` is empty or the first array is
    0-dim.
    """
    if not arrays:
        raise CorpusMetadataError("save_corpus: arrays dict is empty")
    # Prefer 'states' (canonical pretrain.py key) if present, else first
    # insertion-ordered array.
    if "states" in arrays:
        first = arrays["states"]
    else:
        first = next(iter(arrays.values()))
    if first.ndim == 0:
        raise CorpusMetadataError(
            "save_corpus: cannot infer n_positions from 0-dim array; "
            "pass arrays with a leading position-axis"
        )
    return int(first.shape[0])


def save_corpus(
    path: str | pathlib.Path,
    *,
    arrays: dict[str, np.ndarray],
    encoding_name: str,
    source_manifest: str | None = None,
    extra: dict[str, Any] | None = None,
    compress: bool = True,
) -> None:
    """Save `arrays` as npz at `path`; write sidecar metadata.

    Sidecar = `<path>.metadata.json` with:
      encoding_name, sha256, n_positions, source_manifest, created_at,
      created_by_commit, schema_version, extra.

    `n_positions` is the leading dim of `arrays["states"]` if present,
    else the first inserted array's leading dim.

    `compress=True` (default) uses `np.savez_compressed`; pass `False` to
    write an uncompressed archive (lets downstream loaders use
    `np.load(mmap_mode='r')` for near-zero RAM at startup — see
    `scripts/export_corpus_npz.py`).
    """
    npz_path = pathlib.Path(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    n_positions = _infer_n_positions(arrays)

    if compress:
        np.savez_compressed(npz_path, **arrays)
    else:
        np.savez(npz_path, **arrays)
    # np.savez_compressed appends `.npz` if extension absent — match its
    # actual on-disk path so sha + sidecar line up.
    if not npz_path.exists() and npz_path.with_suffix(".npz").exists():
        npz_path = npz_path.with_suffix(".npz")

    sha = compute_npz_sha256(npz_path)
    metadata = {
        "encoding_name": encoding_name,
        "sha256": sha,
        "n_positions": n_positions,
        "source_manifest": source_manifest,
        "created_at": _utc_now_iso(),
        "created_by_commit": _resolve_git_commit(),
        "schema_version": SCHEMA_VERSION,
        "extra": dict(extra) if extra else {},
    }
    sidecar = _sidecar_path(npz_path)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _load_arrays(npz_path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read all arrays from an npz into a dict (eager, not mmap).

    Eager copy keeps caller free to use the file post-close (sidecar
    sha-check needs the file open separately). Callers that want mmap
    should call `np.load(path, mmap_mode='r')` directly — `load_corpus`
    is the validation path, not the streaming path.
    """
    out: dict[str, np.ndarray] = {}
    with np.load(npz_path) as data:
        for key in data.files:
            out[key] = data[key]
    return out


def validate_corpus_sidecar(
    path: str | pathlib.Path,
    *,
    expected_encoding: str | None = None,
    actual_sha: str | None = None,
) -> dict[str, Any]:
    """Validate a corpus npz's sidecar WITHOUT loading the array payload.

    Same validation semantics as `load_corpus` (schema_version, sha256,
    encoding_name) minus the eager `_load_arrays` copy — for a caller that
    only needs "is this the right, non-desynced corpus", not the arrays
    themselves (e.g. `load_pretrained_buffer`, which mmaps separately).

    Args:
        path: npz path.
        expected_encoding: raise if sidecar's `encoding_name` differs.
        actual_sha: an already-computed sha256 of `path`, reused instead of
            re-streaming the file a second time. Computed fresh (one
            stream) when omitted.

    Behaviour:
      - Sidecar present:
          * parse json; reject `schema_version > SCHEMA_VERSION`.
          * compare `actual_sha` (or a fresh streaming sha256 of `path`)
            against the sidecar's declared `sha256`; raise on mismatch.
          * if `expected_encoding` given and sidecar's `encoding_name`
            differs, raise.
          * return metadata.
      - Sidecar absent:
          * emit DeprecationWarning naming the path + backfill script.
          * return {}.

    Raises:
      CorpusMetadataError on any validation failure.
      FileNotFoundError if `path` itself is missing.
    """
    npz_path = pathlib.Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"corpus npz not found: {npz_path}")

    sidecar = _sidecar_path(npz_path)
    if not sidecar.exists():
        warnings.warn(
            f"{npz_path} has no metadata sidecar; "
            f"run python -m scripts.migrations.2026_05_09_stamp_artifact_metadata corpora",
            DeprecationWarning,
            stacklevel=2,
        )
        return {}

    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusMetadataError(
            f"sidecar parse failed: {sidecar} ({exc})"
        ) from exc
    if not isinstance(metadata, dict):
        raise CorpusMetadataError(
            f"sidecar root must be a JSON object: {sidecar}"
        )

    sv = metadata.get("schema_version")
    if not isinstance(sv, int) or sv > SCHEMA_VERSION:
        raise CorpusMetadataError(
            f"sidecar schema_version={sv!r} > supported {SCHEMA_VERSION}: "
            f"{sidecar}; upgrade hexo_rl.bootstrap.corpus_io"
        )

    declared_sha = metadata.get("sha256")
    if not isinstance(declared_sha, str) or not declared_sha:
        raise CorpusMetadataError(
            f"sidecar missing 'sha256' field: {sidecar}"
        )
    resolved_actual_sha = actual_sha if actual_sha is not None else compute_npz_sha256(npz_path)
    if resolved_actual_sha != declared_sha:
        raise CorpusMetadataError(
            f"sha256 mismatch: corpus modified or sidecar stale "
            f"(declared {declared_sha[:12]}…, actual {resolved_actual_sha[:12]}…) "
            f"for {npz_path}"
        )

    if expected_encoding is not None:
        sidecar_enc = metadata.get("encoding_name")
        if sidecar_enc != expected_encoding:
            raise CorpusMetadataError(
                f"encoding mismatch: sidecar says {sidecar_enc!r}, "
                f"expected {expected_encoding!r} ({npz_path})"
            )

    return metadata


def load_corpus(
    path: str | pathlib.Path,
    *,
    expected_encoding: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load `.npz` arrays + validate sidecar metadata.

    Behaviour:
      - Sidecar present:
          * parse json; reject `schema_version > SCHEMA_VERSION`.
          * recompute sha256 of `path`; raise on mismatch.
          * if `expected_encoding` given and sidecar's `encoding_name`
            differs, raise.
          * return (arrays, metadata).
      - Sidecar absent:
          * emit DeprecationWarning naming the path + backfill script.
          * return (arrays, {}).

    Raises:
      CorpusMetadataError on any validation failure.
      FileNotFoundError if `path` itself is missing.
    """
    npz_path = pathlib.Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"corpus npz not found: {npz_path}")

    arrays = _load_arrays(npz_path)
    metadata = validate_corpus_sidecar(npz_path, expected_encoding=expected_encoding)
    return arrays, metadata


__all__ = [
    "CorpusMetadataError",
    "SCHEMA_VERSION",
    "compute_npz_sha256",
    "load_corpus",
    "save_corpus",
    "validate_corpus_sidecar",
]
