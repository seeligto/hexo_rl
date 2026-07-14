"""WP0.4 — launch-pinned corpus sha assertion in `load_pretrained_buffer`.

`load_pretrained_buffer` (`hexo_rl/training/batch_assembly.py`) previously
validated ONLY the corpus's plane count against the active encoding. A
stale/re-exported NPZ of the same shape at the same path passed silently —
run3's launch corpus (`data/bootstrap_corpus_v6_live2_ls.npz`, sha
`3813edc2…`) must be byte-identical on both hosts (docs/registers/
run3_corpus_manifest.md), and a shape-preserving re-export would otherwise
go undetected until a strength regression showed up GPU-days later.

Fix: `hexo_rl.encoding.resolvers._CORPUS_SHA_PINS` + `resolve_corpus_sha_pin`
register a launch-critical sha per encoding name; `load_pretrained_buffer`
recomputes the on-disk file's sha256 (streaming, not trusting the sidecar)
and raises `ValueError` on mismatch BEFORE any load work. It also calls
`hexo_rl.bootstrap.corpus_io.load_corpus(..., expected_encoding=...)` as a
belt-and-braces check — this catches a stale/desynced sidecar even when the
sha pin itself still matches (the pin proves "right corpus"; the sidecar
only proves internal npz<->sidecar consistency).

These tests use tiny synthetic corpora (not the real 2.65 GB launch NPZ) and
monkeypatch `_CORPUS_SHA_PINS` so they run in milliseconds. The real-file
end-to-end check lives in the integration test at
`tests/test_run3_corpus_launch_path.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hexo_rl.bootstrap.corpus_io import CorpusMetadataError, compute_npz_sha256, save_corpus
from hexo_rl.encoding import resolvers as _resolvers
from hexo_rl.training.batch_assembly import load_pretrained_buffer

_BOARD_SIZE = 19
_N_PLANES = 8  # v6 wire-format plane count (registry: lookup("v6").n_planes)
_N_ACTIONS = 362


def _synth_corpus_arrays(t: int = 3) -> dict[str, np.ndarray]:
    states = np.zeros((t, _N_PLANES, _BOARD_SIZE, _BOARD_SIZE), dtype=np.float16)
    policies = np.zeros((t, _N_ACTIONS), dtype=np.float32)
    policies[:, 0] = 1.0
    outcomes = np.zeros((t,), dtype=np.float32)
    return {"states": states, "policies": policies, "outcomes": outcomes}


def _write_pinned_corpus(path: Path, t: int = 3) -> str:
    """Write a tiny v6-shaped corpus + sidecar via save_corpus. Returns actual sha256."""
    save_corpus(path, arrays=_synth_corpus_arrays(t), encoding_name="v6")
    return compute_npz_sha256(path)


def _load(path: Path, encoding: str = "v6"):
    mixing_cfg = {"pretrained_buffer_path": str(path)}
    config = {"seed": 0, "encoding": encoding}
    emitted: list[dict] = []
    return load_pretrained_buffer(
        mixing_cfg, config, emitted.append, buffer_size=0, buffer_capacity=0
    )


# ── match passes ───────────────────────────────────────────────────────────

def test_sha_pin_match_passes(tmp_path, monkeypatch):
    path = tmp_path / "corpus.npz"
    actual_sha = _write_pinned_corpus(path)
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", actual_sha)

    buf = _load(path)

    assert buf is not None


# ── mismatch raises ──────────────────────────────────────────────────────────

def test_sha_pin_mismatch_raises(tmp_path, monkeypatch):
    path = tmp_path / "corpus.npz"
    _write_pinned_corpus(path)
    wrong_sha = "0" * 64
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", wrong_sha)

    with pytest.raises(ValueError) as excinfo:
        _load(path)

    msg = str(excinfo.value)
    assert str(path) in msg
    assert wrong_sha[:12] in msg
    assert "v6" in msg


def test_sha_pin_mismatch_names_actual_sha_in_message(tmp_path, monkeypatch):
    path = tmp_path / "corpus.npz"
    actual_sha = _write_pinned_corpus(path)
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", "f" * 64)

    with pytest.raises(ValueError) as excinfo:
        _load(path)

    assert actual_sha[:12] in str(excinfo.value)


# ── sidecar desync raises (belt-and-braces load_corpus call) ────────────────

def test_sidecar_desync_raises_even_when_pin_matches(tmp_path, monkeypatch):
    path = tmp_path / "corpus.npz"
    actual_sha = _write_pinned_corpus(path)
    # Pin matches the REAL on-disk bytes...
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", actual_sha)

    # ...but the sidecar's declared sha has gone stale (simulates someone
    # editing/replacing the sidecar without re-running save_corpus).
    sidecar_path = path.with_name(path.name + ".metadata.json")
    meta = json.loads(sidecar_path.read_text())
    meta["sha256"] = "a" * 64
    sidecar_path.write_text(json.dumps(meta))

    with pytest.raises(CorpusMetadataError):
        _load(path)


# ── unpinned encoding is unaffected (back-compat) ───────────────────────────

def test_unpinned_encoding_skips_sha_check(tmp_path):
    """No entry in `_CORPUS_SHA_PINS` for this encoding → no sha computed,
    no sidecar required (plain np.savez like every pre-WP0.4 corpus)."""
    path = tmp_path / "corpus.npz"
    arrays = _synth_corpus_arrays(t=2)
    np.savez(path, **arrays)  # no sidecar written at all

    # Real registry has no pin for "v6" by default (only v6_live2_ls is
    # launch-pinned) — this must succeed exactly as before WP0.4.
    buf = _load(path)

    assert buf is not None
