"""S5 (GNN-integration program) — held-out corpus contamination gate.

`docs/registers/run3_corpus_manifest.md` §2 BURNED the prior held-out set
(the 1796-game "R2c" delta) for run3-lineage architecture reads — it was
ACCEPTED into the training corpus, so it is no longer out-of-sample. S5
minted a fresh held-out set (`scripts/mint_s5_heldout_corpus.py`,
`docs/registers/s5_heldout_manifest.md`) explicitly for future BC/value/
architecture reads ONLY. This module gates against the mistake that BURNED
R2c in the first place — a held-out set silently ending up on a training
path — by making it a hard, labeled error at TWO points:

  1. Resolver level (`hexo_rl.encoding.resolvers._assert_no_registry_overlap`,
     called at import time): the launch-pin registry (`_CORPUS_SHA_PINS`)
     and the held-out registry (`_HELDOUT_CORPUS_SHAS`) must never share a
     sha256 — this is a static invariant over the two registries themselves,
     independent of any particular load call.
  2. Load level (`hexo_rl.training.batch_assembly.load_pretrained_buffer`,
     via `assert_not_heldout_sha`): any TRAINING corpus load whose on-disk
     sha256 matches a registered held-out sha raises `ValueError` before any
     load work — unconditional (not gated on the target encoding having a
     `_CORPUS_SHA_PINS` entry), gated only by a cheap file-size pre-check so
     it costs nothing on the real (much larger) training corpora it doesn't
     apply to.

These tests use tiny synthetic corpora + monkeypatched registries (mirrors
`tests/test_corpus_sha_pin_gate.py`) so they run in milliseconds — no
dependency on the real, gitignored S5 held-out NPZ existing on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hexo_rl.bootstrap.corpus_io import compute_npz_sha256, save_corpus
from hexo_rl.encoding import resolvers as _resolvers
from hexo_rl.encoding.registry import EncodingRegistryError
from hexo_rl.training.batch_assembly import load_pretrained_buffer

_BOARD_SIZE = 19
_N_PLANES = 8  # v6 wire-format plane count
_N_ACTIONS = 362


def _synth_corpus_arrays(t: int = 3) -> dict[str, np.ndarray]:
    states = np.zeros((t, _N_PLANES, _BOARD_SIZE, _BOARD_SIZE), dtype=np.float16)
    policies = np.zeros((t, _N_ACTIONS), dtype=np.float32)
    policies[:, 0] = 1.0
    outcomes = np.zeros((t,), dtype=np.float32)
    return {"states": states, "policies": policies, "outcomes": outcomes}


def _write_corpus(path: Path, t: int = 3, encoding: str = "v6") -> str:
    """Write a tiny v6-shaped corpus + sidecar via save_corpus. Returns actual sha256."""
    save_corpus(path, arrays=_synth_corpus_arrays(t), encoding_name=encoding)
    return compute_npz_sha256(path)


def _register_heldout(monkeypatch, path: Path, sha: str) -> None:
    """Register *sha* (+ *path*'s on-disk size) as the sole held-out entry."""
    size = path.stat().st_size
    monkeypatch.setitem(_resolvers._HELDOUT_CORPUS_SHAS, "test_heldout", (sha, size))


def _load(path: Path, encoding: str = "v6"):
    mixing_cfg = {"pretrained_buffer_path": str(path)}
    config = {"seed": 0, "encoding": encoding}
    emitted: list[dict] = []
    return load_pretrained_buffer(
        mixing_cfg, config, emitted.append, buffer_size=0, buffer_capacity=0
    )


# ── direction 1: held-out sha loaded on an UNPINNED encoding's training path
#    (proves the gate is unconditional, not gated on `_CORPUS_SHA_PINS`) ────

def test_heldout_sha_on_unpinned_encoding_raises(tmp_path, monkeypatch):
    path = tmp_path / "heldout.npz"
    actual_sha = _write_corpus(path)
    _register_heldout(monkeypatch, path, actual_sha)

    # Real registry has no pin for "v6" — this must STILL raise; the
    # held-out gate does not depend on a launch pin existing.
    with pytest.raises(ValueError) as excinfo:
        _load(path, encoding="v6")

    msg = str(excinfo.value)
    assert "HELD-OUT" in msg
    assert "test_heldout" in msg
    assert str(path) in msg


# ── direction 2: held-out sha loaded where a PINNED encoding's corpus is
#    expected — must raise the held-out-specific error, not a generic sha-
#    pin mismatch (fires BEFORE the pin-mismatch check) ─────────────────────

def test_heldout_sha_on_pinned_encoding_raises_heldout_error_not_pin_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "heldout.npz"
    actual_sha = _write_corpus(path)
    _register_heldout(monkeypatch, path, actual_sha)
    # Pin some OTHER sha for "v6" (the held-out file does not match it).
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", "f" * 64)

    with pytest.raises(ValueError) as excinfo:
        _load(path, encoding="v6")

    msg = str(excinfo.value)
    assert "HELD-OUT" in msg, f"expected the held-out-specific message, got: {msg}"


# ── non-overlap happy path: a normal pinned corpus load is unaffected by an
#    UNRELATED held-out entry registered alongside it ───────────────────────

def test_pinned_corpus_loads_normally_with_unrelated_heldout_entry_registered(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.npz"
    corpus_sha = _write_corpus(corpus_path, t=3)
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", corpus_sha)

    # A held-out entry for a DIFFERENT (non-colliding) sha/size — must not
    # interfere with the legitimate pinned load.
    heldout_path = tmp_path / "heldout.npz"
    heldout_sha = _write_corpus(heldout_path, t=7)  # different t -> different size
    assert heldout_path.stat().st_size != corpus_path.stat().st_size
    _register_heldout(monkeypatch, heldout_path, heldout_sha)

    buf = _load(corpus_path, encoding="v6")

    assert buf is not None


def test_unpinned_unrelated_corpus_unaffected_by_heldout_registry(tmp_path, monkeypatch):
    """A normal unpinned load whose file size doesn't match any registered
    held-out size must not even attempt the sha256 stream for the held-out
    check (cheap-path proof, mirrors FIX-1's "sha computed once" doctrine)."""
    path = tmp_path / "corpus.npz"
    _write_corpus(path, t=3)

    # Register a held-out entry with a size that provably does NOT match.
    monkeypatch.setitem(
        _resolvers._HELDOUT_CORPUS_SHAS, "unrelated", ("0" * 64, path.stat().st_size + 12345)
    )

    from hexo_rl.bootstrap import corpus_io as _cio
    from hexo_rl.training import batch_assembly as _ba

    calls: list[Path] = []
    orig = _cio.compute_npz_sha256

    def _counting(p):
        calls.append(Path(p))
        return orig(p)

    monkeypatch.setattr(_cio, "compute_npz_sha256", _counting)
    monkeypatch.setattr(_ba, "compute_npz_sha256", _counting)

    buf = _load(path, encoding="v6")  # unpinned -> pin-check also skips hashing

    assert buf is not None
    assert calls == [], "size mismatch must short-circuit before any sha256 stream"


# ── resolver-level static overlap invariant ──────────────────────────────────

def test_registry_overlap_raises(monkeypatch):
    shared_sha = "a" * 64
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", shared_sha)
    monkeypatch.setitem(_resolvers._HELDOUT_CORPUS_SHAS, "bad_entry", (shared_sha, 123))

    with pytest.raises(EncodingRegistryError) as excinfo:
        _resolvers._assert_no_registry_overlap()

    assert shared_sha in str(excinfo.value)


def test_registry_no_overlap_happy_path(monkeypatch):
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", "a" * 64)
    monkeypatch.setitem(_resolvers._HELDOUT_CORPUS_SHAS, "held", ("b" * 64, 123))

    _resolvers._assert_no_registry_overlap()  # must not raise


def test_real_registries_have_no_overlap():
    """Regression guard over the REAL production registries (not
    monkeypatched) — the S5 held-out sha must never coincide with any
    `_CORPUS_SHA_PINS` entry (e.g. run3's v6_live2_ls launch pin)."""
    _resolvers._assert_no_registry_overlap()  # already ran at import; re-check explicitly
    assert set(_resolvers._CORPUS_SHA_PINS.values()).isdisjoint(_resolvers.held_out_shas())
