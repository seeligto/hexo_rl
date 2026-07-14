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


# ── FIX-1: sha computed once, sidecar validated without a full array load ───

def test_sha_computed_once_no_full_array_load(tmp_path, monkeypatch):
    """WP0.4 fix-wave FIX-1 (review Important, batch_assembly.py:203): the
    belt-and-braces sidecar check must reuse the sha already streamed for
    the pin comparison — no second sha256 stream, and no eager full-NPZ
    array materialization via corpus_io._load_arrays (the sidecar check
    only needs the tiny sidecar JSON + the already-known actual sha)."""
    path = tmp_path / "corpus.npz"
    actual_sha = _write_pinned_corpus(path)
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", actual_sha)

    from hexo_rl.bootstrap import corpus_io as _cio
    from hexo_rl.training import batch_assembly as _ba

    _orig_sha = _cio.compute_npz_sha256
    _orig_load_arrays = _cio._load_arrays
    sha_calls: list[Path] = []
    load_arrays_calls: list[Path] = []

    def _counting_sha(p):
        sha_calls.append(Path(p))
        return _orig_sha(p)

    def _counting_load_arrays(npz_path):
        load_arrays_calls.append(Path(npz_path))
        return _orig_load_arrays(npz_path)

    # Patch both the defining module AND batch_assembly's `from ... import`
    # binding — either call site would otherwise dodge the counter.
    monkeypatch.setattr(_cio, "compute_npz_sha256", _counting_sha)
    monkeypatch.setattr(_ba, "compute_npz_sha256", _counting_sha)
    monkeypatch.setattr(_cio, "_load_arrays", _counting_load_arrays)

    buf = _load(path)

    assert buf is not None
    assert len(sha_calls) == 1, (
        f"compute_npz_sha256 called {len(sha_calls)}x, expected 1 (reuse the "
        f"streamed sha across the pin check AND the sidecar check)"
    )
    assert load_arrays_calls == [], (
        "sidecar validation must not eager-load the full NPZ array payload "
        "(corpus_io._load_arrays must not be called)"
    )


# ── FIX-2: pinned encoding + missing/unresolved corpus hard-fails ───────────

def test_pinned_encoding_missing_corpus_raises(tmp_path, monkeypatch):
    """WP0.4 fix-wave FIX-2 (review Minor / plan-law violation,
    batch_assembly.py:165-176): a PIN-REGISTERED encoding declares its
    corpus launch-critical — a missing file must hard ValueError, not warn
    + silently train corpus-less."""
    missing_path = tmp_path / "does_not_exist.npz"
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", "0" * 64)

    with pytest.raises(ValueError) as excinfo:
        _load(missing_path)

    msg = str(excinfo.value)
    assert "v6" in msg
    assert str(missing_path) in msg


def test_pinned_encoding_unexpanded_auto_raises():
    """FIX-2: an unexpanded "<auto>" literal (e.g. expand_auto_paths never
    ran, or the key moved under a different config section) must hard-fail
    for a pinned encoding rather than silently skip the gate."""
    mixing_cfg = {"pretrained_buffer_path": "<auto>"}
    config = {"seed": 0, "encoding": "v6_live2_ls"}  # real registry pin
    emitted: list[dict] = []

    with pytest.raises(ValueError) as excinfo:
        load_pretrained_buffer(
            mixing_cfg, config, emitted.append, buffer_size=0, buffer_capacity=0
        )

    assert "v6_live2_ls" in str(excinfo.value)


def test_unpinned_encoding_missing_corpus_still_warns_not_raises(tmp_path):
    """Back-compat: an unpinned encoding's missing corpus keeps the
    pre-WP0.4-fix-wave warn+None behavior (no launch-critical guarantee to
    enforce)."""
    missing_path = tmp_path / "does_not_exist.npz"

    buf = _load(missing_path)  # default encoding "v6" — unpinned in real registry

    assert buf is None


# ── FIX-3: <auto>-resolved corpus for an unpinned encoding hard-fails ───────

def test_auto_resolved_corpus_without_pin_raises(tmp_path):
    """WP0.4 fix-wave FIX-3 (red-team F2, MEDIUM): a corpus path that came
    from "<auto>" resolution (flagged by `expand_auto_paths`) for an
    encoding with NO registered sha pin must hard ValueError — <auto> is
    exactly the mechanism that lets a one-char encoding typo silently
    redirect to a different, unpinned, possibly host-divergent corpus past
    the plane-count check."""
    path = tmp_path / "corpus.npz"
    _write_pinned_corpus(path)  # real file+sidecar; must still raise — file
                                 # existing/matching is irrelevant to this gate
    mixing_cfg = {
        "pretrained_buffer_path": str(path),
        "_pretrained_buffer_path_auto_resolved": True,
    }
    config = {"seed": 0, "encoding": "v6"}  # real registry: "v6" has no pin
    emitted: list[dict] = []

    with pytest.raises(ValueError) as excinfo:
        load_pretrained_buffer(
            mixing_cfg, config, emitted.append, buffer_size=0, buffer_capacity=0
        )

    msg = str(excinfo.value)
    assert "v6" in msg
    assert "auto" in msg.lower()


def test_auto_resolved_corpus_with_pin_does_not_raise(tmp_path, monkeypatch):
    """<auto>-resolved + PINNED is exactly run3's declared launch path — the
    FIX-3 gate must NOT fire when a pin covers the encoding."""
    path = tmp_path / "corpus.npz"
    actual_sha = _write_pinned_corpus(path)
    monkeypatch.setitem(_resolvers._CORPUS_SHA_PINS, "v6", actual_sha)
    mixing_cfg = {
        "pretrained_buffer_path": str(path),
        "_pretrained_buffer_path_auto_resolved": True,
    }
    config = {"seed": 0, "encoding": "v6"}
    emitted: list[dict] = []

    buf = load_pretrained_buffer(
        mixing_cfg, config, emitted.append, buffer_size=0, buffer_capacity=0
    )

    assert buf is not None


def test_hardcoded_path_without_pin_is_unaffected_by_fix3(tmp_path):
    """Hardcoded explicit paths (the ~30 legacy variants) never go through
    `expand_auto_paths`'s "<auto>" branch, so the auto-resolved flag is
    never set — FIX-3 must not fire for them even though they're unpinned."""
    path = tmp_path / "corpus.npz"
    _write_pinned_corpus(path)

    buf = _load(path)  # _load() constructs mixing_cfg directly — no auto flag

    assert buf is not None
