"""Tests for `hexo_rl.bootstrap.corpus_io` — §172 A5.2.

Cover:
  - save_corpus writes sidecar w/ correct fields + sha matching disk file.
  - load_corpus round-trips arrays + metadata when expected_encoding matches.
  - load_corpus raises on sha tamper.
  - load_corpus emits DeprecationWarning when sidecar absent.
  - load_corpus raises on encoding mismatch.
  - load_corpus rejects future schema_version.

All tests use synthetic numpy under tmp_path; fast (<1s each).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from hexo_rl.bootstrap.corpus_io import (
    SCHEMA_VERSION,
    CorpusMetadataError,
    compute_npz_sha256,
    load_corpus,
    save_corpus,
)
from hexo_rl.bootstrap.corpus_io import _sidecar_path


def _synth_corpus(n: int = 7) -> dict[str, np.ndarray]:
    """Tiny v6w25-ish synthetic corpus; shapes don't need to be canonical."""
    return {
        "states":   np.zeros((n, 8, 5, 5), dtype=np.float32),
        "policies": np.full((n, 25), 1.0 / 25.0, dtype=np.float32),
        "outcomes": np.zeros((n,), dtype=np.float32),
        "weights":  np.ones((n,), dtype=np.float32),
    }


def test_save_corpus_writes_metadata(tmp_path):
    npz = tmp_path / "tiny_v6w25.npz"
    arrays = _synth_corpus(n=7)
    save_corpus(
        npz,
        arrays=arrays,
        encoding_name="v6w25",
        source_manifest="unit-test://synth",
        extra={"note": "synthetic"},
    )

    assert npz.exists(), "npz must be on disk"
    sidecar = _sidecar_path(npz)
    assert sidecar.exists(), f"sidecar missing: {sidecar}"

    meta = json.loads(sidecar.read_text())
    assert meta["encoding_name"] == "v6w25"
    assert meta["n_positions"] == 7
    assert meta["source_manifest"] == "unit-test://synth"
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["extra"] == {"note": "synthetic"}
    assert isinstance(meta["created_at"], str) and meta["created_at"].endswith("Z")
    assert isinstance(meta["created_by_commit"], str)

    # sha256 in sidecar must match a fresh computation against the npz.
    actual = compute_npz_sha256(npz)
    assert meta["sha256"] == actual
    assert len(actual) == 64


def test_load_corpus_validates_metadata(tmp_path):
    npz = tmp_path / "round_trip.npz"
    arrays = _synth_corpus(n=4)
    save_corpus(npz, arrays=arrays, encoding_name="v6w25")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> fail
        loaded_arrays, loaded_meta = load_corpus(npz, expected_encoding="v6w25")

    assert set(loaded_arrays) == set(arrays)
    for k in arrays:
        np.testing.assert_array_equal(loaded_arrays[k], arrays[k])
    assert loaded_meta["encoding_name"] == "v6w25"
    assert loaded_meta["n_positions"] == 4


def test_load_corpus_sha_mismatch_raises(tmp_path):
    npz = tmp_path / "tampered.npz"
    save_corpus(npz, arrays=_synth_corpus(n=3), encoding_name="v6w25")

    # Mutate the npz on disk: append a single byte. sha changes; sidecar
    # remains stale.
    with npz.open("ab") as f:
        f.write(b"X")

    with pytest.raises(CorpusMetadataError, match="sha256"):
        load_corpus(npz)


def test_load_corpus_legacy_warns(tmp_path):
    """Bare .npz (no sidecar) must load with DeprecationWarning."""
    npz = tmp_path / "legacy.npz"
    arrays = _synth_corpus(n=2)
    np.savez_compressed(npz, **arrays)
    assert not _sidecar_path(npz).exists()

    with pytest.warns(DeprecationWarning, match="legacy.npz"):
        loaded_arrays, loaded_meta = load_corpus(npz)
    assert loaded_meta == {}
    assert set(loaded_arrays) == set(arrays)


def test_load_corpus_encoding_mismatch_raises(tmp_path):
    npz = tmp_path / "wrong_enc.npz"
    save_corpus(npz, arrays=_synth_corpus(n=2), encoding_name="v6w25")

    with pytest.raises(CorpusMetadataError, match="encoding mismatch"):
        load_corpus(npz, expected_encoding="v8")


def test_load_corpus_future_schema_rejected(tmp_path):
    npz = tmp_path / "future.npz"
    save_corpus(npz, arrays=_synth_corpus(n=1), encoding_name="v6w25")

    sidecar = _sidecar_path(npz)
    meta = json.loads(sidecar.read_text())
    meta["schema_version"] = SCHEMA_VERSION + 1
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(CorpusMetadataError, match="schema_version"):
        load_corpus(npz)


def test_load_corpus_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does_not_exist.npz")
