"""§6 cross-table tests — INV-1..6 from design §11.6."""
import json
import hashlib
from pathlib import Path
import pytest


def _make_ckpt(tmp_path, name, encoding_name, corpus_sha=None):
    """Synthesize a torch ckpt with metadata block."""
    import torch
    p = tmp_path / name
    payload = {"step": 0, "model_state": {}, "config": {}}
    if encoding_name is not None:
        payload["metadata"] = {
            "encoding_name": encoding_name,
            "corpus_sha256": corpus_sha,
            "schema_version": 1,
        }
    torch.save(payload, p)
    return p


def _make_corpus(tmp_path, name, encoding_name, *, n_pos=10):
    """Synthesize a corpus npz + sidecar."""
    import numpy as np
    p = tmp_path / name
    np.savez(p, positions=np.zeros((n_pos, 8, 19, 19), dtype=np.float32))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    sidecar = p.with_suffix(p.suffix + ".metadata.json")
    sidecar.write_text(json.dumps({
        "encoding_name": encoding_name,
        "sha256": sha,
        "n_positions": n_pos,
        "schema_version": 1,
    }))
    return p, sha


def test_inv5_clean_match(tmp_path):
    """INV-5: ckpt.encoding == corpus.encoding via shared sha → info OK."""
    from hexo_rl.encoding.audit import audit
    corpus, sha = _make_corpus(tmp_path, "v6.npz", "v6")
    _make_ckpt(tmp_path, "ck.pt", "v6", corpus_sha=sha)
    report = audit(checkpoints_dir=tmp_path, corpora_dir=tmp_path)
    cross = [f for f in report.findings if f.section == "§6"]
    assert any("ck.pt" in f.message and "v6" in f.message for f in cross)
    assert all(f.severity != "error" for f in cross)


def test_inv1_enc_mismatch(tmp_path):
    """INV-1: ckpt.encoding != corpus.encoding via shared sha → error."""
    from hexo_rl.encoding.audit import audit
    corpus, sha = _make_corpus(tmp_path, "v6.npz", "v6")
    _make_ckpt(tmp_path, "ck.pt", "v6w25", corpus_sha=sha)
    report = audit(checkpoints_dir=tmp_path, corpora_dir=tmp_path)
    errs = [f for f in report.findings if f.section == "§6" and f.severity == "error"]
    assert any("v6w25" in f.message and "v6" in f.message for f in errs)


def test_inv2_orphan_sha(tmp_path):
    """INV-2: ckpt.corpus_sha references no known corpus → error."""
    from hexo_rl.encoding.audit import audit
    fake_sha = "0" * 64
    _make_ckpt(tmp_path, "ck.pt", "v6", corpus_sha=fake_sha)
    report = audit(checkpoints_dir=tmp_path, corpora_dir=tmp_path)
    errs = [f for f in report.findings if f.section == "§6" and f.severity == "error"]
    assert any("matches no corpus" in f.message.lower() or "orphan" in f.message.lower() for f in errs)


def test_inv3_no_corpus_sha(tmp_path):
    """INV-3: ckpt has metadata but corpus_sha256 is None → warn."""
    from hexo_rl.encoding.audit import audit
    _make_ckpt(tmp_path, "ck.pt", "v6", corpus_sha=None)
    report = audit(checkpoints_dir=tmp_path, corpora_dir=tmp_path)
    warns = [f for f in report.findings if f.section == "§6" and f.severity == "warn"]
    assert any("corpus_sha256" in f.message.lower() for f in warns)


def test_inv4_no_metadata(tmp_path):
    """INV-4: ckpt has no metadata at all → warn."""
    from hexo_rl.encoding.audit import audit
    _make_ckpt(tmp_path, "ck.pt", encoding_name=None)
    report = audit(checkpoints_dir=tmp_path, corpora_dir=tmp_path)
    warns = [f for f in report.findings if f.section == "§6"]
    assert any("no metadata" in f.message.lower() for f in warns)


def test_inv6_orphan_corpus(tmp_path):
    """INV-6: corpus sha unreferenced by any ckpt → info."""
    from hexo_rl.encoding.audit import audit
    _make_corpus(tmp_path, "orphan.npz", "v6")
    report = audit(checkpoints_dir=tmp_path, corpora_dir=tmp_path)
    section6 = [f for f in report.findings if f.section == "§6"]
    assert any("orphan" in f.message.lower() or "unused" in f.message.lower() for f in section6)


def test_skip_when_corpora_empty(tmp_path):
    """When corpora_dir is empty AND ckpts have corpus_sha256, emit one section warn + skip per-row."""
    from hexo_rl.encoding.audit import audit
    _make_ckpt(tmp_path, "ck.pt", "v6", corpus_sha="0" * 64)
    empty = tmp_path / "empty"
    empty.mkdir()
    report = audit(checkpoints_dir=tmp_path, corpora_dir=empty)
    skip_findings = [f for f in report.findings if f.section == "§6" and "skipped" in f.message.lower()]
    assert len(skip_findings) >= 1
