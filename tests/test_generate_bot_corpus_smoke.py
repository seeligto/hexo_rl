"""§178 T1 smoke — exercise scripts/generate_bot_corpus.py with n_games=2.

Skips if `checkpoints/bootstrap_model_v6.pt` absent (CI / pre-anchor envs).
Lowers SealBot think budget to 0.05s so the wall stays under 60s on laptop.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_ANCHOR = REPO_ROOT / "checkpoints" / "bootstrap_model_v6.pt"


@pytest.mark.skipif(not _ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_generate_bot_corpus_smoke(tmp_path: Path) -> None:
    out_npz = tmp_path / "bot_corpus_smoke.npz"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_bot_corpus.py"),
        "--anchor", str(_ANCHOR),
        "--n-games", "2",
        "--max-plies", "30",
        "--out", str(out_npz),
        "--think-seconds", "0.05",
        "--anchor-n-sims", "8",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=90)
    # Allow exit 2 (no decisive games in 2-game smoke); inspect stdout/err.
    assert result.returncode in (0, 2), (
        f"unexpected rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    if result.returncode == 0:
        assert out_npz.exists(), "NPZ not written"
        sidecar = out_npz.with_name(out_npz.name + ".metadata.json")
        assert sidecar.exists(), "sidecar not written"
        meta = json.loads(sidecar.read_text())
        assert meta["encoding_name"] == "v6"
        assert meta["extra"]["anchor_sha256"]
        with np.load(out_npz) as data:
            assert data["states"].shape[0] > 0
            assert data["states"].shape[1:] == (8, 19, 19)
            assert data["policies"].shape[1] == 362
