"""Regression tests for scripts/compute_threat_pos_weight.py.

Covers:
  H-001 — 7-tuple unpack and suppressed-exception fix
  E-003 — HEXB version pin removed; v5 buffers now accepted
  E-006 — probe thresholds raised to match CLAUDE.md §91

Run: pytest tests/test_compute_threat_pos_weight.py -xvs
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Import the script as a module without executing __main__
# ---------------------------------------------------------------------------

def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compute = _load_script("compute_threat_pos_weight")
probe   = _load_script("probe_threat_logits")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHANNELS       = 8
N_CHAIN_PLANES = 6
BOARD_SIZE     = 19
N_ACTIONS      = BOARD_SIZE * BOARD_SIZE + 1   # 362
AUX_STRIDE     = BOARD_SIZE * BOARD_SIZE       # 361


def _random_states(t: int) -> np.ndarray:
    return np.zeros((t, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)


def _zero_chains(t: int) -> np.ndarray:
    return np.zeros((t, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)


def _uniform_policies(t: int) -> np.ndarray:
    p = np.ones((t, N_ACTIONS), dtype=np.float32)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _outcomes(t: int, val: float = 1.0) -> np.ndarray:
    return np.full(t, val, dtype=np.float32)


def _empty_own(t: int) -> np.ndarray:
    return np.ones((t, AUX_STRIDE), dtype=np.uint8)  # 1 = empty


def _wl_with_k_positive(t: int, k: int = 6) -> np.ndarray:
    """Winning-line mask: exactly k cells set per row."""
    arr = np.zeros((t, AUX_STRIDE), dtype=np.uint8)
    arr[:, :k] = 1
    return arr


def _build_v5_buffer(n_rows: int, k_positive: int = 6) -> "ReplayBuffer":
    from engine import ReplayBuffer
    buf = ReplayBuffer(n_rows)
    buf.push_game(
        _random_states(n_rows),
        _zero_chains(n_rows),
        _uniform_policies(n_rows),
        _outcomes(n_rows),
        _empty_own(n_rows),
        _wl_with_k_positive(n_rows, k_positive),
    )
    return buf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_accepts_hexb_v5_buffer():
    """H-001/E-003: from_buffer on a real v5 buffer returns a numeric pos_weight."""
    from engine import ReplayBuffer

    n_rows = 200
    buf = _build_v5_buffer(n_rows)

    with tempfile.NamedTemporaryFile(suffix=".hexb", delete=False) as tf:
        path = Path(tf.name)

    try:
        buf.save_to_path(str(path))
        result = compute.from_buffer(path, sample_n=100)
        assert result is not None, "from_buffer returned None for a valid v5 buffer"
        assert isinstance(result, float)
        assert result > 0.0
    finally:
        path.unlink(missing_ok=True)


def test_empirical_pos_weight_differs_from_theoretical():
    """H-001: empirical pos_weight matches theory within 10% for known label density."""
    from engine import ReplayBuffer

    n_rows = 1000
    k = 6  # positives per row
    buf = _build_v5_buffer(n_rows, k_positive=k)

    with tempfile.NamedTemporaryFile(suffix=".hexb", delete=False) as tf:
        path = Path(tf.name)

    try:
        buf.save_to_path(str(path))
        result = compute.from_buffer(path, sample_n=n_rows)
        assert result is not None

        p_theoretical = k / AUX_STRIDE          # 6 / 361 ≈ 0.01661
        pw_theoretical = (1.0 - p_theoretical) / p_theoretical  # ≈ 59.17

        assert abs(result - pw_theoretical) / pw_theoretical < 0.10, (
            f"pos_weight {result:.2f} deviates >10% from theoretical {pw_theoretical:.2f}"
        )
    finally:
        path.unlink(missing_ok=True)


def test_probe_thresholds_match_claude_md():
    """E-006: probe script thresholds must equal the values documented in the
    corpus/probe-discipline rule file (post-§115 this lives at
    docs/rules/workflow.md; pre-§115 it was CLAUDE.md §91)."""
    rule_text = (REPO_ROOT / "docs" / "rules" / "workflow.md").read_text()

    m5  = re.search(r"ext_in_top5_pct\s*[≥>=]+\s*(\d+)",  rule_text)
    m10 = re.search(r"ext_in_top10_pct\s*[≥>=]+\s*(\d+)", rule_text)

    assert m5,  "docs/rules/workflow.md does not document ext_in_top5_pct threshold"
    assert m10, "docs/rules/workflow.md does not document ext_in_top10_pct threshold"

    rule_top5  = float(m5.group(1))
    rule_top10 = float(m10.group(1))

    assert probe.THRESH_EXT_IN_TOP5_PCT == rule_top5, (
        f"probe THRESH_EXT_IN_TOP5_PCT={probe.THRESH_EXT_IN_TOP5_PCT} "
        f"!= rule value={rule_top5}"
    )
    assert probe.THRESH_EXT_IN_TOP10_PCT == rule_top10, (
        f"probe THRESH_EXT_IN_TOP10_PCT={probe.THRESH_EXT_IN_TOP10_PCT} "
        f"!= rule value={rule_top10}"
    )
