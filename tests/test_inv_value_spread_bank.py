"""INV pin — value-spread canary bank fixtures (§S181 PR-A + PR-C / L48).

The 40-position colony/extension banks are the canary's measurement
surface. They MUST hash to the FU-1 / A3 anchor SHAs — any drift means
the canary measures a different set of positions than FU-1 / A3
calibrated against, silently breaking the +0.617 / +0.212 anchor
references and the SOFT-ABORT gates.

Drift = STOP. Regenerate T3 only via
`scripts/structural_diagnosis/export_value_spread_bank.py`; alt only via
`scripts/structural_diagnosis/track_a/a3_h_bank.py` — and only if the
underlying builders / corpus genuinely changed (each is a wave-level
decision).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "value_spread_bank.json"
ALT_FIXTURE = REPO / "tests" / "fixtures" / "value_spread_bank_alt.json"

# FU-1 anchor — also pinned in hexo_rl/monitoring/value_spread_canary.py and
# audit/structural/05_fu1_value_spread_ladder.md.
EXPECTED_SHA = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"
# PR-C / L48 A3 alt bank anchor — pinned in fixture `meta.sha256`,
# scripts/structural_diagnosis/track_a/a3_h_bank.py, and
# hexo_rl/monitoring/value_spread_canary.py.
EXPECTED_ALT_SHA = "a68b810f27d31a51e06173bfcd3e2d88d8f3275c7773a63b37aafb3fe25a20ff"


def _bank_sha(positions: list[dict]) -> str:
    """Hash over each position name + pos_class + move sequence — identical
    scope to fu1_value_spread_ladder.bank_fixture_sha."""
    h = hashlib.sha256()
    for spec in positions:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        for q, r in spec["moves"]:
            h.update(f"{int(q)},{int(r)};".encode())
    return h.hexdigest()


def _alt_bank_sha(positions: list[dict]) -> str:
    """A3 SHA scope (name + class + state.tobytes()) — different from T3
    because alt positions carry pre-built (8, 19, 19) state arrays rather
    than move sequences. Matches `a3_h_bank.bank_sha`."""
    h = hashlib.sha256()
    for spec in positions:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        arr = np.asarray(spec["state"], dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()


def test_value_spread_bank_fixture_present():
    assert FIXTURE.exists(), f"value-spread bank fixture missing at {FIXTURE}"


def test_value_spread_bank_sha_pinned():
    data = json.loads(FIXTURE.read_text())
    positions = data["positions"]
    sha = _bank_sha(positions)
    assert sha == EXPECTED_SHA, (
        f"value-spread bank SHA drifted: {sha} != {EXPECTED_SHA}. "
        "The canary's measurement surface no longer matches the FU-1 anchor."
    )
    # The fixture's own recorded SHA must agree with the recomputed one.
    assert data["meta"]["sha256"] == EXPECTED_SHA


def test_value_spread_bank_shape():
    data = json.loads(FIXTURE.read_text())
    positions = data["positions"]
    assert len(positions) == 40
    n_col = sum(1 for p in positions if p["pos_class"] == "colony")
    n_ext = sum(1 for p in positions if "extension" in p["pos_class"])
    assert n_col == 20
    assert n_ext == 20


def test_canary_module_sha_constant_matches():
    """The canary module's BANK_SHA256 must equal the pinned anchor."""
    from hexo_rl.monitoring.value_spread_canary import BANK_SHA256

    assert BANK_SHA256 == EXPECTED_SHA


# ── PR-C / L48 alt bank pins ─────────────────────────────────────────────


def test_alt_bank_fixture_present():
    assert ALT_FIXTURE.exists(), f"alt bank fixture missing at {ALT_FIXTURE}"


def test_alt_bank_sha_pinned():
    data = json.loads(ALT_FIXTURE.read_text())
    positions = data["positions"]
    sha = _alt_bank_sha(positions)
    assert sha == EXPECTED_ALT_SHA, (
        f"alt bank SHA drifted: {sha} != {EXPECTED_ALT_SHA}. The canary's "
        "alt-bank measurement surface no longer matches the A3 anchor."
    )
    assert data["meta"]["sha256"] == EXPECTED_ALT_SHA


def test_alt_bank_shape():
    data = json.loads(ALT_FIXTURE.read_text())
    positions = data["positions"]
    assert len(positions) == 40
    n_col = sum(1 for p in positions if p["pos_class"] == "colony")
    n_ext = sum(1 for p in positions if p["pos_class"] == "extension")
    assert n_col == 20
    assert n_ext == 20


def test_canary_module_alt_sha_constant_matches():
    """The canary module's ALT_BANK_SHA256 must equal the pinned anchor."""
    from hexo_rl.monitoring.value_spread_canary import ALT_BANK_SHA256

    assert ALT_BANK_SHA256 == EXPECTED_ALT_SHA
