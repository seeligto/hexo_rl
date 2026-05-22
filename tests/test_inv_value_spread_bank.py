"""INV pin — value-spread canary bank fixture (§S181 PR-A).

The 40-position colony/extension bank is the canary's measurement surface.
It MUST hash to the FU-1 anchor SHA — any drift means the canary measures a
different set of positions than FU-1 calibrated against, silently breaking
the +0.617 anchor reference and the +0.20 abort gate.

Drift = STOP. Regenerate only via
`scripts/structural_diagnosis/export_value_spread_bank.py`, and only if the
T3 builders genuinely changed (which is itself a wave-level decision).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "value_spread_bank.json"

# FU-1 anchor — also pinned in hexo_rl/monitoring/value_spread_canary.py and
# audit/structural/05_fu1_value_spread_ladder.md.
EXPECTED_SHA = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"


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
