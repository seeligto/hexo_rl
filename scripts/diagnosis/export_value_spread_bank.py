#!/usr/bin/env python3
"""§S181 PR-A — one-time export of the value-spread canary bank fixture.

The 40-position bank is procedural (built by `mcts_colony_probe.py` T3
builders). The value-spread canary must NOT depend on script invocation at
training time, so this script freezes the bank into a static JSON fixture at
`tests/fixtures/value_spread_bank.json`.

Bank construction code stays in `mcts_colony_probe.py` — this is purely an
export. Re-run only if the T3 builders change (which would also change the
fixture SHA-256 and break the INV pin — intentional).

Run:  .venv/bin/python scripts/diagnosis/export_value_spread_bank.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcts_colony_probe import (  # noqa: E402
    build_colony_positions,
    build_extension_positions,
)

# FU-1 anchor SHA — the canary + INV pin assert against this exact value.
EXPECTED_SHA = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"


def bank_sha(positions: list[dict]) -> str:
    """Identical to `fu1_value_spread_ladder.bank_fixture_sha` — hash over
    every position's name + class + applied move sequence."""
    h = hashlib.sha256()
    for spec in positions:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        for q, r in spec["moves"]:
            h.update(f"{q},{r};".encode())
    return h.hexdigest()


def main() -> None:
    specs = build_colony_positions() + build_extension_positions()
    positions = [
        {
            "name": s["name"],
            "pos_class": s["pos_class"],
            "moves": [[int(q), int(r)] for q, r in s["moves"]],
        }
        for s in specs
    ]
    sha = bank_sha(positions)
    n_col = sum(1 for p in positions if p["pos_class"] == "colony")
    n_ext = sum(1 for p in positions if "extension" in p["pos_class"])

    if sha != EXPECTED_SHA:
        sys.exit(
            f"FATAL: bank SHA {sha} != expected {EXPECTED_SHA}\n"
            "The T3 builders changed — fixture would diverge from the FU-1 "
            "anchor. STOP."
        )

    fixture = {
        "meta": {
            "source": "mcts_colony_probe.py T3 build_colony_positions + "
                      "build_extension_positions — exported verbatim",
            "n": len(positions),
            "n_colony": n_col,
            "n_extension": n_ext,
            "sha256": sha,
            "sha256_scope": "hash over each position name + pos_class + "
                            "move sequence (q,r;) — matches "
                            "fu1_value_spread_ladder.bank_fixture_sha",
        },
        "positions": positions,
    }
    out = REPO / "tests" / "fixtures" / "value_spread_bank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out}  ({len(positions)} positions, {n_col} colony / "
          f"{n_ext} extension)")
    print(f"bank SHA-256: {sha}  (matches FU-1 anchor)")


if __name__ == "__main__":
    main()
