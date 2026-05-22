#!/usr/bin/env python3
"""§S181 FU-1 — value-spread checkpoint-ladder probe.

Standalone inspection probe. INSPECTION-ONLY. No training, no hot-path edits,
no config edits. Pins WHEN the value head's colony/extension discriminator
flattens across the §S180b training trajectory.

Method
------
Reuses the T3 canonical 40-position bank VERBATIM — imports
`build_colony_positions` / `build_extension_positions` / `realize_board` /
`raw_policy` from `mcts_colony_probe.py`. No bank regeneration: the builders
are deterministic (no RNG; RNG in T3 is MCTS-Dirichlet only, unused here).

For each checkpoint on the ladder, forwards the NN value head once per board
(via `raw_policy` -> `LocalInferenceEngine.infer_batch`) and records:
  mean V(colony), mean V(extension), V_spread = mean V(colony) - mean V(ext).

Reproducibility gate: the anchor `bootstrap_model_v6.pt` MUST reproduce the
T3 JSON value_head numbers (V_colony 0.1635, V_ext -0.4539, spread +0.617).
If it does not, the bank / load path diverged — STOP, do not trust the ladder.

Run:  .venv/bin/python scripts/structural_diagnosis/fu1_value_spread_ladder.py
Output: audit/structural/05_fu1_value_spread_ladder.json (sidecar; md by agent)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Reuse the T3 probe's bank + inference helpers verbatim. Importing the module
# runs only its top-level imports + def statements (main() is __main__-guarded).
from mcts_colony_probe import (  # noqa: E402
    build_colony_positions,
    build_extension_positions,
    realize_board,
    raw_policy,
)
from hexo_rl.selfplay.inference import LocalInferenceEngine  # noqa: E402
from hexo_rl.viewer.model_loader import load_model  # noqa: E402

# Anchor + §S180b ladder. Anchor = step 0 reference.
CKPTS = [
    ("anchor_step0", REPO / "checkpoints" / "bootstrap_model_v6.pt", 0),
    ("step10k", REPO / "archive/s180b_3knob_fail/ckpts/ckpt_step00010000.pt", 10_000),
    ("step20k", REPO / "archive/s180b_3knob_fail/ckpts/ckpt_step00020000.pt", 20_000),
    ("step30k", REPO / "archive/s180b_3knob_fail/ckpts/ckpt_step00030000.pt", 30_000),
    ("step40k", REPO / "archive/s180b_3knob_fail/ckpts/ckpt_step00040000.pt", 40_000),
    ("step50k", REPO / "archive/s180b_3knob_fail/ckpts/ckpt_step00050000.pt", 50_000),
    ("step53500", REPO / "archive/s180b_3knob_fail/ckpts/ckpt_step00053500.pt", 53_500),
]

# T3 JSON value_head reference — the reproducibility gate for the anchor.
T3_ANCHOR_V_COLONY = 0.1635
T3_ANCHOR_V_EXT = -0.4539
T3_ANCHOR_SPREAD = 0.6174  # 0.1635 - (-0.4539)
GATE_TOL = 0.01  # rounding in T3 JSON is 4dp; tolerance covers it.


def build_bank():
    """Realize the 40-position bank. Returns list of (spec, board)."""
    specs = build_colony_positions() + build_extension_positions()
    realized = []
    for spec in specs:
        b = realize_board(spec)
        if b is None:
            print(f"  SKIP illegal: {spec['name']}")
            continue
        realized.append((spec, b))
    return realized


def bank_fixture_sha(realized):
    """Deterministic identity of the realized bank — hash of every position's
    name, class and applied move sequence. A stable fixture SHA despite the
    bank being procedural rather than a static file."""
    h = hashlib.sha256()
    for spec, _ in realized:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        for q, r in spec["moves"]:
            h.update(f"{q},{r};".encode())
    return h.hexdigest()


def probe_checkpoint(path, realized):
    """Forward one checkpoint's value head over the bank. Returns per-class
    value lists + colony/extension/spread summary."""
    net, meta, _ = load_model(path, device=torch.device("cpu"))
    eng = LocalInferenceEngine(net, torch.device("cpu"))
    per_pos = []
    v_colony, v_ext = [], []
    for spec, board in realized:
        rv = raw_policy(board, eng)["value"]
        per_pos.append({"name": spec["name"], "pos_class": spec["pos_class"],
                        "value": rv})
        if spec["pos_class"] == "colony":
            v_colony.append(rv)
        elif "extension" in spec["pos_class"]:
            v_ext.append(rv)
    mc = float(np.mean(v_colony))
    me = float(np.mean(v_ext))
    return {
        "ckpt_step_meta": meta.get("step"),
        "mean_v_colony": round(mc, 4),
        "mean_v_extension": round(me, 4),
        "std_v_colony": round(float(np.std(v_colony)), 4),
        "std_v_extension": round(float(np.std(v_ext)), 4),
        "v_spread": round(mc - me, 4),
        "n_colony": len(v_colony),
        "n_extension": len(v_ext),
        "per_position": per_pos,
    }


def classify_drift(ladder):
    """Drift-signature classifier per the FU-1 spec.

    ladder: list of (label, step, v_spread) ordered by step, anchor first.
    """
    spreads = [s for _, _, s in ladder]
    steps = [st for _, st, _ in ladder]
    total_loss = spreads[0] - spreads[-1]
    # per-interval deltas (spread change; negative = loss)
    intervals = []
    for i in range(1, len(spreads)):
        d = spreads[i] - spreads[i - 1]
        intervals.append((steps[i - 1], steps[i], d))
    losses = [-d for _, _, d in intervals]  # positive = spread lost
    max_single_loss = max(losses) if losses else 0.0
    max_single_frac = (max_single_loss / total_loss) if total_loss > 0 else 0.0
    monotone = all(d <= 1e-9 for _, _, d in intervals)  # non-increasing
    big_gain = max((d for _, _, d in intervals), default=0.0)

    # OSCILLATION: non-monotone with a magnitude swing > 0.10
    if not monotone and big_gain > 0.10:
        verdict = "OSCILLATION"
    elif monotone and max_single_loss <= 0.15:
        verdict = "GRADUAL"
    elif max_single_frac >= 0.50:
        verdict = "PHASE-TRANSITION"
    elif monotone:
        # monotone but one interval drops > 0.15 without being >=50% of total
        verdict = "PHASE-TRANSITION" if max_single_loss > 0.15 else "GRADUAL"
    else:
        verdict = "NON-CLASSIFIABLE"
    return {
        "verdict": verdict,
        "total_spread_loss": round(total_loss, 4),
        "intervals": [{"from": a, "to": b, "delta": round(d, 4)}
                      for a, b, d in intervals],
        "max_single_interval_loss": round(max_single_loss, 4),
        "max_single_interval_frac_of_total": round(max_single_frac, 4),
        "monotone_non_increasing": monotone,
        "max_single_interval_gain": round(big_gain, 4),
    }


def main():
    t0 = time.time()
    print("[1] building T3 canonical 40-position bank ...", flush=True)
    realized = build_bank()
    n_col = sum(1 for s, _ in realized if s["pos_class"] == "colony")
    n_ext = sum(1 for s, _ in realized if "extension" in s["pos_class"])
    sha = bank_fixture_sha(realized)
    print(f"    realized {len(realized)}/40  ({n_col} colony, {n_ext} extension)")
    print(f"    bank fixture SHA-256: {sha}")

    results = {}
    ladder = []
    for label, path, step in CKPTS:
        if not path.exists():
            print(f"  MISSING: {path}")
            sys.exit(f"checkpoint absent: {path} — STOP")
        print(f"[2] probing {label} ({path.name}) ...", flush=True)
        r = probe_checkpoint(path, realized)
        results[label] = r
        ladder.append((label, step, r["v_spread"]))
        print(f"    V_colony {r['mean_v_colony']:+.4f}  "
              f"V_ext {r['mean_v_extension']:+.4f}  "
              f"V_spread {r['v_spread']:+.4f}")

    # reproducibility gate — anchor must match the T3 JSON value_head numbers.
    a = results["anchor_step0"]
    gate = {
        "t3_ref": {"v_colony": T3_ANCHOR_V_COLONY, "v_extension": T3_ANCHOR_V_EXT,
                   "v_spread": T3_ANCHOR_SPREAD},
        "fu1_anchor": {"v_colony": a["mean_v_colony"],
                       "v_extension": a["mean_v_extension"],
                       "v_spread": a["v_spread"]},
        "tol": GATE_TOL,
    }
    gate["pass"] = (
        abs(a["mean_v_colony"] - T3_ANCHOR_V_COLONY) <= GATE_TOL
        and abs(a["mean_v_extension"] - T3_ANCHOR_V_EXT) <= GATE_TOL
        and abs(a["v_spread"] - T3_ANCHOR_SPREAD) <= GATE_TOL
    )
    print(f"\n[3] anchor reproducibility gate vs T3 JSON: "
          f"{'PASS' if gate['pass'] else 'FAIL'}")
    if not gate["pass"]:
        print("    FU-1 anchor does NOT reproduce T3 +0.617 — bank/load path "
              "diverged. STOP — ladder untrustworthy.")

    drift = classify_drift(ladder)
    print(f"[4] drift signature: {drift['verdict']}  "
          f"(total spread loss {drift['total_spread_loss']:+.4f})")

    out = {
        "meta": {
            "script": "scripts/structural_diagnosis/fu1_value_spread_ladder.py",
            "bank_source": "mcts_colony_probe.py (T3) build_colony_positions + "
                           "build_extension_positions — reused verbatim",
            "bank_fixture_sha256": sha,
            "bank_realized": len(realized),
            "n_colony": n_col,
            "n_extension": n_ext,
            "encoding": "v6",
            "wall_s": round(time.time() - t0, 1),
        },
        "reproducibility_gate": gate,
        "ladder": [{"label": l, "step": st, "v_spread": sp}
                   for l, st, sp in ladder],
        "drift_signature": drift,
        "per_checkpoint": results,
    }
    op = REPO / "audit" / "structural" / "05_fu1_value_spread_ladder.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
