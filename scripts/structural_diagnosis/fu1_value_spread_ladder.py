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

import argparse
import hashlib
import json
import re
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

# Bank fixture SHA-256 — the FU-1 anchor. The realized bank MUST hash to this
# (§S181 hard constraint). Drift = STOP, the ladder is non-comparable.
BANK_SHA256 = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"


def build_ckpts_from_dir(ckpt_dir: Path):
    """FU-1.5 finer-ladder mode: build the ladder from a checkpoint directory.

    Globs `ckpt_step*.pt` / `checkpoint_*.pt`, parses the step number from the
    filename, sorts ascending, prepends the anchor at step 0. Used to probe
    the §S181 FU-1.5 2k-cadence re-run without editing the hardcoded CKPTS.
    """
    found = []
    for p in sorted(ckpt_dir.glob("*.pt")):
        m = re.search(r"(\d{4,})", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step <= 0:
            continue
        found.append((f"step{step}", p, step))
    found.sort(key=lambda t: t[2])
    return [("anchor_step0", REPO / "checkpoints" / "bootstrap_model_v6.pt", 0)] + found


def classify_fl_verdict(ladder):
    """§S181 FU-1.5 pre-registered verdict V-FL-A..E (applied literally).

    ladder: list of (label, step, v_spread) ordered by step, anchor first.
    Returns the verdict dict. Conditions are evaluated EXACTLY as
    pre-registered — no post-hoc threshold moves (L13 guard).
    """
    by_step = {st: sp for _, st, sp in ladder}
    steps = sorted(by_step)
    # 2k-interval deltas keyed by the interval end-step.
    intervals = []  # (from, to, delta)
    for i in range(1, len(steps)):
        a, b = steps[i - 1], steps[i]
        intervals.append((a, b, by_step[b] - by_step[a]))

    sp2 = by_step.get(2000)
    sp4 = by_step.get(4000)

    # V-FL-A STEP-0-ONSET: V_spread(2k) <= +0.40 AND V_spread(4k) <= +0.25
    a_match = (sp2 is not None and sp4 is not None
               and sp2 <= 0.40 and sp4 <= 0.25)
    # V-FL-B MID-CLIFF: V_spread(2k) >= +0.50 AND a single 2k interval in
    # [4k,10k] drops V_spread by >= 0.30
    b_cliff = [(f, t, d) for f, t, d in intervals
               if 4000 <= t <= 10000 and d <= -0.30]
    b_match = (sp2 is not None and sp2 >= 0.50 and len(b_cliff) > 0)
    # V-FL-C GRADUAL: monotone non-increasing 0->20k AND no 2k interval
    # loses more than 0.15
    monotone = all(d <= 1e-9 for _, _, d in intervals)
    c_match = monotone and all((-d) <= 0.15 for _, _, d in intervals)
    # V-FL-D RECOVERY: any 2k interval >20k shows >= +0.15 recovery from a
    # sub-+0.20 dip. (A 20k-capped run has no >20k interval — D cannot fire;
    # require >=2 consecutive recovering intervals per the Task-3 guard.)
    d_recov = []
    for i in range(1, len(intervals)):
        f0, t0, d0 = intervals[i - 1]
        f1, t1, d1 = intervals[i]
        if t0 > 20000 and t1 > 20000 and d0 >= 0.15 and d1 >= 0.15 \
                and by_step[f0] < 0.20:
            d_recov.append((f0, t1))
    d_match = len(d_recov) > 0

    if d_match:
        verdict, routing = "V-FL-D", ("RECOVERY — escalate; falsifies the "
                                      "permanent-collapse reading. No FU-2 launch.")
    elif a_match:
        verdict, routing = "V-FL-A", ("STEP-0-ONSET — FU-2 A2 architecture arm "
                                      "(multi-scale avg-pool) load-bearing.")
    elif b_match:
        verdict, routing = "V-FL-B", ("MID-CLIFF — FU-2 A3 aux-loss arm "
                                      "(colony-penalty) load-bearing.")
    elif c_match:
        verdict, routing = "V-FL-C", ("GRADUAL — FU-2 A3 first (cheaper, no "
                                      "re-pretrain); A2 as fallback.")
    else:
        verdict, routing = "V-FL-E", ("INCONCLUSIVE — escalate to operator, "
                                      "no FU-2 launch.")
    return {
        "verdict": verdict,
        "fu2_routing": routing,
        "v_spread_2k": sp2,
        "v_spread_4k": sp4,
        "conditions": {
            "V-FL-A_step0_onset": a_match,
            "V-FL-B_mid_cliff": b_match,
            "V-FL-C_gradual": c_match,
            "V-FL-D_recovery": d_match,
        },
        "intervals_2k": [{"from": f, "to": t, "delta": round(d, 4)}
                         for f, t, d in intervals],
        "mid_cliff_intervals": [{"from": f, "to": t, "delta": round(d, 4)}
                                for f, t, d in b_cliff],
    }


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="FU-1 / FU-1.5 value-spread ladder")
    ap.add_argument("--ckpt-dir", default=None,
                    help="FU-1.5 mode: probe every ckpt in this dir (sorted by "
                         "step) + anchor. Default = the hardcoded FU-1 ladder.")
    ap.add_argument("--out", default=None,
                    help="JSON sidecar output path. Default = FU-1 sidecar.")
    args = ap.parse_args(argv)
    fl_mode = args.ckpt_dir is not None

    t0 = time.time()
    print("[1] building T3 canonical 40-position bank ...", flush=True)
    realized = build_bank()
    n_col = sum(1 for s, _ in realized if s["pos_class"] == "colony")
    n_ext = sum(1 for s, _ in realized if "extension" in s["pos_class"])
    sha = bank_fixture_sha(realized)
    print(f"    realized {len(realized)}/40  ({n_col} colony, {n_ext} extension)")
    print(f"    bank fixture SHA-256: {sha}")

    # §S181 hard SHA gate — the bank MUST hash to the FU-1 anchor. Drift means
    # the ladder is non-comparable to FU-1. STOP.
    if sha != BANK_SHA256:
        sys.exit(f"BANK SHA MISMATCH: {sha} != {BANK_SHA256} — STOP. The bank "
                 "diverged from the FU-1 anchor; the ladder is non-comparable.")
    print(f"    bank SHA gate: PASS (matches FU-1 anchor)")

    ckpts = build_ckpts_from_dir(Path(args.ckpt_dir)) if fl_mode else CKPTS
    print(f"    ladder mode: {'FU-1.5 finer-ladder' if fl_mode else 'FU-1'} "
          f"— {len(ckpts)} checkpoints")

    results = {}
    ladder = []
    for label, path, step in ckpts:
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
            "mode": "fu1.5_finer_ladder" if fl_mode else "fu1",
            "bank_source": "mcts_colony_probe.py (T3) build_colony_positions + "
                           "build_extension_positions — reused verbatim",
            "bank_fixture_sha256": sha,
            "bank_sha_gate_pass": True,
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
    if fl_mode:
        fl = classify_fl_verdict(ladder)
        out["fl_verdict"] = fl
        print(f"[5] FU-1.5 pre-registered verdict: {fl['verdict']}")
        print(f"    {fl['fu2_routing']}")

    if args.out is not None:
        op = Path(args.out)
    else:
        op = REPO / "audit" / "structural" / "05_fu1_value_spread_ladder.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
