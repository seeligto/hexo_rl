#!/usr/bin/env python3
"""§S181 FU-2 — A2 pretrain-quality gates (V-FU2-PT-1/2/3/4).

ALL FOUR must PASS before the T3 sustained run launches.

  PT-1  final pretrain loss   ≤ 3.50   (vs §148 v6 anchor = 3.31)
  PT-2  SealBot WR n=500       ≥ 13%    (vs §148 v6 anchor = 11.4%/16.4%¹)
  PT-3  V_spread on T3 bank    ≥ +0.45  (75% of pre-A2 anchor +0.617)
  PT-4  Threat probes C1/C2/C3 PASS   (default thresholds)

¹ Per sprint log §149: §148 v6 SealBot WR = 11.4%; the 16.4% figure brief
  cites belongs to v7e30 (resumed fine-tune). Brief's PT-2 threshold of
  13% sits between the two — A2 from-scratch is judged against the
  closer-comparator §148 v6 (11.4% + ~2pp headroom).

Outputs:
  - Per-gate verdict line to stdout.
  - JSON sidecar reports/fu2_pretrain/a2_quality_<timestamp>.json with all
    numbers + thresholds + composite verdict.

PT-2 is the long pole (~20-30 min on laptop CPU); pass --skip-pt2 to defer
it to vast and emit a placeholder, or --pt2-results-json to inject results
from a previous run.

Usage:
  python scripts/structural_diagnosis/fu2_pretrain_quality_gates.py \\
      --ckpt checkpoints/bootstrap_model_v6_a2.pt \\
      --pretrain-log reports/fu2_pretrain/pretrain_a2_<ts>.log \\
      --out reports/fu2_pretrain/a2_quality_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _gate_pt1_final_loss(log_path: Path) -> dict:
    """Parse "Epoch N/M  loss=X" lines; report the last epoch's loss."""
    text = log_path.read_text()
    epoch_lines = re.findall(r"Epoch (\d+)/(\d+)\s+loss=(\d+\.\d+)", text)
    if not epoch_lines:
        return {"gate": "PT-1", "status": "ERROR",
                "reason": "no Epoch N/M loss=X line found in log"}
    last = epoch_lines[-1]
    final_loss = float(last[2])
    threshold = 3.50
    passed = final_loss <= threshold
    return {
        "gate": "PT-1",
        "metric": "final_pretrain_loss",
        "value": final_loss,
        "threshold_le": threshold,
        "comparator": f"§148 v6 = 3.31",
        "status": "PASS" if passed else "FAIL",
        "epoch_count": int(last[1]),
        "epochs_completed": int(last[0]),
    }


def _gate_pt3_value_spread(ckpt: Path) -> dict:
    """Load the A2 anchor, run the value_spread canary on the T3 bank."""
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.monitoring.value_spread_canary import (
        BANK_SHA256, compute_value_spread, load_bank,
    )

    model, _spec, label = load_model_with_encoding(ckpt, torch.device("cpu"))
    bank = load_bank()
    assert bank.sha == BANK_SHA256, f"bank SHA drift: {bank.sha}"
    result = compute_value_spread(model, bank, torch.device("cpu"))
    spread = result.spread
    threshold = 0.45
    passed = spread >= threshold
    return {
        "gate": "PT-3",
        "metric": "value_spread_T3_bank",
        "value": spread,
        "threshold_ge": threshold,
        "comparator": "pre-A2 anchor +0.617 (75% target)",
        "status": "PASS" if passed else "FAIL",
        "mean_colony": result.mean_colony,
        "mean_extension": result.mean_ext,
        "encoding_label": label,
        "bank_sha": bank.sha,
    }


def _gate_pt4_threat_probes(ckpt: Path, baseline_ckpt: Path | None) -> dict:
    """Invoke scripts/probe_threat_logits.py; parse C1/C2/C3 verdicts."""
    cmd = [
        sys.executable,
        "scripts/probe_threat_logits.py",
        "--checkpoint", str(ckpt),
    ]
    if baseline_ckpt is not None and baseline_ckpt.exists():
        cmd.extend(["--baseline-checkpoint", str(baseline_ckpt)])
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    stdout = proc.stdout + proc.stderr
    # exit codes: 0=PASS, 1=FAIL, 2=error
    rc = proc.returncode
    status = {0: "PASS", 1: "FAIL"}.get(rc, "ERROR")
    return {
        "gate": "PT-4",
        "metric": "threat_probes_C1_C2_C3",
        "status": status,
        "exit_code": rc,
        "stdout_tail": stdout[-1500:] if stdout else "",
    }


def _gate_pt2_sealbot(ckpt: Path, n_games: int = 500,
                       out_json: Path | None = None) -> dict:
    """SealBot eval n=500 via scripts/run_sealbot_eval.py argmax."""
    if out_json is None:
        out_json = _REPO / "reports/fu2_pretrain/a2_sealbot_argmax.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/run_sealbot_eval.py",
        "--checkpoint", str(ckpt),
        "--inference", "argmax",
        "--n-games", str(n_games),
        "--output", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    stdout = proc.stdout + proc.stderr
    if not out_json.exists():
        return {"gate": "PT-2", "status": "ERROR",
                "reason": f"no output JSON at {out_json}",
                "stdout_tail": stdout[-1500:]}
    data = json.loads(out_json.read_text())
    # SealBot eval JSON typically has "winrate" or "wins"/"losses"/"draws".
    wr = data.get("winrate")
    if wr is None and "wins" in data and "n_games" in data:
        wr = data["wins"] / data["n_games"]
    if wr is None:
        return {"gate": "PT-2", "status": "ERROR",
                "reason": "could not extract winrate from output JSON",
                "json_keys": list(data.keys())}
    threshold = 0.13
    passed = wr >= threshold
    return {
        "gate": "PT-2",
        "metric": "sealbot_winrate_argmax",
        "value": wr,
        "threshold_ge": threshold,
        "comparator": "§148 v6 = 11.4%",
        "n_games": n_games,
        "status": "PASS" if passed else "FAIL",
        "raw_json": str(out_json),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pretrain-log", type=Path, required=True)
    p.add_argument("--baseline-ckpt", type=Path,
                    default=Path("checkpoints/bootstrap_model_v6.pt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-pt2", action="store_true",
                    help="Skip PT-2 SealBot eval (long pole); emit placeholder.")
    p.add_argument("--pt2-results-json", type=Path,
                    help="Use prior PT-2 SealBot eval JSON instead of re-running.")
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"FATAL: ckpt missing at {args.ckpt}", file=sys.stderr)
        return 2
    if not args.pretrain_log.exists():
        print(f"FATAL: pretrain log missing at {args.pretrain_log}", file=sys.stderr)
        return 2

    results = {
        "wave": "§S181 FU-2",
        "ckpt": str(args.ckpt),
        "pretrain_log": str(args.pretrain_log),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gates": {},
    }

    pt1 = _gate_pt1_final_loss(args.pretrain_log)
    results["gates"]["PT-1"] = pt1
    print(f"[PT-1] {pt1['status']:<5} final_loss={pt1.get('value')!r} "
          f"<= {pt1.get('threshold_le')!r}")

    pt3 = _gate_pt3_value_spread(args.ckpt)
    results["gates"]["PT-3"] = pt3
    print(f"[PT-3] {pt3['status']:<5} V_spread={pt3.get('value')!r} "
          f">= {pt3.get('threshold_ge')!r}")

    pt4 = _gate_pt4_threat_probes(args.ckpt, args.baseline_ckpt)
    results["gates"]["PT-4"] = pt4
    print(f"[PT-4] {pt4['status']:<5} threat_probes exit={pt4.get('exit_code')!r}")

    if args.skip_pt2:
        pt2 = {"gate": "PT-2", "status": "DEFERRED",
                "reason": "--skip-pt2 (run on vast separately)"}
    elif args.pt2_results_json is not None:
        with args.pt2_results_json.open() as f:
            data = json.load(f)
        wr = data.get("winrate") or (data["wins"] / data["n_games"])
        threshold = 0.13
        pt2 = {
            "gate": "PT-2",
            "metric": "sealbot_winrate_argmax",
            "value": wr,
            "threshold_ge": threshold,
            "comparator": "§148 v6 = 11.4%",
            "status": "PASS" if wr >= threshold else "FAIL",
            "raw_json": str(args.pt2_results_json),
        }
    else:
        pt2 = _gate_pt2_sealbot(args.ckpt)
    results["gates"]["PT-2"] = pt2
    print(f"[PT-2] {pt2['status']:<5} sealbot_wr={pt2.get('value')!r} "
          f">= {pt2.get('threshold_ge')!r}")

    all_statuses = [results["gates"][k]["status"] for k in ("PT-1", "PT-2", "PT-3", "PT-4")]
    composite_pass = all(s == "PASS" for s in all_statuses)
    composite_deferred = any(s == "DEFERRED" for s in all_statuses)
    if composite_pass:
        composite = "PASS"
    elif composite_deferred and all(s in ("PASS", "DEFERRED") for s in all_statuses):
        composite = "PASS-PENDING-PT2"
    else:
        composite = "FAIL"
    results["composite_verdict"] = composite
    print(f"\nCOMPOSITE: {composite}  ({', '.join(all_statuses)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0 if composite_pass else 1


if __name__ == "__main__":
    sys.exit(main())
