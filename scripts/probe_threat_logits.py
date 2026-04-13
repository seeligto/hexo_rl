"""
Threat-logit probe — step-5k kill criterion for HeXO Phase 4.0 sustained runs.

Loads a checkpoint, runs 20 curated positions through the threat head, and
reports whether the threat head has learned genuine spatial signal (extension
cells score higher than random empty cells).

Kill criterion (§85 / §89 of docs/07_PHASE4_SPRINT_LOG.md):
  At step 5000, re-run this probe. PASS requires ALL THREE:
    1. ext_logit_mean >= bootstrap_ext_mean - margin  (margin default 1.0)
    2. contrast_mean  >= 0.38
    3. ext_in_top5_pct >= 40.0
  Bootstrap baseline numbers come from fixtures/threat_probe_baseline.json,
  written once by `make probe.bootstrap --write-baseline` (or the
  --write-baseline CLI flag). If that file is absent, probe prints FAIL with
  "no baseline recorded — run make probe.bootstrap first".

Exit codes:
  0  PASS
  1  FAIL
  2  error (bad checkpoint, missing fixture file, shape mismatch, etc.)

Usage:
  # Generate baseline (bootstrap_model.pt) and save json:
  python scripts/probe_threat_logits.py \\
      --checkpoint checkpoints/bootstrap_model.pt --write-baseline

  # Probe a trained checkpoint against the saved baseline:
  python scripts/probe_threat_logits.py --checkpoint <ckpt> \\
      --output reports/probes/step5000.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys
from hexo_rl.training.trainer import Trainer

BOARD_SIZE: int = 19

# ── Kill-criterion thresholds (§85 / §89, corrected) ─────────────────────────

THRESH_CONTRAST_MEAN: float = 0.38    # mean contrast must be ≥ 0.38
THRESH_EXT_IN_TOP5_PCT: float = 40.0  # extension cell in policy top-5 ≥ 40%
BASELINE_MARGIN: float = 1.0          # ext_logit_mean ≥ baseline_mean − margin

# Canonical baseline file (generated once via --write-baseline).
BASELINE_JSON_PATH: Path = REPO_ROOT / "fixtures" / "threat_probe_baseline.json"

# Fixed seed for deterministic inference (same value every run).
_PROBE_SEED: int = 42


def _set_determinism() -> None:
    """Lock all RNG sources so two consecutive runs produce identical output."""
    torch.manual_seed(_PROBE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_PROBE_SEED)
    # Disallow non-deterministic CUDA kernels (raises if any are used).
    torch.use_deterministic_algorithms(True, warn_only=False)


# ── Baseline JSON helpers ─────────────────────────────────────────────────────


def load_baseline_json(path: Path = BASELINE_JSON_PATH) -> Optional[Dict]:
    """Return parsed baseline dict, or None if file absent."""
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def save_baseline_json(
    agg: Dict,
    ckpt_name: str,
    path: Path = BASELINE_JSON_PATH,
) -> None:
    """Persist aggregate results as the canonical baseline JSON."""
    record = {
        "ext_logit_mean": agg["ext_logit_mean"],
        "ext_logit_std": agg["ext_logit_std"],
        "ctrl_logit_mean": agg["ctrl_logit_mean"],
        "ctrl_logit_std": agg["ctrl_logit_std"],
        "contrast_mean": agg["contrast_mean"],
        "contrast_std": agg["contrast_std"],
        "ext_in_top5_frac": agg["ext_in_top5_frac"],
        "n": agg["n"],
        "checkpoint": ckpt_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "margin": BASELINE_MARGIN,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"Baseline saved: {path}", file=sys.stderr)


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(
    ckpt_path: Path,
    device: Optional[torch.device] = None,
) -> HexTacToeNet:
    """Load checkpoint → HexTacToeNet in eval mode (FP32 always)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = Trainer._extract_model_state(ckpt)
    state = normalize_model_state_dict_keys(state)
    hparams = Trainer._infer_model_hparams(state)

    model = HexTacToeNet(
        board_size=int(hparams.get("board_size", 19)),
        in_channels=int(hparams.get("in_channels", 18)),
        filters=int(hparams.get("filters", 128)),
        res_blocks=int(hparams.get("res_blocks", 12)),
        se_reduction_ratio=int(hparams.get("se_reduction_ratio", 4)),
    )
    model.load_state_dict(state, strict=False)
    # FP32 throughout — avoids autocast-induced non-determinism at zero accuracy cost.
    model = model.float().to(device).eval()
    return model


# ── Fixture loading ───────────────────────────────────────────────────────────


def load_positions(npz_path: Path) -> Dict:
    """Load fixture NPZ; returns dict with numpy arrays."""
    data = np.load(str(npz_path), allow_pickle=False)
    required = {"states", "ext_cell_idx", "control_cell_idx"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"NPZ missing required arrays: {missing}")

    states = data["states"]  # (N, 18, 19, 19) float16
    if states.ndim != 4 or states.shape[1:] != (18, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"states shape {states.shape} — expected (N, 18, {BOARD_SIZE}, {BOARD_SIZE})"
        )

    # Load cell indices verbatim from NPZ — never regenerate at load time.
    return {
        "states": states,
        "side_to_move": data.get("side_to_move", np.zeros(len(states), dtype=np.int8)),
        "ext_cell_idx": data["ext_cell_idx"].astype(np.int32),
        "control_cell_idx": data["control_cell_idx"].astype(np.int32),
        "game_phase": data.get("game_phase", np.full(len(states), "unknown")),
        "n": len(states),
    }


# ── Per-position probing ──────────────────────────────────────────────────────


def _probe_one(
    model: HexTacToeNet,
    state_fp16: np.ndarray,   # (18, 19, 19) float16
    ext_flat: int,
    ctrl_flat: int,
    device: torch.device,
) -> Tuple[float, float, float, List[int]]:
    """Forward one position in FP32; return (ext_logit, ctrl_logit, contrast, top5)."""
    # Always FP32 — no autocast, deterministic across runs.
    x = torch.from_numpy(state_fp16.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x, threat=True)

    # out = (log_policy, value, v_logit, [opp_reply, sigma2, ownership,] threat_logits)
    threat_logits = out[-1]  # (1, 1, 19, 19)
    log_policy = out[0]       # (1, H*W + 1)

    threat_flat = threat_logits[0, 0].float().cpu().flatten()  # (361,)

    ext_logit = float(threat_flat[ext_flat].item())
    ctrl_logit = float(threat_flat[ctrl_flat].item())
    contrast = ext_logit - ctrl_logit

    policy_spatial = log_policy[0, :BOARD_SIZE * BOARD_SIZE].float().cpu()
    top5_indices = policy_spatial.topk(min(5, len(policy_spatial))).indices.tolist()

    return ext_logit, ctrl_logit, contrast, top5_indices


def probe_positions(
    model: HexTacToeNet,
    positions: Dict,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Run probe on all positions; return per-position result dicts."""
    if device is None:
        device = next(model.parameters()).device

    states = positions["states"]
    ext_idxs = positions["ext_cell_idx"]
    ctrl_idxs = positions["control_cell_idx"]
    phases = positions["game_phase"]
    n = positions["n"]

    results = []
    for i in range(n):
        ext_flat = int(ext_idxs[i])
        ctrl_flat = int(ctrl_idxs[i])

        if not (0 <= ext_flat < BOARD_SIZE * BOARD_SIZE):
            raise ValueError(
                f"Position {i}: ext_cell_idx={ext_flat} out of range [0, {BOARD_SIZE**2})"
            )
        if not (0 <= ctrl_flat < BOARD_SIZE * BOARD_SIZE):
            raise ValueError(
                f"Position {i}: control_cell_idx={ctrl_flat} out of range [0, {BOARD_SIZE**2})"
            )

        ext_logit, ctrl_logit, contrast, top5 = _probe_one(
            model, states[i], ext_flat, ctrl_flat, device
        )

        ext_in_top5 = ext_flat in top5

        results.append({
            "idx": i,
            "game_phase": str(phases[i]) if hasattr(phases[i], "__str__") else "unknown",
            "ext_flat": ext_flat,
            "ctrl_flat": ctrl_flat,
            "ext_logit": ext_logit,
            "ctrl_logit": ctrl_logit,
            "contrast": contrast,
            "policy_top5": top5,
            "ext_in_policy_top5": ext_in_top5,
        })

    return results


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate(results: List[Dict]) -> Dict:
    ext_logits = np.array([r["ext_logit"] for r in results], dtype=np.float64)
    ctrl_logits = np.array([r["ctrl_logit"] for r in results], dtype=np.float64)
    contrasts = np.array([r["contrast"] for r in results], dtype=np.float64)
    ext_in_top5_count = sum(1 for r in results if r["ext_in_policy_top5"])

    return {
        "n": len(results),
        "ext_logit_mean": float(ext_logits.mean()),
        "ext_logit_std": float(ext_logits.std()),
        "ctrl_logit_mean": float(ctrl_logits.mean()),
        "ctrl_logit_std": float(ctrl_logits.std()),
        "contrast_mean": float(contrasts.mean()),
        "contrast_std": float(contrasts.std()),
        "ext_in_top5_frac": ext_in_top5_count / max(len(results), 1),
    }


# ── Pass / fail decision ──────────────────────────────────────────────────────


def check_pass(
    agg: Dict,
    baseline: Optional[Dict] = None,
    margin: float = BASELINE_MARGIN,
) -> Tuple[bool, Dict[str, bool]]:
    """Evaluate three-condition PASS rule (§85 / §89, corrected).

    Returns (overall_pass, per_condition_dict).

    Condition 1 requires a loaded baseline; if None, it is treated as FAIL.
    Condition 2: contrast_mean >= 0.38
    Condition 3: ext_in_top5_pct >= 40.0
    """
    if baseline is not None:
        c1 = agg["ext_logit_mean"] >= baseline["ext_logit_mean"] - margin
    else:
        c1 = False  # no baseline → cannot evaluate

    c2 = agg["contrast_mean"] >= THRESH_CONTRAST_MEAN
    c3 = agg["ext_in_top5_frac"] * 100.0 >= THRESH_EXT_IN_TOP5_PCT

    conditions = {
        "ext_logit_vs_baseline": c1,
        "contrast": c2,
        "ext_in_top5": c3,
    }
    return (c1 and c2 and c3), conditions


# ── Report formatting ─────────────────────────────────────────────────────────


def _fmt(val: float, std: Optional[float] = None) -> str:
    if std is not None:
        return f"{val:+.2f} ± {std:.2f}"
    return f"{val:+.2f}"


def format_report(
    results: List[Dict],
    agg: Dict,
    ckpt_name: str,
    baseline: Optional[Dict] = None,
    baseline_agg: Optional[Dict] = None,
    baseline_name: Optional[str] = None,
) -> str:
    """Render markdown report with three explicit PASS conditions."""
    lines: List[str] = []

    overall_pass, conditions = check_pass(agg, baseline=baseline)
    verdict = "**PASS**" if overall_pass else "**FAIL**"
    lines.append(f"# Threat-Logit Probe: {ckpt_name}\n")
    lines.append(f"Verdict: {verdict}\n")

    # Three explicit conditions
    lines.append("## Pass conditions (§85 / §89 corrected kill criterion)\n")
    lines.append("| # | condition | threshold | value | result |")
    lines.append("|---|-----------|-----------|-------|--------|")

    if baseline is not None:
        bl_mean = baseline["ext_logit_mean"]
        c1_thresh = f"≥ {bl_mean:+.2f} − {BASELINE_MARGIN:.1f} = {bl_mean - BASELINE_MARGIN:+.2f}"
    else:
        c1_thresh = "no baseline — run make probe.bootstrap first"
    c1_val = f"{agg['ext_logit_mean']:+.3f}"
    c1_res = "PASS" if conditions["ext_logit_vs_baseline"] else "FAIL"

    c2_thresh = f"≥ {THRESH_CONTRAST_MEAN}"
    c2_val = f"{agg['contrast_mean']:+.3f}"
    c2_res = "PASS" if conditions["contrast"] else "FAIL"

    c3_thresh = f"≥ {THRESH_EXT_IN_TOP5_PCT:.0f}%"
    c3_val = f"{agg['ext_in_top5_frac'] * 100:.0f}%"
    c3_res = "PASS" if conditions["ext_in_top5"] else "FAIL"

    lines.append(f"| 1 | ext_logit_mean vs baseline | {c1_thresh} | {c1_val} | **{c1_res}** |")
    lines.append(f"| 2 | contrast_mean (ext − ctrl) | {c2_thresh} | {c2_val} | **{c2_res}** |")
    lines.append(f"| 3 | ext cell in policy top-5  | {c3_thresh} | {c3_val} | **{c3_res}** |")
    lines.append("")

    # Summary metrics table (optionally compare with baseline_agg for historic display)
    ref_agg = baseline_agg if baseline_agg is not None else baseline
    ref_name = baseline_name or (baseline.get("checkpoint") if baseline else None)

    if ref_agg is not None and ref_name is not None:
        lines.append(f"| metric | {ckpt_name} | {ref_name} |")
        lines.append("|--------|------------|-------------|")
        lines.append(
            f"| threat logit @ extension cell | "
            f"{_fmt(agg['ext_logit_mean'], agg['ext_logit_std'])} | "
            f"{_fmt(ref_agg['ext_logit_mean'], ref_agg['ext_logit_std'])} |"
        )
        lines.append(
            f"| threat logit @ control cell | "
            f"{_fmt(agg['ctrl_logit_mean'], agg['ctrl_logit_std'])} | "
            f"{_fmt(ref_agg['ctrl_logit_mean'], ref_agg['ctrl_logit_std'])} |"
        )
        lines.append(
            f"| contrast (extension − control) | "
            f"{_fmt(agg['contrast_mean'])} | "
            f"{_fmt(ref_agg['contrast_mean'])} |"
        )
        lines.append(
            f"| extension cell in policy top-5 | "
            f"{agg['ext_in_top5_frac']:.0%} | "
            f"{ref_agg['ext_in_top5_frac']:.0%} |"
        )
    else:
        lines.append(f"| metric | {ckpt_name} |")
        lines.append("|--------|------------|")
        lines.append(
            f"| threat logit @ extension cell | "
            f"{_fmt(agg['ext_logit_mean'], agg['ext_logit_std'])} |"
        )
        lines.append(
            f"| threat logit @ control cell | "
            f"{_fmt(agg['ctrl_logit_mean'], agg['ctrl_logit_std'])} |"
        )
        lines.append(
            f"| contrast (extension − control) | "
            f"{_fmt(agg['contrast_mean'])} |"
        )
        lines.append(
            f"| extension cell in policy top-5 | "
            f"{agg['ext_in_top5_frac']:.0%} |"
        )

    lines.append("")

    # Per-position detail
    lines.append("## Per-position detail\n")
    lines.append("| # | phase | ext_logit | ctrl_logit | contrast | ext∈top5 |")
    lines.append("|---|-------|-----------|------------|----------|----------|")
    for r in results:
        top5_flag = "yes" if r["ext_in_policy_top5"] else "no"
        lines.append(
            f"| {r['idx']+1} | {r['game_phase']} | "
            f"{r['ext_logit']:+.3f} | {r['ctrl_logit']:+.3f} | "
            f"{r['contrast']:+.3f} | {top5_flag} |"
        )

    lines.append("")
    lines.append(
        f"*Positions: {agg['n']}  "
        f"Seed: {_PROBE_SEED}  "
        f"Baseline: {BASELINE_JSON_PATH.name}*"
    )

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Threat-logit probe (step-5k kill criterion).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to probe.")
    parser.add_argument(
        "--positions",
        type=Path,
        default=REPO_ROOT / "fixtures" / "threat_probe_positions.npz",
        help="Path to fixture NPZ (default: fixtures/threat_probe_positions.npz).",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=(
            "Save probe results as the canonical baseline to "
            f"{BASELINE_JSON_PATH}. Use with bootstrap_model.pt."
        ),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Optional: probe this checkpoint and show side-by-side comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to this path instead of stdout.",
    )
    args = parser.parse_args()

    exit_code = 0
    try:
        _set_determinism()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not args.checkpoint.exists():
            print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(2)

        if not args.positions.exists():
            print(f"ERROR: positions file not found: {args.positions}", file=sys.stderr)
            print(
                "Run: python scripts/generate_threat_probe_fixtures.py "
                f"--output {args.positions}",
                file=sys.stderr,
            )
            sys.exit(2)

        print(f"Loading model: {args.checkpoint.name}", file=sys.stderr)
        model = load_model(args.checkpoint, device=device)

        print(f"Loading positions: {args.positions}", file=sys.stderr)
        positions = load_positions(args.positions)
        print(f"  {positions['n']} positions", file=sys.stderr)

        print("Probing...", file=sys.stderr)
        results = probe_positions(model, positions, device=device)
        agg = aggregate(results)

        # Write baseline json before checking pass/fail so the file is available
        # for subsequent probe.latest calls.
        if args.write_baseline:
            save_baseline_json(agg, ckpt_name=args.checkpoint.name)

        # Load saved baseline for PASS evaluation.
        baseline = load_baseline_json()
        if baseline is None:
            print(
                "WARNING: no baseline recorded — run make probe.bootstrap first. "
                "Condition 1 will FAIL.",
                file=sys.stderr,
            )

        # Optional side-by-side baseline checkpoint probe.
        baseline_agg: Optional[Dict] = None
        baseline_name: Optional[str] = None
        if args.baseline_checkpoint is not None and args.baseline_checkpoint.exists():
            print(f"Loading comparison checkpoint: {args.baseline_checkpoint.name}", file=sys.stderr)
            baseline_model = load_model(args.baseline_checkpoint, device=device)
            baseline_results = probe_positions(baseline_model, positions, device=device)
            baseline_agg = aggregate(baseline_results)
            baseline_name = args.baseline_checkpoint.name

        report = format_report(
            results,
            agg,
            ckpt_name=args.checkpoint.name,
            baseline=baseline,
            baseline_agg=baseline_agg,
            baseline_name=baseline_name,
        )

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
            print(f"Report written: {args.output}", file=sys.stderr)
        else:
            print(report)

        overall_pass, conditions = check_pass(agg, baseline=baseline)
        status = "PASS" if overall_pass else "FAIL"
        bl_mean = baseline["ext_logit_mean"] if baseline else float("nan")
        print(
            f"\n{status}  "
            f"[C1] ext_logit_mean={agg['ext_logit_mean']:+.3f} "
            f"(≥ {bl_mean:+.2f} − {BASELINE_MARGIN:.1f} = {bl_mean - BASELINE_MARGIN:+.2f}) "
            f"{'OK' if conditions['ext_logit_vs_baseline'] else 'FAIL'}  "
            f"[C2] contrast={agg['contrast_mean']:+.3f} "
            f"(≥{THRESH_CONTRAST_MEAN}) {'OK' if conditions['contrast'] else 'FAIL'}  "
            f"[C3] top5={agg['ext_in_top5_frac']:.0%} "
            f"(≥{THRESH_EXT_IN_TOP5_PCT:.0f}%) {'OK' if conditions['ext_in_top5'] else 'FAIL'}",
            file=sys.stderr,
        )
        exit_code = 0 if overall_pass else 1

    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        exit_code = 2

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
