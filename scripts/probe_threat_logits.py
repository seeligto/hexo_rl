"""
Threat-logit probe — step-5k kill criterion for HeXO Phase 4.0 sustained runs.

Loads a checkpoint, runs 20 curated positions through the threat head, and
reports whether the policy head has learned to point at genuine spatial
extension cells (i.e. is NOT colony-spamming).

Kill criterion (§85 / §89 of docs/07_PHASE4_SPRINT_LOG.md, revised §91):
  At step 5000, re-run this probe. PASS requires ALL THREE of C1-C3.
  C4 is a warning only — it never causes FAIL.

    C1: contrast_mean >= max(0.38, 0.8 × bootstrap_contrast)
        Position-conditional sharpness must be at least 80% of bootstrap.
        The 0.38 floor preserves the original §85 absolute minimum.
    C2: ext_in_top5_pct  >= 25.0
        Policy head must rank the extension cell in the top-5 spatial moves
        on at least 25% of probe positions. Threshold softened from 40% for
        24-plane model at step 5k (thresholds were calibrated against 18-plane
        bootstrap; 24-plane model needs more steps to match sharpness).
    C3: ext_in_top10_pct >= 40.0
        Looser top-K catches partial sharpness — if extension is rank 6-10
        the policy head is not colony-spamming, just under-sharpened.
        Threshold softened from 60% for same reason as C2.
    C4 (WARNING): abs(ext_logit_mean - bootstrap_ext_logit_mean) < 5.0
        Catches catastrophic decode/mapping bugs without gating training.
        Drift > 5.0 nats prints a WARNING line in the report; does not fail.

  Bootstrap baseline numbers come from fixtures/threat_probe_baseline.json
  (schema v2), written once by `make probe.bootstrap --write-baseline`. If
  that file is absent, probe prints FAIL with "no baseline recorded".

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

# ── Kill-criterion thresholds (§85 / §89, revised §91) ──────────────────────

THRESH_CONTRAST_FLOOR: float = 0.38      # absolute floor for contrast_mean
THRESH_CONTRAST_BOOTSTRAP_FRAC: float = 0.8  # contrast must reach 80% of bootstrap
THRESH_EXT_IN_TOP5_PCT: float = 25.0     # extension cell in policy top-5 ≥ 25% (softened from 40% for 24-plane model at step 5k)
THRESH_EXT_IN_TOP10_PCT: float = 40.0    # extension cell in policy top-10 ≥ 40% (softened from 60% for 24-plane model at step 5k)
THRESH_EXT_LOGIT_DRIFT_WARN: float = 5.0  # |Δ ext_logit_mean| > 5.0 → warning only

BASELINE_SCHEMA_VERSION: int = 4

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
    """Persist aggregate results as the canonical baseline JSON (schema v2)."""
    record = {
        "version": BASELINE_SCHEMA_VERSION,
        "ext_logit_mean": agg["ext_logit_mean"],
        "ext_logit_std": agg["ext_logit_std"],
        "ctrl_logit_mean": agg["ctrl_logit_mean"],
        "ctrl_logit_std": agg["ctrl_logit_std"],
        "contrast_mean": agg["contrast_mean"],
        "contrast_std": agg["contrast_std"],
        "ext_in_top5_frac": agg["ext_in_top5_frac"],
        "ext_in_top5_pct": agg["ext_in_top5_frac"] * 100.0,
        "ext_in_top10_frac": agg["ext_in_top10_frac"],
        "ext_in_top10_pct": agg["ext_in_top10_frac"] * 100.0,
        "n": agg["n"],
        "checkpoint": ckpt_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
    state_fp16: np.ndarray,   # (24, 19, 19) float16
    ext_flat: int,
    ctrl_flat: int,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    """Forward one position in FP32; return (ext_logit, ctrl_logit, contrast, top5, top10)."""
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
    top10_indices = policy_spatial.topk(min(10, len(policy_spatial))).indices.tolist()
    top5_indices = top10_indices[:5]

    return ext_logit, ctrl_logit, contrast, top5_indices, top10_indices


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

        ext_logit, ctrl_logit, contrast, top5, top10 = _probe_one(
            model, states[i], ext_flat, ctrl_flat, device
        )

        ext_in_top5 = ext_flat in top5
        ext_in_top10 = ext_flat in top10

        results.append({
            "idx": i,
            "game_phase": str(phases[i]) if hasattr(phases[i], "__str__") else "unknown",
            "ext_flat": ext_flat,
            "ctrl_flat": ctrl_flat,
            "ext_logit": ext_logit,
            "ctrl_logit": ctrl_logit,
            "contrast": contrast,
            "policy_top5": top5,
            "policy_top10": top10,
            "ext_in_policy_top5": ext_in_top5,
            "ext_in_policy_top10": ext_in_top10,
        })

    return results


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate(results: List[Dict]) -> Dict:
    ext_logits = np.array([r["ext_logit"] for r in results], dtype=np.float64)
    ctrl_logits = np.array([r["ctrl_logit"] for r in results], dtype=np.float64)
    contrasts = np.array([r["contrast"] for r in results], dtype=np.float64)
    ext_in_top5_count = sum(1 for r in results if r["ext_in_policy_top5"])
    ext_in_top10_count = sum(
        1 for r in results if r.get("ext_in_policy_top10", r["ext_in_policy_top5"])
    )

    n = max(len(results), 1)
    return {
        "n": len(results),
        "ext_logit_mean": float(ext_logits.mean()),
        "ext_logit_std": float(ext_logits.std()),
        "ctrl_logit_mean": float(ctrl_logits.mean()),
        "ctrl_logit_std": float(ctrl_logits.std()),
        "contrast_mean": float(contrasts.mean()),
        "contrast_std": float(contrasts.std()),
        "ext_in_top5_frac": ext_in_top5_count / n,
        "ext_in_top10_frac": ext_in_top10_count / n,
    }


# ── Pass / fail decision ──────────────────────────────────────────────────────


def contrast_floor(baseline: Optional[Dict]) -> float:
    """C1 floor: max(0.38, 0.8 × bootstrap_contrast). Returns 0.38 if no baseline."""
    if baseline is None:
        return THRESH_CONTRAST_FLOOR
    bootstrap_contrast = float(baseline.get("contrast_mean", 0.0))
    return max(THRESH_CONTRAST_FLOOR, THRESH_CONTRAST_BOOTSTRAP_FRAC * bootstrap_contrast)


def check_pass(
    agg: Dict,
    baseline: Optional[Dict] = None,
) -> Tuple[bool, Dict[str, bool]]:
    """Evaluate three-condition PASS rule (§85 / §89 revised §91).

    Returns (overall_pass, per_condition_dict).

    All of C1, C2, C3 must pass for overall PASS. C4 is a warning only and
    is reported separately via :func:`check_warning`.

    C1: contrast_mean >= max(0.38, 0.8 × bootstrap_contrast)
    C2: ext_in_top5_pct  >= 25.0
    C3: ext_in_top10_pct >= 40.0

    Conditions never require a baseline to FAIL — C1 falls back to the 0.38
    absolute floor when baseline is None. (Without a baseline we cannot check
    the warning, but the gate still functions.)
    """
    floor = contrast_floor(baseline)
    c1 = agg["contrast_mean"] >= floor
    c2 = agg["ext_in_top5_frac"] * 100.0 >= THRESH_EXT_IN_TOP5_PCT
    c3 = agg.get("ext_in_top10_frac", 0.0) * 100.0 >= THRESH_EXT_IN_TOP10_PCT

    conditions = {
        "contrast": c1,
        "ext_in_top5": c2,
        "ext_in_top10": c3,
    }
    return (c1 and c2 and c3), conditions


def check_warning(
    agg: Dict,
    baseline: Optional[Dict] = None,
) -> Tuple[bool, float]:
    """C4 warning check: |Δ ext_logit_mean| < 5.0.

    Returns (warning_triggered, drift). `warning_triggered` is True when drift
    exceeds the threshold (i.e. the metric is suspicious). Returns (False, 0.0)
    if no baseline is loaded (cannot compute drift).
    """
    if baseline is None:
        return False, 0.0
    drift = abs(agg["ext_logit_mean"] - float(baseline["ext_logit_mean"]))
    return drift >= THRESH_EXT_LOGIT_DRIFT_WARN, drift


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
    """Render markdown report with three explicit PASS conditions + C4 warning."""
    lines: List[str] = []

    overall_pass, conditions = check_pass(agg, baseline=baseline)
    warn_triggered, drift = check_warning(agg, baseline=baseline)
    verdict = "**PASS**" if overall_pass else "**FAIL**"
    lines.append(f"# Threat-Logit Probe: {ckpt_name}\n")
    lines.append(f"Verdict: {verdict}\n")

    # Three explicit conditions
    lines.append("## Pass conditions (§85 / §89 revised §91 kill criterion)\n")
    lines.append("| # | condition | threshold | value | result |")
    lines.append("|---|-----------|-----------|-------|--------|")

    floor = contrast_floor(baseline)
    if baseline is not None:
        bl_contrast = float(baseline.get("contrast_mean", 0.0))
        c1_thresh = (
            f"≥ max(0.38, 0.8 × {bl_contrast:+.3f}) = {floor:+.3f}"
        )
    else:
        c1_thresh = f"≥ {THRESH_CONTRAST_FLOOR:+.3f} (no baseline; floor only)"
    c1_val = f"{agg['contrast_mean']:+.3f}"
    c1_res = "PASS" if conditions["contrast"] else "FAIL"

    c2_thresh = f"≥ {THRESH_EXT_IN_TOP5_PCT:.0f}%"
    c2_val = f"{agg['ext_in_top5_frac'] * 100:.0f}%"
    c2_res = "PASS" if conditions["ext_in_top5"] else "FAIL"

    c3_thresh = f"≥ {THRESH_EXT_IN_TOP10_PCT:.0f}%"
    c3_val = f"{agg.get('ext_in_top10_frac', 0.0) * 100:.0f}%"
    c3_res = "PASS" if conditions["ext_in_top10"] else "FAIL"

    lines.append(f"| 1 | contrast_mean (ext − ctrl) | {c1_thresh} | {c1_val} | **{c1_res}** |")
    lines.append(f"| 2 | ext cell in policy top-5  | {c2_thresh} | {c2_val} | **{c2_res}** |")
    lines.append(f"| 3 | ext cell in policy top-10 | {c3_thresh} | {c3_val} | **{c3_res}** |")
    lines.append("")

    # C4 warning row (informational only)
    if baseline is not None:
        bl_ext = float(baseline["ext_logit_mean"])
        warn_state = "WARNING" if warn_triggered else "ok"
        lines.append(
            f"**C4 (warning, not gated):** |Δ ext_logit_mean| = "
            f"|{agg['ext_logit_mean']:+.3f} − {bl_ext:+.3f}| = {drift:.3f} "
            f"(threshold {THRESH_EXT_LOGIT_DRIFT_WARN:.1f}) → **{warn_state}**\n"
        )
        if warn_triggered:
            lines.append(
                "> WARNING: ext_logit_mean drifted ≥ 5.0 nats from bootstrap. "
                "Probe still **passes** if C1-C3 are met, but investigate for "
                "BCE scale drift or decode/mapping bugs.\n"
            )
    else:
        lines.append("**C4 (warning):** skipped — no baseline loaded.\n")

    # Summary metrics table (optionally compare with baseline_agg for historic display)
    ref_agg = baseline_agg if baseline_agg is not None else baseline
    ref_name = baseline_name or (baseline.get("checkpoint") if baseline else None)

    def _ref_top10(d: Dict) -> float:
        return float(d.get("ext_in_top10_frac", d.get("ext_in_top10_pct", 0.0) / 100.0))

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
        lines.append(
            f"| extension cell in policy top-10 | "
            f"{agg.get('ext_in_top10_frac', 0.0):.0%} | "
            f"{_ref_top10(ref_agg):.0%} |"
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
        lines.append(
            f"| extension cell in policy top-10 | "
            f"{agg.get('ext_in_top10_frac', 0.0):.0%} |"
        )

    lines.append("")

    # Per-position detail
    lines.append("## Per-position detail\n")
    lines.append("| # | phase | ext_logit | ctrl_logit | contrast | ext∈top5 | ext∈top10 |")
    lines.append("|---|-------|-----------|------------|----------|----------|-----------|")
    for r in results:
        top5_flag = "yes" if r["ext_in_policy_top5"] else "no"
        top10_flag = "yes" if r.get("ext_in_policy_top10", False) else "no"
        lines.append(
            f"| {r['idx']+1} | {r['game_phase']} | "
            f"{r['ext_logit']:+.3f} | {r['ctrl_logit']:+.3f} | "
            f"{r['contrast']:+.3f} | {top5_flag} | {top10_flag} |"
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
    parser.add_argument(
        "--zero-chain-planes",
        action="store_true",
        default=False,
        help=(
            "Experiment C ablation: zero input planes 18-23 (chain-length planes) "
            "before running inference. Use with Experiment C checkpoints."
        ),
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

        # Experiment C: zero chain-length input planes before inference.
        if args.zero_chain_planes:
            positions = dict(positions)
            states_zeroed = positions["states"].copy()
            states_zeroed[:, 18:24] = 0
            positions["states"] = states_zeroed
            print("  [Experiment C] planes 18-23 zeroed", file=sys.stderr)

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
                "C1 will fall back to the absolute 0.38 floor; C4 warning skipped.",
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
        warn_triggered, drift = check_warning(agg, baseline=baseline)
        status = "PASS" if overall_pass else "FAIL"
        floor = contrast_floor(baseline)
        print(
            f"\n{status}  "
            f"[C1] contrast={agg['contrast_mean']:+.3f} "
            f"(≥{floor:+.3f}) {'OK' if conditions['contrast'] else 'FAIL'}  "
            f"[C2] top5={agg['ext_in_top5_frac']:.0%} "
            f"(≥{THRESH_EXT_IN_TOP5_PCT:.0f}%) {'OK' if conditions['ext_in_top5'] else 'FAIL'}  "
            f"[C3] top10={agg.get('ext_in_top10_frac', 0.0):.0%} "
            f"(≥{THRESH_EXT_IN_TOP10_PCT:.0f}%) {'OK' if conditions['ext_in_top10'] else 'FAIL'}",
            file=sys.stderr,
        )
        if baseline is not None:
            warn_label = "WARN" if warn_triggered else "ok"
            print(
                f"[C4] |Δ ext_logit_mean|={drift:.3f} "
                f"(<{THRESH_EXT_LOGIT_DRIFT_WARN:.1f}) {warn_label}",
                file=sys.stderr,
            )
        if args.write_baseline:
            # Baseline-set mode: the PASS/FAIL verdict is diagnostic-only. The
            # bootstrap's threat_head is untrained (pretrain doesn't feed
            # winning_line targets), so its contrast is essentially random and
            # the absolute 0.38 floor doesn't apply — future probes compare
            # against this recorded baseline, not against an absolute floor.
            # See sprint §92 for the Q13 landing context.
            if not overall_pass:
                print(
                    "NOTE: running in --write-baseline mode; FAIL verdict is "
                    "informational only — baseline written as v3 reference for "
                    "future probes. Exiting 0.",
                    file=sys.stderr,
                )
            exit_code = 0
        else:
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
