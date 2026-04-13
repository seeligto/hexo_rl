"""
Threat-logit probe — step-5k kill criterion for HeXO Phase 4.0 sustained runs.

Loads a checkpoint, runs 20 curated positions through the threat head, and
reports whether the threat head has learned genuine spatial signal (extension
cells score higher than random empty cells).

Kill criterion (§85 of docs/07_PHASE4_SPRINT_LOG.md):
  At step 5000, re-run this probe.
  PASS = mean extension-cell logit > 0  AND  mean contrast ≥ 0.38
  FAIL = either condition missed → investigate before continuing.

Exit codes:
  0  PASS
  1  FAIL
  2  error (bad checkpoint, missing fixture file, shape mismatch, etc.)

Usage:
  python scripts/probe_threat_logits.py --checkpoint checkpoints/bootstrap_model.pt
  python scripts/probe_threat_logits.py --checkpoint <ckpt> \\
      --baseline-checkpoint checkpoints/bootstrap_model.pt \\
      --output reports/probes/step5000.md
"""

from __future__ import annotations

import argparse
import math
import sys
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

# ── Kill-criterion thresholds (§85) ──────────────────────────────────────────

THRESH_EXT_LOGIT_MEAN: float = 0.0   # mean extension-cell logit must be > 0
THRESH_CONTRAST_MEAN: float = 0.38   # mean contrast must be ≥ 0.38

# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(
    ckpt_path: Path,
    device: Optional[torch.device] = None,
) -> HexTacToeNet:
    """Load checkpoint → HexTacToeNet in eval mode.

    Uses Trainer._extract_model_state + _infer_model_hparams + strict=False,
    matching the pattern established in diagnose_policy_sharpness.py.
    """
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
    model = model.to(device).eval()
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
    """Forward one position; return (ext_logit, ctrl_logit, contrast, policy_top5_indices)."""
    x = torch.from_numpy(state_fp16.astype(np.float16)).unsqueeze(0).to(device)

    use_amp = device.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x, threat=True)
        else:
            # CPU: run in float32 for numerical stability
            x = x.float()
            out = model(x, threat=True)

    # out = (log_policy, value, v_logit, [opp_reply, sigma2, ownership,] threat_logits)
    # threat head is always appended last when threat=True
    threat_logits = out[-1]  # (1, 1, 19, 19)
    log_policy = out[0]       # (1, H*W + 1)

    threat_flat = threat_logits[0, 0].float().cpu().flatten()  # (361,)

    ext_logit = float(threat_flat[ext_flat].item())
    ctrl_logit = float(threat_flat[ctrl_flat].item())
    contrast = ext_logit - ctrl_logit

    # Policy top-5 (spatial cells only, ignore pass token)
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


def check_pass(agg: Dict) -> bool:
    """PASS = mean extension logit > 0 AND mean contrast ≥ 0.38 (§85 kill criterion)."""
    return (
        agg["ext_logit_mean"] > THRESH_EXT_LOGIT_MEAN
        and agg["contrast_mean"] >= THRESH_CONTRAST_MEAN
    )


# ── Report formatting ─────────────────────────────────────────────────────────


def _fmt(val: float, std: Optional[float] = None) -> str:
    if std is not None:
        return f"{val:+.2f} ± {std:.2f}"
    return f"{val:+.2f}"


def format_report(
    results: List[Dict],
    agg: Dict,
    ckpt_name: str,
    baseline_agg: Optional[Dict] = None,
    baseline_name: Optional[str] = None,
) -> str:
    """Render markdown report matching §85 table structure."""
    lines: List[str] = []

    verdict = "**PASS**" if check_pass(agg) else "**FAIL**"
    lines.append(f"# Threat-Logit Probe: {ckpt_name}\n")
    lines.append(f"Verdict: {verdict}\n")
    lines.append(
        f"Kill criterion (§85): ext_logit_mean > {THRESH_EXT_LOGIT_MEAN} "
        f"AND contrast_mean ≥ {THRESH_CONTRAST_MEAN}\n"
    )

    # Summary table
    if baseline_agg is not None and baseline_name is not None:
        lines.append(
            f"| metric | {ckpt_name} | {baseline_name} |"
        )
        lines.append("|--------|------------|-------------|")
        lines.append(
            f"| threat logit @ extension cell | "
            f"{_fmt(agg['ext_logit_mean'], agg['ext_logit_std'])} | "
            f"{_fmt(baseline_agg['ext_logit_mean'], baseline_agg['ext_logit_std'])} |"
        )
        lines.append(
            f"| threat logit @ control cell | "
            f"{_fmt(agg['ctrl_logit_mean'], agg['ctrl_logit_std'])} | "
            f"{_fmt(baseline_agg['ctrl_logit_mean'], baseline_agg['ctrl_logit_std'])} |"
        )
        lines.append(
            f"| contrast (extension − control) | "
            f"{_fmt(agg['contrast_mean'])} | "
            f"{_fmt(baseline_agg['contrast_mean'])} |"
        )
        lines.append(
            f"| extension cell in policy top-5 | "
            f"{agg['ext_in_top5_frac']:.0%} | "
            f"{baseline_agg['ext_in_top5_frac']:.0%} |"
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

    # Per-position detail table
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
        f"*Positions: {agg['n']}   "
        f"Thresholds: ext_logit_mean > {THRESH_EXT_LOGIT_MEAN}, "
        f"contrast_mean ≥ {THRESH_CONTRAST_MEAN}*"
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
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Optional baseline checkpoint for side-by-side comparison.",
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

        baseline_agg: Optional[Dict] = None
        baseline_name: Optional[str] = None
        if args.baseline_checkpoint is not None and args.baseline_checkpoint.exists():
            print(f"Loading baseline: {args.baseline_checkpoint.name}", file=sys.stderr)
            baseline_model = load_model(args.baseline_checkpoint, device=device)
            baseline_results = probe_positions(baseline_model, positions, device=device)
            baseline_agg = aggregate(baseline_results)
            baseline_name = args.baseline_checkpoint.name

        report = format_report(
            results,
            agg,
            ckpt_name=args.checkpoint.name,
            baseline_agg=baseline_agg,
            baseline_name=baseline_name,
        )

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
            print(f"Report written: {args.output}", file=sys.stderr)
        else:
            print(report)

        passed = check_pass(agg)
        status = "PASS" if passed else "FAIL"
        print(
            f"\n{status}  ext_logit_mean={agg['ext_logit_mean']:+.3f} "
            f"(>{THRESH_EXT_LOGIT_MEAN})  "
            f"contrast_mean={agg['contrast_mean']:+.3f} "
            f"(≥{THRESH_CONTRAST_MEAN})",
            file=sys.stderr,
        )
        exit_code = 0 if passed else 1

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
