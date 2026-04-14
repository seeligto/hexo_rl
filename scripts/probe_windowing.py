"""
Windowing diagnostic probe — answers three questions from a fixed 50-position set.

Q_cov:    Coverage rate of level-4+ threat cells by the network's cluster windows.
Q_anchor: For positions where raw-policy top-1 disagrees with the actual next move
          (MCTS proxy), is the actual next move inside any window?
Q_stability: Across adjacent plies in self-play games, how often does the primary
             anchor (K=0 cluster centre) shift by more than 3 hex cells?

Deliverables:
    scripts/probe_windowing.py          ← this file
    fixtures/windowing_probe_positions.npz
    reports/probes/windowing_<date>.md

Usage:
    python scripts/probe_windowing.py \\
        --checkpoint checkpoints/checkpoint_00020496.pt \\
        --output reports/probes/windowing_20260414.md

Exit codes: 0 = success, 1 = fixture-generation failure, 2 = other error.
"""

from __future__ import annotations

import argparse
import sys
import math
from collections import defaultdict
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
from hexo_rl.eval.windowing_diagnostic import (
    BOARD_SIZE, HALF, SENTINEL,
    hex_dist,
    generate_fixture_positions,
    save_fixture_npz,
    load_fixture_npz,
    replay_board,
    analyse_position,
)

DEFAULT_FIXTURE_PATH = REPO_ROOT / "fixtures" / "windowing_probe_positions.npz"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"


# ── Model loading (mirrors probe_threat_logits.py) ─────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> HexTacToeNet:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = Trainer._extract_model_state(ckpt)
    state = normalize_model_state_dict_keys(state)
    hparams = Trainer._infer_model_hparams(state)

    model = HexTacToeNet(
        board_size=int(hparams.get("board_size", 19)),
        in_channels=int(hparams.get("in_channels", 24)),
        filters=int(hparams.get("filters", 128)),
        res_blocks=int(hparams.get("res_blocks", 12)),
        se_reduction_ratio=int(hparams.get("se_reduction_ratio", 4)),
    )
    model.load_state_dict(state, strict=False)
    return model.float().to(device).eval()


# ── ASCII histogram helper ─────────────────────────────────────────────────────

def ascii_hist(values: List[float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "  (no data)\n"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  all = {lo:.3f}\n"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo_bin = lo + i * step
        hi_bin = lo_bin + step
        bar = "#" * round(c / max_count * width)
        lines.append(f"  [{lo_bin:+6.2f},{hi_bin:+6.2f}) {bar:<{width}} {c}")
    return "\n".join(lines) + "\n"


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def aggregate(results: List[Dict]) -> Dict:
    """Compute Q_cov, Q_anchor, Q_stability from per-position result dicts."""

    # ── Q_cov ──────────────────────────────────────────────────────────────────
    total_threats = 0
    covered_threats = 0
    cov_by_phase: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "covered": 0})
    for r in results:
        ph = r["phase"]
        for _q, _r, _lv, covered in r["threats_l4plus"]:
            total_threats += 1
            cov_by_phase[ph]["total"] += 1
            if covered:
                covered_threats += 1
                cov_by_phase[ph]["covered"] += 1

    q_cov = covered_threats / total_threats if total_threats > 0 else float("nan")

    # ── Q_anchor ───────────────────────────────────────────────────────────────
    disagree_positions = [r for r in results if r["nn_disagrees"]]
    n_disagree = len(disagree_positions)
    n_anchor_covered = sum(
        1 for r in disagree_positions if r["next_covered"] is True
    )
    q_anchor = n_anchor_covered / n_disagree if n_disagree > 0 else float("nan")

    # Per-phase breakdown for Q_anchor
    anchor_by_phase: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "covered": 0})
    for r in disagree_positions:
        ph = r["phase"]
        anchor_by_phase[ph]["total"] += 1
        if r["next_covered"]:
            anchor_by_phase[ph]["covered"] += 1

    # ── Q_stability ────────────────────────────────────────────────────────────
    # Group positions by game_id; sort by ply; find consecutive (ply N, ply N+1) pairs
    by_game: Dict[int, List[Dict]] = defaultdict(list)
    for r in results:
        by_game[r["game_id"]].append(r)
    for lst in by_game.values():
        lst.sort(key=lambda x: x["ply"])

    churn_events = 0
    total_adjacent = 0
    anchor_shifts: List[float] = []

    for gid, positions in by_game.items():
        for i in range(len(positions) - 1):
            a = positions[i]
            b = positions[i + 1]
            # Only count truly adjacent plies
            if b["ply"] != a["ply"] + 1:
                continue
            total_adjacent += 1

            # Primary anchor = centers[0] for each position
            if not a["centers"] or not b["centers"]:
                continue
            aq, ar = a["centers"][0]
            bq, br = b["centers"][0]
            shift = hex_dist(aq, ar, bq, br)
            anchor_shifts.append(float(shift))
            if shift > 3:
                churn_events += 1

    q_stability = churn_events / total_adjacent if total_adjacent > 0 else float("nan")

    return {
        "q_cov": q_cov,
        "q_cov_total_threats": total_threats,
        "q_cov_covered_threats": covered_threats,
        "q_cov_by_phase": dict(cov_by_phase),
        "q_anchor": q_anchor,
        "q_anchor_n_disagree": n_disagree,
        "q_anchor_covered": n_anchor_covered,
        "q_anchor_by_phase": dict(anchor_by_phase),
        "q_stability": q_stability,
        "q_stability_churn": churn_events,
        "q_stability_total_adjacent": total_adjacent,
        "anchor_shifts": anchor_shifts,
        "n_positions": len(results),
    }


# ── Markdown report ────────────────────────────────────────────────────────────

def _pct(n: float, d: float) -> str:
    if d == 0:
        return "n/a"
    return f"{100 * n / d:.1f}%"


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "n/a (no data)"
    return f"{v:.3f} ({100*v:.1f}%)"


def build_report(
    agg: Dict,
    results: List[Dict],
    ckpt_name: str,
    fixture_path: str,
) -> str:
    lines = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines += [
        f"# Windowing Diagnostic Probe — {ts}",
        "",
        f"**Checkpoint:** `{ckpt_name}`",
        f"**Fixture:** `{fixture_path}`",
        f"**Positions:** {agg['n_positions']} total",
        "",
        "---",
        "",
    ]

    # ── Q_cov ──────────────────────────────────────────────────────────────────
    lines += [
        "## Q_cov — Threat Coverage",
        "",
        "For every level-4+ threat cell detected by `board.get_threats()`, "
        "is the cell inside at least one cluster window?",
        "",
        f"**Overall coverage: {_fmt(agg['q_cov'])}** "
        f"({agg['q_cov_covered_threats']} / {agg['q_cov_total_threats']} threat cells)",
        "",
        "### Per-phase breakdown",
        "",
        "| Phase      | Threats | Covered | Rate |",
        "|------------|---------|---------|------|",
    ]
    for ph in ["empty", "early_mid", "mid_late"]:
        d = agg["q_cov_by_phase"].get(ph, {"total": 0, "covered": 0})
        lines.append(
            f"| {ph:<10} | {d['total']:>7} | {d['covered']:>7} | "
            f"{_pct(d['covered'], d['total']):>4} |"
        )
    lines += [""]

    if agg["q_cov_total_threats"] == 0:
        lines.append(
            "_No level-4+ threats in probe set — "
            "positions may be too early-game for threat detection._"
        )
    lines += [""]

    # ── Q_anchor ───────────────────────────────────────────────────────────────
    lines += [
        "## Q_anchor — Policy-Disagreement Coverage",
        "",
        "When raw-policy top-1 disagrees with the actual next move "
        "(MCTS proxy), is the actual next move inside any cluster window?",
        "",
        f"**Disagreement positions: {agg['q_anchor_n_disagree']} / {agg['n_positions']}**",
        f"**Coverage of disagreement cell: {_fmt(agg['q_anchor'])}** "
        f"({agg['q_anchor_covered']} / {agg['q_anchor_n_disagree']})",
        "",
        "### Per-phase breakdown",
        "",
        "| Phase      | Disagree | Covered | Rate |",
        "|------------|----------|---------|------|",
    ]
    for ph in ["empty", "early_mid", "mid_late"]:
        d = agg["q_anchor_by_phase"].get(ph, {"total": 0, "covered": 0})
        lines.append(
            f"| {ph:<10} | {d['total']:>8} | {d['covered']:>7} | "
            f"{_pct(d['covered'], d['total']):>4} |"
        )
    lines += [""]

    # ── Q_stability ────────────────────────────────────────────────────────────
    lines += [
        "## Q_stability — Anchor Churn",
        "",
        "Across adjacent plies in self-play games, how often does the "
        "primary anchor (K=0 cluster centre) shift by > 3 hex cells?",
        "",
        f"**Adjacent ply pairs analysed: {agg['q_stability_total_adjacent']}**",
        f"**Churn events (shift > 3): {agg['q_stability_churn']}**",
        f"**Anchor-churn rate: {_fmt(agg['q_stability'])}**",
        "",
    ]

    shifts = agg["anchor_shifts"]
    if shifts:
        lines += [
            "### Anchor-shift distribution (hex cells)",
            "",
            "```",
            ascii_hist(shifts, bins=8, width=30),
            "```",
            f"mean={sum(shifts)/len(shifts):.2f}  "
            f"median={sorted(shifts)[len(shifts)//2]:.2f}  "
            f"max={max(shifts):.0f}",
            "",
        ]
    else:
        lines += [
            "_No adjacent ply pairs found in probe set._",
            "(Positions from different games; increase fixture size for stability data.)",
            "",
        ]

    # ── Per-position dump ──────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Per-Position Detail",
        "",
    ]
    for i, r in enumerate(results):
        centers_str = ", ".join(f"({cq},{cr})" for cq, cr in r["centers"])
        lines.append(f"### Position {i+1} — ply={r['ply']}  phase={r['phase']}  K={r['K']}")
        lines.append(f"- Anchors: {centers_str}")

        bboxes_str = ", ".join(
            f"q∈[{q0},{q1}] r∈[{r0},{r1}]"
            for q0, q1, r0, r1 in r["bboxes"]
        )
        lines.append(f"- Bboxes:  {bboxes_str}")

        if r["threats_l4plus"]:
            threat_lines = [
                f"({q},{r}) lv{lv} {'✓' if cov else '✗'}"
                for q, r, lv, cov in r["threats_l4plus"]
            ]
            lines.append(f"- Threats L4+: {'; '.join(threat_lines)}")
        else:
            lines.append("- Threats L4+: none")

        if r["top5_policy"]:
            pol_lines = [
                f"({q},{r}) {'in' if cov else 'OUT'} lp={lp:.2f}"
                for q, r, cov, lp in r["top5_policy"]
            ]
            lines.append(f"- Top-5 policy: {' | '.join(pol_lines)}")

        if r["next_move"] is not None:
            nq, nr = r["next_move"]
            nn_q, nn_r = r["nn_top1"] if r["nn_top1"] else (None, None)
            disagree = "DISAGREE" if r["nn_disagrees"] else "agree"
            cov_str = "covered" if r["next_covered"] else "NOT COVERED"
            lines.append(
                f"- Next move: ({nq},{nr}) [{cov_str}] | NN top-1: ({nn_q},{nn_r}) | {disagree}"
            )
        lines.append("")

    # ── Verdict ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Verdict",
        "",
    ]

    issues: List[str] = []
    notes: List[str] = []

    if not math.isnan(agg["q_cov"]):
        if agg["q_cov"] < 0.80:
            issues.append(
                f"Q_cov = {agg['q_cov']:.1%} < 80%: windowing truncates "
                f">20% of level-4+ threats from the network's view."
            )
        else:
            notes.append(f"Q_cov = {agg['q_cov']:.1%} ✓")
    else:
        notes.append("Q_cov: no level-4+ threats in probe set (too early-game).")

    if not math.isnan(agg["q_anchor"]):
        if agg["q_anchor"] < 0.80:
            issues.append(
                f"Q_anchor = {agg['q_anchor']:.1%} < 80%: decisive cells "
                f"often lie outside all cluster windows."
            )
        else:
            notes.append(f"Q_anchor = {agg['q_anchor']:.1%} ✓")
    else:
        notes.append("Q_anchor: no policy-disagreement positions observed.")

    if not math.isnan(agg["q_stability"]):
        if agg["q_stability"] > 0.25:
            issues.append(
                f"Q_stability churn = {agg['q_stability']:.1%} > 25%: "
                f"primary anchor jumps frequently, making temporal history "
                f"inconsistent across plies."
            )
        else:
            notes.append(f"Q_stability churn = {agg['q_stability']:.1%} ✓")
    else:
        notes.append("Q_stability: insufficient adjacent-ply pairs for measurement.")

    if issues:
        lines.append("**FINDINGS (require investigation):**")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("**No critical windowing failures detected.**")
    lines.append("")
    if notes:
        lines.append("**Passing:**")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Windowing diagnostic probe (Q_cov, Q_anchor, Q_stability)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "checkpoint_00020496.pt",
        help="Model checkpoint to probe.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="NPZ fixture file (generated if absent).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory of self-play run dirs (for fixture generation).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=50,
        help="Number of probe positions (default: 50).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = REPO_ROOT / "reports" / "probes" / f"windowing_{ts}.md"

    # ── Validate checkpoint ────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    # ── Generate fixture if needed ─────────────────────────────────────────────
    if not args.fixture.exists():
        print(f"[probe_windowing] fixture not found — generating {args.n_positions} positions…")
        positions_raw = generate_fixture_positions(
            args.runs_root, n_total=args.n_positions, seed=args.seed
        )
        if not positions_raw:
            print("ERROR: could not generate any fixture positions.", file=sys.stderr)
            return 1
        save_fixture_npz(positions_raw, args.fixture)

    positions = load_fixture_npz(args.fixture)
    print(f"[probe_windowing] loaded {len(positions)} fixture positions from {args.fixture}")

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe_windowing] loading {args.checkpoint.name} on {device}…")
    model = load_model(args.checkpoint, device)
    print("[probe_windowing] model loaded.")

    # ── Run probe ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    results: List[Dict] = []
    for i, pos in enumerate(positions):
        board, state = replay_board(pos["moves_q"], pos["moves_r"], pos["n_moves"])
        result = analyse_position(board, state, model, device, pos)
        results.append(result)

        K = result["K"]
        n_threats = len(result["threats_l4plus"])
        cov_threats = sum(1 for *_, cov in result["threats_l4plus"] if cov)
        print(
            f"  pos {i+1:2d}/{len(positions)}  ply={pos['n_moves']:3d}  "
            f"phase={pos['phase']:<10}  K={K}  "
            f"threats-L4+={n_threats}  covered={cov_threats}  "
            f"nn_disagrees={result['nn_disagrees']}"
        )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    agg = aggregate(results)

    print()
    print("=" * 60)
    print(f"Q_cov    = {_fmt(agg['q_cov'])}  "
          f"({agg['q_cov_covered_threats']}/{agg['q_cov_total_threats']} threat cells)")
    print(f"Q_anchor = {_fmt(agg['q_anchor'])}  "
          f"({agg['q_anchor_covered']}/{agg['q_anchor_n_disagree']} disagree positions covered)")
    print(f"Q_stability churn = {_fmt(agg['q_stability'])}  "
          f"({agg['q_stability_churn']}/{agg['q_stability_total_adjacent']} adjacent pairs)")
    print("=" * 60)

    # ── Write report ───────────────────────────────────────────────────────────
    report = build_report(agg, results, args.checkpoint.name, str(args.fixture))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"\nReport → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
