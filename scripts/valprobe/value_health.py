"""D-C VALPROBE WP4 — value-head health trend across banked run2 checkpoints.

SOLVER-INDEPENDENT: no T_provable, no tactics solver. Forward-only.

Measures per banked ckpt (50k → latest):
  (a) value ECE (Expected Calibration Error, 10-bin, P(win) = (v+1)/2)
  (b) decided-row accuracy  sign(v_t) == sign(outcome), restricted to |v_t| > margin
  (c) value MAE = mean|v_t - outcome|  where outcome ∈ {+1, -1}
  (d) phase split (early / late halves) for context

Skips draws/censored games (no clean ground truth).
Samples positions at HEAD TURN-STARTS only (same §4.1 rationale as WP1:
value head is min-pooled to-move perspective + plane-2 moves_remaining confound).

Output schema (STABLE — run3 external eval consumes):
  {ckpt_step, ckpt_sha, radius, book_id, n_positions,
   value_ece, decided_accuracy, decided_margin, value_mae,
   n_win_games, n_loss_games}

CLI::

    .venv/bin/python scripts/valprobe/value_health.py \\
      [--out reports/valprobe/value_health_series.jsonl] \\
      [--no-plot]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# ── constants ─────────────────────────────────────────────────────────────────

EXPECT_ENCODING = "v6_live2_ls"
ECE_N_BINS = 10
DECIDED_MARGIN = 0.05   # |v_t| > this to count as "decided"

# Ordered ckpt inventory: (step, ckpt_rel_path, games_rel_dir)
# 175k lives in scripts/arena/weights/run2_175k.pt; games in retro_slope/run2_175k/
CKPT_INVENTORY: List[Tuple[int, str, str]] = [
    (50000,  "checkpoints/run2_retro/checkpoint_00050000.pt",
             "reports/evalfair/retro_slope/checkpoint_00050000"),
    (70000,  "checkpoints/run2_retro/checkpoint_00070000.pt",
             "reports/evalfair/retro_slope/checkpoint_00070000"),
    (90000,  "checkpoints/run2_retro/checkpoint_00090000.pt",
             "reports/evalfair/retro_slope/checkpoint_00090000"),
    (110000, "checkpoints/run2_retro/checkpoint_00110000.pt",
             "reports/evalfair/retro_slope/checkpoint_00110000"),
    (130000, "checkpoints/run2_retro/checkpoint_00130000.pt",
             "reports/evalfair/retro_slope/checkpoint_00130000"),
    (150000, "checkpoints/run2_retro/checkpoint_00150000.pt",
             "reports/evalfair/retro_slope/checkpoint_00150000"),
    (170000, "checkpoints/run2_retro/checkpoint_00170000.pt",
             "reports/evalfair/retro_slope/checkpoint_00170000"),
    (175000, "scripts/arena/weights/run2_175k.pt",
             "reports/evalfair/retro_slope/run2_175k"),
    (195000, "checkpoints/run2_retro/checkpoint_00195000.pt",
             "reports/evalfair/retro_slope/checkpoint_00195000"),
    (200000, "checkpoints/run2_retro/checkpoint_00200000.pt",
             "reports/evalfair/retro_slope/checkpoint_00200000"),
    (210000, "checkpoints/run2_retro/checkpoint_00210000.pt",
             "reports/evalfair/retro_slope/checkpoint_00210000"),
    (220000, "checkpoints/run2_retro/checkpoint_00220000.pt",
             "reports/evalfair/retro_slope/checkpoint_00220000"),
    (230000, "checkpoints/run2_retro/checkpoint_00230000.pt",
             "reports/evalfair/retro_slope/checkpoint_00230000"),
    (240000, "checkpoints/run2_retro/checkpoint_00240000.pt",
             "reports/evalfair/retro_slope/checkpoint_00240000"),
    (248000, "checkpoints/run2_retro/checkpoint_00248000.pt",
             "reports/evalfair/retro_slope/checkpoint_00248000"),
    # 272k: stamped v6_live2 (not _ls) — will fail gated loader; skip with log
    (272357, "checkpoints/checkpoint_00272357.pt",
             None),  # no retro_slope dir either
]


# ── sha util ──────────────────────────────────────────────────────────────────


def ckpt_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── head-turn-start filter (mirrors WP1 §4.1) ────────────────────────────────


def is_head_turn_start(cp: int, mr: int, ply: int, head_pn: int) -> bool:
    """True iff position is a head-to-move turn-start."""
    is_turn_start = (ply == 0) or (mr == 2)
    return is_turn_start and (cp == head_pn)


# ── game loading ──────────────────────────────────────────────────────────────


def load_games_jsonl(path: str) -> List[Dict]:
    games = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


# ── value ECE ────────────────────────────────────────────────────────────────


def compute_ece(v_vals: List[float], outcomes: List[float], n_bins: int = ECE_N_BINS) -> float:
    """Expected Calibration Error (equal-width bins on P_win = (v+1)/2).

    Args:
        v_vals:   raw value head outputs in [-1, 1], head-perspective
        outcomes: ground-truth outcomes ∈ {+1.0, -1.0}, head-perspective
        n_bins:   number of equal-width bins over [0, 1]
    Returns:
        ECE scalar (lower is better calibrated)
    """
    if not v_vals:
        return float("nan")
    p_wins = [(v + 1.0) / 2.0 for v in v_vals]
    y_wins = [(o + 1.0) / 2.0 for o in outcomes]  # 1 for win, 0 for loss

    n = len(p_wins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [j for j in range(n) if lo <= p_wins[j] < hi]
        if i == n_bins - 1:
            # include right edge in last bin
            in_bin = [j for j in range(n) if lo <= p_wins[j] <= hi]
        if not in_bin:
            continue
        bin_n = len(in_bin)
        avg_conf = np.mean([p_wins[j] for j in in_bin])
        avg_acc = np.mean([y_wins[j] for j in in_bin])
        ece += (bin_n / n) * abs(avg_conf - avg_acc)
    return float(ece)


# ── decided accuracy ──────────────────────────────────────────────────────────


def compute_decided_accuracy(
    v_vals: List[float],
    outcomes: List[float],
    margin: float = DECIDED_MARGIN,
) -> Tuple[float, int]:
    """Fraction where sign(v_t) matches sign(outcome), restricted to |v_t| > margin.

    Returns:
        (accuracy, n_decided)
    """
    decided = [(v, o) for v, o in zip(v_vals, outcomes) if abs(v) > margin]
    if not decided:
        return float("nan"), 0
    correct = sum(1 for v, o in decided if (v > 0) == (o > 0))
    return correct / len(decided), len(decided)


# ── value MAE ────────────────────────────────────────────────────────────────


def compute_mae(v_vals: List[float], outcomes: List[float]) -> float:
    if not v_vals:
        return float("nan")
    return float(np.mean([abs(v - o) for v, o in zip(v_vals, outcomes)]))


# ── process one checkpoint ─────────────────────────────────────────────────────


def process_ckpt(
    step: int,
    ckpt_path: Path,
    games_dir: Optional[Path],
    device,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load ckpt + replay games.jsonl, compute value-head health metrics.

    Returns metric dict or None on skip.
    """
    import torch

    from hexo_rl.encoding import normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model
    from hexo_rl.eval.eval_board import make_eval_board

    log = lambda s: print(f"  [step={step}] {s}") if verbose else None

    # ── games dir check ──────────────────────────────────────────────────────
    if games_dir is None:
        log("SKIP: no games dir configured")
        return None

    games_path = games_dir / "games.jsonl"
    if not games_path.exists():
        log(f"SKIP: games.jsonl not found at {games_path}")
        return None

    if not ckpt_path.exists():
        log(f"SKIP: ckpt not found at {ckpt_path}")
        return None

    # ── load model (gated, v6_live2_ls) ──────────────────────────────────────
    try:
        model, spec, label = load_model_with_encoding(
            str(ckpt_path), device, declared_encoding=EXPECT_ENCODING
        )
    except Exception as e:
        log(f"SKIP: loader failed — {e}")
        return None

    enc_name = normalize_encoding_name(EXPECT_ENCODING)
    eng = _build_engine_for_model(model, enc_name, device)
    sha = ckpt_sha(str(ckpt_path))
    log(f"loaded {label} sha={sha}")

    # ── load games ────────────────────────────────────────────────────────────
    games = load_games_jsonl(str(games_path))
    if not games:
        log("SKIP: empty games.jsonl")
        return None

    radius = games[0].get("radius")
    book_id = games[0].get("book_id", "unknown")

    # Filter: only decided games (no draws, no censored)
    decided_games = [
        g for g in games
        if g.get("winner") not in ("draw", None)
        and not g.get("censored", False)
    ]
    n_win_games = sum(
        1 for g in decided_games
        if g["winner"] == ("p1" if g["head_as_p1"] else "p2")
    )
    n_loss_games = len(decided_games) - n_win_games
    log(f"games: {len(games)} total, {len(decided_games)} decided (w={n_win_games} l={n_loss_games})")

    # ── replay + collect value head outputs ───────────────────────────────────
    all_v: List[float] = []        # head-perspective v_t
    all_outcomes: List[float] = [] # head-perspective outcome (+1 win / -1 loss)
    all_phases: List[str] = []     # "early" / "late" (split at game midpoint)

    for g in decided_games:
        head_pn = 1 if g["head_as_p1"] else -1
        outcome = +1.0 if g["winner"] == ("p1" if g["head_as_p1"] else "p2") else -1.0

        # Replay to collect board snapshots
        board = make_eval_board(enc_name, radius)
        snaps = []
        for q, r in g["moves"]:
            # snapshot before applying move
            snaps.append({
                "cp": int(board.current_player),
                "mr": int(board.moves_remaining),
                "ply": int(board.ply),
                "board": board.clone(),
            })
            board.apply_move(int(q), int(r))

        # Filter to head-turn-starts
        head_snaps = [
            s for s in snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        if not head_snaps:
            continue

        # Batch inference
        boards = [s["board"] for s in head_snaps]
        _, vals = eng.infer_batch(boards)

        game_len = len(snaps)
        midpoint_ply = game_len // 2

        for si, snap in enumerate(head_snaps):
            v = float(vals[si])
            all_v.append(v)
            all_outcomes.append(outcome)
            phase = "early" if snap["ply"] < midpoint_ply else "late"
            all_phases.append(phase)

    n_positions = len(all_v)
    log(f"n_positions: {n_positions}")

    if n_positions == 0:
        log("SKIP: zero positions after filtering")
        return None

    # ── metrics ───────────────────────────────────────────────────────────────
    value_ece = compute_ece(all_v, all_outcomes)
    decided_acc, n_decided = compute_decided_accuracy(all_v, all_outcomes)
    value_mae = compute_mae(all_v, all_outcomes)

    # Phase split
    early_v = [v for v, ph in zip(all_v, all_phases) if ph == "early"]
    early_o = [o for o, ph in zip(all_outcomes, all_phases) if ph == "early"]
    late_v  = [v for v, ph in zip(all_v, all_phases) if ph == "late"]
    late_o  = [o for o, ph in zip(all_outcomes, all_phases) if ph == "late"]

    early_ece = compute_ece(early_v, early_o) if early_v else float("nan")
    late_ece  = compute_ece(late_v, late_o)   if late_v  else float("nan")
    early_acc, _ = compute_decided_accuracy(early_v, early_o)
    late_acc, _  = compute_decided_accuracy(late_v,  late_o)

    row = {
        "ckpt_step":        step,
        "ckpt_sha":         sha,
        "radius":           radius,
        "book_id":          book_id,
        "n_positions":      n_positions,
        "value_ece":        round(value_ece, 6),
        "decided_accuracy": round(decided_acc, 6) if decided_acc == decided_acc else None,
        "decided_margin":   DECIDED_MARGIN,
        "value_mae":        round(value_mae, 6),
        "n_win_games":      n_win_games,
        "n_loss_games":     n_loss_games,
        # phase split (not in primary schema but useful context)
        "early_ece":        round(early_ece, 6) if early_ece == early_ece else None,
        "late_ece":         round(late_ece, 6)  if late_ece  == late_ece  else None,
        "early_decided_acc": round(early_acc, 6) if early_acc == early_acc else None,
        "late_decided_acc":  round(late_acc, 6)  if late_acc  == late_acc  else None,
        "n_early_positions": len(early_v),
        "n_late_positions":  len(late_v),
    }

    log(f"ECE={value_ece:.4f}  acc={decided_acc:.4f}  MAE={value_mae:.4f}")
    return row


# ── plotting ──────────────────────────────────────────────────────────────────


def make_trend_plot(rows: List[Dict], out_path: Path) -> None:
    """ECE + decided_accuracy vs step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["ckpt_step"] for r in rows]
    ece   = [r["value_ece"] for r in rows]
    acc   = [r["decided_accuracy"] for r in rows if r["decided_accuracy"] is not None]
    acc_steps = [r["ckpt_step"] for r in rows if r["decided_accuracy"] is not None]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(steps, ece, "o-", color="tab:blue", markersize=4, linewidth=1.5, label="ECE")
    ax1.axvline(200000, color="gray", linestyle="--", alpha=0.5, label="r4→r5 boundary")
    ax1.set_ylabel("ECE (↓ better)")
    ax1.set_title("Value-Head Health Trend — run2 banked checkpoints")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(acc_steps, acc, "o-", color="tab:green", markersize=4, linewidth=1.5,
             label=f"Decided acc (|v|>{DECIDED_MARGIN})")
    ax2.axvline(200000, color="gray", linestyle="--", alpha=0.5, label="r4→r5 boundary")
    ax2.set_ylabel("Decided accuracy (↑ better)")
    ax2.set_xlabel("Checkpoint step")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"Plot → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WP4 value-head health trend")
    parser.add_argument(
        "--out", default="reports/valprobe/value_health_series.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument("--pilot", type=int, default=0, metavar="N",
                        help="Only process first N checkpoints (smoke-test)")
    args = parser.parse_args()

    import torch
    dev = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"device: {dev}")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inventory = CKPT_INVENTORY
    if args.pilot:
        inventory = inventory[:args.pilot]

    rows: List[Dict] = []
    skipped: List[Tuple[int, str]] = []

    for step, ckpt_rel, games_rel in inventory:
        print(f"\n=== step {step} ===")
        ckpt_path = REPO / ckpt_rel
        games_dir = REPO / games_rel if games_rel else None

        row = process_ckpt(step, ckpt_path, games_dir, dev)
        if row is None:
            skipped.append((step, "see log above"))
        else:
            rows.append(row)

    # Write JSONL sorted by step
    rows.sort(key=lambda r: r["ckpt_step"])
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(rows)} rows → {out_path}")

    if skipped:
        print(f"Skipped {len(skipped)} ckpts: {[s for s, _ in skipped]}")

    if not args.no_plot and rows:
        plot_path = out_path.parent / "value_health_trend.png"
        make_trend_plot(rows, plot_path)

    # Print summary table
    print("\n=== Series table ===")
    print(f"{'step':>8}  {'ECE':>8}  {'dec_acc':>8}  {'MAE':>8}  {'n_pos':>7}  {'r':>3}")
    for r in rows:
        acc_str = f"{r['decided_accuracy']:.4f}" if r["decided_accuracy"] is not None else "  n/a "
        print(f"{r['ckpt_step']:>8}  {r['value_ece']:>8.4f}  {acc_str:>8}  "
              f"{r['value_mae']:>8.4f}  {r['n_positions']:>7}  {r['radius']:>3}")


if __name__ == "__main__":
    main()
