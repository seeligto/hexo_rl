"""Phase 4.0 mode-collapse diagnostic — policy sharpness across checkpoints.

This is Diagnostic B from `archive/diagnosis_2026-04-10/`. It loads several
checkpoints, forwards 500 board positions drawn from recent self-play game
records, and reports the distribution of raw-policy entropy (H(π_raw)) per
checkpoint. It also parses the diag-A training-path JSONL (if present) and
emits per-move H(π_prior) / H(π_visits) / Δ / top-1 visit fraction for
Diagnostic C.

**Scope of changes:** this script is strictly read-only w.r.t. trainer.py,
network.py, and the engine. It only loads checkpoints, runs forward passes
in eval() + no_grad(), and writes markdown / JSON / CSV into
`archive/diagnosis_2026-04-10/`. Nothing else is touched.

**K=0 caveat.** Entropy is measured on the K=0 (centroid) cluster window
only, not the full min-pool aggregation used by the training path.
Cross-checkpoint comparison on identical positions is unaffected by the
K choice because every checkpoint is evaluated on the same inputs. See the
diag_B_sharpness.md header for the full caveat text.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board
from hexo_rl.env.game_state import GameState, HISTORY_LEN
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys
from hexo_rl.training.trainer import Trainer

DIAG_DIR = REPO_ROOT / "archive" / "diagnosis_2026-04-10"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
RUNS_DIR = REPO_ROOT / "runs"

CHECKPOINT_NAMES = [
    "bootstrap_model.pt",
    "checkpoint_00013000.pt",
    "checkpoint_00014000.pt",
    "checkpoint_00015000.pt",
    "checkpoint_00016000.pt",
    "checkpoint_00017000.pt",
    "checkpoint_00017428.pt",
    "best_model.pt",
]

N_POSITIONS = 500
SEED = 2026_04_10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Position sampling from recorded games
# ---------------------------------------------------------------------------


@dataclass
class Position:
    tensor: np.ndarray  # (24, 19, 19) float16 — K=0 slice only
    compound_move: int
    game_id: str
    ply: int


def _latest_run_dir() -> Path:
    candidates = sorted(
        p for p in RUNS_DIR.iterdir()
        if p.is_dir() and (p / "games").exists() and any((p / "games").iterdir())
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories with games found under {RUNS_DIR}")
    # Sort by mtime so "latest" actually means most recent.
    candidates.sort(key=lambda p: (p / "games").stat().st_mtime)
    return candidates[-1]


def _parse_move(token: str) -> Tuple[int, int]:
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):
        token = token[1:-1]
    q_str, r_str = token.split(",")
    return int(q_str), int(r_str)


def _replay_to_state(moves: List[Tuple[int, int]], target_ply: int) -> Optional[GameState]:
    """Replay the first `target_ply` moves of `moves` through Rust Board + GameState."""
    if target_ply <= 0 or target_ply > len(moves):
        return None
    board = Board()
    history: deque = deque(maxlen=HISTORY_LEN)
    state = GameState.from_board(board, history=history)
    try:
        for i in range(target_ply):
            q, r = moves[i]
            state = state.apply_move(board, q, r)
    except Exception:
        return None
    return state


def sample_positions(n: int) -> List[Position]:
    run_dir = _latest_run_dir()
    game_dir = run_dir / "games"
    game_files = sorted(game_dir.glob("*.json"))
    if not game_files:
        raise FileNotFoundError(f"No *.json game records in {game_dir}")

    print(f"[diag_B] sampling from {len(game_files)} games in {run_dir.name}")

    rng = random.Random(SEED)
    rng.shuffle(game_files)

    positions: List[Position] = []
    for gf in game_files:
        if len(positions) >= n:
            break
        try:
            doc = json.loads(gf.read_text())
        except Exception:
            continue
        moves_raw = doc.get("moves_list") or doc.get("moves")
        if not moves_raw or len(moves_raw) < 2:
            continue
        try:
            moves = [_parse_move(tok) for tok in moves_raw]
        except Exception:
            continue

        # Pull one or two positions per game at stratified phases so we don't
        # over-represent long games. Stratification: early / mid / late.
        phases = []
        max_ply = len(moves)
        early_ply = max(1, min(max_ply - 1, rng.randint(1, min(max_ply - 1, 10))))
        phases.append(early_ply)
        if max_ply > 20:
            mid_ply = rng.randint(11, min(max_ply - 1, 48))
            phases.append(mid_ply)
        if max_ply > 50:
            late_ply = rng.randint(50, max_ply - 1)
            phases.append(late_ply)

        for ply in phases:
            if len(positions) >= n:
                break
            state = _replay_to_state(moves, ply)
            if state is None:
                continue
            tensor, _centers = state.to_tensor()
            if tensor.shape[0] == 0:
                continue
            # K=0 window — document in the report.
            k0 = tensor[0]
            compound_move = (ply + 1) // 2
            positions.append(Position(
                tensor=k0,
                compound_move=compound_move,
                game_id=gf.stem,
                ply=ply,
            ))

    if len(positions) < n:
        print(f"[diag_B] WARNING: only sampled {len(positions)} / {n} positions")
    return positions[:n]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _weight_fingerprint(state: Dict) -> str:
    """SHA-256 of the first conv weight tensor bytes — stable across save formats."""
    stem_key = next(
        (k for k in sorted(state) if "conv" in k and "weight" in k),
        next(iter(state)),  # fallback: any first key
    )
    t = state[stem_key]
    raw = t.numpy().tobytes() if hasattr(t, "numpy") else bytes(t)
    return hashlib.sha256(raw).hexdigest()[:12]


def _load_model(ckpt_path: Path) -> tuple[HexTacToeNet, str]:
    """Load checkpoint; return (model, weight_fingerprint)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = Trainer._extract_model_state(ckpt)
    state = normalize_model_state_dict_keys(state)
    fingerprint = _weight_fingerprint(state)
    hparams = Trainer._infer_model_hparams(state)
    model = HexTacToeNet(
        board_size=int(hparams.get("board_size", 19)),
        in_channels=int(hparams.get("in_channels", 24)),
        filters=int(hparams.get("filters", 128)),
        res_blocks=int(hparams.get("res_blocks", 12)),
    )
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    return model, fingerprint


# ---------------------------------------------------------------------------
# Entropy computation
# ---------------------------------------------------------------------------


def _entropy_stats(H: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(H.mean()),
        "median": float(np.median(H)),
        "p10": float(np.percentile(H, 10)),
        "p90": float(np.percentile(H, 90)),
        "min": float(H.min()),
        "max": float(H.max()),
    }


def _histogram(H: np.ndarray, bins: int = 10, lo: float = 0.0, hi: float = 4.0) -> List[int]:
    counts, _edges = np.histogram(H, bins=bins, range=(lo, hi))
    return counts.tolist()


def _phase_split(compound_moves: np.ndarray, H: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    masks = {
        "early (cm<10)": compound_moves < 10,
        "mid (10<=cm<25)": (compound_moves >= 10) & (compound_moves < 25),
        "late (cm>=25)": compound_moves >= 25,
    }
    for label, mask in masks.items():
        if mask.sum() == 0:
            out[label] = {"count": 0}
            continue
        stats = _entropy_stats(H[mask])
        stats["count"] = int(mask.sum())
        out[label] = stats
    return out


def evaluate_checkpoint(ckpt_path: Path, positions: List[Position]) -> Dict:
    print(f"[diag_B] evaluating {ckpt_path.name} on {len(positions)} positions...")
    model, fingerprint = _load_model(ckpt_path)

    tensors = np.stack([p.tensor for p in positions]).astype(np.float16)
    x = torch.from_numpy(tensors).to(DEVICE)

    batch_size = 64
    log_policies: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            chunk = x[i:i + batch_size]
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):
                log_policy, _value, _value_logit = model(chunk)
            log_policies.append(log_policy.detach().float().cpu())

    log_policy = torch.cat(log_policies, dim=0)  # (N, 362)
    probs = torch.exp(log_policy)
    # Same as trainer.py:402-405 — sum over action dim.
    entropy = torch.special.entr(probs).sum(dim=-1).numpy()  # (N,)
    top1 = probs.max(dim=-1).values.numpy()  # (N,)
    effective_support = np.exp(entropy)

    compound_moves = np.array([p.compound_move for p in positions], dtype=np.int32)

    # Free GPU memory before next checkpoint.
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": ckpt_path.name,
        "weight_fingerprint": fingerprint,
        "n_positions": int(len(positions)),
        "entropy_stats": _entropy_stats(entropy),
        "top1_stats": _entropy_stats(top1),
        "effective_support_mean": float(effective_support.mean()),
        "effective_support_median": float(np.median(effective_support)),
        "histogram": _histogram(entropy),
        "phase_split": _phase_split(compound_moves, entropy),
    }


# ---------------------------------------------------------------------------
# Diagnostic C — parse the diag A training-path JSONL
# ---------------------------------------------------------------------------


def _entropy(vec: List[float]) -> float:
    arr = np.asarray(vec, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    # Handle zeros cleanly.
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def parse_training_trace(trace_path: Path) -> Optional[List[Dict]]:
    if not trace_path.exists():
        return None
    records: List[Dict] = []
    with trace_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("site") != "game_runner":
                continue
            priors = rec["root_priors"]
            visits = rec["root_visit_counts"]
            visit_total = int(rec.get("total_root_visits", sum(visits)))
            H_prior = _entropy(priors)
            H_visits = _entropy([float(v) for v in visits])
            top1_visit = (
                max(visits) / visit_total if visit_total > 0 else 0.0
            )
            children_over_5pct = (
                sum(1 for v in visits if visit_total > 0 and v / visit_total > 0.05)
                if visit_total > 0 else 0
            )
            records.append({
                "game_index": rec["game_index"],
                "worker_id": rec["worker_id"],
                "compound_move": rec["compound_move"],
                "ply": rec["ply"],
                "legal_move_count": rec["legal_move_count"],
                "root_n_children": rec["root_n_children"],
                "total_root_visits": visit_total,
                "temperature": rec["temperature"],
                "is_fast_game": rec["is_fast_game"],
                "H_prior": H_prior,
                "H_visits": H_visits,
                "delta_entropy": H_prior - H_visits,
                "top1_visit_fraction": float(top1_visit),
                "children_over_5pct": int(children_over_5pct),
                "effective_support_visits": float(math.exp(H_visits)),
            })
    return records


# ---------------------------------------------------------------------------
# Temperature schedule audit
# ---------------------------------------------------------------------------


def _load_selfplay_temp_config() -> Tuple[float, int]:
    """Read temp_min + temperature_threshold_compound_moves from configs."""
    import yaml
    selfplay_yaml = yaml.safe_load((REPO_ROOT / "configs" / "selfplay.yaml").read_text())
    mcts = selfplay_yaml.get("mcts", {})
    temp_min = float(mcts.get("temperature_min", 0.05))
    threshold = int(mcts.get("temperature_threshold_compound_moves", 15))
    return temp_min, threshold


def code_temperature_schedule(temp_min: float, threshold: int) -> List[Tuple[int, float]]:
    """Mirror the Rust formula at engine/src/game_runner.rs:510-515."""
    out: List[Tuple[int, float]] = []
    for cm in [0, 5, 10, 14, 15, 16, 20, 30]:
        if cm >= threshold:
            tau = temp_min
        else:
            tau = max(temp_min, math.cos(math.pi / 2 * cm / threshold))
        out.append((cm, float(tau)))
    return out


def sprint36_temperature_schedule(temp_min: float, anneal_moves: int = 60) -> List[Tuple[int, float]]:
    """Sprint log §36's formula for comparison."""
    out: List[Tuple[int, float]] = []
    for ply in [0, 5, 10, 14, 15, 16, 20, 30, 60, 120]:
        if ply >= anneal_moves:
            tau = temp_min
        else:
            tau = temp_min + 0.5 * (1.0 - temp_min) * (1.0 + math.cos(math.pi * ply / anneal_moves))
        out.append((ply, float(tau)))
    return out


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------


def _fmt_stats(s: Dict[str, float]) -> str:
    return (
        f"mean={s['mean']:.3f} median={s['median']:.3f} "
        f"p10={s['p10']:.3f} p90={s['p90']:.3f} "
        f"min={s['min']:.3f} max={s['max']:.3f}"
    )


def write_diag_B(results: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Diagnostic B — policy sharpness across checkpoints\n")
    lines.append("## K=0 caveat (must be read first)\n")
    lines.append(
        "> Entropy values were measured on the K=0 (centroid) cluster window\n"
        "> only, not the full min-pool aggregation used by the training path.\n"
        "> Absolute comparison against the sprint log §1 heuristic (\"expected\n"
        "> 3-6 nats; < 1.0 signals collapse\") is **indicative, not strict** --\n"
        "> that heuristic was derived on the full min-pool path and may differ\n"
        "> by several tenths of a nat from K=0 numbers. The **primary signal is\n"
        "> the progression across checkpoints on the same positions**, which is\n"
        "> unaffected by the K choice because every checkpoint is evaluated on\n"
        "> identical inputs. Do not argue about whether \"ckpt_15000 at 0.2\n"
        "> nats\" is really collapsed or really at 0.8 nats under min-pool --\n"
        "> argue about whether the gap between `best_checkpoint.pt` and\n"
        "> ckpt_15000 is catastrophic.\n"
    )
    lines.append("")
    lines.append(f"Positions evaluated: {results[0]['n_positions']} (same 500 positions per checkpoint)\n")
    lines.append("Source: recent self-play games from the latest run directory.\n")
    lines.append("Device: " + str(DEVICE) + "\n")
    lines.append("")

    lines.append("## Per-checkpoint summary\n")
    lines.append("| Checkpoint | weight_fp | H(π) mean | median | p10 | p90 | top-1 mean | eff support mean |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        es = r["entropy_stats"]
        ts = r["top1_stats"]
        fp = r.get("weight_fingerprint", "?")
        lines.append(
            f"| {r['checkpoint']} | `{fp}` | {es['mean']:.3f} | {es['median']:.3f} "
            f"| {es['p10']:.3f} | {es['p90']:.3f} "
            f"| {ts['mean']:.3f} | {r['effective_support_mean']:.2f} |"
        )
    lines.append("")

    # Warn if any two checkpoints share the same weight fingerprint.
    fp_to_names: Dict[str, List[str]] = {}
    for r in results:
        fp = r.get("weight_fingerprint", "?")
        fp_to_names.setdefault(fp, []).append(r["checkpoint"])
    dupes = {fp: names for fp, names in fp_to_names.items() if len(names) > 1}
    if dupes:
        lines.append("### WARNING — duplicate weight fingerprints detected\n")
        for fp, names in dupes.items():
            lines.append(
                f"- `{fp}`: {', '.join(f'`{n}`' for n in names)} share identical "
                f"model weights despite being different files on disk. "
                f"The most likely cause is that one file was initialised as a copy "
                f"of the other and the training gating never promoted a challenger.\n"
            )
        lines.append("")

    lines.append("## Phase split — H(π) per checkpoint per phase bucket\n")
    lines.append("| Checkpoint | Phase | n | mean | median | p10 | p90 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        for phase, stats in r["phase_split"].items():
            if stats.get("count", 0) == 0:
                lines.append(f"| {r['checkpoint']} | {phase} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {r['checkpoint']} | {phase} | {stats['count']} "
                f"| {stats['mean']:.3f} | {stats['median']:.3f} "
                f"| {stats['p10']:.3f} | {stats['p90']:.3f} |"
            )
    lines.append("")

    lines.append("## Histograms (10 bins over 0.0 - 4.0 nats)\n")
    lines.append("```")
    bin_header = "bin edge:  " + " ".join(f"{v:4.1f}" for v in np.linspace(0, 4, 11)[:-1])
    lines.append(bin_header)
    for r in results:
        hist = r["histogram"]
        hist_str = " ".join(f"{c:4d}" for c in hist)
        lines.append(f"{r['checkpoint']:30s}  {hist_str}")
    lines.append("```")
    lines.append("")

    lines.append("## Restart candidate heuristic\n")
    # Gather post-bootstrap entropy values to detect flat-band vs. trend.
    post_bootstrap = [
        r for r in results
        if r["checkpoint"] not in ("bootstrap_model.pt", "best_model.pt")
    ]
    if post_bootstrap:
        h_vals = [r["entropy_stats"]["mean"] for r in post_bootstrap]
        h_min, h_max = min(h_vals), max(h_vals)
        band_width = h_max - h_min
        is_flat = band_width < 0.3  # oscillation < 0.3 nats → no trend
    else:
        is_flat = False

    if is_flat and post_bootstrap:
        lines.append(
            f"**Entropy is FLAT across all post-bootstrap checkpoints** "
            f"(band {h_min:.2f}–{h_max:.2f} nats, width {band_width:.2f} nats "
            f"< 0.3 nat threshold). This is a **stuck fixed point**, not a "
            f"progressive collapse. No checkpoint in the post-bootstrap range "
            f"is meaningfully less collapsed than any other — do NOT use "
            f"entropy rank to choose the restart point.\n"
        )
        lines.append(
            "**Restart point selection:** choose the earliest checkpoint "
            "before self-play dominated the replay buffer — approximately "
            "step 10k, where the pretrain share was still ≥70%. Entropy "
            "ordering across the 13k–17k range is noise, not signal.\n"
        )
        lines.append(
            "**Restart from `bootstrap_model.pt`** (clean pretrained weights "
            "with no self-play contamination) once the Dirichlet port is "
            "complete. Do not use `best_model.pt` as a restart candidate "
            "unless you have confirmed it has different weights from "
            "`bootstrap_model.pt` (check the weight_fingerprint column above).\n"
        )
    else:
        candidate = None
        for r in post_bootstrap:
            if r["entropy_stats"]["mean"] >= 1.5:
                candidate = r["checkpoint"]
        if candidate is None:
            lines.append(
                "All post-bootstrap checkpoints have mean H(π) < 1.5 nats on "
                "the K=0 window. Recommend restart from `bootstrap_model.pt` "
                "(clean pretrained weights). Do NOT use `best_model.pt` unless "
                "its weight_fingerprint differs from `bootstrap_model.pt`."
            )
        else:
            lines.append(
                f"`{candidate}` is the latest checkpoint with mean raw-policy "
                f"entropy ≥ 1.5 nats. Restart candidates for the Phase 4.0 fix "
                f"session are this checkpoint or earlier — but see flat-band "
                f"caveat: if band_width < 0.3, entropy rank is noise."
            )
    lines.append("")

    (DIAG_DIR / "diag_B_sharpness.md").write_text("\n".join(lines))
    (DIAG_DIR / "diag_B_sharpness.json").write_text(json.dumps(results, indent=2))
    print(f"[diag_B] wrote {DIAG_DIR/'diag_B_sharpness.md'}")


def write_diag_C(records: Optional[List[Dict]], temp_min: float, threshold: int) -> None:
    # -- C.1: temperature schedule comparison --
    code_sched = code_temperature_schedule(temp_min, threshold)
    s36_sched = sprint36_temperature_schedule(temp_min, anneal_moves=60)

    lines: List[str] = []
    lines.append("# Diagnostic C.1 — temperature schedule audit\n")
    lines.append(
        "The Rust training path uses a compound-move cosine schedule with a\n"
        "hard floor at `temperature_threshold_compound_moves`. Sprint log\n"
        "§36 describes a different formula (per-ply cosine with\n"
        "`temp_anneal_moves=60`). **Both are reproduced below and differ.**\n"
        "This is a live docs-vs-code drift. It is not the cause of the mode\n"
        "collapse on its own (no root noise is), but it belongs in §70 as a\n"
        "separate bullet so it is greppable later.\n"
    )
    lines.append("")
    lines.append(f"Config values: `temperature_min = {temp_min}`, "
                 f"`temperature_threshold_compound_moves = {threshold}`\n")
    lines.append("")
    lines.append("## Rust code formula (`engine/src/game_runner.rs:510-515`)\n")
    lines.append("```\n"
                 "tau(cm) = temp_min                                 if cm >= threshold\n"
                 "        = max(temp_min, cos(pi/2 * cm / threshold)) otherwise\n"
                 "```\n")
    lines.append("| compound_move | tau |")
    lines.append("|---|---|")
    for cm, tau in code_sched:
        lines.append(f"| {cm} | {tau:.4f} |")
    lines.append("")
    lines.append("## Sprint log §36 formula (not implemented)\n")
    lines.append("```\n"
                 "tau(ply) = temp_min + 0.5 * (1 - temp_min) * (1 + cos(pi * ply / anneal_moves))\n"
                 "           with anneal_moves = 60, per-ply (not per compound_move)\n"
                 "```\n")
    lines.append("| ply | tau (sprint log §36) |")
    lines.append("|---|---|")
    for ply, tau in s36_sched:
        lines.append(f"| {ply} | {tau:.4f} |")
    lines.append("")
    lines.append("## Conclusion\n")
    lines.append(
        "- The code drops to the `temp_min = 0.05` floor at compound_move 15\n"
        "  (ply 29 or 30 depending on player-1's solo opener). Between cm=0\n"
        "  and cm=15 the schedule is a quarter-cosine falling from 1.0 to 0.0,\n"
        "  clamped at temp_min. After cm=15 there is zero further annealing.\n"
        "- Sprint log §36 would keep tau above the floor until ply 60, with a\n"
        "  symmetric half-cosine shape. Under that schedule a game has roughly\n"
        "  four times as many moves with meaningfully stochastic sampling.\n"
        "- Since the root policy on the collapsed ckpt_15000 is already 54%\n"
        "  concentrated on a single move at cm=0 (see diag_A_trace_summary.md),\n"
        "  even a temperature of 1.0 cannot produce meaningful variation --\n"
        "  pi[top] / sum(pi^(1/1)) is still the top draw with >50% probability.\n"
        "  The docs-vs-code drift therefore does not explain the collapse, but\n"
        "  it does mean the implemented schedule gives the network even less\n"
        "  room to escape a sharpened prior than the documented one did.\n"
    )
    (DIAG_DIR / "diag_C_temp_schedule.md").write_text("\n".join(lines))
    print(f"[diag_C] wrote {DIAG_DIR/'diag_C_temp_schedule.md'}")

    # -- C.2: per-move CSV + summary --
    if records is None:
        print("[diag_C] training trace JSONL not found; skipping per-move analysis")
        return

    import csv
    csv_path = DIAG_DIR / "diag_C_per_move.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "game_index", "worker_id", "compound_move", "ply",
            "legal_move_count", "root_n_children", "total_root_visits",
            "temperature", "is_fast_game",
            "H_prior", "H_visits", "delta_entropy",
            "top1_visit_fraction", "children_over_5pct",
            "effective_support_visits",
        ])
        for r in records:
            w.writerow([
                r["game_index"], r["worker_id"], r["compound_move"], r["ply"],
                r["legal_move_count"], r["root_n_children"], r["total_root_visits"],
                f"{r['temperature']:.4f}", r["is_fast_game"],
                f"{r['H_prior']:.4f}", f"{r['H_visits']:.4f}",
                f"{r['delta_entropy']:.4f}",
                f"{r['top1_visit_fraction']:.4f}", r["children_over_5pct"],
                f"{r['effective_support_visits']:.4f}",
            ])
    print(f"[diag_C] wrote {csv_path}")

    # Summary markdown.
    H_prior = np.array([r["H_prior"] for r in records])
    H_visits = np.array([r["H_visits"] for r in records])
    delta = np.array([r["delta_entropy"] for r in records])
    top1_visit = np.array([r["top1_visit_fraction"] for r in records])
    eff = np.array([r["effective_support_visits"] for r in records])

    lines = []
    lines.append("# Diagnostic C.2 — per-move entropy from the training trace\n")
    lines.append(
        f"Records parsed: {len(records)} (from "
        f"`diag_A_trace_training.jsonl`, cap 30 in game_runner).\n"
    )
    lines.append("## Summary statistics\n")
    lines.append("| Metric | mean | median | p10 | p90 | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, arr in [
        ("H(pi_prior)", H_prior),
        ("H(pi_visits)", H_visits),
        ("delta (prior - visits)", delta),
        ("top-1 visit fraction", top1_visit),
        ("effective support (exp H)", eff),
    ]:
        lines.append(
            f"| {label} | {arr.mean():.3f} | {np.median(arr):.3f} "
            f"| {np.percentile(arr, 10):.3f} | {np.percentile(arr, 90):.3f} "
            f"| {arr.min():.3f} | {arr.max():.3f} |"
        )
    lines.append("")

    mean_delta = delta.mean()
    if mean_delta > 0.2:
        verdict = (
            "H(pi_visits) is meaningfully **lower** than H(pi_prior) on average "
            "-- MCTS is sharpening the already-sharp prior. The collapsed "
            "self-play loop is self-referential: a sharp prior produces a "
            "sharp visit distribution, which trains the next policy toward "
            "an even sharper prior."
        )
    elif mean_delta < -0.2:
        verdict = (
            "H(pi_visits) is meaningfully **higher** than H(pi_prior) on "
            "average -- MCTS is diversifying beyond the prior. Unexpected on "
            "a collapsed run; investigate whether Gumbel or FPU is somehow "
            "active."
        )
    else:
        verdict = (
            "H(pi_visits) is approximately equal to H(pi_prior) on average -- "
            "MCTS is a no-op that rubber-stamps the prior. This is the worst "
            "signal because it means every simulation budget we spend on "
            "self-play produces training targets that are identical to the "
            "current network's output, so training converges to a fixed point."
        )
    lines.append("## Verdict\n")
    lines.append(verdict)
    lines.append("")

    lines.append("## First 5 records (illustrative)\n")
    lines.append("```")
    for r in records[:5]:
        lines.append(
            f"g={r['game_index']:2d} w={r['worker_id']:2d} cm={r['compound_move']:2d} "
            f"ply={r['ply']:2d} H_prior={r['H_prior']:.3f} "
            f"H_visits={r['H_visits']:.3f} delta={r['delta_entropy']:+.3f} "
            f"top1={r['top1_visit_fraction']:.3f} "
            f"eff_support={r['effective_support_visits']:.2f}"
        )
    lines.append("```")

    (DIAG_DIR / "diag_C_summary.md").write_text("\n".join(lines))
    print(f"[diag_C] wrote {DIAG_DIR/'diag_C_summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4.0 mode-collapse diagnostics")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="Run diag B on these checkpoint names only (e.g. best_model.pt). "
             "Writes an *appended* corrected row to the existing report.",
    )
    args = parser.parse_args()

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    names_to_run = args.only if args.only else CHECKPOINT_NAMES

    positions = sample_positions(N_POSITIONS)
    print(f"[diag_B] sampled {len(positions)} positions; "
          f"cm range = {min(p.compound_move for p in positions)}"
          f"..{max(p.compound_move for p in positions)}")

    results: List[Dict] = []
    for name in names_to_run:
        path = CHECKPOINT_DIR / name
        if not path.exists():
            print(f"[diag_B] skipping missing checkpoint: {name}")
            continue
        try:
            results.append(evaluate_checkpoint(path, positions))
        except Exception as exc:
            print(f"[diag_B] ERROR on {name}: {exc!r}")

    if results:
        if args.only:
            # Append corrected rows to existing report rather than overwriting.
            report_path = DIAG_DIR / "diag_B_sharpness.md"
            lines: List[str] = []
            lines.append("\n---\n")
            lines.append("## Corrected rows (re-run with --only)\n")
            lines.append("| Checkpoint | weight_fp | H(π) mean | median | p10 | p90 | top-1 mean | eff support mean |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in results:
                es = r["entropy_stats"]
                ts = r["top1_stats"]
                fp = r.get("weight_fingerprint", "?")
                lines.append(
                    f"| {r['checkpoint']} | `{fp}` | {es['mean']:.3f} | {es['median']:.3f} "
                    f"| {es['p10']:.3f} | {es['p90']:.3f} "
                    f"| {ts['mean']:.3f} | {r['effective_support_mean']:.2f} |"
                )
            lines.append("")
            # Check for dupes against bootstrap (first result fingerprint is known reference).
            for r in results:
                fp = r.get("weight_fingerprint", "?")
                print(f"[diag_B] {r['checkpoint']:30s}  weight_fp={fp}  H_mean={r['entropy_stats']['mean']:.3f}")
            with report_path.open("a") as f:
                f.write("\n".join(lines))
            print(f"[diag_B] appended corrected rows to {report_path}")
        else:
            write_diag_B(results)

    # Diagnostic C — only on full run, not --only partial re-runs.
    if not args.only:
        trace_path = DIAG_DIR / "diag_A_trace_training.jsonl"
        records = parse_training_trace(trace_path)
        temp_min, threshold = _load_selfplay_temp_config()
        write_diag_C(records, temp_min, threshold)


if __name__ == "__main__":
    main()
