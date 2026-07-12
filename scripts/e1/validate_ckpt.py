"""E1 T7 — per-checkpoint value-health validation CLI.

E1 reads are ALL external (in-loop eval is off). This is the ONE CLI that scores
any E1 checkpoint on the frozen 234-probe (loss set) + 651 win-controls (safe set)
and emits the frozen metrics M1-M4 per docs/designs/e1_metric_freeze.md (FROZEN,
authoritative). It is the value-health stage of WP3's external eval script and
backs the WP4 red-team forward-smoke.

Scoring path (RAW multi-window forward, NO search)
--------------------------------------------------
The 234-probe is a FIXED position set. Its `v_raw` was selected by the
MULTI-WINDOW MIN-POOL forward (LocalInferenceEngine.infer_batch:112
`v = float(board_values.min())` over the K legal-set cluster windows). We
replicate that exact deploy decode with the checkpoint's OWN baked value head
(loaded via the gated eval loader), NOT the HEADSWAP swappable-head:

  - reconstruct each board from its source game (opening_idx, head_as_p1, t),
    verifying the row zobrist (score_probe.reconstruct_board);
  - build the K-cluster (K, C, H, W) tensor (GameState.to_tensor + kept-plane
    slice, identical to inference.py's hot-path slice);
  - forward the FULL net once -> per-cluster (log_policy, value, v_logit):
      * value   = the DEPLOY-DECODED scalar (tanh for scalar; E[softmax·support]
                  for dist65 — network.py routes dist decode through
                  decode_binned_value);
      * v_logit = raw pre-tanh logit (scalar) / (K,65) bin logits (dist65);
  - argmin the decoded scalar over K clusters (== infer_batch's min-pool) -> the
    scored decoded-v; for the dist arm, take THAT cluster's tail-mass P(v<=-0.5)
    (loss_tail_mass, bins 0..16 inclusive).

No search, no sims: a fixed-position value read scores the raw forward the probe
was built with. Deterministic given (ckpt, probe files) — eval-mode, no sampling,
no wall-clock in the row.

Metrics (frozen — e1_metric_freeze.md §4)
------------------------------------------
  M1  mean_v_on_losses  = mean decoded-v over the LOSS positions.
  M2  ece               = ECE over the FULL set (loss ∪ safe), 10-bin,
                          P_win=(v+1)/2 (value_health.compute_ece).
  M3  tail_mass_auc     = dist arm: AUC(tail-mass, loss vs safe).
      decoded_auc       = scalar arm: AUC(decoded-v, loss vs safe).
                          (both emitted; one null per arm.)
  M4  false_pessimism   = fraction of SAFE controls with decoded-v <= -0.5.
  diagnostic (non-gating): recognition_lag_mean_v_on_losses (== M1 here; the
      recognition-lag delta is a longitudinal quantity across checkpoints, not a
      per-checkpoint scalar — this CLI emits the per-checkpoint mean-v-on-losses
      which the WP3 series consumer diffs across steps).

REGISTER GUARD (INV-D1): scores only. SealBot/solver appear only as the frozen
probe LABEL (set=loss/safe); never a target, never a gradient here.

CLI::

    .venv/bin/python scripts/e1/validate_ckpt.py \\
      --ckpt <path> --arm <scalar|dist65> \\
      --out reports/e1/value_health_series.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo-root on sys.path for bare `python scripts/e1/validate_ckpt.py` invocation
# (pytest resolves this via rootdir; a direct script run does not).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.eval_board import make_eval_board
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.binned_value import decode_binned_value
from scripts.headswap.targets import LOSS_TAIL_BIN
from scripts.headswap.metrics import auc, false_pessimism
from scripts.valprobe.value_health import compute_ece

# Frozen default probe/negatives + source-game paths (e1_metric_freeze.md §1).
REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROBE = str(REPO / "reports/valprobe/probe_set_v1.jsonl")
DEFAULT_NEGATIVES = str(REPO / "reports/valprobe/negatives_v1.jsonl")
DEFAULT_GAMES = str(
    REPO / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"
)
# WP2 loss positions (193/234) reconstruct from REGENERATED per-book games, not
# the retro_slope 248k games (e1_metric_freeze.md §1; score_all.resolve_game).
# Default location = the HEADSWAP wp2_regen layout book_id/games.jsonl. Absent on
# a fresh box (operator-run regen) -> those rows are unresolvable; the caller must
# supply --wp2-games or accept the missing-board skip.
DEFAULT_WP2_GAMES_DIR = str(REPO / "reports/headswap/wp2_regen")
_WP2_BOOK_IDS = (
    "evalfair_r5_wp2_b0",
    "evalfair_r5_wp2_b1",
    "evalfair_r5_wp2_b2",
    "evalfair_r5_wp2_b3",
    "evalfair_r5_wp2_b4",
)

FALSE_PESS_THRESHOLD = -0.5  # metrics.py:36 (also loss-tail threshold)

# STABLE row schema (WP3 + the run3 watcher consume it). Order fixed.
ROW_KEYS: Tuple[str, ...] = (
    "step",
    "arm",
    "ckpt_sha",
    "encoding",
    "mean_v_on_losses",
    "ece",
    "tail_mass_auc",
    "decoded_auc",
    "false_pessimism",
    "recognition_lag_mean_v_on_losses",
    "n_loss",
    "n_safe",
)


# ── checkpoint sha ────────────────────────────────────────────────────────────


def ckpt_sha(path: str) -> str:
    """Short content sha (first 16 hex), matching value_health.ckpt_sha."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _ckpt_step(ckpt_path: str) -> Optional[int]:
    """Best-effort step read from the checkpoint payload (never loads the net)."""
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    for key in ("step", "global_step", "training_step"):
        v = raw.get(key)
        if isinstance(v, int):
            return v
    meta = raw.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("step"), int):
        return int(meta["step"])
    return None


# ── source-game indexing + board reconstruction (reuse HEADSWAP) ─────────────


def index_games(games_path: str) -> Dict[Tuple[int, bool], dict]:
    """Map (opening_idx, head_as_p1) -> game record (retro_slope games.jsonl)."""
    from scripts.headswap.score_all import index_games as _idx

    return _idx(games_path)


def _default_wp2_games() -> Dict[str, str]:
    """Discover per-book WP2 regen games under DEFAULT_WP2_GAMES_DIR.

    Returns {book_id: path} only for books whose games.jsonl EXISTS (absent on a
    fresh box — those rows stay unresolvable and are reported as skipped)."""
    out: Dict[str, str] = {}
    root = Path(DEFAULT_WP2_GAMES_DIR)
    for book in _WP2_BOOK_IDS:
        p = root / book / "games.jsonl"
        if p.exists():
            out[book] = str(p)
    return out


def _build_game_indices(
    games_path: str, wp2_games: Dict[str, str]
) -> Tuple[Dict[Tuple[int, bool], dict], Dict[str, Dict[Tuple[int, bool], dict]]]:
    """(retro_index, wp2_indices) — mirrors score_all.run's index build."""
    from scripts.headswap.score_all import index_games as _idx

    retro_index = _idx(games_path)
    wp2_indices = {bid: _idx(p) for bid, p in wp2_games.items()}
    return retro_index, wp2_indices


def _load_rows(path: str) -> List[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


# ── gated model load (arm-consistency enforced) ──────────────────────────────


def _load_net(
    ckpt_path: str, arm: str, device: torch.device
) -> Tuple[HexTacToeNet, "object", str]:
    """Load the checkpoint via the gated eval loader (encoding asserted
    v6_live2_ls; dist65 value head auto-detected from value_fc2_bins) and
    enforce that the checkpoint's value-head type MATCHES the declared arm.

    A dist65 checkpoint scored as arm='scalar' (or vice versa) is a hard
    ValueError — never a silent mis-decode.
    """
    if arm not in ("scalar", "dist65"):
        raise ValueError(f"unknown arm {arm!r}; expected 'scalar' or 'dist65'")
    model, spec, label = load_model_with_encoding(
        ckpt_path, device, declared_encoding="v6_live2_ls"
    )
    ckpt_vh = model.value_head_type
    if ckpt_vh != arm:
        raise ValueError(
            f"arm/checkpoint value_head_type mismatch: declared arm={arm!r} but "
            f"checkpoint {ckpt_path} has value_head_type={ckpt_vh!r}. Scoring a "
            "dist65 checkpoint as scalar (or vice versa) would silently "
            "mis-decode the value head — refusing."
        )
    model.eval()
    return model, spec, label


# ── per-position multi-window min-pool forward (RAW, deploy decode) ──────────


def _board_to_cluster_tensor(
    model: HexTacToeNet, spec, board, device: torch.device
) -> torch.Tensor:
    """Board -> (K, in_channels, H, W) float tensor on device.

    Front-half of the multi-window inference path (identical to
    inference.py:81-88 and scripts.headswap.model_heads.board_to_cluster_tensor):
    GameState.to_tensor yields the 18-plane wire tensor for K legal-set cluster
    windows; slice to this encoding's kept planes.
    """
    state = GameState.from_board(board)
    tensor, _centers = state.to_tensor()  # (K, 18, H, W)
    if tensor.shape[1] != model.in_channels:
        tensor = tensor[:, list(spec.kept_plane_indices)]
    return torch.from_numpy(tensor).to(device).float()


@torch.inference_mode()
def _score_board(
    model: HexTacToeNet, spec, arm: str, board, device: torch.device
) -> Dict[str, float]:
    """Deploy-decoded, min-pooled score for ONE board via the FULL net's own
    value head.

    Returns {"v": decoded scalar (min-pool over K clusters, == infer_batch),
             "tail_mass": dist arm P(v<=-0.5) of the min-pool cluster, else None}.
    Matches inference.py's argmin-over-decoded-scalar min-pool exactly.
    """
    x = _board_to_cluster_tensor(model, spec, board, device)  # (K, C, H, W)
    # AMP off: this is a deterministic eval read (fp32 end-to-end, no autocast).
    log_policy, value, v_logit = model(x)  # value: (K,1) decoded; v_logit per arm
    v_k = value.squeeze(-1).float()  # (K,) decoded scalar in [-1,1]
    k_star = int(torch.argmin(v_k).item())  # min-pool cluster
    if arm == "dist65":
        # v_logit is (K, 65) bin logits; tail-mass of the min-pool cluster.
        probs = torch.softmax(v_logit[k_star].float(), dim=-1)
        tail = float(probs[: LOSS_TAIL_BIN + 1].sum().item())
        return {"v": float(v_k[k_star].item()), "tail_mass": tail}
    return {"v": float(v_k[k_star].item()), "tail_mass": None}


def _score_positions(
    model: HexTacToeNet,
    spec,
    device: torch.device,
    rows: List[dict],
    retro_index: Dict[Tuple[int, bool], dict],
    wp2_indices: Optional[Dict[str, Dict[Tuple[int, bool], dict]]] = None,
) -> Tuple[List[Dict[str, float]], int]:
    """Score every resolvable row -> ([{v, tail_mass}], n_skipped).

    Routes WP2 rows (book_id / source='wp2_regen_b<N>') to their regen games and
    WP1/NEG rows to the retro_slope index (score_all.resolve_game). Rows whose
    source game is UNAVAILABLE (WP2 regen not present) are SKIPPED and counted —
    NOT a hard error. Zobrist mismatch on a RESOLVED board IS a hard error (wrong
    reconstruction)."""
    from scripts.headswap.score_all import resolve_game
    from scripts.headswap.score_probe import reconstruct_board

    wp2_indices = wp2_indices or {}
    arm = model.value_head_type
    scored: List[Dict[str, float]] = []
    n_skipped = 0
    for r in rows:
        game = resolve_game(r, retro_index, wp2_indices)
        if game is None:
            n_skipped += 1
            continue
        board, zob = reconstruct_board(game, int(r["t"]))
        if zob != str(r["zobrist"]):
            raise RuntimeError(
                f"zobrist mismatch opening=({r['opening_idx']},{r['head_as_p1']}) "
                f"t={r['t']} wp={r.get('wp')}: recon {zob} != row {r['zobrist']}"
            )
        scored.append(_score_board(model, spec, arm, board, device))
    return scored, n_skipped


# ── metric assembly ───────────────────────────────────────────────────────────


def _round(x: Optional[float], nd: int = 8) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return round(float(x), nd)


def _compute_metrics(
    arm: str,
    loss_scores: List[Dict[str, float]],
    safe_scores: List[Dict[str, float]],
) -> Dict[str, Optional[float]]:
    """M1-M4 per e1_metric_freeze.md §4."""
    loss_v = [s["v"] for s in loss_scores]
    safe_v = [s["v"] for s in safe_scores]

    # M1 — mean decoded-v on the LOSS positions.
    m1 = float(np.mean(loss_v)) if loss_v else float("nan")

    # M2 — ECE over the FULL set = loss ∪ safe. Loss outcome y=-1, win y=+1.
    all_v = loss_v + safe_v
    all_outcomes = [-1.0] * len(loss_v) + [1.0] * len(safe_v)
    ece = compute_ece(all_v, all_outcomes)

    # M3 — discrimination AUC, loss(=positive label 1) vs safe.
    labels = [1] * len(loss_v) + [0] * len(safe_v)
    if arm == "dist65":
        tail = [s["tail_mass"] for s in loss_scores] + [
            s["tail_mass"] for s in safe_scores
        ]
        tail_mass_auc = auc(tail, labels)  # higher tail-mass -> more likely lost
        decoded_auc = None
    else:
        # scalar arm has no distribution; AUC of NEGATED decoded-v (lower v ->
        # more likely lost, so -v is the "likely-lost" score for label==1).
        neg_v = [-v for v in all_v]
        decoded_auc = auc(neg_v, labels)
        tail_mass_auc = None

    # M4 — fraction of SAFE controls with decoded-v <= -0.5.
    fp = false_pessimism(safe_v, threshold=FALSE_PESS_THRESHOLD)

    return {
        "mean_v_on_losses": m1,
        "ece": ece,
        "tail_mass_auc": tail_mass_auc,
        "decoded_auc": decoded_auc,
        "false_pessimism": fp,
        # Diagnostic (non-gating): per-checkpoint mean-v-on-losses; the WP3
        # series consumer diffs this across steps for recognition lag.
        "recognition_lag_mean_v_on_losses": m1,
    }


# ── public entry point ─────────────────────────────────────────────────────────


def validate_ckpt(
    ckpt_path: str,
    arm: str,
    out_jsonl: str,
    step: Optional[int] = None,
    probe_path: str = DEFAULT_PROBE,
    negatives_path: str = DEFAULT_NEGATIVES,
    games_path: str = DEFAULT_GAMES,
    wp2_games: Optional[Dict[str, str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Score one E1 checkpoint on the frozen loss/safe sets, compute M1-M4,
    APPEND one JSON row to ``out_jsonl`` and return it (stable ROW_KEYS schema).

    Read-only: no training, no gradient, no checkpoint mutation. Deterministic
    given (ckpt, probe files) — no wall-clock in the row.

    ``wp2_games`` maps book_id -> regen games.jsonl for the 193/234 WP2 loss
    positions. ``None`` -> auto-discover under DEFAULT_WP2_GAMES_DIR; books whose
    regen games are absent leave their rows UNRESOLVED (skipped + counted in
    ``n_loss_skipped``), NOT a hard failure.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wp2_games is None:
        wp2_games = _default_wp2_games()

    model, spec, label = _load_net(ckpt_path, arm, device)
    sha = ckpt_sha(ckpt_path)
    if step is None:
        step = _ckpt_step(ckpt_path)

    loss_rows = _load_rows(probe_path)
    safe_rows = _load_rows(negatives_path)
    retro_index, wp2_indices = _build_game_indices(games_path, wp2_games)

    loss_scores, loss_skipped = _score_positions(
        model, spec, device, loss_rows, retro_index, wp2_indices
    )
    safe_scores, safe_skipped = _score_positions(
        model, spec, device, safe_rows, retro_index, wp2_indices
    )

    if loss_skipped or safe_skipped:
        import sys

        print(
            f"[validate_ckpt] WARN unresolved source games: "
            f"loss_skipped={loss_skipped}/{len(loss_rows)} "
            f"safe_skipped={safe_skipped}/{len(safe_rows)} "
            f"(WP2 regen games under {DEFAULT_WP2_GAMES_DIR} — supply --wp2-games "
            f"or accept the reduced set). n_loss={len(loss_scores)} "
            f"n_safe={len(safe_scores)}",
            file=sys.stderr,
        )
    if not loss_scores:
        raise RuntimeError(
            f"no loss positions resolved (all {len(loss_rows)} skipped: WP2 regen "
            f"games absent under {DEFAULT_WP2_GAMES_DIR}). M1/M2/M3 undefined — "
            "supply --wp2-games with the regenerated per-book games.jsonl."
        )
    if not safe_scores:
        raise RuntimeError(
            f"no safe controls resolved (all {len(safe_rows)} skipped). "
            "M2/M3/M4 undefined."
        )

    metrics = _compute_metrics(arm, loss_scores, safe_scores)

    row = {
        "step": step,
        "arm": arm,
        "ckpt_sha": sha,
        "encoding": label,
        "mean_v_on_losses": _round(metrics["mean_v_on_losses"]),
        "ece": _round(metrics["ece"]),
        "tail_mass_auc": _round(metrics["tail_mass_auc"]),
        "decoded_auc": _round(metrics["decoded_auc"]),
        "false_pessimism": _round(metrics["false_pessimism"]),
        "recognition_lag_mean_v_on_losses": _round(
            metrics["recognition_lag_mean_v_on_losses"]
        ),
        "n_loss": len(loss_scores),
        "n_safe": len(safe_scores),
    }
    # Enforce stable key set/order.
    row = {k: row[k] for k in ROW_KEYS}

    out_p = Path(out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="E1 T7 — per-checkpoint value-health CLI (frozen M1-M4)"
    )
    ap.add_argument("--ckpt", required=True, help="E1 checkpoint .pt")
    ap.add_argument("--arm", required=True, choices=["scalar", "dist65"])
    ap.add_argument(
        "--out",
        default=str(REPO / "reports/e1/value_health_series.jsonl"),
        help="append-target JSONL (WP3 series)",
    )
    ap.add_argument("--step", type=int, default=None, help="override step")
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--negatives", default=DEFAULT_NEGATIVES)
    ap.add_argument("--games", default=DEFAULT_GAMES)
    ap.add_argument(
        "--wp2-games", nargs="*", default=None,
        help="book_id=path pairs for regenerated WP2 games.jsonl (the 193/234 WP2 "
             "loss positions). Omit to auto-discover under DEFAULT_WP2_GAMES_DIR.",
    )
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    wp2_games: Optional[Dict[str, str]] = None
    if args.wp2_games is not None:
        wp2_games = {}
        for spec in args.wp2_games:
            if "=" not in spec:
                raise SystemExit(f"--wp2-games entry must be book_id=path, got {spec!r}")
            bid, p = spec.split("=", 1)
            wp2_games[bid] = p

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    row = validate_ckpt(
        args.ckpt,
        args.arm,
        args.out,
        step=args.step,
        probe_path=args.probe,
        negatives_path=args.negatives,
        games_path=args.games,
        wp2_games=wp2_games,
        device=device,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
