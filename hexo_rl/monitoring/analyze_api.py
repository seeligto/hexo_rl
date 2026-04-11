"""Policy viewer API — read-only analysis endpoint for /analyze.

MONITORING INVARIANT: This module must NOT import from hexo_rl.training
or hexo_rl.selfplay.pool. Must NOT call events.emit_event. Must NOT
register itself as a renderer. It is a passive read-only endpoint.

Per-cluster inspection (per-K policy distributions, per-K value heads)
deferred to v2 — see HexCanvas.js for the rendering side.
"""
from __future__ import annotations

import math
import os
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
from flask import Blueprint, jsonify, request

from engine import Board, MCTSTree
from hexo_rl.env.game_state import GameState
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.viewer.model_loader import load_model

log = structlog.get_logger(__name__)

analyze_bp = Blueprint("analyze", __name__)

# ── Checkpoint LRU cache ─────────────────────────────────────────────────────

_MAX_CACHE = 3
_cache_lock = threading.Lock()
_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()  # path → entry

# Single worker thread for inference + MCTS (prevents blocking Flask/SocketIO)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyze")


def _get_model(checkpoint_path: str) -> Tuple[Any, torch.device, dict]:
    """Load model from cache or disk. Thread-safe, LRU eviction."""
    abs_path = str(Path(checkpoint_path).resolve())

    if not Path(abs_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    disk_mtime = os.path.getmtime(abs_path)

    with _cache_lock:
        if abs_path in _cache:
            entry = _cache[abs_path]
            if entry["mtime"] >= disk_mtime:
                _cache.move_to_end(abs_path)
                return entry["model"], entry["device"], entry["metadata"]
            # Stale — evict and reload
            del _cache[abs_path]

    # Load outside lock (can be slow)
    model, metadata, device = load_model(abs_path)
    engine = LocalInferenceEngine(model, device)

    with _cache_lock:
        _cache[abs_path] = {
            "model": model,
            "engine": engine,
            "device": device,
            "metadata": metadata,
            "mtime": disk_mtime,
        }
        _cache.move_to_end(abs_path)
        while len(_cache) > _MAX_CACHE:
            _cache.popitem(last=False)

    return model, device, metadata


def _get_engine(checkpoint_path: str) -> LocalInferenceEngine:
    """Get cached inference engine for a checkpoint."""
    abs_path = str(Path(checkpoint_path).resolve())
    _get_model(checkpoint_path)  # ensure loaded
    with _cache_lock:
        return _cache[abs_path]["engine"]


# ── Board reconstruction ─────────────────────────────────────────────────────

def _build_board(moves: List[Dict[str, Any]]) -> Board:
    """Build Board from move list [{q, r, player}, ...]."""
    board = Board()
    for m in moves:
        q, r = int(m["q"]), int(m["r"])
        board.apply_move(q, r)
    return board


# ── Raw policy analysis ──────────────────────────────────────────────────────

def _analyze_raw(
    engine: LocalInferenceEngine,
    board: Board,
) -> Dict[str, Any]:
    """Run raw NN forward pass, return policy/value/entropy."""
    state = GameState.from_board(board)
    tensor, centers = state.to_tensor()
    K = len(centers)
    BOARD_SIZE = 19
    N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
    half = (BOARD_SIZE - 1) // 2

    batch = torch.from_numpy(tensor).to(engine.device)
    engine.model.eval()
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=engine.device.type,
            enabled=(engine.device.type in ("cuda", "mps")),
        ):
            log_policy, value, _v_logit = engine.model(batch.float())

    policies_np = log_policy.exp().cpu().float().numpy()
    log_policies_np = log_policy.cpu().float().numpy()
    values_np = value.squeeze(-1).cpu().float().numpy()

    # Min-pool value across clusters
    v = float(values_np.min())

    # Build per-legal-move policy with logits
    legal_moves = board.legal_moves()
    policy_entries = []
    for q, r in legal_moves:
        flat_idx = board.to_flat(q, r)
        if flat_idx >= N_ACTIONS - 1:
            continue
        max_prob = 0.0
        best_logit = -float("inf")
        for k, (cq, cr) in enumerate(centers):
            wq = q - cq + half
            wr = r - cr + half
            if 0 <= wq < BOARD_SIZE and 0 <= wr < BOARD_SIZE:
                local_idx = wq * BOARD_SIZE + wr
                p = float(policies_np[k, local_idx])
                if p > max_prob:
                    max_prob = p
                    best_logit = float(log_policies_np[k, local_idx])
        policy_entries.append({"q": q, "r": r, "prob": max_prob, "logit": best_logit})

    # Normalize
    total = sum(e["prob"] for e in policy_entries)
    if total > 1e-9:
        for e in policy_entries:
            e["prob"] /= total
    else:
        uniform = 1.0 / max(len(policy_entries), 1)
        for e in policy_entries:
            e["prob"] = uniform

    # Sort descending by prob
    policy_entries.sort(key=lambda e: e["prob"], reverse=True)

    # Entropy
    probs = np.array([e["prob"] for e in policy_entries], dtype=np.float64)
    probs = probs[probs > 1e-10]
    entropy_nats = float(-np.sum(probs * np.log(probs)))
    uniform_entropy = float(np.log(max(len(policy_entries), 1)))
    entropy_frac = entropy_nats / uniform_entropy if uniform_entropy > 0 else 0.0

    return {
        "policy": policy_entries,
        "value": round(v, 6),
        "entropy_nats": round(entropy_nats, 4),
        "entropy_uniform_fraction": round(entropy_frac, 4),
        "top_k": policy_entries[:10],
        "legal_moves_count": len(legal_moves),
        "next_to_move": int(board.current_player),
        "moves_remaining": int(board.moves_remaining),
    }


# ── MCTS analysis ────────────────────────────────────────────────────────────

def _infer_and_expand(
    tree: MCTSTree,
    engine: LocalInferenceEngine,
    batch_size: int,
    n_sims: int,
) -> int:
    """Drive n_sims through the tree using Python inference. Returns sims done."""
    done = 0
    while done < n_sims:
        batch = min(batch_size, n_sims - done)
        leaves = tree.select_leaves(batch)
        if not leaves:
            break
        policies, values = engine.infer_batch(leaves)
        tree.expand_and_backup(policies, values)
        done += len(leaves)
    return done


def _run_puct(
    board: Board,
    engine: LocalInferenceEngine,
    n_sims: int,
    c_puct: float = 1.5,
) -> Dict[str, Any]:
    """Run PUCT MCTS, return visit distribution."""
    tree = MCTSTree(c_puct=c_puct)
    tree.new_game(board)
    _infer_and_expand(tree, engine, batch_size=1, n_sims=1)  # expand root
    _infer_and_expand(tree, engine, batch_size=8, n_sims=max(0, n_sims - 1))
    return _collect_mcts_results(tree)


def _run_gumbel(
    board: Board,
    engine: LocalInferenceEngine,
    n_sims: int,
    c_puct: float = 1.5,
    m: int = 16,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> Dict[str, Any]:
    """Run Gumbel MCTS with Sequential Halving, return visit distribution.

    Simplified SH for analyze-only. Production SH lives in
    engine/src/game_runner.rs (GumbelSearchState). Do not treat
    this as source of truth for training/selfplay.
    """
    tree = MCTSTree(c_puct=c_puct)
    tree.new_game(board)
    _infer_and_expand(tree, engine, batch_size=1, n_sims=1)  # expand root

    children = tree.get_root_children_info()
    n_children = len(children)
    if n_children == 0:
        return _collect_mcts_results(tree)

    effective_m = min(m, n_children)

    # Gumbel noise + log-prior scores → top-m candidates
    gumbels = [-math.log(-math.log(random.uniform(1e-10, 1 - 1e-7))) for _ in range(n_children)]
    log_priors = [math.log(max(c[2], 1e-8)) for c in children]
    scored = sorted(range(n_children), key=lambda i: gumbels[i] + log_priors[i], reverse=True)
    candidates = scored[:effective_m]

    num_phases = max(1, math.ceil(math.log2(effective_m))) if effective_m > 1 else 1
    sims_used = 0
    budget = max(0, n_sims - 1)

    for phase in range(num_phases):
        if sims_used >= budget or len(candidates) <= 1:
            break
        remaining = budget - sims_used
        remaining_phases = num_phases - phase
        sims_per = max(1, remaining // (remaining_phases * len(candidates)))

        for cand_idx in candidates:
            if sims_used >= budget:
                break
            pool_idx = children[cand_idx][1]
            tree.forced_root_child = int(pool_idx)
            done = _infer_and_expand(tree, engine, batch_size=8, n_sims=sims_per)
            sims_used += done

        tree.forced_root_child = None

        # Halve: re-score with sigma(Q), keep top half
        if len(candidates) > 1:
            refreshed = tree.get_root_children_info()
            info_by_pool = {c[1]: c for c in refreshed}
            max_n = max((info_by_pool.get(children[i][1], (None, None, None, 0, 0))[3]
                         for i in candidates), default=0)

            def _score(i: int) -> float:
                pool_idx = children[i][1]
                info = info_by_pool.get(pool_idx, (None, None, None, 0, 0.0))
                q_hat = info[4] if info[3] > 0 else 0.0
                sigma = (c_visit + max_n) * c_scale * q_hat
                return gumbels[i] + log_priors[i] + sigma

            candidates = sorted(candidates, key=_score, reverse=True)
            candidates = candidates[:(len(candidates) + 1) // 2]

    tree.forced_root_child = None
    result = _collect_mcts_results(tree)

    # Add improved policy for Gumbel mode
    improved = tree.get_improved_policy(c_visit=c_visit, c_scale=c_scale)
    BOARD_SIZE = 19
    half = (BOARD_SIZE - 1) // 2
    improved_entries = []
    for q, r in board.legal_moves():
        flat_idx = board.to_flat(q, r)
        if flat_idx < len(improved):
            p = improved[flat_idx]
            if p > 1e-6:
                improved_entries.append({"q": q, "r": r, "prob": round(float(p), 6)})
    improved_entries.sort(key=lambda e: e["prob"], reverse=True)
    result["improved_policy"] = improved_entries[:10]

    return result


def _collect_mcts_results(tree: MCTSTree) -> Dict[str, Any]:
    """Extract visit distribution from a searched tree."""
    top = tree.get_top_visits(20)
    visits = []
    for coord_str, v, prior, q_val in top:
        # Parse "(q,r)" → ints
        parts = coord_str.strip("()").split(",")
        q, r = int(parts[0]), int(parts[1])
        visits.append({
            "q": q, "r": r,
            "visits": int(v),
            "prior": round(float(prior), 6),
            "q_value": round(float(q_val), 6),
        })

    return {
        "visits": visits,
        "root_value": round(float(tree.root_value()), 6),
        "total_sims": int(tree.root_visits()),
        "top_k": visits[:10],
    }


# ── Flask routes ──────────────────────────────────────────────────────────────

@analyze_bp.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze a board position: raw policy + optional MCTS."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON", "detail": "Request body must be valid JSON"}), 400

    checkpoint = data.get("checkpoint")
    if not checkpoint:
        return jsonify({"error": "Missing checkpoint", "detail": "\"checkpoint\" field required"}), 400

    moves = data.get("moves", [])
    if not isinstance(moves, list):
        return jsonify({"error": "Invalid moves", "detail": "\"moves\" must be a list"}), 400

    mcts_cfg = data.get("mcts", {})

    def _do_analyze() -> Dict[str, Any]:
        try:
            engine = _get_engine(checkpoint)
        except FileNotFoundError:
            return {"_status": 404, "error": "Checkpoint not found", "detail": str(checkpoint)}
        except Exception as exc:
            log.error("model_load_failed", checkpoint=checkpoint, exc=str(exc))
            return {"_status": 500, "error": "Model load failed", "detail": str(exc)}

        try:
            board = _build_board(moves)
        except Exception as exc:
            return {"_status": 400, "error": "Invalid moves", "detail": str(exc)}

        result = _analyze_raw(engine, board)

        if mcts_cfg.get("enabled"):
            mode = mcts_cfg.get("mode", "puct")
            n_sims = min(int(mcts_cfg.get("simulations", 200)), 2000)
            try:
                if mode == "gumbel":
                    mcts_result = _run_gumbel(
                        board, engine, n_sims,
                        c_puct=float(mcts_cfg.get("c_puct", 1.5)),
                        m=int(mcts_cfg.get("m", 16)),
                        c_visit=float(mcts_cfg.get("c_visit", 50.0)),
                        c_scale=float(mcts_cfg.get("c_scale", 1.0)),
                    )
                else:
                    mcts_result = _run_puct(
                        board, engine, n_sims,
                        c_puct=float(mcts_cfg.get("c_puct", 1.5)),
                    )
                result["mcts"] = mcts_result
            except Exception as exc:
                log.error("mcts_failed", exc=str(exc))
                result["mcts_error"] = str(exc)

        return result

    future = _executor.submit(_do_analyze)
    result = future.result(timeout=60)

    status = result.pop("_status", 200)
    return jsonify(result), status


@analyze_bp.route("/api/analyze/checkpoints", methods=["GET"])
def list_checkpoints():
    """List available checkpoint files."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return jsonify([])

    entries = []
    for p in ckpt_dir.glob("*.pt"):
        try:
            stat = p.stat()
            entries.append({
                "path": str(p),
                "size_mb": round(stat.st_size / 1e6, 1),
                "mtime": stat.st_mtime,
            })
        except OSError:
            continue

    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return jsonify(entries)
