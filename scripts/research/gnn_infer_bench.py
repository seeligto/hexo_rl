#!/usr/bin/env python3
"""gnn_infer_bench.py — WP-A laptop shakeout (D-M R-LADDER GNN-integration COND-3).

Answers COND-3 of the GNN-integration program (docs/designs/gnn_integration_scope.md
§Measurements / §C1 / §C6): does the CPU-derived cost story (ORT best; forward
dominates, build small) hold under CUDA, and what is graph-build cost relative
to the dense CNN forward — on REAL self-play position data, not synthetic.

Cell matrix (see gnn_integration_scope.md §Measurements for the priors this
re-tests):
  GNN (net=GnnBcNet, block-diagonal disjoint-union batching, same convention
  as its forward_batch):
    1. torch-CPU fp32
    2. torch-CUDA fp32 (cuda-synchronized timing)
    3. torch-CUDA autocast fp16
    4. ORT-CPU, intra_op_num_threads in {1,2,4,8}
    5. ORT-CUDA (CUDAExecutionProvider)
  at two scales (probe: hidden=128/4L, 283,970 params, loaded from a real BC
  checkpoint; prod-proxy: hidden=256/4L, fresh init) x batch sizes.

  Comparator: production CNN (HexTacToeNet, v6_live2_ls, filters=128,
  blocks=12, dist65) forward, torch-CPU fp32 + torch-CUDA autocast fp16.

Also measures: the Python graph-builder (build_axis_graph_raw) cost/position,
the Python-reachable dense plane-encode cost/position (GameState.to_tensor),
a Rust native-builder proxy (hexo-strix's axis_graph.rs, via an untracked
ad hoc harness bin), an ONNX-export + parity gate for the ORT cells, and
per-cell mean GPU utilization (nvidia-smi, 1 Hz) on every CUDA cell.

CRITICAL import-order gotcha: `torch` MUST be imported before `onnxruntime`,
or the CUDA execution provider fails to load libcudart.so.13 (torch preloads
the bundled CUDA 13 libs onnxruntime otherwise can't find).

Usage:
    .venv/bin/python scripts/research/gnn_infer_bench.py \\
        --device all --scales both --batch-sizes 16,32,64,128,256 --n-runs 10

    # fast smoke (pipeline sanity check, NOT the real measurement):
    .venv/bin/python scripts/research/gnn_infer_bench.py \\
        --n-positions 40 --batch-sizes 16,32 --n-runs 2 --warmup 1
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml package is deprecated.*")

import argparse
import copy
import gc
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

import numpy as np
import torch  # MUST precede `import onnxruntime` (see module docstring).
import torch.nn as nn

import onnxruntime as ort  # noqa: E402  (import-order gotcha, see docstring)

from engine import Board  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw  # noqa: E402
from hexo_rl.probes.gnn_bc.gnn_bc_net import GnnBcNet  # noqa: E402
from hexo_rl.model.network import HexTacToeNet  # noqa: E402
from hexo_rl.encoding import lookup as lookup_encoding  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY_DIR = Path("/home/timmy/Work/Hexo/hexo_rl/logs/replays")
PRIMARY_REPLAY_FILES = [
    "games_2026-07-10.jsonl",
    "games_2026-07-12.jsonl",
    "games_2026-07-14.jsonl",
]
FALLBACK_REPLAY_GLOB = "games_2026-06-*.jsonl"

# Fixed probe geometry — matches hexo_rl.bots.strix_v1_graph / gnn_bc_bot.py /
# train_bc.py convention (_STRIX_WIN_LENGTH / _STRIX_RADIUS), decoupled from
# whatever radius curriculum the source self-play games were actually played
# under.
WIN_LENGTH = 6
RADIUS = 6

IQR_UNSTABLE_THRESHOLD = 0.15
PARITY_THRESHOLD = 1e-4
N_PARITY_GRAPHS = 24


# ─────────────────────────────────────────────────────────────────────────
# Timing primitives (docs/rules/perf-targets.md methodology: warmup + n
# timed runs, median + IQR, IQR-gate rerun once if IQR/median > 0.15).
# ─────────────────────────────────────────────────────────────────────────

class GpuUtilSampler:
    """Background 1 Hz nvidia-smi GPU-utilization sampler."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.samples: List[float] = []
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                line = out.stdout.strip().splitlines()[0]
                self.samples.append(float(line))
            except Exception:
                pass
            self._stop_evt.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def mean(self) -> Optional[float]:
        return statistics.mean(self.samples) if self.samples else None


def compute_stats(times_ms: List[float]) -> Dict[str, Any]:
    times_ms = sorted(times_ms)
    med = statistics.median(times_ms)
    if len(times_ms) >= 4:
        lower = times_ms[: len(times_ms) // 2]
        upper = times_ms[(len(times_ms) + 1) // 2:]
        q1 = statistics.median(lower)
        q3 = statistics.median(upper)
    else:
        q1, q3 = times_ms[0], times_ms[-1]
    iqr = q3 - q1
    return {
        "median_ms": med,
        "iqr_ms": iqr,
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "unstable": (iqr / med) > IQR_UNSTABLE_THRESHOLD if med > 0 else False,
        "raw_ms": times_ms,
    }


def measure_cell(
    fn: Callable[[], Any],
    n_runs: int = 10,
    warmup: int = 5,
    sync_cuda: bool = False,
    gpu_util: bool = False,
    min_gpu_util_duration: float = 1.5,
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()

    sampler = GpuUtilSampler() if gpu_util else None
    if sampler:
        sampler.start()

    times_ms: List[float] = []
    t_wall_start = time.perf_counter()
    for _ in range(n_runs):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    # Pad with untimed reps so fast cells still get >=1-2 util samples.
    while sampler is not None and (time.perf_counter() - t_wall_start) < min_gpu_util_duration:
        fn()
        if sync_cuda:
            torch.cuda.synchronize()

    if sampler:
        sampler.stop()

    stats = compute_stats(times_ms)
    stats["gpu_util_mean"] = sampler.mean() if sampler else None
    stats["gpu_util_n_samples"] = len(sampler.samples) if sampler else 0
    return stats


def measure_cell_with_gate(fn: Callable[[], Any], **kwargs: Any) -> Dict[str, Any]:
    """IQR-gate: rerun once if unstable, keep the rerun's numbers (note both)."""
    stats = measure_cell(fn, **kwargs)
    if stats["unstable"]:
        rerun = measure_cell(fn, **kwargs)
        rerun["unstable_first_attempt"] = True
        rerun["first_attempt_median_ms"] = stats["median_ms"]
        rerun["first_attempt_unstable"] = stats["unstable"]
        return rerun
    return stats


# ─────────────────────────────────────────────────────────────────────────
# Real-position sampling (standing red-team order: no synthetic uniform).
# ─────────────────────────────────────────────────────────────────────────

def load_games_from_file(path: Path) -> List[Tuple[List[Tuple[int, int]], str]]:
    games: List[Tuple[List[Tuple[int, int]], str]] = []
    if not path.exists():
        return games
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            moves = d.get("moves") or []
            if len(moves) < 4:
                continue
            games.append(([(int(q), int(r)) for q, r in moves], d.get("outcome", "draw")))
    return games


def replay_all_plies(moves: List[Tuple[int, int]]):
    """Yield (ply, stones_dict, current_player, moves_remaining, state) for
    every ply BEFORE move `ply` is applied. Mirrors
    hexo_rl.probes.gnn_bc.bc_data.replay_positions's stones/current_player/
    moves_remaining derivation, but (a) unrestricted to the full ply range
    (we want the raw self-play ply distribution, not the §114 [2,150)
    corpus-training window) and (b) also yields the live GameState (needed
    for the dense plane-encode timing seam, GameState.to_tensor())."""
    board = Board()
    state = GameState.from_board(board)
    for ply, (q, r) in enumerate(moves):
        stones = {(sq, sr): p for (sq, sr, p) in board.get_stones()}
        yield ply, stones, state.current_player, state.moves_remaining, state
        try:
            state = state.apply_move(board, q, r)
        except Exception:
            break


def _collect_candidates(paths: List[Path]) -> List[dict]:
    candidates: List[dict] = []
    for p in paths:
        games = load_games_from_file(p)
        for gi, (moves, outcome) in enumerate(games):
            for ply, stones, cur, mr, state in replay_all_plies(moves):
                if len(stones) < 2:
                    continue  # skip degenerate near-empty boards
                candidates.append({
                    "source_file": p.name,
                    "game_idx": gi,
                    "ply": ply,
                    "stones": stones,
                    "current_player": cur,
                    "moves_remaining": mr,
                    "state": state,
                })
    return candidates


def sample_positions(n_positions: int, positions_dir: Path, seed: int) -> Tuple[List[dict], dict]:
    """Stratified-by-ply sample of >= n_positions real self-play positions."""
    primary_paths = [positions_dir / f for f in PRIMARY_REPLAY_FILES]
    files_used: List[str] = [str(p) for p in primary_paths if p.exists()]
    candidates = _collect_candidates(primary_paths)

    fallback_files_used: List[str] = []
    if len(candidates) < n_positions * 3:
        fallback_files = sorted(positions_dir.glob(FALLBACK_REPLAY_GLOB))
        extra_paths = []
        for p in fallback_files:
            extra_paths.append(p)
            probe = _collect_candidates([p])
            candidates.extend(probe)
            if probe:
                fallback_files_used.append(str(p))
            if len(candidates) >= n_positions * 3:
                break

    n_games_pool = len({(c["source_file"], c["game_idx"]) for c in candidates})
    if not candidates:
        raise RuntimeError(f"no candidate positions found under {positions_dir}")

    n_target = min(n_positions, len(candidates))
    max_ply = max(c["ply"] for c in candidates)
    min_ply = min(c["ply"] for c in candidates)

    # Stratified-by-ply sampling: bucket into 20 equal-width ply bins,
    # round-robin draw so the selected set is roughly uniform over plies.
    rng = random.Random(seed)
    n_bins = 20
    bin_width = max(1.0, (max_ply - min_ply + 1) / n_bins)
    bins: Dict[int, List[dict]] = {}
    for c in candidates:
        b = min(n_bins - 1, int((c["ply"] - min_ply) / bin_width))
        bins.setdefault(b, []).append(c)
    for b in bins:
        rng.shuffle(bins[b])

    selected: List[dict] = []
    bin_keys = sorted(bins.keys())
    ptrs = {b: 0 for b in bin_keys}
    while len(selected) < n_target:
        progressed = False
        for b in bin_keys:
            if len(selected) >= n_target:
                break
            if ptrs[b] < len(bins[b]):
                selected.append(bins[b][ptrs[b]])
                ptrs[b] += 1
                progressed = True
        if not progressed:
            break

    ply_hist_bins = {str(b): len(bins.get(b, [])) for b in bin_keys}
    selected_ply_hist: Dict[int, int] = {}
    for c in selected:
        selected_ply_hist[c["ply"]] = selected_ply_hist.get(c["ply"], 0) + 1

    provenance = {
        "files_used_primary": files_used,
        "files_used_fallback": fallback_files_used,
        "n_candidates_pool": len(candidates),
        "n_games_pool": n_games_pool,
        "n_positions_requested": n_positions,
        "n_positions_sampled": len(selected),
        "seed": seed,
        "ply_range_pool": [min_ply, max_ply],
        "pool_ply_histogram_20bin": ply_hist_bins,
        "sampling_method": "stratified round-robin over 20 equal-width ply bins, seeded shuffle within bin",
    }
    return selected, provenance


def positions_to_json_records(selected: List[dict]) -> List[dict]:
    out = []
    for i, c in enumerate(selected):
        stones_list = [[int(q), int(r), int(p)] for (q, r), p in c["stones"].items()]
        out.append({
            "idx": i,
            "source_file": c["source_file"],
            "game_idx": c["game_idx"],
            "ply": c["ply"],
            "current_player": int(c["current_player"]),
            "moves_remaining": int(c["moves_remaining"]),
            "stones": stones_list,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────
# Graph building + Python builder timing + dense-encode timing.
# ─────────────────────────────────────────────────────────────────────────

def _build_one(stones: dict, current_player: int, moves_remaining: int) -> dict:
    return build_axis_graph_raw(
        stones, current_player, moves_remaining,
        win_length=WIN_LENGTH, radius=RADIUS,
        prune_empty_edges=True, threat_features=True, relative_stones=True,
    )


def build_graphs_and_time(selected: List[dict], n_runs: int, warmup: int) -> Tuple[List[dict], dict]:
    for _ in range(warmup):
        for c in selected:
            _build_one(c["stones"], c["current_player"], c["moves_remaining"])

    per_rep_ms: List[float] = []
    graphs: Optional[List[dict]] = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        gs = [_build_one(c["stones"], c["current_player"], c["moves_remaining"]) for c in selected]
        elapsed = time.perf_counter() - t0
        per_rep_ms.append(elapsed * 1000.0 / len(selected))
        graphs = gs

    stats = compute_stats(per_rep_ms)
    stats["unit"] = "ms/position"

    assert graphs is not None
    nodes = [g["num_nodes"] for g in graphs]
    edges = [len(g["edge_src"]) for g in graphs]
    histogram = {
        "n_positions": len(selected),
        "mean_nodes": statistics.mean(nodes), "p50_nodes": statistics.median(nodes),
        "p90_nodes": float(np.percentile(nodes, 90)), "max_nodes": max(nodes),
        "mean_edges": statistics.mean(edges), "p50_edges": statistics.median(edges),
        "p90_edges": float(np.percentile(edges, 90)), "max_edges": max(edges),
    }
    return graphs, {"python_build_timing": stats, "graph_size_histogram": histogram}


def time_dense_encode(selected: List[dict], n_runs: int, warmup: int) -> dict:
    states = [c["state"] for c in selected if c.get("state") is not None]
    if not states:
        return {"status": "SKIPPED", "reason": "no live GameState objects (positions reused from JSON)"}
    for _ in range(warmup):
        for s in states:
            s.to_tensor()
    per_rep_ms: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for s in states:
            s.to_tensor()
        elapsed = time.perf_counter() - t0
        per_rep_ms.append(elapsed * 1000.0 / len(states))
    stats = compute_stats(per_rep_ms)
    stats["unit"] = "ms/position"
    stats["status"] = "MEASURED"
    stats["note"] = (
        "Python-reachable GameState.to_tensor() dense plane-encode (K cluster "
        "views x 18ch, numpy). NOT the Rust engine::encode_state_to_buffer_channels "
        "<50us/position figure cited in the scoping doc (that path is "
        "Rust-internal, not Python-reachable per-position — cited, not measured, "
        "in this report)."
    )
    return stats


def collate_batch(graphs: List[dict], device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-diagonal disjoint-union batch — same convention as GnnBcNet.forward_batch
    / hexo_rl.probes.gnn_bc.train_bc._collate_gnn_dict (the reference oracle)."""
    xs, eis, eas, lms = [], [], [], []
    node_offset = 0
    for g in graphs:
        n, fdim = g["num_nodes"], g["fdim"]
        x = torch.tensor(g["features"], dtype=torch.float32).reshape(n, fdim)
        E = len(g["edge_src"])
        if E:
            ei = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.int64) + node_offset
            ea = torch.tensor(g["edge_attr"], dtype=torch.float32).reshape(E, 5)
        else:
            ei = torch.zeros((2, 0), dtype=torch.int64)
            ea = torch.zeros((0, 5), dtype=torch.float32)
        lm = torch.tensor(g["legal_mask"], dtype=torch.bool)
        xs.append(x); eis.append(ei); eas.append(ea); lms.append(lm)
        node_offset += n
    x = torch.cat(xs, 0).to(device)
    edge_index = torch.cat(eis, 1).to(device)
    edge_attr = torch.cat(eas, 0).to(device)
    legal_mask = torch.cat(lms, 0).to(device)
    return x, edge_index, edge_attr, legal_mask


def build_batch_index_sets(n_graphs: int, batch_sizes: List[int], seed: int) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    perm = list(range(n_graphs))
    rng.shuffle(perm)
    out: Dict[int, List[int]] = {}
    for bs in batch_sizes:
        if bs > n_graphs:
            idxs = [perm[i % n_graphs] for i in range(bs)]
        else:
            idxs = perm[:bs]
        out[bs] = idxs
    return out


# ─────────────────────────────────────────────────────────────────────────
# ONNX export (export-only reimplementation — does NOT modify gnn_bc_net.py).
#
# torch's ONNX symbolic for `Tensor.index_add_` (used by strix_v1_net's
# _GINEConv message aggregation) does NOT accumulate duplicate destination
# indices correctly (verified empirically: max|eager - ORT| ~0.37 on a real
# 204-node/816-edge graph, with torch itself emitting "ONNX export does not
# support duplicated values in 'index' field, this will cause the ONNX model
# to be incorrect"). `Tensor.scatter_add` DOES export correctly (opset 18
# ScatterElements w/ reduction='add') and is mathematically identical to
# index_add_ for this use — verified eager-vs-eager max diff 0.0, ORT parity
# ~4e-7. So the export path below is a parameter-identical reimplementation
# of RepresentationNetwork/_GINEConv using scatter_add instead of index_add_,
# loaded via strict state_dict copy from the real (trained or fresh-init) net.
# ─────────────────────────────────────────────────────────────────────────

class _ExportGINEConv(nn.Module):
    def __init__(self, hidden: int, edge_in: int) -> None:
        super().__init__()
        self.register_buffer("eps", torch.zeros(1))
        self.nn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.lin = nn.Linear(edge_in, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        msg = (x.index_select(0, src) + self.lin(edge_attr)).relu()
        idx = dst.unsqueeze(-1).expand(-1, x.shape[1])
        agg = torch.zeros((n, x.shape[1]), dtype=x.dtype, device=x.device).scatter_add(0, idx, msg)
        out = agg + (1.0 + self.eps) * x
        return self.nn(out)


class _ExportRepresentationNetwork(nn.Module):
    """ONNX-export-safe reimpl of strix_v1_net.RepresentationNetwork. Identical
    parameter names/shapes (state_dict strict=True compatible) — only the
    aggregation op differs (scatter_add vs index_add_). Export-only; never
    used for the eager/torch timing cells."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int, edge_dim: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList([_ExportGINEConv(hidden, hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        projected_edge_attr = self.edge_proj(edge_attr)
        hs = []
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            xn = norm(x)
            xc = conv(xn, edge_index, projected_edge_attr)
            x = xc + residual
            x = self.activation(x)
            hs.append(x)
        hs = [self.final_norm(h) for h in hs]
        return torch.cat(hs, dim=-1)


class _ExportForwardBatch(nn.Module):
    """Matches GnnBcNet.forward_batch exactly (policy-only, disjoint-union batch)."""

    def __init__(self, representation: _ExportRepresentationNetwork, policy_mlp: nn.Sequential) -> None:
        super().__init__()
        self.representation = representation
        self.policy_mlp = policy_mlp

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                legal_mask: torch.Tensor) -> torch.Tensor:
        emb = self.representation(x, edge_index, edge_attr)
        legal_emb = emb[legal_mask]
        return self.policy_mlp(legal_emb).squeeze(-1)


def build_export_wrapper(net: GnnBcNet) -> _ExportForwardBatch:
    rep = net.representation
    in_dim = rep.input_proj.in_features
    hidden = rep.hidden
    num_layers = rep.num_layers
    edge_dim = rep.edge_proj.in_features
    export_rep = _ExportRepresentationNetwork(in_dim, hidden, num_layers, edge_dim)
    export_rep.load_state_dict(rep.state_dict(), strict=True)
    export_rep.eval()
    wrapper = _ExportForwardBatch(export_rep, net.policy_head.mlp)
    wrapper.eval()
    return wrapper


def export_onnx(net: GnnBcNet, example_batch: Tuple[torch.Tensor, ...], out_path: Path) -> None:
    x, edge_index, edge_attr, legal_mask = (t.cpu() for t in example_batch)
    wrapper = build_export_wrapper(net)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper, (x, edge_index, edge_attr, legal_mask), str(out_path),
            input_names=["x", "edge_index", "edge_attr", "legal_mask"],
            output_names=["policy_logits"],
            dynamic_axes={
                "x": {0: "n_nodes"},
                "edge_index": {1: "n_edges"},
                "edge_attr": {0: "n_edges"},
                "legal_mask": {0: "n_nodes"},
                "policy_logits": {0: "n_legal"},
            },
            opset_version=18,
            dynamo=False,
        )


def check_parity(net: GnnBcNet, onnx_path: Path, graphs: List[dict], n_check: int) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    net_cpu = net.to("cpu").eval()
    max_diff = 0.0
    n_done = 0
    per_graph_diffs = []
    for g in graphs[:n_check]:
        x, edge_index, edge_attr, legal_mask = collate_batch([g], "cpu")
        with torch.no_grad():
            eager = net_cpu.forward_batch(x, edge_index, edge_attr, legal_mask).numpy()
        ort_out = sess.run(None, {
            "x": x.numpy(), "edge_index": edge_index.numpy(),
            "edge_attr": edge_attr.numpy(), "legal_mask": legal_mask.numpy(),
        })[0]
        if eager.shape != ort_out.shape or eager.size == 0:
            continue
        d = float(np.abs(eager - ort_out).max())
        per_graph_diffs.append(d)
        max_diff = max(max_diff, d)
        n_done += 1
    return {
        "n_graphs_checked": n_done,
        "max_abs_diff": max_diff,
        "mean_abs_diff": statistics.mean(per_graph_diffs) if per_graph_diffs else None,
        "threshold": PARITY_THRESHOLD,
        "pass": max_diff < PARITY_THRESHOLD,
    }


def do_onnx_export_and_parity(scales: List[int], scale_names: List[str], out_dir: Path,
                               graphs: List[dict], build_net_fn: Callable[[str, str], GnnBcNet]) -> Tuple[dict, dict]:
    onnx_paths: Dict[str, Optional[Path]] = {}
    export_results: Dict[str, dict] = {}
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for scale in scale_names:
        net = build_net_fn(scale, "cpu")
        onnx_path = onnx_dir / f"gnn_bc_{scale}.onnx"
        example_batch = collate_batch(graphs[:4], "cpu")
        entry: Dict[str, Any] = {}
        try:
            export_onnx(net, example_batch, onnx_path)
            entry["status"] = "OK"
            entry["path"] = str(onnx_path)
            entry["error"] = None
        except Exception as e:  # noqa: BLE001 — record verbatim, don't crash the run
            entry["status"] = "EXPORT-BLOCKED"
            entry["path"] = None
            entry["error"] = f"{type(e).__name__}: {e}"
        if entry["status"] == "OK":
            try:
                parity = check_parity(net, onnx_path, graphs, N_PARITY_GRAPHS)
                entry["parity"] = parity
                if not parity["pass"]:
                    entry["status"] = "PARITY-FAIL"
            except Exception as e:  # noqa: BLE001
                entry["status"] = "PARITY-CHECK-FAILED"
                entry["parity_error"] = f"{type(e).__name__}: {e}"
        export_results[scale] = entry
        onnx_paths[scale] = Path(entry["path"]) if entry.get("status") == "OK" and entry.get("path") else None
    return onnx_paths, export_results


# ─────────────────────────────────────────────────────────────────────────
# ORT session helpers.
# ─────────────────────────────────────────────────────────────────────────

def make_ort_cpu_session(onnx_path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


ORT_GPU_MEM_GB = 3.0  # overridden by --ort-gpu-mem-gb


def make_ort_cuda_session(onnx_path: Path) -> ort.InferenceSession:
    # Release torch's cached VRAM first — on the 8 GB laptop card the caching
    # allocator's reserved blocks otherwise starve the ORT BFC arena (first
    # shakeout run died here: 29.5 MB alloc failure after the torch cells).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    cuda_opts = {
        "gpu_mem_limit": int(ORT_GPU_MEM_GB * 1024 * 1024 * 1024),
        "arena_extend_strategy": "kSameAsRequested",
    }
    return ort.InferenceSession(
        str(onnx_path),
        providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
    )


# ─────────────────────────────────────────────────────────────────────────
# Net builders.
# ─────────────────────────────────────────────────────────────────────────

def build_gnn_net(scale: str, device: str, checkpoint_path: Optional[Path]) -> GnnBcNet:
    if scale == "probe":
        net = GnnBcNet(hidden=128, num_layers=4)
        if checkpoint_path is not None and checkpoint_path.exists():
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            net.load_state_dict(sd, strict=True)
    elif scale == "prod":
        torch.manual_seed(0)
        net = GnnBcNet(hidden=256, num_layers=4)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    net.to(device).eval()
    return net


def build_cnn_net(device: str) -> HexTacToeNet:
    torch.manual_seed(0)
    net = HexTacToeNet(filters=128, res_blocks=12, encoding="v6_live2_ls", value_head_type="dist65")
    net.to(device).eval()
    return net


# ─────────────────────────────────────────────────────────────────────────
# Cell runners.
# ─────────────────────────────────────────────────────────────────────────

def record_cell(results: dict, **kwargs: Any) -> None:
    stats = kwargs.pop("stats")
    bs = kwargs.get("batch_size")
    entry = dict(kwargs)
    med = stats["median_ms"]
    entry.update({
        "median_ms": med,
        "iqr_ms": stats["iqr_ms"],
        "iqr_over_median": (stats["iqr_ms"] / med) if med else None,
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "unstable": stats.get("unstable", False),
        "unstable_first_attempt": stats.get("unstable_first_attempt", False),
        "gpu_util_mean": stats.get("gpu_util_mean"),
        "gpu_util_n_samples": stats.get("gpu_util_n_samples"),
        "throughput_pos_per_s": (bs / (med / 1000.0)) if bs and med else None,
    })
    results["cells"].append(entry)
    label = f"{entry.get('component')}/{entry.get('scale')}/{entry.get('device')}/{entry.get('precision')}/{entry.get('backend')}"
    extra = f" T={entry['ort_threads']}" if entry.get("ort_threads") else ""
    print(
        f"[cell] {label}{extra} bs={bs}: median={med:.3f}ms iqr={stats['iqr_ms']:.3f}ms "
        f"unstable={entry['unstable']} gpu_util={entry['gpu_util_mean']}",
        flush=True,
    )


def gnn_torch_cells(args: argparse.Namespace, results: dict, graphs: List[dict],
                     idx_by_bs: Dict[int, List[int]], checkpoint_path: Optional[Path]) -> None:
    for scale in args.scales_list:
        combos: List[Tuple[str, str]] = []
        if args.device in ("cpu", "all"):
            combos.append(("cpu", "fp32"))
        if args.device in ("cuda", "all") and torch.cuda.is_available():
            combos.append(("cuda", "fp32"))
            combos.append(("cuda", "fp16_autocast"))
        for device, precision in combos:
            net = build_gnn_net(scale, device, checkpoint_path)
            n_params = net.num_params()
            for bs in args.batch_sizes:
                batch_graphs = [graphs[i] for i in idx_by_bs[bs]]
                x, ei, ea, lm = collate_batch(batch_graphs, device)

                if precision == "fp16_autocast":
                    def fn(net=net, x=x, ei=ei, ea=ea, lm=lm):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            with torch.no_grad():
                                net.forward_batch(x, ei, ea, lm)
                else:
                    def fn(net=net, x=x, ei=ei, ea=ea, lm=lm):
                        with torch.no_grad():
                            net.forward_batch(x, ei, ea, lm)

                stats = measure_cell_with_gate(
                    fn, n_runs=args.n_runs, warmup=args.warmup,
                    sync_cuda=(device == "cuda"), gpu_util=(device == "cuda"),
                    min_gpu_util_duration=args.gpu_util_duration,
                )
                record_cell(results, component="gnn_torch", scale=scale, device=device,
                            precision=precision, backend="torch", ort_threads=None,
                            batch_size=bs, n_params=n_params, stats=stats)


def ort_cells(args: argparse.Namespace, results: dict, graphs: List[dict],
               idx_by_bs: Dict[int, List[int]], onnx_paths: Dict[str, Optional[Path]]) -> None:
    for scale in args.scales_list:
        onnx_path = onnx_paths.get(scale)
        if onnx_path is None:
            results.setdefault("notes", []).append(
                f"ORT cells SKIPPED for scale={scale}: no valid ONNX export (see onnx_export section)."
            )
            continue
        n_params = build_gnn_net(scale, "cpu", None).num_params() if scale == "prod" else None

        if args.device in ("cpu", "all"):
            for T in (1, 2, 4, 8):
                sess = make_ort_cpu_session(onnx_path, T)
                for bs in args.batch_sizes:
                    batch_graphs = [graphs[i] for i in idx_by_bs[bs]]
                    x, ei, ea, lm = collate_batch(batch_graphs, "cpu")
                    feed = {"x": x.numpy(), "edge_index": ei.numpy(),
                            "edge_attr": ea.numpy(), "legal_mask": lm.numpy()}

                    def fn(sess=sess, feed=feed):
                        sess.run(None, feed)

                    try:
                        stats = measure_cell_with_gate(
                            fn, n_runs=args.n_runs, warmup=args.warmup, sync_cuda=False, gpu_util=False,
                        )
                    except Exception as e:  # noqa: BLE001
                        results.setdefault("notes", []).append(
                            f"ORT-CPU cell FAILED-SKIPPED scale={scale} T={T} bs={bs}: {type(e).__name__}: {e}"
                        )
                        print(f"[cell] gnn_ort/{scale}/cpu T={T} bs={bs}: FAILED-SKIPPED ({type(e).__name__})", flush=True)
                        continue
                    record_cell(results, component="gnn_ort", scale=scale, device="cpu",
                                precision="fp32", backend="ort", ort_threads=T,
                                batch_size=bs, n_params=n_params, stats=stats)

        if (args.device in ("cuda", "all") and torch.cuda.is_available()
                and "CUDAExecutionProvider" in ort.get_available_providers()):
            try:
                sess = make_ort_cuda_session(onnx_path)
            except Exception as e:  # noqa: BLE001
                results.setdefault("notes", []).append(
                    f"ORT-CUDA session creation FAILED for scale={scale}: {type(e).__name__}: {e}"
                )
                sess = None
            if sess is not None:
                for bs in args.batch_sizes:
                    batch_graphs = [graphs[i] for i in idx_by_bs[bs]]
                    x, ei, ea, lm = collate_batch(batch_graphs, "cpu")
                    feed = {"x": x.numpy(), "edge_index": ei.numpy(),
                            "edge_attr": ea.numpy(), "legal_mask": lm.numpy()}

                    def fn(sess=sess, feed=feed):
                        sess.run(None, feed)

                    try:
                        stats = measure_cell_with_gate(
                            fn, n_runs=args.n_runs, warmup=args.warmup, sync_cuda=False, gpu_util=True,
                            min_gpu_util_duration=args.gpu_util_duration,
                        )
                    except Exception as e:  # noqa: BLE001
                        results.setdefault("notes", []).append(
                            f"ORT-CUDA cell FAILED-SKIPPED scale={scale} bs={bs}: {type(e).__name__}: {e}"
                        )
                        print(f"[cell] gnn_ort/{scale}/cuda bs={bs}: FAILED-SKIPPED ({type(e).__name__})", flush=True)
                        continue
                    record_cell(results, component="gnn_ort", scale=scale, device="cuda",
                                precision="fp32", backend="ort", ort_threads=None,
                                batch_size=bs, n_params=n_params, stats=stats)
                del sess
                gc.collect()


def cnn_cells(args: argparse.Namespace, results: dict) -> None:
    spec = lookup_encoding("v6_live2_ls")
    n_planes, H, W = spec.n_planes, spec.trunk_size, spec.trunk_size

    combos: List[Tuple[str, str]] = []
    if args.device in ("cpu", "all"):
        combos.append(("cpu", "fp32"))
    if args.device in ("cuda", "all") and torch.cuda.is_available():
        combos.append(("cuda", "fp16_autocast"))

    for device, precision in combos:
        net = build_cnn_net(device)
        n_params = sum(p.numel() for p in net.parameters())
        for bs in args.batch_sizes:
            torch.manual_seed(1)
            x = torch.randn(bs, n_planes, H, W, device=device)

            if precision == "fp16_autocast":
                def fn(net=net, x=x):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        with torch.no_grad():
                            net(x)
            else:
                def fn(net=net, x=x):
                    with torch.no_grad():
                        net(x)

            stats = measure_cell_with_gate(
                fn, n_runs=args.n_runs, warmup=args.warmup,
                sync_cuda=(device == "cuda"), gpu_util=(device == "cuda"),
                min_gpu_util_duration=args.gpu_util_duration,
            )
            record_cell(results, component="cnn_comparator", scale="prod_cnn_4.27M", device=device,
                        precision=precision, backend="torch", ort_threads=None,
                        batch_size=bs, n_params=n_params, stats=stats)


def check_strix_origin_compat(position_records: List[dict]) -> dict:
    """hexo-strix's public reconstruction API (Board::new() / GameState::from_state)
    HARD-CODES a rule our engine does not share: P1's opening stone is always at
    (0,0) (Board::new() unconditionally inserts it, with no way to override via
    the public API — verified by reading hexo-engine/src/board.rs). Our engine's
    Board() starts empty and P1's opening move is a free choice (verified:
    `Board().get_stones() == []`). A real position is only faithfully
    reconstructible through hexo-strix's API if (0,0) happens to be occupied by
    P1 in that position; if it's occupied by P2 the reconstruction hard-collides
    (Board::new()'s phantom P1-at-origin stone vs. the real P2 stone), and if
    it's empty the reconstruction would silently PHANTOM a P1 stone that was
    never actually played (representationally wrong, not just a crash) — so
    "empty" positions are excluded too, not just "P2" ones."""
    compat, incompat_p2, incompat_empty = [], 0, 0
    for r in position_records:
        origin_player = None
        for (q, rr, p) in r["stones"]:
            if q == 0 and rr == 0:
                origin_player = p
                break
        if origin_player == 1:
            compat.append(r)
        elif origin_player == -1:
            incompat_p2 += 1
        else:
            incompat_empty += 1
    return {
        "n_total": len(position_records),
        "n_compat_p1_at_origin": len(compat),
        "n_incompat_p2_at_origin": incompat_p2,
        "n_incompat_origin_empty": incompat_empty,
        "compat_records": compat,
    }


def run_rust_builder_proxy(position_records: List[dict], provenance: dict, rust_repo: Path,
                            n_reps: int, out_dir: Path) -> dict:
    hexo_rs_dir = rust_repo / "hexo-rs"
    if not hexo_rs_dir.exists():
        return {"status": "SKIPPED", "reason": f"{hexo_rs_dir} not found"}

    build_cmd = ["cargo", "build", "--release", "-j4", "-p", "hexo-mcts", "--bin", "wpa_axis_graph_bench"]
    try:
        proc = subprocess.run(build_cmd, cwd=str(hexo_rs_dir), capture_output=True, text=True, timeout=600)
    except Exception as e:  # noqa: BLE001
        return {"status": "SKIPPED", "reason": f"cargo build exception: {type(e).__name__}: {e}"}
    if proc.returncode != 0:
        return {"status": "SKIPPED", "reason": "cargo build failed (non-zero exit)",
                "stderr_tail": proc.stderr[-4000:]}

    binary = hexo_rs_dir / "target" / "release" / "wpa_axis_graph_bench"
    if not binary.exists():
        return {"status": "SKIPPED", "reason": f"binary not found at {binary} after successful build"}

    compat = check_strix_origin_compat(position_records)
    compat_summary = {k: v for k, v in compat.items() if k != "compat_records"}
    if compat["n_compat_p1_at_origin"] < 20:
        return {
            "status": "BLOCKED-BY-API-CONSTRAINT",
            "reason": (
                "hexo-strix's Board::new()/GameState::from_state hard-codes P1-opens-at-(0,0); "
                "our real self-play corpus does not share that rule (P1's opening move is a free "
                f"choice — only {compat['n_compat_p1_at_origin']}/{compat['n_total']} sampled "
                "positions have a real P1 stone at (0,0); see origin_compat below). Fewer than 20 "
                "of the sampled positions have a REAL P1 stone at (0,0), so no faithful subsample is "
                "reconstructible through hexo-strix's public API without either a hard collision "
                "(origin occupied by P2 in our data) or silently phantoming a stone that was never "
                "played (origin empty in our data). This is a genuine representation-compatibility "
                "finding, not a build failure: it independently confirms gnn_integration_scope.md "
                "§C1's recommendation to PORT the axis-graph algorithm against OUR OWN "
                "engine::Board type rather than reuse hexo-strix's Board/GameState wholesale."
            ),
            "origin_compat": compat_summary,
        }

    # Faithful-enough subsample exists — run the proxy on ONLY the positions
    # hexo-strix's API can reconstruct without corruption (biased subsample:
    # positions where a stone happens to sit at the origin; noted in the report).
    subset_records = compat["compat_records"]
    positions_json_path = out_dir / "wpa_positions_strix_compat_subset.json"
    positions_json_path.write_text(json.dumps({
        "provenance": provenance, "positions": subset_records,
    }, indent=1))

    out_json = out_dir / "rust_builder_proxy_raw.json"
    try:
        run = subprocess.run(
            [str(binary), str(positions_json_path), str(n_reps), str(out_json)],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as e:  # noqa: BLE001
        return {"status": "SKIPPED", "reason": f"harness run exception: {type(e).__name__}: {e}",
                "origin_compat": compat_summary}
    if run.returncode != 0:
        return {"status": "SKIPPED", "reason": "harness run failed (non-zero exit)",
                "stderr_tail": run.stderr[-4000:], "origin_compat": compat_summary}
    try:
        data = json.loads(out_json.read_text())
    except Exception as e:  # noqa: BLE001
        return {"status": "SKIPPED", "reason": f"could not parse harness output: {type(e).__name__}: {e}",
                "stdout_tail": run.stdout[-2000:], "origin_compat": compat_summary}
    data["status"] = "OK"
    data["binary"] = str(binary)
    data["origin_compat"] = compat_summary
    data["caveat"] = (
        "Biased subsample: only positions where a real stone happens to sit at (0,0) are "
        "reconstructible through hexo-strix's public API (see origin_compat). Node/edge counts "
        "and ns/pos on this subset are directionally informative but not a random sample of the "
        "full graph-size distribution reported in graph_size_histogram."
    )
    return data


# ─────────────────────────────────────────────────────────────────────────
# Markdown rendering.
# ─────────────────────────────────────────────────────────────────────────

def render_markdown(results: dict) -> str:
    lines: List[str] = []
    meta = results["meta"]
    lines.append("# GNN inference bench — raw results\n")
    lines.append(f"Hardware: {meta.get('hardware')}  \nGPU: {meta.get('gpu_name')}  \n"
                  f"torch {meta.get('torch_version')}, onnxruntime {meta.get('onnxruntime_version')}  \n"
                  f"Generated: {meta.get('timestamp')}\n")

    prov = results.get("positions_provenance", {})
    lines.append("## Position provenance\n")
    lines.append(f"- files (primary): {prov.get('files_used_primary')}")
    lines.append(f"- files (fallback, June): {prov.get('files_used_fallback')}")
    lines.append(f"- candidate pool: {prov.get('n_candidates_pool')} positions across "
                  f"{prov.get('n_games_pool')} games")
    lines.append(f"- sampled: {prov.get('n_positions_sampled')} (requested {prov.get('n_positions_requested')})")
    lines.append(f"- ply range in pool: {prov.get('ply_range_pool')}")
    lines.append(f"- seed: {prov.get('seed')}\n")

    hist = results.get("graph_size_histogram", {})
    if hist:
        lines.append("## Graph size histogram (sampled positions, win_length=6, radius=6)\n")
        lines.append("| metric | nodes | edges |")
        lines.append("|---|---|---|")
        lines.append(f"| mean | {hist.get('mean_nodes'):.1f} | {hist.get('mean_edges'):.1f} |")
        lines.append(f"| p50 | {hist.get('p50_nodes'):.1f} | {hist.get('p50_edges'):.1f} |")
        lines.append(f"| p90 | {hist.get('p90_nodes'):.1f} | {hist.get('p90_edges'):.1f} |")
        lines.append(f"| max | {hist.get('max_nodes')} | {hist.get('max_edges')} |\n")

    bt = results.get("python_build_timing", {})
    if bt:
        lines.append("## Python graph-builder timing (build_axis_graph_raw)\n")
        lines.append(f"- median {bt.get('median_ms'):.4f} ms/position, IQR {bt.get('iqr_ms'):.4f} ms "
                      f"(unstable={bt.get('unstable')})\n")

    de = results.get("dense_encode_timing", {})
    if de:
        lines.append("## Dense plane-encode timing (GameState.to_tensor, Python-reachable)\n")
        if de.get("status") == "MEASURED":
            lines.append(f"- median {de.get('median_ms'):.4f} ms/position, IQR {de.get('iqr_ms'):.4f} ms "
                          f"(unstable={de.get('unstable')})")
            lines.append(f"- {de.get('note')}\n")
        else:
            lines.append(f"- {de.get('status')}: {de.get('reason')}\n")

    rb = results.get("rust_builder_proxy", {})
    if rb:
        lines.append("## Rust builder proxy (hexo-strix axis_graph.rs)\n")
        if rb.get("status") == "OK":
            lines.append(f"- median {rb.get('median_ns_per_pos'):.0f} ns/position "
                          f"({rb.get('median_ns_per_pos', 0) / 1e6:.4f} ms/position), "
                          f"IQR {rb.get('iqr_ns_per_pos'):.0f} ns (unstable={rb.get('unstable')})")
            lines.append(f"- n_positions={rb.get('n_positions')}, n_reps={rb.get('n_reps')}")
            lines.append(f"- mean_nodes={rb.get('mean_nodes'):.1f}, mean_edges={rb.get('mean_edges'):.1f}")
            if rb.get("caveat"):
                lines.append(f"- CAVEAT: {rb.get('caveat')}")
            if rb.get("origin_compat"):
                lines.append(f"- origin_compat: {rb.get('origin_compat')}")
            lines.append("")
        else:
            lines.append(f"- {rb.get('status')}: {rb.get('reason')}")
            if rb.get("origin_compat"):
                lines.append(f"- origin_compat: {rb.get('origin_compat')}")
            lines.append("")

    onnx = results.get("onnx_export", {})
    if onnx:
        lines.append("## ONNX export + parity gate\n")
        for scale, entry in onnx.items():
            lines.append(f"### scale={scale}: {entry.get('status')}")
            if entry.get("error"):
                lines.append(f"- error: `{entry['error']}`")
            if "parity" in entry:
                p = entry["parity"]
                lines.append(f"- parity: max|Δ|={p.get('max_abs_diff'):.3e} over "
                              f"{p.get('n_graphs_checked')} real graphs "
                              f"(threshold {p.get('threshold')}) -> {'PASS' if p.get('pass') else 'FAIL'}")
            lines.append("")

    # cell tables, grouped
    lines.append("## Cell timing tables (median ms; IQR shown as ±)\n")
    cells = results.get("cells", [])
    groups: Dict[Tuple, List[dict]] = {}
    for c in cells:
        key = (c.get("component"), c.get("scale"), c.get("device"), c.get("precision"),
               c.get("backend"), c.get("ort_threads"))
        groups.setdefault(key, []).append(c)

    for key in sorted(groups.keys(), key=lambda k: tuple(str(x) for x in k)):
        component, scale, device, precision, backend, threads = key
        rows = sorted(groups[key], key=lambda c: c["batch_size"])
        title = f"{component} | scale={scale} | device={device} | precision={precision} | backend={backend}"
        if threads:
            title += f" | intra_op_threads={threads}"
        lines.append(f"### {title}\n")
        lines.append("| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            gu = r.get("gpu_util_mean")
            gu_s = f"{gu:.1f} (n={r.get('gpu_util_n_samples')})" if gu is not None else "n/a"
            lines.append(
                f"| {r['batch_size']} | {r['median_ms']:.3f} | {r['iqr_ms']:.3f} | "
                f"{r.get('throughput_pos_per_s', 0):.0f} | {gu_s} | {r.get('unstable')} |"
            )
        lines.append("")

    notes = results.get("notes", [])
    if notes:
        lines.append("## Notes / warnings\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Main.
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    ap.add_argument("--scales", choices=["probe", "prod", "both"], default="both")
    ap.add_argument("--batch-sizes", default="16,32,64,128,256")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--positions-dir", default=str(DEFAULT_REPLAY_DIR))
    ap.add_argument("--n-positions", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="reports/probes/gnn_integration")
    ap.add_argument("--checkpoint", default="checkpoints/probes/gnn_bc/gnn_bc_040000.pt")
    ap.add_argument("--rust-repo", default="/home/timmy/Work/Hexo/hexo-strix")
    ap.add_argument("--skip-rust", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--positions-json", default=None,
                     help="reuse an existing positions JSON (from a prior run) instead of resampling; "
                          "dense-encode timing is skipped in this mode (no live GameState).")
    ap.add_argument("--cells", default="builder,torch,ort,cnn,rust",
                    help="comma-separated cell groups to run (crash-recovery reruns): "
                         "builder,torch,ort,cnn,rust")
    ap.add_argument("--ort-gpu-mem-gb", type=float, default=3.0,
                    help="ORT CUDA-EP BFC arena cap in GiB (raise for big block-diagonal batches)")
    ap.add_argument("--gpu-util-duration", type=float, default=1.5)
    ap.add_argument("--rust-reps", type=int, default=10)
    ap.add_argument("--builder-reps", type=int, default=10, help="n reps for the Python builder timing cell")
    args = ap.parse_args()
    args.batch_sizes = sorted({int(x) for x in args.batch_sizes.split(",") if x.strip()})
    args.scales_list = ["probe", "prod"] if args.scales == "both" else [args.scales]
    args.cells_list = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = set(args.cells_list) - {"builder", "torch", "ort", "cnn", "rust"}
    if unknown:
        ap.error(f"unknown --cells groups: {sorted(unknown)}")
    return args


def flush_results(results: dict, out_dir: Path) -> None:
    """Persist partial results after every cell group — a crash must never
    lose completed cells (first shakeout run lost ~1 h of torch cells)."""
    (out_dir / "gnn_infer_bench_results.json").write_text(
        json.dumps(results, indent=1, default=str)
    )


def main() -> int:
    global ORT_GPU_MEM_GB
    args = parse_args()
    ORT_GPU_MEM_GB = args.ort_gpu_mem_gb
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "meta": {
            "hardware": "laptop RTX 4060 Max-Q Ada Lovelace (sm_89) — NOT the authoritative 5080 verdict "
                        "run; script is runnable unchanged on vast",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "ort_providers": ort.get_available_providers(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": {k: v for k, v in vars(args).items()},
        },
        "cells": [],
        "notes": [],
    }

    t_start = time.time()

    # 1. Positions.
    if args.positions_json:
        pj = json.loads(Path(args.positions_json).read_text())
        position_records = pj["positions"]
        results["positions_provenance"] = pj.get("provenance", {})
        selected = [
            {
                "source_file": r["source_file"], "game_idx": r["game_idx"], "ply": r["ply"],
                "stones": {(s[0], s[1]): s[2] for s in r["stones"]},
                "current_player": r["current_player"], "moves_remaining": r["moves_remaining"],
                "state": None,
            }
            for r in position_records
        ]
        print(f"[positions] reused {len(selected)} positions from {args.positions_json}", flush=True)
    else:
        print(f"[positions] sampling >= {args.n_positions} real positions from {args.positions_dir} ...", flush=True)
        selected, provenance = sample_positions(args.n_positions, Path(args.positions_dir), args.seed)
        results["positions_provenance"] = provenance
        position_records = positions_to_json_records(selected)
        positions_json_path = out_dir / "wpa_positions.json"
        positions_json_path.write_text(json.dumps({
            "provenance": provenance, "positions": position_records,
        }, indent=1))
        print(f"[positions] sampled {len(selected)} positions, wrote {positions_json_path}", flush=True)

    positions_json_path = out_dir / "wpa_positions.json"
    if not positions_json_path.exists() and args.positions_json:
        positions_json_path = Path(args.positions_json)

    # 2. Graph build + histogram + Python builder timing.
    # (graph build always runs — every downstream cell needs the graphs;
    #  only the TIMING is gated by --cells.)
    print("[builder] timing build_axis_graph_raw ...", flush=True)
    graphs, build_info = build_graphs_and_time(selected, n_runs=args.builder_reps, warmup=2)
    if "builder" in args.cells_list:
        results["python_build_timing"] = build_info["python_build_timing"]
        results["graph_size_histogram"] = build_info["graph_size_histogram"]

    # 3. Dense encode timing (Python-reachable seam).
    if "builder" in args.cells_list:
        print("[dense-encode] timing GameState.to_tensor() ...", flush=True)
        results["dense_encode_timing"] = time_dense_encode(selected, n_runs=args.builder_reps, warmup=2)
    flush_results(results, out_dir)

    # Free live GameState refs (large arrays) — no longer needed.
    for c in selected:
        c["state"] = None

    n_graphs = len(graphs)
    idx_by_bs = build_batch_index_sets(n_graphs, args.batch_sizes, args.seed)

    # 4. ONNX export + parity.
    checkpoint_path = Path(args.checkpoint)
    need_onnx = not args.skip_onnx and "ort" in args.cells_list
    if need_onnx:
        print("[onnx] exporting + parity-checking ...", flush=True)
        onnx_paths, export_results = do_onnx_export_and_parity(
            args.batch_sizes, args.scales_list, out_dir, graphs,
            lambda scale, device: build_gnn_net(scale, device, checkpoint_path),
        )
        results["onnx_export"] = export_results
        flush_results(results, out_dir)
    else:
        onnx_paths = {s: None for s in args.scales_list}
        results["onnx_export"] = {s: {"status": "SKIPPED", "reason": "--skip-onnx or --cells"} for s in args.scales_list}

    # 5. GNN torch cells.
    if "torch" in args.cells_list:
        print("[gnn-torch] running cells ...", flush=True)
        gnn_torch_cells(args, results, graphs, idx_by_bs, checkpoint_path)
        flush_results(results, out_dir)

    # 6. GNN ORT cells.
    if need_onnx:
        print("[gnn-ort] running cells ...", flush=True)
        ort_cells(args, results, graphs, idx_by_bs, onnx_paths)
        flush_results(results, out_dir)

    # 7. CNN comparator cells.
    if "cnn" in args.cells_list:
        print("[cnn] running cells ...", flush=True)
        cnn_cells(args, results)
        flush_results(results, out_dir)

    # 8. Rust builder proxy.
    if args.skip_rust or "rust" not in args.cells_list:
        results["rust_builder_proxy"] = {"status": "SKIPPED", "reason": "--skip-rust or --cells"}
    else:
        print("[rust] building + running native builder proxy ...", flush=True)
        results["rust_builder_proxy"] = run_rust_builder_proxy(
            position_records, results.get("positions_provenance", {}), Path(args.rust_repo),
            args.rust_reps, out_dir,
        )

    results["meta"]["wall_clock_s"] = time.time() - t_start

    # 9. Write outputs.
    json_path = out_dir / "gnn_infer_bench_results.json"
    json_path.write_text(json.dumps(results, indent=1, default=str))
    md_path = out_dir / "gnn_infer_bench_results.md"
    md_path.write_text(render_markdown(results))

    print(f"\n[done] wrote {json_path}", flush=True)
    print(f"[done] wrote {md_path}", flush=True)
    print(f"[done] wall clock: {results['meta']['wall_clock_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
