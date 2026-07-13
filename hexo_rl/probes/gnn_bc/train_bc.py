"""BC trainer for the GNN-BC probe — both arms, matched protocol (D-L WP3).

Supervised policy cross-entropy on the played move, v7 human corpus, §114
window + Elo-band weights. One CLI, two arms (``--arm gnn`` / ``--arm cnn``),
run once per LR (§3.3 two-LR fairness sweep). Optimizer/schedule mirror the
corpus pretrain path (AdamW + CosineAnnealingLR). Value head is present in both
nets but NOT supervised — this is a POLICY probe.

GNN arm: axis-graphs (fidelity-gated builder, REUSED) batched by disjoint union
(block-diagonal node concat, edge_index offset); per-node policy CE over the
played move's legal node. CNN arm: v6_live2_ls per-cluster-row scatter (OUR
encoding), dense 362 softmax CE. Both consume the identical BcPosition stream +
weights (``bc_data.iter_corpus_positions``).

Usage (operator, on a 5080 — see docs/handoffs/gnn_bc_probe_runbook.md):
    .venv/bin/python -m hexo_rl.probes.gnn_bc.train_bc \
        --arm gnn --lr 1e-3 --steps 40000 --batch-size 256 \
        --out checkpoints/probes/gnn_bc/gnn_lr1e-3

    # tiny CPU smoke (build-verify only; NOT the real run):
    .venv/bin/python -m hexo_rl.probes.gnn_bc.train_bc \
        --arm gnn --lr 1e-3 --steps 3 --batch-size 8 --smoke --out /tmp/gnn_smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque, namedtuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, List

import numpy as np
import torch
import torch.nn.functional as F

from hexo_rl.corpus.sources.human_game_source import HumanGameSource
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.env.game_state import GameState
from engine import Board
from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.probes.gnn_bc.bc_data import iter_corpus_positions, BcPosition
from hexo_rl.probes.gnn_bc.gnn_bc_net import GnnBcNet
from hexo_rl.probes.gnn_bc.cnn_bc_net import build_cnn_bc_net

LABEL_SMOOTH = 0.05           # corpus default
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5

_STRIX_WIN_LENGTH = 6
_STRIX_RADIUS = 6


# ── GNN example building ──────────────────────────────────────────────────────

def _gnn_example(pos: BcPosition):
    """Build a single axis-graph example. Returns None if the played move is not
    a representable legal node (out of radius of every stone — rare)."""
    g = build_axis_graph_raw(
        pos.stones, pos.current_player, pos.moves_remaining,
        win_length=_STRIX_WIN_LENGTH, radius=_STRIX_RADIUS,
        prune_empty_edges=True, threat_features=True, relative_stones=True,
    )
    legal = g["legal_coords"]
    try:
        target_local = legal.index(pos.move)  # index within THIS graph's legal nodes
    except ValueError:
        return None
    return g, target_local, pos.weight


# Compact per-example representation (the MEMORY fix): numpy arrays instead of
# the Python-list dict from build_axis_graph_raw (~100KB/example → ~27KB). Small
# dtypes (float32/int32/bool). Picklable (namedtuple of np arrays) → workers ship
# it fast (buffer protocol, not slow list-pickle). ``n_legal`` precomputed so the
# hot collate never recomputes it.
CompactGnnExample = namedtuple(
    "CompactGnnExample", "x edge eattr lmask target_local weight n_legal")


def _compact_from_graph(g, target_local: int, weight) -> "CompactGnnExample":
    """Pack a build_axis_graph_raw dict into a CompactGnnExample (numpy)."""
    n, fdim, E = g["num_nodes"], g["fdim"], len(g["edge_src"])
    x = np.asarray(g["features"], dtype=np.float32).reshape(n, fdim)
    if E:
        edge = np.asarray([g["edge_src"], g["edge_dst"]], dtype=np.int32)      # (2, E)
        eattr = np.asarray(g["edge_attr"], dtype=np.float32).reshape(E, 5)
    else:
        edge = np.zeros((2, 0), dtype=np.int32)
        eattr = np.zeros((0, 5), dtype=np.float32)
    lmask = np.asarray(g["legal_mask"], dtype=np.bool_)
    return CompactGnnExample(x, edge, eattr, lmask, int(target_local),
                             np.float32(weight), int(lmask.sum()))


def _compact_example(pos: BcPosition):
    """Build ONE axis-graph example as compact numpy. None if the played move is
    not a representable legal node (out of radius — rare)."""
    g = build_axis_graph_raw(
        pos.stones, pos.current_player, pos.moves_remaining,
        win_length=_STRIX_WIN_LENGTH, radius=_STRIX_RADIUS,
        prune_empty_edges=True, threat_features=True, relative_stones=True,
    )
    try:
        target_local = g["legal_coords"].index(pos.move)
    except ValueError:
        return None
    return _compact_from_graph(g, target_local, pos.weight)


def _collate_gnn_dict(examples, device):
    """REFERENCE disjoint-union collate from the raw build_axis_graph_raw dicts.
    Kept as the equivalence oracle for the fast compact ``_collate_gnn`` below.
    Returns (x, edge_index, edge_attr, legal_mask, target_idx, legal_offsets, w)."""
    xs, eis, eas, lms = [], [], [], []
    legal_offsets = []          # start index of each graph's legal block (in concat legal order)
    per_graph_targets = []
    weights = []
    node_offset = 0
    legal_count = 0
    for (g, target_local, w) in examples:
        n = g["num_nodes"]
        fdim = g["fdim"]
        x = torch.tensor(g["features"], dtype=torch.float32).reshape(n, fdim)
        E = len(g["edge_src"])
        if E:
            ei = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.int64) + node_offset
            ea = torch.tensor(g["edge_attr"], dtype=torch.float32).reshape(E, 5)
        else:
            ei = torch.zeros((2, 0), dtype=torch.int64)
            ea = torch.zeros((0, 5), dtype=torch.float32)
        lm = torch.tensor(g["legal_mask"], dtype=torch.bool)
        n_legal = int(lm.sum().item())
        xs.append(x); eis.append(ei); eas.append(ea); lms.append(lm)
        legal_offsets.append(legal_count)
        per_graph_targets.append(legal_count + target_local)
        weights.append(w)
        node_offset += n
        legal_count += n_legal
    x = torch.cat(xs, 0).to(device)
    edge_index = torch.cat(eis, 1).to(device)
    edge_attr = torch.cat(eas, 0).to(device)
    legal_mask = torch.cat(lms, 0).to(device)
    target_idx = torch.tensor(per_graph_targets, dtype=torch.int64, device=device)
    legal_offsets = torch.tensor(legal_offsets + [legal_count], dtype=torch.int64, device=device)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return x, edge_index, edge_attr, legal_mask, target_idx, legal_offsets, w


def _collate_gnn(examples, device):
    """FAST disjoint-union collate over CompactGnnExample (numpy). One
    ``np.concatenate`` + ``torch.from_numpy`` per field (no per-graph
    ``torch.tensor(list)``). Tensors are IDENTICAL to ``_collate_gnn_dict``."""
    xs, edges, eas, lms = [], [], [], []
    legal_offsets = []          # start index of each graph's legal block
    per_graph_targets = []
    weights = []
    node_offset = 0
    legal_count = 0
    for e in examples:
        xs.append(e.x)
        # offset this graph's LOCAL edge indices into the concatenated node space
        edges.append(e.edge + np.int32(node_offset) if e.edge.shape[1] else e.edge)
        eas.append(e.eattr)
        lms.append(e.lmask)
        legal_offsets.append(legal_count)
        per_graph_targets.append(legal_count + e.target_local)
        weights.append(e.weight)
        node_offset += e.x.shape[0]
        legal_count += e.n_legal
    x = torch.from_numpy(np.concatenate(xs, axis=0)).to(device)
    edge_index = torch.from_numpy(
        np.concatenate(edges, axis=1).astype(np.int64, copy=False)).to(device)
    edge_attr = torch.from_numpy(np.concatenate(eas, axis=0)).to(device)
    legal_mask = torch.from_numpy(np.concatenate(lms, axis=0)).to(device)
    target_idx = torch.tensor(per_graph_targets, dtype=torch.int64, device=device)
    legal_offsets = torch.tensor(legal_offsets + [legal_count], dtype=torch.int64, device=device)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return x, edge_index, edge_attr, legal_mask, target_idx, legal_offsets, w


def _gnn_loss_reference(net, batch):
    """Per-graph cross-entropy over each graph's legal-node logits (segmented
    softmax via legal_offsets), weighted by the Elo-band weight.

    REFERENCE implementation — the frozen spec. Kept as the equivalence oracle
    for ``test_vectorized_gnn_loss_matches_reference``; the training path uses the
    vectorized ``_gnn_loss`` below (256 per-graph ``.item()`` GPU syncs/step here
    made this the ~24h bottleneck — see the perf note in the WP3 runbook)."""
    x, edge_index, edge_attr, legal_mask, target_idx, legal_offsets, w = batch
    logits = net.forward_batch(x, edge_index, edge_attr, legal_mask)  # (num_legal_total,)
    n_graphs = legal_offsets.numel() - 1
    losses = []
    correct = 0
    for gi in range(n_graphs):
        lo = int(legal_offsets[gi].item())
        hi = int(legal_offsets[gi + 1].item())
        seg = logits[lo:hi]
        tgt = int(target_idx[gi].item()) - lo
        logp = F.log_softmax(seg, dim=0)
        # label smoothing over this graph's legal set
        n = seg.numel()
        smooth = LABEL_SMOOTH / n
        loss_g = -((1 - LABEL_SMOOTH) * logp[tgt] + smooth * logp.sum())
        losses.append(w[gi] * loss_g)
        if int(seg.argmax().item()) == tgt:
            correct += 1
    loss = torch.stack(losses).sum() / w.sum()
    return loss, correct, n_graphs


def _gnn_loss(net, batch):
    """Vectorized equivalent of ``_gnn_loss_reference`` — segmented log-softmax
    over ``legal_offsets`` with NO Python per-graph loop and NO per-graph
    ``.item()`` GPU sync. Math-identical (up to float reduction order)."""
    x, edge_index, edge_attr, legal_mask, target_idx, legal_offsets, w = batch
    logits = net.forward_batch(x, edge_index, edge_attr, legal_mask)  # (num_legal_total,)
    n_graphs = legal_offsets.numel() - 1
    dev, dt = logits.device, logits.dtype
    counts = (legal_offsets[1:] - legal_offsets[:-1])                 # (n_graphs,) legal nodes/graph
    seg_id = torch.repeat_interleave(torch.arange(n_graphs, device=dev), counts)  # (num_legal_total,)
    # per-segment max (detached stabilization constant, exactly as F.log_softmax does)
    seg_max = torch.full((n_graphs,), float("-inf"), device=dev, dtype=dt)
    seg_max = seg_max.scatter_reduce(0, seg_id, logits, reduce="amax", include_self=True).detach()
    seg_sumexp = torch.zeros(n_graphs, device=dev, dtype=dt).index_add(
        0, seg_id, (logits - seg_max[seg_id]).exp())
    seg_logz = seg_max + seg_sumexp.log()                            # (n_graphs,)
    logp = logits - seg_logz[seg_id]                                 # per-segment log-softmax, all nodes
    logp_tgt = logp[target_idx]                                      # (n_graphs,) target is a GLOBAL index
    seg_sum_logp = torch.zeros(n_graphs, device=dev, dtype=dt).index_add(0, seg_id, logp)
    smooth = LABEL_SMOOTH / counts.to(dt)                            # (n_graphs,) label smoothing over legal set
    loss_g = -((1 - LABEL_SMOOTH) * logp_tgt + smooth * seg_sum_logp)
    loss = (w * loss_g).sum() / w.sum()
    # top1 (diagnostic only, not in the gradient): target achieves its segment max
    correct = int((logits[target_idx] == seg_max).sum().item())
    return loss, correct, n_graphs


def parallel_ordered_map(func: Callable, iterable: Iterable, workers: int,
                         max_inflight: int) -> Iterator:
    """Apply ``func`` across ``workers`` processes, yielding results in INPUT
    order, with at most ``max_inflight`` tasks in flight (bounded memory over an
    unbounded input). Order-preserving so the built-example stream is identical
    to the serial path."""
    it = iter(iterable)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()
        exhausted = False
        while len(pending) < max_inflight and not exhausted:
            try:
                pending.append(pool.submit(func, next(it)))
            except StopIteration:
                exhausted = True
        while pending:
            result = pending.popleft().result()   # FIFO → yields in submission (input) order
            if not exhausted:
                try:
                    pending.append(pool.submit(func, next(it)))
                except StopIteration:
                    exhausted = True
            yield result


# ── CNN example building (v6_live2_ls per-cluster-row scatter) ─────────────────

def iter_cnn_examples(game_records, spec) -> Iterator:
    """Stream v6_live2_ls per-cluster-row examples at GAME granularity (so the
    board is replayed once per game, matching dataset.replay_game_to_triples_ls)."""
    from hexo_rl.probes.gnn_bc.bc_data import (
        MIN_GAME_LENGTH, POSITION_START, POSITION_END, elo_band_weight,
    )
    kept = list(spec.kept_plane_indices)
    k_max = int(getattr(spec, "k_max", 8) or 8)
    for rec in game_records:
        if len(rec.moves) < MIN_GAME_LENGTH:
            continue
        w = elo_band_weight((rec.metadata or {}).get("elo_p1"),
                            (rec.metadata or {}).get("elo_p2"))
        board = Board()
        state = GameState.from_board(board)
        for ply, (q, r) in enumerate(rec.moves):
            if POSITION_START <= ply < POSITION_END:
                tensor, centers = state.to_tensor()  # (K, 18, S, S)
                _, _, H, W = tensor.shape
                half = (H - 1) // 2
                for k, (cq, cr) in enumerate(centers[:k_max]):
                    wq = q - cq + half
                    wr = r - cr + half
                    if 0 <= wq < H and 0 <= wr < W:
                        planes = tensor[k][kept].astype(np.float32)
                        yield planes, wq * W + wr, w
            try:
                state = state.apply_move(board, q, r)
            except Exception:
                break


def _collate_cnn(examples, device):
    planes = np.stack([e[0] for e in examples])
    x = torch.from_numpy(planes).to(device)
    tgt = torch.tensor([e[1] for e in examples], dtype=torch.int64, device=device)
    w = torch.tensor([e[2] for e in examples], dtype=torch.float32, device=device)
    return x, tgt, w


def _cnn_loss(net, batch):
    x, tgt, w = batch
    out = net(x)              # (log_policy, value, v_logit)
    log_policy = out[0]       # (B, 362) log-softmax
    n_actions = log_policy.shape[1]
    logp_t = log_policy.gather(1, tgt.unsqueeze(1)).squeeze(1)   # (B,)
    smooth = LABEL_SMOOTH / n_actions
    per = -((1 - LABEL_SMOOTH) * logp_t + smooth * log_policy.sum(dim=1))
    loss = (w * per).sum() / w.sum()
    correct = int((log_policy.argmax(dim=1) == tgt).sum().item())
    return loss, correct, x.shape[0]


# ── batching over an example stream ───────────────────────────────────────────

def _batched(stream, batch_size):
    buf = []
    for item in stream:
        if item is None:
            continue
        buf.append(item)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def train(arm: str, lr: float, steps: int, batch_size: int, out: Path,
          raw_dir: str, device: str, smoke: bool, seed: int = 42, workers: int = 8):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    out.mkdir(parents=True, exist_ok=True)

    if arm == "gnn":
        net = GnnBcNet().to(dev)
    elif arm == "cnn":
        net = build_cnn_bc_net().to(dev)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    n_params = sum(p.numel() for p in net.parameters())
    par = workers if (arm == "gnn" and workers > 0 and not smoke) else 0
    print(f"[bc] arm={arm} lr={lr} steps={steps} bs={batch_size} params={n_params:,} "
          f"device={device} build_workers={par}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=ETA_MIN)

    spec = _lookup_encoding("v6_live2_ls")

    # ── GNN dataset MATERIALIZATION (the perf fix) ────────────────────────────
    # build_axis_graph_raw is ~95% of GNN main-thread cost (cProfile), and a
    # 40k-step run re-walks the ~500k-position corpus ~20× — rebuilding every
    # graph every epoch. Instead build each example ONCE as COMPACT numpy
    # (_compact_example — float32/int32, ~72KB vs the raw dict's ~250KB+ of
    # Python-list objects; the dict path OOM'd a single 60G box). ~500k examples
    # ≈ ~36GB → fits ONE dataset at a time; run the two arms SEQUENTIALLY (two
    # concurrent = ~72GB = OOM). Hold the list, re-iterate per epoch → steady
    # state is fast numpy collate + GPU only. Parallel build ships numpy via the
    # buffer protocol (cheap pickle), scaling far better than the dict path.
    gnn_dataset: List = []
    if arm == "gnn" and not smoke:
        t_build = time.time()
        src = HumanGameSource(raw_dir)
        if par > 0:
            built = parallel_ordered_map(_compact_example, iter_corpus_positions(src),
                                         par, max_inflight=max(par * 64, batch_size * 4))
        else:
            built = (_compact_example(p) for p in iter_corpus_positions(src))
        gnn_dataset = [e for e in built if e is not None]
        print(f"[bc] materialized {len(gnn_dataset):,} gnn examples in "
              f"{time.time() - t_build:.1f}s (workers={par})")

    def example_stream():
        if arm == "gnn":
            if smoke:
                # single pass, serial — a 3-step build-verify never materializes
                for pos in iter_corpus_positions(HumanGameSource(raw_dir)):
                    yield _compact_example(pos)   # None dropped by _batched
            else:
                while True:                   # re-iterate the materialized dataset
                    yield from gnn_dataset
        else:
            while True:
                source = HumanGameSource(raw_dir)
                yield from iter_cnn_examples(source, spec)
                if smoke:
                    return

    net.train()
    step = 0
    t0 = time.time()
    log_rows = []
    for buf in _batched(example_stream(), batch_size):
        if arm == "gnn":
            batch = _collate_gnn(buf, dev)
            loss, correct, n = _gnn_loss(net, batch)
        else:
            batch = _collate_cnn(buf, dev)
            loss, correct, n = _cnn_loss(net, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % 200 == 0 or smoke:
            acc = correct / max(n, 1)
            lr_now = opt.param_groups[0]["lr"]
            row = {"step": step, "loss": float(loss.item()), "top1": acc, "lr": lr_now,
                   "elapsed_s": round(time.time() - t0, 1)}
            log_rows.append(row)
            print(f"[bc] step={step} loss={row['loss']:.4f} top1={acc:.3f} lr={lr_now:.2e}")
        if step >= steps:
            break

    ckpt = out / f"{arm}_bc_{step:06d}.pt"
    torch.save({
        "arm": arm, "lr": lr, "steps": step, "n_params": n_params,
        "model_state_dict": net.state_dict(),
        "encoding": "v6_live2_ls" if arm == "cnn" else "strix_axis_graph",
    }, ckpt)
    (out / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in log_rows))
    print(f"[bc] saved {ckpt}")
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["gnn", "cnn"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--raw-dir", default="data/corpus/raw_human")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=8,
                    help="GNN graph-build worker processes (order-preserving; 0=serial). "
                         "No effect on the CNN arm or --smoke.")
    ap.add_argument("--smoke", action="store_true", help="single-pass build-verify (NOT the real run)")
    args = ap.parse_args()
    train(args.arm, args.lr, args.steps, args.batch_size, Path(args.out),
          args.raw_dir, args.device, args.smoke, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
