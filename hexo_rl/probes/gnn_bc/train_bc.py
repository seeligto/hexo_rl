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
from pathlib import Path
from typing import Iterator, List

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


def _collate_gnn(examples, device):
    """Disjoint-union batch of axis-graphs. Node/edge tensors concatenated with
    per-graph offsets; legal-node targets remapped into the concatenated legal
    index space. Returns (x, edge_index, edge_attr, legal_mask, target_idx, w)."""
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


def _gnn_loss(net, batch):
    """Per-graph cross-entropy over each graph's legal-node logits (segmented
    softmax via legal_offsets), weighted by the Elo-band weight."""
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
          raw_dir: str, device: str, smoke: bool, seed: int = 42):
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
    print(f"[bc] arm={arm} lr={lr} steps={steps} bs={batch_size} params={n_params:,} device={device}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=ETA_MIN)

    spec = _lookup_encoding("v6_live2_ls")

    def example_stream():
        # re-instantiate the source each epoch pass (streaming over the corpus)
        while True:
            source = HumanGameSource(raw_dir)
            if arm == "gnn":
                for pos in iter_corpus_positions(source):
                    ex = _gnn_example(pos)
                    if ex is not None:
                        yield ex
            else:
                yield from iter_cnn_examples(source, spec)
            if smoke:
                return  # one pass only in smoke

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
    ap.add_argument("--smoke", action="store_true", help="single-pass build-verify (NOT the real run)")
    args = ap.parse_args()
    train(args.arm, args.lr, args.steps, args.batch_size, Path(args.out),
          args.raw_dir, args.device, args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
