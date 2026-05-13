#!/usr/bin/env python3
"""§174 bootstrap probe — G4 + value-output dist + first-conv L2 + policy entropy.

Mirrors the Track-1 T3 probe so transfer + e50 + 30ep are directly comparable.

Usage:
    python scripts/s174_probe_model.py \
        --checkpoint checkpoints/bootstrap_model_v6w25_transfer_ft.pt \
        --out reports/s174/probe_ft.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hexo_rl.utils.config import load_config
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.encoding import resolve_from_config as _rfc, resolve_corpus_path, lookup


def _build_model(state: dict, encoding: str) -> torch.nn.Module:
    config = load_config("configs/model.yaml", "configs/training.yaml")
    config["encoding"] = encoding
    for k in ("board_size", "in_channels", "n_planes",
              "cluster_window_size", "cluster_threshold", "legal_move_radius"):
        config.pop(k, None)
    spec = _rfc(config)
    model = HexTacToeNet(
        board_size=spec.trunk_size,
        in_channels=int(spec.n_planes),
        filters=int(config["filters"]),
        res_blocks=int(config["res_blocks"]),
        se_reduction_ratio=int(config.get("se_reduction_ratio", 4)),
        encoding=encoding,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--encoding", default="v6w25")
    p.add_argument("--n-positions", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model = _build_model(state, args.encoding).to(device)

    # G4 — value head |max| band [0.154, 0.462]
    vmax = float(model.state_dict()["value_fc2.weight"].abs().max().item())
    g4_band = (0.154, 0.462)
    g4_pass = g4_band[0] <= vmax <= g4_band[1]
    print(f"G4 value_fc2 |max| = {vmax:.6f}  band {g4_band} → {'PASS' if g4_pass else 'FAIL'}")

    # First-conv per-plane L2
    w = model.state_dict()["trunk.input_conv.weight"]  # [filters, in_ch, k, k]
    per_plane_l2 = w.float().pow(2).sum(dim=(0, 2, 3)).sqrt().cpu().numpy().tolist()
    plane_names = list(lookup(args.encoding).plane_layout)
    print("first-conv per-plane L2:")
    for i, (name, l2) in enumerate(zip(plane_names, per_plane_l2)):
        print(f"  plane {i:2d} {name:25s} L2={l2:.4f}")

    # Sample corpus positions, forward, record value + entropy
    corpus_path = resolve_corpus_path(lookup(args.encoding))
    print(f"corpus: {corpus_path}")
    npz = np.load(corpus_path, mmap_mode="r")
    states = npz["states"]
    n_total = states.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(n_total, size=min(args.n_positions, n_total), replace=False)
    sample = np.ascontiguousarray(states[idx])
    if sample.dtype != np.float32:
        sample = sample.astype(np.float32)
    print(f"sample shape: {sample.shape}")

    x = torch.from_numpy(sample).to(device)
    with torch.no_grad():
        out = model(x)

    # out is (policy_logits, value, aux_opp_reply) — figure out which is which
    policy_logits, value, opp_reply = out[0], out[1], out[2]
    if policy_logits.dim() == 2 and policy_logits.shape[1] >= 2:
        ent = (-F.log_softmax(policy_logits, dim=-1) * F.softmax(policy_logits, dim=-1)).sum(dim=-1)
    else:
        raise RuntimeError(f"unexpected policy_logits shape {policy_logits.shape}")
    ent_np = ent.float().cpu().numpy()
    val_np = value.float().squeeze(-1).cpu().numpy()

    rec = {
        "checkpoint": str(args.checkpoint),
        "encoding": args.encoding,
        "n_positions": int(sample.shape[0]),
        "g4": {
            "value_fc2_abs_max": vmax,
            "band": list(g4_band),
            "pass": g4_pass,
        },
        "first_conv_per_plane_l2": {name: float(l2) for name, l2 in zip(plane_names, per_plane_l2)},
        "policy_entropy_nats": {
            "mean": float(ent_np.mean()),
            "std":  float(ent_np.std()),
            "min":  float(ent_np.min()),
            "max":  float(ent_np.max()),
        },
        "value_output": {
            "mean": float(val_np.mean()),
            "std":  float(val_np.std()),
            "min":  float(val_np.min()),
            "max":  float(val_np.max()),
            "abs_mean": float(np.abs(val_np).mean()),
        },
    }
    print(json.dumps({k: v for k, v in rec.items() if k != "first_conv_per_plane_l2"}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
