"""§S181-AUDIT Wave 1 Track B — trunk feature drift analysis.

Post-run script. Walks a checkpoint ladder, forwards the alt-bank
40-position fixture through each model's TRUNK output (pre-head), and
records:

  - per-class trunk-feature centroid (mean activation across positions)
  - intra-class variance (mean squared distance to centroid)
  - across-class centroid distance (L2)

Goal (V-B-D). If trunk centroids collapse (colony+extension distance
shrinks ≥ 50%) by step 1k, trunk co-adapts to colony pattern —
representation collapse, not head-only. Requires aux heads forcing
trunk discrimination in the real run.

Usage:
    python -m scripts.structural_diagnosis.track_b.trunk_feature_drift \\
        --ckpt-dir /path/to/checkpoints \\
        --output audit/structural/track_b/B3_trunk_drift.json

The alt-bank fixture is loaded via the canary's `load_alt_bank`
(SHA-pinned). Trunk output is taken from the HexTacToeNet's
`self.trunk` submodule via a forward-pre-hook on the policy head — no
model surgery, no architecture coupling.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from hexo_rl.monitoring.value_spread_canary import load_alt_bank
from hexo_rl.viewer.model_loader import load_model


def _extract_trunk_features(
    net: torch.nn.Module,
    states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Forward `states` through the model and capture the trunk output.

    Returns features shape (B, C, H, W) as float32 on CPU.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: torch.nn.Module, args: tuple) -> None:
        # forward-pre-hook signature is (module, args); args[0] is the
        # trunk output tensor handed to the policy head (its first input).
        if args and isinstance(args[0], torch.Tensor):
            captured["features"] = args[0].detach()

    target = None
    for name in ("policy_head", "policy_conv"):
        mod = getattr(net, name, None)
        if mod is not None:
            target = mod
            break
    if target is None:
        raise RuntimeError("could not find a policy head to hook trunk output")

    handle = target.register_forward_pre_hook(_hook)
    try:
        was_training = net.training
        net.eval()
        try:
            x = torch.from_numpy(states).to(device)
            with torch.no_grad():
                _ = net(x)
        finally:
            if was_training:
                net.train()
    finally:
        handle.remove()

    if "features" not in captured:
        raise RuntimeError("trunk feature pre-hook did not fire")
    return captured["features"].cpu().float().numpy()


def _per_class_centroid_stats(
    features: np.ndarray, classes: np.ndarray,
) -> dict[str, Any]:
    """Compute per-class centroid + intra-class spread + across-class distance.

    `features` shape (B, C, H, W) → flatten to (B, D).
    """
    B = features.shape[0]
    flat = features.reshape(B, -1)

    col = flat[classes == "colony"]
    ext = flat[classes == "extension"]
    col_centroid = col.mean(axis=0) if col.shape[0] > 0 else None
    ext_centroid = ext.mean(axis=0) if ext.shape[0] > 0 else None

    if col_centroid is None or ext_centroid is None:
        return dict(
            colony_intra_var=float("nan"),
            extension_intra_var=float("nan"),
            inter_centroid_dist=float("nan"),
            inter_intra_ratio=float("nan"),
            feature_dim=int(flat.shape[1]),
        )

    col_intra = float(np.mean(np.linalg.norm(col - col_centroid, axis=1) ** 2))
    ext_intra = float(np.mean(np.linalg.norm(ext - ext_centroid, axis=1) ** 2))
    inter = float(np.linalg.norm(col_centroid - ext_centroid))

    denom = (col_intra + ext_intra) ** 0.5
    ratio = (inter / denom) if denom > 1e-12 else float("nan")

    return dict(
        colony_intra_var=round(col_intra, 6),
        extension_intra_var=round(ext_intra, 6),
        inter_centroid_dist=round(inter, 6),
        inter_intra_ratio=round(ratio, 6),
        feature_dim=int(flat.shape[1]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, type=Path,
                    help="directory containing checkpoint_NNNNNNNN.pt files")
    ap.add_argument("--ckpt-glob", default="checkpoint_*.pt",
                    help="glob pattern within --ckpt-dir")
    ap.add_argument("--include-anchor", type=Path, default=None,
                    help="optional baseline checkpoint to forward FIRST as step 0")
    ap.add_argument("--output", type=Path,
                    default=REPO / "audit" / "structural" / "track_b"
                            / "B3_trunk_drift.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    bank = load_alt_bank()
    print(f"alt bank: {bank.states.shape[0]} positions, SHA={bank.sha[:16]}…")

    ladder: list[dict[str, Any]] = []
    t0 = time.time()

    if args.include_anchor is not None:
        print(f"forwarding anchor {args.include_anchor.name} ...")
        net, _meta, _ = load_model(args.include_anchor, device=device)
        feats = _extract_trunk_features(net, bank.states, device)
        stats = _per_class_centroid_stats(feats, bank.classes)
        ladder.append(dict(step=0, ckpt=str(args.include_anchor.name), **stats))
        print(f"  inter_centroid_dist = {stats['inter_centroid_dist']}  "
              f"inter/intra ratio = {stats['inter_intra_ratio']}")

    ckpts = sorted(args.ckpt_dir.glob(args.ckpt_glob))
    for ckpt in ckpts:
        # Step number parsed from filename `checkpoint_<step>.pt`.
        try:
            step = int(ckpt.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        print(f"forwarding {ckpt.name} (step {step}) ...")
        net, _meta, _ = load_model(ckpt, device=device)
        feats = _extract_trunk_features(net, bank.states, device)
        stats = _per_class_centroid_stats(feats, bank.classes)
        ladder.append(dict(step=step, ckpt=str(ckpt.name), **stats))
        print(f"  inter_centroid_dist = {stats['inter_centroid_dist']}  "
              f"inter/intra ratio = {stats['inter_intra_ratio']}")

    out = dict(
        meta=dict(
            wave="§S181-AUDIT Wave 1 — Track B / B3",
            ckpt_dir=str(args.ckpt_dir),
            alt_bank_sha=bank.sha,
            n_ladder=len(ladder),
            wall_s=round(time.time() - t0, 2),
        ),
        ladder=ladder,
    )
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
