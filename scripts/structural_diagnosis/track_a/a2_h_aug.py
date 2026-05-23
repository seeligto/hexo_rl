"""§S181 Track A — A2 H-AUG: augmentation symmetry asymmetry.

Tests whether 12-fold hex-symmetry augmentation produces:
  - FEWER unique augmented variants for colony positions (rotationally
    near-invariant blobs)
  - LOWER feature-space variance through `bootstrap_model_v6.pt` trunk
    for colony positions

Mechanism: a position with high rotational symmetry maps to itself (or
near-itself) under some of the 12 symmetries → only a few unique
canonical variants → augmentation provides less gradient diversity →
fewer effective training samples per epoch. If colony positions show
strong asymmetry vs extension, augmentation oversamples them
effectively (each unique colony variant gets >1× the gradient pressure
of an extension variant).

Verdict (pre-registered, LITERAL L13):
  ASYMMETRIC-CONFIRMED  colony_uniq ≤ 0.6 × ext_uniq
                        AND colony_feat_var ≤ 0.6 × ext_feat_var
  NEUTRAL               both ratios in [0.85, 1.15]
  INCONCLUSIVE          otherwise

Outputs:
  audit/structural/track_a/A2_h_aug_symmetry.json
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from engine import Board
from hexo_rl.augment.luts import get_policy_scatters
from hexo_rl.viewer.model_loader import load_model
from hexo_rl.encoding import lookup as lookup_encoding

BANK = REPO / "tests" / "fixtures" / "value_spread_bank.json"
ANCHOR = REPO / "checkpoints" / "bootstrap_model_v6.pt"
BANK_SHA = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"


def bank_sha(positions: list[dict]) -> str:
    h = hashlib.sha256()
    for spec in positions:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        for q, r in spec["moves"]:
            h.update(f"{int(q)},{int(r)};".encode())
    return h.hexdigest()


def realize_state_v6(spec: dict) -> np.ndarray | None:
    """Build Board, get 18-plane tensor, slice to 8 v6 planes."""
    b = Board()
    for q, r in spec["moves"]:
        try:
            b.apply_move(int(q), int(r))
        except Exception:
            return None
    full = b.to_tensor().reshape(18, 19, 19)
    v6_spec = lookup_encoding("v6")
    state = full[v6_spec.kept_plane_indices].copy()  # (8, 19, 19)
    return state


def apply_sym(state: np.ndarray, scatter: np.ndarray, n_cells: int) -> np.ndarray:
    """Apply one symmetry to (n_planes, H, W) state via scatter."""
    n_planes = state.shape[0]
    flat = state.reshape(n_planes, -1)  # (n_planes, n_cells)
    new_flat = flat[:, scatter[:n_cells]]
    return new_flat.reshape(state.shape)


def main():
    t0 = time.time()
    print(f"loading bank {BANK} ...")
    data = json.loads(BANK.read_text())
    positions = data["positions"]
    sha = bank_sha(positions)
    if sha != BANK_SHA:
        raise SystemExit(f"bank SHA {sha} != pinned {BANK_SHA} — STOP")
    print(f"  bank SHA verified: {sha[:16]}…")

    print(f"loading model {ANCHOR} ...")
    device = torch.device("cpu")
    net, meta, _ = load_model(ANCHOR, device=device)
    net.eval()
    print(f"  model loaded; step={meta.get('step')}")

    scatters = get_policy_scatters(board_size=19, has_pass=True)
    print(f"  {len(scatters)} symmetry scatters")

    n_cells = 19 * 19
    per_pos: list[dict[str, Any]] = []
    for spec in positions:
        state = realize_state_v6(spec)
        if state is None:
            print(f"  SKIP illegal: {spec['name']}")
            continue
        # Apply each of 12 symmetries; canonicalize via state-byte hash
        aug_states: list[np.ndarray] = []
        hashes: set[bytes] = set()
        for sc in scatters:
            aug = apply_sym(state, sc, n_cells)
            aug_states.append(aug)
            hashes.add(aug.tobytes())
        unique_variants = len(hashes)

        # Forward all 12 variants through model.trunk for feature variance
        batch = np.stack(aug_states)  # (12, 8, 19, 19)
        x = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            trunk_out = net.trunk(x, mask=None, mask_sum_hw=None)  # (12, 128, 19, 19)
        feat = trunk_out.cpu().numpy()
        # Per-cell variance across 12 augmentations, mean over channels+cells
        cell_var = feat.var(axis=0, ddof=0)  # (128, 19, 19)
        mean_feat_var = float(cell_var.mean())
        # Also: per-cell mean abs deviation across 12 augs, mean
        # (alternative scalar — more robust to outliers)
        feat_std = float(np.sqrt(cell_var).mean())

        per_pos.append(dict(
            name=spec["name"],
            pos_class=spec["pos_class"],
            unique_variants=unique_variants,
            mean_feat_var=round(mean_feat_var, 6),
            mean_feat_std=round(feat_std, 6),
        ))

    # Aggregate by class
    def agg(filt):
        sub = [p for p in per_pos if filt(p["pos_class"])]
        if not sub:
            return None
        return dict(
            n=len(sub),
            mean_unique_variants=round(float(np.mean([p["unique_variants"] for p in sub])), 3),
            std_unique_variants=round(float(np.std([p["unique_variants"] for p in sub])), 3),
            mean_feat_var=round(float(np.mean([p["mean_feat_var"] for p in sub])), 6),
            mean_feat_std=round(float(np.mean([p["mean_feat_std"] for p in sub])), 6),
        )

    by_class = dict(
        colony=agg(lambda c: c == "colony"),
        extension=agg(lambda c: "extension" in c),
    )

    # Ratios
    col = by_class["colony"]
    ext = by_class["extension"]
    if col and ext and ext["mean_unique_variants"] > 0 and ext["mean_feat_var"] > 0:
        ratio_uniq = round(col["mean_unique_variants"] / ext["mean_unique_variants"], 4)
        ratio_var = round(col["mean_feat_var"] / ext["mean_feat_var"], 4)
        ratio_std = round(col["mean_feat_std"] / ext["mean_feat_std"], 4)
    else:
        ratio_uniq = ratio_var = ratio_std = None

    # Pre-registered verdict (LITERAL L13)
    if ratio_uniq is not None and ratio_var is not None:
        if ratio_uniq <= 0.6 and ratio_var <= 0.6:
            verdict = "ASYMMETRIC-CONFIRMED"
        elif 0.85 <= ratio_uniq <= 1.15 and 0.85 <= ratio_var <= 1.15:
            verdict = "NEUTRAL"
        else:
            verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"

    result = dict(
        meta=dict(
            bank_sha256=sha,
            anchor=str(ANCHOR.name),
            anchor_step=meta.get("step"),
            n_positions=len(per_pos),
            wall_s=round(time.time() - t0, 1),
        ),
        per_position=per_pos,
        by_class=by_class,
        ratios=dict(
            colony_over_extension_unique=ratio_uniq,
            colony_over_extension_feat_var=ratio_var,
            colony_over_extension_feat_std=ratio_std,
        ),
        verdict=verdict,
    )

    out = REPO / "audit" / "structural" / "track_a" / "A2_h_aug_symmetry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}  ({result['meta']['wall_s']}s)")
    print(json.dumps({
        "by_class": by_class,
        "ratios": result["ratios"],
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
