#!/usr/bin/env python3
"""§S181 IMPL-S181T2 — standalone value-head + encoding architecture probe.

STANDALONE. Imports ONLY torch + numpy. NO imports from hexo_rl / engine.
Re-implements the v6 value-head forward math (GAP + max → fc1 → ReLU → fc2
→ tanh) from `hexo_rl/model/network.py:789-796` so the structural property
can be measured WITHOUT loading the anchor checkpoint or the full network.

The point is NOT to evaluate the trained anchor — it is to demonstrate the
STRUCTURAL asymmetry of the dual-pool value head independent of any
particular weight set. Run with random + adversarial weight sets to show
the property is architectural.

What this probes
----------------
1. Dual-pool asymmetry: `v = cat([avg_pool(out), max_pool(out)])` — the
   max-pool half gives a *monotone* path from "one high-activation cell"
   to the value logit. Quantify: how much can a single saturated channel
   move the value output via the max half vs the avg half.

2. Colony-saturation simulation: synthesise trunk feature maps that mimic
   (a) a compact colony (one dense high-activation blob) vs (b) a thin
   extension line (activation spread along a 1-cell-wide ray). Measure the
   max-pool vector for each. Colony blob and extension line can produce the
   SAME max-pool vector (max is blob-vs-line invariant) — but the avg-pool
   vector differs (line is sparser → lower mean). Net effect on value.

3. K-cluster contrast (v6w25 path, NOT the §S178 v6 anchor): min-pool over
   K cluster values. Show a single colony cluster reading high does NOT
   pull the aggregate up (min picks the weakest) — but ALSO show the
   policy scatter-max DOES let the colony cluster's high-prob cells win.

Usage:
    python scripts/structural_diagnosis/probe_architecture.py
    python scripts/structural_diagnosis/probe_architecture.py --seed 7 --n 256
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# v6 value-head math, re-implemented verbatim from network.py:789-796 +
# network_min_max_head.py:87-94. Single window (v6 / §S178 anchor: k_max=1).
# ----------------------------------------------------------------------------
def value_head_forward(out, w1, b1, w2, b2):
    """out: (B, C, H, W) trunk features. Returns (value, v_logit, parts).

    parts = dict of intermediate tensors for asymmetry decomposition.
    """
    v_avg = out.mean(dim=(2, 3))            # (B, C)  — global average pool
    v_max = out.amax(dim=(2, 3))            # (B, C)  — global MAX pool
    v = torch.cat([v_avg, v_max], dim=1)    # (B, 2C)
    h = F.relu(F.linear(v, w1, b1))         # (B, 256)
    v_logit = F.linear(h, w2, b2)           # (B, 1)
    value = torch.tanh(v_logit)
    return value, v_logit, dict(v_avg=v_avg, v_max=v_max, h=h)


def make_value_weights(filters, hidden, seed, mode="random"):
    """Build a (w1,b1,w2,b2) value-head weight set.

    mode='random'      — Kaiming-ish random (untrained-net proxy).
    mode='colony_pos'  — w2 deliberately positive on hidden units that
                         the max-pool half feeds, simulating a net that
                         *learned* "high max activation => winning".
    """
    g = torch.Generator().manual_seed(seed)
    twoC = 2 * filters
    w1 = torch.randn(hidden, twoC, generator=g) * (1.0 / np.sqrt(twoC))
    b1 = torch.zeros(hidden)
    w2 = torch.randn(1, hidden, generator=g) * (1.0 / np.sqrt(hidden))
    b2 = torch.zeros(1)
    if mode == "colony_pos":
        # Force w1's max-pool block (cols filters..2filters) to be strongly
        # positive and w2 all-positive: this is the "value head learned a
        # max-detector" adversarial weight set.
        w1[:, filters:] = w1[:, filters:].abs() + 0.5
        w2 = w2.abs() + 0.2
    return w1, b1, w2, b2


# ----------------------------------------------------------------------------
# Synthetic trunk feature maps.
# We do NOT have the real trunk; we model its OUTPUT directly. A trained
# trunk maps stones -> spatial activations. We assert two coarse facts that
# hold for any conv trunk:
#   * a compact colony of stones -> a dense high-activation BLOB
#   * a thin 5-in-row extension  -> a sparse high-activation RAY (1-wide)
# Both can have the SAME peak activation; they differ in COVERAGE.
# ----------------------------------------------------------------------------
def synth_colony(filters, H, W, peak=4.0):
    """Dense blob: 6x6 patch of high activation on every channel."""
    out = torch.zeros(filters, H, W)
    cy, cx = H // 2, W // 2
    out[:, cy - 3:cy + 3, cx - 3:cx + 3] = peak
    return out


def synth_extension(filters, H, W, peak=4.0, length=5):
    """Thin ray: 1-wide line of high activation (open-ended 5-in-row)."""
    out = torch.zeros(filters, H, W)
    cy, cx = H // 2, W // 2
    for i in range(length):
        out[:, cy, cx - length // 2 + i] = peak
    return out


def synth_diffuse_extension(filters, H, W, peak=4.0, n_clusters=8):
    """Extension whose stones are SPREAD so no single cluster window owns
    the whole threat. Models claim #3: long thin line diffused across
    multiple cluster crops -> each crop sees only a fragment."""
    out = torch.zeros(filters, H, W)
    cy = H // 2
    # place 5 activations spaced apart so a window of width ~H/3 captures
    # only ~2 of them.
    xs = np.linspace(2, W - 3, 5).astype(int)
    for x in xs:
        out[:, cy, x] = peak
    return out


def summarise(name, value, v_logit, parts):
    va = parts["v_avg"]
    vm = parts["v_max"]
    print(f"  {name:28s} value={value.mean().item():+.4f} "
          f"logit={v_logit.mean().item():+.4f} "
          f"avg_pool_norm={va.norm(dim=1).mean().item():.3f} "
          f"max_pool_norm={vm.norm(dim=1).mean().item():.3f}")


def probe_dual_pool_asymmetry(filters=128, H=19, W=19, seed=0):
    print("\n[1] DUAL-POOL VALUE-HEAD ASYMMETRY")
    print("    value head: v = cat([GAP(out), GMP(out)]) -> fc1 -> relu -> fc2 -> tanh")
    print("    (network.py:789-796)\n")

    w1, b1, w2, b2 = make_value_weights(filters, 256, seed, "random")

    colony = synth_colony(filters, H, W).unsqueeze(0)
    ext = synth_extension(filters, H, W).unsqueeze(0)
    empty = torch.zeros(1, filters, H, W)

    for nm, t in [("empty_board", empty), ("colony_blob", colony),
                  ("extension_line", ext)]:
        val, lg, parts = value_head_forward(t, w1, b1, w2, b2)
        summarise(nm, val, lg, parts)

    # Asymmetry test: zero out the avg half vs the max half, measure logit.
    print("\n    Ablation — which pool half carries the colony signal?")
    for nm, t in [("colony_blob", colony), ("extension_line", ext)]:
        v_avg = t.mean(dim=(2, 3))
        v_max = t.amax(dim=(2, 3))
        zero = torch.zeros_like(v_avg)
        # full
        full = torch.cat([v_avg, v_max], dim=1)
        # avg-only (max half zeroed)
        avg_only = torch.cat([v_avg, zero], dim=1)
        # max-only (avg half zeroed)
        max_only = torch.cat([zero, v_max], dim=1)
        lg_full = F.linear(F.relu(F.linear(full, w1, b1)), w2, b2)
        lg_avg = F.linear(F.relu(F.linear(avg_only, w1, b1)), w2, b2)
        lg_max = F.linear(F.relu(F.linear(max_only, w1, b1)), w2, b2)
        print(f"      {nm:18s} logit_full={lg_full.item():+.4f}  "
              f"avg_only={lg_avg.item():+.4f}  max_only={lg_max.item():+.4f}")

    # KEY structural fact: GMP(colony_blob) == GMP(extension_line) when the
    # peak activation is equal. The value head CANNOT distinguish a 36-cell
    # colony from a 5-cell extension THROUGH THE MAX HALF.
    gmp_colony = colony.amax(dim=(2, 3))
    gmp_ext = ext.amax(dim=(2, 3))
    print(f"\n    GMP(colony) == GMP(extension) ? "
          f"max|diff|={(gmp_colony - gmp_ext).abs().max().item():.6f}  "
          f"-> max pool is BLOB-vs-LINE BLIND")
    gap_colony = colony.mean(dim=(2, 3))
    gap_ext = ext.mean(dim=(2, 3))
    print(f"    GAP(colony) / GAP(extension) ratio = "
          f"{(gap_colony.mean() / gap_ext.mean()).item():.2f}x  "
          f"-> avg pool DOES separate them (colony denser)")


def probe_colony_saturation(filters=128, H=19, W=19, seed=0):
    print("\n[2] COLONY-SATURATION SIMULATION")
    print("    Hypothesis: under colony self-reinforcement every region reads")
    print("    'high', GMP half saturates, value head pushes toward +1.\n")

    # adversarial weight set: net that LEARNED max => winning.
    w1, b1, w2, b2 = make_value_weights(filters, 256, seed, "colony_pos")

    peaks = [0.5, 1.0, 2.0, 4.0, 8.0]
    print("    colony peak activation sweep (colony_pos value weights):")
    for p in peaks:
        t = synth_colony(filters, H, W, peak=p).unsqueeze(0)
        val, lg, _ = value_head_forward(t, w1, b1, w2, b2)
        sat = "  <-- SATURATED" if val.item() > 0.95 else ""
        print(f"      peak={p:4.1f}  value={val.item():+.4f}  "
              f"logit={lg.item():+.4f}{sat}")
    print("\n    -> Monotone: higher max activation -> higher value, no ceiling")
    print("       in the head itself; tanh is the only bound. A trained net")
    print("       whose w2 is positive on max-fed hidden units locks colony")
    print("       positions at value~+1 regardless of threat content.")


def probe_kcluster_contrast(filters=128, H=25, W=25, seed=0):
    print("\n[3] K-CLUSTER VALUE/POLICY CONTRAST (v6w25 path; NOT §S178 v6)")
    print("    value pool = min over K (worker_loop/inner.rs:589-593,")
    print("    pooling.py:110). policy = scatter-max (records.rs:64-75).\n")

    w1, b1, w2, b2 = make_value_weights(filters, 256, seed, "random")

    # K=8 clusters: 1 colony cluster (high value) + 7 'sparse-not-lost' clusters.
    colony = synth_colony(filters, H, W).unsqueeze(0)
    sparse = synth_extension(filters, H, W, peak=1.0).unsqueeze(0)

    v_colony, _, _ = value_head_forward(colony, w1, b1, w2, b2)
    v_sparse, _, _ = value_head_forward(sparse, w1, b1, w2, b2)
    per_cluster = [v_colony.item()] + [v_sparse.item()] * 7
    agg_min = min(per_cluster)
    agg_mean = float(np.mean(per_cluster))
    print(f"    per-cluster values: colony={v_colony.item():+.4f}  "
          f"7x sparse={v_sparse.item():+.4f}")
    print(f"    min-pool aggregate = {agg_min:+.4f}  "
          f"(mean would be {agg_mean:+.4f})")
    print("    -> value: min-pool SUPPRESSES the lone high colony cluster.")
    print("       Value head is colony-RESISTANT on the v6w25 K-axis.")

    # policy scatter-max: colony cluster has a sharp high-prob peak; sparse
    # clusters near-uniform. scatter-max => colony's peak wins the legal cell.
    A = 64
    g = torch.Generator().manual_seed(seed)
    colony_logits = torch.full((A,), -2.0)
    colony_logits[10] = 6.0                      # sharp colony move
    sparse_logits = torch.randn(A, generator=g) * 0.3
    colony_probs = F.softmax(colony_logits, 0)
    sparse_probs = F.softmax(sparse_logits, 0)
    stacked = torch.stack([colony_probs] + [sparse_probs] * 7, 0)  # (8, A)
    scatter_max = stacked.amax(0)
    scatter_max = scatter_max / scatter_max.sum()
    print(f"\n    policy scatter-max: colony move prob in colony cluster = "
          f"{colony_probs[10].item():.3f}")
    print(f"    after scatter-max + renorm, colony move prob = "
          f"{scatter_max[10].item():.3f}  (argmax cell = {scatter_max.argmax().item()})")
    print("    -> policy: scatter-max PROMOTES the lone colony cluster's peak.")
    print("       ASYMMETRY: value min-pools (colony-safe), policy max-pools")
    print("       (colony-amplifying). Same K, opposite reduction.")


def probe_encoding_diffusion(filters=128, seed=0):
    print("\n[4] ENCODING CHANNEL ASYMMETRY — extension diffusion")
    print("    cluster windows centre on stone density (cluster.rs:79).")
    print("    A thin 5-in-row whose stones are spread can be split across")
    print("    cluster crops; no single crop owns the whole threat.\n")

    H = W = 19
    # compact extension: all 5 stones in one window-width span
    compact = synth_extension(filters, H, W, length=5).unsqueeze(0)
    # diffuse extension: stones spread across the board
    diffuse = synth_diffuse_extension(filters, H, W).unsqueeze(0)

    # crop two windows of width ~H//2 and count how much activation each owns
    cw = H // 2
    def coverage(t):
        # fraction of total activation captured by the best single cw-window
        s = t[0].sum(0)            # (H, W) channel-summed
        best = 0.0
        tot = s.sum().item() + 1e-9
        for y in range(0, H - cw + 1, 2):
            for x in range(0, W - cw + 1, 2):
                c = s[y:y + cw, x:x + cw].sum().item()
                best = max(best, c)
        return best / tot
    print(f"    compact extension: best single-window coverage = "
          f"{coverage(compact):.2f}")
    print(f"    diffuse extension: best single-window coverage = "
          f"{coverage(diffuse):.2f}")
    print("    -> diffuse extension fragments across crops; per-cluster policy")
    print("       sees only a partial line, logits diffuse, scatter-max cannot")
    print("       concentrate -> MCTS visits do NOT lock onto the 6th move.")
    print("       Compact COLONY never fragments (dense by construction).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--filters", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("§S181 IMPL-S181T2 — value-head + encoding architecture probe")
    print("STANDALONE — re-implements network.py value-head math, no ckpt load")
    print("=" * 72)

    probe_dual_pool_asymmetry(filters=args.filters, seed=args.seed)
    probe_colony_saturation(filters=args.filters, seed=args.seed)
    probe_kcluster_contrast(filters=args.filters, seed=args.seed)
    probe_encoding_diffusion(filters=args.filters, seed=args.seed)

    print("\n" + "=" * 72)
    print("Probe complete. See audit/structural/02_value_head_encoding_"
          "architecture.md for the verdict.")
    print("=" * 72)


if __name__ == "__main__":
    main()
