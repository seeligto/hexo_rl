#!/usr/bin/env python3
"""Value-head calibration ladder — clean, ground-truth-anchored value/loss data.

WHY THIS EXISTS
---------------
In-training value diagnostics cannot discriminate a healthy value head from a
collapsing one:

  * trainer ``value_loss`` is the main BCE term (verified: trainer.py logs
    ``compute_value_loss`` alone; uncertainty/aux are separate keys) — but it
    is a BCE, while head "improvement" questions are usually asked in MSE/MAE
    units against ground-truth z. This probe reports BOTH (``value_bce`` is
    directly comparable to the trainer curve).
  * ``avg_sigma`` predicts the *magnitude of the squared error*, not variance —
    it falls both when the head gets MORE accurate AND when the head regresses
    toward a constant and predicts low error regardless.
  * ``V(colony)/V(ext)`` are synthetic T3 positions, not real labelled games.

This probe is FIXTURE-AGNOSTIC (§D-VALPROBE Phase 2): it forwards each
checkpoint's value head over a FIXED slice of any labelled position bank
(npz with ``states`` + ``outcomes``) and computes per-checkpoint:

  sign_acc    fraction sign(v)==sign(z) on decided positions  (is it right?)
  value_mse   mean (v - z)^2                                  (clean value loss)
  value_bce   BCE(v_logit, (z+1)/2)            (trainer ``value_loss`` units)
  mae         mean |v - z|
  value_std   std(v) over the bank          (compression toward constant?)
  value_mean  mean(v)
  ece         n-bin expected calibration error of p=(v+1)/2 vs empirical
  per-phase   sign_acc / value_std in early/mid/late stone-count terciles
  spread      sign_acc / value_std in least/mid/most own-component terciles
              (hex connected components of the mover's stones — the G3 axis)

Run the SAME bank over the whole checkpoint ladder; the TREND across 10k->50k
is the discriminator (train/test overlap is constant across the ladder, so it
cancels out of every delta).

PRE-REGISTERED VERDICTS
  E1/E2 (corpus fixture, §D-VALPROBE banked 2026-06-11):
    T1 std-compression  std(50k)/std(10k)   benign >=0.85 ; collapse <0.85
    T2 sign-accuracy    benign d>=-0.02 AND 50k>=0.68 ; collapse d<-0.02 OR 50k<0.62
    T3 clean value-MSE  benign d<=+0.02 ; collapse d>+0.05
    T4 endgame discrim  sign_acc(late)-sign_acc(mid) >=+0.05 benign
  G1/G3 (self-play fixture, §D-VALPROBE §2):
    G1 GENERALIZES       sign_acc d>=+0.05 AND value_mse d<=-0.05 (50k vs 10k)
    G2 TRAIN-SET-ONLY    G1 fails on selfplay while the corpus trends hold
                         (read at report level against the banked corpus JSON)
    G3 SPREAD-BLIND      sign_acc(most_spread) < sign_acc(least_spread) - 0.10 @tip

Example (corpus fixture, vast):
  python scripts/diagnosis/value_calibration_ladder.py \
    --fixture corpus --encoding v6_live2_ls \
    --ckpt-anchor checkpoints/bootstrap_model_v6_live2.pt \
    --ckpts checkpoints/checkpoint_000{1,2,3,4,5}0000.pt \
    --out audit/structural/value_calibration_ladder.json

Self-play fixture (leak-free, generated post-training at the tip checkpoint):
  ... --fixture selfplay --selfplay-path data/selfplay_fixture_<run>.npz \
      --verdict-mode generalization
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hexo_rl.encoding import cur_stone_slot, lookup, opp_stone_slot  # noqa: E402
from hexo_rl.model.network import HexTacToeNet  # noqa: E402

# ── defaults (single surfaced block — every value overridable via CLI) ───────
DEFAULTS = {
    "draw_band": 0.10,   # |z| < this => not decided, excluded from sign_acc
    "ece_bins": 10,
    "n": 4000,
    "seed": 20260611,
    "batch": 512,
    "min_sign_acc": 0.55,    # perspective self-check floor (best ckpt)
    "baseline_min_step": 10000,  # "lo" rung of every pre-registered delta
    "corpus_path": "data/bootstrap_corpus_v6_live2.npz",
    "selfplay_path": "data/selfplay_fixture_v6_live2_ls_50k.npz",
}

E1E2_THRESHOLDS = {
    "t1_std_ratio": 0.85,
    "t2_sign_delta": -0.02,
    "t2_sign_floor": 0.68,
    "t2_sign_kill": 0.62,
    "t3_mse_benign": 0.02,
    "t3_mse_collapse": 0.05,
    "t4_endgame": 0.05,
}

G_THRESHOLDS = {
    "g1_sign_delta": 0.05,    # sign_acc must rise by at least this
    "g1_mse_delta": -0.05,    # value_mse must fall by at least this
    "g3_spread_gap": -0.10,   # most-spread sign_acc gap vs least-spread
}

HEX_NEIGHBORS: Tuple[Tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
)


# ── metric kernels (pure, unit-tested in tests/test_value_calibration_metrics.py)


def compute_metrics(
    v: np.ndarray, z: np.ndarray, draw_band: float,
    ece_bins: int = DEFAULTS["ece_bins"],
) -> Dict[str, float]:
    """Core calibration metrics of value predictions v against outcomes z."""
    decided = np.abs(z) >= draw_band
    sign_acc = (
        float(np.mean(np.sign(v[decided]) == np.sign(z[decided])))
        if decided.any() else float("nan")
    )
    return {
        "sign_acc": sign_acc,
        "value_mse": float(np.mean((v - z) ** 2)),
        "mae": float(np.mean(np.abs(v - z))),
        "value_std": float(np.std(v)),
        "value_mean": float(np.mean(v)),
        "ece": expected_calibration_error(v, z, n_bins=ece_bins),
        "n_decided": int(decided.sum()),
    }


def expected_calibration_error(v: np.ndarray, z: np.ndarray, n_bins: int = 10) -> float:
    """ECE of p=(v+1)/2 vs empirical win-prob (z+1)/2 over equal-width bins."""
    p = np.clip((v + 1.0) / 2.0, 0.0, 1.0)
    y = (z + 1.0) / 2.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def bce_with_logits(logits: np.ndarray, z: np.ndarray) -> float:
    """Numerically stable BCE(v_logit, (z+1)/2) — the trainer ``value_loss`` unit.

    Identical to torch.nn.functional.binary_cross_entropy_with_logits with
    targets (z+1)/2: mean over max(x,0) - x*y + log(1+exp(-|x|)).
    """
    x = logits.astype(np.float64).reshape(-1)
    y = ((z + 1.0) / 2.0).astype(np.float64).reshape(-1)
    return float(np.mean(np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))))


def tercile_masks(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Disjoint early/mid/late masks at the 1/3 and 2/3 quantiles of x."""
    q1, q2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "early": x <= q1,
        "mid": (x > q1) & (x <= q2),
        "late": x > q2,
    }


def stratified_tercile_masks(
    x: np.ndarray, stratum_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Tercile masks of x computed WITHIN a stratum (in-stratum quantiles only).

    Fixes the §D-VALPROBE within-stratum padding artifact (open item 4):
    ``tercile_masks(np.where(m, x, -1))`` computes the 1/3- and 2/3-quantiles
    over the FULL array, so with ~2/3 of rows padded to -1 both boundaries
    collapse onto the padding value and the within-stratum bins degenerate
    (the whole stratum lands in "late"). Here q1/q2 come from
    ``x[stratum_mask]`` alone.

    Returned masks are pairwise disjoint, each a subset of ``stratum_mask``,
    and their union covers the stratum exactly. Small-int tie degeneracy: x is
    typically a small-int count (hex components), so q1 == q2 is possible —
    then the mid bin (q1 < x <= q2) is EMPTY by construction; acceptable, the
    early/late bins still partition the stratum. An all-False stratum returns
    three all-False masks.
    """
    stratum_mask = np.asarray(stratum_mask, dtype=bool)
    if not stratum_mask.any():
        empty = np.zeros(stratum_mask.shape, dtype=bool)
        return {"early": empty, "mid": empty.copy(), "late": empty.copy()}
    q1, q2 = np.quantile(x[stratum_mask], [1.0 / 3.0, 2.0 / 3.0])
    return {
        "early": stratum_mask & (x <= q1),
        "mid": stratum_mask & (x > q1) & (x <= q2),
        "late": stratum_mask & (x > q2),
    }


def hex_component_count(plane: np.ndarray, threshold: float = 0.5) -> int:
    """Connected components of a binary stone plane under HEX adjacency.

    Axial-coordinate neighbors: the six offsets in HEX_NEIGHBORS — identical
    to the engine's HEX_AXES± set. Window crops are pure translations of
    axial coords (engine window_flat_idx_at_geom), so axial adjacency carries
    to array indices unchanged.
    """
    occ = plane > threshold
    h, w = occ.shape
    seen = np.zeros_like(occ, dtype=bool)
    count = 0
    for a in range(h):
        for b in range(w):
            if not occ[a, b] or seen[a, b]:
                continue
            count += 1
            stack = [(a, b)]
            seen[a, b] = True
            while stack:
                ca, cb = stack.pop()
                for da, db in HEX_NEIGHBORS:
                    na, nb = ca + da, cb + db
                    if 0 <= na < h and 0 <= nb < w and occ[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        stack.append((na, nb))
    return count


def spread_bin_metrics(
    v: np.ndarray, z: np.ndarray, components: np.ndarray, draw_band: float,
) -> Dict[str, Dict[str, float]]:
    """sign_acc / value_std per own-component-count tercile (the G3 axis)."""
    masks = tercile_masks(components)
    names = {"early": "least_spread", "mid": "mid_spread", "late": "most_spread"}
    out: Dict[str, Dict[str, float]] = {}
    for key, m in masks.items():
        vv, zz = v[m], z[m]
        dec = np.abs(zz) >= draw_band
        out[names[key]] = {
            "sign_acc": float(np.mean(np.sign(vv[dec]) == np.sign(zz[dec]))) if dec.any() else float("nan"),
            "value_std": float(np.std(vv)) if m.any() else float("nan"),
            "components_min": float(components[m].min()) if m.any() else float("nan"),
            "components_max": float(components[m].max()) if m.any() else float("nan"),
            "n": int(m.sum()),
        }
    return out


def perspective_check(sign_accs: Sequence[float], min_sign_acc: float) -> Tuple[bool, float]:
    """Built-in flipped-perspective guard: the BEST checkpoint must clear the floor.

    A perspective-flipped extraction reads ~(1 - true sign_acc) ≪ 0.5; a healthy
    ladder's best rung sits well above 0.5.
    """
    finite = [s for s in sign_accs if s == s]  # drop NaN
    best = max(finite) if finite else float("nan")
    return (best == best and best >= min_sign_acc), best


# ── pre-registered classifiers ───────────────────────────────────────────────


def _lo_hi(ladder: List[dict], baseline_min_step: int) -> Tuple[dict, dict, int, int]:
    by_step = {e["step"]: e for e in ladder if e["step"]}
    lo = min(s for s in by_step if s >= baseline_min_step)
    hi = max(by_step)
    return by_step[lo], by_step[hi], lo, hi


def classify(
    ladder: List[dict],
    thresholds: Dict[str, float],
    baseline_min_step: int = DEFAULTS["baseline_min_step"],
) -> Dict[str, object]:
    """Pre-registered E1/E2 verdict over the ladder (lo rung vs tip)."""
    t = thresholds
    e_lo, e_hi, lo, hi = _lo_hi(ladder, baseline_min_step)
    a, b = e_lo["metrics"], e_hi["metrics"]
    r_std = b["value_std"] / a["value_std"] if a["value_std"] else float("inf")
    d_sign = b["sign_acc"] - a["sign_acc"]
    d_mse = b["value_mse"] - a["value_mse"]
    t1 = "benign" if r_std >= t["t1_std_ratio"] else "collapse"
    if d_sign >= t["t2_sign_delta"] and b["sign_acc"] >= t["t2_sign_floor"]:
        t2 = "benign"
    elif d_sign < t["t2_sign_delta"] or b["sign_acc"] < t["t2_sign_kill"]:
        t2 = "collapse"
    else:
        t2 = "ambiguous"
    t3 = "benign" if d_mse <= t["t3_mse_benign"] else (
        "collapse" if d_mse > t["t3_mse_collapse"] else "ambiguous"
    )
    ph = e_hi.get("phase") or {}
    t4 = None
    if ph.get("late") and ph.get("mid"):
        t4 = ph["late"]["sign_acc"] - ph["mid"]["sign_acc"]
    if t1 == "benign" and t2 == "benign" and t3 == "benign":
        verdict = "E1_BENIGN_SHARPENING"
    elif t1 == "collapse" and (t2 == "collapse" or t3 == "collapse"):
        verdict = "E2_MODE_COLLAPSE"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "compare_steps": [lo, hi],
        "T1_std_ratio": round(r_std, 4),
        "T1": t1,
        "T2_sign_delta": round(d_sign, 4),
        "T2_sign_tip": round(b["sign_acc"], 4),
        "T2": t2,
        "T3_mse_delta": round(d_mse, 4),
        "T3": t3,
        "T4_endgame_discrim": round(t4, 4) if t4 is not None else None,
        "T4": "benign" if (t4 is not None and t4 >= t["t4_endgame"]) else "weak",
        "thresholds": dict(t),
    }


def classify_generalization(
    ladder: List[dict],
    thresholds: Dict[str, float],
    baseline_min_step: int = DEFAULTS["baseline_min_step"],
) -> Dict[str, object]:
    """Pre-registered §D-VALPROBE G1/G3 verdict on a self-play-distribution ladder.

    G1 GENERALIZES requires BOTH trends on this fixture; failing G1 yields
    TRAIN_SET_ONLY_CANDIDATE — the final G2 call is made at report level
    against the banked corpus ladder (corpus trends must HOLD for G2).
    G3 reads the tip checkpoint's spread-binned breakdown.
    """
    t = thresholds
    e_lo, e_hi, lo, hi = _lo_hi(ladder, baseline_min_step)
    d_sign = e_hi["metrics"]["sign_acc"] - e_lo["metrics"]["sign_acc"]
    d_mse = e_hi["metrics"]["value_mse"] - e_lo["metrics"]["value_mse"]
    g1 = "GENERALIZES" if (d_sign >= t["g1_sign_delta"] and d_mse <= t["g1_mse_delta"]) \
        else "TRAIN_SET_ONLY_CANDIDATE"
    out: Dict[str, object] = {
        "G1": g1,
        "compare_steps": [lo, hi],
        "g1_sign_delta": round(d_sign, 4),
        "g1_mse_delta": round(d_mse, 4),
        "thresholds": dict(t),
    }
    spread = e_hi.get("spread")
    if spread:
        gap = spread["most_spread"]["sign_acc"] - spread["least_spread"]["sign_acc"]
        out["G3"] = "SPREAD_BLIND" if gap < t["g3_spread_gap"] else "NOT_SPREAD_BLIND"
        out["g3_gap"] = round(gap, 4)
        out["g3_bin_n"] = {k: d["n"] for k, d in spread.items()}
    else:
        out["G3"] = "NOT_EVALUATED"
    return out


# ── bank loading + forwards ──────────────────────────────────────────────────


def load_bank(fixture_path: str, n: int, seed: int, encoding: str):
    """Fixed random slice of a labelled bank: states + ground-truth z + phase
    proxy (occupied-cell count) + spread proxy (mover hex component count)."""
    bank = np.load(fixture_path)
    states = bank["states"]
    outcomes = bank["outcomes"].astype(np.float32)
    total = states.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=min(n, total), replace=False)
    idx.sort()
    st = np.asarray(states[idx], dtype=np.float32)
    zz = outcomes[idx]
    # slots resolved per-encoding (opp t0 is slot 4 v6-family / 1 v6_live2 — L65)
    spec = lookup(encoding)
    cur, opp = cur_stone_slot(spec), opp_stone_slot(spec)
    occ = (st[:, cur] + st[:, opp] > 0.5).sum(axis=(1, 2)).astype(np.int32)
    comps = np.array([hex_component_count(p) for p in st[:, cur]], dtype=np.int32)
    return st, zz, occ, comps


def bank_spread_stats(occ: np.ndarray, comps: np.ndarray) -> Dict[str, object]:
    """FIXTURE-VALID inputs: the bank's spread/occupancy distribution, to be
    compared against the live run's self-play stats before verdicts are read."""
    def _q(x):
        return [float(v) for v in np.quantile(x, [0.0, 1 / 3, 0.5, 2 / 3, 1.0])]
    binc = np.bincount(comps)
    return {
        "n": int(len(comps)),
        "components_mean": float(comps.mean()),
        "components_quantiles_0_33_50_66_100": _q(comps),
        "components_histogram": {str(k): int(c) for k, c in enumerate(binc) if c},
        "occupancy_mean": float(occ.mean()),
        "occupancy_quantiles_0_33_50_66_100": _q(occ),
    }


def forward_values(model, states: np.ndarray, batch: int, device: str):
    """Forward the bank; returns (value tanh-space, v_logit) arrays."""
    out_vals = np.empty(states.shape[0], dtype=np.float32)
    out_logits = np.empty(states.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, states.shape[0], batch):
            t = torch.from_numpy(states[i : i + batch]).to(device)
            out = model(t)  # (log_policy, value, v_logit, *extras)
            out_vals[i : i + batch] = out[1].squeeze(-1).float().cpu().numpy()
            out_logits[i : i + batch] = out[2].squeeze(-1).float().cpu().numpy()
    return out_vals, out_logits


def phase_metrics(v, z, occ, draw_band: float):
    """sign_acc / value_std per early/mid/late stone-count tercile."""
    out = {}
    for name, m in tercile_masks(occ).items():
        vv, zz = v[m], z[m]
        dec = np.abs(zz) >= draw_band
        out[name] = {
            "sign_acc": float(np.mean(np.sign(vv[dec]) == np.sign(zz[dec]))) if dec.any() else float("nan"),
            "value_std": float(np.std(vv)),
            "n": int(m.sum()),
        }
    return out


def load_model(ckpt: str, encoding: str, device: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    model = HexTacToeNet(encoding=encoding)
    model.load_state_dict(state)
    return model.to(device)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _load_thresholds(path: Optional[str], defaults: Dict[str, float]) -> Dict[str, float]:
    if not path:
        return dict(defaults)
    merged = dict(defaults)
    merged.update(json.loads(pathlib.Path(path).read_text()))
    return merged


def main():
    ap = argparse.ArgumentParser(
        description="Value-head calibration ladder over a labelled fixture",
    )
    ap.add_argument("--fixture", default="corpus",
                    help="'corpus' | 'selfplay' | path to a labelled npz (states+outcomes)")
    ap.add_argument("--corpus-path", default=DEFAULTS["corpus_path"])
    ap.add_argument("--selfplay-path", default=DEFAULTS["selfplay_path"])
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--ckpt-anchor", default=None)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=DEFAULTS["n"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--draw-band", type=float, default=DEFAULTS["draw_band"])
    ap.add_argument("--ece-bins", type=int, default=DEFAULTS["ece_bins"])
    ap.add_argument("--min-sign-acc", type=float, default=DEFAULTS["min_sign_acc"],
                    help="perspective self-check floor on the best rung")
    ap.add_argument("--baseline-min-step", type=int, default=DEFAULTS["baseline_min_step"])
    ap.add_argument("--verdict-mode", choices=["e1e2", "generalization", "both"],
                    default=None,
                    help="default: e1e2 for --fixture corpus, generalization for selfplay")
    ap.add_argument("--thresholds-json", default=None,
                    help="JSON file overriding E1E2/G threshold keys")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fixture_path = {
        "corpus": args.corpus_path, "selfplay": args.selfplay_path,
    }.get(args.fixture, args.fixture)
    mode = args.verdict_mode or ("e1e2" if args.fixture == "corpus" else "generalization")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st, z, occ, comps = load_bank(fixture_path, args.n, args.seed, args.encoding)
    spread_stats = bank_spread_stats(occ, comps)
    print(f"[bank] fixture={fixture_path} n={len(z)} "
          f"decided={int((np.abs(z) >= args.draw_band).sum())} z_mean={z.mean():+.3f} "
          f"occ_range=[{occ.min()},{occ.max()}] comp_mean={comps.mean():.2f} device={device}")
    print(f"[bank] FIXTURE-VALID inputs: {json.dumps(spread_stats['components_histogram'])}")

    def step_of(path):
        import re
        m = re.search(r"(\d{4,})", pathlib.Path(path).stem)
        return int(m.group(1)) if m else None

    ladder = []
    targets = ([("anchor", args.ckpt_anchor)] if args.ckpt_anchor else []) + [
        (pathlib.Path(c).stem, c) for c in args.ckpts
    ]
    for label, path in targets:
        model = load_model(path, args.encoding, device)
        v, v_logit = forward_values(model, st, args.batch, device)
        m = compute_metrics(v, z, draw_band=args.draw_band, ece_bins=args.ece_bins)
        m["value_bce"] = bce_with_logits(v_logit, z)
        ph = phase_metrics(v, z, occ, draw_band=args.draw_band)
        sp = spread_bin_metrics(v, z, comps, draw_band=args.draw_band)
        step = 0 if label == "anchor" else step_of(path)
        ladder.append({"label": label, "step": step, "metrics": m, "phase": ph, "spread": sp})
        print(f"  {label:28s} step={str(step):>6} | sign_acc={m['sign_acc']:.3f} "
              f"value_mse={m['value_mse']:.4f} value_bce={m['value_bce']:.4f} "
              f"mae={m['mae']:.4f} std={m['value_std']:.4f} "
              f"mean={m['value_mean']:+.3f} ece={m['ece']:.4f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    persp_ok, persp_best = perspective_check(
        [e["metrics"]["sign_acc"] for e in ladder], args.min_sign_acc,
    )
    print(f"\n[perspective] best sign_acc={persp_best:.3f} floor={args.min_sign_acc} "
          f"=> {'PASS' if persp_ok else 'FAIL — labels likely perspective-flipped'}")

    verdicts: Dict[str, Dict[str, object]] = {}
    n_stepped = sum(1 for e in ladder if e["step"] and e["step"] >= args.baseline_min_step)
    if n_stepped < 2:
        print(f"\n[verdicts] SKIPPED — need >=2 rungs at step>={args.baseline_min_step}, "
              f"have {n_stepped} (smoke/anchor-only run)")
        mode = "none"
    if mode in ("e1e2", "both"):
        verdicts["e1e2"] = classify(
            ladder, _load_thresholds(args.thresholds_json, E1E2_THRESHOLDS),
            baseline_min_step=args.baseline_min_step,
        )
    if mode in ("generalization", "both"):
        verdicts["generalization"] = classify_generalization(
            ladder, _load_thresholds(args.thresholds_json, G_THRESHOLDS),
            baseline_min_step=args.baseline_min_step,
        )

    stepped = [e for e in ladder if e["step"]]
    tip = max(stepped, key=lambda e: e["step"]) if stepped else ladder[-1]
    print("\n=== PER-PHASE (tip) ===")
    for name, d in tip["phase"].items():
        print(f"  {name:6s} sign_acc={d['sign_acc']:.3f} value_std={d['value_std']:.4f} n={d['n']}")
    print("=== SPREAD BINS (tip) ===")
    for name, d in tip["spread"].items():
        print(f"  {name:13s} sign_acc={d['sign_acc']:.3f} value_std={d['value_std']:.4f} "
              f"comps=[{d['components_min']:.0f},{d['components_max']:.0f}] n={d['n']}")
    print("\n=== VERDICTS ===")
    for vk, vd in verdicts.items():
        for k, val in vd.items():
            print(f"  [{vk}] {k}: {val}")

    report = {
        "script": "scripts/diagnosis/value_calibration_ladder.py",
        "fixture": fixture_path,
        "encoding": args.encoding,
        "n": int(len(z)),
        "seed": args.seed,
        "draw_band": args.draw_band,
        "bank_spread_stats": spread_stats,
        "perspective_check": {"pass": persp_ok, "best_sign_acc": persp_best,
                              "floor": args.min_sign_acc},
        "ladder": ladder,
        "verdicts": verdicts,
    }
    if args.out:
        op = pathlib.Path(args.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(report, indent=2))
        print(f"\n[wrote] {op}")
    if not persp_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
