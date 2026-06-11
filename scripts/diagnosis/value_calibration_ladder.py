#!/usr/bin/env python3
"""Value-head calibration ladder — clean, ground-truth-anchored value/loss data.

WHY THIS EXISTS
---------------
The in-training value diagnostics are AMBIGUOUS by construction and cannot
discriminate a healthy value head from a collapsing one:

  * trainer ``value_loss`` mixes the main value term with the Huber-on-squared-
    error uncertainty target and the aux heads → "loss flat" tells you nothing
    about the main value term in isolation.
  * ``avg_sigma`` predicts the *magnitude of the squared error*, not variance —
    it falls both when the head gets MORE accurate (errors shrink) AND when the
    head regresses toward a constant and simply predicts low error regardless.
  * ``V(colony)/V(ext)`` are synthetic T3 positions, not real labelled games.

This probe forwards each checkpoint's value head over a FIXED slice of REAL
positions carrying ground-truth Monte-Carlo outcome labels ``z`` (the corpus
``outcomes`` array) and computes interpretable numbers that DO discriminate:

  sign_acc   fraction sign(v)==sign(z) on decided positions  (is it right?)
  value_mse  mean (v - z)^2                                  (clean value loss)
  mae        mean |v - z|
  value_std  std(v) over the bank          (compression toward constant?)
  value_mean mean(v)
  ece        10-bin expected calibration error of p=(v+1)/2 vs empirical
  per-phase  sign_acc / value_std in early/mid/late stone-count terciles

Run the SAME bank over the whole checkpoint ladder; the TREND across 10k->50k
is the discriminator (train/test overlap is constant across the ladder, so it
cancels out of every delta).

TWO-ARM HYPOTHESES (pre-registered, see thresholds below)
  E1 BENIGN SHARPENING : value head genuinely improving; trainer value_loss
                         flat because flatness is in aux/uncertainty terms.
  E2 MODE-COLLAPSE     : value/uncertainty head regressing toward a constant.

PRE-REGISTERED THRESHOLDS (50k vs 10k)
  T1 std-compression  std(50k)/std(10k)   benign >=0.85 ; collapse <0.85
  T2 sign-accuracy    benign d>=-0.02 AND 50k>=0.68 ; collapse d<-0.02 OR 50k<0.62
  T3 clean value-MSE  benign d<=+0.02 ; collapse d>+0.05
  T4 endgame discrim  sign_acc(late)-sign_acc(mid) >=+0.05 benign

VERDICT
  E1 if T1&T2&T3 all benign ; E2 if T1 collapse AND (T2 OR T3 collapse) ;
  else INCONCLUSIVE.

Run (vast, where checkpoints + corpus live):
  python scripts/diagnosis/value_calibration_ladder.py \
    --corpus data/bootstrap_corpus_v6_live2.npz \
    --encoding v6_live2_ls \
    --ckpt-anchor checkpoints/bootstrap_model_v6_live2.pt \
    --ckpts checkpoints/checkpoint_00010000.pt checkpoints/checkpoint_00020000.pt \
            checkpoints/checkpoint_00030000.pt checkpoints/checkpoint_00040000.pt \
            checkpoints/checkpoint_00050000.pt \
    --n 4000 --out audit/structural/value_calibration_ladder.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hexo_rl.encoding import cur_stone_slot, lookup, opp_stone_slot  # noqa: E402
from hexo_rl.model.network import HexTacToeNet  # noqa: E402

DRAW_BAND = 0.10  # |z| < this => position not decided, excluded from sign_acc


def load_bank(corpus_path: str, n: int, seed: int, encoding: str):
    """Fixed random slice of the corpus: states + ground-truth z + phase proxy."""
    z = np.load(corpus_path)
    states = z["states"]
    outcomes = z["outcomes"].astype(np.float32)
    total = states.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=min(n, total), replace=False)
    idx.sort()
    st = np.asarray(states[idx], dtype=np.float32)
    zz = outcomes[idx]
    # game-phase proxy: occupied-cell count from the stone planes, slots
    # resolved per-encoding (opp t0 is slot 4 v6-family / 1 v6_live2 — L65).
    # Monotone-ish with ply; used only for coarse early/mid/late terciles.
    spec = lookup(encoding)
    cur, opp = cur_stone_slot(spec), opp_stone_slot(spec)
    occ = (st[:, cur] + st[:, opp] > 0.5).sum(axis=(1, 2)).astype(np.int32)
    return st, zz, occ


def forward_values(model, states: np.ndarray, batch: int, device: str):
    out_vals = np.empty(states.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, states.shape[0], batch):
            t = torch.from_numpy(states[i : i + batch]).to(device)
            out = model(t)
            value = out[1]  # (log_policy, value, v_logit, *extras)
            out_vals[i : i + batch] = value.squeeze(-1).float().cpu().numpy()
    return out_vals


def metrics(v: np.ndarray, z: np.ndarray):
    decided = np.abs(z) >= DRAW_BAND
    sign_acc = float(np.mean(np.sign(v[decided]) == np.sign(z[decided]))) if decided.any() else float("nan")
    mse = float(np.mean((v - z) ** 2))
    mae = float(np.mean(np.abs(v - z)))
    # ECE: p=(v+1)/2 binned vs empirical win-prob (z+1)/2, 10 bins.
    p = np.clip((v + 1.0) / 2.0, 0.0, 1.0)
    y = (z + 1.0) / 2.0
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return {
        "sign_acc": sign_acc,
        "value_mse": mse,
        "mae": mae,
        "value_std": float(np.std(v)),
        "value_mean": float(np.mean(v)),
        "ece": float(ece),
        "n_decided": int(decided.sum()),
    }


def phase_metrics(v, z, occ):
    """sign_acc / value_std per early/mid/late stone-count tercile."""
    q1, q2 = np.quantile(occ, [1 / 3, 2 / 3])
    out = {}
    for name, m in (("early", occ <= q1), ("mid", (occ > q1) & (occ <= q2)), ("late", occ > q2)):
        vv, zz = v[m], z[m]
        dec = np.abs(zz) >= DRAW_BAND
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


def classify(ladder):
    """Pre-registered E1/E2 verdict over the ladder (10k anchor vs 50k tip)."""
    by_step = {e["step"]: e["metrics"] for e in ladder if e["step"] is not None}
    lo = min(s for s in by_step if s >= 10000)
    hi = max(by_step)
    a, b = by_step[lo], by_step[hi]
    r_std = b["value_std"] / a["value_std"] if a["value_std"] else float("inf")
    d_sign = b["sign_acc"] - a["sign_acc"]
    d_mse = b["value_mse"] - a["value_mse"]
    t1 = "benign" if r_std >= 0.85 else "collapse"
    if d_sign >= -0.02 and b["sign_acc"] >= 0.68:
        t2 = "benign"
    elif d_sign < -0.02 or b["sign_acc"] < 0.62:
        t2 = "collapse"
    else:
        t2 = "ambiguous"
    t3 = "benign" if d_mse <= 0.02 else ("collapse" if d_mse > 0.05 else "ambiguous")
    ph = next((e["phase"] for e in ladder if e["step"] == hi), None)
    t4 = None
    if ph:
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
        "T2_sign_50k": round(b["sign_acc"], 4),
        "T2": t2,
        "T3_mse_delta": round(d_mse, 4),
        "T3": t3,
        "T4_endgame_discrim": round(t4, 4) if t4 is not None else None,
        "T4": "benign" if (t4 is not None and t4 >= 0.05) else "weak",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--ckpt-anchor", default=None)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st, z, occ = load_bank(args.corpus, args.n, args.seed, args.encoding)
    print(f"[bank] n={len(z)} decided={int((np.abs(z)>=DRAW_BAND).sum())} "
          f"z_mean={z.mean():+.3f} occ_range=[{occ.min()},{occ.max()}] device={device}")

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
        v = forward_values(model, st, args.batch, device)
        m = metrics(v, z)
        ph = phase_metrics(v, z, occ)
        step = 0 if label == "anchor" else step_of(path)
        ladder.append({"label": label, "step": step, "metrics": m, "phase": ph})
        print(f"  {label:28s} step={str(step):>6} | sign_acc={m['sign_acc']:.3f} "
              f"value_mse={m['value_mse']:.4f} mae={m['mae']:.4f} "
              f"std={m['value_std']:.4f} mean={m['value_mean']:+.3f} ece={m['ece']:.4f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    verdict = classify(ladder)
    print("\n=== PER-PHASE (50k tip) ===")
    tip = max((e for e in ladder if e["step"]), key=lambda e: e["step"])
    for name, d in tip["phase"].items():
        print(f"  {name:6s} sign_acc={d['sign_acc']:.3f} value_std={d['value_std']:.4f} n={d['n']}")
    print("\n=== VERDICT ===")
    for k, val in verdict.items():
        print(f"  {k}: {val}")

    report = {
        "script": "scripts/diagnosis/value_calibration_ladder.py",
        "corpus": args.corpus,
        "encoding": args.encoding,
        "n": int(len(z)),
        "seed": args.seed,
        "draw_band": DRAW_BAND,
        "ladder": ladder,
        "verdict": verdict,
    }
    if args.out:
        op = pathlib.Path(args.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(report, indent=2))
        print(f"\n[wrote] {op}")


if __name__ == "__main__":
    main()
