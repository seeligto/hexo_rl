#!/usr/bin/env python3
"""INDEPENDENT re-derivation of D-FULLSPEC E2 input-feature ablation.

Does NOT rerun dvderisk_e2_featablation.py. Re-implements, from scratch:
  - own game-disjoint whole-game 3-way split (FRESH seeds), assert shared_games==0
  - own small from-scratch conv value net (4-plane CONTROL vs 8-plane TREATMENT)
  - own class-balanced value-MSE train loop + early stop on game-disjoint VAL
  - own turn-phase-matched holdout eval (tp==0 INTERSECT overlapping stone support)
  - own KILL-A / KILL-C

REUSED ONLY (read-only data ingestion / verified threat compute):
  build_pool                 byte-exact npz->DS1-CSV game_id join (read-only banks)
  compute_threat_planes_batch threat-plane numpy (re-verified independently here)

Verdict logic re-derived from MY numbers. Read-only on the npz banks.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dvderisk_lighttrunk_probe import build_pool
from scripts.dvderisk_e2_reprobe import (
    compute_threat_planes_batch, compute_threat_planes_single,
)
from hexo_rl.utils.device import best_device


# ---------------------------------------------------------------------------
# Independent small from-scratch conv value net. Only conv1.in_channels differs.
# ---------------------------------------------------------------------------
class IndepConvNet(nn.Module):
    def __init__(self, in_channels, width=32, fc=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(2 * width, fc), nn.ReLU(inplace=True), nn.Linear(fc, 1))

    def forward(self, x):
        h = self.body(x)
        z = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return torch.tanh(self.head(z)).squeeze(1)


def ka(vl):  # KILL-A: frac losses called loss (v<0)
    return float((vl < 0).mean()) if len(vl) else float("nan")


def kc(vw):  # KILL-C: frac wins called win (v>0)
    return float((vw > 0).mean()) if len(vw) else float("nan")


@torch.no_grad()
def predict(net, states, device, bs=256):
    net.eval()
    out = []
    for s in range(0, len(states), bs):
        out.append(net(torch.from_numpy(states[s:s + bs]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def my_split(gids, holdout_frac, val_frac, seed):
    """Independent whole-game 3-way split. Each distinct game -> exactly one side."""
    uniq = np.unique(gids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_ho = max(1, int(round(len(uniq) * holdout_frac)))
    rest = perm[n_ho:]
    n_val = max(1, int(round(len(rest) * val_frac)))
    ho_g = set(perm[:n_ho].tolist())
    val_g = set(rest[:n_val].tolist())
    tr_g = set(rest[n_val:].tolist())
    ho = np.array([g in ho_g for g in gids])
    val = np.array([g in val_g for g in gids])
    tr = np.array([g in tr_g for g in gids])
    return tr, val, ho, len(tr_g), len(val_g), len(ho_g)


def my_matched(lab, tp, stones, mask):
    """Independent turn-phase-matched stratum: tp==0 INTERSECT overlapping stone
    support between losses and wins inside `mask`. Returns (loss_mask, win_mask)."""
    l0 = (lab < 0) & mask & (tp == 0)
    w0 = (lab > 0) & mask & (tp == 0)
    if l0.sum() == 0 or w0.sum() == 0:
        return None
    lo = max(stones[l0].min(), stones[w0].min())
    hi = min(stones[l0].max(), stones[w0].max())
    lm = l0 & (stones >= lo) & (stones <= hi)
    wm = w0 & (stones >= lo) & (stones <= hi)
    return lm, wm, float(lo), float(hi)


def train_net(net, tr_s, tr_y, val_s, val_y, device, *, seed, max_epochs=120, bpc=32, patience=20):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    li = np.where(tr_y < 0)[0]
    wi = np.where(tr_y > 0)[0]
    vl_s, vw_s = val_s[val_y < 0], val_s[val_y > 0]

    def val_mse():
        net.eval()
        with torch.no_grad():
            vl = predict(net, vl_s, device)
            vw = predict(net, vw_s, device)
        return 0.5 * float(((vl + 1) ** 2).mean()) + 0.5 * float(((vw - 1) ** 2).mean())

    rng = np.random.default_rng(seed)
    nl, nw = len(li), len(wi)
    steps = max(nl, nw) // bpc + 1
    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    for ep in range(max_epochs):
        net.train()
        pl, pw = rng.permutation(nl), rng.permutation(nw)
        for st in range(steps):
            bl = li[pl[(st * bpc) % nl:][:bpc]]
            bw = wi[pw[(st * bpc) % nw:][:bpc]]
            if len(bl) == 0 or len(bw) == 0:
                continue
            xs = np.concatenate([tr_s[bl], tr_s[bw]])
            ts = np.concatenate([np.full(len(bl), -1.0, "float32"), np.full(len(bw), 1.0, "float32")])
            opt.zero_grad()
            v = net(torch.from_numpy(xs).to(device))
            mse(v, torch.from_numpy(ts).to(device)).backward()
            opt.step()
        vm = val_mse()
        if vm < best - 1e-5:
            best, bad, best_ep = vm, 0, ep
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return best, best_ep


def run_cond(states_pool, lab, tp, stones, tr, val, ho, device, *, in_ch, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = IndepConvNet(in_channels=in_ch).to(device)
    train_net(net, states_pool[tr], lab[tr], states_pool[val], lab[val], device, seed=seed)
    v_all = predict(net, states_pool, device)
    out = {}
    mh = my_matched(lab, tp, stones, ho)
    mt = my_matched(lab, tp, stones, tr)
    lm, wm, lo, hi = mh
    out["matched_ho"] = {"ka": ka(v_all[lm]), "kc": kc(v_all[wm]),
                         "n_loss": int(lm.sum()), "n_win": int(wm.sum()), "support": [lo, hi]}
    lmt, wmt, _, _ = mt
    out["matched_tr"] = {"ka": ka(v_all[lmt]), "kc": kc(v_all[wmt]),
                         "n_loss": int(lmt.sum()), "n_win": int(wmt.sum())}
    out["naive_ho"] = {"ka": ka(v_all[(lab < 0) & ho]), "kc": kc(v_all[(lab > 0) & ho])}
    return out


def verify_threat():
    """Independent re-check of threat-plane unit semantics."""
    checks = {}
    Z = np.zeros((19, 19), np.float32)
    t = compute_threat_planes_single(Z, Z)
    checks["empty_all_zero"] = bool(np.allclose(t, 0.0))
    cur = Z.copy(); cur[9, 9] = 1.0
    t = compute_threat_planes_single(cur, Z)
    # single stone: 3 hex axes each contribute open windows; open_line_count at the
    # stone cell normalized /6 -> count of unblocked windows containing it /6.
    # plane0 raw count = (# windows containing (9,9)). We only check the script's stated 3.0.
    checks["single_stone_open_line_count_3"] = bool(abs(float(t[0, 9, 9]) - 3.0) < 1e-5)
    cur = Z.copy()
    for r in range(9, 14):
        cur[r, 9] = 1.0  # 5 stones in a column
    t = compute_threat_planes_single(cur, Z)
    # completion cell (14,9): best window fill = 5 stones in the 6-window => 5/6
    checks["five_in_row_best_fill_5_6"] = bool(abs(float(t[2, 14, 9]) - 5.0 / 6.0) < 1e-4)
    b = compute_threat_planes_batch(np.stack([np.stack([cur, Z, Z, Z])]))
    checks["batch_eq_single"] = bool(np.allclose(b[0], t))
    return checks, all(checks.values())


def main():
    t0 = time.time()
    device = best_device()
    SEEDS = [11, 42, 77]   # FRESH seeds (reprobe used 101/202/303)
    HOLD, VALF = 0.25, 0.18

    tp_checks, tp_ok = verify_threat()
    print(f"[indep] threat verify {tp_checks} all_pass={tp_ok}", flush=True)
    assert tp_ok, "threat plane verification failed"

    pool = build_pool()
    lab = pool["label"]
    states4 = pool["states"]
    tp = pool["tp"]
    stones = pool["stones"]
    gid = pool["gid"]
    print(f"[indep] pool n={len(lab)} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
          f"games={len(np.unique(gid))} p23_eq={pool['p23_eq']}", flush=True)

    threat = compute_threat_planes_batch(states4)
    states8 = np.concatenate([states4, threat], axis=1).astype("float32")
    print(f"[indep] states4={states4.shape} states8={states8.shape}", flush=True)

    # empirical fire-rate sanity (opp_best_window_fill plane idx 3 of threat -> plane 7 of 8)
    obwf = threat[:, 3].reshape(len(threat), -1).max(axis=1)
    fires = obwf >= (4.0 / 6.0 - 1e-6)
    print(f"[indep] opp_bwf>=4/6 fires loss={float(fires[lab<0].mean()):.3f} win={float(fires[lab>0].mean()):.3f}", flush=True)

    per_seed = []
    for sd in SEEDS:
        tr, val, ho, ng_tr, ng_val, ng_ho = my_split(gid, HOLD, VALF, sd)
        g_tr = set(gid[tr].tolist()); g_val = set(gid[val].tolist()); g_ho = set(gid[ho].tolist())
        shared = len(g_tr & g_ho) + len(g_tr & g_val) + len(g_val & g_ho)
        assert shared == 0, f"seed {sd} shared_games={shared}"
        # turn-phase matched stratum verification: every matched pos must have plane2==plane3==0
        mh = my_matched(lab, tp, stones, ho)
        lm, wm, _, _ = mh
        matched_idx = np.where(lm | wm)[0]
        p2_ok = bool(np.all(states4[matched_idx, 2] == 0.0))
        p3_ok = bool(np.all(states4[matched_idx, 3] == 0.0))

        ctrl = run_cond(states4, lab, tp, stones, tr, val, ho, device, in_ch=4, seed=sd)
        trt = run_cond(states8, lab, tp, stones, tr, val, ho, device, in_ch=8, seed=sd)
        print(f"[seed {sd}] games tr/val/ho={ng_tr}/{ng_val}/{ng_ho} shared={shared} "
              f"tpmatch(p2==0,p3==0)={p2_ok},{p3_ok}", flush=True)
        print(f"[seed {sd}] CONTROL   matched KA={ctrl['matched_ho']['ka']:.3f} KC={ctrl['matched_ho']['kc']:.3f} "
              f"(l/w={ctrl['matched_ho']['n_loss']}/{ctrl['matched_ho']['n_win']}) trfitKC={ctrl['matched_tr']['kc']:.3f}", flush=True)
        print(f"[seed {sd}] TREATMENT matched KA={trt['matched_ho']['ka']:.3f} KC={trt['matched_ho']['kc']:.3f} "
              f"(l/w={trt['matched_ho']['n_loss']}/{trt['matched_ho']['n_win']}) trfitKC={trt['matched_tr']['kc']:.3f}", flush=True)
        per_seed.append({"seed": sd, "shared": shared, "p2_ok": p2_ok, "p3_ok": p3_ok,
                         "control": ctrl, "treatment": trt})

    def mean_ho(cond, f):
        return float(np.mean([s[cond]["matched_ho"][f] for s in per_seed]))

    def mean_tr(cond):
        return float(np.mean([s[cond]["matched_tr"]["kc"] for s in per_seed]))

    c_ka, c_kc, c_tkc = mean_ho("control", "ka"), mean_ho("control", "kc"), mean_tr("control")
    t_ka, t_kc, t_tkc = mean_ho("treatment", "ka"), mean_ho("treatment", "kc"), mean_tr("treatment")
    delta = t_kc - c_kc
    c_overfit = c_tkc - c_kc
    t_overfit = t_tkc - t_kc

    control_separates = (c_ka > 0.35) and (c_kc >= 0.85)
    reproduces_entangled = not control_separates
    treat_sep = (t_ka > 0.35) and (t_kc >= 0.85)
    treat_all_sep = all((s["treatment"]["matched_ho"]["ka"] > 0.35 and
                         s["treatment"]["matched_ho"]["kc"] >= 0.85) for s in per_seed)
    big_overfit_t = t_overfit > 0.20

    if not reproduces_entangled:
        verdict = "ENTANGLED_R_INSTRUMENT_BROKEN"
    elif treat_sep and treat_all_sep and not big_overfit_t:
        verdict = "SEPARABLE_R"
    elif delta >= 0.15 and t_kc > c_kc:
        verdict = "PARTIAL_R"
    else:
        verdict = "ENTANGLED_R"

    result = {
        "ran": True,
        "device": str(device),
        "seeds": SEEDS,
        "threat_checks": tp_checks,
        "pool": {"n": int(len(lab)), "loss": int((lab < 0).sum()), "win": int((lab > 0).sum()),
                 "games": int(len(np.unique(gid))), "p23_eq": pool["p23_eq"]},
        "opp_bwf_fire": {"loss": float(fires[lab < 0].mean()), "win": float(fires[lab > 0].mean())},
        "control_4plane": {"matched_ka": round(c_ka, 4), "matched_kc": round(c_kc, 4),
                           "train_kc": round(c_tkc, 4), "overfit": round(c_overfit, 4),
                           "reproduces_entangled": bool(reproduces_entangled)},
        "treatment_8plane": {"matched_ka": round(t_ka, 4), "matched_kc": round(t_kc, 4),
                             "train_kc": round(t_tkc, 4), "overfit": round(t_overfit, 4)},
        "delta_matched_kc": round(delta, 4),
        "per_seed_kc": {"control": [round(s["control"]["matched_ho"]["kc"], 3) for s in per_seed],
                        "treatment": [round(s["treatment"]["matched_ho"]["kc"], 3) for s in per_seed]},
        "per_seed_ka": {"control": [round(s["control"]["matched_ho"]["ka"], 3) for s in per_seed],
                        "treatment": [round(s["treatment"]["matched_ho"]["ka"], 3) for s in per_seed]},
        "matched_n_per_seed": [[s["control"]["matched_ho"]["n_loss"], s["control"]["matched_ho"]["n_win"]] for s in per_seed],
        "tp_match_ok_all": all(s["p2_ok"] and s["p3_ok"] for s in per_seed),
        "shared_games_all_zero": all(s["shared"] == 0 for s in per_seed),
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }
    out_dir = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "E2_INDEP_REVIEW_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("INDEP_RESULT_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
