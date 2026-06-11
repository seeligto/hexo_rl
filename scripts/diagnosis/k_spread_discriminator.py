"""§D-VALPROBE rt3/rt4-mandated analyses (eval-only):
(a) sign_acc binned by per-row k_counts (global cluster-view count) per rung;
(b) position-level K-aggregated value (min-pool = deployment semantics) sign_acc;
(c) game-bootstrap CIs for the G1 deltas (effective-n = distinct games);
(d) spread terciles WITHIN occupancy strata at the tip (phase-controlled G3).
"""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/hexo_rl")
import torch
from hexo_rl.encoding import cur_stone_slot, lookup, opp_stone_slot
from scripts.diagnosis.value_calibration_ladder import (
    load_model, forward_values, hex_component_count, tercile_masks)

BANKS = {
    "uniform": "data/selfplay_fixture_v6_live2_ls_50k.npz",
    "occmatched": "data/selfplay_fixture_v6_live2_ls_50k_occmatched.npz",
    "livetail": "data/livetail_bank_e928c854.npz",
}
CKPTS = {0: "checkpoints/bootstrap_model_v6_live2.pt",
         10000: "checkpoints/checkpoint_00010000.pt",
         20000: "checkpoints/checkpoint_00020000.pt",
         40000: "checkpoints/checkpoint_00040000.pt",
         50000: "checkpoints/checkpoint_00050000.pt"}
SEED, N = 20260611, 4000
device = "cuda" if torch.cuda.is_available() else "cpu"
spec = lookup("v6_live2_ls")
cur, opp = cur_stone_slot(spec), opp_stone_slot(spec)
out = {}
for bname, path in BANKS.items():
    bank = np.load(path)
    n_total = len(bank["outcomes"])
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=min(N, n_total), replace=False))
    st = bank["states"][idx].astype(np.float32)
    z = bank["outcomes"][idx].astype(np.float32)
    k = bank["k_counts"][idx]; plies = bank["plies"][idx]; gids = bank["game_ids"][idx]
    occ = (st[:, cur] + st[:, opp] > 0.5).sum(axis=(1, 2))
    preds = {}
    for step, ck in CKPTS.items():
        model = load_model(ck, "v6_live2_ls", device)
        v, _ = forward_values(model, st, 512, device)
        preds[step] = v
        del model; torch.cuda.empty_cache()
    b = {"n": int(len(z)), "n_games": int(len(np.unique(gids)))}
    # (a) k-binned sign_acc
    kb = {}
    for step in (0, 10000, 50000):
        v = preds[step]
        row = {}
        for lab, m in (("k1", k == 1), ("k2", k == 2), ("k3", k == 3), ("k4+", k >= 4)):
            if m.sum() >= 30:
                row[lab] = {"sign_acc": round(float(np.mean(np.sign(v[m]) == np.sign(z[m]))), 4), "n": int(m.sum())}
        kb[str(step)] = row
    b["k_binned"] = kb
    # (b) position-level min-pool aggregation (deployment semantics)
    pos_key = gids.astype(np.int64) * 100000 + plies.astype(np.int64)
    agg = {}
    for step in (0, 10000, 50000):
        v = preds[step]
        accs, zs = {}, {}
        for i, pk in enumerate(pos_key):
            accs[pk] = min(accs.get(pk, 1e9), v[i]); zs[pk] = z[i]
        va = np.array(list(accs.values())); za = np.array([zs[pk] for pk in accs])
        kk = np.array([k[pos_key == pk][0] for pk in list(accs)[:0]])  # skip per-pos k for speed
        agg[str(step)] = {"sign_acc": round(float(np.mean(np.sign(va) == np.sign(za))), 4), "n_pos": int(len(va))}
    b["minpool_position"] = agg
    # min-pool sign_acc binned by position K (10k vs 50k)
    posk = {}
    for i, pk in enumerate(pos_key):
        posk[pk] = k[i]
    pos_list = list(posk)
    for step in (10000, 50000):
        v = preds[step]
        accs, zs = {}, {}
        for i, pk in enumerate(pos_key):
            accs[pk] = min(accs.get(pk, 1e9), v[i]); zs[pk] = z[i]
        row = {}
        for lab, lo, hi in (("K1", 1, 1), ("K2", 2, 2), ("K3", 3, 3), ("K4+", 4, 99)):
            sel = [pk for pk in pos_list if lo <= posk[pk] <= hi]
            if len(sel) >= 30:
                va = np.array([accs[pk] for pk in sel]); za = np.array([zs[pk] for pk in sel])
                row[lab] = {"sign_acc": round(float(np.mean(np.sign(va) == np.sign(za))), 4), "n_pos": len(sel)}
        b[f"minpool_K_binned_{step}"] = row
    # (c) game-bootstrap CI for deltas
    def boot(step_a, step_b, nb=2000):
        ga = np.unique(gids); res_s, res_m = [], []
        brng = np.random.default_rng(SEED)
        va, vb = preds[step_a], preds[step_b]
        for _ in range(nb):
            sample = brng.choice(ga, size=len(ga), replace=True)
            rows = np.concatenate([np.where(gids == g)[0] for g in sample])
            zz = z[rows]
            ds = np.mean(np.sign(vb[rows]) == np.sign(zz)) - np.mean(np.sign(va[rows]) == np.sign(zz))
            dm = np.mean((vb[rows]-zz)**2) - np.mean((va[rows]-zz)**2)
            res_s.append(ds); res_m.append(dm)
        return ([round(float(q), 4) for q in np.quantile(res_s, [.025, .5, .975])],
                [round(float(q), 4) for q in np.quantile(res_m, [.025, .5, .975])])
    for (a_, b_) in ((10000, 50000), (10000, 40000), (20000, 50000)):
        s_ci, m_ci = boot(a_, b_)
        b[f"boot_dsign_{a_}_{b_}"] = s_ci; b[f"boot_dmse_{a_}_{b_}"] = m_ci
    # (d) occupancy-stratified spread terciles at tip
    comps = np.array([hex_component_count(p) for p in st[:, cur]])
    v50 = preds[50000]
    strata = {}
    for sname, m in tercile_masks(occ).items():
        row = {}
        for tname, tm in tercile_masks(np.where(m, comps, -1)).items():
            mm = m & tm
            if mm.sum() >= 50:
                row[tname] = {"sign_acc": round(float(np.mean(np.sign(v50[mm]) == np.sign(z[mm]))), 4), "n": int(mm.sum())}
        strata[sname] = row
    b["tip_spread_within_occ"] = strata
    out[bname] = b
    print(f"[{bname}] done")
json.dump(out, open("audit/structural/valprobe_k_discriminator.json", "w"), indent=1)
print(json.dumps(out, indent=1))
