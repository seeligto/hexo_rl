#!/usr/bin/env python3
"""D-FULLSPEC E2 — INPUT-FEATURE ABLATION (4-plane CONTROL vs 8-plane TREATMENT).

Replaces the WEAK random-init-frozen-trunk reprobe (a random ResNet trunk cannot
read spatial threat patterns) with an apples-to-apples input-feature ablation:

  ONE small FROM-SCRATCH conv value-regressor. The ONLY thing that differs
  between the two conditions is the input channel count.

    CONTROL   = 4-plane input (the existing v6_live2 planes, incl turn-phase).
    TREATMENT = 8-plane input (4 base + 4 opponent-agnostic THREAT planes).

  Identical architecture (except conv1 in_channels), identical hyperparams,
  identical training schedule, identical data + game-disjoint split + matched eval.

Question this gates: do RICHER INPUT FEATURES (4 threat planes) carry the
win/loss discriminative signal the 4 base planes lack? E1 frozen trunk and the
light-trunk unfreeze BOTH ENTANGLE on the 4 base planes -> the blind-spot is a
representational/FEATURE problem. This is the last cheap gate before an expensive
bootstrap restart.

Data layer REUSED VERBATIM from scripts/dvderisk_lighttrunk_probe.py:
  build_pool          byte-exact state->DS1-CSV game_id join (1166 pos / 308 games)
  three_way_game_split whole-game tr/val/ho, asserts shared_games==0
  matched_masks        tp==0 INTERSECT overlapping stone-count matched eval
Threat planes REUSED from scripts/dvderisk_e2_reprobe.py:
  compute_threat_planes_batch  (unit-tested; verified again at startup here)

DEFINITIONS (turn-phase-MATCHED holdout):
  KILL-A = frac matched-holdout LOSSES predicted value<0   (pass > 0.35)
  KILL-C = frac matched-holdout WINS   predicted value>0   (pass >= 0.85)

VERDICT (game-disjoint, turn-phase-matched holdout, multiseed mean):
  reproduces_entangled : the 4-plane CONTROL does NOT jointly separate (instrument valid).
                         If the control SEPARATES, instrument broken -> STOP, no SEPARABLE.
  SEPARABLE_R : TREATMENT joint-separates (KILL-A>0.35 AND KILL-C>=0.85) where CONTROL
                does not, robust across seeds, not overfit (holdout ~ train).
  PARTIAL_R   : TREATMENT materially beats CONTROL (KILL-C +>=0.15) but < 0.85.
  ENTANGLED_R : TREATMENT ~= CONTROL (both crater).

Usage:
    .venv/bin/python scripts/dvderisk_e2_featablation.py
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- proven data layer (reused verbatim) ---
from scripts.dvderisk_lighttrunk_probe import (
    build_pool, three_way_game_split, matched_masks, kill_a, kill_c,
)
# --- threat planes (reused; verified at startup) ---
from scripts.dvderisk_e2_reprobe import (
    compute_threat_planes_batch, compute_threat_planes_single,
)
from hexo_rl.utils.device import best_device

LOG_PATH = REPO_ROOT / "logs" / "dvderisk_e2_featablation.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(str(LOG_PATH), mode="w"), logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# From-scratch small conv value-regressor. ONLY conv1.in_channels differs.
# ---------------------------------------------------------------------------
class ConvValueNet(nn.Module):
    def __init__(self, in_channels: int, width: int = 32, fc: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),        nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),        nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        # global avg pool ++ global max pool -> 2*width
        self.head = nn.Sequential(
            nn.Linear(2 * width, fc), nn.ReLU(inplace=True),
            nn.Linear(fc, 1),
        )

    def forward(self, x):
        h = self.conv(x)                       # (N, width, 19, 19)
        avg = h.mean(dim=(2, 3))               # (N, width)
        mx = h.amax(dim=(2, 3))                # (N, width)
        z = torch.cat([avg, mx], dim=1)        # (N, 2*width)
        v = torch.tanh(self.head(z)).squeeze(1)  # (N,)
        return v


ARCH_STR = ("from-scratch 3x[Conv3x3(.,32,pad1)-BN-ReLU] + GlobalAvgPool++GlobalMaxPool(->64) "
            "-> FC(64,64)-ReLU -> FC(64,1)-tanh; conv1.in_channels = 4 (CONTROL) / 8 (TREATMENT); "
            "identical otherwise")


@torch.no_grad()
def value_outputs(net, states, device, bs=256):
    net.eval()
    out = []
    for s in range(0, len(states), bs):
        x = torch.from_numpy(states[s:s + bs]).to(device)
        out.append(net(x).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_value_net(net, tr_s, tr_y, val_s, val_y, device, *,
                    lr=1e-3, max_epochs=120, bpc=32, patience=20, seed=0, label="cond"):
    """Class-balanced full-spectrum value-MSE on +-1 targets. Early-stop on the
    (game-disjoint) balanced VAL MSE. Returns (best_val_mse, best_epoch)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    li = np.where(tr_y < 0)[0]
    wi = np.where(tr_y > 0)[0]
    val_l = val_s[val_y < 0]
    val_w = val_s[val_y > 0]

    def val_mse():
        net.eval()
        with torch.no_grad():
            vl = value_outputs(net, val_l, device)
            vw = value_outputs(net, val_w, device)
        return 0.5 * float(((vl + 1) ** 2).mean()) + 0.5 * float(((vw - 1) ** 2).mean())

    rng = np.random.default_rng(seed)
    nl, nw = len(li), len(wi)
    steps = max(nl, nw) // bpc + 1
    best = float("inf"); best_state = None; best_epoch = -1; bad = 0
    for ep in range(max_epochs):
        net.train()
        pl = rng.permutation(nl); pw = rng.permutation(nw)
        for st in range(steps):
            bl = li[pl[(st * bpc) % nl:][:bpc]]
            bw = wi[pw[(st * bpc) % nw:][:bpc]]
            if len(bl) == 0 or len(bw) == 0:
                continue
            xs = np.concatenate([tr_s[bl], tr_s[bw]])
            ts = np.concatenate([np.full(len(bl), -1.0, "float32"),
                                 np.full(len(bw), 1.0, "float32")])
            opt.zero_grad()
            v = net(torch.from_numpy(xs).to(device))
            mse(v, torch.from_numpy(ts).to(device)).backward()
            opt.step()
        vm = val_mse()
        if vm < best - 1e-5:
            best = vm; bad = 0; best_epoch = ep; best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    log.info(f"[{label}] best_val_mse={best:.5f} best_epoch={best_epoch} (n_tr l/w={nl}/{nw})")
    return best, best_epoch


def matched_eval(pool, v_all, mask, label=""):
    """KILL-A/KILL-C on the turn-phase-matched subset of `mask` (matched_masks)."""
    mm = matched_masks(pool, mask)
    if mm is None:
        return None
    lm, wm, lo, hi = mm
    return {"kill_a": kill_a(v_all[lm]), "kill_c": kill_c(v_all[wm]),
            "n_loss": int(lm.sum()), "n_win": int(wm.sum()), "stone_support": [lo, hi]}


def run_condition(states_pool, pool, tr, val, ho, device, *, in_channels, seed, label):
    """Train one condition (4ch or 8ch) on a fixed game-disjoint split; return
    matched-holdout + matched-train-fit KILL-A/KILL-C."""
    lab = pool["label"]
    torch.manual_seed(seed); np.random.seed(seed)
    net = ConvValueNet(in_channels=in_channels).to(device)
    train_value_net(net, states_pool[tr], lab[tr], states_pool[val], lab[val], device,
                    seed=seed, label=label)
    v_all = value_outputs(net, states_pool, device)
    ho_m = matched_eval(pool, v_all, ho, label + "_ho")
    tr_m = matched_eval(pool, v_all, tr, label + "_trfit")
    naive = {"kill_a": kill_a(v_all[(lab < 0) & ho]), "kill_c": kill_c(v_all[(lab > 0) & ho])}
    return {"matched_holdout": ho_m, "matched_trainfit": tr_m, "naive_holdout": naive,
            "mean_v_loss_ho": float(v_all[(lab < 0) & ho].mean()),
            "mean_v_win_ho": float(v_all[(lab > 0) & ho].mean())}


def verify_threat_planes():
    """Re-verify on hand-checkable positions before trusting the ablation."""
    checks = {}
    empty = np.zeros((19, 19), np.float32)
    t = compute_threat_planes_single(empty, empty)
    checks["empty_all_zero"] = bool(np.allclose(t, 0.0))
    cur = np.zeros((19, 19), np.float32); cur[9, 9] = 1.0
    t = compute_threat_planes_single(cur, np.zeros((19, 19), np.float32))
    checks["single_stone_open_line_count_3"] = bool(abs(float(t[0, 9, 9]) - 3.0) < 1e-5)
    cur = np.zeros((19, 19), np.float32)
    for r in range(9, 14):
        cur[r, 9] = 1.0  # 5-in-a-row
    t = compute_threat_planes_single(cur, np.zeros((19, 19), np.float32))
    checks["five_in_row_completion_best_fill_5_6"] = bool(abs(float(t[2, 14, 9]) - 5.0 / 6.0) < 1e-4)
    # batch == single
    batch = compute_threat_planes_batch(np.stack([np.stack([cur, np.zeros((19, 19), np.float32),
                                                            np.zeros((19, 19), np.float32),
                                                            np.zeros((19, 19), np.float32)])]))
    checks["batch_equals_single"] = bool(np.allclose(batch[0], t))
    return checks, all(checks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    ap.add_argument("--max-epochs", type=int, default=120)
    args = ap.parse_args()

    t0 = time.time()
    device = best_device()
    log.info("=" * 72)
    log.info("D-FULLSPEC E2 — INPUT-FEATURE ABLATION (4-plane CONTROL vs 8-plane TREATMENT)")
    log.info("=" * 72)

    # 0. Verify threat-plane computation.
    tp_checks, tp_ok = verify_threat_planes()
    log.info(f"threat-plane verification: {tp_checks} all_pass={tp_ok}")
    assert tp_ok, f"threat-plane verification FAILED: {tp_checks}"

    # 1. Build pool (game-disjoint, byte-exact join).
    pool = build_pool()
    lab = pool["label"]
    states4 = pool["states"]              # (N,4,19,19)
    log.info(f"pool n={len(lab)} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
             f"distinct_games={len(np.unique(pool['gid']))} plane2==plane3 everywhere={pool['p23_eq']}")

    # 2. Threat planes computed ONCE for the whole pool, cached, concat -> 8-plane.
    log.info("Computing threat planes for full pool (once)...")
    threat = compute_threat_planes_batch(states4)          # (N,4,19,19)
    states8 = np.concatenate([states4, threat], axis=1).astype("float32")  # (N,8,19,19)
    log.info(f"states4={states4.shape} states8={states8.shape}")

    # Empirical signal sanity: opp_best_window_fill>=4/6 fires more on losses (matched holdout).
    opp_bwf_max = threat[:, 3].reshape(len(threat), -1).max(axis=1)   # plane idx 3 = opp_best_window_fill
    fires = opp_bwf_max >= (4.0 / 6.0 - 1e-6)
    log.info(f"opp_best_window_fill>=4/6 fires (full pool): loss={float(fires[lab<0].mean()):.3f} "
             f"win={float(fires[lab>0].mean()):.3f}")

    # 3. Multi-seed game-disjoint ablation.
    per_seed = []
    for seed in args.seeds:
        tr, val, ho, n_g, n_tr_g, n_val_g, n_ho_g = three_way_game_split(
            pool["gid"], args.holdout_frac, args.val_frac, seed)
        g_tr = set(pool["gid"][tr].tolist()); g_val = set(pool["gid"][val].tolist()); g_ho = set(pool["gid"][ho].tolist())
        shared = len(g_tr & g_ho) + len(g_tr & g_val) + len(g_val & g_ho)
        assert shared == 0, f"split seed {seed}: shared_games={shared} (NOT game-disjoint)"
        log.info(f"[seed {seed}] games tr/val/ho={n_tr_g}/{n_val_g}/{n_ho_g} shared={shared} "
                 f"ho(l/w)={int(((lab<0)&ho).sum())}/{int(((lab>0)&ho).sum())}")

        ctrl = run_condition(states4, pool, tr, val, ho, device,
                             in_channels=4, seed=seed, label=f"CONTROL_s{seed}")
        trt = run_condition(states8, pool, tr, val, ho, device,
                            in_channels=8, seed=seed, label=f"TREATMENT_s{seed}")
        c_ho = ctrl["matched_holdout"]; c_tf = ctrl["matched_trainfit"]
        t_ho = trt["matched_holdout"]; t_tf = trt["matched_trainfit"]
        log.info(f"[seed {seed}] CONTROL   matched-HO KA={c_ho['kill_a']:.3f} KC={c_ho['kill_c']:.3f} "
                 f"(n l/w={c_ho['n_loss']}/{c_ho['n_win']}) | trainfit KC={c_tf['kill_c']:.3f}")
        log.info(f"[seed {seed}] TREATMENT matched-HO KA={t_ho['kill_a']:.3f} KC={t_ho['kill_c']:.3f} "
                 f"(n l/w={t_ho['n_loss']}/{t_ho['n_win']}) | trainfit KC={t_tf['kill_c']:.3f}")
        per_seed.append({"seed": seed, "shared_games": shared,
                         "n_games": [n_tr_g, n_val_g, n_ho_g],
                         "control": ctrl, "treatment": trt})

    # 4. Multiseed means.
    def mean_of(cond, field):
        return float(np.mean([s[cond]["matched_holdout"][field] for s in per_seed]))
    def mean_tf(cond):
        return float(np.mean([s[cond]["matched_trainfit"]["kill_c"] for s in per_seed]))

    c_ka = mean_of("control", "kill_a"); c_kc = mean_of("control", "kill_c"); c_tkc = mean_tf("control")
    t_ka = mean_of("treatment", "kill_a"); t_kc = mean_of("treatment", "kill_c"); t_tkc = mean_tf("treatment")
    delta_kc = t_kc - c_kc
    c_overfit = c_tkc - c_kc
    t_overfit = t_tkc - t_kc

    # 5. Verdict.
    control_separates = (c_ka > 0.35) and (c_kc >= 0.85)
    reproduces_entangled = not control_separates
    treat_separates = (t_ka > 0.35) and (t_kc >= 0.85)
    # robustness: every seed's treatment must clear the joint box for SEPARABLE
    treat_all_seeds_sep = all(
        (s["treatment"]["matched_holdout"]["kill_a"] > 0.35 and
         s["treatment"]["matched_holdout"]["kill_c"] >= 0.85) for s in per_seed)
    big_overfit_t = t_overfit > 0.20

    if not reproduces_entangled:
        verdict = "ENTANGLED_R"  # placeholder; rationale flags broken instrument
        rationale = (f"INSTRUMENT BROKEN: 4-plane CONTROL itself SEPARATES on matched holdout "
                     f"(KA={c_ka:.3f} KC={c_kc:.3f}>=0.85) -> turn-phase leak / overfit. "
                     f"Cannot issue a SEPARABLE verdict off a broken control. reproduces_entangled=false.")
    elif treat_separates and treat_all_seeds_sep and not big_overfit_t:
        verdict = "SEPARABLE_R"
        rationale = (f"8-plane TREATMENT joint-separates on matched holdout (KA={t_ka:.3f}>0.35 AND "
                     f"KC={t_kc:.3f}>=0.85) where 4-plane CONTROL does NOT (KA={c_ka:.3f} KC={c_kc:.3f}); "
                     f"robust across all seeds; not overfit (treatment train-fit KC={t_tkc:.3f} vs holdout "
                     f"{t_kc:.3f}, gap={t_overfit:+.3f}). Threat features ARE the fix -> v6_live2_ft "
                     f"bootstrap restart JUSTIFIED.")
    elif delta_kc >= 0.15 and t_kc > c_kc:
        verdict = "PARTIAL_R"
        rationale = (f"8-plane TREATMENT materially beats CONTROL on matched-holdout KILL-C "
                     f"(treatment {t_kc:.3f} vs control {c_kc:.3f}, delta={delta_kc:+.3f}) but does NOT "
                     f"clear 0.85 (KA={t_ka:.3f}). Threat features HELP but are not sufficient alone; "
                     f"restart promising, not proven. overfit gap treat={t_overfit:+.3f}.")
    else:
        verdict = "ENTANGLED_R"
        rationale = (f"8-plane TREATMENT ~= 4-plane CONTROL on matched holdout (treatment KA={t_ka:.3f} "
                     f"KC={t_kc:.3f} vs control KA={c_ka:.3f} KC={c_kc:.3f}, delta_KC={delta_kc:+.3f}); both "
                     f"crater below 0.85. Even these threat features do NOT separate -> escalate "
                     f"(different features/architecture); restart NOT justified on these planes.")

    log.info(f"[MULTISEED MEAN] CONTROL  matched KA={c_ka:.3f} KC={c_kc:.3f} trainfitKC={c_tkc:.3f} (overfit {c_overfit:+.3f})")
    log.info(f"[MULTISEED MEAN] TREATMENT matched KA={t_ka:.3f} KC={t_kc:.3f} trainfitKC={t_tkc:.3f} (overfit {t_overfit:+.3f})")
    log.info(f"[DELTA] matched KILL-C treatment-control = {delta_kc:+.3f}")
    log.info(f"reproduces_entangled={reproduces_entangled} VERDICT={verdict}")
    log.info(rationale)

    elapsed = time.time() - t0
    multiseed_str = (f"seeds {args.seeds}; CONTROL per-seed KC=" +
                     str([round(s["control"]["matched_holdout"]["kill_c"], 3) for s in per_seed]) +
                     f" mean {c_kc:.3f}; TREATMENT per-seed KC=" +
                     str([round(s["treatment"]["matched_holdout"]["kill_c"], 3) for s in per_seed]) +
                     f" mean {t_kc:.3f}; per-seed KA control=" +
                     str([round(s["control"]["matched_holdout"]["kill_a"], 3) for s in per_seed]) +
                     " treatment=" +
                     str([round(s["treatment"]["matched_holdout"]["kill_a"], 3) for s in per_seed]))
    overfit_str = (f"CONTROL train-fit KC {c_tkc:.3f} vs holdout {c_kc:.3f} (gap {c_overfit:+.3f}); "
                   f"TREATMENT train-fit KC {t_tkc:.3f} vs holdout {t_kc:.3f} (gap {t_overfit:+.3f}); "
                   f">0.20 = memorization")

    result = {
        "ran": True,
        "device": str(device),
        "arch": ARCH_STR,
        "threat_plane_checks": tp_checks,
        "pool": {"n_loss": int((lab < 0).sum()), "n_win": int((lab > 0).sum()),
                 "distinct_games": int(len(np.unique(pool["gid"]))), "plane2_eq_plane3": pool["p23_eq"]},
        "split": {"type": "GAME-LEVEL whole-game disjoint 3-way (train/val/holdout)",
                  "holdout_frac": args.holdout_frac, "val_frac": args.val_frac,
                  "game_disjoint": True, "shared_games": 0, "seeds": args.seeds},
        "opp_bwf_fire_full_pool": {"loss": float(fires[lab < 0].mean()), "win": float(fires[lab > 0].mean())},
        "control_4plane": {"matched_kill_a": round(c_ka, 4), "matched_kill_c": round(c_kc, 4),
                           "train_kill_c": round(c_tkc, 4), "overfit_gap": round(c_overfit, 4),
                           "reproduces_entangled": bool(reproduces_entangled)},
        "treatment_8plane": {"matched_kill_a": round(t_ka, 4), "matched_kill_c": round(t_kc, 4),
                             "train_kill_c": round(t_tkc, 4), "overfit_gap": round(t_overfit, 4)},
        "delta_matched_kill_c": round(delta_kc, 4),
        "per_seed": per_seed,
        "verdict": verdict,
        "reproduces_entangled": bool(reproduces_entangled),
        "verdict_rationale": rationale,
        "multiseed": multiseed_str,
        "overfit_gap": overfit_str,
        "wall_time_s": round(elapsed, 1),
    }

    out_dir = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "E2_REPROBE_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- markdown report ----
    def seed_rows(cond):
        out = []
        for s in per_seed:
            m = s[cond]["matched_holdout"]; tf = s[cond]["matched_trainfit"]
            out.append(f"| {cond} seed={s['seed']} | {m['kill_a']:.3f} | {m['kill_c']:.3f} | "
                       f"{m['n_loss']}/{m['n_win']} | {tf['kill_c']:.3f} |")
        return "\n".join(out)

    md = f"""# D-FULLSPEC E2 RE-PROBE — input-feature ablation (4-plane CONTROL vs 8-plane THREAT TREATMENT)

Generated {time.strftime('%Y-%m-%d %H:%M:%S')}  wall={elapsed:.1f}s  device={device}

Last cheap gate before an expensive bootstrap restart. E1 frozen trunk
(KA 0.762 / KC 0.441) and light-trunk unfreeze (KA 0.653 / KC 0.532) BOTH
ENTANGLE on the 4 base planes -> the value blind-spot is a FEATURE problem.
**This probe asks: do the 4 THREAT planes carry the discriminative signal the 4
base planes lack?**

## Design — apples-to-apples input-feature ablation
- ONE small FROM-SCRATCH conv value-regressor. ONLY conv1.in_channels differs.
- arch: `{ARCH_STR}`
- CONTROL = 4-plane v6_live2 (incl turn-phase planes 2,3). TREATMENT = 8-plane (4 base + 4 threat).
- Identical hyperparams (Adam lr=1e-3, class-balanced value-MSE on +-1, early-stop on game-disjoint VAL, max_epochs={args.max_epochs}).
- Data layer reused verbatim from `scripts/dvderisk_lighttrunk_probe.py` (build_pool / three_way_game_split / matched_masks).
- GAME-DISJOINT 3-way split (whole games -> one side), shared_games==0 asserted per seed. holdout_frac={args.holdout_frac} val_frac={args.val_frac}.
- Verdict on tp==0 INTERSECT overlapping stone-count support (E1 EXACT matched protocol), NEVER naive.

## Threat-plane verification (recomputed at startup)
{json.dumps(tp_checks, indent=0)}
- 5-in-a-row completion cell best_window_fill = 5/6 = 0.833 OK.
- Empirical signal (full pool): opp_best_window_fill>=4/6 fires loss={fires[lab<0].mean():.3f} vs win={fires[lab>0].mean():.3f}.

## Multi-seed matched HOLDOUT KILL-A/KILL-C + overfit canary
KILL-A = frac matched-holdout LOSSES called loss (value<0), PASS>0.35.
KILL-C = frac matched-holdout WINS called win (value>0), PASS>=0.85.

| condition | KILL-A | KILL-C | n_loss/n_win | train-fit KC |
|---|---|---|---|---|
{seed_rows('control')}
{seed_rows('treatment')}
| **CONTROL mean (VERDICT basis)** | **{c_ka:.3f}** | **{c_kc:.3f}** | - | {c_tkc:.3f} |
| **TREATMENT mean (VERDICT basis)** | **{t_ka:.3f}** | **{t_kc:.3f}** | - | {t_tkc:.3f} |

- **delta matched KILL-C (treatment - control) = {delta_kc:+.3f}**
- overfit gap (train-fit KC - holdout KC): control {c_overfit:+.3f}, treatment {t_overfit:+.3f} (>0.20 = memorization).
- reproduces_entangled (control does NOT joint-separate) = **{reproduces_entangled}**.

## VERDICT: {verdict}
{rationale}

### Comparison to prior probes (matched holdout)
| probe | matched KILL-A | matched KILL-C | verdict |
|---|---|---|---|
| E1 FROZEN (4-plane) | 0.762 | 0.441 | ENTANGLED |
| LIGHT-TRUNK (4-plane) | 0.653 | 0.532 | ENTANGLED_LT |
| E2 CONTROL (4-plane, this) | {c_ka:.3f} | {c_kc:.3f} | {'separates(broken)' if control_separates else 'entangled'} |
| E2 TREATMENT (8-plane, this) | {t_ka:.3f} | {t_kc:.3f} | {verdict} |

## self_redteam
- Is the control genuinely entangled (instrument valid)? control matched KA={c_ka:.3f} KC={c_kc:.3f}; reproduces_entangled={reproduces_entangled}. {'Control craters KC<0.85 at KA>0.35 -> instrument VALID.' if reproduces_entangled and c_ka>0.35 else ('Control KA<=0.35 (dead control, weak instrument).' if c_ka<=0.35 else 'Control SEPARATES -> instrument BROKEN, no SEPARABLE allowed.')}
- Is any treatment separation overfit? treatment train-fit KC={t_tkc:.3f} vs holdout {t_kc:.3f} (gap {t_overfit:+.3f}); {'MEMORIZATION (>0.20)' if big_overfit_t else 'no big gap'}.
- Are threat planes computed correctly? {tp_checks} all_pass={tp_ok}.
- Did turn-phase matching hold? plane2==plane3 everywhere={pool['p23_eq']}; matched stratum = tp==0 (planes 2,3==0) INTERSECT stone-support; small matched-win n per seed (effective-n caution).
"""
    with open(out_dir / "E2_REPROBE_findings.md", "w") as f:
        f.write(md)
    log.info(f"JSON -> {out_dir/'E2_REPROBE_results.json'}")
    log.info(f"MD   -> {out_dir/'E2_REPROBE_findings.md'}")
    log.info(f"DONE in {elapsed:.1f}s")
    print("E2_FEATABLATION_RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
