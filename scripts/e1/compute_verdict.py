"""E1 frozen verdict — REVIVE / CONFIRM-DEMOTE per docs/designs/e1_metric_freeze.md §5.

Verdict points = +5k/+10k/+20k/+50k into E1 = grid ckpts 255k/260k/270k/298k
(250k=+2k warm-start snapshot EXCLUDED). gap = scalar_M - dist_M (positive = dist better).

REVIVE = ( gap_M1 positive Theil-Sen slope over the 4 pts AND 298k gap_M1 bootstrap-CI
lower-bound > 0 ) OR ( same for gap_M2 ), with the OTHER co-primary's 298k gap >= 0 AND the
M4 guard (dist_FP <= scalar_FP + 0.05) held at all 4 pts. Else CONFIRM-DEMOTE.

CI = position-level paired bootstrap, 10k resamples. Slope = Theil-Sen over the 4 points.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from scripts.e1 import validate_ckpt as V
from scripts.valprobe.value_health import compute_ece

VERDICT_STEPS = [255000, 260000, 270000, 298000]
SCALAR_298 = "/tmp/e1_ckpts/scalar_00298000.pt"
DIST_298 = "/tmp/e1_ckpts/dist_00298000.pt"
SERIES = str(REPO / "reports/e1/t7_series.jsonl")
N_BOOT = 10000
SEED = 12345
FP_THRESH = -0.5
FP_SLACK = 0.05

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def theil_sen(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    sl = [(ys[j] - ys[i]) / (xs[j] - xs[i]) for i in range(len(xs)) for j in range(i + 1, len(xs))]
    return float(np.median(sl))


def score_arm(ckpt, arm, loss_rows, safe_rows, ri, wp2):
    model, spec, _ = V._load_net(ckpt, arm, dev)
    ls, nsl = V._score_positions(model, spec, dev, loss_rows, ri, wp2)
    ss, nss = V._score_positions(model, spec, dev, safe_rows, ri, wp2)
    return np.array([s["v"] for s in ls], float), np.array([s["v"] for s in ss], float), nsl, nss


def ece(v, y):  # y in {-1 loss, +1 safe/win}
    return compute_ece(list(v), list(y))


def main():
    rows = [json.loads(l) for l in open(SERIES)]
    def get(a, s, k):
        m = [r for r in rows if r["arm"] == a and r["step"] == s]
        return m[-1][k] if m else None

    loss_rows = V._load_rows(V.DEFAULT_PROBE)
    safe_rows = V._load_rows(V.DEFAULT_NEGATIVES)
    ri, wp2 = V._build_game_indices(V.DEFAULT_GAMES, V._default_wp2_games())

    # per-position v at 298k, both arms (paired: same rows/order)
    sv_loss, sv_safe, a, b = score_arm(SCALAR_298, "scalar", loss_rows, safe_rows, ri, wp2)
    dv_loss, dv_safe, c, d = score_arm(DIST_298, "dist65", loss_rows, safe_rows, ri, wp2)
    assert a == c == 0 and b == d == 0, f"skips: scalar(loss={a},safe={b}) dist(loss={c},safe={d})"
    assert len(sv_loss) == len(dv_loss) and len(sv_safe) == len(dv_safe)
    nL, nS = len(sv_loss), len(sv_safe)

    # 298k aggregates
    M1_s, M1_d = sv_loss.mean(), dv_loss.mean()
    yL, yS = np.full(nL, -1.0), np.full(nS, 1.0)
    ECE_s = ece(np.concatenate([sv_loss, sv_safe]), np.concatenate([yL, yS]))
    ECE_d = ece(np.concatenate([dv_loss, dv_safe]), np.concatenate([yL, yS]))
    FP_s = float((sv_safe <= FP_THRESH).mean())
    FP_d = float((dv_safe <= FP_THRESH).mean())
    gapM1_298 = float(M1_s - M1_d)
    gapM2_298 = float(ECE_s - ECE_d)

    # append 298k dist to series record (M1/M2/M4)
    with open(SERIES, "a") as f:
        f.write(json.dumps({"step": 298000, "arm": "dist65", "mean_v_on_losses": float(M1_d),
                            "ece": float(ECE_d), "false_pessimism": FP_d, "n_loss": nL, "n_safe": nS,
                            "note": "verdict-recompute"}) + "\n")

    # gaps at the 4 verdict points (255/260/270 from series, 298 fresh)
    gM1, gM2, m4 = [], [], []
    for s in VERDICT_STEPS:
        if s == 298000:
            g1, g2, fs, fd = gapM1_298, gapM2_298, FP_s, FP_d
        else:
            g1 = get("scalar", s, "mean_v_on_losses") - get("dist65", s, "mean_v_on_losses")
            g2 = get("scalar", s, "ece") - get("dist65", s, "ece")
            fs, fd = get("scalar", s, "false_pessimism"), get("dist65", s, "false_pessimism")
        gM1.append(g1); gM2.append(g2); m4.append(fd <= fs + FP_SLACK)

    slopeM1 = theil_sen(VERDICT_STEPS, gM1)
    slopeM2 = theil_sen(VERDICT_STEPS, gM2)

    # position-level paired bootstrap of the 298k gaps
    rng = np.random.default_rng(SEED)
    bM1 = np.empty(N_BOOT); bM2 = np.empty(N_BOOT)
    union_s = np.concatenate([sv_loss, sv_safe]); union_d = np.concatenate([dv_loss, dv_safe])
    yU = np.concatenate([yL, yS]); nU = nL + nS
    for i in range(N_BOOT):
        li = rng.integers(0, nL, nL)
        bM1[i] = sv_loss[li].mean() - dv_loss[li].mean()
        ui = rng.integers(0, nU, nU)
        bM2[i] = ece(union_s[ui], yU[ui]) - ece(union_d[ui], yU[ui])
    ciM1 = (float(np.percentile(bM1, 2.5)), float(np.percentile(bM1, 97.5)))
    ciM2 = (float(np.percentile(bM2, 2.5)), float(np.percentile(bM2, 97.5)))

    m4_held = all(m4)
    revive_M1 = (slopeM1 > 0) and (ciM1[0] > 0) and (gapM2_298 >= 0) and m4_held
    revive_M2 = (slopeM2 > 0) and (ciM2[0] > 0) and (gapM1_298 >= 0) and m4_held
    verdict = "REVIVE" if (revive_M1 or revive_M2) else "CONFIRM-DEMOTE"

    out = {
        "verdict": verdict, "revive_via_M1": revive_M1, "revive_via_M2": revive_M2,
        "verdict_steps": VERDICT_STEPS,
        "gap_M1": [round(x, 4) for x in gM1], "gap_M2": [round(x, 4) for x in gM2],
        "theil_sen_slope_M1_per_step": slopeM1, "theil_sen_slope_M2_per_step": slopeM2,
        "gap_M1_298k": round(gapM1_298, 4), "gap_M1_298k_CI95": [round(x, 4) for x in ciM1],
        "gap_M2_298k": round(gapM2_298, 4), "gap_M2_298k_CI95": [round(x, 4) for x in ciM2],
        "M4_held_all4": m4_held, "M4_per_pt_FP_scalar_dist": None,
        "raw_298k": {"M1_scalar": round(float(M1_s), 4), "M1_dist": round(float(M1_d), 4),
                     "ECE_scalar": round(float(ECE_s), 4), "ECE_dist": round(float(ECE_d), 4),
                     "FP_scalar": FP_s, "FP_dist": FP_d, "n_loss": nL, "n_safe": nS},
    }
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
