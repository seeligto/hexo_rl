#!/usr/bin/env python3
"""§D-VALCEIL Q1/Q2 analyses over the per-row prediction dumps (eval-only, CPU):
(a) SANITY GATE: recompute aggregate sign_acc@50k per bank + livetail per-phase
    cells from the dumps; must equal the banked §D-VALPROBE JSONs to 4dp;
(b) Q1 CEIL (pre-registered): per-phase (per-bank occupancy terciles, the
    ladder's tercile_masks — the registered binning) sign_acc + value_mse for
    livetail@50k vs corpus@50k, gaps, game-bootstrap 95% CIs (2000 resamples,
    distinct game_ids, seed 20260611; corpus dump has NO game_ids -> per-row
    bootstrap, caveat stated in output), registered verdict:
    CEIL-INTRINSIC  iff late livetail >= late corpus - 0.05
    CEIL-HEADROOM   iff late livetail <  late corpus - 0.10, else AMBIGUOUS;
(c) Q1 across-rungs: per-phase livetail sign_acc at all 6 rungs + saturation
    read (delta per rung; 10k->20k / 20k->50k / 10k->50k game-bootstrap CIs);
(d) Q2 KSTRAT (pre-registered): occ-stratum (same terciles, boundaries pinned
    to the banked Phase-1 JSONs) x K-bin (K1/K2/K3/K4+ per-row k_counts)
    sign_acc cells on livetail/uniform/occmatched @50k (+10k informational),
    adequacy floor n_rows>=100 AND n_games>=30, DEFF per cell, registered
    verdict: REOPENED iff ANY adequate 50k cell has
    sign_acc(K>=2) < sign_acc(K1 same stratum) - 0.10, else BANKED;
(e) open item 4 re-read: within-occ-stratum spread-COMPS terciles at tip on
    livetail via the FIXED stratified_tercile_masks kernel (the banked read
    degenerated under the tercile-padding artifact).

Inputs are the §D-VALCEIL per-row dumps (z/occ/comps/preds_<step> per bank,
+ game_ids/plies/k_counts where the source bank had them), produced on vast
from the exact ladder slices (verified bit-identical to the banked JSONs).
No model forward here — pure numpy over the dumps (tercile kernels imported
from value_calibration_ladder for bit-fidelity; that pulls torch in
transitively but no tensor is ever created).

Example:
  python scripts/diagnosis/valceil_analysis.py \
    --dumps-dir tmp/valceil/dumps --banked-dir tmp/valprobe \
    --out tmp/valceil/valceil_analysis.json \
    --out-audit audit/structural/valceil_analysis.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.value_calibration_ladder import (  # noqa: E402
    stratified_tercile_masks,
    tercile_masks,
)

# ── registered constants (fixed before reading any cell) ────────────────────
DEFAULTS = {
    "dumps_dir": "tmp/valceil/dumps",
    "banked_dir": "tmp/valprobe",
    "out": "tmp/valceil/valceil_analysis.json",
    "out_audit": "audit/structural/valceil_analysis.json",
    "draw_band": 0.10,        # identical to value_calibration_ladder DEFAULTS
    "n_boot": 2000,
    "seed": 20260611,
}

RUNGS = (0, 10000, 20000, 30000, 40000, 50000)
PHASES = ("early", "mid", "late")
KBINS = ("K1", "K2", "K3", "K4+")

# bank -> banked §D-VALPROBE ladder JSON (sanity-gate targets + pinned terciles)
BANKED_JSON = {
    "corpus": "value_calibration_ladder_repro.json",
    "uniform": "value_calibration_ladder_selfplay.json",
    "occmatched": "value_calibration_ladder_selfplay_occmatched.json",
    "livetail": "value_calibration_ladder_livetail.json",
}

CEIL_THRESHOLDS = {"intrinsic": -0.05, "headroom": -0.10}   # late gap vs corpus
KSTRAT_DEFICIT = -0.10                                       # K>=2 vs K1 gap
ADEQUACY = {"n_rows": 100, "n_games": 30}
SANITY_TOL = 5e-5                                            # "equal to 4dp"


# ── dump loading ─────────────────────────────────────────────────────────────


def load_dump(dumps_dir: pathlib.Path, bank: str) -> Dict[str, np.ndarray]:
    d = np.load(dumps_dir / f"{bank}.npz")
    out = {k: d[k] for k in d.files}
    for r in RUNGS:
        assert f"preds_{r}" in out, f"{bank}: missing preds_{r}"
    return out


# ── masked-cell kernels (sign_acc masking identical to phase_metrics) ────────


def cell_sign_acc(corr: np.ndarray, decided: np.ndarray, m: np.ndarray) -> float:
    """sign_acc over decided rows of a cell — same masking as
    value_calibration_ladder.phase_metrics (dec = |z| >= draw_band within m)."""
    mm = m & decided
    return float(corr[mm].mean()) if mm.any() else float("nan")


def cell_mse(serr: np.ndarray, m: np.ndarray) -> float:
    return float(serr[m].mean()) if m.any() else float("nan")


def ci95(samples: np.ndarray) -> List[float]:
    return [round(float(q), 4) for q in np.nanquantile(samples, [0.025, 0.975])]


# ── generic cluster bootstrap over precomputed per-row stats ─────────────────


class StatSpec:
    """One bootstrap statistic: mean of values[rows] over mask[rows]."""

    def __init__(self, name: str, values: np.ndarray, mask: np.ndarray):
        self.name = name
        self.values = values.astype(np.float64)
        self.mask = mask.astype(bool)

    def point(self) -> float:
        return float(self.values[self.mask].mean()) if self.mask.any() else float("nan")


def bootstrap(
    stats: List[StatSpec],
    n_rows: int,
    n_boot: int,
    seed: int,
    game_ids: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Cluster bootstrap: resample DISTINCT game_ids with replacement (the
    §D-ARGMAX effective-n lesson — CI per-game, not per-row). game_ids=None
    -> per-row bootstrap (corpus dump lacks game_ids; rows of a 392k-row
    corpus are near-independent; livetail clustering dominates the gap CI)."""
    rng = np.random.default_rng(seed)
    if game_ids is not None:
        games = np.unique(game_ids)
        rows_by_game = {g: np.flatnonzero(game_ids == g) for g in games}
    out = {s.name: np.empty(n_boot) for s in stats}
    for i in range(n_boot):
        if game_ids is not None:
            sample = rng.choice(games, size=len(games), replace=True)
            rows = np.concatenate([rows_by_game[g] for g in sample])
        else:
            rows = rng.choice(n_rows, size=n_rows, replace=True)
        for s in stats:
            m = s.mask[rows]
            out[s.name][i] = s.values[rows][m].mean() if m.any() else np.nan
    return out


# ── per-bank precomputation ──────────────────────────────────────────────────


def prep_bank(dump: Dict[str, np.ndarray], draw_band: float) -> Dict[str, object]:
    z = dump["z"].astype(np.float64)
    occ = dump["occ"]
    decided = np.abs(z) >= draw_band
    phase = tercile_masks(occ)  # the registered binning — per-bank terciles
    corr = {r: (np.sign(dump[f"preds_{r}"]) == np.sign(z)).astype(np.float64)
            for r in RUNGS}
    serr = {r: (dump[f"preds_{r}"].astype(np.float64) - z) ** 2 for r in RUNGS}
    b: Dict[str, object] = {
        "z": z, "occ": occ, "decided": decided, "phase": phase,
        "corr": corr, "serr": serr, "n": len(z),
        "occ_terciles": [float(q) for q in np.quantile(occ, [1 / 3, 2 / 3])],
        "game_ids": dump.get("game_ids"),
        "k_counts": dump.get("k_counts"),
        "comps": dump.get("comps"),
    }
    if b["k_counts"] is not None:
        k = np.clip(b["k_counts"], 1, 4)
        b["kmask"] = {"K1": k == 1, "K2": k == 2, "K3": k == 3, "K4+": k == 4}
    return b


def n_games_of(game_ids: Optional[np.ndarray], m: np.ndarray) -> Optional[int]:
    return int(len(np.unique(game_ids[m]))) if game_ids is not None else None


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="§D-VALCEIL CEIL + KSTRAT analysis")
    ap.add_argument("--dumps-dir", default=DEFAULTS["dumps_dir"])
    ap.add_argument("--banked-dir", default=DEFAULTS["banked_dir"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--out-audit", default=DEFAULTS["out_audit"])
    ap.add_argument("--draw-band", type=float, default=DEFAULTS["draw_band"])
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = ap.parse_args()

    dumps_dir = ROOT / args.dumps_dir
    banked_dir = ROOT / args.banked_dir
    banks = {name: prep_bank(load_dump(dumps_dir, name), args.draw_band)
             for name in BANKED_JSON}
    banked = {name: json.loads((banked_dir / fn).read_text())
              for name, fn in BANKED_JSON.items()}

    def banked_rung(name: str, step: int) -> dict:
        return next(e for e in banked[name]["ladder"] if e["step"] == step)

    # ── (a) sanity gate: dump-recomputed cells == banked JSON to 4dp ────────
    gate = {"tol": SANITY_TOL, "cells": {}, "pass": True}
    for name, b in banks.items():
        agg = cell_sign_acc(b["corr"][50000], b["decided"],
                            np.ones(b["n"], dtype=bool))
        tgt = banked_rung(name, 50000)["metrics"]["sign_acc"]
        ok = abs(agg - tgt) < SANITY_TOL
        gate["cells"][f"{name}_agg_sign_acc_50k"] = {
            "recomputed": round(agg, 6), "banked": tgt, "match": ok}
        gate["pass"] &= ok
        # pinned occ-tercile boundaries must equal the banked Phase-1 quantiles
        bq = banked[name]["bank_spread_stats"]["occupancy_quantiles_0_33_50_66_100"]
        ok_b = (b["occ_terciles"][0] == bq[1]) and (b["occ_terciles"][1] == bq[3])
        gate["cells"][f"{name}_occ_terciles"] = {
            "recomputed": b["occ_terciles"], "banked": [bq[1], bq[3]], "match": ok_b}
        gate["pass"] &= ok_b
    lt = banks["livetail"]
    for ph in PHASES:
        v = cell_sign_acc(lt["corr"][50000], lt["decided"], lt["phase"][ph])
        tgt = banked_rung("livetail", 50000)["phase"][ph]["sign_acc"]
        ok = abs(v - tgt) < SANITY_TOL
        gate["cells"][f"livetail_phase_{ph}_sign_acc_50k"] = {
            "recomputed": round(v, 6), "banked": tgt, "match": ok}
        gate["pass"] &= ok
    if not gate["pass"]:
        print(json.dumps(gate, indent=1))
        sys.exit("SANITY GATE FAIL — dumps do not reproduce the banked JSONs")
    print(f"[gate] PASS — {len(gate['cells'])} cells match banked to 4dp")

    # ── bootstrap stat manifests per bank ────────────────────────────────────
    boots: Dict[str, Dict[str, np.ndarray]] = {}
    for name, b in banks.items():
        stats: List[StatSpec] = []
        for ph in PHASES:
            pm = b["phase"][ph]
            for r in RUNGS:
                stats.append(StatSpec(f"phase_{ph}_sign_{r}", b["corr"][r],
                                      pm & b["decided"]))
                stats.append(StatSpec(f"phase_{ph}_mse_{r}", b["serr"][r], pm))
            if b["k_counts"] is not None:
                for kb in KBINS:
                    cm = pm & b["kmask"][kb]
                    for r in (10000, 50000):
                        stats.append(StatSpec(f"k_{ph}_{kb}_sign_{r}",
                                              b["corr"][r], cm & b["decided"]))
        if name == "livetail":  # open item 4: spread-within-occ at tip
            for ph in PHASES:
                for tn, tm in stratified_tercile_masks(
                        b["comps"], b["phase"][ph]).items():
                    stats.append(StatSpec(f"spread_{ph}_{tn}_sign_50000",
                                          b["corr"][50000], tm & b["decided"]))
        boots[name] = bootstrap(stats, b["n"], args.n_boot, args.seed,
                                b["game_ids"])
        print(f"[boot] {name}: {len(stats)} stats x {args.n_boot} resamples "
              f"({'game' if b['game_ids'] is not None else 'row'}-cluster)")

    # ── (b) Phase-1 CEIL table + gaps + verdict ──────────────────────────────
    corpus_caveat = (
        "corpus dump lacks game_ids (source npz has none) -> corpus side is a "
        "PER-ROW bootstrap; rows subsampled from a 392,251-row corpus are "
        "near-independent, livetail game-clustering dominates the gap CI")
    table: Dict[str, dict] = {}
    for name in ("livetail", "corpus"):
        b = banks[name]
        table[name] = {}
        for ph in PHASES:
            pm = b["phase"][ph]
            table[name][ph] = {
                "sign_acc": round(cell_sign_acc(b["corr"][50000], b["decided"], pm), 4),
                "value_mse": round(cell_mse(b["serr"][50000], pm), 4),
                "n_rows": int(pm.sum()),
                "n_games": n_games_of(b["game_ids"], pm),
                "sign_acc_ci95": ci95(boots[name][f"phase_{ph}_sign_50000"]),
                "value_mse_ci95": ci95(boots[name][f"phase_{ph}_mse_50000"]),
            }
    gaps = {}
    for ph in PHASES:
        d_sign = table["livetail"][ph]["sign_acc"] - table["corpus"][ph]["sign_acc"]
        d_mse = table["livetail"][ph]["value_mse"] - table["corpus"][ph]["value_mse"]
        gaps[ph] = {
            "sign_acc_gap": round(d_sign, 4),
            "sign_acc_gap_ci95": ci95(
                boots["livetail"][f"phase_{ph}_sign_50000"]
                - boots["corpus"][f"phase_{ph}_sign_50000"]),
            "value_mse_gap": round(d_mse, 4),
            "value_mse_gap_ci95": ci95(
                boots["livetail"][f"phase_{ph}_mse_50000"]
                - boots["corpus"][f"phase_{ph}_mse_50000"]),
        }
    late_gap, mid_gap = gaps["late"]["sign_acc_gap"], gaps["mid"]["sign_acc_gap"]
    if late_gap >= CEIL_THRESHOLDS["intrinsic"]:
        ceil_verdict = "CEIL-INTRINSIC"
    elif late_gap < CEIL_THRESHOLDS["headroom"]:
        ceil_verdict = "CEIL-HEADROOM"
    else:
        ceil_verdict = "AMBIGUOUS"
    lg_ci = gaps["late"]["sign_acc_gap_ci95"]
    ci_note = {
        "late_gap_ci_vs_minus0.05": "entirely below" if lg_ci[1] < -0.05 else
        ("entirely above" if lg_ci[0] >= -0.05 else "straddles"),
        "late_gap_ci_vs_minus0.10": "entirely below" if lg_ci[1] < -0.10 else
        ("entirely above" if lg_ci[0] >= -0.10 else "straddles"),
    }

    # ── (c) across-rungs livetail per-phase + saturation ─────────────────────
    across: Dict[str, dict] = {}
    for ph in PHASES:
        pm = lt["phase"][ph]
        per_rung = {str(r): round(cell_sign_acc(lt["corr"][r], lt["decided"], pm), 4)
                    for r in RUNGS}
        deltas = {f"{a//1000}k->{b_//1000}k": round(per_rung[str(b_)] - per_rung[str(a)], 4)
                  for a, b_ in zip(RUNGS[:-1], RUNGS[1:])}
        bb = boots["livetail"]
        across[ph] = {
            "sign_acc_per_rung": per_rung,
            "delta_per_rung": deltas,
            "d_10k_20k": {"point": round(per_rung["20000"] - per_rung["10000"], 4),
                          "ci95": ci95(bb[f"phase_{ph}_sign_20000"]
                                       - bb[f"phase_{ph}_sign_10000"])},
            "d_20k_50k": {"point": round(per_rung["50000"] - per_rung["20000"], 4),
                          "ci95": ci95(bb[f"phase_{ph}_sign_50000"]
                                       - bb[f"phase_{ph}_sign_20000"])},
            "d_10k_50k": {"point": round(per_rung["50000"] - per_rung["10000"], 4),
                          "ci95": ci95(bb[f"phase_{ph}_sign_50000"]
                                       - bb[f"phase_{ph}_sign_10000"])},
        }

    # supplementary: uniform + occmatched per-phase @50k (context only)
    supplementary = {}
    for name in ("uniform", "occmatched"):
        b = banks[name]
        supplementary[name] = {ph: {
            "sign_acc": round(cell_sign_acc(b["corr"][50000], b["decided"],
                                            b["phase"][ph]), 4),
            "value_mse": round(cell_mse(b["serr"][50000], b["phase"][ph]), 4),
            "n_rows": int(b["phase"][ph].sum()),
            "sign_acc_ci95": ci95(boots[name][f"phase_{ph}_sign_50000"]),
        } for ph in PHASES}

    # ── (d) Phase-2 KSTRAT cells + verdict ───────────────────────────────────
    kstrat: Dict[str, dict] = {"corpus": "SKIPPED — source corpus npz has no "
                               "k_counts (verified in dump manifest)"}
    deficits: List[dict] = []
    deffs: Dict[str, List[float]] = {}
    for name in ("livetail", "uniform", "occmatched"):
        b = banks[name]
        kstrat[name] = {}
        deffs[name] = []
        for r in (50000, 10000):
            rows = {}
            for ph in PHASES:
                rows[ph] = {}
                for kb in KBINS:
                    cm = b["phase"][ph] & b["kmask"][kb]
                    nd = int((cm & b["decided"]).sum())
                    ng = n_games_of(b["game_ids"], cm)
                    p = cell_sign_acc(b["corr"][r], b["decided"], cm)
                    bs = boots[name][f"k_{ph}_{kb}_sign_{r}"]
                    var_b = float(np.nanvar(bs))
                    deff = (min(nd * var_b / (p * (1 - p)), float(nd))
                            if p == p and 0.0 < p < 1.0 and nd > 0 else float("nan"))
                    adequate = bool(cm.sum() >= ADEQUACY["n_rows"]
                                    and ng is not None
                                    and ng >= ADEQUACY["n_games"])
                    rows[ph][kb] = {
                        "sign_acc": round(p, 4) if p == p else None,
                        "n_rows": int(cm.sum()), "n_games": ng,
                        "sign_acc_ci95": ci95(bs), "deff": round(deff, 2)
                        if deff == deff else None, "adequate": adequate,
                    }
                    if r == 50000 and adequate and deff == deff:
                        deffs[name].append(deff)
                # registered verdict reads: K>=2 vs K1, both cells adequate, 50k
                if r == 50000 and rows[ph]["K1"]["adequate"]:
                    for kb in ("K2", "K3", "K4+"):
                        c = rows[ph][kb]
                        if not c["adequate"] or c["sign_acc"] is None:
                            continue
                        d = c["sign_acc"] - rows[ph]["K1"]["sign_acc"]
                        dci = ci95(boots[name][f"k_{ph}_{kb}_sign_50000"]
                                   - boots[name][f"k_{ph}_K1_sign_50000"])
                        deficits.append({"bank": name, "stratum": ph, "kbin": kb,
                                         "gap_vs_K1": round(d, 4), "gap_ci95": dci})
            kstrat[name][str(r)] = rows
    worst = min(deficits, key=lambda d: d["gap_vs_K1"])
    kstrat_verdict = ("REOPENED" if worst["gap_vs_K1"] < KSTRAT_DEFICIT
                      else "BANKED")
    deff_summary = {name: {"n_adequate_cells_50k": len(v),
                           "median": round(float(np.median(v)), 2),
                           "max": round(float(np.max(v)), 2)}
                    for name, v in deffs.items() if v}

    # ── (e) open item 4: spread-within-occ terciles at tip (fixed kernel) ────
    spread = {}
    for ph in PHASES:
        spread[ph] = {}
        masks = stratified_tercile_masks(banks["livetail"]["comps"],
                                         lt["phase"][ph])
        for tn, tm in masks.items():
            spread[ph][tn] = {
                "sign_acc": round(cell_sign_acc(lt["corr"][50000],
                                                lt["decided"], tm), 4),
                "n_rows": int(tm.sum()),
                "n_games": n_games_of(lt["game_ids"], tm),
                "comps_range": ([int(banks["livetail"]["comps"][tm].min()),
                                 int(banks["livetail"]["comps"][tm].max())]
                                if tm.any() else None),
                "sign_acc_ci95": ci95(boots["livetail"][f"spread_{ph}_{tn}_sign_50000"]),
            }
        g = (spread[ph]["late"]["sign_acc"] - spread[ph]["early"]["sign_acc"]
             if spread[ph]["late"]["sign_acc"] is not None
             and spread[ph]["early"]["sign_acc"] is not None else None)
        gci = ci95(boots["livetail"][f"spread_{ph}_late_sign_50000"]
                   - boots["livetail"][f"spread_{ph}_early_sign_50000"])
        spread[ph]["most_minus_least"] = {
            "gap": round(g, 4) if g is not None else None, "gap_ci95": gci,
            "ci_excludes_zero": (gci[0] > 0 or gci[1] < 0)}

    report = {
        "script": "scripts/diagnosis/valceil_analysis.py",
        "mandate": "§D-VALCEIL Q1 CEIL + Q2 KSTRAT over §D-VALPROBE per-row dumps",
        "seed": args.seed, "n_boot": args.n_boot, "draw_band": args.draw_band,
        "sanity_gate": gate,
        "phase1_ceil": {
            "binning": "per-bank occupancy terciles via tercile_masks(occ) — "
                       "registered, identical to the corpus probe",
            "occ_tercile_boundaries": {n: banks[n]["occ_terciles"] for n in banks},
            "table_50k": table,
            "gaps_livetail_minus_corpus_50k": gaps,
            "corpus_bootstrap_caveat": corpus_caveat,
            "verdict": {
                "rule": "CEIL-INTRINSIC iff late_gap >= -0.05; CEIL-HEADROOM "
                        "iff late_gap < -0.10; else AMBIGUOUS (point estimates)",
                "late_gap": late_gap, "mid_gap": mid_gap,
                "verdict": ceil_verdict,
                "ci_robustness_note": ci_note,
            },
            "across_rungs_livetail": across,
            "supplementary_50k": supplementary,
        },
        "phase2_kstrat": {
            "strata": "same per-bank occupancy terciles as Phase 1 (boundaries "
                      "pinned to banked Phase-1 JSONs, asserted in sanity gate)",
            "adequacy_floor": ADEQUACY,
            "cells": kstrat,
            "adequate_K_vs_K1_gaps_50k": deficits,
            "worst_adequate_deficit": worst,
            "verdict": {
                "rule": f"REOPENED iff any adequate 50k cell K>=2 gap < "
                        f"{KSTRAT_DEFICIT} vs same-stratum K1; else BANKED",
                "verdict": kstrat_verdict,
            },
            "deff_summary": deff_summary,
        },
        "open_item4_spread_within_occ_livetail_50k": {
            "kernel": "stratified_tercile_masks (fixed — in-stratum quantiles; "
                      "banked read degenerated under -1 padding)",
            "cells": spread,
        },
    }
    for out_path in (args.out, args.out_audit):
        op = ROOT / out_path
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(report, indent=1))
        print(f"[wrote] {op}")
    print(f"\n[Q1 CEIL] late_gap={late_gap} mid_gap={mid_gap} -> {ceil_verdict}")
    print(f"[Q2 KSTRAT] worst adequate deficit {worst} -> {kstrat_verdict}")


if __name__ == "__main__":
    main()
