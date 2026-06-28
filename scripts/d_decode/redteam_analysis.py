#!/usr/bin/env python3
"""D-DECODE red-team analysis: premise attacks on d_solver_offwindow data.

Runs three attacks:
1. Byte-identical arm check (exploit vs control)
2. Effective-n for deploy 0.335
3. Kcluster DEFENDS vs AVOIDS the off-window forcing band

Usage: .venv/bin/python scripts/d_decode/redteam_analysis.py
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "reports" / "d_solver_offwindow"
OUT_DIR = REPO / "reports" / "d_decode"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fingerprint(r: dict) -> str:
    return json.dumps({k: v for k, v in r.items() if k != "arm"}, sort_keys=True)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(center - margin, 4), round(center + margin, 4)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def main() -> None:
    deploy = load_jsonl(DATA_DIR / "run1.jsonl")
    kcluster = load_jsonl(DATA_DIR / "ref_kcluster.jsonl")
    mp = load_jsonl(DATA_DIR / "ref_modelplayer.jsonl")

    deploy_e = [r for r in deploy if r["arm"] == "exploit"]
    deploy_c = [r for r in deploy if r["arm"] == "control"]
    k_e = [r for r in kcluster if r["arm"] == "exploit"]
    k_c = [r for r in kcluster if r["arm"] == "control"]
    mp_e = [r for r in mp if r["arm"] == "exploit"]
    mp_c = [r for r in mp if r["arm"] == "control"]

    # === Attack 1: Byte-identical arms ===
    deploy_ids = [fingerprint(e) == fingerprint(c) for e, c in zip(deploy_e, deploy_c)]
    k_ids = [fingerprint(e) == fingerprint(c) for e, c in zip(k_e, k_c)]
    mp_diffs = sum(1 for e, c in zip(mp_e, mp_c) if fingerprint(e) != fingerprint(c))

    print("=== ATTACK 1: Byte-identical arms ===")
    print(f"Deploy all-identical: {all(deploy_ids)}")
    print(f"KCluster all-identical: {all(k_ids)}")
    print(f"ModelPlayer diffs: {mp_diffs}/200")
    print()

    # === Attack 2: Effective-n ===
    deploy_distinct = len(set(fingerprint(r) for r in deploy_e))
    deploy_owfw = sum(r["off_window_win"] for r in deploy_e)
    ci = wilson_ci(deploy_owfw, deploy_distinct)
    print("=== ATTACK 2: Effective-n ===")
    print(f"Deploy distinct: {deploy_distinct}/200, off_window_wins: {deploy_owfw}")
    print(f"Rate: {deploy_owfw/deploy_distinct:.4f}, Wilson 95% CI: {ci}")
    print()

    # === Attack 3: kcluster DEFENDS vs AVOIDS ===
    k_any = [r for r in k_e if r["any_offwindow_forcing_position"]]
    k_no_any = [r for r in k_e if not r["any_offwindow_forcing_position"]]
    d_any = [r for r in deploy_e if r["any_offwindow_forcing_position"]]
    mp_c_any = [r for r in mp_c if r["any_offwindow_forcing_position"]]

    deploy_win_games = {r["game"] for r in deploy_e if r["adversary_won"]}
    k_forcing_games = {r["game"] for r in k_any}

    print("=== ATTACK 3: kcluster DEFENDS vs AVOIDS ===")
    print(f"any_offwindow_forcing_position_rate:")
    print(f"  kcluster: {len(k_any)}/200 = {len(k_any)/200:.3f}")
    print(f"  deploy:   {len(d_any)}/200 = {len(d_any)/200:.3f}")
    print(f"  mp-ctrl:  {len(mp_c_any)}/200 = {len(mp_c_any)/200:.3f}")
    print(f"Kcluster forcing games: survived={sum(not r['adversary_won'] for r in k_any)}, lost={sum(r['adversary_won'] for r in k_any)}")
    print(f"Deploy forcing games: survived={sum(not r['adversary_won'] for r in d_any)}, lost={sum(r['adversary_won'] for r in d_any)}")
    print(f"Games deploy loses AND kcluster also sees forcing: {len(deploy_win_games & k_forcing_games)}")
    print(f"Games deploy loses AND kcluster avoids forcing band: {len(deploy_win_games - k_forcing_games)}")
    print()

    # Save JSON
    out = {
        "attack1_byte_identical": {
            "deploy": all(deploy_ids),
            "kcluster": all(k_ids),
            "modelplayer_diffs": mp_diffs,
            "mechanism": (
                "adversary _is_off uses CURRENT board (after model's move); "
                "probe is_off_window uses MODEL SNAPSHOT (before model's move). "
                "Model's defensive stones shift bbox centroid toward adversary tendril, "
                "making off-window win appear in-window to adversary. "
                "Control arm then takes the 'in-window' (by current centroid) win that "
                "the probe classifies as off-window via the snapshot centroid. "
                "ModelPlayer avoids this because its single-window policy can't see "
                "the off-window region, so it plays elsewhere, keeping centroid centered."
            ),
        },
        "attack2_effective_n": {
            "deploy_distinct": deploy_distinct,
            "deploy_total": len(deploy_e),
            "off_window_wins": deploy_owfw,
            "rate": round(deploy_owfw / deploy_distinct, 4),
            "wilson_ci_95": ci,
            "robust": ci[0] > 0.05,
        },
        "attack3_kcluster": {
            "kcluster_any_forcing_rate": len(k_any) / 200,
            "kcluster_forcing_survived": sum(not r["adversary_won"] for r in k_any),
            "kcluster_forcing_lost": sum(r["adversary_won"] for r in k_any),
            "deploy_any_forcing_rate": len(d_any) / 200,
            "deploy_forcing_lost": sum(r["adversary_won"] for r in d_any),
            "overlap_deploy_lose_kcluster_see_forcing": len(deploy_win_games & k_forcing_games),
            "deploy_lose_kcluster_avoids_band": len(deploy_win_games - k_forcing_games),
            "verdict": (
                "MIXED: kcluster DEFENDS all 33 games where forcing band is reached "
                "(0/33 losses). But kcluster reaches the band in only 33/200 games vs "
                "deploy's 67/200, meaning kcluster also AVOIDS the band in 34 extra "
                "games that deploy fails. The lower band-reach rate reflects kcluster's "
                "no-drop policy (can see+disrupt the adversary's off-window tendril "
                "earlier) plus possible centroid-shift artifact (kcluster defensive moves "
                "shift centroid toward adversary, making adversary's plan appear 'in-window' "
                "and the adversary may abandon it). Dominant interpretation: DEFENDS + "
                "early prevention, but partial avoidance cannot be ruled out."
            ),
        },
    }
    out_path = OUT_DIR / "redteam_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
