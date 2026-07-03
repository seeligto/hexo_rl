#!/usr/bin/env python
"""D-SOLVER A1 — deploy solver-backup WR gate (PAIRED).

Measures whether wiring a deploy-time tactical solver-backup (``SolverBackupBot``) lifts WR
vs a fixed-depth SealBot. Arms:
  * baseline   — DeployHeadBot (Gumbel-SH, g=0, no temp, deploy sims) alone.
  * backup_dN  — SolverBackupBot(DeployHeadBot, depth=N): at each turn start a depth-N SealBot
                 root probe; on a PROVEN mate, override the model with the proven move.

PAIRED design (the load-bearing fix): all arms share the SAME RNG-seeded openings and the
same deterministic g=0 head + fixed-depth opponent, so a per-seed game is byte-identical
across arms UNTIL the backup's first override. The A1 statistic is the PAIRED bootstrap of
(backup - baseline) per-seed delta: every non-firing game contributes an exact 0, collapsing
the CI onto the fired games (the §D-TACTICAL short tactical band). ``n_fired`` (games the
backup changed) is the true power unit. The opponent is a FRESH fixed-depth SealBot per game
(pure function of position — fixed depth + cold TT) so pairing holds on non-fired turns.

MATCHED-DEPTH control: pass ``--backup-depths 5,6`` (with ``--sealbot-depth 5``). If the
lift persists at backup_d5 (== opponent depth), the net is missing mates WITHIN the
opponent's own horizon → trap-fix, not a generic d6>d5 depth edge.

PRE-REGISTERED (D-SOLVER A1): per backup arm —
  LIFT_IN_WINDOW          — paired delta CI lower bound > 0 AND n_fired >= --min-fired.
  FLAT                    — CI straddles/under 0 AND n_fired >= --min-fired.
  INDETERMINATE_UNDERPOWERED — n_fired < --min-fired (too few overrides to resolve).
LIFT is a PARTIAL lift (short band only) and is NECESSARY-NOT-SUFFICIENT: the override is
sound (terminal mate only) but SealBot is an in-window FLOOR (38% in-corpus fail, colony-OOB
guarded), so a fixed-bot WR lift CANNOT clear an off-window defect (CLAUDE.md off-window
false-clear gate). Phase B is gated on A1 LIFT_IN_WINDOW *and* a separate spread-uncapped /
adversarial probe — do not let this verdict alone green-light the native build.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from hexo_rl.bots.sealbot_bot import SealBotBot
from hexo_rl.eval.a1_stats import cand_outcome, dedup_distinct, paired_delta, soundness_violations
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot,
    _build_engine_for_model,
    _play_one_game,
    extract_deploy_knobs,
)
from hexo_rl.eval.solver_backup_bot import SolverBackupBot


def _build_cand(arm: str, engine, knobs, backup_depth: int, seed: int, window_half,
                legal_set: bool = False, probe_engine: str = "sealbot"):
    # §D-RECONFIRM R1: legal_set threads the MULTI-WINDOW no-drop decode into BOTH arms'
    # deploy head (baseline + the solver-backup's inner). The original A1 (+0.165) ran
    # legal_set=False = the SINGLE-window handicapped instrument; this re-measures the
    # paired delta on the corrected multi-window head the net was trained under.
    head = DeployHeadBot(engine, knobs, label=arm, seed=seed, legal_set=legal_set)
    if arm == "baseline":
        return head
    # §D-ZVALID Z1: probe_engine="native" routes the backup through engine::tactics (sound,
    # spread-immune, no SealBot dep; capped at threat-forcing reach until the quiet-move body
    # lands). The native solver's own window_half guard handles off-window, so the Python
    # window_half/colony guards are disabled on that path (see SolverBackupBot).
    return SolverBackupBot(head, depth=backup_depth, window_half=window_half,
                           probe_engine=probe_engine)  # arm == backup_dN


def _play_arm(
    cand,
    cand_label: str,
    sealbot_depth: int,
    encoding: str,
    n_games: int,
    opening_plies: int,
    seed_base: int,
) -> List[Dict[str, Any]]:
    """Color-balanced cand-vs-SealBot games. FRESH opponent per game (pure function of
    position) + cand.reset() per game. Seeds shared across arms (paired)."""
    is_backup = isinstance(cand, SolverBackupBot)
    games: List[Dict[str, Any]] = []
    for gi in range(n_games):
        if hasattr(cand, "reset"):
            cand.reset()
        opp = SealBotBot(time_limit=60.0, max_depth=sealbot_depth)  # fixed-depth, cold TT per game
        before = (cand.fired_win, cand.fired_loss, cand.skipped_colony) if is_backup else None
        if gi % 2 == 0:
            p1, p2, l1, l2 = cand, opp, cand_label, "sealbot"
        else:
            p1, p2, l1, l2 = opp, cand, "sealbot", cand_label
        seed = seed_base + gi
        random.seed(seed)
        np.random.seed(seed % (2**31))
        g = _play_one_game(p1, p2, l1, l2, encoding, opening_plies=opening_plies, seed=seed)
        g["arm"] = cand_label
        g["seed"] = seed  # paired join key
        if is_backup:
            after = (cand.fired_win, cand.fired_loss, cand.skipped_colony)
            g["fired_win"] = after[0] - before[0]
            g["fired_loss"] = after[1] - before[1]
            g["skipped_colony"] = after[2] - before[2]
            g["cand_won"] = (g["winner"] == "p1" and g["p1"] == cand_label) or (
                g["winner"] == "p2" and g["p2"] == cand_label
            )
        games.append(g)
    return games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="deploy net .pt")
    ap.add_argument("--encoding", default="v6_live2_ls", help="board/engine encoding (deploy)")
    ap.add_argument("--n-games", type=int, default=200, help="games PER ARM (paired by seed)")
    ap.add_argument("--sealbot-depth", type=int, default=5, help="opponent SealBot fixed depth")
    ap.add_argument("--backup-depths", default="6",
                    help="comma list of backup probe depths; include the opponent depth for the matched control")
    ap.add_argument("--legal-set", action="store_true",
                    help="§D-RECONFIRM R1: route BOTH arms' deploy head through the MULTI-WINDOW "
                         "no-drop legal-set decode (the corrected, trained-under instrument). "
                         "Default off = single-window — the handicapped instrument A1's +0.165 "
                         "was measured on. Set this to re-measure LIFT-HOLDS vs LIFT-SHRINKS.")
    ap.add_argument("--window-half", type=int, default=9,
                    help="off-window guard: SealBot proofs of moves >cheb this from the stone "
                         "bbox center are untrusted (9 = v6_live2_ls window; 0/neg disables). "
                         "For --probe-engine native this is passed to the solver's OWN guard.")
    ap.add_argument("--probe-engine", default="sealbot", choices=["sealbot", "native"],
                    help="D-ZVALID Z1: 'native' routes the backup through engine::tactics "
                         "(sound + spread-immune, no SealBot dep; capped at threat-forcing reach "
                         "until the quiet-move body lands). 'sealbot' = interim vendored probe.")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-fired", type=int, default=10, help="power floor: n_fired below this = INDETERMINATE")
    ap.add_argument("--seed-base", type=int, default=20260627)
    ap.add_argument("--n-sims-full", type=int, default=0,
                    help="§D-RECONFIRM R2: override the deploy n_sims_full for ALL checkpoints "
                         "(0 = use each ckpt's embedded knob). The d1m lineage embeds 100 sims at "
                         "95k-200k but 150 at 272357 — fix this to ONE value (150 = the D-LADDER "
                         "Gumbel@150 deploy regime) for a same-instrument cross-checkpoint WR ladder.")
    ap.add_argument("--gumbel-m", type=int, default=0,
                    help="override the deploy gumbel_m for ALL checkpoints (0 = embedded knob).")
    ap.add_argument("--out", default="reports/d_solver_A1", help="output dir")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_depths = [int(d) for d in str(args.backup_depths).split(",") if str(d).strip()]

    # D-EVALGATE fix wave: --encoding (the deploy board/engine encoding, usually
    # v6_live2_ls) is threaded as decode_override — this script routinely
    # cross-decodes stale/single-window-stamped checkpoints (see the module
    # docstring's d1m lineage caveats), so a disagreeing stamp must log loudly,
    # never raise.
    model, _spec, auto_label = load_model_with_encoding(
        args.checkpoint, device, decode_override=args.encoding,
    )
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    # §D-RECONFIRM R2: fix the deploy search budget across a cross-checkpoint lineage so the WR
    # ladder is same-instrument (the d1m lineage embeds 100 sims at 95k-200k vs 150 at 272357).
    knob_overrides = {}
    if args.n_sims_full:
        knob_overrides["n_sims_full"] = (knobs.get("n_sims_full"), args.n_sims_full)
        knobs["n_sims_full"] = args.n_sims_full
    if args.gumbel_m:
        knob_overrides["gumbel_m"] = (knobs.get("gumbel_m"), args.gumbel_m)
        knobs["gumbel_m"] = args.gumbel_m
    if knob_overrides:
        print(f"[A1] KNOB OVERRIDE (same-instrument): {knob_overrides}", flush=True)
    encoding = args.encoding
    engine = _build_engine_for_model(model, encoding, device)
    print(f"[A1] ckpt={Path(args.checkpoint).name} auto_enc={auto_label} board_enc={encoding} "
          f"device={device} knobs={knobs} sealbot_d{args.sealbot_depth} backup_d{backup_depths} "
          f"n={args.n_games}/arm PAIRED decode={'MULTI-WINDOW(legal_set)' if args.legal_set else 'single-window'}",
          flush=True)

    arm_labels = ["baseline"] + [f"backup_d{d}" for d in backup_depths]
    games_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    arm_results: Dict[str, Dict[str, Any]] = {}
    all_games: List[Dict[str, Any]] = []

    for arm in arm_labels:
        depth = int(arm.split("_d")[1]) if arm != "baseline" else 0
        wh = args.window_half if args.window_half and args.window_half > 0 else None
        cand = _build_cand(arm, engine, knobs, depth, seed=args.seed_base, window_half=wh,
                           legal_set=args.legal_set, probe_engine=args.probe_engine)  # same seed: g=0 deterministic
        t0 = time.time()
        games = _play_arm(cand, arm, args.sealbot_depth, encoding, args.n_games,
                          args.opening_plies, args.seed_base)  # SAME seed_base => paired
        dt = time.time() - t0
        scores = [cand_outcome(g, arm) for g in games]
        distinct = dedup_distinct(games, arm)
        head_frac = sum(1 for g in games if g.get("head_fired")) / max(len(games), 1)
        res: Dict[str, Any] = {
            "arm": arm, "wr_raw": float(np.mean(scores)) if scores else 0.0,
            "wr_distinct": float(np.mean(distinct)) if distinct else 0.0,
            "n_raw": len(scores), "n_distinct": len(distinct),
            "head_fired_frac": head_frac, "secs": round(dt, 1),
        }
        if isinstance(cand, SolverBackupBot):
            res["backup"] = {"fired_win": cand.fired_win, "fired_loss": cand.fired_loss,
                             "skipped_colony": cand.skipped_colony,
                             "skipped_offwindow": cand.skipped_offwindow, "probes": cand.probes}
        arm_results[arm] = res
        games_by_arm[arm] = games
        all_games.extend(games)
        print(f"[A1] arm={arm} wr={res['wr_raw']:.3f} (distinct {res['wr_distinct']:.3f}, "
              f"n_distinct={res['n_distinct']}/{res['n_raw']}) head={head_frac:.2f} "
              f"{res.get('backup','')} {dt:.0f}s", flush=True)

    # ── Paired delta + verdict per backup arm ────────────────────────────────────────
    deltas: Dict[str, Any] = {}
    for arm in arm_labels:
        if arm == "baseline":
            continue
        boot = paired_delta(games_by_arm["baseline"], games_by_arm[arm], "baseline", arm,
                            args.n_boot, args.seed_base)
        viol = soundness_violations(games_by_arm[arm], arm)
        boot["soundness_violations"] = len(viol)
        if boot["n_fired"] < args.min_fired:
            verdict = "INDETERMINATE_UNDERPOWERED"
        elif boot["ci_lo"] > 0.0:
            verdict = "LIFT_IN_WINDOW"
        else:
            verdict = "FLAT"
        boot["verdict"] = verdict
        boot["matched_depth"] = (int(arm.split("_d")[1]) == args.sealbot_depth)
        deltas[arm] = boot
        tag = " [MATCHED-DEPTH control]" if boot["matched_depth"] else ""
        print(f"[A1] {arm} vs baseline{tag}: delta={boot['delta']:+.3f} "
              f"paired-bootstrap 95% CI [{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}] "
              f"n_fired={boot['n_fired']}/{boot['n_paired']} P(>0)={boot['p_gt_0']:.3f} "
              f"-> {verdict}", flush=True)
        if viol:
            print(f"[A1][!!!] UNSOUND: {len(viol)} {arm} games fired a proven win but did NOT win "
                  f"(loss or draw) — FALSE PROOF, investigate. seeds={viol[:10]}", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint), "encoding": encoding, "knobs": knobs,
        "legal_set": bool(args.legal_set),  # §D-RECONFIRM R1: multi-window(True) vs single-window(False) decode
        "knob_overrides": {k: {"embedded": v[0], "forced": v[1]} for k, v in knob_overrides.items()},
        "deploy_knobs": knobs,  # the ACTUAL knobs used (post-override) — same-instrument provenance for R2
        "sealbot_depth": args.sealbot_depth, "backup_depths": backup_depths,
        "n_games_per_arm": args.n_games, "opening_plies": args.opening_plies,
        "min_fired": args.min_fired, "design": "paired", "arms": arm_results, "deltas": deltas,
        "off_window_caveat": "LIFT_IN_WINDOW is necessary-not-sufficient; Phase B also requires a "
                             "separate spread-uncapped/adversarial off-window probe (SealBot is in-window).",
    }
    with open(out_dir / "per_game.jsonl", "w") as f:
        for g in all_games:
            f.write(json.dumps(g) + "\n")
    with open(out_dir / "A1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[A1] wrote {out_dir}/A1_summary.json + per_game.jsonl  (reports/ gitignored — copy to persist)",
          flush=True)


if __name__ == "__main__":
    main()
