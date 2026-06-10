#!/usr/bin/env python3
"""§D-RECONVERGE Phase 1a — off-window PLACEMENT conversion-lift discriminator (THE GATE).

EVAL-ONLY. Read-only on banked checkpoints + golong self-play replays. NO Rust, NO training.
Reuses ``_ControlDropMCTSBot`` (single-window: off-window priors dropped, production path) and
``KClusterMCTSBot`` (multi-window legal-set: scatter-max over K cluster windows), the SAME model
for both — one switch (the off-window drop). The ablation-control discipline (hold builder skill
constant; differ by one switch; verify via a shared invariant — here the drop instrumentation).

THE QUESTION (the re-opened §D-COHERENCE Branch-C test). §D-GLOBALCONC Phase-2b corrected the
win-cell unit to the COMPLETING cell and re-opened off-window as the LARGER, significant, rising leg
of the ``forced_win_conversion`` decline (shift-share 46/54). Is off-window PLACEMENT (reachability)
the binding constraint? Take self-play forced-win positions whose completing cell is OFF-WINDOW under
the corrected unit (``winning_turn_cells`` + ``is_off_window``). Compare conversion:
  (i)  single-window legal set  — cannot place off-window -> ~0 by construction on the all-off leg.
  (ii) multi-window legal set   — CAN place off-window (picks an off-window move ~9% in S-PRE).
Normalise the multi-window off-window conversion by the model's IN-WINDOW finishing reference R_in
(its intrinsic finishing skill when it CAN reach the cell) -> the recovery fraction isolates the
PLACEMENT constraint from the finding/skill constraint.

PRE-REG verdict (LOCKED in investigation/reconverge_2026-06-08/PREREGISTRATION.md before the run):
  LIFT     : multi recovers a SUBSTANTIAL fraction of R_in via OFF-WINDOW placement
             -> off-window placement is the binding constraint -> Branch C (action space) validated.
  NO-LIFT  : multi ALSO fails to convert (model can't FIND the win even given the action space)
             -> placement is NOT the constraint; the win is unseen -> upstream -> Phase 2.

GUARD (named, not assumed): this is an INFERENCE test on banked checkpoints. It bounds whether
off-window placement HELPS self-play conversion; it does NOT clear the §D-MULTICLUSTER training gates
(S1 >50% fail / S0-362-multiwindow / S3 exploit_probe<=0.06) and it does NOT prove the
self-play -> SealBot-WR (the kill) link. Named as an explicit open gate.

RED-TEAM guard (lift must be REAL off-window placement, not a different IN-WINDOW win on the same
position): the ALL-OFF subleg (every winning cell off-window; control_conv == 0 by construction) is
the primary, and every multi conversion is decomposed by whether the LANDING move was off-window
(``last_move in W_off``). On the all-off leg any conversion REQUIRES an off-window placement.

Run:
  cd $REPO_ROOT && PYTHONPATH=. .venv/bin/python \
    scripts/structural_diagnosis/offwindow_placement_lift.py --dry-pool   # size first
  ... then drop --dry-pool for the full conversion measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.diagnostics.forced_win_detector import (  # noqa: E402
    cheb, is_off_window, window_center, winning_turn_cells,
)
from hexo_rl.encoding import lookup, normalize_encoding_name  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402


# ── pool: off-window vs in-window forced-win turn-starts ─────────────────────────────────
def _binding(win_cells, center):
    """The §D-COHERENCE binding cell = the winning completion farthest (chebyshev) from the
    single-global-window center (the one whose off-window-ness decides the leg)."""
    return max(win_cells, key=lambda c: cheb(c, center))


def build_pool(files, name, spec, want_steps, max_per_leg, seed):
    """Replay golong self-play; at each mover turn-start with a forced win, classify the leg by
    the binding completing cell's off-window flag. Returns (off_rows, in_rows) record lists.

    Each record: {prefix, mover, src, ply, gid, W (sorted), W_off, W_in, all_off, binding_off}.
    A turn is a forced win iff ``winning_turn_cells`` is non-empty (turn-correct, completing cell).
    """
    off_rows, in_rows = [], []
    gid = 0
    n_forced = 0
    for fn in files:
        try:
            fh = open(fn)
        except FileNotFoundError:
            print(f"  (skip missing {fn})", file=sys.stderr)
            continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            src = int(d.get("checkpoint_step", 0))
            if want_steps and src not in want_steps:
                continue
            mv = [(int(q), int(r)) for (q, r) in d["moves"]]
            this_gid = gid
            gid += 1
            board = Board.with_encoding_name(name)
            i, n = 0, len(mv)
            while i < n:
                mover = int(board.current_player)
                snap = board.clone()
                ply = int(board.ply)
                W = winning_turn_cells(snap, mover)
                if W:
                    n_forced += 1
                    center = window_center(
                        [(int(s[0]), int(s[1])) for s in snap.get_stones()])
                    Wl = sorted(W)
                    W_off = [c for c in Wl if is_off_window(snap, c, spec)]
                    W_in = [c for c in Wl if c not in W_off]
                    binding = _binding(Wl, center)
                    binding_off = is_off_window(snap, binding, spec)
                    rec = {
                        "prefix": mv[:i], "mover": mover, "src": src, "ply": ply,
                        "gid": this_gid, "W": Wl, "W_off": W_off, "W_in": W_in,
                        "all_off": (len(W_in) == 0), "binding_off": bool(binding_off),
                        "n_win": len(Wl), "n_off": len(W_off),
                    }
                    (off_rows if binding_off else in_rows).append(rec)
                # advance one full turn (until player flips or a win lands)
                while i < n:
                    q, r = mv[i]
                    try:
                        board.apply_move(q, r)
                    except Exception:
                        i = n
                        break
                    i += 1
                    if board.check_win():
                        break
                    if int(board.current_player) != mover:
                        break
    rng = np.random.default_rng(seed)
    for rows in (off_rows, in_rows):
        rng.shuffle(rows)
    if max_per_leg:
        off_rows = off_rows[:max_per_leg]
        in_rows = in_rows[:max_per_leg]
    return off_rows, in_rows, n_forced


# ── conversion playout (one mover turn under a given bot) ─────────────────────────────────
def rebuild(name, prefix):
    board = Board.with_encoding_name(name)
    state = GameState.from_board(board)
    for (q, r) in prefix:
        state = state.apply_move(board, q, r)
    return board, state


def play_turn(bot, name, prefix, mover, max_stones=3):
    """Play ``mover``'s full turn from ``prefix`` under ``bot``. Returns (converted, last_move)."""
    board, state = rebuild(name, prefix)
    if int(board.current_player) != mover:
        return None
    bot.reset()
    last = None
    stones = 0
    while stones < max_stones:
        if board.check_win() or board.legal_move_count() == 0:
            break
        mvq, mvr = bot.get_move(state, board)
        state = state.apply_move(board, int(mvq), int(mvr))
        last = (int(mvq), int(mvr))
        stones += 1
        if board.check_win():
            break
        if int(board.current_player) != mover:
            break
    converted = bool(board.check_win() and int(board.winner()) == mover)
    return converted, last


# ── bootstrap over games ─────────────────────────────────────────────────────────────────
def boot_ci(vals, gids, nboot, rng, stat=np.mean):
    by_g = defaultdict(list)
    for v, g in zip(vals, gids):
        by_g[g].append(v)
    glist = list(by_g)
    if not glist:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(glist), len(glist))
        sel = []
        for gi in pick:
            sel.extend(by_g[glist[gi]])
        if sel:
            out.append(stat(sel))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def boot_diff_ci(a_vals, a_gids, b_vals, b_gids, nboot, rng):
    """CI of mean(a)-mean(b) resampling games jointly (a=multi, b=control on the SAME positions,
    so games align; resample the shared game set)."""
    by_g_a = defaultdict(list)
    by_g_b = defaultdict(list)
    for v, g in zip(a_vals, a_gids):
        by_g_a[g].append(v)
    for v, g in zip(b_vals, b_gids):
        by_g_b[g].append(v)
    glist = sorted(set(by_g_a) | set(by_g_b))
    if not glist:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(glist), len(glist))
        sa, sb = [], []
        for gi in pick:
            g = glist[gi]
            sa.extend(by_g_a.get(g, []))
            sb.extend(by_g_b.get(g, []))
        if sa and sb:
            out.append(np.mean(sa) - np.mean(sb))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--replays", nargs="+",
                    default=sorted(str(p) for p in (REPO / "investigation/coherence_2026-06-08/replays").glob("games_2026-06-0*.jsonl")))
    ap.add_argument("--ckpts", nargs="+", default=[
        "investigation/coherence_2026-06-08/checkpoints/checkpoint_00030000.pt",
        "checkpoints/golong_bank/checkpoint_00050000_PEAK_sb0.38.pt",
        "investigation/fragility_2026-06-07/checkpoints/checkpoint_00087500.pt",
    ])
    ap.add_argument("--ckpt-labels", nargs="+", default=["30k", "50k_PEAK", "87.5k"])
    ap.add_argument("--pool-steps", type=int, nargs="+", default=[30000, 53000, 87500])
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--max-per-leg", type=int, default=160)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--dry-pool", action="store_true", help="size the pool and exit (no GPU)")
    ap.add_argument("--out", default="investigation/reconverge_2026-06-08/offwindow_placement_lift.json")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    print(f"[cfg] encoding={name} policy_logit_count={spec.policy_logit_count} "
          f"trunk={spec.trunk_size} kept={list(spec.kept_plane_indices)}", flush=True)
    print(f"[cfg] replays={len(args.replays)} files  pool_steps={args.pool_steps}", flush=True)

    t0 = time.time()
    off_rows, in_rows, n_forced = build_pool(
        args.replays, name, spec, set(args.pool_steps), args.max_per_leg, args.seed)
    print(f"[pool] forced-win turn-starts total={n_forced}  "
          f"OFF-leg(binding off-window)={len(off_rows)}  IN-leg={len(in_rows)}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    if off_rows:
        all_off = [r for r in off_rows if r["all_off"]]
        print(f"[pool] OFF-leg: all_off(strict, every winner off-window)={len(all_off)}  "
              f"binding-only(has in-window winner too)={len(off_rows)-len(all_off)}", flush=True)
        print(f"[pool] OFF-leg by src={dict(sorted(Counter(r['src'] for r in off_rows).items()))}  "
              f"mean n_win={np.mean([r['n_win'] for r in off_rows]):.2f} "
              f"mean n_off={np.mean([r['n_off'] for r in off_rows]):.2f}", flush=True)
    if in_rows:
        print(f"[pool] IN-leg by src={dict(sorted(Counter(r['src'] for r in in_rows).items()))}",
              flush=True)
    if args.dry_pool:
        print("[dry-pool] sized only; exiting before GPU work.")
        return 0

    import torch  # noqa: E402  (defer the heavy import past --dry-pool)
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot  # noqa: E402
    from scripts.multicluster_s174_precheck import _ControlDropMCTSBot  # noqa: E402
    from hexo_rl.utils.device import best_device  # noqa: E402

    device = best_device()
    kept = list(spec.kept_plane_indices)
    n_actions = spec.policy_logit_count
    rng = np.random.default_rng(args.seed)

    out = {
        "encoding": name, "sims": args.sims, "pool_steps": args.pool_steps,
        "n_forced_total": n_forced, "n_off_leg": len(off_rows), "n_in_leg": len(in_rows),
        "n_all_off": sum(1 for r in off_rows if r["all_off"]),
        "prereg": "investigation/reconverge_2026-06-08/PREREGISTRATION.md",
        "checkpoints": [],
    }

    for ckpt_path, lab in zip(args.ckpts, args.ckpt_labels):
        cp = REPO / ckpt_path if not Path(ckpt_path).is_absolute() else Path(ckpt_path)
        model, _, label = load_model_with_encoding(cp, device)
        model.eval()
        control = _ControlDropMCTSBot(model, device, n_sims=args.sims, temperature=0.0,
                                      n_actions=n_actions, kept_plane_indices=kept)
        multi = KClusterMCTSBot(model, device, n_sims=args.sims, temperature=0.0,
                                kept_plane_indices=kept)
        print(f"\n[ckpt {lab}] {cp.name} enc={label} device={device}", flush=True)

        # IN-leg reference: production single-window finishing skill (control), the normaliser R_in.
        rin_v, rin_g = [], []
        for r in in_rows:
            res = play_turn(control, name, r["prefix"], r["mover"])
            if res is None:
                continue
            rin_v.append(int(res[0]))
            rin_g.append(r["gid"])
        R_in = float(np.mean(rin_v)) if rin_v else float("nan")
        rin_ci = boot_ci(rin_v, rin_g, args.nboot, rng)

        # OFF-leg: control (expect ~0 on all-off) vs multi; decompose multi by off-window placement.
        c_v, c_g, m_v, m_g = [], [], [], []
        m_place_off = []          # among OFF-leg positions multi converted, was the landing off-window
        ao_c_v, ao_m_v, ao_g = [], [], []   # all-off subleg
        for r in off_rows:
            W_off_set = {tuple(c) for c in r["W_off"]}
            rc = play_turn(control, name, r["prefix"], r["mover"])
            rm = play_turn(multi, name, r["prefix"], r["mover"])
            if rc is None or rm is None:
                continue
            cconv, _ = rc
            mconv, mlast = rm
            c_v.append(int(cconv)); c_g.append(r["gid"])
            m_v.append(int(mconv)); m_g.append(r["gid"])
            if mconv and mlast is not None:
                m_place_off.append(int(tuple(mlast) in W_off_set))
            if r["all_off"]:
                ao_c_v.append(int(cconv)); ao_m_v.append(int(mconv)); ao_g.append(r["gid"])

        def _m(v):
            return float(np.mean(v)) if v else float("nan")

        off_control = _m(c_v)
        off_multi = _m(m_v)
        lift = off_multi - off_control
        lift_ci = boot_diff_ci(m_v, m_g, c_v, c_g, args.nboot, rng)
        recovery = (off_multi / R_in) if (R_in and R_in == R_in and R_in > 0) else float("nan")
        place_frac = _m(m_place_off)
        ao = {
            "n": len(ao_c_v), "control_conv": _m(ao_c_v), "multi_conv": _m(ao_m_v),
            "multi_ci": boot_ci(ao_m_v, ao_g, args.nboot, rng),
        }
        rec = {
            "label": lab, "ckpt": cp.name,
            "R_in_finishing_ref": R_in, "R_in_ci": list(rin_ci), "n_in_used": len(rin_v),
            "off_control_conv": off_control, "off_multi_conv": off_multi,
            "off_multi_ci": list(boot_ci(m_v, m_g, args.nboot, rng)),
            "lift": lift, "lift_ci": list(lift_ci),
            "recovery_frac_of_Rin": recovery,
            "off_placement_frac": place_frac,        # of multi conversions, share via off-window cell
            "n_off_used": len(c_v),
            "all_off": ao,
            "control_drop_diag": {
                "expansions": control.expansions,
                "expansions_k_gt1": control.expansions_k_gt1,
                "expansions_with_drop": control.expansions_with_drop,
                "total_priors_dropped": control.total_priors_dropped,
                "dropped_all_turns": control.dropped_all_turns,
                "max_k": control.max_k,
            },
        }
        out["checkpoints"].append(rec)
        print(f"  R_in(in-window finishing, control) = {R_in:.3f} CI[{rin_ci[0]:.3f},{rin_ci[1]:.3f}] "
              f"(n={len(rin_v)})", flush=True)
        print(f"  OFF-leg control={off_control:.3f}  multi={off_multi:.3f}  "
              f"lift={lift:+.3f} CI[{lift_ci[0]:+.3f},{lift_ci[1]:+.3f}]  "
              f"recovery(multi/R_in)={recovery:.2f}  off-placement-frac={place_frac:.2f}", flush=True)
        print(f"  ALL-OFF subleg n={ao['n']}: control={ao['control_conv']:.3f} (expect 0) "
              f"multi={ao['multi_conv']:.3f} CI[{ao['multi_ci'][0]:.3f},{ao['multi_ci'][1]:.3f}]",
              flush=True)
        print(f"  [drop-diag] expansions={control.expansions} k>1={control.expansions_k_gt1} "
              f"with_drop={control.expansions_with_drop} dropped={control.total_priors_dropped} "
              f"all-dropped-turns={control.dropped_all_turns} max_k={control.max_k}", flush=True)
        del model, control, multi
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── pooled verdict across checkpoints (pre-registered rule) ──────────────────────────
    recs = out["checkpoints"]
    pooled_recovery = float(np.nanmean([r["recovery_frac_of_Rin"] for r in recs]))
    pooled_lift = float(np.nanmean([r["lift"] for r in recs]))
    pooled_place = float(np.nanmean([r["off_placement_frac"] for r in recs]))
    pooled_ao_multi = float(np.nanmean([r["all_off"]["multi_conv"] for r in recs
                                        if r["all_off"]["n"] > 0])) if any(
        r["all_off"]["n"] > 0 for r in recs) else float("nan")
    # LIFT if multi recovers >=0.50 of R_in AND every-ckpt lift CI lower bound >0 AND placement majority
    all_lift_pos = all(r["lift_ci"][0] > 0 for r in recs)
    if pooled_recovery >= 0.50 and all_lift_pos and pooled_place >= 0.5:
        verdict = "LIFT"
    elif pooled_recovery <= 0.25 or not any(r["lift_ci"][0] > 0 for r in recs):
        verdict = "NO-LIFT"
    else:
        verdict = "AMBIGUOUS"
    out["pooled"] = {
        "recovery_frac_of_Rin": pooled_recovery, "lift": pooled_lift,
        "off_placement_frac": pooled_place, "all_off_multi_conv": pooled_ao_multi,
        "all_ckpt_lift_ci_lower_positive": all_lift_pos,
    }
    out["verdict"] = verdict
    print(f"\n{'='*100}")
    print(f"VERDICT: {verdict}  (pre-reg: LIFT if recovery>=0.50 & every-ckpt lift CI>0 & "
          f"placement>=0.5; NO-LIFT if recovery<=0.25 or no ckpt lift CI>0)")
    print(f"  pooled recovery(multi/R_in)={pooled_recovery:.2f}  lift={pooled_lift:+.3f}  "
          f"off-placement-frac={pooled_place:.2f}  all-off multi_conv={pooled_ao_multi:.3f}")
    print(f"  GUARD: inference-only — does NOT clear §D-MULTICLUSTER S0/S1/S3 gates, does NOT "
          f"prove the self-play -> SealBot-WR link.")
    print(f"{'='*100}")

    Path(REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(REPO / args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
