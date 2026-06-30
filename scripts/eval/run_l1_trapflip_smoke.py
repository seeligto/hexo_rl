#!/usr/bin/env python
"""D-WS3 W1 — the L1 solver-in-loop GENERALIZES/MEMORIZES gate (move-level trap-flip).

The PRE-REGISTERED headline metric of the GPU-day smoke. Decides whether the L1
solver-in-loop fine-tune taught the POLICY to propose the saving move STANDALONE
(GENERALIZES, held-out trap-flip >=25%) or capped at the ~16% memorization floor
(MEMORIZES -> deploy-backup permanent for the class). Spec:
docs/designs/coupled_valuez_decode_design.md §0/§5 (Stage-1A); runbook:
docs/handoffs/d_ws3_l1_smoke_runbook.md.

WHAT IT MEASURES (the dpfit-validated move-level deploy-flip, P1b anchors:
control 12% / single-window held-out 16% / in-sample 31%): for each GAME-DISJOINT
held-out trap, replay the PARENT board (net to-move, about to blunder) and run the
DEPLOY search (g=0, gumbel_m=16, n_sims=150, multi-window legal_set decode); the
trap "flips" iff the search plays the solver's saving move (refuting_move). Flip
RATE = fraction "saving", measured for the BASELINE (pre-finetune anchor) and the
CANDIDATE (post-finetune) net. Standalone — NO solver/backup at eval time (Z3
discipline: never re-measure the crutch).

PRE-REGISTERED VERDICT (dispatcher W1):
  * GENERALIZES — candidate held-out flip >= --pass (0.25) AND > baseline flip
    (a real margin over the 16% single-window / 12% control floor) AND the KILL
    co-gate clears -> the net internalised the lever beyond memorization -> the
    GPU-week (full training-z) is justified.
  * MEMORIZES   — candidate held-out flip <= --floor (0.16): no lift over the
    memorization floor -> the decode-recovers-stranded-priors wager is false ->
    deploy-backup is PERMANENT for the class. Saves the GPU-week.
  * KILL>16%    — normal-position policy corrupted by the injection
    (deploy-disagree vs baseline on a held-out NORMAL set > --kill 0.16) ->
    soften the injection (lower solver_visit_weight) and re-smoke before any
    verdict. The soft/one-hot boundary is the knob.
  * INDETERMINATE — candidate flip in (floor, pass): mixed; thin power (31 traps)
    -> do not over-read; expand the corpus or run longer.

CO-GATES (run SEPARATELY per the runbook; this script reminds, does NOT run them):
  * off-window forced rate must HOLD 0.0 (no-drop floor): exploit_probe.py
    --defender deploy --legal-set --adv-ref current (absolute rate; the contrast
    is contaminated on centroid-shifting bots).
  * ModelPlayer false-clear cross-check (arm-aliasing-immune): exploit_probe.py
    --defender modelplayer.
  * threat-probe C1-C3 PASS: scripts/probe_threat_logits.py on the candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import engine  # noqa: E402
from scripts.eval.gumbel_greedy_bot import _build_engine  # noqa: E402
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board  # noqa: E402

# Deploy knobs — the regime the net actually deploys under (deploy_strength_eval /
# §D-LADDER). g=0 (gumbel_scale=0.0) => deterministic head.
DEPLOY = dict(n_sims=150, m=16, c_visit=50.0, c_scale=1.0, c_puct=1.5,
              dirichlet=False, gumbel_scale=0.0)

DEFAULT_ANCHOR = "reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt"
DEFAULT_TRAPS = "reports/d_tactical_2026-06-26/heldout_traps.jsonl"


def replay(seq, encoding):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def classify(played, saving, blunder) -> str:
    if played is None:
        return "none"
    pl = (int(played[0]), int(played[1]))
    if pl == (int(saving[0]), int(saving[1])):
        return "saving"
    if pl == (int(blunder[0]), int(blunder[1])):
        return "blunder"
    return "other"


def deploy_played(eng, board, legal_set: bool):
    # g=0 (gumbel_scale=0.0) => deterministic argmax head, so a fresh rng(0) per call
    # is sound. If gumbel_scale were >0 this fixed seed would correlate every board's
    # Gumbel draw and collapse the effective-n (red-team F5) — guard it.
    assert DEPLOY["gumbel_scale"] == 0.0, "deploy_played assumes a deterministic g=0 head"
    g = run_gumbel_on_board(eng, board, **DEPLOY, legal_set=legal_set,
                            rng=np.random.default_rng(0))
    return g["played_move"]


def flip_rates(eng, traps, encoding, legal_set: bool):
    """Per-trap classification + flip rate, split in/off-window."""
    cls = []
    for t in traps:
        seq = t.get("parent_move_seq")
        if seq is None:
            print(f"[l1] WARN trap {t.get('pos_id')!r} has no parent_move_seq; skipping", flush=True)
            continue
        board = replay(seq, t.get("encoding", encoding))
        played = deploy_played(eng, board, legal_set)
        cls.append((t, classify(played, t["saving_move"], t["blunder_move"])))
    n = len(cls)
    flip = sum(c == "saving" for _, c in cls)
    inw = [(t, c) for t, c in cls if t.get("in_window", True)]
    off = [(t, c) for t, c in cls if not t.get("in_window", True)]
    return {
        "n": n,
        "flip_rate": flip / n if n else 0.0,
        "flip": flip,
        "blunder_rate": sum(c == "blunder" for _, c in cls) / n if n else 0.0,
        "in_window": {"n": len(inw), "flip": sum(c == "saving" for _, c in inw),
                      "flip_rate": (sum(c == "saving" for _, c in inw) / len(inw)) if inw else None},
        "off_window": {"n": len(off), "flip": sum(c == "saving" for _, c in off),
                       "flip_rate": (sum(c == "saving" for _, c in off) / len(off)) if off else None},
        "per_trap": [{"pos_id": t["pos_id"], "in_window": t.get("in_window", True), "cls": c}
                     for t, c in cls],
    }


def normal_boards(traps, encoding, lo=16, stride=8, cap_per_game=2, cap_total=60):
    """Held-out NORMAL (non-trap) positions for the KILL co-gate. Red-team F1: drawing
    these from the LATE-game prefixes adjacent to the blunder leaks the trap's tactical
    context — the candidate's legitimate localized improvement there reads as
    "corruption" (false KILL). Two defenses:
      (1) sample only the EARLY-MIDGAME (plies `lo`..L//2) — structurally distant from
          the late-game blunder tail;
      (2) keep only QUIET positions: skip any board where the side-to-move has a
          within-turn forced win (cheap `forced_win_move`) OR the native solver proves
          a forced win (`window_half=None`, modest budget) — those ARE tactical/trap-
          like, not normal. A genuinely localized trap-lever must NOT shift quiet,
          non-tactical moves; deploy-disagree HERE is honest collateral corruption.
    Still same-game prefixes (no separate normal-game corpus exists), but quiet +
    early so the leakage path is closed. (The symmetric-corruption FALSE-CLEAR — a
    candidate that shifts every move to a same-argmax-nearby cell scores disagree 0 —
    is NOT caught here; the authoritative normal-play canary is threat-probe C1-C3,
    run separately per the runbook.)"""
    try:
        import engine as _engine
        solver = _engine.TacticalSolver(window_half=None, cand_cap=40)
    except Exception:  # noqa: BLE001 — solver filter is best-effort; fall back to forced_win_move only
        solver = None
    seen = set()
    boards = []
    for t in traps:
        seq = t.get("parent_move_seq")
        if seq is None:
            continue
        L = len(seq)
        hi = max(lo + 1, L // 2)  # first half only — away from the trap tail
        got = 0
        for cut in range(lo, hi, stride):
            sub = seq[:cut]
            key = tuple((int(q), int(r)) for q, r in sub)
            if key in seen:
                continue
            seen.add(key)
            b = replay(sub, t.get("encoding", encoding))
            if b.check_win() or b.legal_move_count() == 0:
                continue
            if b.forced_win_move(2) is not None:  # cheap: within-turn tactical -> not quiet
                continue
            if solver is not None and solver.prove(b, 8, 10000)[0] == 1:  # deep forced win -> tactical
                continue
            boards.append(b)
            got += 1
            if got >= cap_per_game or len(boards) >= cap_total:
                break
        if len(boards) >= cap_total:
            break
    return boards


def kill_gate(base_eng, cand_eng, boards, legal_set: bool):
    """Normal-position policy-corruption: rate at which the candidate's deploy move
    DIFFERS from the baseline's on held-out normal positions. High = the injection
    bled into normal play (corruption)."""
    if not boards:
        return {"n": 0, "deploy_disagree_rate": None}
    disagree = 0
    for b in boards:
        pr = deploy_played(base_eng, b, legal_set)
        pf = deploy_played(cand_eng, b, legal_set)
        if pr is not None and pf is not None and tuple(pr) != tuple(pf):
            disagree += 1
    n = len(boards)
    return {"n": n, "deploy_disagree": disagree, "deploy_disagree_rate": disagree / n}


def decide(cand_flip, base_flip, kill_rate, pass_thr, floor_thr, kill_thr):
    # KILL co-gate (normal-play corruption) dominates a positive flip — but ONLY when
    # it was actually MEASURED. A None rate (no quiet normal positions found) means the
    # co-gate did not run, so it cannot CLEAR the pre-reg requirement.
    if kill_rate is not None and kill_rate > kill_thr:
        return "KILL>16%"
    # GENERALIZES = clears the absolute pass bar AND lifts over the SAME-DECODE baseline
    # (F4: the static 16% floor was calibrated single-window; under multi-window decode
    # the baseline may already be >16%, so require a real lift over the measured base,
    # not just over the static floor).
    would_generalize = cand_flip >= pass_thr and cand_flip > base_flip
    if would_generalize:
        if kill_rate is None:
            return "INDETERMINATE_KILL_UNRUN"  # pre-reg: the KILL co-gate must CLEAR
        return "GENERALIZES"
    # No meaningful lift over the same-decode baseline (or below the floor) = no learning.
    if cand_flip <= floor_thr or cand_flip <= base_flip:
        return "MEMORIZES"
    return "INDETERMINATE"  # partial lift, below pass — thin power / starved-recall; do not over-read


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-ckpt", default=DEFAULT_ANCHOR,
                    help="pre-finetune anchor (default checkpoint_00200000)")
    ap.add_argument("--candidate-ckpt", required=True, help="post-finetune net (the L1 candidate)")
    ap.add_argument("--encoding", default="v6_live2_ls",
                    help="MUST be v6_live2_ls — the .pt auto-detects as v6_live2 (policy_pool none); "
                         "pass the _ls form to resolve the trained multi-window action space")
    ap.add_argument("--trap-set", default=DEFAULT_TRAPS)
    ap.add_argument("--legal-set", dest="legal_set", action="store_true", default=True,
                    help="multi-window no-drop decode (the L1 regime; DEFAULT)")
    ap.add_argument("--no-legal-set", dest="legal_set", action="store_false",
                    help="single-window decode (P1b contrast; off-window saves un-expressible)")
    ap.add_argument("--pass", dest="pass_thr", type=float, default=0.25)
    ap.add_argument("--floor", dest="floor_thr", type=float, default=0.16)
    ap.add_argument("--kill", dest="kill_thr", type=float, default=0.16)
    ap.add_argument("--normal-cap", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="reports/d_zvalid_z2/l1_trapflip")
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    traps = [json.loads(l) for l in open(args.trap_set) if l.strip()]
    if not traps:
        raise SystemExit(f"no traps in {args.trap_set} — run scripts/dpfit_export_heldout_traps.py first")
    print(f"[l1] {len(traps)} held-out traps "
          f"({sum(t.get('in_window', True) for t in traps)} in-window), legal_set={args.legal_set}")

    base_eng = _build_engine(args.baseline_ckpt, args.encoding, device)
    cand_eng = _build_engine(args.candidate_ckpt, args.encoding, device)

    base = flip_rates(base_eng, traps, args.encoding, args.legal_set)
    cand = flip_rates(cand_eng, traps, args.encoding, args.legal_set)
    nb = normal_boards(traps, args.encoding, cap_total=args.normal_cap)
    kill = kill_gate(base_eng, cand_eng, nb, args.legal_set)

    verdict = decide(cand["flip_rate"], base["flip_rate"], kill["deploy_disagree_rate"],
                     args.pass_thr, args.floor_thr, args.kill_thr)

    summary = {
        "verdict": verdict,
        "baseline_ckpt": args.baseline_ckpt,
        "candidate_ckpt": args.candidate_ckpt,
        "legal_set": args.legal_set,
        "thresholds": {"pass": args.pass_thr, "floor": args.floor_thr, "kill": args.kill_thr},
        "baseline_flip": base,
        "candidate_flip": cand,
        "delta_flip": cand["flip_rate"] - base["flip_rate"],
        "kill_gate": kill,
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "l1_trapflip_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[l1] baseline flip={base['flip_rate']:.3f} ({base['flip']}/{base['n']})  "
          f"candidate flip={cand['flip_rate']:.3f} ({cand['flip']}/{cand['n']})  "
          f"delta={summary['delta_flip']:+.3f}")
    if base["off_window"]["n"]:
        print(f"[l1] off-window flip: baseline {base['off_window']['flip']}/{base['off_window']['n']} -> "
              f"candidate {cand['off_window']['flip']}/{cand['off_window']['n']} "
              f"(the multi-window-decode-recovered band)")
    print(f"[l1] KILL gate (normal-position deploy-disagree): "
          f"{kill['deploy_disagree_rate']} on n={kill['n']} (threshold {args.kill_thr})")
    print(f"[l1] VERDICT = {verdict}  -> {outdir / 'l1_trapflip_summary.json'}")
    if verdict.startswith("INDETERMINATE"):
        print("[l1] NOTE: INDETERMINATE can be a WEAK lever OR STARVED RECALL (the per-move solver "
              "found few forced wins to inject). Check the fine-tune's solver fire-rate / throughput "
              "before concluding the lever is weak — a low fire-rate means re-run with higher "
              "solver_depth/node_budget/neighbor_dist, not MEMORIZES.")
    print("[l1] co-gates (run separately): exploit_probe.py --defender deploy --legal-set "
          "(off-window floor 0.0); --defender modelplayer (arm-aliasing-immune); "
          "scripts/probe_threat_logits.py (C1-C3). The KILL gate here is a collateral proxy — "
          "C1-C3 is the authoritative normal-play canary (catches symmetric corruption this misses).")


if __name__ == "__main__":
    main()
