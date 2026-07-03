#!/usr/bin/env python
"""D-ZVALID Z2 — training-z efficacy discriminator: STANDALONE (backup-OFF) ladder.

THE GPU-WEEK GATE. After a short solver-in-loop fine-tune from the 272k anchor
(``checkpoints/checkpoint_00272357.pt``), this script answers ONE question:

    Did the net INTERNALISE the solver lever, or does it still need the runtime crutch?

It measures the **STANDALONE** net (deploy head ALONE, ``SolverBackupBot`` OFF) of the
fine-tuned candidate against the 272k baseline, on the eval ladder we actually have:

    1. trap-loss-rate on a GAME-DISJOINT held-out trap set (NOT the fine-tune set),
    2. fixed-depth SealBot WR (reproducible external bar),
    3. self-play BT-Elo (candidate vs baseline), opening-diversified + distinct-game CI.

KrakenBot is NOT in the ladder (we do not have its weights — D-ZVALID user constraint).

WHY STANDALONE / BACKUP-OFF IS LOAD-BEARING (Z3 discipline, the whole point of Z2):
    The deploy-backup (Z1, +0.195 over SealBot) is the RUNTIME SOLVER doing the work —
    measuring backup-ON vs SealBot re-measures the CRUTCH, not learning. The training-z
    claim is that policy-via-guided-search teaches the net to play the trap moves WITHOUT
    the solver. So the success metric is the STANDALONE net (this script never wraps the
    candidate in SolverBackupBot). DISTINCT from the value-distillation D-FULLSPEC killed;
    Probe-C (0/14) is the pessimistic prior (policy can't pick up ~0-prior moves even
    surfaced).

PRE-REGISTERED VERDICT (set BEFORE the run — a post-hoc move is storytelling; see
``docs/handoffs/d_zvalid_z2_training_z_discriminator.md`` for the full pre-reg):
    TEACHES        — candidate standalone trap-loss-rate DROPS by >= --teach-trap-drop AND
                     (SealBot WR rises OR self-play Elo delta CI lower bound > 0), backup
                     OFF  ->  full GPU-week training-z justified.
    DOESN'T-TEACH  — trap-loss-rate flat/within noise AND neither strength axis rises  ->
                     the backup is the PERMANENT mechanism (solver-grade cap); do NOT
                     spend the GPU-week. Reconsider (from-bootstrap solver-in-loop, etc.).
    INDETERMINATE  — too few distinct games / no trap set / underpowered CI.

STATUS: SCAFFOLD. The candidate checkpoint does not exist until the solver-in-loop
fine-tune runs (gated on Z0-C, the native quiet-move alpha-beta body, + the Rust
``finalize_game`` z-correction hook — see the runbook). The trap-set loader is a
documented STUB until the held-out trap corpus is wired. Everything else (the standalone
ladder, the paired/Elo stats, the JSON verdict) is real and reuses the A1 / deploy-
strength / round-robin stacks verbatim.

Run (once the candidate exists, on vast):
    .venv/bin/python scripts/eval/run_z2_standalone_ladder.py \
        --baseline-ckpt checkpoints/checkpoint_00272357.pt \
        --candidate-ckpt checkpoints/<finetuned_z>.pt \
        --encoding v6_live2_ls --sealbot-depth 5 --n-games 200 \
        --trap-set reports/d_tactical_2026-06-26/heldout_traps.jsonl \
        --out reports/d_zvalid_z2/run1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Trap set (held-out) ──────────────────────────────────────────────────────────
# A trap position = a board the net mis-evaluates and loses to SealBot from (the
# §D-TACTICAL / D-LOCALIZE proven-loss class — net_value ~ +0.6..1.0 at a d6 forced
# loss). MOVE-SEQUENCE JSONL row format (one position per line), emitted by
# scripts/dpfit_export_heldout_traps.py — replays through the audited legal apply
# path (zero bbox/turn-phase drift):
#   {"pos_id", "source_game_id", "bucket", "encoding": "v6_live2_ls",
#    "parent_move_seq": [[q,r],...],   # net to-move WITH the saving choice (playout start)
#    "post_move_seq":   [[q,r],...],   # proven-loss-to-move POST board (forced loss — diagnostic)
#    "current_player_parent": -1, "current_player_post": -1,
#    "saving_move": [q,r], "blunder_move": [q,r], "in_window": true, ...}
# GAME-DISJOINT: the held-out trap set MUST be drawn from games NOT in the solver-in-
# loop fine-tune corpus (else the drop measures memorisation, not generalisation).
def load_trap_set(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load held-out trap positions (move-sequence JSONL from
    scripts/dpfit_export_heldout_traps.py). STUB-SAFE: returns [] when no path / file
    (the caller reports the trap axis as unavailable)."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _board_from_trap(row: Dict[str, Any], encoding: str, which: str = "parent"):
    """Reconstruct an engine Board at a trap position via MOVE-SEQUENCE replay (the
    audited legal apply path — no bbox/turn-phase drift). ``which`` selects the board:
      * "parent" (DEFAULT, the playout start): net to-move WITH the saving choice. The
        methodologically-correct start for a generalisation-measuring playout — a net
        that learned the saving move AVOIDS the loss here.
      * "post": the proven-loss-to-move board (net already blundered). A FORCED loss —
        a net loses regardless of strength, so it carries ~no generalisation signal;
        kept only as a diagnostic. (This is why trap_loss_rate plays from "parent".)
    Reuses the same Board surface the eval bots use (``Board.with_encoding_name`` + the
    apply path). Soft-validates the recorded turn phase against the replayed board."""
    from engine import Board  # lazy: avoid importing the binding at module load
    from hexo_rl.encoding import normalize_encoding_name

    board = Board.with_encoding_name(normalize_encoding_name(row.get("encoding", encoding)))
    seq = row.get(f"{which}_move_seq") or row.get("move_seq")
    if seq is None:
        raise ValueError(
            f"trap row {row.get('pos_id')!r} has no {which}_move_seq (re-run the exporter "
            "scripts/dpfit_export_heldout_traps.py to emit the move-sequence form)"
        )
    for q, r in seq:
        board.apply_move(int(q), int(r))
    return board


def assert_traps_game_disjoint(
    traps: List[Dict[str, Any]], finetune_game_ids: Optional[set]
) -> str:
    """Fail-closed guard: the held-out trap set MUST be drawn from games NOT in the
    solver-in-loop fine-tune corpus (else the trap-loss drop measures memorisation, not
    generalisation). Asserts ``{trap.source_game_id} ∩ finetune_game_ids == ∅``. Returns a
    status string for the summary; raises on overlap. SKIPPED when the fine-tune corpus
    game-ids aren't supplied (the exporter that emits them isn't built yet — operator
    discipline holds until then)."""
    if not finetune_game_ids:
        return "SKIPPED_no_finetune_ids"
    trap_ids = {t.get("source_game_id") for t in traps if t.get("source_game_id") is not None}
    overlap = trap_ids & finetune_game_ids
    if overlap:
        raise AssertionError(
            f"held-out trap set NOT game-disjoint from the fine-tune corpus: "
            f"{len(overlap)} shared game-id(s) (e.g. {sorted(overlap)[:3]}) — the trap-loss "
            f"drop would measure memorisation. Fix the exporter's split."
        )
    return f"OK_disjoint(n_trap_games={len(trap_ids)})"


def trap_loss_rate(
    head_factory,
    traps: List[Dict[str, Any]],
    sealbot_depth: int,
    encoding: str,
) -> Optional[float]:
    """Fraction of trap positions the STANDALONE head loses from, playing out vs a
    fixed-depth SealBot. Returns None when the trap set is unavailable OR the playout is
    not yet implemented — NEVER raises, so the strength ladder still runs to an
    INDETERMINATE verdict (the runbook command passes --trap-set; a hard crash there would
    abort the whole eval)."""
    if not traps:
        return None
    from hexo_rl.bots.sealbot_bot import SealBotBot
    from hexo_rl.env.game_state import GameState

    head = head_factory()
    losses = 0
    counted = 0
    skipped = 0
    for t in traps:
        try:
            # PARENT board: net to-move WITH the saving choice. A net that internalised
            # the saving move avoids the loss here (POST is a forced loss -> no signal).
            board = _board_from_trap(t, encoding, which="parent")
        except Exception as e:  # noqa: BLE001 — skip unreplayable rows, never abort the eval
            print(f"[Z2] WARN trap {t.get('pos_id')!r} replay failed ({e}); skipping", flush=True)
            skipped += 1
            continue
        net_side = int(t.get("current_player_parent", board.current_player))  # 1 (P1) / -1 (P2)
        if hasattr(head, "reset"):
            head.reset()
        opp = SealBotBot(time_limit=60.0, max_depth=sealbot_depth)  # fixed depth, cold TT
        state = GameState.from_board(board)
        ply = 0
        while ply < 200 and not board.check_win() and board.legal_move_count() > 0:
            mover = head if board.current_player == net_side else opp
            q, r = mover.get_move(state, board)
            state = state.apply_move(board, q, r)
            ply += 1
        winner = board.winner() if board.check_win() else None  # 1 / -1 / None(draw)
        counted += 1
        if winner is not None and winner != net_side:  # opponent won -> net lost (draw = survived)
            losses += 1
    if counted == 0:
        print(f"[Z2] WARN: all {skipped} trap playouts failed to replay -> trap-loss-rate=None", flush=True)
        return None
    if skipped:
        print(f"[Z2] trap playout: {skipped} rows skipped (replay failure)", flush=True)
    return losses / counted


# ── Standalone ladder ────────────────────────────────────────────────────────────
def _build_standalone_head(ckpt: str, encoding: str, device, seed: int, legal_set: bool):
    """The deploy head ALONE (backup OFF). This is what 'standalone' means: the candidate
    is NEVER wrapped in SolverBackupBot here."""
    import torch

    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.deploy_strength_eval import (
        DeployHeadBot,
        _build_engine_for_model,
        extract_deploy_knobs,
    )

    # D-EVALGATE fix wave: `encoding` (the --encoding CLI board/engine encoding,
    # default v6_live2_ls) is threaded as decode_override — the baseline/candidate
    # ladder routinely cross-decodes checkpoints under the multi-window action
    # space regardless of their own stamp, so a disagreeing stamp must log
    # loudly, never raise (this call is not user-overridable per-checkpoint;
    # both baseline and candidate always decode under the SAME `--encoding`).
    model, _spec, auto_label = load_model_with_encoding(
        ckpt, device, decode_override=encoding,
    )
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    engine = _build_engine_for_model(model, encoding, device)
    head = DeployHeadBot(engine, knobs, label=Path(ckpt).stem, seed=seed, legal_set=legal_set)
    return head, knobs, auto_label


def sealbot_wr(head, label, sealbot_depth, encoding, n_games, opening_plies, seed_base) -> Dict[str, Any]:
    """Standalone head vs fixed-depth SealBot WR, distinct-game-deduped (the deterministic
    g=0 regime collapses to ~2 games/pair without opening diversity — §D-ARGMAX)."""
    import numpy as np

    from hexo_rl.bots.sealbot_bot import SealBotBot
    from hexo_rl.eval.a1_stats import cand_outcome, dedup_distinct
    from hexo_rl.eval.deploy_strength_eval import _play_one_game

    games: List[Dict[str, Any]] = []
    for gi in range(n_games):
        if hasattr(head, "reset"):
            head.reset()
        opp = SealBotBot(time_limit=60.0, max_depth=sealbot_depth)  # fixed depth, cold TT
        if gi % 2 == 0:
            p1, p2, l1, l2 = head, opp, label, "sealbot"
        else:
            p1, p2, l1, l2 = opp, head, "sealbot", label
        seed = seed_base + gi
        g = _play_one_game(p1, p2, l1, l2, encoding, opening_plies=opening_plies, seed=seed)
        g["arm"] = label
        games.append(g)
    scores = [cand_outcome(g, label) for g in games]
    distinct = dedup_distinct(games, label)
    return {
        "wr_raw": float(np.mean(scores)) if scores else 0.0,
        "wr_distinct": float(np.mean(distinct)) if distinct else 0.0,
        "n_raw": len(scores),
        "n_distinct": len(distinct),
        "_games": games,
    }


def selfplay_elo(cand_head, base_head, encoding, n_games, opening_plies, seed_base) -> Dict[str, Any]:
    """Candidate-vs-baseline self-play BT-Elo, distinct-game cluster-bootstrap CI. Reuses
    the round-robin aggregate stack verbatim (the §D-ARGMAX pseudo-replication guard).
    Anchor = baseline (labels[0]); a candidate CI lower bound > 0 Elo = candidate
    stronger over DISTINCT games."""
    from hexo_rl.eval.deploy_strength_eval import _play_one_game
    from hexo_rl.eval.round_robin import (
        bootstrap_ratings_ci,
        distinct_per_pair,
        effective_n_guard,
    )

    labels = ["baseline", "candidate"]  # labels[0] anchors the BT gauge at 0 Elo
    games: List[Dict[str, Any]] = []
    for gi in range(n_games):
        for h in (cand_head, base_head):
            if hasattr(h, "reset"):
                h.reset()
        if gi % 2 == 0:
            p1, p2, l1, l2 = cand_head, base_head, "candidate", "baseline"
        else:
            p1, p2, l1, l2 = base_head, cand_head, "baseline", "candidate"
        seed = seed_base + gi
        games.append(_play_one_game(p1, p2, l1, l2, encoding, opening_plies=opening_plies, seed=seed))
    ci = bootstrap_ratings_ci(games, labels)  # {label: (lo, hi)} in Elo units
    dpp = distinct_per_pair(games, labels)    # {(a, b): n_distinct}
    return {
        "ratings_ci": {lbl: [float(lo), float(hi)] for lbl, (lo, hi) in ci.items()},
        "distinct_per_pair": {f"{a}|{b}": int(v) for (a, b), v in dpp.items()},
        "effective_n": effective_n_guard(games, labels),
        "_games": games,
    }


def decide(
    base_trap: Optional[float],
    cand_trap: Optional[float],
    base_wr: float,
    cand_wr: float,
    elo_ci_lo: Optional[float],
    teach_trap_drop: float,
    teach_wr_rise: float,
    elo_distinct_ok: bool,
    z_loss_coverage: Optional[float] = None,
    min_z_coverage: float = 0.5,
) -> str:
    """Pre-registered TEACHES / DOESN'T-TEACH / INDETERMINATE verdict (backup OFF).

    Strength corroboration is falsifiable on BOTH arms: the SealBot-WR arm needs a min
    ABSOLUTE rise ``teach_wr_rise`` (a bare ``>0`` is noise-satisfiable and manufactures a
    spurious TEACHES); the self-play arm needs the distinct-game Elo CI lower bound > 0.

    DOESN'T-TEACH is only EARNED when the fine-tune actually HAD z-LOSS recall on the
    held-out quiet-trap class (``z_loss_coverage >= min_z_coverage``). Without the perf
    layer the budget-bound full-legal verify certifies ~0 not-in-check quiet traps within
    budget (the binding constraint is RECALL, not soundness), so a starved fine-tune yields
    flat trap-loss BY CONSTRUCTION — calling that DOESN'T-TEACH mis-attributes starved
    recall to a dead lever and wrongly kills the GPU-week. Below the floor we return
    INDETERMINATE_STARVED_RECALL instead."""
    strength_rose = (
        (cand_wr - base_wr) >= teach_wr_rise
        or (elo_ci_lo is not None and elo_ci_lo > 0.0)
    )
    if not elo_distinct_ok:
        return "INDETERMINATE_UNDERPOWERED"
    if base_trap is None or cand_trap is None:
        # No trap set / unrun playout: the primary internalisation signal is unavailable.
        return "INDETERMINATE_NO_TRAPSET"
    trap_dropped = (base_trap - cand_trap) >= teach_trap_drop
    if trap_dropped and strength_rose:
        return "TEACHES"
    if not trap_dropped and not strength_rose:
        # DOESN'T-TEACH is only EARNED with MEASURED adequate recall: unmeasured (None) or
        # below-floor z-coverage withholds the GPU-week-killing verdict (the safe failure).
        if z_loss_coverage is None or z_loss_coverage < min_z_coverage:
            return "INDETERMINATE_STARVED_RECALL"
        return "DOESNT_TEACH"
    return "INDETERMINATE_MIXED"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-ckpt", default="checkpoints/checkpoint_00272357.pt",
                    help="the 272k anchor (pre-fine-tune standalone reference)")
    ap.add_argument("--candidate-ckpt", required=True, help="the solver-in-loop fine-tuned net (standalone)")
    ap.add_argument("--encoding", default="v6_live2_ls")
    ap.add_argument("--sealbot-depth", type=int, default=5, help="fixed-depth external bar (reproducible)")
    ap.add_argument("--n-games", type=int, default=200, help="games per ladder rung")
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="RNG-seeded opening diversity — distinct games for the g=0 deterministic "
                         "regime (§D-ARGMAX effective-n; 0 collapses the CI)")
    ap.add_argument("--trap-set", default=None, help="held-out trap corpus JSONL (game-disjoint)")
    ap.add_argument("--finetune-game-ids", default=None,
                    help="JSON list of the solver-in-loop fine-tune corpus game-ids; when given, "
                         "the held-out trap set is asserted DISJOINT from it (fail-closed)")
    ap.add_argument("--teach-trap-drop", type=float, default=0.10,
                    help="min absolute trap-loss-rate drop (candidate vs baseline) to count as taught")
    ap.add_argument("--teach-wr-rise", type=float, default=0.03,
                    help="min absolute SealBot-WR rise (candidate vs baseline, distinct games) for the "
                         "WR corroboration arm — a bare >0 is noise-satisfiable and manufactures TEACHES")
    ap.add_argument("--z-loss-coverage", type=float, default=None,
                    help="MEASURED fraction of held-out quiet traps the solver-in-loop certified a "
                         "z-correction for within budget (from the fine-tune logs). Below --min-z-coverage "
                         "a flat trap-loss is STARVED RECALL, not a dead lever -> INDETERMINATE_STARVED_RECALL")
    ap.add_argument("--min-z-coverage", type=float, default=0.5,
                    help="z-LOSS coverage floor below which DOESN'T-TEACH is NOT earned (starved recall)")
    ap.add_argument("--min-distinct-per-pair", type=int, default=10,
                    help="§D-ARGMAX power floor: below this the self-play CI is untrusted -> INDETERMINATE")
    ap.add_argument("--legal-set", action="store_true",
                    help="route the standalone deploy head through the multi-window no-drop decode "
                         "(Track 2). Match whatever the fine-tune deployed under.")
    ap.add_argument("--seed-base", type=int, default=20260629)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default="reports/d_zvalid_z2", help="output dir (gitignored — copy to persist)")
    args = ap.parse_args()

    import torch

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_head, base_knobs, base_enc = _build_standalone_head(
        args.baseline_ckpt, args.encoding, device, args.seed_base, args.legal_set
    )
    cand_head, cand_knobs, cand_enc = _build_standalone_head(
        args.candidate_ckpt, args.encoding, device, args.seed_base, args.legal_set
    )
    print(f"[Z2] STANDALONE ladder (backup OFF). baseline={Path(args.baseline_ckpt).name} "
          f"candidate={Path(args.candidate_ckpt).name} enc={args.encoding} "
          f"sealbot_d{args.sealbot_depth} n={args.n_games} knobs(base)={base_knobs}", flush=True)

    # 1) trap-loss-rate (held-out, game-disjoint). Fail-closed disjointness check first.
    traps = load_trap_set(args.trap_set)
    finetune_game_ids = None
    if args.finetune_game_ids:
        finetune_game_ids = set(json.loads(Path(args.finetune_game_ids).read_text()))
    disjoint_status = assert_traps_game_disjoint(traps, finetune_game_ids)
    base_trap = trap_loss_rate(lambda: base_head, traps, args.sealbot_depth, args.encoding)
    cand_trap = trap_loss_rate(lambda: cand_head, traps, args.sealbot_depth, args.encoding)
    trap_playout_unimplemented = bool(traps) and (base_trap is None or cand_trap is None)

    # 2) SealBot fixed-depth WR (standalone)
    base_sb = sealbot_wr(base_head, "baseline", args.sealbot_depth, args.encoding,
                         args.n_games, args.opening_plies, args.seed_base)
    cand_sb = sealbot_wr(cand_head, "candidate", args.sealbot_depth, args.encoding,
                         args.n_games, args.opening_plies, args.seed_base)

    # 3) self-play BT-Elo (candidate vs baseline)
    elo = selfplay_elo(cand_head, base_head, args.encoding, args.n_games,
                       args.opening_plies, args.seed_base)
    elo_ci = elo.get("ratings_ci", {})  # {label: [lo, hi]} Elo, anchored baseline=0
    cand_ci = elo_ci.get("candidate", [None, None])
    elo_ci_lo = cand_ci[0] if cand_ci else None
    dpp = elo.get("distinct_per_pair", {})
    min_dpp = min(dpp.values()) if isinstance(dpp, dict) and dpp else 0
    elo_distinct_ok = min_dpp >= args.min_distinct_per_pair

    verdict = decide(
        base_trap, cand_trap, base_sb["wr_distinct"], cand_sb["wr_distinct"],
        elo_ci_lo, args.teach_trap_drop, args.teach_wr_rise, elo_distinct_ok,
        z_loss_coverage=args.z_loss_coverage, min_z_coverage=args.min_z_coverage,
    )

    summary = {
        "design": "z2_standalone_ladder_backup_OFF",
        "baseline_ckpt": str(args.baseline_ckpt),
        "candidate_ckpt": str(args.candidate_ckpt),
        "encoding": args.encoding,
        # D-EVALGATE fix wave: the RESOLVED decode label each head was actually
        # built under (post decode_override — always == args.encoding today
        # since the override is unconditional here; recorded so a future
        # per-checkpoint divergence is visible in the artifact rather than
        # silently dropped, per review point "dead auto_label/base_enc/cand_enc").
        "decode_label": {"baseline": base_enc, "candidate": cand_enc},
        "sealbot_depth": args.sealbot_depth,
        "trap_loss_rate": {"baseline": base_trap, "candidate": cand_trap,
                           "available": bool(traps), "n_traps": len(traps),
                           "playout_unimplemented": trap_playout_unimplemented,
                           "game_disjoint_check": disjoint_status},
        "sealbot_wr": {"baseline": base_sb["wr_distinct"], "candidate": cand_sb["wr_distinct"],
                       "baseline_n_distinct": base_sb["n_distinct"], "candidate_n_distinct": cand_sb["n_distinct"]},
        "selfplay_elo": {"ratings_ci": elo_ci, "min_distinct_per_pair": min_dpp,
                         "distinct_ok": elo_distinct_ok, "candidate_ci_lo": elo_ci_lo},
        "z_loss_coverage": {"measured": args.z_loss_coverage, "floor": args.min_z_coverage},
        "thresholds": {"teach_trap_drop": args.teach_trap_drop, "teach_wr_rise": args.teach_wr_rise,
                       "min_distinct_per_pair": args.min_distinct_per_pair},
        "verdict": verdict,
        "note": "STANDALONE (backup OFF). Verdict drives the GPU-week gate — see "
                "docs/handoffs/d_zvalid_z2_training_z_discriminator.md.",
    }
    games = {"sealbot_baseline": base_sb["_games"], "sealbot_candidate": cand_sb["_games"],
             "selfplay": elo["_games"]}
    with open(out_dir / "Z2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_game.jsonl", "w") as f:
        for grp, gl in games.items():
            for g in gl:
                g["_group"] = grp
                f.write(json.dumps(g) + "\n")
    print(f"[Z2] verdict={verdict}  trap(base={base_trap},cand={cand_trap})  "
          f"sealbotWR(base={base_sb['wr_distinct']:.3f},cand={cand_sb['wr_distinct']:.3f})  "
          f"elo_ci_lo={elo_ci_lo}  wrote {out_dir}/Z2_summary.json (reports/ gitignored — copy to persist)",
          flush=True)


if __name__ == "__main__":
    main()
