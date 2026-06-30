#!/usr/bin/env python
"""D-WS3 / Z2 — held-out trap-corpus exporter (game-disjoint split).

Closes the Z2 OPEN item (`docs/handoffs/d_zvalid_z2_training_z_discriminator.md`
§9.1): emit the game-disjoint held-out trap JSONL + `finetune_game_ids.json` that
`scripts/eval/run_z2_standalone_ladder.py` and `scripts/eval/run_l1_trapflip_smoke.py`
consume. Read-only, CPU, commits nothing.

SOURCE: `reports/d_tactical_2026-06-26/corpus.jsonl` (the D-TACTICAL proven-loss
class; boards stored ONLY as move sequences — replay through the audited
`Board.apply_move` path, zero bbox/turn-phase drift). FILTER to the honest core:
`is_proven_core AND is_value_blind` (the D-PERCEPT honest 33; the 92->54%/33-core
correction) AND drop the 2 degenerate rows where `refuting_move == blunder_move`
-> 31 clean traps. Each carries a SealBot proven mate (|score|>=99_999_000, d6/d7/
d8) with a proof PV (the fresh deep re-score the DS1-stale-label memory mandates;
this corpus does NOT reuse the stale depth-5 soft `sealbot_score`).

SPLIT semantics (the memorization-vs-generalization rigor, REVIEW concern):
  * The D-WS3 L1 smoke fine-tune is PURE FRESH solver-in-loop self-play — it does
    NOT replay these corpus traps. The traps are from FROZEN d_ladder games
    (s150k/s175k/s200k buckets), disjoint from fresh self-play BY CONSTRUCTION.
    So the correct default is `--holdout-frac 1.0`: ALL 31 traps are held-out,
    `finetune_game_ids.json` is empty (the disjointness assertion is then
    trivially — and correctly — satisfied: nothing was trained on), and the gate
    gets maximum power from a thin (31) set.
  * For a future REALIZE-mode fine-tune that DOES replay the finetune traps, pass
    `--holdout-frac < 1.0`: a deterministic, bucket-stratified game-disjoint split
    -> the held-out slice is never injected, `finetune_game_ids.json` is non-empty,
    and the run_z2 `--finetune-game-ids` fail-closed assertion has teeth.

OUTPUT (default `reports/d_tactical_2026-06-26/`):
  * `heldout_traps.jsonl`    — the eval set (run_z2 `--trap-set`, l1_trapflip)
  * `finetune_traps.jsonl`   — the reserved finetune slice (REALIZE-mode)
  * `finetune_game_ids.json` — sorted source_game_ids of the finetune slice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402
import hexo_rl.encoding as enc  # noqa: E402


def replay(seq, encoding):
    """Reconstruct a Board via the audited legal apply path (no bbox/turn drift)."""
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def _cp_to_pm1(player) -> int:
    """Coerce the engine current_player (enum or int) to the corpus +-1 sign."""
    s = str(player)
    if "One" in s or s in ("1", "P1"):
        return 1
    if "Two" in s or s in ("-1", "2", "P2"):
        return -1
    try:
        v = int(player)
        return 1 if v == 1 else -1
    except (TypeError, ValueError):
        raise ValueError(f"cannot map current_player {player!r} to +-1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="reports/d_tactical_2026-06-26/corpus.jsonl")
    ap.add_argument("--encoding", default="v6_live2_ls")
    ap.add_argument("--out-dir", default="reports/d_tactical_2026-06-26")
    ap.add_argument("--holdout-frac", type=float, default=1.0,
                    help="fraction of distinct games into the held-out slice (default 1.0 = all "
                         "held-out, correct for pure fresh-self-play fine-tune)")
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--drop-off-window", action="store_true",
                    help="exclude traps whose saving move is off the single window "
                         "(default: KEEP all + tag in_window — the off-window ones are exactly "
                         "what the multi-window decode must recover)")
    args = ap.parse_args()

    spec = enc.lookup(args.encoding)
    plc = int(spec.policy_logit_count)

    rows = [json.loads(l) for l in open(args.corpus) if l.strip()]
    core = [
        r for r in rows
        if r.get("is_proven_core") and r.get("is_value_blind")
        and list(r["refuting_move"]) != list(r["blunder_move"])
    ]
    print(f"[export] corpus={len(rows)} rows -> {len(core)} clean (proven_core & value_blind & non-degenerate)")

    traps = []
    n_off = n_drop = n_cpmismatch = 0
    for r in core:
        enc_name = r.get("encoding", args.encoding) or args.encoding
        try:
            parent = replay(r["parent_move_seq"], enc_name)
            post = replay(r["postblunder_move_seq"], enc_name)
        except Exception as e:  # noqa: BLE001 — drop unreplayable rows, never silently include
            print(f"[export] DROP {r['pos_id']}: replay failed: {e}")
            n_drop += 1
            continue
        sq, sr = int(r["refuting_move"][0]), int(r["refuting_move"][1])
        in_window = parent.to_flat(sq, sr) < plc  # saving move expressible in the single window?
        if not in_window:
            n_off += 1
            if args.drop_off_window:
                continue
        cp_parent = _cp_to_pm1(parent.current_player)
        cp_post = _cp_to_pm1(post.current_player)
        if cp_parent != int(r["current_player_parent"]) or cp_post != int(r["current_player_post"]):
            n_cpmismatch += 1  # soft: trust corpus V2 parity flags, but surface drift
        traps.append({
            "pos_id": r["pos_id"],
            "source_game_id": r["game_id"],
            "game_idx": r["game_idx"],
            "bucket": r["bucket"],
            "encoding": enc_name,
            "parent_move_seq": [[int(q), int(s)] for q, s in r["parent_move_seq"]],
            "post_move_seq": [[int(q), int(s)] for q, s in r["postblunder_move_seq"]],
            "current_player_parent": int(r["current_player_parent"]),
            "current_player_post": int(r["current_player_post"]),
            "moves_remaining_parent": int(parent.moves_remaining),
            "moves_remaining_post": int(post.moves_remaining),
            "saving_move": [sq, sr],
            "blunder_move": [int(r["blunder_move"][0]), int(r["blunder_move"][1])],
            "mate_distance": r.get("mate_distance"),
            "proven_depth": r.get("proven_depth"),
            "depth_band": r.get("depth_band"),
            "refuting_pv": r.get("refuting_pv"),
            "is_value_blind": bool(r.get("is_value_blind")),
            "in_window": bool(in_window),
        })
    print(f"[export] kept={len(traps)} (off_window={n_off}, replay_drop={n_drop}, cp_mismatch_warn={n_cpmismatch})")

    # Deterministic bucket-stratified game-disjoint split.
    rng = np.random.default_rng(args.seed)
    holdout_ids: set = set()
    finetune_ids: set = set()
    by_bucket: dict = {}
    for t in traps:
        by_bucket.setdefault(t["bucket"], []).append(t["source_game_id"])
    for bucket, ids in sorted(by_bucket.items()):
        uids = sorted(set(ids))
        perm = rng.permutation(uids)
        n_hold = int(round(len(perm) * args.holdout_frac))
        holdout_ids.update(perm[:n_hold].tolist())
        finetune_ids.update(perm[n_hold:].tolist())
    assert holdout_ids.isdisjoint(finetune_ids), "split is not game-disjoint"

    heldout = [t for t in traps if t["source_game_id"] in holdout_ids]
    finetune = [t for t in traps if t["source_game_id"] in finetune_ids]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "heldout_traps.jsonl", "w") as f:
        for t in heldout:
            f.write(json.dumps(t) + "\n")
    with open(out / "finetune_traps.jsonl", "w") as f:
        for t in finetune:
            f.write(json.dumps(t) + "\n")
    with open(out / "finetune_game_ids.json", "w") as f:
        json.dump(sorted(finetune_ids), f, indent=0)

    print(f"[export] held-out={len(heldout)} traps ({sum(t['in_window'] for t in heldout)} in-window) "
          f"-> {out / 'heldout_traps.jsonl'}")
    print(f"[export] finetune={len(finetune)} traps -> {out / 'finetune_traps.jsonl'} "
          f"(+ {len(finetune_ids)} ids -> finetune_game_ids.json)")
    if args.holdout_frac >= 1.0:
        print("[export] holdout_frac=1.0 → ALL traps held-out, finetune slice empty "
              "(correct for pure fresh-self-play; for REALIZE-mode pass --holdout-frac < 1.0)")


if __name__ == "__main__":
    main()
