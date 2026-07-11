"""D-F HEADSWAP — full-probe scoring driver (per arm/seed -> tidy scores table).

Binds to scripts/headswap/RECIPE.md §"Scoring", §"Metrics". For ONE trained arm
head (+ tower[11] for C/D) this scores EVERY positive (loss) + EVERY negative
(safe) probe row on the SAME K>1 multi-window instrument the positives were
selected by (score_probe.score_board), reconstructing each board from its
source game, then writes a tidy JSONL:

    reports/headswap/scores_{ARM}_seed{S}.jsonl
      {position_id, set, ply_band, source_game_id, v, tail_mass}

Board reconstruction routing (RECIPE §"Probe set"):
  - WP1 positive  : retro_slope 248k game by (opening_idx, head_as_p1), replay to t.
  - WP2 positive  : REGEN game by book_id (games.jsonl produced later; slotted in
                    via --wp2-games book_id=path). MISSING boards skipped + counted.
  - safe negative : retro_slope 248k game by (opening_idx, head_as_p1), replay to t.

ply_band = t // 10 (single phase per RECIPE — ply_band is the only match axis).
source_game_id groups per source game for the cluster bootstrap (metrics.py):
  (book_id_or_'retro', opening_idx, head_as_p1).

MEMORY: torch.inference_mode + one board at a time; per-position K-tensors are
built and dropped inside score_board (no accumulation). Resident set stays well
under 2 GB (trunk ~17 MB + one K-cluster batch).

REGISTER GUARD (INV-D1): this module scores only. SealBot/solver appear only as
the probe-set LABEL that assigns set=loss/safe; never a target, never a gradient.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts.headswap.model_heads import load_trunk
from scripts.headswap.score_probe import (
    load_scored_head,
    reconstruct_board,
    score_board,
)

RETRO_SOURCE = "retro"  # source_game_id prefix for retro_slope-reconstructed rows


# ── source-game indexing ──────────────────────────────────────────────────────


def index_games(games_path: str) -> Dict[Tuple[int, bool], dict]:
    """Map (opening_idx, head_as_p1) -> game record from a games.jsonl file."""
    idx: Dict[Tuple[int, bool], dict] = {}
    with open(games_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            idx[(int(g["opening_idx"]), bool(g["head_as_p1"]))] = g
    return idx


def _row_book(row: dict) -> Optional[str]:
    """Regen book_id for a row that routes to WP2 games, else None. WP2 positives
    carry book_id; WP2-harvested negatives carry source='wp2_regen_b<N>' (no book_id)."""
    if row.get("wp") == "WP2" and row.get("book_id"):
        return str(row["book_id"])
    src = str(row.get("source") or "")
    if src.startswith("wp2_regen_b"):
        return "evalfair_r5_wp2_b" + src.split("_b")[-1]
    return None


def _source_game_id(row: dict) -> str:
    """Stable grouping key for cluster bootstrap: one id per distinct source game."""
    book = _row_book(row) or RETRO_SOURCE
    return f"{book}:{int(row['opening_idx'])}:{int(bool(row['head_as_p1']))}"


def _ply_band(row: dict) -> int:
    return int(row["t"]) // 10


# ── per-row board resolution (skip-gracefully for missing WP2) ────────────────


def resolve_game(
    row: dict,
    retro_index: Dict[Tuple[int, bool], dict],
    wp2_indices: Dict[str, Dict[Tuple[int, bool], dict]],
) -> Optional[dict]:
    """Return the source game for a probe row, or None if unavailable (WP2 not
    yet regenerated). Routing by _row_book: WP2 positives (book_id) and
    WP2-harvest negatives (source='wp2_regen_b<N>') -> regen wp2 games; WP1
    positives + retro negatives -> retro_slope 248k index."""
    key = (int(row["opening_idx"]), bool(row["head_as_p1"]))
    book = _row_book(row)  # WP2 positive (book_id) OR WP2-harvest negative (source)
    if book is not None:
        book_idx = wp2_indices.get(book)
        if book_idx is None:
            return None
        return book_idx.get(key)
    return retro_index.get(key)


def score_rows(
    rows: List[dict],
    set_label: str,
    retro_index: Dict[Tuple[int, bool], dict],
    wp2_indices: Dict[str, Dict[Tuple[int, bool], dict]],
    model,
    head,
    is_bin: bool,
    verify_zobrist: bool = True,
) -> Tuple[List[dict], Dict[str, int]]:
    """Score a list of probe rows -> tidy score records + a skip/error tally.

    Each record: {position_id, set, ply_band, source_game_id, v, tail_mass,
    zobrist_match, wp}. Missing-board rows are SKIPPED and counted (WP2 regen
    gap). Zobrist mismatch is a HARD error (wrong reconstruction) unless
    verify_zobrist=False.
    """
    out: List[dict] = []
    tally = {"scored": 0, "skipped_no_game": 0, "zobrist_mismatch": 0}
    for r in rows:
        game = resolve_game(r, retro_index, wp2_indices)
        if game is None:
            tally["skipped_no_game"] += 1
            continue
        board, zob = reconstruct_board(game, int(r["t"]))
        zob_match = (zob == str(r["zobrist"]))
        if not zob_match:
            tally["zobrist_mismatch"] += 1
            if verify_zobrist:
                raise RuntimeError(
                    f"zobrist mismatch pos_id={r['zobrist']} "
                    f"opening={r['opening_idx']} head_as_p1={r['head_as_p1']} "
                    f"t={r['t']} wp={r.get('wp')}: recon {zob}"
                )
        sc = score_board(model, head, is_bin, board)
        out.append({
            "position_id": str(r["zobrist"]),
            "set": set_label,
            "ply_band": _ply_band(r),
            "source_game_id": _source_game_id(r),
            "v": sc["v"],
            "tail_mass": sc["tail_mass"],
            "zobrist_match": zob_match,
            "wp": r.get("wp"),
            "t": int(r["t"]),
        })
        tally["scored"] += 1
        del board  # drop per-position board; K-tensors already freed in score_board
    return out, tally


def load_jsonl(path: str) -> List[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def run(
    head_path: str,
    trunk_path: str,
    positives_path: str,
    negatives_path: str,
    games_path: str,
    wp2_games: Dict[str, str],
    out_path: str,
    verify_zobrist: bool = True,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trunk(trunk_path, device)
    model.eval()
    head, is_bin = load_scored_head(head_path, model, device)

    retro_index = index_games(games_path)
    wp2_indices = {bid: index_games(p) for bid, p in wp2_games.items()}

    positives = load_jsonl(positives_path)
    negatives = load_jsonl(negatives_path)

    with torch.inference_mode():
        pos_scored, pos_tally = score_rows(
            positives, "loss", retro_index, wp2_indices, model, head, is_bin,
            verify_zobrist,
        )
        neg_scored, neg_tally = score_rows(
            negatives, "safe", retro_index, wp2_indices, model, head, is_bin,
            verify_zobrist,
        )

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        for rec in pos_scored + neg_scored:
            f.write(json.dumps({
                "position_id": rec["position_id"],
                "set": rec["set"],
                "ply_band": rec["ply_band"],
                "source_game_id": rec["source_game_id"],
                "v": rec["v"],
                "tail_mass": rec["tail_mass"],
            }) + "\n")

    finite = all(np.isfinite(r["v"]) for r in pos_scored + neg_scored)
    summary = {
        "head": head_path,
        "is_bin": is_bin,
        "n_positive_scored": pos_tally["scored"],
        "n_negative_scored": neg_tally["scored"],
        "positive_tally": pos_tally,
        "negative_tally": neg_tally,
        "all_finite": bool(finite),
        "out": str(out_p),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="D-F HEADSWAP full-probe scoring driver")
    ap.add_argument("--head", required=True, help="trained head .pt (train_arm blob)")
    ap.add_argument("--trunk", required=True, help="run2 trunk ckpt")
    ap.add_argument("--positives", default="reports/valprobe/probe_set_v1.jsonl")
    ap.add_argument("--negatives", default="reports/valprobe/negatives_v1.jsonl")
    ap.add_argument(
        "--games",
        default="reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl",
        help="retro_slope 248k games.jsonl (WP1 + negatives boards)",
    )
    ap.add_argument(
        "--wp2-games", nargs="*", default=[],
        help="book_id=path pairs for regenerated WP2 games.jsonl "
             "(e.g. evalfair_r5_wp2_b0=reports/.../b0/games.jsonl)",
    )
    ap.add_argument("--out", required=True, help="scores_{ARM}_seed{S}.jsonl")
    ap.add_argument(
        "--no-verify-zobrist", action="store_true",
        help="downgrade zobrist mismatch to a counted warning (default: hard error)",
    )
    args = ap.parse_args()

    wp2_games: Dict[str, str] = {}
    for spec in args.wp2_games:
        if "=" not in spec:
            raise SystemExit(f"--wp2-games entry must be book_id=path, got {spec!r}")
        bid, p = spec.split("=", 1)
        wp2_games[bid] = p

    summary = run(
        head_path=args.head,
        trunk_path=args.trunk,
        positives_path=args.positives,
        negatives_path=args.negatives,
        games_path=args.games,
        wp2_games=wp2_games,
        out_path=args.out,
        verify_zobrist=not args.no_verify_zobrist,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
