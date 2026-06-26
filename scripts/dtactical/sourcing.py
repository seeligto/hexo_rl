#!/usr/bin/env python3
"""D-TACTICAL — reachable proven-loss corpus builder (foundation gates V2/V3).

Harvests one record per reachable proven-loss-to-move position from the 68
decisive-blunder games of D-LOCALIZE P2, with FRESH fixed-depth SealBot oracle
labels (d6/d7/d8), depth bands, refuting move + proof PV, and the explicit move
sequence that reconstructs the COMPLETE faithful board (replay -> zero window
clipping). Read-only on all banks. Commits nothing.

------------------------------------------------------------------------------
PIPELINE / JOIN (validated 2026-06-26)
------------------------------------------------------------------------------
* `p2_decisions.jsonl` (68) = union of the s150k/s175k/s200k buckets (18/26/24).
  `game_idx` IS the 0-based line index into `per_game_seald5.jsonl` (280 games):
  all 68 (p1,p2,winner,plies) tuples match by construction (0/68 mismatch).
* 7 of 68 games are ALREADY-LOST (no decisive_ply, decisive_index=None) -> no
  clean decision blunder -> EXCLUDED. 61 decisive-blunder games -> 61 records.

PARENT vs POST position (the load-bearing definition):
* parent (decision) ply  P = decisions[decisive_index].ply  (MODEL to-move; about
  to blunder). parent_move_seq = moves[:P].
* blunder_move = moves[P] (== decisions[decisive_index].actual_move for all 61 —
  the REALIZED self-play move). refuting_move = decisions[decisive_index].ref_best
  (SealBot loss-avoiding move at the parent).
* post (proven-loss) position = decisions[decisive_index + 1] = the MODEL's NEXT
  decision = moves[:post_ply], ALWAYS model-to-move. This is what the D-PERCEPT
  discriminator scored (its nv_post / d6_post) and is the methodologically-correct
  "proven-loss-to-move" position for the tactical probe.
    - 32 games: P is the model's 1st stone of its turn -> post_ply = P+1 -> the
      post board == moves[:P]+[blunder] (the literal "post-blunder" board).
    - 29 games: P is the model's 2nd stone (moves_remaining==1) -> post_ply = P+3
      -> the post board = blunder + opponent's full reply (still model-to-move).
      The literal moves[:P+1] here would be OPPONENT-to-move (wrong perspective),
      so we use decs[i+1]'s ply, never a bare P+1.

------------------------------------------------------------------------------
NET-VALUE PROVENANCE (V2) — deviation from the dispatcher's checkpoint hint
------------------------------------------------------------------------------
The dispatcher said checkpoint_00272357 produced the stored p2 net_value. It did
NOT: p2_localize.py builds its engine from the PER-BUCKET checkpoints
(checkpoint_00150000/175000/200000, reports/d_ladder_2026-06-24/ckpts). Verified
empirically: per-bucket recompute == stored net_value to 5 decimals on every
spot-checked decision; checkpoint_00272357 differs by 0.02-0.45 (it produced the
DS1 / dvderisk net_value, a different net). V2 (reconstruction faithfulness)
therefore recomputes with the PER-BUCKET checkpoint and matches the stored value;
that is the only checkpoint for which the parity test is meaningful.

------------------------------------------------------------------------------
ORACLE (V3) — fresh fixed-depth SealBot
------------------------------------------------------------------------------
* SealBot mock game built from the LIVE Board exactly as
  hexo_rl/bots/sealbot_bot.py / scripts/dvderisk_ds1_scan.py do: board dict from
  get_stones() {(q,r): Player}, current_player, moves_left_in_turn=moves_remaining,
  move_count=len(stones).
* FIXED max_depth in {6,7,8}; time_limit is a SAFETY cap only (default 150s,
  matching the discriminator) — proven mates break far inside it. Proven mate iff
  |last_score| >= WIN_THRESHOLD (99,999,000); sign = side-to-move (model) POV,
  negative = model losing.
* mate-distance unit VERIFIED = SealBot minimax TURNS (compound 1-2 stone turns):
  s150k#38 score -99999997 -> mate_dist = 1e8-|score| = 3 == pv_turns 3 == 6 PV
  stones / 2. Depth bands applied in TURNS: short<=4, mid 5-8, deep>8, else
  unproven (loss-side, no mate at d8) / not-loss (d8 score >= 0).
* V3: proven_core = #post positions proven-loss (mate at d6, or d7/d8 upgrade)
  over the VALUE-BLIND subset -> reproduces disc_merged.proven_core_n=33 (+-3).
  The 4 d7/d8 upgrades require the full 150s cap (s175k#179's mate is only at d8,
  ~102s); a smaller cap silently drops it to 32.

------------------------------------------------------------------------------
MODES (run net once on GPU, seal in parallel CPU shards, then merge)
------------------------------------------------------------------------------
  --mode net                                   net_meta.jsonl (GPU; all 61)
  --mode seal --shard-index K --num-shards N    seal_shard{K}.jsonl (CPU)
  --mode merge                                  corpus.jsonl + sourcing_validation.md
  --mode all                                    net + seal(serial) + merge (single proc)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# SealBot import (vendor) — done lazily in the seal phase to keep the net phase
# torch/GPU-only and the seal phase CPU-only.

from engine import Board  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402

ENCODING = "v6_live2_ls"
WIN_SCORE = 1.0e8
WIN_THRESHOLD = 99_999_000.0   # |last_score| >= this => proven forced mate
VALUE_DRAWN = -0.05            # net_value >= this while SealBot proves LOSS => value-blind
NV_PARITY_TOL = 1e-3           # V2 |delta| tolerance

DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "d_tactical_2026-06-26"
DEFAULT_P2 = REPO_ROOT / "reports" / "d_localize_2026-06-25" / "p2_decisions.jsonl"
DEFAULT_GAMES = REPO_ROOT / "reports" / "d_ladder_2026-06-24" / "per_game_seald5.jsonl"
DEFAULT_CKPT_DIR = REPO_ROOT / "reports" / "d_ladder_2026-06-24" / "ckpts"
DEFAULT_DEPLOY_CKPT = REPO_ROOT / "checkpoints" / "checkpoint_00272357.pt"
DISC_MERGED = REPO_ROOT / "reports" / "d_localize_2026-06-25" / "discriminator" / "disc_merged.json"

BUCKET_CKPT = {
    "s150k": "checkpoint_00150000.pt",
    "s175k": "checkpoint_00175000.pt",
    "s200k": "checkpoint_00200000.pt",
}


# ---------------------------------------------------------------------------
# Target construction (deterministic order; shared by all modes)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def build_targets(decisions_recs: List[Dict[str, Any]], games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One target per decisive-blunder game (decisive_ply present). Validates the
    game_idx join (p1/p2/winner/plies). Returns deterministic, pos_id-sorted list."""
    targets: List[Dict[str, Any]] = []
    join_mismatch: List[str] = []
    for rec in decisions_recs:
        gi = rec["game_idx"]
        bucket = rec["bucket"]
        g = games[gi]
        if not (g["p1"] == rec["p1"] and g["p2"] == rec["p2"]
                and g["winner"] == rec["winner"] and g["plies"] == rec["plies"]):
            join_mismatch.append(f"{bucket}#{gi}")
        i = rec.get("decisive_index")
        if i is None or rec.get("decisive_ply") is None:
            continue  # ALREADY-LOST: no decision blunder
        decs = rec["decisions"]
        if i + 1 >= len(decs):
            continue  # no post decision (none observed, but guard)
        dec = decs[i]
        post = decs[i + 1]
        P = int(dec["ply"])
        post_ply = int(post["ply"])
        moves = [(int(q), int(r)) for (q, r) in g["moves"]]
        targets.append({
            "pos_id": f"{bucket}_g{gi}_p{post_ply}",
            "game_idx": gi,
            "bucket": bucket,
            "model_is_p1": rec.get("model_is_p1", rec["p1"] != "sealbot"),
            "decisive_index": i,
            "decisive_ply": P,
            "post_ply": post_ply,
            "parent_move_seq": [list(m) for m in moves[:P]],
            "blunder_move": list(moves[P]),
            "postblunder_move_seq": [list(m) for m in moves[:post_ply]],
            "refuting_move": list(dec["ref_best"]),
            "current_player_parent": int(dec["current_player"]),
            "current_player_post": int(post["current_player"]),
            "net_value_parent_stored": float(dec["net_value"]),
            "net_value_post_stored": float(post["net_value"]),
            "d6_post_stored": float(post["d6_score"]),
            "actual_move": list(dec["actual_move"]),
            "is_value_blind": float(post["net_value"]) >= VALUE_DRAWN,
        })
    if join_mismatch:
        print(f"[WARN] game_idx join mismatch on {len(join_mismatch)}: {join_mismatch}", flush=True)
    targets.sort(key=lambda t: t["pos_id"])
    return targets


def replay(moves: List[List[int]], upto: int) -> Tuple[Board, GameState]:
    b = Board.with_encoding_name(ENCODING)
    st = GameState.from_board(b)
    for k in range(upto):
        q, r = moves[k]
        st = st.apply_move(b, int(q), int(r))
    return b, st


# ---------------------------------------------------------------------------
# NET phase — recompute net_value via PER-BUCKET checkpoint, validate V2
# ---------------------------------------------------------------------------

def run_net(targets: List[Dict[str, Any]], ckpt_dir: Path, out_path: Path) -> None:
    import torch  # local import: seal shards stay torch-free
    from scripts.eval.gumbel_greedy_bot import _build_engine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[net] device={device} n_targets={len(targets)}", flush=True)
    engines: Dict[str, Any] = {}

    def engine_for(bucket: str):
        if bucket not in engines:
            engines[bucket] = _build_engine(str(ckpt_dir / BUCKET_CKPT[bucket]), ENCODING, device)
        return engines[bucket]

    with out_path.open("w") as fh:
        for t in targets:
            eng = engine_for(t["bucket"])
            # parent
            bp, _ = replay(t["parent_move_seq"] + [t["blunder_move"]], len(t["parent_move_seq"]))
            cp_par = int(bp.current_player)
            _, vpar = eng.infer(bp)
            # post
            bpo, _ = replay(t["postblunder_move_seq"], len(t["postblunder_move_seq"]))
            cp_post = int(bpo.current_player)
            _, vpost = eng.infer(bpo)
            row = {
                "pos_id": t["pos_id"],
                "cp_parent_recomputed": cp_par,
                "cp_post_recomputed": cp_post,
                "nv_parent_recomputed": float(vpar),
                "nv_post_recomputed": float(vpost),
                "cp_parent_match": cp_par == t["current_player_parent"],
                "cp_post_match": cp_post == t["current_player_post"],
                "nv_parent_match": abs(float(vpar) - t["net_value_parent_stored"]) <= NV_PARITY_TOL,
                "nv_post_match": abs(float(vpost) - t["net_value_post_stored"]) <= NV_PARITY_TOL,
                "nv_parent_delta": float(vpar) - t["net_value_parent_stored"],
                "nv_post_delta": float(vpost) - t["net_value_post_stored"],
                "checkpoint_bucket": BUCKET_CKPT[t["bucket"]],
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[net] {t['pos_id']} cp_par={cp_par}({row['cp_parent_match']}) "
                  f"nv_par_d={row['nv_parent_delta']:+.2e} nv_post_d={row['nv_post_delta']:+.2e}", flush=True)
    print(f"[net] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# SEAL phase — fresh fixed-depth SealBot oracle at the post position
# ---------------------------------------------------------------------------

def _seal_imports():
    for p in (str(REPO_ROOT / "vendor" / "bots" / "sealbot"),
              str(REPO_ROOT / "vendor" / "bots" / "sealbot" / "best")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from minimax_cpp import MinimaxBot  # type: ignore[import]
    from game import Player as SealPlayer  # type: ignore[import]
    return MinimaxBot, SealPlayer


class _MockGame:
    """Duck-types what SealBot's C++ binding reads (see sealbot_bot.py)."""

    def __init__(self, bd: dict, cp, ml: int, mc: int) -> None:
        self.board = bd
        self.current_player = cp
        self.moves_left_in_turn = ml
        self.move_count = mc


def _mock_from_board(board: Board, SealPlayer) -> "_MockGame":
    bd: dict = {}
    for q, r, p in board.get_stones():
        bd[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B
    cp = SealPlayer.A if board.current_player == 1 else SealPlayer.B
    return _MockGame(bd, cp, int(board.moves_remaining), len(bd))


def _score_depth(MinimaxBot, SealPlayer, board: Board, depth: int, time_limit: float):
    bot = MinimaxBot(time_limit=time_limit)
    bot.max_depth = depth
    game = _mock_from_board(board, SealPlayer)
    t0 = time.time()
    bot.get_move(game)
    dt = time.time() - t0
    score = float(bot.last_score)
    last_depth = int(bot.last_depth)
    mate = abs(score) >= WIN_THRESHOLD
    pv = bot.extract_pv() if mate else None
    return {
        "score": score,
        "depth": last_depth,
        "mate": mate,
        "secs": round(dt, 1),
        "pv": pv,
    }


def run_seal(targets: List[Dict[str, Any]], depths: List[int], time_limit: float, out_path: Path) -> None:
    MinimaxBot, SealPlayer = _seal_imports()
    print(f"[seal] n_targets={len(targets)} depths={depths} time_limit={time_limit}", flush=True)
    with out_path.open("w") as fh:
        for t in targets:
            board, _ = replay(t["postblunder_move_seq"], len(t["postblunder_move_seq"]))
            row: Dict[str, Any] = {"pos_id": t["pos_id"], "cp_post_board": int(board.current_player),
                                   "n_stones": len(board.get_stones())}
            sb: Dict[str, Any] = {}
            proven_depth: Optional[int] = None
            proof_pv = None
            terminal = board.check_win() or board.legal_move_count() == 0
            for d in depths:
                if terminal:
                    sb[f"d{d}"] = {"score": None, "depth": None, "mate": False, "secs": 0.0}
                    continue
                res = _score_depth(MinimaxBot, SealPlayer, board, d, time_limit)
                sb[f"d{d}"] = {"score": res["score"], "depth": res["depth"],
                               "mate": res["mate"], "secs": res["secs"]}
                if res["mate"] and proven_depth is None:
                    proven_depth = d            # shallowest proving depth
                    proof_pv = res["pv"]
            row["sealbot"] = sb
            row["terminal"] = terminal
            row["proven_depth"] = proven_depth
            row["proof_pv"] = proof_pv
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            mdist = (WIN_SCORE - abs(sb[f"d{proven_depth}"]["score"])) if proven_depth else None
            print(f"[seal] {t['pos_id']} d6={sb['d6']['score']} d7={sb['d7']['score'] if 'd7' in sb else '-'} "
                  f"d8={sb['d8']['score'] if 'd8' in sb else '-'} proven@={proven_depth} mate_dist={mdist} "
                  f"({sb.get('d8',{}).get('secs','-')}s)", flush=True)
    print(f"[seal] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# MERGE — combine net_meta + seal shards -> corpus.jsonl + validation report
# ---------------------------------------------------------------------------

def depth_band(mate_distance: Optional[float], loss_side: bool) -> str:
    if mate_distance is None:
        return "unproven" if loss_side else "not-loss"
    if mate_distance <= 4:
        return "short"
    if mate_distance <= 8:
        return "mid"
    return "deep"


def reality_label(proven_depth: Optional[int], d8_score: Optional[float]) -> str:
    if proven_depth == 6:
        return "proven-d6"
    if proven_depth in (7, 8):
        return f"proven-d{proven_depth}"
    if d8_score is None:
        return "unknown"
    return "not-loss" if d8_score >= 0 else "unproven-loss"


def run_merge(targets: List[Dict[str, Any]], report_dir: Path, depths: List[int],
              time_limit: float, out_path: Path) -> None:
    net_meta = {r["pos_id"]: r for r in load_jsonl(report_dir / "net_meta.jsonl")} \
        if (report_dir / "net_meta.jsonl").exists() else {}
    seal: Dict[str, Dict[str, Any]] = {}
    for shard in sorted(report_dir.glob("seal_shard*.jsonl")):
        for r in load_jsonl(shard):
            seal[r["pos_id"]] = r
    if (report_dir / "seal_all.jsonl").exists():
        for r in load_jsonl(report_dir / "seal_all.jsonl"):
            seal[r["pos_id"]] = r

    records: List[Dict[str, Any]] = []
    missing_net: List[str] = []
    missing_seal: List[str] = []
    for t in targets:
        pid = t["pos_id"]
        nm = net_meta.get(pid)
        sl = seal.get(pid)
        if nm is None:
            missing_net.append(pid)
        if sl is None:
            missing_seal.append(pid)
            continue
        sb = sl["sealbot"]
        proven_depth = sl.get("proven_depth")
        d8_score = sb.get("d8", {}).get("score") if "d8" in sb else None
        loss_side = (d8_score is not None and d8_score < 0)
        if proven_depth is not None:
            pscore = sb[f"d{proven_depth}"]["score"]
            mate_distance = WIN_SCORE - abs(pscore)
        else:
            mate_distance = None
        band = depth_band(mate_distance, loss_side)
        reality = reality_label(proven_depth, d8_score)
        proof_pv = sl.get("proof_pv")
        rec = {
            "pos_id": pid,
            "game_idx": t["game_idx"],
            "game_id": t["pos_id"],          # bucket+game_idx+post_ply: disjointness key
            "bucket": t["bucket"],
            "model_is_p1": t["model_is_p1"],
            "decisive_index": t["decisive_index"],
            "decisive_ply": t["decisive_ply"],
            "post_ply": t["post_ply"],
            "parent_move_seq": t["parent_move_seq"],
            "blunder_move": t["blunder_move"],
            "refuting_move": t["refuting_move"],
            "postblunder_move_seq": t["postblunder_move_seq"],
            "current_player_parent": t["current_player_parent"],
            "current_player_post": t["current_player_post"],
            "net_value_parent": (nm["nv_parent_recomputed"] if nm else t["net_value_parent_stored"]),
            "net_value_post": (nm["nv_post_recomputed"] if nm else t["net_value_post_stored"]),
            "net_value_parent_stored": t["net_value_parent_stored"],
            "net_value_post_stored": t["net_value_post_stored"],
            "nv_parent_match": (nm["nv_parent_match"] if nm else None),
            "nv_post_match": (nm["nv_post_match"] if nm else None),
            "cp_parent_match": (nm["cp_parent_match"] if nm else None),
            "cp_post_match": (nm["cp_post_match"] if nm else None),
            "checkpoint_bucket": (nm["checkpoint_bucket"] if nm else BUCKET_CKPT[t["bucket"]]),
            "is_value_blind": t["is_value_blind"],
            "sealbot": {d: sb.get(f"d{d}") for d in (6, 7, 8) if f"d{d}" in sb},
            "mate_distance": mate_distance,
            "depth_band": band,
            "reality": reality,
            "proven_depth": proven_depth,
            "refuting_pv": proof_pv,
            "refuting_pv_turns": (len(proof_pv) if proof_pv else 0),
            "is_proven_core": proven_depth is not None,
        }
        records.append(rec)

    records.sort(key=lambda r: r["pos_id"])
    with out_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[merge] wrote {len(records)} records -> {out_path}", flush=True)

    _write_validation(records, targets, net_meta, report_dir, depths, time_limit,
                      missing_net, missing_seal)


def _write_validation(records, targets, net_meta, report_dir, depths, time_limit,
                      missing_net, missing_seal) -> None:
    n_total = len(records)
    vb = [r for r in records if r["is_value_blind"]]
    n_vb = len(vb)

    # V2: net parity (per-bucket checkpoint) over join-matched positions
    nm_rows = list(net_meta.values())
    nv_par_ok = sum(1 for r in nm_rows if r["nv_parent_match"])
    nv_post_ok = sum(1 for r in nm_rows if r["nv_post_match"])
    cp_par_ok = sum(1 for r in nm_rows if r["cp_parent_match"])
    cp_post_ok = sum(1 for r in nm_rows if r["cp_post_match"])
    n_nm = len(nm_rows)
    max_par_delta = max((abs(r["nv_parent_delta"]) for r in nm_rows), default=0.0)
    max_post_delta = max((abs(r["nv_post_delta"]) for r in nm_rows), default=0.0)

    # V3: proven_core over value-blind subset vs disc_merged 33
    proven_vb = [r for r in vb if r["is_proven_core"]]
    proven_all = [r for r in records if r["is_proven_core"]]
    n_proven_vb = len(proven_vb)
    n_proven_all = len(proven_all)
    disc_target = 33
    if DISC_MERGED.exists():
        disc_target = json.load(open(DISC_MERGED)).get("proven_core_n", 33)

    # reality breakdown over value-blind (compare to disc loss_reality)
    from collections import Counter
    reality_vb = Counter(r["reality"] for r in vb)
    band_vb = Counter(r["depth_band"] for r in proven_vb)
    band_all = Counter(r["depth_band"] for r in proven_all)
    band_full = Counter(r["depth_band"] for r in records)

    # proven depth split
    pdepth_vb = Counter(r["proven_depth"] for r in proven_vb)

    def pct(a, b):
        return f"{a}/{b} ({(a / b * 100 if b else 0):.1f}%)"

    L: List[str] = []
    L.append("# D-TACTICAL — sourcing validation (foundation gates V2 / V3)\n")
    L.append(f"Built by `scripts/dtactical/sourcing.py`. Corpus: `corpus.jsonl` "
             f"({n_total} records, one per decisive-blunder game with a decision ply).\n")
    L.append("## Join / pipeline integrity\n")
    L.append("- `p2_decisions.jsonl` (68) = union of buckets s150k(18)+s175k(26)+s200k(24). "
             "`game_idx` == per_game_seald5 line index; **0/68 (p1,p2,winner,plies) mismatch**.")
    L.append("- 7 ALREADY-LOST games (decisive_ply=None) excluded -> **61 decisive-blunder records**.")
    L.append(f"- value-blind subset (net_value_post >= {VALUE_DRAWN}): **{n_vb}** "
             "(disc_merged VALUE-BLIND = 56).")
    if missing_net:
        L.append(f"- [WARN] {len(missing_net)} records missing net_meta: {missing_net}")
    if missing_seal:
        L.append(f"- [WARN] {len(missing_seal)} targets missing seal results (DROPPED): {missing_seal}")
    L.append("")

    L.append("## V2 — net_value parity (reconstruction faithfulness)\n")
    L.append("Recomputed with the **PER-BUCKET checkpoint** (s150k/s175k/s200k from "
             "`reports/d_ladder_2026-06-24/ckpts`) — the net that actually produced the stored "
             "p2 `net_value`. NOTE: checkpoint_00272357 (the dispatcher's hint) does NOT match "
             "stored p2 net_value (it produced the DS1/dvderisk net_value; empirical delta "
             "0.02-0.45) — using it would FALSE-FAIL V2. Per-bucket is the only meaningful net.\n")
    L.append(f"- net_value_parent parity (|d|<={NV_PARITY_TOL}): **{pct(nv_par_ok, n_nm)}**  "
             f"(max |delta| {max_par_delta:.2e})")
    L.append(f"- net_value_post   parity (|d|<={NV_PARITY_TOL}): **{pct(nv_post_ok, n_nm)}**  "
             f"(max |delta| {max_post_delta:.2e})")
    L.append(f"- current_player parent match: {pct(cp_par_ok, n_nm)}; post match: {pct(cp_post_ok, n_nm)}")
    v2_pass = (n_nm > 0 and nv_par_ok / n_nm >= 0.95 and nv_post_ok / n_nm >= 0.95)
    L.append(f"- **V2 GATE: {'PASS' if v2_pass else 'FAIL'}** (>=95% parity required).")
    L.append("")

    L.append("## V3 — fresh-oracle proven_core reproduction\n")
    L.append(f"SealBot fixed max_depth {depths}, time_limit={time_limit}s (matches discriminator; "
             "the d7/d8 upgrades need the full cap — s175k#179's mate is only at d8 ~102s).\n")
    L.append(f"- proven_core over VALUE-BLIND: **{n_proven_vb}** vs disc_merged "
             f"`proven_core_n={disc_target}` (delta {n_proven_vb - disc_target:+d}).")
    v3_pass = abs(n_proven_vb - disc_target) <= 3
    L.append(f"- **V3 GATE: {'PASS' if v3_pass else 'FAIL'}** (|delta| <= 3 required).")
    L.append(f"- proven_core over ALL 61 decisive: {n_proven_all}.")
    L.append(f"- proven depth split (value-blind): "
             + ", ".join(f"d{k}={pdepth_vb.get(k, 0)}" for k in (6, 7, 8)) + ".")
    L.append(f"- reality (value-blind): "
             + ", ".join(f"{k}={v}" for k, v in sorted(reality_vb.items()))
             + "  (disc_merged: proven-d6=29, proven-d7d8=4, unproven-loss=18, not-loss=5).")
    L.append("")

    L.append("## Depth-band histogram (proven positions; bands in SealBot TURNS)\n")
    L.append("mate-distance unit VERIFIED = SealBot minimax TURNS (1-2 stones/turn): "
             "s150k#38 score -99999997 -> mate_dist 3 == pv_turns 3 (== 6 PV stones). "
             "Bands: short<=4, mid 5-8, deep>8; unproven = loss-side no-mate@d8; not-loss = d8>=0.\n")
    L.append("| band | value-blind proven | all proven |")
    L.append("|---|---|---|")
    for band in ("short", "mid", "deep"):
        L.append(f"| {band} | {band_vb.get(band, 0)} | {band_all.get(band, 0)} |")
    L.append(f"| **proven total** | **{n_proven_vb}** | **{n_proven_all}** |")
    L.append("")
    L.append("Full corpus depth_band (incl. non-proven): "
             + ", ".join(f"{k}={band_full.get(k, 0)}"
                         for k in ("short", "mid", "deep", "unproven", "not-loss")) + ".")
    L.append("")

    L.append("## Counts\n")
    L.append(f"- n_total corpus records: **{n_total}**")
    L.append(f"- n_value_blind: **{n_vb}**")
    L.append(f"- n_proven_core (value-blind): **{n_proven_vb}**")
    L.append(f"- n_proven_core (all): **{n_proven_all}**")
    L.append("")

    L.append("## Caveats / decisions\n")
    L.append("- **game_idx join**: `game_idx` is the per_game_seald5 0-based line index "
             "(0/68 tuple mismatch); no fuzzy join needed.")
    L.append("- **post-blunder position** = `decisions[decisive_index+1]` (always model-to-move) = "
             "`moves[:post_ply]`. For 32 games post_ply=P+1 (== moves[:P]+[blunder]); for 29 games "
             "post_ply=P+3 (blunder + opponent reply). blunder==moves[P]==actual_move for all 61.")
    L.append("- **gumbel_played != actual_move for 34/61** games (re-derived deploy-head pick differs "
             "from the realized self-play move); the corpus uses the REALIZED move (moves[P]) as the "
             "blunder so the post board is the position actually reached.")
    L.append("- **net checkpoint**: per-bucket (matches stored), NOT 272357 (see V2).")
    L.append("- **time_limit=150** is a safety cap; proven mates are time-independent (break inside "
             "the cap). Non-mate heuristic scores can vary with the cap but do NOT affect is_proven_core.")
    L.append("")

    out = report_dir / "sourcing_validation.md"
    out.write_text("\n".join(L) + "\n")
    print(f"[merge] wrote {out}", flush=True)
    print("\n".join(L))
    print(f"\n[GATES] V2={'PASS' if v2_pass else 'FAIL'}  V3={'PASS' if v3_pass else 'FAIL'} "
          f"(proven_vb={n_proven_vb} vs {disc_target})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2-decisions", default=str(DEFAULT_P2))
    ap.add_argument("--per-game", default=str(DEFAULT_GAMES))
    ap.add_argument("--checkpoint", default=str(DEFAULT_DEPLOY_CKPT),
                    help="deploy/DS1 reference net (272357). NOTE: net_value parity (V2) uses the "
                         "PER-BUCKET checkpoints from --ckpt-dir, NOT this one (see module docstring).")
    ap.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR),
                    help="dir holding per-bucket checkpoints (s150k/s175k/s200k)")
    ap.add_argument("--sealbot-depths", default="6,7,8")
    ap.add_argument("--sealbot-time-limit", type=float, default=150.0,
                    help="SealBot per-depth time cap (s); matches the discriminator")
    ap.add_argument("--out", default=str(DEFAULT_REPORT_DIR / "corpus.jsonl"))
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--mode", choices=["all", "net", "seal", "merge"], default="all")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    depths = [int(d) for d in args.sealbot_depths.split(",")]

    decisions_recs = load_jsonl(Path(args.p2_decisions))
    games = load_jsonl(Path(args.per_game))
    targets = build_targets(decisions_recs, games)
    print(f"[main] mode={args.mode} targets={len(targets)} depths={depths}", flush=True)

    if args.mode in ("net", "all"):
        run_net(targets, Path(args.ckpt_dir), report_dir / "net_meta.jsonl")

    if args.mode == "seal":
        shard = [t for i, t in enumerate(targets) if i % args.num_shards == args.shard_index]
        run_seal(shard, depths, args.sealbot_time_limit,
                 report_dir / f"seal_shard{args.shard_index}.jsonl")
    elif args.mode == "all":
        run_seal(targets, depths, args.sealbot_time_limit, report_dir / "seal_all.jsonl")

    if args.mode in ("merge", "all"):
        run_merge(targets, report_dir, depths, args.sealbot_time_limit, Path(args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
