#!/usr/bin/env python3
"""D-PFIT P1a — trap plane-export + alignment validation (correctness gate).

Builds the data artifact for D-PFIT P1: "can the frozen-trunk POLICY head
represent the solver-proven saving moves?" This script ONLY exports the planes
+ validates alignment. It does NOT run the fine-tune (P1b).

WHAT IS EXPORTED (`data/dpfit_traps.npz`)
-----------------------------------------
One record per proven-core position (38; `is_proven_core==True` in
`reports/d_tactical_2026-06-26/corpus.jsonl`). For each, the PARENT board
(`parent_move_seq`, MODEL to-move, about to blunder) is replayed under the
`v6_live2_ls` encoding and encoded into the SINGLE PRIMARY (window_center /
to_flat) 4-plane frame the policy head fits on:

  planes[i]  : (4, 19, 19) float32 — [my_t0, opp_t0, moves_rem==2 bcast, ply%2 bcast]
               built at the to_flat window_center frame (see INDEX CORRESPONDENCE).
  target_idx : to_flat(refuting_move)  (the SAVING move) for in-window; -1 off-window.
  in_window  : bool (32 True / 6 False). Off-window route to the D-DECODE multi-window
               lever; EXCLUDE from the single-window policy-fit set, kept here flagged.
  checkpoint : per-bucket net that MADE the blunder (s150k/s175k/s200k ->
               reports/d_ladder_2026-06-24/ckpts/checkpoint_00{150000,175000,200000}.pt)

INDEX CORRESPONDENCE (the airtight bit the fit stage relies on)
---------------------------------------------------------------
`board.to_flat(q,r)` == Rust `window_flat_idx` == an index in the SINGLE window
centered on `window_center()` = ((min_q+max_q)//2, (min_r+max_r)//2) over all
stones (trunc-toward-zero), trunk_sz = cluster_window_size = 19, half = 9:
    wq = q - cq + 9 ; wr = r - cr + 9 ; flat = wq*19 + wr   (in [0,361); else usize::MAX)
The policy head (`policy_conv`->`policy_fc`->log_softmax) emits a length-362 vector
per window indexed by THIS SAME wq*19+wr local index (pass slot = 361). So building
the planes at window_center (stone placement uses board.to_flat per stone, exactly
as Rust `to_planes`) makes `exp(log_policy)[to_flat(ref)]` the raw-net mass on the
saving move — index space identical by construction. `Board.to_tensor()` PANICS for
v6_live2_ls (multi-window guard), so planes are built from `get_stones()` + `to_flat`
(validated byte-identical to the engine `get_cluster_views` kernel, V2).

VALIDATION GATES (printed; --validate)
  V1 index parity   : window_coords(target_idx) round-trips to refuting_move (32 in-win).
  V2 stone kernel   : my stone placement == engine GameState.to_tensor planes[0,8] at
                      EVERY cluster center (byte-identical).
  V3 scalar planes  : my planes[2,3] == engine to_tensor planes[16,17] (all 38).
  V4 gold forward   : for boards whose primary window IS a cluster center, model forward
                      on my cached planes vs on engine-assembled primary-cluster planes ->
                      identical mass+rank @ to_flat (the cached-vs-engine forward proof).
  V5 npz round-trip : reload npz, forward, == fresh-rebuild forward @ to_flat.

RAW-NET BASELINE (pre-fine-tune; the start the fit must improve from)
  For the 32 in-window positions: raw-net policy-head mass + rank on the saving move,
  per-bucket + overall. (D-SOLVER A1: ~67% of saving moves have ~0 deploy prior.)

Run: .venv/bin/python scripts/dpfit_export.py --validate
CPU is fine + deterministic (no autocast); 38 positions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import engine  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from scripts.eval.gumbel_greedy_bot import _build_engine  # noqa: E402

ENC = "v6_live2_ls"
S = 19
HALF = (S - 1) // 2          # 9
N_CELLS = S * S              # 361 (pass slot = 361, policy_logit_count = 362)
NEAR_ZERO = 1e-3             # "~0 prior" threshold (matches run_t0 near-zero cut)

CKPT_DIR = REPO_ROOT / "reports" / "d_ladder_2026-06-24" / "ckpts"
BUCKET_CKPT = {
    "s150k": "checkpoint_00150000.pt",
    "s175k": "checkpoint_00175000.pt",
    "s200k": "checkpoint_00200000.pt",
}
DEFAULT_CORPUS = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "corpus.jsonl"
DEFAULT_T0 = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "t0_results.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "dpfit_traps.npz"


# ---------------------------------------------------------------------------
# board replay + primary-window plane construction (engine-kernel-faithful)
# ---------------------------------------------------------------------------

def replay(seq: List[List[int]]) -> "engine.Board":
    b = engine.Board.with_encoding_name(ENC)
    for (q, r) in seq:
        b.apply_move(int(q), int(r))
    return b


def window_center(board: "engine.Board") -> Optional[Tuple[int, int]]:
    """Recover (cq, cr) = the to_flat window_center from any in-window stone.

    cq = q - (flat//19) + 9. Independent of the trunc-toward-zero midpoint
    formula (derived from the authoritative Rust to_flat), so it cannot drift.
    """
    for (q, r, _p) in board.get_stones():
        flat = board.to_flat(int(q), int(r))
        if flat < N_CELLS:
            wq, wr = flat // S, flat % S
            return int(q) - wq + HALF, int(r) - wr + HALF
    return None


def primary_planes(board: "engine.Board") -> np.ndarray:
    """(4,19,19) float32 frame at window_center, matching Rust `to_planes`.

    plane0 my stones, plane1 opp stones (placed via board.to_flat per stone,
    dropping stones outside the primary window exactly as Rust does), plane2
    moves_remaining==2 broadcast, plane3 ply%2 broadcast. The 4 planes are
    v6_live2_ls kept_plane_indices [0,8,16,17].
    """
    pl = np.zeros((4, S, S), dtype=np.float32)
    cp = int(board.current_player)              # 1 (P1) or -1 (P2)
    for (q, r, p) in board.get_stones():
        flat = board.to_flat(int(q), int(r))
        if flat >= N_CELLS:                     # off-primary-window (incl. usize::MAX)
            continue
        wq, wr = flat // S, flat % S
        pl[0 if int(p) == cp else 1, wq, wr] = 1.0
    mr = int(board.moves_remaining)
    pl[2, :, :] = 0.0 if mr == 1 else 1.0       # == (mr==2 ? 1 : 0) for mr in {1,2}
    pl[3, :, :] = float(int(board.ply) % 2)
    return pl


def stone_view_at(board: "engine.Board", cq: int, cr: int) -> np.ndarray:
    """(2,19,19) [my, opp] stone view at an ARBITRARY center — for V2 kernel parity
    against engine GameState.to_tensor (which centers each cluster at its own center)."""
    v = np.zeros((2, S, S), dtype=np.float32)
    cp = int(board.current_player)
    for (q, r, p) in board.get_stones():
        wq, wr = int(q) - cq + HALF, int(r) - cr + HALF
        if 0 <= wq < S and 0 <= wr < S:
            v[0 if int(p) == cp else 1, wq, wr] = 1.0
    return v


def to_flat_idx(board: "engine.Board", move: List[int]) -> int:
    flat = board.to_flat(int(move[0]), int(move[1]))
    return int(flat) if flat < N_CELLS else -1


# ---------------------------------------------------------------------------
# target build
# ---------------------------------------------------------------------------

def build_targets(corpus_path: Path) -> List[Dict[str, Any]]:
    recs = [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]
    core = [r for r in recs if r.get("is_proven_core")]
    core.sort(key=lambda r: r["pos_id"])
    out = []
    for r in core:
        out.append({
            "pos_id": r["pos_id"],
            "game_id": r["game_id"],
            "bucket": r["bucket"],
            "depth_band": r.get("depth_band", "unknown"),
            "parent_move_seq": r["parent_move_seq"],
            "refuting_move": r["refuting_move"],
            "blunder_move": r["blunder_move"],
            "current_player_parent": int(r["current_player_parent"]),
        })
    return out


# ---------------------------------------------------------------------------
# engine forward helpers (CPU, fp32, deterministic — no autocast)
# ---------------------------------------------------------------------------

def raw_policy(model, planes: np.ndarray) -> np.ndarray:
    """Single primary-window RAW policy-head forward -> prob vector (362,)."""
    x = torch.from_numpy(np.ascontiguousarray(planes)).float().unsqueeze(0)
    with torch.no_grad():
        log_policy, _v, _vl = model(x)
    return log_policy.exp().squeeze(0).double().numpy()


def mass_rank(prob: np.ndarray, idx: int) -> Tuple[float, int]:
    """mass on idx + rank (0 = top), matching run_t0 `_policy_prior_on`."""
    mass = float(prob[idx])
    rank = int((prob > prob[idx]).sum())
    return mass, rank


# ---------------------------------------------------------------------------
# main: export + validate + baseline
# ---------------------------------------------------------------------------

def run(corpus_path: Path, out_path: Path, do_validate: bool, t0_path: Path) -> int:
    device = torch.device("cpu")            # deterministic; 38 positions
    targets = build_targets(corpus_path)
    print(f"[dpfit] proven-core targets: {len(targets)}")

    engines: Dict[str, Any] = {}

    def eng_for(bucket: str):
        if bucket not in engines:
            engines[bucket] = _build_engine(
                str(CKPT_DIR / BUCKET_CKPT[bucket]), ENC, device)
        return engines[bucket]

    # ---- build records ----
    planes_arr = np.zeros((len(targets), 4, S, S), dtype=np.float32)
    target_idx = np.full(len(targets), -1, dtype=np.int64)
    in_window = np.zeros(len(targets), dtype=bool)
    pos_ids, game_ids, buckets, bands, ckpts = [], [], [], [], []
    cur_players = np.zeros(len(targets), dtype=np.int64)
    moves_rem = np.zeros(len(targets), dtype=np.int64)
    plys = np.zeros(len(targets), dtype=np.int64)
    ref_q = np.zeros(len(targets), dtype=np.int64)
    ref_r = np.zeros(len(targets), dtype=np.int64)

    for i, t in enumerate(targets):
        b = replay(t["parent_move_seq"])
        planes_arr[i] = primary_planes(b)
        tidx = to_flat_idx(b, t["refuting_move"])
        target_idx[i] = tidx
        in_window[i] = tidx >= 0
        pos_ids.append(t["pos_id"]); game_ids.append(t["game_id"])
        buckets.append(t["bucket"]); bands.append(t["depth_band"])
        ckpts.append(str(CKPT_DIR / BUCKET_CKPT[t["bucket"]]))
        cur_players[i] = int(b.current_player)
        moves_rem[i] = int(b.moves_remaining)
        plys[i] = int(b.ply)
        ref_q[i] = int(t["refuting_move"][0]); ref_r[i] = int(t["refuting_move"][1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        planes=planes_arr,
        target_idx=target_idx,
        in_window=in_window,
        pos_id=np.array(pos_ids),
        game_id=np.array(game_ids),
        bucket=np.array(buckets),
        depth_band=np.array(bands),
        checkpoint=np.array(ckpts),
        current_player=cur_players,
        moves_remaining=moves_rem,
        ply=plys,
        refuting_q=ref_q,
        refuting_r=ref_r,
        encoding=np.array(ENC),
        schema=np.array("dpfit_traps_v1: primary-window frame; target_idx=to_flat(refuting_move) "
                        "(-1 off-window); planes [my,opp,mr==2,ply%2] @ window_center"),
    )
    n_in = int(in_window.sum())
    print(f"[dpfit] wrote {out_path}  records={len(targets)}  in_window={n_in}  "
          f"off_window={len(targets) - n_in}")

    if not do_validate:
        return 0

    # ---- VALIDATION ----
    print("\n========== VALIDATION ==========")
    v1_ok = v1_tot = 0
    v2_ok = v2_tot = 0
    v3_ok = v3_tot = 0
    v4_ok = v4_tot = 0
    v5_ok = v5_tot = 0
    v4_max_mass_delta = 0.0
    v5_max_mass_delta = 0.0
    primary_in_centers = 0

    for i, t in enumerate(targets):
        b = replay(t["parent_move_seq"])
        wc = window_center(b)
        # V1 index parity (in-window only): window_coords(target_idx) == refuting_move
        if in_window[i]:
            v1_tot += 1
            tidx = int(target_idx[i])
            wq, wr = tidx // S, tidx % S
            recon = (wq - HALF + wc[0], wr - HALF + wc[1])
            if recon == (int(t["refuting_move"][0]), int(t["refuting_move"][1])):
                v1_ok += 1
            else:
                print(f"  [V1 FAIL] {t['pos_id']} recon {recon} != ref {t['refuting_move']}")

        # engine plane assembly
        st = GameState.from_board(b)
        t18, centers = st.to_tensor()
        centers = [tuple(int(x) for x in c) for c in centers]

        # V2 stone-kernel parity at every center
        for k, c in enumerate(centers):
            v2_tot += 1
            mine = stone_view_at(b, c[0], c[1])
            eng = t18[k][[0, 8]].astype(np.float32)
            if np.array_equal(mine, eng):
                v2_ok += 1
            else:
                print(f"  [V2 FAIL] {t['pos_id']} center {c} L1 {np.abs(mine-eng).sum()}")

        # V3 scalar-plane parity (planes 16/17 identical across clusters)
        v3_tot += 1
        my_pl = primary_planes(b)
        if (np.array_equal(my_pl[2], t18[0][16].astype(np.float32))
                and np.array_equal(my_pl[3], t18[0][17].astype(np.float32))):
            v3_ok += 1
        else:
            print(f"  [V3 FAIL] {t['pos_id']} scalar planes differ")

        model = eng_for(t["bucket"]).model

        # V4 gold forward parity: primary window IS a cluster center
        if wc in centers and in_window[i]:
            primary_in_centers += 1
            v4_tot += 1
            k = centers.index(wc)
            eng_planes = t18[k][[0, 8, 16, 17]].astype(np.float32)
            p_cache = raw_policy(model, my_pl)
            p_eng = raw_policy(model, eng_planes)
            tidx = int(target_idx[i])
            m_c, r_c = mass_rank(p_cache, tidx)
            m_e, r_e = mass_rank(p_eng, tidx)
            d = abs(m_c - m_e)
            v4_max_mass_delta = max(v4_max_mass_delta, d)
            if d < 1e-6 and r_c == r_e:
                v4_ok += 1
            else:
                print(f"  [V4 FAIL] {t['pos_id']} mass {m_c:.3e} vs {m_e:.3e} "
                      f"rank {r_c} vs {r_e}")

        # V5 npz round-trip: reloaded planes forward == fresh-rebuild forward
        if in_window[i]:
            v5_tot += 1
            npz = np.load(out_path, allow_pickle=True)
            p_npz = raw_policy(model, npz["planes"][i])
            p_fresh = raw_policy(model, my_pl)
            tidx = int(target_idx[i])
            d = abs(float(p_npz[tidx]) - float(p_fresh[tidx]))
            v5_max_mass_delta = max(v5_max_mass_delta, d)
            if d < 1e-9:
                v5_ok += 1
            else:
                print(f"  [V5 FAIL] {t['pos_id']} npz {p_npz[tidx]:.3e} vs {p_fresh[tidx]:.3e}")

    def gate(name, ok, tot, extra=""):
        verdict = "PASS" if ok == tot and tot > 0 else "FAIL"
        print(f"  {name}: {ok}/{tot}  {verdict}  {extra}")
        return ok == tot and tot > 0

    g1 = gate("V1 index parity (in-window)", v1_ok, v1_tot)
    g2 = gate("V2 stone-kernel parity (all centers)", v2_ok, v2_tot)
    g3 = gate("V3 scalar-plane parity (all 38)", v3_ok, v3_tot)
    g4 = gate("V4 gold forward parity (primary-in-centers)", v4_ok, v4_tot,
              f"max|d_mass|={v4_max_mass_delta:.2e}")
    g5 = gate("V5 npz round-trip (in-window)", v5_ok, v5_tot,
              f"max|d_mass|={v5_max_mass_delta:.2e}")
    print(f"  primary window IS a cluster center: {primary_in_centers}/{n_in} in-window")
    all_pass = g1 and g2 and g3 and g4 and g5
    print(f"\n  ===== VALIDATION {'PASS' if all_pass else 'FAIL'} =====")

    # ---- RAW-NET BASELINE (32 in-window) ----
    print("\n========== RAW-NET BASELINE (pre-fine-tune; 32 in-window) ==========")
    # t0 search-improved cross-ref (index-consistency note)
    t0 = {}
    if t0_path.exists():
        for l in t0_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                t0[r["pos_id"]] = r.get("parent", {})

    rows = []
    for i, t in enumerate(targets):
        if not in_window[i]:
            continue
        b = replay(t["parent_move_seq"])
        model = eng_for(t["bucket"]).model
        p_raw = raw_policy(model, primary_planes(b))           # primary-window raw
        m_raw, r_raw = mass_rank(p_raw, int(target_idx[i]))
        # deploy multi-window scatter-max raw prior (cross-ref, engine.infer)
        pol_sm, _v = eng_for(t["bucket"]).infer(b)
        p_sm = np.asarray(pol_sm, dtype=np.float64)
        m_sm, r_sm = mass_rank(p_sm, int(target_idx[i]))
        tp = t0.get(t["pos_id"], {})
        rows.append({
            "pos_id": t["pos_id"], "bucket": t["bucket"], "band": t["depth_band"],
            "raw_mass": m_raw, "raw_rank": r_raw,
            "sm_mass": m_sm, "sm_rank": r_sm,
            "t0_improved_mass": tp.get("ref_prior_mass"), "t0_improved_rank": tp.get("ref_rank"),
        })

    def summarize(label, subset):
        if not subset:
            return
        masses = np.array([x["raw_mass"] for x in subset])
        ranks = np.array([x["raw_rank"] for x in subset])
        n = len(subset)
        nz = int((masses < NEAR_ZERO).sum())
        r0 = int((ranks == 0).sum())
        rle9 = int((ranks < 10).sum())
        print(f"  [{label}] n={n}  rank0={r0}  rank<10={rle9}  ~0prior(<1e-3)={nz} "
              f"({100*nz/n:.0f}%)  mass[median={np.median(masses):.2e} "
              f"max={masses.max():.2e}]  rank[median={np.median(ranks):.0f} "
              f"max={int(ranks.max())}]")

    print("  (raw_mass/rank = primary-window policy-head prior on the saving move)")
    summarize("ALL in-window", rows)
    for bk in ("s150k", "s175k", "s200k"):
        summarize(bk, [x for x in rows if x["bucket"] == bk])

    print("\n  per-position (raw primary-window | deploy scatter-max | t0 improved):")
    print(f"  {'pos_id':24s} {'band':6s} {'raw_mass':>10s} {'rk':>4s} "
          f"{'sm_mass':>10s} {'rk':>4s} {'t0_mass':>10s} {'rk':>4s}")
    for x in rows:
        t0m = f"{x['t0_improved_mass']:.2e}" if x["t0_improved_mass"] is not None else "  -"
        t0r = x["t0_improved_rank"] if x["t0_improved_rank"] is not None else "-"
        print(f"  {x['pos_id']:24s} {x['band']:6s} {x['raw_mass']:10.2e} {x['raw_rank']:4d} "
              f"{x['sm_mass']:10.2e} {x['sm_rank']:4d} {t0m:>10s} {str(t0r):>4s}")

    return 0 if all_pass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--t0", default=str(DEFAULT_T0))
    ap.add_argument("--validate", action="store_true", help="run V1-V5 + baseline")
    args = ap.parse_args()
    return run(Path(args.corpus), Path(args.out), args.validate, Path(args.t0))


if __name__ == "__main__":
    raise SystemExit(main())
