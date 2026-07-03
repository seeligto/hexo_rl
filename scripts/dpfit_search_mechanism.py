#!/usr/bin/env python3
"""D-PFIT P1b — Deliverable (B): does a boosted policy prior SURVIVE the deploy search?

THE decisive training-z necessary-condition test. Mechanism premise: training-z
surfaces the saving move → policy prior rises → MCTS finds it WITHOUT the solver.
A frozen-trunk one-hot fit (Deliverable A) drives a NEAR-MAXIMAL prior boost
(deploy root prior 0.15 → ~1.0, verified) — stronger than training-z could
realistically reach. So the fitted-net deploy search is the UPPER BOUND on the
policy lever: if a ~one-hot prior does NOT flip the search, the policy route is
DEAD and the BLIND VALUE is the binding constraint.

Three conditions per in-window trap PARENT board (deploy knobs: gumbel_m=16,
n_sims=150, c_visit=50, c_scale=1, c_puct=1.5, dirichlet=False, gumbel_scale=0.0
— the deterministic deploy head, matching run_t0):

  CONTROL (raw net)        : raw deploy played_move (the dilution baseline).
  B1-insample (fit incl.)  : insample-fitted head (value UNCHANGED=blind) deploy
                             played_move — the upper bound.
  B1-heldout  (fit excl.)  : game-disjoint LOO head → deploy played_move
                             (memorization vs generalization).

Classification per condition: played == saving (refuting_move) / == blunder
(blunder_move) / other. Plus deploy ROOT prior on the saving move (engine.infer
scatter-max) before vs after the fit (transfer check), and saving-vs-blunder
visit count + Q in the search (the non-flip diagnosis: prior-transferred-but-
value-diluted vs prior-not-transferred).

Collateral-on-(B): raw vs insample-fitted deploy played_move on N normal
(non-trap) same-bucket positions → top-1 disagreement + worse-move proxy.

B2 prior-injection sweep: SKIPPED. `run_gumbel_on_board` reads root priors from
the NN (`tree.get_root_children_info`) with no clean external override hook; the
near-one-hot fit IS the maximal prior injection, so B1 is the upper bound
regardless (documented, not a gap).

Run (after dpfit_fit.py): .venv/bin/python scripts/dpfit_search_mechanism.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import engine  # noqa: E402

from scripts.eval.gumbel_greedy_bot import _build_engine  # noqa: E402
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from scripts.dpfit_export import replay, ENC, CKPT_DIR, BUCKET_CKPT, DEFAULT_CORPUS  # noqa: E402
from scripts.dpfit_fit import NPZ, FIT_DIR, BUCKETS  # noqa: E402

DEPLOY = dict(n_sims=150, m=16, c_visit=50.0, c_scale=1.0, c_puct=1.5,
              dirichlet=False, gumbel_scale=0.0)
REPORT = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "dpfit_p1b_report.md"


def deploy_search(eng, board) -> Dict[str, Any]:
    rng = np.random.default_rng(0)  # gumbel_scale=0 ⇒ deterministic; rng unused for noise
    return run_gumbel_on_board(eng, board, **DEPLOY, rng=rng)


def classify(played, saving, blunder) -> str:
    if played is None:
        return "none"
    pl = (int(played[0]), int(played[1]))
    if pl == saving:
        return "saving"
    if pl == blunder:
        return "blunder"
    return "other"


def saving_prior(eng, board, saving_idx: int) -> float:
    pol, _v = eng.infer(board)
    return float(pol[saving_idx])


def is_primary_center(board) -> bool:
    """Is the to_flat window_center one of the deploy K cluster centers?"""
    from scripts.dpfit_export import window_center
    wc = window_center(board)
    _t, centers = GameState.from_board(board).to_tensor()
    centers = [(int(c[0]), int(c[1])) for c in centers]
    return wc in centers


def restore_head(model, head_sd: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(head_sd, strict=False)


# ---------------------------------------------------------------------------
# normal (non-trap) boards for collateral search
# ---------------------------------------------------------------------------

def normal_boards(corpus, bucket_pids: List[str], stride: int = 6,
                  floor: int = 20, cap: int = 10) -> List[engine.Board]:
    seen = set()
    boards = []
    for pid in bucket_pids:
        seq = corpus[pid]["parent_move_seq"]
        L = len(seq)
        for cut in range(L - stride, floor - 1, -stride):
            sub = seq[:cut]
            key = tuple((int(q), int(r)) for q, r in sub)
            if key in seen:
                continue
            seen.add(key)
            b = replay(sub)
            if b.check_win() or b.legal_move_count() == 0:
                continue
            boards.append(b)
            if len(boards) >= cap:
                return boards
    return boards


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(device_str: str, collat_cap: int) -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    d = np.load(NPZ, allow_pickle=True)
    inw = d["in_window"]
    pos_ids = d["pos_id"][inw].astype(str)
    buckets = d["bucket"][inw].astype(str)
    bands = d["depth_band"][inw].astype(str)
    recs = [json.loads(l) for l in DEFAULT_CORPUS.read_text().splitlines() if l.strip()]
    corpus = {r["pos_id"]: r for r in recs if r.get("is_proven_core")}

    # engines per bucket: raw, insample-fitted, work (for heldout head-swap)
    raw_eng, ins_eng, work_eng, loo_heads = {}, {}, {}, {}
    for bk in BUCKETS:
        raw_eng[bk] = _build_engine(str(CKPT_DIR / BUCKET_CKPT[bk]), ENC, device)
        ins_eng[bk] = _build_engine(str(FIT_DIR / f"insample_{bk}.pt"), ENC, device)
        work_eng[bk] = _build_engine(str(CKPT_DIR / BUCKET_CKPT[bk]), ENC, device)
        loo_heads[bk] = torch.load(FIT_DIR / f"loo_heads_{bk}.pt", map_location=device,
                                   weights_only=False)
        print(f"  [engines built] {bk}")

    rows = []
    t0 = time.time()
    for i, pid in enumerate(pos_ids):
        bk = buckets[i]
        r = corpus[pid]
        saving = (int(r["refuting_move"][0]), int(r["refuting_move"][1]))
        blunder = (int(r["blunder_move"][0]), int(r["blunder_move"][1]))
        board = replay(r["parent_move_seq"])
        sidx = board.to_flat(*saving)
        prim_ctr = is_primary_center(board)

        # CONTROL
        g_raw = deploy_search(raw_eng[bk], board)
        c_raw = classify(g_raw["played_move"], saving, blunder)
        sm_raw = saving_prior(raw_eng[bk], board, sidx)

        # B1-insample
        g_ins = deploy_search(ins_eng[bk], board)
        c_ins = classify(g_ins["played_move"], saving, blunder)
        sm_ins = saving_prior(ins_eng[bk], board, sidx)

        # B1-heldout (restore LOO head into work engine)
        restore_head(work_eng[bk].model, loo_heads[bk][pid])
        g_ho = deploy_search(work_eng[bk], board)
        c_ho = classify(g_ho["played_move"], saving, blunder)
        sm_ho = saving_prior(work_eng[bk], board, sidx)

        # non-flip diagnosis: saving vs blunder visits + Q in the insample search
        sv_visits = g_ins["child_visits"].get(saving, 0)
        bl_visits = g_ins["child_visits"].get(blunder, 0)
        sv_q = g_ins["child_q"].get(saving)
        bl_q = g_ins["child_q"].get(blunder)

        rows.append({
            "pos_id": pid, "bucket": bk, "band": str(bands[i]),
            "primary_center": prim_ctr,
            "saving": saving, "blunder": blunder,
            "ctrl_played": g_raw["played_move"], "ctrl_class": c_raw,
            "ins_played": g_ins["played_move"], "ins_class": c_ins,
            "ho_played": g_ho["played_move"], "ho_class": c_ho,
            "sm_raw": sm_raw, "sm_ins": sm_ins, "sm_ho": sm_ho,
            "root_value_ins": g_ins.get("root_value"),
            "sv_visits": int(sv_visits), "bl_visits": int(bl_visits),
            "sv_q": (float(sv_q) if sv_q is not None else None),
            "bl_q": (float(bl_q) if bl_q is not None else None),
        })
        if (i + 1) % 8 == 0:
            print(f"  search {i+1}/{len(pos_ids)}  ({time.time()-t0:.0f}s)")

    # ---- collateral search (raw vs insample-fitted on normal positions) ----
    collat = {}
    for bk in BUCKETS:
        bpids = [pid for i, pid in enumerate(pos_ids) if buckets[i] == bk]
        nbs = normal_boards(corpus, bpids, cap=collat_cap)
        disagree = 0
        zero_prior_played = 0   # fitted plays a move the RAW net gave ~0 prior (proxy: worse)
        for b in nbs:
            gr = deploy_search(raw_eng[bk], b)["played_move"]
            gf = deploy_search(ins_eng[bk], b)["played_move"]
            if gr is None or gf is None:
                continue
            if tuple(gr) != tuple(gf):
                disagree += 1
                pol_raw, _ = raw_eng[bk].infer(b)
                fidx = b.to_flat(int(gf[0]), int(gf[1]))
                if 0 <= fidx < len(pol_raw) and pol_raw[fidx] < 1e-3:
                    zero_prior_played += 1
        collat[bk] = {"n": len(nbs), "disagree": disagree,
                      "disagree_rate": (disagree / len(nbs)) if nbs else float("nan"),
                      "zero_prior_played": zero_prior_played}
        print(f"  [collateral {bk}] n={len(nbs)} disagree={disagree} "
              f"({collat[bk]['disagree_rate']:.2f}) zero_prior_played={zero_prior_played}")

    out = {"rows": rows, "collateral": collat, "deploy_knobs": DEPLOY}
    _report(out)
    return out


def _counts(rows, key):
    from collections import Counter
    return Counter(r[key] for r in rows)


def _report(out: Dict[str, Any]) -> None:
    rows = out["rows"]
    n = len(rows)
    cc = _counts(rows, "ctrl_class")
    ci = _counts(rows, "ins_class")
    ch = _counts(rows, "ho_class")

    def flip_saving(key):
        return sum(1 for r in rows if r[key] == "saving")

    def flip_offblunder(key):
        return sum(1 for r in rows if r[key] != "blunder")

    lines = []
    lines.append("# D-PFIT P1b — Deliverable (B): policy-prior survival under the deploy search\n")
    lines.append(f"\ndeploy knobs: {out['deploy_knobs']}\n")
    lines.append(f"\nn in-window traps = {n}\n")
    lines.append("\n## Flip table (played-move classification)\n\n")
    lines.append("| condition | saving | blunder | other | none | flip→saving | flip-off-blunder |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for label, key, ctr in [("CONTROL (raw)", "ctrl_class", cc),
                            ("B1-insample", "ins_class", ci),
                            ("B1-heldout", "ho_class", ch)]:
        lines.append(f"| {label} | {ctr.get('saving',0)} | {ctr.get('blunder',0)} | "
                     f"{ctr.get('other',0)} | {ctr.get('none',0)} | "
                     f"{flip_saving(key)}/{n} | {flip_offblunder(key)}/{n} |\n")

    # per-bucket
    lines.append("\n## Per-bucket flip→saving\n\n")
    lines.append("| bucket | n | CONTROL | B1-insample | B1-heldout |\n|---|---|---|---|---|\n")
    for bk in BUCKETS:
        br = [r for r in rows if r["bucket"] == bk]
        nb = len(br)
        lines.append(f"| {bk} | {nb} | {sum(1 for r in br if r['ctrl_class']=='saving')}/{nb} | "
                     f"{sum(1 for r in br if r['ins_class']=='saving')}/{nb} | "
                     f"{sum(1 for r in br if r['ho_class']=='saving')}/{nb} |\n")

    # primary-center split (transfer cleanliness)
    pc = [r for r in rows if r["primary_center"]]
    npc = [r for r in rows if not r["primary_center"]]
    lines.append("\n## B1-insample flip→saving by primary-window-is-deploy-center\n\n")
    lines.append(f"- primary IS center (n={len(pc)}): "
                 f"{sum(1 for r in pc if r['ins_class']=='saving')}/{len(pc)} flip; "
                 f"mean deploy saving prior {np.mean([r['sm_ins'] for r in pc]):.3f}\n")
    lines.append(f"- primary NOT center (n={len(npc)}): "
                 f"{sum(1 for r in npc if r['ins_class']=='saving')}/{len(npc)} flip; "
                 f"mean deploy saving prior {np.mean([r['sm_ins'] for r in npc]):.3f}\n")

    # prior transfer
    sm_raw = np.array([r["sm_raw"] for r in rows])
    sm_ins = np.array([r["sm_ins"] for r in rows])
    lines.append("\n## Deploy ROOT prior on saving move (engine.infer scatter-max)\n\n")
    lines.append(f"- raw : median {np.median(sm_raw):.3f}  mean {sm_raw.mean():.3f}  "
                 f">0.5: {(sm_raw>0.5).sum()}/{n}\n")
    lines.append(f"- insample-fit : median {np.median(sm_ins):.3f}  mean {sm_ins.mean():.3f}  "
                 f">0.5: {(sm_ins>0.5).sum()}/{n}\n")
    lines.append(f"- prior transferred (sm_ins>0.5 i.e. boost reached deploy root): "
                 f"{(sm_ins>0.5).sum()}/{n}\n")

    # NON-FLIP diagnosis for B1-insample
    nonflip = [r for r in rows if r["ins_class"] != "saving"]
    transferred_nonflip = [r for r in nonflip if r["sm_ins"] > 0.5]
    lines.append("\n## NON-FLIP diagnosis (B1-insample did NOT play saving)\n\n")
    lines.append(f"- non-flips: {len(nonflip)}/{n}\n")
    lines.append(f"- of those, prior DID transfer to deploy root (sm_ins>0.5): "
                 f"{len(transferred_nonflip)}/{len(nonflip)} "
                 f"→ BLIND VALUE diluted a near-one-hot prior (search-diluted)\n")
    lines.append(f"- prior did NOT transfer (sm_ins<=0.5): "
                 f"{len(nonflip)-len(transferred_nonflip)}/{len(nonflip)} "
                 f"→ multi-window decoding gap (fit window != deploy center)\n")
    lines.append("\n### per non-flip: saving vs blunder visits + Q (insample search)\n\n")
    lines.append("| pos_id | bkt | band | primC | sm_raw→ins | ins_class | "
                 "sv_visits | bl_visits | sv_Q | bl_Q | root_v |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in nonflip:
        svq = f"{r['sv_q']:.2f}" if r["sv_q"] is not None else "-"
        blq = f"{r['bl_q']:.2f}" if r["bl_q"] is not None else "-"
        rv = f"{r['root_value_ins']:.2f}" if r["root_value_ins"] is not None else "-"
        lines.append(f"| {r['pos_id']} | {r['bucket']} | {r['band']} | "
                     f"{'Y' if r['primary_center'] else 'n'} | "
                     f"{r['sm_raw']:.2f}→{r['sm_ins']:.2f} | {r['ins_class']} | "
                     f"{r['sv_visits']} | {r['bl_visits']} | {svq} | {blq} | {rv} |\n")

    # collateral
    lines.append("\n## Collateral-on-(B): raw vs insample-fit deploy played move (normal positions)\n\n")
    lines.append("| bucket | n | disagree | rate | fit-plays-~0-prior-move |\n|---|---|---|---|---|\n")
    for bk, cm in out["collateral"].items():
        lines.append(f"| {bk} | {cm['n']} | {cm['disagree']} | {cm['disagree_rate']:.2f} | "
                     f"{cm['zero_prior_played']} |\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(lines))
    print("".join(lines))
    # also dump raw rows for audit
    (FIT_DIR / "search_B_rows.json").write_text(json.dumps(out, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--collat-cap", type=int, default=10)
    args = ap.parse_args()
    run(args.device, args.collat_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
