"""D-TACTICAL — T0: deploy Gumbel-SH-150 (g=0) baseline + policy-prior decomposition.

Establishes the GAP these positions represent for the DEPLOY search (not just the raw
value head), and decomposes WHY the search misses the refutation:

  (a) POST-blunder proven-loss position: deploy root value (value-blind iff >= MISS),
      played move. -> "the deploy search mis-evaluates a proven forced loss."
  (b) PARENT decision position: does the deploy POLICY assign nonzero prior to the
      refuting/saving move ref_best, and does the search play the blunder instead?
      prior~0 on ref_best  -> MCTS structurally can't explore it (override-the-prior fix)
      prior>0 but not played -> selection/averaging dilutes it (solver-backup fix)

GPU. Uses the deploy net (checkpoint_00272357 — the net whose value blind-spot was
characterized, matching the corpus net_value). Defines M = the deploy-search-miss set.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, "/home/timmy/Work/Hexo/hexo_rl")
import numpy as np
import torch
import engine
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board

ENC = "v6_live2_ls"
MISS_THRESH = -0.05

# Per-bucket deploy checkpoints — the nets whose value blind-spot these positions
# exhibit (V2 confirmed: stored net_value matches these, NOT checkpoint_00272357).
CKPT_DIR = "reports/d_ladder_2026-06-24/ckpts"
BUCKET_CKPT = {
    "s150k": "checkpoint_00150000.pt",
    "s175k": "checkpoint_00175000.pt",
    "s200k": "checkpoint_00200000.pt",
}
DEPLOY_KNOBS = dict(gumbel_m=16, n_sims_full=150, c_visit=50.0, c_scale=1.0,
                    c_puct=1.5, dirichlet_enabled=False)


def _engine_for(bucket, ckpt_dir, device, cache):
    """Lazily build (and cache) a deploy engine for the position's bucket."""
    if bucket not in cache:
        from scripts.eval.gumbel_greedy_bot import _build_engine
        ckpt = os.path.join(ckpt_dir, BUCKET_CKPT[bucket])
        cache[bucket] = _build_engine(ckpt, ENC, device)
        print(f"  [engine] built {bucket} <- {BUCKET_CKPT[bucket]}")
    return cache[bucket]


def _replay(seq):
    b = engine.Board.with_encoding_name(ENC)
    for (q, r) in seq:
        b.apply_move(int(q), int(r))
    return b


def _deploy_search(eng, knobs, board, rng):
    return run_gumbel_on_board(
        eng, board,
        n_sims=int(knobs["n_sims_full"]), m=int(knobs["gumbel_m"]),
        c_visit=float(knobs["c_visit"]), c_scale=float(knobs["c_scale"]),
        c_puct=float(knobs["c_puct"]), dirichlet=bool(knobs.get("dirichlet_enabled", False)),
        gumbel_scale=0.0, rng=rng)


def _policy_prior_on(board, policy, move):
    """Prior mass + rank of `move` in the improved policy (window-relative to_flat)."""
    if move is None:
        return None, None
    try:
        idx = board.to_flat(int(move[0]), int(move[1]))
    except Exception:
        return None, None
    if idx is None or idx >= len(policy):
        return None, None
    mass = float(policy[idx])
    rank = int((policy > policy[idx]).sum())  # 0 = top
    return mass, rank


def _get(rec, *keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def run(corpus_path, out_dir, ckpt_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knobs = DEPLOY_KNOBS
    print(f"  deploy knobs: {knobs}")
    engines = {}
    records = [json.loads(l) for l in open(corpus_path) if l.strip()]
    rng = np.random.default_rng(0)
    out = []
    for i, rec in enumerate(records):
        pos_id = _get(rec, "pos_id", default=f"pos{i}")
        bucket = _get(rec, "bucket", default="s150k")
        eng = _engine_for(bucket, ckpt_dir, device, engines)
        post_seq = _get(rec, "postblunder_move_seq", "move_seq", "moves")
        parent_seq = _get(rec, "parent_move_seq")
        ref_best = _get(rec, "refuting_move", "ref_best")
        band = _get(rec, "depth_band", "band", default="unknown")

        # (a) post-blunder proven-loss position
        post = {}
        if post_seq is not None:
            b = _replay(post_seq)
            g = _deploy_search(eng, knobs, b, rng)
            rv = g.get("root_value")
            post = {"root_value": rv, "played_move": g.get("played_move"),
                    "miss": (rv is not None and rv >= MISS_THRESH)}

        # (b) parent decision position: policy prior on the refuting move
        parent = {}
        if parent_seq is not None:
            pb = _replay(parent_seq)
            gp = _deploy_search(eng, knobs, pb, rng)
            pol = gp.get("improved_policy")
            mass, rank = (None, None)
            if pol is not None:
                mass, rank = _policy_prior_on(pb, np.asarray(pol), ref_best)
            parent = {"root_value": gp.get("root_value"),
                      "played_move": gp.get("played_move"),
                      "ref_best": ref_best, "ref_prior_mass": mass, "ref_rank": rank,
                      "played_is_ref": (gp.get("played_move") == (list(ref_best)
                                        if ref_best else None))}

        out.append({"pos_id": pos_id, "band": band, "post": post, "parent": parent})
        if (i + 1) % 5 == 0:
            print(f"  T0 {i+1}/{len(records)}")

    os.makedirs(out_dir, exist_ok=True)
    rp = os.path.join(out_dir, "t0_results.jsonl")
    with open(rp, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    _summary(out, out_dir)
    return out


def _summary(out, out_dir):
    n = len(out)
    miss = [r for r in out if r["post"].get("miss")]
    # policy decomposition over parents with a ref prior
    parents = [r for r in out if r["parent"].get("ref_prior_mass") is not None]
    near_zero = [r for r in parents if r["parent"]["ref_prior_mass"] < 1e-3]
    nonzero_not_played = [r for r in parents
                          if r["parent"]["ref_prior_mass"] >= 1e-3
                          and not r["parent"]["played_is_ref"]]
    lines = ["# D-TACTICAL T0 — deploy baseline + policy-prior decomposition\n\n",
             f"n={n}\n",
             f"- deploy MIS-EVALUATES (root_value >= {MISS_THRESH}) the proven loss: "
             f"{len(miss)}/{n} = {100*len(miss)/max(n,1):.0f}%  (= M, the search-miss set)\n",
             f"- parents with a ref_best prior measured: {len(parents)}\n",
             f"  - ref prior ~0 (<1e-3): {len(near_zero)}/{len(parents) or 1} "
             f"-> MCTS structurally cannot explore the refutation (override-prior fix)\n",
             f"  - ref prior >0 but search did NOT play it: {len(nonzero_not_played)} "
             f"-> selection/averaging dilutes it (solver-backup fix)\n"]
    with open(os.path.join(out_dir, "t0_summary.md"), "w") as f:
        f.writelines(lines)
    print("".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="reports/d_tactical_2026-06-26/corpus.jsonl")
    ap.add_argument("--out-dir", default="reports/d_tactical_2026-06-26")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR,
                    help="dir with per-bucket deploy ckpts (matches stored net_value)")
    args = ap.parse_args()
    run(args.corpus, args.out_dir, args.ckpt_dir)
