#!/usr/bin/env python3
"""§D-OVERSPREAD D3 — does the TARGET reward building toward a TURN-fork? (turn-correct)

DRIVER D3 (PREREGISTRATION.md): the MCTS visit target may not reward building toward a
TURN-fork (``count_winning_turns >= 3``, the turn-correct analogue of the Rust depth-1
quiescence). Pure-terminal reward + diffuse visit mass teaches spread. No stored visits in
the banked replays -> REGEN MCTS on banked fork-approach positions and read the visit
distribution.

POOL (FIXED, encoding-independent, pure board replay — built ONCE, shared by every
checkpoint):
  turn-start snapshots where the MOVER's ``count_winning_turns in {1, 2}`` (one move from a
  turn-fork) AND >=1 legal stone RAISES ``count_winning_turns`` (a turn-fork-building move
  exists) AND >=1 in-window winning-turn cell. Subsampled to ~150-300, fixed seed.

FORK-BUILDING MOVE (turn-correct): a single legal stone ``c`` such that
``count_winning_turns(board_after_c, mover) > count_winning_turns(board_before, mover)``.
A turn places 2 stones; the first stone keeps the side to move (moves_remaining 2->1), so
placing ``c`` and re-measuring the mover's winning-turn set on the resulting (still-mover)
position is the turn-correct "did this move grow my multi-threat" test. (When the snapshot is
already on the last stone of a turn, moves_remaining==1, depth2 contributes nothing and the
test reduces to the depth-1 growth — still correct.)

PRIMARY metric per checkpoint: regen MCTS (Python ``KClusterMCTSBot`` — the §D-MULTICLUSTER
instrument, modest sims) -> visit distribution over root children. fork-affinity =
visit-mass fraction landing on fork-building moves. PROXY: eval-path MCTS != the Rust
self-play target generator (n_sims_full 600 + Dirichlet + completed_q). The TREND across the
arc is the signal, not the absolute level. CROSS-CHECK: cheap network POLICY-PRIOR mass on
fork-building moves, computed on ALL 11 checkpoints.

LIT iff target fork-affinity is LOW AND FALLS over the arc (pre-registered).

EVAL-ONLY, read-only. New file only. Zero geometry literals (all from spec / detector /
turn_wins primitive). Run:
  cd $REPO_ROOT && PYTHONPATH=. .venv/bin/python \
    scripts/diagnosis/overspread_d3_fork_affinity.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import is_off_window
from hexo_rl.encoding import lookup, normalize_encoding_name
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.k_cluster_mcts_bot import (
    KClusterMCTSBot,
    _Node,
    _aggregate_priors,
)
from hexo_rl.training.checkpoints import load_inference_model

# turn-correct primitive (NOT the depth-1 Rust count_winning_moves)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turn_wins import count_winning_turns, winning_turn_cells, FORK_THRESHOLD  # noqa: E402


# ----------------------------------------------------------------------------- pool build
def fork_building_moves(board, mover):
    """Legal single stones that RAISE the mover's count_winning_turns (turn-correct).

    Restrict the candidate set for cost: a fork-building move must touch the mover's
    existing winning-turn neighbourhood OR be adjacent to its own stones — but to stay
    HONEST (no geometry heuristic that could bias the metric) we test EVERY legal move.
    depth2_wins is already O(threats^2) and gated to threat cells, so this is bounded.
    """
    base = count_winning_turns(board, mover)
    builders = []
    for (q, r) in board.legal_moves():
        b2 = board.clone()
        try:
            b2.apply_move(q, r)
        except Exception:
            continue
        # After the first stone of a 2-stone turn the SAME side is to move; measure its
        # winning-turn set on the resulting position. If the single stone already finished
        # the game, that is the strongest possible "fork-build" (it IS a win) -> count it.
        if b2.check_win() and b2.winner() == mover:
            builders.append((q, r))
            continue
        if int(b2.current_player) == mover:
            if count_winning_turns(b2, mover) > base:
                builders.append((q, r))
        # If the stone ended the turn (current_player flipped), it cannot raise THIS turn's
        # finishing set (opp moves next) -> not a fork-building move by the turn-correct def.
    return builders, base


def build_pool(files, name, spec, want_steps, max_pos, seed):
    """One fixed pool of fork-approach snapshots, tagged by source checkpoint_step.

    Criteria (turn-correct): mover count_winning_turns in {1,2}, >=1 fork-building move
    exists, >=1 in-window winning-turn cell. Stored as move-prefix so any checkpoint can
    rebuild the exact board.
    """
    rows = []
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
            if d.get("game_length", 0) <= 0:
                continue
            src = int(d.get("checkpoint_step", 0))
            if src not in want_steps:
                continue
            mv = [(int(q), int(r)) for (q, r) in d["moves"]]
            b = Board.with_encoding_name(name)
            i, n = 0, len(mv)
            while i < n:
                cp = int(b.current_player)
                snap = b.clone()
                wt = winning_turn_cells(snap, cp)
                cwt = len(wt)
                if cwt in (1, 2):
                    inwin = [c for c in wt if not is_off_window(snap, c, spec)]
                    if inwin:
                        builders, base = fork_building_moves(snap, cp)
                        if builders:
                            rows.append({
                                "prefix": mv[:i],
                                "side": cp,
                                "src_step": src,
                                "cwt": cwt,
                                "builders": [list(c) for c in builders],
                                "n_legal": len(snap.legal_moves()),
                            })
                # advance one turn
                while i < n:
                    q, r = mv[i]
                    try:
                        b.apply_move(q, r)
                    except Exception:
                        i = n
                        break
                    i += 1
                    if b.check_win():
                        break
                    if int(b.current_player) != cp:
                        break
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_pos:
        rows = rows[:max_pos]
    return rows


def rebuild(name, prefix):
    b = Board.with_encoding_name(name)
    st = GameState.from_board(b)
    for (q, r) in prefix:
        st = st.apply_move(b, q, r)
    return b, st


# ------------------------------------------------------------------ MCTS visit affinity
def mcts_visit_affinity(bot, name, row):
    """Regen MCTS on the snapshot; return fork-building visit-mass fraction.

    Reads the root's child visit counts directly (the visit TARGET signal). fork-building
    moves are recomputed on the rebuilt board (encoding-independent), matched against the
    root children by (q,r).
    """
    b, st = rebuild(name, row["prefix"])
    builders = {tuple(c) for c in row["builders"]}
    root = _Node(prior=0.0)
    bot._expand(root, b)
    if root.is_terminal or not root.children:
        return None
    for _ in range(bot.n_sims):
        bot._simulate(root, b)
    total = 0
    fork_mass = 0
    for a, child in root.children.items():
        v = int(child.visits)
        total += v
        if tuple(a) in builders:
            fork_mass += v
    if total == 0:
        return None
    return fork_mass / total


# ---------------------------------------------------------------- policy-prior affinity
def policy_prior_affinity(model, device, kept, name, rows):
    """Cheap cross-check on ALL checkpoints: network policy-prior (scatter-max aggregated,
    same pooling the bot uses) mass on fork-building moves. One forward per snapshot.
    """
    out = []
    for row in rows:
        b, st = rebuild(name, row["prefix"])
        legal = list(b.legal_moves())
        if not legal:
            continue
        tensor, centers = st.to_tensor()
        if tensor.shape[1] != model.in_channels:
            tensor = tensor[:, kept]
        view_size = tensor.shape[-1]
        x = torch.from_numpy(tensor).float().to(device)
        with torch.no_grad():
            log_p, _v, _ = model(x)
        priors = _aggregate_priors(
            log_p.cpu().numpy(), list(centers), legal, view_size
        )
        builders = {tuple(c) for c in row["builders"]}
        mass = sum(p for (mv, p) in zip(legal, priors) if tuple(mv) in builders)
        out.append(mass)
    return out


def discover_ckpts(dirs):
    found = {}
    for dd in dirs:
        for cp in Path(dd).glob("checkpoint_*.pt"):
            tail = cp.stem.replace("checkpoint_", "")
            digits = tail.split("_")[0]
            if not digits.isdigit():
                continue
            step = int(digits.lstrip("0") or "0")
            found.setdefault(step, cp)
    return found


def boot_ci(vals, nboot, rng):
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    arr = np.asarray(vals, dtype=np.float64)
    bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(nboot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--replays", nargs="+",
                    default=sorted(str(p) for p in Path(
                        "investigation/coherence_2026-06-08/replays").glob(
                        "games_2026-06-0*.jsonl")))
    ap.add_argument("--ckpt-dirs", nargs="+",
                    default=["investigation/coherence_2026-06-08/checkpoints",
                             "investigation/fragility_2026-06-07/checkpoints"])
    ap.add_argument("--pool-steps", type=int, nargs="+", default=[30000, 53000, 87500],
                    help="replay checkpoint_step buckets to draw pool snapshots from")
    ap.add_argument("--mcts-steps", type=int, nargs="+",
                    default=[30000, 50000, 75000, 87500],
                    help="checkpoint steps to run REGEN MCTS on (subset, to bound cost). "
                         "50000 stands in for the 53000 replay bucket (no 53k ckpt).")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--max-pos", type=int, default=240)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/d3_fork_affinity.json")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    kept = list(spec.kept_plane_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[pool] building fork-approach pool (cwt in {{1,2}} & fork-builder exists & "
          f">=1 in-window) from {len(args.replays)} replays, steps={args.pool_steps} ...",
          flush=True)
    rows = build_pool(args.replays, name, spec, set(args.pool_steps),
                      args.max_pos, args.seed)
    by_src = Counter(r["src_step"] for r in rows)
    by_cwt = Counter(r["cwt"] for r in rows)
    print(f"[pool] n={len(rows)}  by_src={dict(sorted(by_src.items()))}  "
          f"by_cwt={dict(sorted(by_cwt.items()))}", flush=True)
    if not rows:
        print("NO fork-approach positions -> cannot rate D3"); return 1

    ckpts = discover_ckpts(args.ckpt_dirs)
    print(f"[ckpts] available steps: {sorted(ckpts)}", flush=True)
    all_steps = sorted(ckpts)
    mcts_steps = [s for s in args.mcts_steps if s in ckpts]
    rng = np.random.default_rng(args.seed)

    # baseline: random-policy affinity = expected fork-mass if visits/priors were uniform
    # over legal moves (the NULL "no fork credit at all"). builders/legal per row.
    uniform_aff = float(np.mean([len(r["builders"]) / max(r["n_legal"], 1) for r in rows]))
    print(f"[null] uniform-over-legal fork-mass (no credit) = {uniform_aff:.4f}", flush=True)

    out = {"encoding": name, "n_pool": len(rows), "sims": args.sims,
           "pool_steps": args.pool_steps, "by_src": dict(sorted(by_src.items())),
           "by_cwt": dict(sorted(by_cwt.items())), "uniform_fork_mass": uniform_aff,
           "null_note": "uniform_fork_mass = fork-building visit mass if play were uniform "
                        "over legal moves (no fork credit). MCTS/policy ABOVE this = the "
                        "target gives fork-building some credit; AT/BELOW = none.",
           "proxy_disclosure": "eval-path KClusterMCTSBot (sims=%d, no Dirichlet, no "
                               "completed_q) != Rust self-play target generator "
                               "(n_sims_full 600 + Dirichlet + completed_q). The TREND "
                               "across the arc is the signal, not the absolute level. "
                               "50000 ckpt stands in for the 53000 replay bucket." % args.sims,
           "policy_prior_all_ckpts": [], "mcts": []}

    # ---- POLICY-PRIOR proxy on ALL 11 checkpoints (cheap cross-check) ----
    print(f"\n{'='*78}\nPOLICY-PRIOR fork-mass (scatter-max prior) — ALL checkpoints\n{'='*78}",
          flush=True)
    print(f"{'step':>7} | {'policy fork-mass':>18} | {'95% CI':>20}")
    print("-" * 52)
    for st in all_steps:
        model, _sp, _lab = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        vals = policy_prior_affinity(model, device, kept, name, rows)
        m = float(np.mean(vals)) if vals else float("nan")
        ci = boot_ci(vals, args.nboot, rng)
        out["policy_prior_all_ckpts"].append(
            {"step": st, "policy_fork_mass": m, "ci": ci, "n": len(vals)})
        print(f"{st:>7} | {m:>18.4f} | [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- REGEN MCTS on the subset of checkpoints ----
    print(f"\n{'='*78}\nREGEN-MCTS visit fork-affinity (sims={args.sims}) — subset {mcts_steps}\n{'='*78}",
          flush=True)
    print(f"{'step':>7} | {'MCTS fork-mass':>15} | {'95% CI':>20} | {'n':>4} | {'sec':>6}")
    print("-" * 64)
    for st in mcts_steps:
        model, _sp, _lab = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        bot = KClusterMCTSBot(model, device, n_sims=args.sims, temperature=0.0,
                              kept_plane_indices=kept)
        vals, t0 = [], time.time()
        for j, row in enumerate(rows):
            a = mcts_visit_affinity(bot, name, row)
            if a is not None:
                vals.append(a)
            if (j + 1) % 60 == 0:
                print(f"    [{st}] {j+1}/{len(rows)} mean={np.mean(vals):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        m = float(np.mean(vals)) if vals else float("nan")
        ci = boot_ci(vals, args.nboot, rng)
        dt = time.time() - t0
        out["mcts"].append({"step": st, "mcts_fork_mass": m, "ci": ci, "n": len(vals),
                            "sec": dt})
        print(f"{st:>7} | {m:>15.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {len(vals):>4} | {dt:>6.0f}",
              flush=True)
        del model, bot
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- arc trend ----
    pol = out["policy_prior_all_ckpts"]
    mc = out["mcts"]
    out["arc"] = {}
    if len(pol) >= 2:
        a, b = pol[0], pol[-1]
        out["arc"]["policy_delta"] = {"from": a["step"], "to": b["step"],
                                      "delta": b["policy_fork_mass"] - a["policy_fork_mass"]}
    if len(mc) >= 2:
        a, b = mc[0], mc[-1]
        out["arc"]["mcts_delta"] = {"from": a["step"], "to": b["step"],
                                    "delta": b["mcts_fork_mass"] - a["mcts_fork_mass"]}

    print(f"\n{'='*78}\nARC TREND\n{'='*78}")
    if "policy_delta" in out["arc"]:
        d = out["arc"]["policy_delta"]
        print(f"POLICY-PRIOR fork-mass {d['from']}->{d['to']}: "
              f"{pol[0]['policy_fork_mass']:.4f} -> {pol[-1]['policy_fork_mass']:.4f} "
              f"(d={d['delta']:+.4f})  [null uniform={uniform_aff:.4f}]")
    if "mcts_delta" in out["arc"]:
        d = out["arc"]["mcts_delta"]
        print(f"MCTS-VISIT  fork-mass {d['from']}->{d['to']}: "
              f"{mc[0]['mcts_fork_mass']:.4f} -> {mc[-1]['mcts_fork_mass']:.4f} "
              f"(d={d['delta']:+.4f})  [null uniform={uniform_aff:.4f}]")
    print("\nLIT iff target fork-affinity LOW *and* FALLS over the arc (pre-registered).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
