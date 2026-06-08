#!/usr/bin/env python3
"""§D-OVERSPREAD D3 RED-TEAM — split fork-affinity into outright-WIN vs true RAISE builders.

The D3 OUT verdict rests on "fork-affinity HIGH (155-233x null)". But ~52% of the pool's
"builder" moves are OUTRIGHT WINS (a single stone finishes the game NOW) — that is FINISHING
affinity (the O1 concept), NOT fork-BUILDING. This script re-measures, on the SAME pool, the
policy-prior mass on:
  - ALL builders          (the D3 metric)
  - WIN-only builders     (outright single-stone wins)
  - RAISE-only builders   (genuinely raise count_winning_turns WITHOUT winning = true fork-build)
over ALL 11 checkpoints, plus the MCTS visit mass on the same three subsets on the 4 MCTS
checkpoints. If the RAISE-only (true fork-build) affinity is LOW and FALLS while the headline
is propped up by WIN affinity, the D3 OUT verdict is challenged.

EVAL-ONLY, read-only, new file. Reuses the D3 script's helpers verbatim (imported).
"""
from __future__ import annotations

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
from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot, _Node, _aggregate_priors
from hexo_rl.training.checkpoints import load_inference_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turn_wins import count_winning_turns, winning_turn_cells  # noqa: E402


def classify_builders(board, mover):
    """Return (win_builders, raise_builders) — outright single-stone wins vs true cwt-raises."""
    base = count_winning_turns(board, mover)
    win_b, raise_b = [], []
    for (q, r) in board.legal_moves():
        b2 = board.clone()
        try:
            b2.apply_move(q, r)
        except Exception:
            continue
        if b2.check_win() and b2.winner() == mover:
            win_b.append((q, r))
            continue
        if int(b2.current_player) == mover and count_winning_turns(b2, mover) > base:
            raise_b.append((q, r))
    return win_b, raise_b


def build_pool(files, name, spec, want_steps, max_pos, seed):
    rows = []
    for fn in files:
        try:
            fh = open(fn)
        except FileNotFoundError:
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
                        win_b, raise_b = classify_builders(snap, cp)
                        if win_b or raise_b:
                            rows.append({
                                "prefix": mv[:i], "side": cp, "src_step": src, "cwt": cwt,
                                "win_b": [list(c) for c in win_b],
                                "raise_b": [list(c) for c in raise_b],
                                "n_legal": len(snap.legal_moves()),
                            })
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


def policy_mass_split(model, device, kept, name, rows):
    """Per-row policy-prior mass on win-builders, raise-builders, all-builders.

    Mass is computed only over rows that HAVE that builder type (so the raise metric is the
    pure fork-build credit, undiluted by rows with no fork-build move)."""
    win_v, raise_v, all_v = [], [], []
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
        priors = _aggregate_priors(log_p.cpu().numpy(), list(centers), legal, view_size)
        pmap = {tuple(mv): p for mv, p in zip(legal, priors)}
        win_set = {tuple(c) for c in row["win_b"]}
        raise_set = {tuple(c) for c in row["raise_b"]}
        all_set = win_set | raise_set
        if win_set:
            win_v.append(sum(pmap.get(c, 0.0) for c in win_set))
        if raise_set:
            raise_v.append(sum(pmap.get(c, 0.0) for c in raise_set))
        all_v.append(sum(pmap.get(c, 0.0) for c in all_set))
    return win_v, raise_v, all_v


def mcts_mass_split(bot, name, rows):
    win_v, raise_v, all_v = [], [], []
    for row in rows:
        b, st = rebuild(name, row["prefix"])
        win_set = {tuple(c) for c in row["win_b"]}
        raise_set = {tuple(c) for c in row["raise_b"]}
        all_set = win_set | raise_set
        root = _Node(prior=0.0)
        bot._expand(root, b)
        if root.is_terminal or not root.children:
            continue
        for _ in range(bot.n_sims):
            bot._simulate(root, b)
        total = sum(int(c.visits) for c in root.children.values())
        if total == 0:
            continue
        wm = sum(int(ch.visits) for a, ch in root.children.items() if tuple(a) in win_set)
        rm = sum(int(ch.visits) for a, ch in root.children.items() if tuple(a) in raise_set)
        am = sum(int(ch.visits) for a, ch in root.children.items() if tuple(a) in all_set)
        if win_set:
            win_v.append(wm / total)
        if raise_set:
            raise_v.append(rm / total)
        all_v.append(am / total)
    return win_v, raise_v, all_v


def boot_ci(vals, nboot, rng):
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    arr = np.asarray(vals, dtype=np.float64)
    bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(nboot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


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


def main():
    name = normalize_encoding_name("v6_live2")
    spec = lookup(name)
    kept = list(spec.kept_plane_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(str(p) for p in Path(
        "investigation/coherence_2026-06-08/replays").glob("games_2026-06-0*.jsonl"))
    rows = build_pool(files, name, spec, {30000, 53000, 87500}, 240, 20260608)
    by_src = Counter(r["src_step"] for r in rows)
    n_raise_rows = sum(1 for r in rows if r["raise_b"])
    n_win_rows = sum(1 for r in rows if r["win_b"])
    print(f"[pool] n={len(rows)} by_src={dict(sorted(by_src.items()))} "
          f"rows_with_raise_builder={n_raise_rows} rows_with_win_builder={n_win_rows}",
          flush=True)
    # null fork-mass for RAISE-only builders (true fork-build): uniform over legal
    raise_null = float(np.mean([len(r["raise_b"]) / max(r["n_legal"], 1)
                                for r in rows if r["raise_b"]]))
    win_null = float(np.mean([len(r["win_b"]) / max(r["n_legal"], 1)
                              for r in rows if r["win_b"]]))
    print(f"[null] raise-only uniform fork-mass={raise_null:.4f}  "
          f"win-only uniform fork-mass={win_null:.4f}", flush=True)

    ckpts = discover_ckpts(["investigation/coherence_2026-06-08/checkpoints",
                            "investigation/fragility_2026-06-07/checkpoints"])
    rng = np.random.default_rng(20260608)

    print("\nPOLICY-PRIOR mass split (all 11 ckpts):")
    print(f"{'step':>7} | {'WIN-only':>22} | {'RAISE-only(true fork)':>26} | {'ALL':>10}")
    pol = []
    for st in sorted(ckpts):
        model, _s, _l = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        wv, rv, av = policy_mass_split(model, device, kept, name, rows)
        wm, rm, am = (float(np.mean(x)) if x else float("nan") for x in (wv, rv, av))
        rci = boot_ci(rv, 1000, rng)
        pol.append({"step": st, "win": wm, "raise": rm, "all": am, "raise_ci": rci,
                    "n_raise": len(rv)})
        print(f"{st:>7} | {wm:>22.4f} | {rm:>11.4f} [{rci[0]:.3f},{rci[1]:.3f}] | {am:>10.4f}",
              flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nREGEN-MCTS mass split (sims=100, ckpts 30k/50k/75k/87.5k):")
    print(f"{'step':>7} | {'WIN-only':>22} | {'RAISE-only(true fork)':>26} | {'ALL':>10}")
    mc = []
    for st in [30000, 50000, 75000, 87500]:
        if st not in ckpts:
            continue
        model, _s, _l = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        bot = KClusterMCTSBot(model, device, n_sims=100, temperature=0.0,
                              kept_plane_indices=kept)
        t0 = time.time()
        wv, rv, av = mcts_mass_split(bot, name, rows)
        wm, rm, am = (float(np.mean(x)) if x else float("nan") for x in (wv, rv, av))
        rci = boot_ci(rv, 1000, rng)
        mc.append({"step": st, "win": wm, "raise": rm, "all": am, "raise_ci": rci,
                   "n_raise": len(rv)})
        print(f"{st:>7} | {wm:>22.4f} | {rm:>11.4f} [{rci[0]:.3f},{rci[1]:.3f}] | {am:>10.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        del model, bot
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(pol) >= 2:
        print(f"\nARC raise-only(true fork) policy {pol[0]['step']}->{pol[-1]['step']}: "
              f"{pol[0]['raise']:.4f} -> {pol[-1]['raise']:.4f} "
              f"(d={pol[-1]['raise']-pol[0]['raise']:+.4f})  [null {raise_null:.4f}]")
    if len(mc) >= 2:
        print(f"ARC raise-only(true fork) MCTS  {mc[0]['step']}->{mc[-1]['step']}: "
              f"{mc[0]['raise']:.4f} -> {mc[-1]['raise']:.4f} "
              f"(d={mc[-1]['raise']-mc[0]['raise']:+.4f})  [null {raise_null:.4f}]")

    out = {"n_pool": len(rows), "by_src": dict(sorted(by_src.items())),
           "raise_null": raise_null, "win_null": win_null,
           "n_raise_rows": n_raise_rows, "n_win_rows": n_win_rows,
           "policy": pol, "mcts": mc}
    Path("investigation/overspread_2026-06-08/d3_redteam_split.json").write_text(
        json.dumps(out, indent=2))
    print("\n[out] investigation/overspread_2026-06-08/d3_redteam_split.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
