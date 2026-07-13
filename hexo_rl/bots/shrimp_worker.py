#!/usr/bin/env python3
"""shrimp subprocess worker — runs INSIDE the hexo-bot venv only.

WHY A SUBPROCESS: shrimp's move oracle is a RUST Gumbel-MCTS
(`shrimp._rust.ShrimpMctsSession`) driven by a RUST support-set featurizer
(`shrimp._rust.featurize_states`). Neither is portable to our venv — our
interpreter has no `shrimp._rust`/`hexo_engine` extensions, and reimplementing
either in pure Python is infeasible + forbidden by the "no home-grown minimax"
rule. So the honest adapter DELEGATES search to shrimp's own code, run in a
network-off subprocess in shrimp's built venv, and never imports shrimp into
OUR working env. This module is that subprocess. It is invoked ONLY by
`hexo_rl/bots/shrimp_bot.py` via subprocess.Popen with the hexo-bot venv python;
it must never be imported into the hexo_rl interpreter.

SECURITY: no network is opened here. The checkpoint is loaded with
`weights_only=True`. shrimp's Rust/Python is executed (this is the sanctioned
"run shrimp verbatim inside a sandbox" path — same as scripts/arena's
shrimp_child.py prior art), but only inside this isolated child process.

PROTOCOL (one JSON object per line, request on stdin -> reply on stdout):
  {"op":"ready"}                                   -> {"ok":true,...}
  {"op":"reset"}                                   -> {"ok":true}
  {"op":"move","stones":[[q,r,player],...],        -> {"q":..,"r":..,
        "current_player":1|-1,"moves_remaining":1|2,      "root_value":..,
        "ply":N}                                          "value":..,"policy":[...]}
  {"op":"eval","stones":[...],"current_player":..}  -> {"value":..,"policy":[[q,r,logit],...]}
        (raw forward_policy_value forward — no search; for the fidelity harness)
  {"op":"quit"}                                    -> (exit)

`player` in the wire stones list uses OUR convention: 1 = opener (shrimp
player0), -1 = responder (shrimp player1). Coordinates are raw axial (q, r),
identical frame to shrimp's AxialCoord and our engine (no transform).

STDOUT carries ONLY protocol JSON. All logging/diagnostics go to stderr. The
load-bearing SHRIMP_* env (support radius / channels / heads / trunk) is set
here before importing shrimp, exactly as scripts/arena/bots/shrimp_child.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

# Arch/featurization env — read once at shrimp import; SUPPORT_RADIUS is NOT in
# the checkpoint and default 8 silently degrades play. channels/heads/trunk are
# auto-detected from the state dict but are set here for parity with the arena
# child. Values from scripts/arena/bots/shrimp_child.py + the as-trained profile.
os.environ.setdefault("SHRIMP_SUPPORT_RADIUS", "4")
os.environ.setdefault("SHRIMP_CHANNELS", "192")
os.environ.setdefault("SHRIMP_ATTENTION_HEADS", "3")
os.environ.setdefault("SHRIMP_TRUNK", "CCACCACCACCACCA")


def _log(msg: str) -> None:
    print(f"[shrimp_worker] {msg}", file=sys.stderr, flush=True)


class _Worker:
    def __init__(self, checkpoint: str, visits: int, seed: int, profile: str) -> None:
        import numpy as np
        import torch
        import hexo_engine as engine
        from hexo_engine.types import AxialCoord, PlacementAction
        from shrimp import _rust
        from shrimp.config import build_divergence_overrides, parse_shrimp_config
        from shrimp.geometry import unpack_action_id
        from shrimp.inference import ShrimpEvaluator
        from shrimp.losses import decode_binned_value
        from shrimp.constants import NUM_FEATURES
        from shrimp.model import ShrimpNet, infer_net_kwargs_from_state_dict

        self._np = np
        self._torch = torch
        self._engine = engine
        self._AxialCoord = AxialCoord
        self._PlacementAction = PlacementAction
        self._rust = _rust
        self._unpack_action_id = unpack_action_id
        self._decode_binned_value = decode_binned_value
        self._NUM_FEATURES = NUM_FEATURES

        # weights_only=True load (security): the checkpoint is a {meta, model}
        # dict of plain tensors. The arena child uses weights_only=False; we do
        # not need to — the state dict loads fine under the safe unpickler.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = payload["model"]
        net = ShrimpNet(**infer_net_kwargs_from_state_dict(sd))
        net.load_state_dict(sd, strict=True)
        net.eval()
        self._net = net
        self._evaluator = ShrimpEvaluator(net, device="cpu")
        self._session = _rust.ShrimpMctsSession(max_states=65_536)

        # As-trained search invocation, parsed value-for-value from the profile
        # TOML (mirrors scripts/arena/bots/shrimp_child.py::_SearchProfile).
        raw = tomllib.load(open(profile, "rb"))
        mc = raw.get("model", {}).get("config", {})
        cfg = parse_shrimp_config(
            {
                "device": "cpu",
                "selfplay": mc.get("selfplay", {}),
                "multi_stage_eval": mc.get("multi_stage_eval", {}),
            }
        )
        self._sp = cfg.selfplay
        self._overrides = build_divergence_overrides(cfg.selfplay)
        self._virtual_batch_size = int(cfg.multi_stage_eval.eval_virtual_batch_size or 32)
        self._opening_plies = int(cfg.multi_stage_eval.opening_plies)
        self._opening_temperature = float(cfg.multi_stage_eval.opening_temperature)

        self._visits = int(visits)
        self._seed = int(seed)
        self._game_key = 0

        # side<->player map derived from the engine (opener is current on new_game),
        # exactly as the arena child does. our_wire 1 -> opener, -1 -> other.
        probe = engine.new_game()
        self._opener = engine.current_player(probe)
        engine.apply_action(probe, PlacementAction(AxialCoord(0, 0)))
        self._other = engine.current_player(probe)
        self._player_for = {1: self._opener, -1: self._other}
        _log(
            f"ready: ckpt={checkpoint} visits={self._visits} "
            f"support_radius={os.environ['SHRIMP_SUPPORT_RADIUS']} "
            f"opener={self._opener} other={self._other} "
            f"opening_plies={self._opening_plies}"
        )

    # ── state reconstruction ─────────────────────────────────────────────────

    def _build_state(self, stones, current_player, moves_remaining, origin):
        """Reconstruct a shrimp engine state from an UNORDERED stone list.

        TRANSLATION (load-bearing): shrimp's hexo_engine HARD-REQUIRES the
        opening placement at (0, 0); OUR engine allows the opener to open
        anywhere (translation-free infinite board). The support-set featurizer
        is translation-EQUIVARIANT (local relative geometry), so the adapter
        passes `origin` = the opener's OPENING stone in OUR coords, and every
        stone is shifted by -origin so the opening lands at (0, 0). The searched
        move is translated back by +origin in the adapter. This makes the
        adapter correct for arbitrary (non-origin) openings our engine produces.

        HTTT cadence is deterministic (opener 1 stone, then 2-per-turn
        alternating), so a valid placement order that reaches the target
        (current_player, moves_remaining) is reconstructible from stone counts
        alone. The opener's OPENING stone (== `origin`, now at (0,0) after the
        shift) is placed first. The one order-dependent featurizer field is the
        most-recent-stone recency flag (feature col 8); its intra-turn ambiguity
        for the last opponent turn is the documented fidelity ceiling.
        """
        engine = self._engine
        AxialCoord = self._AxialCoord
        PlacementAction = self._PlacementAction

        oq, orr = int(origin[0]), int(origin[1])
        shift = lambda q, r: (int(q) - oq, int(r) - orr)  # noqa: E731

        opener_all = [shift(q, r) for (q, r, p) in stones if p == 1]
        other_stones = [shift(q, r) for (q, r, p) in stones if p == -1]

        # The opening stone (origin -> (0,0)) must be placed FIRST; the remaining
        # opener stones keep their (shifted) positions. Guard: (0,0) must be an
        # opener stone (the caller's origin claim).
        if (0, 0) not in opener_all:
            raise ValueError(
                f"origin {origin} is not an opener stone after shift "
                f"(opener={opener_all[:6]}...)"
            )
        opener_stones = [(0, 0)] + [c for c in opener_all if c != (0, 0)]

        st = engine.new_game()
        oi = ii = 0  # opener index, other index

        def place(q, r):
            engine.apply_action(st, PlacementAction(AxialCoord(int(q), int(r))))

        # Opener's opening single stone (now at (0,0)).
        if opener_stones:
            place(*opener_stones[oi]); oi += 1
        # Then alternating 2-stone turns: other, opener, other, ...
        turn = -1  # -1 = other to move next after the opener
        while oi < len(opener_stones) or ii < len(other_stones):
            if turn == -1:
                for _ in range(2):
                    if ii < len(other_stones):
                        place(*other_stones[ii]); ii += 1
                turn = 1
            else:
                for _ in range(2):
                    if oi < len(opener_stones):
                        place(*opener_stones[oi]); oi += 1
                turn = -1

        # Sanity: reconstructed turn phase must match the caller's declaration.
        cp = engine.current_player(st)
        want_cp = self._player_for[current_player]
        if cp != want_cp:
            raise ValueError(
                f"reconstruction desync: engine at {cp}, expected {want_cp} "
                f"(cur={current_player} mr={moves_remaining} "
                f"n_opener={len(opener_stones)} n_other={len(other_stones)})"
            )
        return st

    # ── raw forward (fidelity reference) ─────────────────────────────────────

    def _raw_forward(self, st):
        """forward_policy_value on a single state assembled from featurize_states.

        Returns (value_scalar, [(q, r, logit), ...] over the legal prefix). This
        bypasses the wire-ABI group packer but calls the identical net forward
        and identical value decode the serve path uses."""
        np = self._np; torch = self._torch
        d = self._rust.featurize_states([st])[0]
        N = d["num_nodes"]
        coords = np.frombuffer(d["coords"], dtype=np.int16).reshape(N, 2).astype(np.int64)
        nbr = np.frombuffer(d["nbr"], dtype=np.int32).reshape(N, 6).copy()
        feats = np.frombuffer(d["feats"], dtype=np.float32).reshape(N, self._NUM_FEATURES).copy()
        nbr = np.where(nbr < 0, N, nbr)  # row-local missing -> appended pad row
        ft = torch.from_numpy(feats).unsqueeze(0)
        nt = torch.from_numpy(nbr.astype(np.int64)).unsqueeze(0)
        mt = torch.ones(1, N, dtype=torch.bool)
        ct = torch.from_numpy(coords).unsqueeze(0)
        with torch.no_grad():
            out = self._net.forward_policy_value(ft, nt, mt, ct)
        value = float(self._decode_binned_value(out["value"].float()).item())
        legal = int(d["legal_count"])
        pol = out["policy"][0][:legal].float().tolist()
        # legal prefix cells are the first `legal` rows in featurize row order.
        cells = [(int(coords[i][0]), int(coords[i][1])) for i in range(legal)]
        policy = [(q, r, lg) for (q, r), lg in zip(cells, pol)]
        return value, policy

    # ── search (move oracle) ─────────────────────────────────────────────────

    def _search_one(self, st, ply, seed):
        temperature = (
            self._opening_temperature
            if (ply < self._opening_plies and self._opening_temperature > 0.0)
            else 0.0
        )
        sp = self._sp
        return self._session.search(
            [int(self._game_key)],
            (st,),
            visits=int(self._visits),
            c_puct=sp.c_puct,
            temperature=0.0,
            seed=int(seed) & 0x7FFFFFFF,
            evaluator=self._evaluator,
            move_temperatures=[float(temperature)],
            divergence_overrides=self._overrides,
            virtual_batch_size=self._virtual_batch_size,
            active_root_limit=sp.active_root_limit,
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            fpu_reduction=sp.fpu_reduction,
            tss_enabled=sp.tss_enabled,
            search_parity_mode=sp.search_parity_mode,
        )[0]

    def handle_move(self, req):
        """Run search for ONE stone (the caller assembles compound turns by
        calling move twice, applying the first stone in between). Coordinates
        are translated so the opener's opening stone is at (0,0) for shrimp; the
        searched move is translated BACK to our frame by +origin."""
        origin = req["origin"]
        oq, orr = int(origin[0]), int(origin[1])
        st = self._build_state(
            req["stones"], int(req["current_player"]), int(req["moves_remaining"]), origin
        )
        ply = int(req.get("ply", 0))
        seed = self._seed * 100003 + ply
        res = self._search_one(st, ply, seed)
        q, r = self._unpack_action_id(int(res["action_id"]))
        # Raw forward diagnostics (value scalar + n policy) alongside the move.
        value, policy = self._raw_forward(st)
        return {
            "q": int(q) + oq,  # translate back to our frame
            "r": int(r) + orr,
            "root_value": float(res["root_value"]),
            "value": value,
            "n_policy": len(policy),
        }

    def handle_eval(self, req):
        """Raw forward_policy_value (no search) — for the fidelity harness.
        Policy cells are translated BACK to our frame by +origin."""
        origin = req["origin"]
        oq, orr = int(origin[0]), int(origin[1])
        st = self._build_state(
            req["stones"], int(req["current_player"]), int(req.get("moves_remaining", 2)), origin
        )
        value, policy = self._raw_forward(st)
        policy = [(q + oq, r + orr, lg) for (q, r, lg) in policy]
        return {"value": value, "policy": policy}


def main() -> None:
    ap = argparse.ArgumentParser(description="shrimp subprocess worker (hexo-bot venv)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--visits", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--profile",
        default="/home/timmy/Work/Hexo/hexo-bot/apps/showcase/profiles/shrimp_main_7.toml",
    )
    args = ap.parse_args()

    worker = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ready":
                if worker is None:
                    worker = _Worker(args.checkpoint, args.visits, args.seed, args.profile)
                reply = {"ok": True, "visits": worker._visits}
            elif op == "reset":
                if worker is not None:
                    worker._game_key += 1
                reply = {"ok": True}
            elif op == "move":
                assert worker is not None, "send ready before move"
                reply = worker.handle_move(req)
            elif op == "eval":
                assert worker is not None, "send ready before eval"
                reply = worker.handle_eval(req)
            elif op == "quit":
                return
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as exc:  # surface faults to the parent via stderr + reply
            import traceback

            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
