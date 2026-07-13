#!/usr/bin/env python3
"""Sandboxed shrimp fidelity REFERENCE — runs INSIDE the hexo-bot venv only.

Emits raw forward_policy_value (value scalar + legal-prefix policy logits) for a
list of positions, built DIRECTLY from shrimp's own code with NO adapter bridge
in the loop. The fidelity harness (tests/test_shrimp_bot.py) compares this
against the adapter's raw_eval() (which drives shrimp_worker.py through the
BotProtocol bridge). Both invoke identical shrimp code; a mismatch localizes to
the adapter's state-reconstruction / wire seam.

NETWORK-OFF: reads a local checkpoint (weights_only=True), no sockets. Same
sanctioned "run shrimp verbatim in a sandbox" path as scripts/arena.

Positions arrive on stdin as one JSON per line:
  {"id":.., "stones":[[q,r,player],...], "current_player":1|-1, "moves_remaining":1|2}
player is OUR convention (1 opener / -1 responder). One reply JSON per line:
  {"id":.., "value":float, "policy":[[q,r,logit],...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SHRIMP_SUPPORT_RADIUS", "4")
os.environ.setdefault("SHRIMP_CHANNELS", "192")
os.environ.setdefault("SHRIMP_ATTENTION_HEADS", "3")
os.environ.setdefault("SHRIMP_TRUNK", "CCACCACCACCACCA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction
    from shrimp import _rust
    from shrimp.losses import decode_binned_value
    from shrimp.constants import NUM_FEATURES
    from shrimp.model import ShrimpNet, infer_net_kwargs_from_state_dict

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    net = ShrimpNet(**infer_net_kwargs_from_state_dict(sd))
    net.load_state_dict(sd, strict=True)
    net.eval()

    probe = engine.new_game()
    opener = engine.current_player(probe)
    engine.apply_action(probe, PlacementAction(AxialCoord(0, 0)))
    other = engine.current_player(probe)
    player_for = {1: opener, -1: other}

    def build(stones, current_player, origin):
        oq, orr = int(origin[0]), int(origin[1])
        shift = lambda q, r: (int(q) - oq, int(r) - orr)  # noqa: E731
        opener_all = [shift(q, r) for q, r, p in stones if p == 1]
        other_st = [shift(q, r) for q, r, p in stones if p == -1]
        assert (0, 0) in opener_all, "origin is not an opener stone after shift"
        opener_st = [(0, 0)] + [c for c in opener_all if c != (0, 0)]
        st = engine.new_game()
        oi = ii = 0

        def place(q, r):
            engine.apply_action(st, PlacementAction(AxialCoord(int(q), int(r))))

        if opener_st:
            place(*opener_st[oi]); oi += 1
        turn = -1
        while oi < len(opener_st) or ii < len(other_st):
            if turn == -1:
                for _ in range(2):
                    if ii < len(other_st):
                        place(*other_st[ii]); ii += 1
                turn = 1
            else:
                for _ in range(2):
                    if oi < len(opener_st):
                        place(*opener_st[oi]); oi += 1
                turn = -1
        assert engine.current_player(st) == player_for[current_player], "ref desync"
        return st

    def raw_forward(st):
        d = _rust.featurize_states([st])[0]
        N = d["num_nodes"]
        coords = np.frombuffer(d["coords"], dtype=np.int16).reshape(N, 2).astype(np.int64)
        nbr = np.frombuffer(d["nbr"], dtype=np.int32).reshape(N, 6).copy()
        feats = np.frombuffer(d["feats"], dtype=np.float32).reshape(N, NUM_FEATURES).copy()
        nbr = np.where(nbr < 0, N, nbr)
        ft = torch.from_numpy(feats).unsqueeze(0)
        nt = torch.from_numpy(nbr.astype(np.int64)).unsqueeze(0)
        mt = torch.ones(1, N, dtype=torch.bool)
        ct = torch.from_numpy(coords).unsqueeze(0)
        with torch.no_grad():
            out = net.forward_policy_value(ft, nt, mt, ct)
        value = float(decode_binned_value(out["value"].float()).item())
        legal = int(d["legal_count"])
        pol = out["policy"][0][:legal].float().tolist()
        policy = [
            [int(coords[i][0]), int(coords[i][1]), lg] for i, lg in enumerate(pol)
        ]
        return value, policy

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        origin = req["origin"]
        oq, orr = int(origin[0]), int(origin[1])
        st = build(req["stones"], int(req["current_player"]), origin)
        value, policy = raw_forward(st)
        # translate policy cells back to our frame
        policy = [[q + oq, r + orr, lg] for (q, r, lg) in policy]
        sys.stdout.write(json.dumps({"id": req["id"], "value": value, "policy": policy}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
