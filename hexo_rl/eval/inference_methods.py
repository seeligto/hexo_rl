"""Inference-method dispatcher for the encoding-aware SealBot harness.

Encoding × inference are independent axes:
- Encoding (v6 / v6w25 / v8) is detected from the checkpoint by
  `hexo_rl.eval.checkpoint_loader.load_model_with_encoding`.
- Inference method (argmax / mcts-N / fast) is operator-selected via the
  `--inference` CLI flag on `scripts/run_sealbot_eval.py`.

`build_inference_method(name, model, device, encoding_label)` returns a
`BotProtocol` ready for play. Adding a new method requires editing this
single dispatcher; harness + tests stay untouched.
"""
from __future__ import annotations

import re
from typing import Tuple

import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot
from hexo_rl.eval.v8_argmax_bot import V8ArgmaxBot
from hexo_rl.eval.v8_mcts_bot import V8MCTSBot
from hexo_rl.model.network import HexTacToeNet


# Aliases that map to a concrete name. Single source of truth for shorthand.
_ALIASES = {
    "fast": "mcts-50",
    "fast-mode": "mcts-50",
    "mcts": "mcts-128",
}


def _parse_method(name: str) -> Tuple[str, int]:
    """Return (kind, n_sims). kind ∈ {'argmax', 'mcts'}.

    'argmax' → ('argmax', 0).
    'mcts-N' → ('mcts', N) for N >= 1.
    Aliases are resolved before parsing.
    """
    name = _ALIASES.get(name, name)
    if name == "argmax":
        return "argmax", 0
    m = re.fullmatch(r"mcts-(\d+)", name)
    if m:
        n = int(m.group(1))
        if n < 1:
            raise ValueError(f"mcts sims must be >= 1; got {n}")
        return "mcts", n
    raise ValueError(
        f"unknown inference method {name!r}; expected 'argmax', "
        "'mcts-N', or 'fast'"
    )


def list_methods() -> list[str]:
    """Return canonical method names. Used by --help text."""
    return ["argmax", "mcts-N (e.g. mcts-128)", "fast (= mcts-50)"]


def build_inference_method(
    name: str,
    model: HexTacToeNet,
    device: torch.device,
    encoding_label: str,
    *,
    temperature: float = 0.0,
    c_puct: float = 1.5,
    kept_plane_indices: list[int] | None = None,
) -> BotProtocol:
    """Build a BotProtocol for the (encoding, method) tuple.

    encoding_label ∈ {'v6', 'v6w25', 'v8'} — distinct from
    `EncodingSpec.version` (which is just a state-dict-compat marker).

    Raises NotImplementedError for combinations that haven't been wired
    yet (currently: v6w25 with any method, since v6w25 inference paths
    are gated on §168 Gate 3 Rust unification).
    """
    kind, n_sims = _parse_method(name)

    if encoding_label == "v8":
        if kind == "argmax":
            return V8ArgmaxBot(model, device, temperature=temperature)
        return V8MCTSBot(
            model, device, n_sims=n_sims, c_puct=c_puct, temperature=temperature
        )

    # v6tp (§P5-CT CF-2) and v6_live2 (§P5-CT H-PLANE fix) share v6 runtime
    # inference; the only difference is the wire-plane slice (v6tp 10 incl.
    # turn-phase 16/17, v6_live2 4 = [0,8,16,17] vs v6's 8), threaded via
    # kept_plane_indices into the shape-aware bots.
    if encoding_label in ("v6", "v6tp", "v6_live2", "v6_live2_ls"):
        if kind == "argmax":
            if encoding_label == "v6_live2_ls":
                # The legal-set (multi-window) action space has no single-window
                # argmax bot — V6ArgmaxBot would drop off-window legal moves, the
                # same mis-route defender_dispatch fixes. Use mcts-N (no-drop
                # KClusterMCTSBot) for v6_live2_ls strength reads.
                raise NotImplementedError(
                    "v6_live2_ls argmax bot is not wired (single-window argmax "
                    "drops off-window legal moves); use --inference mcts-N for the "
                    "no-drop KClusterMCTSBot path."
                )
            return V6ArgmaxBot(model, device, temperature=temperature)
        # v6 MCTS: Python K-cluster MCTS (KClusterMCTSBot) since §169 P1.
        # Rust MCTSTree is also available via evaluator.ModelPlayer but
        # this dispatcher uses the Python path uniformly across v6 / v6w25
        # for matched-MCTS comparison apples-to-apples.
        return KClusterMCTSBot(
            model, device, n_sims=n_sims, c_puct=c_puct, temperature=temperature,
            kept_plane_indices=kept_plane_indices,
        )

    if encoding_label == "v6w25":
        # v6w25 — same K-cluster encoding as v6 at runtime, 25×25 cluster
        # window. The bots are shape-aware (read view dims from the tensor),
        # and the script seeds the Board with set_cluster_window_size(25) +
        # cluster_threshold(8) + legal_move_radius(8) before each game.
        if kind == "argmax":
            return V6ArgmaxBot(model, device, temperature=temperature)
        # v6w25 native MCTS via Rust MCTSTree deferred to α (§172 Phase A7
        # design). §172 registry: lookup("v6w25").is_multi_window=True;
        # KClusterMCTSBot is the Python port that handles K-cluster fan-out
        # at MCTS expansion time and is the canonical matched-MCTS path
        # until α lands the multi-window Rust unification.
        return KClusterMCTSBot(
            model, device, n_sims=n_sims, c_puct=c_puct, temperature=temperature
        )

    raise ValueError(f"unknown encoding label {encoding_label!r}")
