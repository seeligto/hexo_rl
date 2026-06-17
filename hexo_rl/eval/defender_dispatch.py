"""Single source of truth for the legal-set vs single-window eval-bot dispatch.

Single-window ``ModelPlayer`` DROPS off-window legal moves (evaluator.py:111-113):
it zeroes the probability of any legal move whose flat index exceeds the single
window's ``policy_logit_count``. On a legal-set / multi-window encoding
(``policy_pool == "legal_set_scatter_max"``, e.g. ``v6_live2_ls``) that funnels the
model back to a single window and MIS-ROUTES it — the false-clear that fix
``a71ae28`` found in ``scripts/exploit_probe.py`` (dmulticluster_362_legalset_design
§7 P4). Those encodings must play through ``KClusterMCTSBot`` (no-drop scatter-max),
which actually exercises the off-window logits the model was trained on.

This module is the ONE place that decision lives, so every eval instrument
(``exploit_probe``, ``round_robin``, the in-run ``Evaluator``) routes identically —
per the registry-by-name / no-scattered-constants discipline in CLAUDE.md.

Dispatch key = ``spec.policy_pool``, NOT the encoding name or ``is_multi_window``:
the drop bug is exactly "legal moves exist off the single window", which is what
``legal_set_scatter_max`` marks. So v6 / v6_live2 / v6w25 / v8 (``policy_pool ==
"none"``) keep ``ModelPlayer`` and are **bitwise-unchanged**; only legal-set
encodings switch to the no-drop bot.

NB (the load-bearing caveat for standalone loaders): ``v6_live2_ls`` is
state-dict-identical to ``v6_live2``, so checkpoint auto-detection returns the arch
family (``v6_live2``, ``policy_pool="none"``) and auto-dispatch would WRONGLY pick
ModelPlayer. Standalone tools that load a checkpoint (exploit_probe, round_robin)
must pass an explicit ``--encoding v6_live2_ls`` override so the spec resolves to the
trained action space. The in-run ``Evaluator`` needs no override — its config
already carries ``encoding: v6_live2_ls``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol

LEGAL_SET_POLICY_POOL = "legal_set_scatter_max"


def policy_pool_of(spec_or_str: Any) -> str:
    """Extract the policy_pool from an EncodingSpec or accept a bare string.

    Accepting a string keeps the dispatch decision pure-logic testable without
    constructing a spec (mirrors the ``a71ae28`` ``_defender_kind`` contract)."""
    if isinstance(spec_or_str, str):
        return spec_or_str or "none"
    return getattr(spec_or_str, "policy_pool", "none") or "none"


def needs_no_drop_bot(spec_or_str: Any) -> bool:
    """True iff the encoding's legal action set extends off the single window, so
    single-window ModelPlayer would drop off-window legal moves and mis-route it."""
    return policy_pool_of(spec_or_str) == LEGAL_SET_POLICY_POOL


def defender_kind(spec_or_policy_pool: Any, mode: str = "auto") -> str:
    """Return the bot kind: ``'kcluster'`` | ``'modelplayer'``.

    ``mode`` ∈ {"auto", "kcluster", "modelplayer"}. A forced mode overrides the
    policy-pool dispatch — used for the drop-vs-no-drop causal-uncapping
    counterfactual (force ModelPlayer on a legal-set encoding to reproduce the
    drop baseline, or KClusterMCTSBot on single-window). ``"auto"`` dispatches by
    ``spec.policy_pool``."""
    if mode in ("kcluster", "modelplayer"):
        return mode
    return "kcluster" if needs_no_drop_bot(spec_or_policy_pool) else "modelplayer"


def build_model_bot(
    model: Any,
    spec: Any,
    device: torch.device,
    *,
    n_sims: int,
    encoding_label: str,
    temperature: float = 0.0,
    c_puct: float = 1.5,
    config: Optional[dict] = None,
    mode: str = "auto",
) -> BotProtocol:
    """Build the encoding-correct bot for a model that is the thing UNDER TEST.

    Legal-set encodings → ``KClusterMCTSBot`` (no-drop); everything else →
    ``ModelPlayer`` (the call is byte-identical to the prior inline construction,
    so single-window runs are unchanged).

    ``encoding_label`` is the resolved encoding name (after any override). It is
    stamped onto ``model.encoding`` for the KClusterMCTSBot branch (the bot reads
    ``model.encoding`` to look up its cluster geometry + validate the pass slot),
    and used to build the ModelPlayer config when ``config`` is not supplied.
    ``config`` (the full eval config) is passed straight through to ModelPlayer so
    its other knobs (c_puct, etc.) survive."""
    if defender_kind(spec, mode) == "kcluster":
        from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot

        # KClusterMCTSBot reads model.encoding for the cluster views + pass-slot
        # check; a state-dict-identical legal-set checkpoint detects as the arch
        # family, so stamp the resolved (overridden) label.
        try:
            model.encoding = encoding_label
        except Exception:
            pass
        return KClusterMCTSBot(
            model, device, n_sims=n_sims, c_puct=c_puct, temperature=temperature,
            kept_plane_indices=list(spec.kept_plane_indices),
        )

    from hexo_rl.eval.evaluator import ModelPlayer

    cfg = config if config is not None else {
        "encoding": encoding_label, "mcts": {"c_puct": c_puct},
    }
    return ModelPlayer(model, cfg, device, n_sims=n_sims, temperature=temperature)
