"""CONFRES batch 7 — the single player-construction authority (design §4 build_player, B1, N5).

``build_player`` is the ONE entry every eval/deploy play seam routes through. It reads the resolved
``PlannerSpec`` (dispatch target + physics) and the resolved encoding label, then constructs the
correct DISTINCT player — ModelPlayer (bare PUCT) / KClusterMCTSBot (K-batched no-drop) /
DeployHeadBot (Gumbel g=0). A mis-dispatch (eval building deploy's player, a ``policy_pool`` edit
swapping the eval bot) is impossible BY CONSTRUCTION: the factory selects on ``spec.bot_impl``, and
the pin test asserts the resolved dispatch per (context, subcontext).

Two hard constraints (design §4):
  (i)  UNIFY the construction ENTRY, NOT the physics — the three algorithms stay distinct; the
       factory SELECTS among them from the resolved spec (flat vs legal-set vs deploy differ
       legitimately). No physics merger (§11).
  (ii) architecture comes from the resolved encoding authority (registry TOML + stamp), never from
       shape-inference as the happy path (F1/canary silent-mislabel class). The ``encoding_label``
       IS the resolved authority; ``detect_encoding_from_state_dict`` stays a loud last-resort
       gated by ``require_encoding_source`` (in ``load_model_with_encoding``), never the factory's
       default path.

N5 (label channel): the encoding label is passed EXPLICITLY to KClusterMCTSBot, never via a
``model.encoding`` attribute mutation (last-writer-wins when a model is shared across two arms with
different labels).

Batched carve-out (B1a): ``build_player`` returns a stateful Player object; the coroutine-based
``eval_batcher.batched_evaluate`` cannot receive one. It calls :func:`assert_batched_dispatch_ok`
to HARD-REFUSE ``batched:true`` on a legal-set encoding (whose off-window legal moves the
single-window batched physics would silently drop) rather than mis-route.

Design: docs/designs/confres_design.md §4, §6 P1, §13 (B1/B1a/N5).
"""
from __future__ import annotations

from typing import Any

import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.config.resolve.planner import (
    BOT_DEPLOY_GUMBEL,
    BOT_KCLUSTER,
    BOT_MODEL_PLAYER,
    PlannerSpec,
)


class BatchedLegalSetRefusedError(ValueError):
    """``batched:true`` requested on a legal-set encoding (B1a).

    The coroutine batched path assembles single-window physics inline and DROPS off-window legal
    moves (``eval_batcher.py`` ``_pick_move``); running it on a legal-set encoding mis-routes the
    net exactly like the ModelPlayer drop bug. Refuse loudly — the full batched-KCluster path is a
    follow-up (§11); the CONFRES fix is this guard.
    """


def assert_batched_dispatch_ok(encoding_label: str) -> None:
    """HARD-REFUSE ``batched:true`` on a legal-set encoding (B1a batched carve-out)."""
    from hexo_rl.config.resolve.encoding import window_set
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot

    if needs_no_drop_bot(window_set(encoding_label)):
        raise BatchedLegalSetRefusedError(
            f"batched eval refused for legal-set encoding {encoding_label!r}: the batched "
            "coroutine runs single-window physics that DROPS off-window legal moves "
            "(eval_batcher._pick_move), mis-routing the net just like the ModelPlayer drop bug. "
            "Use the non-batched KClusterMCTSBot path (build_player), or implement the "
            "batched-KCluster branch (design §11 follow-up)."
        )


def build_player(
    spec: PlannerSpec,
    *,
    encoding_label: str,
    model: Any = None,
    engine: Any = None,
    device: torch.device | None = None,
    knobs: dict | None = None,
    config: dict | None = None,
    seed: int = 0,
    veto: bool = False,
) -> BotProtocol:
    """Construct the player for a resolved ``PlannerSpec`` — the ONE play-construction entry.

    Dispatch on ``spec.bot_impl`` (the DISTINCT algorithms):

    - ``ModelPlayer`` / ``KClusterMCTSBot`` (eval contexts): need ``model`` + ``device``. Routes
      through ``defender_dispatch.build_model_bot`` (the existing in-loop dispatch authority), which
      selects ModelPlayer vs KClusterMCTSBot from the encoding — CONSISTENT with the resolved
      ``bot_impl`` (a divergence raises, so the resolver and the dispatch can't drift). The label is
      passed EXPLICITLY (N5).
    - ``DeployHeadBot`` (deploy context): needs ``engine`` + ``knobs``. Gumbel g=0 head with the
      resolved ``legal_set`` decode flag.

    Architecture is NEVER inferred here — ``encoding_label`` is the resolved authority (constraint
    ii). Passing a shape-inference fallback label is the caller's loud last-resort, not this path.
    """
    if spec.bot_impl in (BOT_MODEL_PLAYER, BOT_KCLUSTER):
        if model is None or device is None:
            raise ValueError(
                f"build_player({spec.bot_impl}) requires model + device (eval context)"
            )
        from hexo_rl.config.resolve.encoding import window_set
        from hexo_rl.eval.defender_dispatch import build_model_bot, defender_kind

        enc_spec = window_set(encoding_label)
        # Consistency guard: the resolved bot_impl MUST match the dispatch the shared rule computes
        # from the encoding — a mismatch means the resolver and defender_dispatch drifted.
        dispatch_kind = defender_kind(enc_spec, "auto")
        expected = BOT_KCLUSTER if dispatch_kind == "kcluster" else BOT_MODEL_PLAYER
        if spec.bot_impl != expected:
            raise ValueError(
                f"build_player dispatch mismatch: PlannerSpec.bot_impl={spec.bot_impl!r} but "
                f"encoding {encoding_label!r} (policy_pool) dispatches {expected!r}. The resolver "
                "and defender_dispatch disagree — one drifted."
            )
        return build_model_bot(
            model, enc_spec, device,
            n_sims=spec.n_sims,
            encoding_label=encoding_label,
            temperature=spec.temperature,
            c_puct=spec.c_puct,
            config=config,
            mode="auto",
        )

    if spec.bot_impl == BOT_DEPLOY_GUMBEL:
        if engine is None or knobs is None:
            raise ValueError(
                "build_player(DeployHeadBot) requires engine + knobs (deploy context)"
            )
        from hexo_rl.eval.deploy_strength_eval import DeployHeadBot

        return DeployHeadBot(
            engine, knobs, label=spec.subcontext, seed=seed,
            legal_set=spec.legal_set, veto=veto,
        )

    raise ValueError(f"build_player: unknown bot_impl {spec.bot_impl!r}")
