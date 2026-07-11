"""CONFRES batch 7 — the planner/dispatch resolution AUTHORITY (design §4 planner, B1, N2, N5).

One resolver from ``(context, subcontext)`` → a ``PlannerSpec`` naming the DISPATCH TARGET
(``bot_impl``) + the search physics each play seam runs. The single ``build_player`` factory
(``hexo_rl.eval.player_factory``) reads a ``PlannerSpec`` and constructs the correct player, so a
mis-dispatch (eval building deploy's player, or a ``policy_pool`` edit swapping the eval bot) is
impossible by construction (design §4 player-factory).

Two hard constraints (design §4):
  (i)  the factory UNIFIES the construction ENTRY, NOT the physics — ModelPlayer (bare PUCT),
       KClusterMCTSBot (K-batched, fpu_q=0), Gumbel-g0 stay DISTINCT algorithms the factory SELECTS
       among from ``bot_impl``. Forced merger is out of scope (§11).
  (ii) ``bot_impl`` for the eval subcontexts is DISPATCHED from the encoding's ``policy_pool``
       (legal-set → KClusterMCTSBot no-drop; flat → ModelPlayer bare PUCT), the ONE dispatch rule
       ``hexo_rl.eval.defender_dispatch`` already owns — re-exported here so the resolver + the
       factory agree with every offline instrument.

Import constraint (§8 N3): this module imports ``hexo_rl.eval.defender_dispatch`` LAZILY (inside
the resolver) so ``hexo_rl.config.resolve.planner`` does not pull ``hexo_rl.eval`` at import time
and cycle via ``hexo_rl/eval/__init__.py``.

Design: docs/designs/confres_design.md §4 (planner + build_player), §6 P1, §13 (B1/N2/N5).
"""
from __future__ import annotations

from dataclasses import dataclass


# Canonical bot-impl dispatch targets (the DISTINCT physics the factory selects among).
BOT_MODEL_PLAYER = "ModelPlayer"          # bare single-window PUCT (drops off-window)
BOT_KCLUSTER = "KClusterMCTSBot"          # K-batched no-drop scatter-max (legal-set encodings)
BOT_DEPLOY_GUMBEL = "DeployHeadBot"       # Gumbel g=0 deploy head (deterministic)


@dataclass(frozen=True)
class PlannerSpec:
    """The resolved dispatch + search physics for one (context, subcontext).

    ``bot_impl`` names WHICH player the factory builds; the physics fields carry the knobs that
    player needs. Not every field applies to every ``bot_impl`` (e.g. ``gumbel_scale`` is
    deploy-only, ``legal_set`` is a deploy-decode flag) — the factory reads the fields its target
    consumes. ``root``/``interior`` document the search regime for the emitted provenance.
    """

    context: str            # "selfplay" | "eval" | "deploy"
    subcontext: str         # e.g. "best" | "sealbot" | "bootstrap_anchor" | "default"
    bot_impl: str           # one of the BOT_* constants
    root: str               # "puct" | "gumbel"
    interior: str           # "puct" | ...
    n_sims: int
    c_puct: float = 1.5
    fpu: float = 0.0
    virtual_loss: bool = False
    legal_set: bool = False       # decode flag (deploy multi-window no-drop / KCluster no-drop)
    temperature: float = 0.0
    gumbel_scale: float = 0.0     # deploy g=0 determinism

    def as_event_dict(self) -> dict:
        """Render for the ``resolved_config`` planner.<context>.<subcontext> provenance rows."""
        return {
            "bot_impl": self.bot_impl,
            "root": self.root,
            "interior": self.interior,
            "n_sims": self.n_sims,
            "c_puct": self.c_puct,
            "fpu": self.fpu,
            "virtual_loss": self.virtual_loss,
            "legal_set": self.legal_set,
            "temperature": self.temperature,
            "gumbel_scale": self.gumbel_scale,
        }


def _dispatch_eval_bot_impl(encoding_name: str, mode: str = "auto") -> str:
    """Dispatch the eval bot_impl from the encoding's policy_pool (the ONE dispatch rule).

    Delegates to ``defender_dispatch.defender_kind`` (lazy import, N3) so the resolver, the factory,
    and every offline instrument route identically. ``mode`` ∈ {"auto","kcluster","modelplayer"}
    forwards the drop-vs-no-drop counterfactual force.
    """
    from hexo_rl.config.resolve.encoding import window_set
    from hexo_rl.eval.defender_dispatch import defender_kind

    spec = window_set(encoding_name)
    kind = defender_kind(spec, mode)  # "kcluster" | "modelplayer"
    return BOT_KCLUSTER if kind == "kcluster" else BOT_MODEL_PLAYER


def resolve_eval_planner(
    encoding_name: str,
    subcontext: str,
    *,
    n_sims: int,
    c_puct: float = 1.5,
    temperature: float = 0.0,
    mode: str = "auto",
) -> PlannerSpec:
    """Resolve the in-loop / offline EVAL planner for a (subcontext) on ``encoding_name``.

    bot_impl DISPATCHES from policy_pool (legal-set → KClusterMCTSBot; flat → ModelPlayer). Bare
    PUCT root, fpu_q=0, no virtual loss (eval measures; it does not train). The pin test asserts
    THIS bot_impl per subcontext, so a policy_pool edit or a ``mode=`` force FIRES the test.
    """
    bot_impl = _dispatch_eval_bot_impl(encoding_name, mode)
    return PlannerSpec(
        context="eval",
        subcontext=subcontext,
        bot_impl=bot_impl,
        root="puct",
        interior="puct",
        n_sims=int(n_sims),
        c_puct=float(c_puct),
        fpu=0.0,
        virtual_loss=False,
        legal_set=(bot_impl == BOT_KCLUSTER),
        temperature=float(temperature),
        gumbel_scale=0.0,
    )


def resolve_deploy_planner(
    encoding_name: str,
    *,
    n_sims: int,
    c_puct: float = 1.5,
) -> PlannerSpec:
    """Resolve the DEPLOY planner: Gumbel g=0 head + legal_set decode flag from policy_pool.

    The deploy head is a DISTINCT algorithm (Gumbel Sequential-Halving, deterministic at
    gumbel_scale=0.0) — NOT ModelPlayer/KClusterMCTSBot. ``legal_set`` (multi-window no-drop
    decode) is dispatched from the encoding's policy_pool (design §4 deploy row).
    """
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.config.resolve.encoding import window_set

    legal_set = needs_no_drop_bot(window_set(encoding_name))
    return PlannerSpec(
        context="deploy",
        subcontext="default",
        bot_impl=BOT_DEPLOY_GUMBEL,
        root="gumbel",
        interior="puct",
        n_sims=int(n_sims),
        c_puct=float(c_puct),
        fpu=0.25,             # selfplay/deploy Gumbel fpu_reduction
        virtual_loss=True,
        legal_set=legal_set,
        temperature=0.0,
        gumbel_scale=0.0,
    )
