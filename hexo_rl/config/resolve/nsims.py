"""CONFRES P5 — one default authority for eval opponent n_sims (``model_sims``).

Pre-CONFRES the eval ``model_sims`` default was resolved at two seams with DIFFERENT values:
``EvalPipeline`` injected 96/128 (``defaults.DEFAULT_RANDOM/SEALBOT_MODEL_SIMS``) while a
directly-constructed ``Evaluator`` fell back to 100/200
(``defaults.DEFAULT_EVALUATOR_*_MODEL_SIMS``). Production always injects, so 100/200 only ever
fired on direct-init paths — and every such path (tests) sets ``model_sims`` explicitly, so
NOTHING depended on 100/200. The "intentional split" was drift; this module is the single
default authority both seams now read (design §6 P5).

Imports stdlib only — safe to import from ``hexo_rl.eval`` without a package-init cycle (§8 N3).
"""
from __future__ import annotations

# The ONE eval model_sims default per opponent (collapsed onto the production 96/128).
EVAL_MODEL_SIMS_DEFAULT: dict[str, int] = {"random": 96, "sealbot": 128}


def resolve_eval_model_sims(opponent: str, cfg_value: int | None) -> int:
    """Resolve eval ``model_sims`` for ``opponent`` ("random"|"sealbot").

    A config-supplied value (from ``evaluation.<opponent>_model_sims`` or the opponent block's
    ``model_sims``) always wins; otherwise the single per-opponent default. Unknown opponent →
    ``ValueError`` (no silent fallback — the caller passed a name with no default authority).
    """
    if cfg_value is not None:
        return int(cfg_value)
    try:
        return EVAL_MODEL_SIMS_DEFAULT[opponent]
    except KeyError:
        raise ValueError(
            f"no eval model_sims default for opponent {opponent!r}; "
            f"known: {sorted(EVAL_MODEL_SIMS_DEFAULT)}"
        ) from None
