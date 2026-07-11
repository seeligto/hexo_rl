"""Centralized eval-pipeline defaults (SSR16 close-out, §176 P37).

Inline ``cfg.get("foo", N)`` fallbacks for the same conceptual default were
scattered across ``eval_pipeline.py`` and ``evaluator.py``.  Drift between
the two sites silently corrupts comparisons (an evaluator built without a
pipeline merge would see a different default than one built through the
pipeline).  This module is the single source of repetition for those
constants.

Canonical override channel: ``configs/eval.yaml`` (and per-variant
``configs/variants/*.yaml`` blocks that target it).  These module
constants are the fallbacks the code uses when the YAML does not pin the
key — they MUST stay byte-for-byte aligned with prior inline literals so
no behavior changes for existing variants.

The eval ``model_sims`` defaults formerly lived here as TWO divergent sets (96/128
pipeline vs 100/200 evaluator-direct); CONFRES P5 collapsed them into one authority at
``hexo_rl.config.resolve.nsims`` (the 100/200 set was dead — no production path or test
depended on it). The remaining constants below are single-valued eval defaults not yet
migrated to a resolver.
"""

from __future__ import annotations

# ── EvalPipeline.run_evaluation setdefault block ──────────────────────
# Defaults the pipeline injects into the ``evaluation`` config section
# before constructing the Evaluator. (model_sims moved to resolve/nsims.py — P5.)
DEFAULT_COLONY_CENTROID_THRESHOLD: float = 6.0
# eval_temperature moved to resolve/temperature.py — P4.
DEFAULT_EVAL_RANDOM_OPENING_PLIES: int = 4
DEFAULT_EVAL_SEED_BASE: int = 42

# ── MCTS exploration constant for ModelPlayer ─────────────────────────
# Mirrored from ``mcts.c_puct`` in the live config when present.
DEFAULT_C_PUCT: float = 1.5
