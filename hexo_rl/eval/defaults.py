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

Two of the canonical purposes have intentionally distinct evaluator-side
fallbacks (``random_model_sims``, ``sealbot_model_sims``): the pipeline
merges its 96/128 values into the evaluation section before the Evaluator
runs, so the Evaluator's 100/200 fallbacks only fire on direct
instantiation paths (notably tests).  Both sets are exposed here so each
prod site imports its own constant rather than inlining the literal.
"""

from __future__ import annotations

# ── EvalPipeline.run_evaluation setdefault block (lines 216–224) ───────
# These six are the SSR16 set: defaults the pipeline injects into the
# ``evaluation`` config section before constructing the Evaluator.
DEFAULT_RANDOM_MODEL_SIMS: int = 96
DEFAULT_SEALBOT_MODEL_SIMS: int = 128
DEFAULT_COLONY_CENTROID_THRESHOLD: float = 6.0
DEFAULT_EVAL_TEMPERATURE: float = 0.5
DEFAULT_EVAL_RANDOM_OPENING_PLIES: int = 4
DEFAULT_EVAL_SEED_BASE: int = 42

# ── Evaluator direct-init fallbacks ────────────────────────────────────
# Used when Evaluator is instantiated without an upstream pipeline merge
# (tests, ad-hoc scripts).  Distinct from the pipeline values by design —
# do NOT collapse onto DEFAULT_*_MODEL_SIMS above.
DEFAULT_EVALUATOR_RANDOM_MODEL_SIMS: int = 100
DEFAULT_EVALUATOR_SEALBOT_MODEL_SIMS: int = 200

# ── MCTS exploration constant for ModelPlayer ─────────────────────────
# Mirrored from ``mcts.c_puct`` in the live config when present.
DEFAULT_C_PUCT: float = 1.5
