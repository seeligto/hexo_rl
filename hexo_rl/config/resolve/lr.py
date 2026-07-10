"""CONFRES S1/B2 — loud declared-vs-effective LR on a full-checkpoint resume.

On a full resume the learning rate is checkpoint-STATE-owned (``optimizer_state.param_groups[].lr``
+ ``scheduler_state``), so a declared variant ``lr:`` is intentionally NOT applied — but
pre-CONFRES that override was SILENT (the v2-LR strike). This makes it LOUD without changing
precedence (the checkpoint still wins).

Refinement over the design's §5 sketch: the WARN compares the declared lr against the
checkpoint's BAKED initial lr (``config["lr"]`` — what the scheduler anneals FROM), NOT the
annealed effective lr. The effective lr is ALWAYS below the initial lr after any annealing, so a
declared-vs-effective comparison would fire on every normal resume. The real strike is a declared
override that differs from the baked lr → silently ignored. The annealed effective (state-blob)
value is still carried on ``LrProvenance`` for the batch-6 ``resolved_config`` emission (B2:
read the state blob, not just config). (design §6 S1.)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LrProvenance:
    declared: float | None       # operator variant/cli lr (intent)
    baked: float | None          # checkpoint's baked initial lr (scheduler anneals from this)
    effective: float | None      # restored optimizer param_group lr (annealed — what runs now)
    override_ignored: bool       # declared present AND declared != baked → override silently dropped


def resolve_lr_provenance(
    declared: float | None,
    baked: float | None,
    effective: float | None,
    *,
    rel_tol: float = 1e-9,
) -> LrProvenance:
    """Build LR provenance for a full-checkpoint resume.

    ``override_ignored`` is True only when a declared lr is present AND differs (beyond
    ``rel_tol``) from the checkpoint's baked initial lr — i.e. the operator asked for an lr the
    resume silently drops. A normal resume (declared == baked, effective merely annealed) does
    NOT flag.
    """
    override_ignored = (
        declared is not None
        and baked is not None
        and abs(float(declared) - float(baked)) > rel_tol * max(abs(float(baked)), 1.0)
    )
    return LrProvenance(
        declared=None if declared is None else float(declared),
        baked=None if baked is None else float(baked),
        effective=None if effective is None else float(effective),
        override_ignored=override_ignored,
    )
