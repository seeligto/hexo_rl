"""CONFRES 6c — the encoding resolution AUTHORITY (design §2, §4, B5b, N3).

One rule module for the encoding name, delegated to by BOTH invocation surfaces so they cannot
diverge (design law #1):

- Surface A (launch builder) — ``hexo_rl.config.resolve.run_config._resolve_encoding`` calls
  :func:`resolve_encoding` to fold the variant declaration + checkpoint stamp into one
  ``EncodingResolution`` (I1 presence-before-normalize, I2 conflict-raise, B5a absent→stamp).
- Surface B (per-artifact eval loader) — ``hexo_rl.eval.checkpoint_loader.load_model_with_encoding``
  reconciles a caller-declared encoding against the checkpoint's OWN trusted stamp; that
  reconciliation rule is :func:`reconcile_declared_vs_stamp` here, so the eval gate and the launch
  builder apply provably identical raise-on-conflict semantics.

The registry lookup (name → ``EncodingSpec``) is :func:`window_set`, the ONE encoding→spec seam
6c migrates the ReplayBuffer / recent-buffer / batch-buffer consumers onto (design §4 window_set).

Import constraint (§8 N3): stdlib + ``hexo_rl.encoding`` (registry + ``normalize_encoding_name``)
ONLY — NO ``hexo_rl.eval`` / ``.selfplay`` / ``.training`` import, so the resolve package stays
cycle-free (``hexo_rl/eval/__init__.py`` eagerly imports eval_pipeline → evaluator → player_factory
→ config.resolve; a back-edge here would cycle).

Design: docs/designs/confres_design.md §2 (I1/I2), §3 (encoding row), §4 (window_set), §13 (B5a).
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Mapping

from hexo_rl.encoding import lookup as _lookup
from hexo_rl.encoding import normalize_encoding_name as _normalize

# Distinct sentinel for an ABSENT encoding declaration (I1). NOT None — ``normalize_encoding_name``
# coerces None → "v6", which would make an absent key look like a PRESENT "v6" and I2-raise against
# any non-v6 stamp, breaking every no-`encoding:` variant on resume (B5a).
UNSPECIFIED = types.new_class("_UnspecifiedEncodingSentinel", (), {})()


class EncodingConflictError(ValueError):
    """A PRESENT declared encoding disagrees with a PRESENT checkpoint stamp (I2, design §2/§3).

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers still catch it. The
    message NAMES both sources + values — the F1-forensic requirement that a checkpoint may INFORM
    but never SILENTLY override a declared value. The launch builder wraps this in its own
    ``ConfigConflictError`` (also a ``ValueError`` subclass) with the same contract; the eval loader
    raises its own ``DeclaredEncodingMismatchError``. Both delegate the DECISION to
    :func:`reconcile_declared_vs_stamp` so the rule is single-sourced.
    """

    def __init__(self, declared: str, stamp: str, *, stamp_source: str = "checkpoint"):
        self.declared = declared
        self.stamp = stamp
        self.stamp_source = stamp_source
        super().__init__(
            f"encoding conflict: declared={declared!r} vs {stamp_source}={stamp!r} "
            "(declared value and checkpoint stamp disagree; a checkpoint may INFORM but never "
            "silently override a declared encoding — the two are state-dict-shape-identical for "
            "v6_live2 vs v6_live2_ls so shape inference cannot catch this. Re-stamp the checkpoint "
            "or fix the declared encoding.)"
        )


@dataclass(frozen=True)
class EncodingResolution:
    """The resolved encoding name + which source won (for provenance emission).

    ``source`` ∈ {"variant","checkpoint","default"}. ``declared`` is the normalized declared name
    or ``UNSPECIFIED``; ``stamp`` is the normalized stamp name or ``None``. The launch builder maps
    this onto a ``ResolvedValue`` with the full ``inputs_seen`` provenance.
    """

    name: str
    source: str
    declared: Any  # normalized str, or UNSPECIFIED sentinel
    stamp: str | None


def reconcile_declared_vs_stamp(
    declared: Any,
    stamp: str | None,
    *,
    stamp_source: str = "checkpoint",
) -> EncodingResolution:
    """The single raise-on-conflict rule, shared by BOTH surfaces (design law #1, I1/I2/B5a).

    ``declared`` is either the ``UNSPECIFIED`` sentinel (no operator declaration — I1 presence test
    said absent BEFORE normalization) or a normalized encoding name string. ``stamp`` is the
    checkpoint's normalized stamp name or ``None`` (fresh run / bare weights).

    Precedence (design §3 encoding row):
      - declared PRESENT and stamp PRESENT and they DIFFER → raise :class:`EncodingConflictError`.
      - declared PRESENT (agree with stamp, or no stamp) → declared wins, source="variant".
      - declared ABSENT and stamp PRESENT → stamp wins, source="checkpoint" (metadata-wins, B5a).
      - declared ABSENT and no stamp → the "v6" compat default, source="default".
    """
    declared_present = declared is not UNSPECIFIED
    if declared_present and stamp is not None:
        if declared != stamp:
            raise EncodingConflictError(declared, stamp, stamp_source=stamp_source)
        return EncodingResolution(declared, "variant", declared, stamp)
    if declared_present:
        return EncodingResolution(declared, "variant", declared, stamp)
    if stamp is not None:
        return EncodingResolution(stamp, "checkpoint", declared, stamp)
    default = _normalize(None)  # "v6"
    return EncodingResolution(default, "default", declared, stamp)


def normalize_declared(raw_present: bool, raw_value: Any) -> Any:
    """I1 presence-before-normalize: a key present in a declaration layer → normalized name;
    an absent key → the ``UNSPECIFIED`` sentinel (NOT ``normalize(None)`` == "v6", which would
    make absent look like a PRESENT "v6" and I2-raise against any non-v6 stamp — B5a).
    """
    return _normalize(raw_value) if raw_present else UNSPECIFIED


def normalize_stamp(stamps: Mapping[str, Any]) -> str | None:
    """Normalize the checkpoint's encoding stamp (or ``None`` when no stamp is present)."""
    if "encoding" not in stamps:
        return None
    return _normalize(stamps["encoding"])


def resolve_encoding(
    declared_present: bool,
    declared_raw: Any,
    stamps: Mapping[str, Any],
) -> EncodingResolution:
    """Surface-A convenience: fold a raw declaration presence/value + a stamp mapping into an
    :class:`EncodingResolution` via the shared :func:`reconcile_declared_vs_stamp` rule.
    """
    declared = normalize_declared(declared_present, declared_raw)
    stamp = normalize_stamp(stamps)
    return reconcile_declared_vs_stamp(declared, stamp)


def window_set(name: Any):
    """Resolve an encoding NAME (str / dict / EncodingSpec) to its registry ``EncodingSpec``.

    The ONE encoding→spec authority (design §4 window_set). Consumers that used to size the
    ReplayBuffer / recent-buffer / batch buffers from a raw ``config.get("encoding")`` +
    ``resolve_from_config`` (or a v6 literal fallback) call THIS instead, so a metadata-wins resume
    sizes them from the RESOLVED spec, not the pre-checkpoint one. Funnels through
    ``normalize_encoding_name`` so the dict form (``{"version": ...}`` post-propagation) resolves.
    """
    return _lookup(_normalize(name))
