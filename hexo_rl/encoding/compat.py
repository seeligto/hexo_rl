"""Backward-compat encoding inference for legacy checkpoints.

Authored §172 Phase A3 (2026-05-09). Used by `resolve_from_checkpoint`
when a checkpoint lacks `metadata['encoding_name']`. The §172 A5
migration script will stamp metadata onto every existing checkpoint;
after that, this fallback path becomes unreachable for in-tree work.

Heuristic order (per A2 §5.5 + §5.6):
  1. Filename match — longest-first against registered names.
  2. State-dict shape inference — first conv `in_channels` × policy
     `out_features` → unique encoding (or ambiguous → error).
"""
from __future__ import annotations

from typing import Any, Mapping

from hexo_rl.encoding._probes import FIRST_CONV_KEYS as _FIRST_CONV_KEYS
from hexo_rl.encoding._probes import POLICY_FC_KEYS as _POLICY_FC_KEYS
from hexo_rl.encoding.registry import EncodingRegistryError, _load


def legacy_spec_for_registry_name(name: str):
    """Map a registry encoding name to a legacy NamedTuple ``EncodingSpec``.

    The legacy resolver (``hexo_rl.utils.encoding.resolve_encoding``) only
    knows the historical names (v6 / v6w25 / v8). Registry-only names
    (v7full, v7, v7e30, v7mw, v8_canvas_realness) bridge to the wire-
    compatible legacy spec so downstream consumers reading
    ``state_stride`` / ``chain_stride`` keep working.

      v7full / v7 / v7e30  → v6_spec()  (same wire format, distinct anchor tag)
      v7mw                  → v6_spec()  (same wire format; multi-window label
                                          surfaces via the registry spec)
      v8_canvas_realness    → v8_spec()  (same wire format, plane-8 polarity differs)

    Lazy imports avoid a circular-import path through
    ``hexo_rl.utils.encoding`` at package init.
    """
    from hexo_rl.utils.encoding import resolve_encoding, v6_spec, v8_spec

    if name in ("v6", "v6w25", "v8"):
        return resolve_encoding({"encoding": {"version": name}})
    if name in ("v7full", "v7", "v7e30", "v7mw"):
        return v6_spec()
    if name == "v8_canvas_realness":
        return v8_spec()
    raise ValueError(
        f"legacy_spec_for_registry_name: no legacy bridge for registry "
        f"encoding {name!r}. Add a mapping if the wire format matches an "
        f"existing legacy spec."
    )


def _filename_match(path_hint: str) -> str | None:
    """Return registered name found in `path_hint`, longest-first."""
    if not path_hint:
        return None
    candidates = sorted(_load().keys(), key=len, reverse=True)
    for name in candidates:
        if name in path_hint:
            return name
    return None


def _shape_match(state_dict: Mapping[str, Any]) -> list[str]:
    """Return all registered names whose (n_planes, policy_logit_count) match.

    Returns [] if no probe key found in state_dict.
    """
    pfc = None
    for k in _POLICY_FC_KEYS:
        if k in state_dict:
            pfc = state_dict[k]
            break
    conv = None
    for k in _FIRST_CONV_KEYS:
        if k in state_dict:
            conv = state_dict[k]
            break
    if pfc is None and conv is None:
        return []

    matches: list[str] = []
    for name, spec in _load().items():
        ok = True
        if pfc is not None:
            if int(pfc.shape[0]) != spec.policy_logit_count:
                ok = False
        if ok and conv is not None:
            if int(conv.shape[1]) != spec.n_planes:
                ok = False
        if ok:
            matches.append(name)
    return matches


def infer_encoding_from_state_dict(
    state_dict: Mapping[str, Any], path_hint: str = ""
) -> str:
    """Return registered encoding name for a legacy checkpoint.

    Filename heuristic first (longest-first); falls back to state-dict
    shape inference. Raises `EncodingRegistryError` on ambiguity or
    no-match.
    """
    name = _filename_match(path_hint)
    if name is not None:
        return name

    matches = _shape_match(state_dict)
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise EncodingRegistryError(
            "could not infer encoding: no filename match and no state-dict probe key "
            f"matched any registered (n_planes, policy_logit_count). path_hint={path_hint!r}"
        )
    raise EncodingRegistryError(
        f"could not disambiguate encoding from state-dict shape alone; "
        f"matches: {sorted(matches)}. Stamp metadata['encoding_name'] explicitly "
        f"(see §172 A5 migration). path_hint={path_hint!r}"
    )
