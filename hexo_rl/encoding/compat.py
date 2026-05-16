"""Backward-compat encoding inference for legacy checkpoints.

Authored §172 Phase A3 (2026-05-09). Used by `resolve_from_checkpoint`
when a checkpoint lacks `metadata['encoding_name']`. The §172 A5
migration script will stamp metadata onto every existing checkpoint;
after that, this fallback path becomes unreachable for in-tree work.

Heuristic order (per A2 §5.5 + §5.6):
  1. Filename match — longest-first against registered names.
  2. State-dict shape inference — first conv `in_channels` × policy
     `out_features` → unique encoding (or ambiguous → error).

§176 P3 retired the legacy `hexo_rl.utils.encoding` NamedTuple shim.
The trainer-side wire-format propagation (cluster_window_size /
cluster_threshold / legal_move_radius / board_size that the Rust
SelfPlayRunner kwargs consume) used to ride on that NamedTuple. Now
the wire-format mapping lives here as ``WIRE_FORMAT_SPECS`` keyed by
registry name; ``legacy_spec_for_registry_name`` returns a tiny
``WireFormatSpec`` dataclass carrying those four scalars plus the
registry ``name``.

For v6-family (v6 / v7full / v7 / v7e30 / v7mw) the wire format is
the legacy v6 16-plane layout (cw=19, ct=5, lmr=5, bs=19). v6w25 is
its own wire format (cw=25, ct=8, lmr=8, bs=25). v8 / v8_canvas_realness
share the v8 11-plane layout (no cluster fields, lmr=8, bs=25); v8 is
not on a Rust selfplay path today (loud-fail in pool resolver).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from hexo_rl.encoding._probes import FIRST_CONV_KEYS as _FIRST_CONV_KEYS
from hexo_rl.encoding._probes import POLICY_FC_KEYS as _POLICY_FC_KEYS
from hexo_rl.encoding.registry import EncodingRegistryError, _load


@dataclass(frozen=True, slots=True)
class WireFormatSpec:
    """Wire-format scalars for a registry encoding name (§176 P3).

    Replaces the four-field surface of the retired legacy
    ``hexo_rl.utils.encoding.EncodingSpec`` NamedTuple — the only
    contract downstream consumers (trainer ckpt-load propagation +
    pool runner-kwarg construction) actually relied on.

    ``cluster_window_size`` / ``cluster_threshold`` are ``None`` for
    v8 family encodings (K-aggregation retired).
    """

    name: str
    cluster_window_size: Optional[int]
    cluster_threshold: Optional[int]
    legal_move_radius: int
    board_size: int

    def to_pyo3(self):
        """Build the 4-field engine.EncodingSpec the Rust SelfPlayRunner kwarg expects.

        Raises ValueError if cluster_window_size or cluster_threshold is None
        (v8 family — no cluster plumbing).

        TODO(§P3.2): retire alongside SelfPlayRunner encoding kwarg deletion.
        """
        from engine import EncodingSpec as _PyEnc
        if self.cluster_window_size is None or self.cluster_threshold is None:
            raise ValueError(
                f"WireFormatSpec(name={self.name!r}) has no cluster window/threshold; "
                "to_pyo3 is only defined for v6-family encodings"
            )
        return _PyEnc(
            cluster_window_size=int(self.cluster_window_size),
            cluster_threshold=int(self.cluster_threshold),
            legal_move_radius=int(self.legal_move_radius),
            board_size=int(self.board_size),
        )


# Registry name → wire-format scalars (§176 P3).
#
# v6-family (v6 / v7full / v7 / v7e30 / v7mw) all share the legacy v6
# wire format: 19×19 cluster window, threshold=5, radius=5, canvas
# board_size=19. v6w25 is its own wire format. v8 / v8_canvas_realness
# share the v8 wire format; cluster fields stay None because v8 retired
# K-aggregation (Rust selfplay path also blocked at the pool resolver).
_V6_WIRE = WireFormatSpec(
    name="v6", cluster_window_size=19, cluster_threshold=5,
    legal_move_radius=5, board_size=19,
)
_V6W25_WIRE = WireFormatSpec(
    name="v6w25", cluster_window_size=25, cluster_threshold=8,
    legal_move_radius=8, board_size=25,
)
_V8_WIRE = WireFormatSpec(
    name="v8", cluster_window_size=None, cluster_threshold=None,
    legal_move_radius=8, board_size=25,
)
WIRE_FORMAT_SPECS: dict[str, WireFormatSpec] = {
    "v6":                  _V6_WIRE,
    "v6w25":               _V6W25_WIRE,
    "v7full":              _V6_WIRE,
    "v7":                  _V6_WIRE,
    "v7e30":               _V6_WIRE,
    "v7mw":                _V6_WIRE,
    "v8":                  _V8_WIRE,
    "v8_canvas_realness":  _V8_WIRE,
}


def legacy_spec_for_registry_name(name: str) -> WireFormatSpec:
    """Map a registry encoding name to a ``WireFormatSpec`` (§176 P3).

    Wire-format mapping (preserves byte-exact pre-§176 selfplay
    behaviour: v6-family Rust workers see v6 wire constants, v6w25
    workers see v6w25 wire constants):

      v6 / v7full / v7 / v7e30 / v7mw  → v6 wire (cw=19, ct=5, lmr=5, bs=19)
      v6w25                            → v6w25 wire (cw=25, ct=8, lmr=8, bs=25)
      v8 / v8_canvas_realness          → v8 wire (cw=None, ct=None, lmr=8, bs=25)
    """
    spec = WIRE_FORMAT_SPECS.get(name)
    if spec is None:
        raise ValueError(
            f"legacy_spec_for_registry_name: no wire-format mapping for "
            f"registry encoding {name!r}. Add an entry to "
            f"WIRE_FORMAT_SPECS if the wire format matches an existing "
            f"layout."
        )
    return spec


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
