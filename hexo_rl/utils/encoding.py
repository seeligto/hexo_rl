"""Encoding-version resolver for v6 / v8 path routing.

Single source of truth for numeric constants when code must operate on
either encoding version. v6-only call sites should keep importing
`hexo_rl.utils.constants.*` directly; only thread an `EncodingSpec`
through code that genuinely routes both paths.

Contract: `docs/designs/encoding_v8_contract.md` §4.2.

Note (§172 A3): hexo_rl.encoding is the new canonical registry; this module
remains for backward compat until A4 migrates consumers.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, NamedTuple, Optional

from hexo_rl.utils import constants as _c


EncodingVersion = Literal["v6", "v6w25", "v8"]


class EncodingSpec(NamedTuple):
    """Numeric constants bundle for one encoding version.

    All fields are derived deterministically from `version`; the named
    tuple shape is identical for v6 and v8 so callers can branch on the
    `version` field while reading values uniformly.
    """

    version: EncodingVersion
    board_size: int
    half: int
    n_cells: int
    n_actions: int
    n_planes: int
    legal_move_radius: int
    cluster_threshold: Optional[int]  # None for v8 (K-aggregation retired)
    cluster_window_size: Optional[int]  # None for v8; 19 for v6; 25 for v6w25 (§168 Gate 3, §171 plumbing)
    state_stride: int
    chain_stride: int
    policy_stride: int
    aux_stride: int

    @property
    def has_pass_slot(self) -> bool:
        return self.n_actions != self.n_cells

    def to_pyo3(self) -> "engine.EncodingSpec":  # type: ignore[name-defined]
        """Return a PyO3 engine.EncodingSpec mirror of this NamedTuple.

        Raises ValueError if any of cluster_window_size / cluster_threshold /
        legal_move_radius is None (v8 has cluster_threshold=cluster_window_size=None
        because K-aggregation is retired; v8 self-play does NOT use the cluster
        plumbing surface).
        """
        from engine import EncodingSpec as _PyEnc
        if self.cluster_window_size is None or self.cluster_threshold is None:
            raise ValueError(
                f"EncodingSpec(version={self.version!r}) has no cluster window / threshold; "
                "to_pyo3 is only defined for v6-family encodings"
            )
        return _PyEnc(
            cluster_window_size=int(self.cluster_window_size),
            cluster_threshold=int(self.cluster_threshold),
            legal_move_radius=int(self.legal_move_radius),
            board_size=int(self.board_size),
        )


# v6 chain-plane count is shared with v8: 6 axes × 1 = 6 (Q13).
_N_CHAIN_PLANES = 6

_V6_SPEC = EncodingSpec(
    version="v6",
    board_size=_c.BOARD_SIZE,
    half=(_c.BOARD_SIZE - 1) // 2,
    n_cells=_c.NUM_CELLS,
    n_actions=_c.NUM_CELLS + 1,  # 362 (361 cells + 1 pass slot)
    n_planes=_c.BUFFER_CHANNELS,
    legal_move_radius=5,
    cluster_threshold=5,
    cluster_window_size=19,
    state_stride=_c.BUFFER_CHANNELS * _c.NUM_CELLS,  # 2888
    chain_stride=_N_CHAIN_PLANES * _c.NUM_CELLS,  # 2166
    policy_stride=_c.NUM_CELLS + 1,  # 362
    aux_stride=_c.NUM_CELLS,  # 361
)

_V8_SPEC = EncodingSpec(
    version="v8",
    board_size=_c.BOARD_SIZE_V8,
    half=(_c.BOARD_SIZE_V8 - 1) // 2,
    n_cells=_c.NUM_CELLS_V8,
    n_actions=_c.N_ACTIONS_V8,  # 625, no pass slot
    n_planes=_c.BUFFER_CHANNELS_V8,
    legal_move_radius=_c.LEGAL_MOVE_RADIUS_V8,
    cluster_threshold=None,
    cluster_window_size=None,
    state_stride=_c.BUFFER_CHANNELS_V8 * _c.NUM_CELLS_V8,  # 6875
    chain_stride=_N_CHAIN_PLANES * _c.NUM_CELLS_V8,  # 3750
    policy_stride=_c.N_ACTIONS_V8,  # 625
    aux_stride=_c.NUM_CELLS_V8,  # 625
)

# v6w25 (§168 Gate 3 / §171 P2 reopen): widened 25×25 cluster window with
# threshold=8, legal-move radius=8. Same canvas board_size as v6 (19); only
# perception parameters differ. Wire-format planes and stride identical to v6.
_V6W25_SPEC = EncodingSpec(
    version="v6w25",
    board_size=_c.BOARD_SIZE,
    half=(_c.BOARD_SIZE - 1) // 2,
    n_cells=_c.NUM_CELLS,
    n_actions=_c.NUM_CELLS + 1,
    n_planes=_c.BUFFER_CHANNELS,
    legal_move_radius=8,
    cluster_threshold=8,
    cluster_window_size=25,
    state_stride=_c.BUFFER_CHANNELS * _c.NUM_CELLS,
    chain_stride=_N_CHAIN_PLANES * _c.NUM_CELLS,
    policy_stride=_c.NUM_CELLS + 1,
    aux_stride=_c.NUM_CELLS,
)


def resolve_encoding(config: Mapping[str, Any]) -> EncodingSpec:
    """Resolve an EncodingSpec from a config mapping.

    Accepts BOTH forms (§172 A4.5 canonical = string form):
      - `cfg['encoding'] = "v6w25"`           string form (preferred)
      - `cfg['encoding'] = {'version': 'v6'}` mapping form (legacy)

    Default: "v6". Anything other than "v6" / "v6w25" / "v8" raises
    ValueError. Configs without an "encoding" key resolve to v6
    (canonical default preserves byte-exact pre-§166 behavior).
    """
    section = config.get("encoding") if config else None
    if section is None:
        version: str = "v6"
    elif isinstance(section, str):
        version = section
    elif isinstance(section, Mapping):
        version = section.get("version", "v6")
    else:
        raise ValueError(
            f"encoding section must be a string or mapping; got {type(section).__name__}"
        )

    if version == "v6":
        return _V6_SPEC
    if version == "v6w25":
        return _V6W25_SPEC
    if version == "v8":
        return _V8_SPEC
    raise ValueError(
        f"unknown encoding.version {version!r}; expected 'v6', 'v6w25', or 'v8'"
    )


def v6_spec() -> EncodingSpec:
    """Return the v6 spec without consulting any config (test helper)."""
    return _V6_SPEC


def v6w25_spec() -> EncodingSpec:
    """Return the v6w25 spec without consulting any config (test helper)."""
    return _V6W25_SPEC


def v8_spec() -> EncodingSpec:
    """Return the v8 spec without consulting any config (test helper)."""
    return _V8_SPEC
