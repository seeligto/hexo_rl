"""Encoding-version resolver for v6 / v8 path routing.

Single source of truth for numeric constants when code must operate on
either encoding version. v6-only call sites should keep importing
`hexo_rl.utils.constants.*` directly; only thread an `EncodingSpec`
through code that genuinely routes both paths.

Contract: `docs/designs/encoding_v8_contract.md` §4.2.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, NamedTuple, Optional

from hexo_rl.utils import constants as _c


EncodingVersion = Literal["v6", "v8"]


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
    state_stride: int
    chain_stride: int
    policy_stride: int
    aux_stride: int

    @property
    def has_pass_slot(self) -> bool:
        return self.n_actions != self.n_cells


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
    state_stride=_c.BUFFER_CHANNELS_V8 * _c.NUM_CELLS_V8,  # 6875
    chain_stride=_N_CHAIN_PLANES * _c.NUM_CELLS_V8,  # 3750
    policy_stride=_c.N_ACTIONS_V8,  # 625
    aux_stride=_c.NUM_CELLS_V8,  # 625
)


def resolve_encoding(config: Mapping[str, Any]) -> EncodingSpec:
    """Resolve an EncodingSpec from a config mapping.

    Reads `config["encoding"]["version"]`. Default: "v6". Anything other
    than "v6" or "v8" raises ValueError.

    Mappings without an "encoding" key resolve to v6 (canonical default
    preserves byte-exact pre-§166 behavior).
    """
    section = config.get("encoding") if config else None
    if section is None:
        version: str = "v6"
    elif isinstance(section, Mapping):
        version = section.get("version", "v6")
    else:
        raise ValueError(
            f"encoding section must be a mapping; got {type(section).__name__}"
        )

    if version == "v6":
        return _V6_SPEC
    if version == "v8":
        return _V8_SPEC
    raise ValueError(
        f"unknown encoding.version {version!r}; expected 'v6' or 'v8'"
    )


def v6_spec() -> EncodingSpec:
    """Return the v6 spec without consulting any config (test helper)."""
    return _V6_SPEC


def v8_spec() -> EncodingSpec:
    """Return the v8 spec without consulting any config (test helper)."""
    return _V8_SPEC
