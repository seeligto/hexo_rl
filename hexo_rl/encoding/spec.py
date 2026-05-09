"""Full-schema EncodingSpec dataclass — mirror of Rust `RegistrySpec`.

Authored §172 Phase A3 (2026-05-09). Pure-Python; no torch/engine imports.
PyO3 binding extension lives in §172 A4.

Schema: docs/designs/encoding_registry_design.md §3.1, §5.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ValuePool = Literal["none", "min", "max", "mean"]
PolicyPool = Literal["none", "scatter_max", "scatter_mean"]


@dataclass(frozen=True, slots=True)
class EncodingSpec:
    """Immutable record for one registered encoding.

    Field set matches `engine/src/encoding/spec.rs::RegistrySpec` 1-to-1.
    Validation runs at registry load time; consumers can trust invariants.
    """

    name: str
    board_size: int
    trunk_size: int
    cluster_window_size: int | None
    cluster_threshold: int | None
    legal_move_radius: int
    n_planes: int
    plane_layout: tuple[str, ...]
    policy_logit_count: int
    has_pass_slot: bool
    is_multi_window: bool
    value_pool: ValuePool
    policy_pool: PolicyPool
    sym_table_id: str
    schema_version: int
    notes: str

    @property
    def n_actions(self) -> int:
        return self.policy_logit_count

    @property
    def n_cells(self) -> int:
        return self.board_size * self.board_size
