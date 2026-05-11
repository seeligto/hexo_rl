"""Full-schema EncodingSpec dataclass — mirror of Rust `RegistrySpec`.

Authored §172 Phase A3 (2026-05-09). Pure-Python; no torch/engine imports.
PyO3 binding extension lives in §172 A4.

§173 A3 additions (2026-05-11):
  - kept_plane_indices: tuple[int, ...] — physical source-plane indices retained
    by wire format. len == n_planes. Canonical v6-family block: [0,1,2,3,8,9,10,11].
  - n_source_planes: int — source tensor plane count before slicing.
    v6 family = 18; v8 family = 21.
  - state_stride, chain_stride, aux_stride, policy_stride: @property helpers
    mirroring Rust RegistrySpec methods.
  - n_cells: FIX — was board_size² (wrong for multi-window v6w25 which returned
    361 where 625 is correct). Now trunk_size² matching Rust. Bug was pre-existing
    but never manifested because v6w25 Python path was runtime-blocked.

Schema: docs/designs/encoding_registry_design.md §3.1, §5.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ValuePool = Literal["none", "min", "max", "mean"]
PolicyPool = Literal["none", "scatter_max", "scatter_mean"]

# Module-level mirror of Rust `replay_buffer::sym_tables::N_CHAIN_PLANES`.
# Tied to D6 hex geometry (2 players × 3 hex axes = 6 chain planes). Not
# an encoding choice — see §173 A2 §0 SSoT principle for why it stays as
# a Rust const + Python mirror rather than a TOML field.
# If a non-hex sym table ever lands, update both this constant and the
# Rust const in the same commit.
_N_CHAIN_PLANES: int = 6


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

    # ── §173 A3 additions ────────────────────────────────────────────────
    kept_plane_indices: tuple[int, ...]
    n_source_planes: int

    # ── properties ───────────────────────────────────────────────────────

    @property
    def n_actions(self) -> int:
        return self.policy_logit_count

    @property
    def n_cells(self) -> int:
        """Cells per trunk input tensor = trunk_size².

        §173 A3 semantic fix: was board_size² (wrong for multi-window v6w25
        which returned 361 instead of 625). Now trunk_size² matching Rust.
        """
        return self.trunk_size * self.trunk_size

    @property
    def state_stride(self) -> int:
        """State plane stride = n_planes × n_cells."""
        return self.n_planes * self.n_cells

    @property
    def chain_stride(self) -> int:
        """Chain plane stride = _N_CHAIN_PLANES × n_cells.

        N_CHAIN_PLANES = 6 is D6 hex geometry (2 players × 3 axes), not an
        encoding choice. See module-level `_N_CHAIN_PLANES` for details.
        """
        return _N_CHAIN_PLANES * self.n_cells

    @property
    def aux_stride(self) -> int:
        """Aux plane stride = n_cells (single aux plane per buffer layout)."""
        return self.n_cells

    @property
    def policy_stride(self) -> int:
        """Policy stride = policy_logit_count."""
        return self.policy_logit_count
