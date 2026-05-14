"""Canonical constants for the Hex Tac Toe board (v6 only).

v6 symbols stay here as the legacy default; v8 / v6w25 / v7full / etc.
are sourced from the canonical encoding registry at
``engine/src/encoding/registry.toml`` via ``hexo_rl.encoding.lookup(name)``.

§176 P5 retired the v8 block from this module; downstream v8 callers
import their numeric constants from ``hexo_rl.bootstrap.dataset_v8``
(which now sources them from the registry). §176 P3 retired the legacy
``hexo_rl.utils.encoding`` NamedTuple shim entirely; downstream
consumers route through ``hexo_rl.encoding`` (registry) and
``hexo_rl.encoding.compat`` (wire-format scalars).
"""

# ── v6 (canonical default) ──────────────────────────────────────────────────

BOARD_SIZE: int = 19
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 361

# HEXB v6 buffer wire format: 8 planes (cur ply-0..3 + opp ply-0..3).
BUFFER_CHANNELS: int = 8

# §173 A6 DEPRECATED: use EncodingSpec.kept_plane_indices from the registry instead.
# Retained for callers not yet migrated to registry-sourced construction.
# Equivalent to: hexo_rl.encoding.lookup("v6").kept_plane_indices
# Will be removed in §174 cleanup.
KEPT_PLANE_INDICES: list[int] = [0, 1, 2, 3, 8, 9, 10, 11]

# AlphaZero history length (current + 7 prior timesteps).
HISTORY_LEN: int = 8
