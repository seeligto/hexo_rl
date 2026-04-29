"""Canonical constants for the Hex Tac Toe board."""

BOARD_SIZE: int = 19
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 361

# HEXB v6 buffer wire format: 8 planes (cur ply-0..3 + opp ply-0..3).
# Python-side mirror of engine/src/replay_buffer/sym_tables.rs:KEPT_PLANE_INDICES.
BUFFER_CHANNELS: int = 8
KEPT_PLANE_INDICES: list[int] = [0, 1, 2, 3, 8, 9, 10, 11]
