"""Canonical constants for the Hex Tac Toe board.

v6 (canonical default) and v8 (gated, Path β) symbol sets coexist. v6
symbols are unprefixed and unchanged from pre-§166 master; v8 symbols
carry a `_V8` suffix and are only used when `encoding.version == "v8"`.

Use `hexo_rl.utils.encoding.resolve_encoding(config)` to get an
`EncodingSpec` for version-aware code; do not reach for the `_V8`
symbols directly unless you know you are on the v8 path.
"""

# ── v6 (canonical default) ──────────────────────────────────────────────────

BOARD_SIZE: int = 19
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 361

# HEXB v6 buffer wire format: 8 planes (cur ply-0..3 + opp ply-0..3).
# Python-side mirror of engine/src/replay_buffer/sym_tables.rs:KEPT_PLANE_INDICES.
BUFFER_CHANNELS: int = 8
KEPT_PLANE_INDICES: list[int] = [0, 1, 2, 3, 8, 9, 10, 11]

# ── v8 Path β (gated) ───────────────────────────────────────────────────────
#
# 25×25 fixed-max bbox-of-all-stones, 11 planes (8 KEPT + off_window +
# moves_remaining_bcast + ply_parity_bcast), R=8 perception, no pass slot,
# K-aggregation removed. See docs/designs/encoding_v8_contract.md §1.2.

BOARD_SIZE_V8: int = 25
NUM_CELLS_V8: int = BOARD_SIZE_V8 * BOARD_SIZE_V8  # 625

# v8 native plane count. NO KEPT_PLANE_INDICES slice under v8 — the encoder
# emits 11 planes directly (no 18-plane intermediate).
BUFFER_CHANNELS_V8: int = 11

# Spatial-only policy (no pass slot, P1 close-out: pass dead in HTTT).
N_ACTIONS_V8: int = NUM_CELLS_V8  # 625

# HTTT rule baseline; replaces v6's r=5 cap (§145 Option α').
LEGAL_MOVE_RADIUS_V8: int = 8

# bbox dilation margin — equals LEGAL_MOVE_RADIUS_V8.
MARGIN_M_V8: int = 8
