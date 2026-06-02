"""Non-encoding training/selfplay hyperparameters.

§176 P4 retired the v6 encoding constants (BOARD_SIZE / NUM_CELLS /
BUFFER_CHANNELS / KEPT_PLANE_INDICES) from this module. Per §172 the
canonical source of truth for every encoding is the registry at
``engine/src/encoding/registry.toml``; route through
``hexo_rl.encoding.lookup(name)`` (Python) or
``engine::encoding::lookup_or_panic(name)`` (Rust).

§176 P5 retired the v8 block; downstream v8 callers use
``hexo_rl.bootstrap.dataset_v8``. §176 P3 retired the legacy
``hexo_rl.utils.encoding`` NamedTuple shim entirely; consumers route
through ``hexo_rl.encoding`` (registry) and ``hexo_rl.encoding.compat``
(wire-format scalars).

What remains here are non-encoding hyperparameters — values that are not
geometry / plane-layout / action-space attributes of any encoding and
therefore have no place in the registry.
"""

# AlphaZero history length (current + 7 prior timesteps). Self-play /
# training hyperparameter, not an encoding parameter — kept here.
#
# B4 (2026-06-02): SINGLE SoT for HISTORY_LEN (game_state.py now imports this,
# was a second independent definition). LOAD-BEARING COUPLING: the 18-plane
# layout places HISTORY_LEN my-stone history planes (source 0..7) then the
# opponent block starting at source plane HISTORY_LEN — i.e. HISTORY_LEN ==
# OPP_STONE_PLANE (== resolvers._OPP_STONE_SRC_PLANE == 8 == engine
# core.rs OPP_STONE_PLANE). A future history-depth change here MUST move the
# opponent-block offset in lockstep; test_b4_history_len_sot pins the equality.
HISTORY_LEN: int = 8
