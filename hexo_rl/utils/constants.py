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
HISTORY_LEN: int = 8
