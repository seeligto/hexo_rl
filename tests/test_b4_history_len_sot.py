"""B4 / MAGIC-6 — HISTORY_LEN single source of truth + opp-plane coupling pin.

HISTORY_LEN was defined independently in `hexo_rl/utils/constants.py` and
`hexo_rl/env/game_state.py` (no cross-check). It is load-bearing-coupled to the
plane layout: the opponent-stone block begins at source plane HISTORY_LEN
(== OPP_STONE_PLANE == 8). A future history-depth experiment editing one
definition but not the other (or not the opp-plane offset) would silently
corrupt the corpus tensor build. game_state now imports the single SoT from
constants; these pins guard the coupling.
"""
from hexo_rl.utils.constants import HISTORY_LEN as CONST_HISTORY_LEN
from hexo_rl.env.game_state import HISTORY_LEN as GS_HISTORY_LEN
from hexo_rl.encoding.resolvers import _OPP_STONE_SRC_PLANE


def test_history_len_single_sot():
    """game_state re-exports the constants SoT, not a second definition."""
    assert GS_HISTORY_LEN == CONST_HISTORY_LEN


def test_history_len_couples_to_opp_stone_plane():
    """The opponent block begins at source plane HISTORY_LEN; if HISTORY_LEN
    ever moves, the opp-stone source plane MUST move with it (else the
    `tensor[k, HISTORY_LEN]` opp write in game_state corrupts silently)."""
    assert CONST_HISTORY_LEN == _OPP_STONE_SRC_PLANE
