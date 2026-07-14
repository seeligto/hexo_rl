"""20 fixed HeXO positions for the strix fidelity check.

Each: stones = list of ((q,r), player) with player 1=P1, -1=P2;
current_player (1/-1); moves_remaining (1 or 2). Positions are hand-built
to cover: openings, both players to move, both moves_remaining values,
near-win threats (to exercise threat features), blocked lines, and
multi-axis structure. Radius 6 / win_length 6 (strix checkpoint game_config).
"""

WIN_LENGTH = 6
RADIUS = 6

POSITIONS = [
    # 1: single seeded stone, P1 to move mr=1 (real opening-style position)
    {"name": "p1_single_seed", "stones": [[[0, 0], 1]], "current_player": 1, "moves_remaining": 1},
    # 2: after P1 (0,0); P2 compound turn
    {"name": "p2_after_open", "stones": [[[0, 0], 1]], "current_player": -1, "moves_remaining": 2},
    # 3: P2 mid double-turn (moves_remaining 1)
    {"name": "p2_mid_turn", "stones": [[[0, 0], 1], [[1, 0], -1]], "current_player": -1, "moves_remaining": 1},
    # 4: P1 to move, small cluster
    {"name": "p1_small_cluster", "stones": [[[0, 0], 1], [[1, 0], -1], [[2, 0], -1]], "current_player": 1, "moves_remaining": 2},
    # 5: P1 three-in-a-row threat on axis (1,0)
    {"name": "p1_three_row", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 0], 1], [[0, 1], -1], [[1, 1], -1]], "current_player": 1, "moves_remaining": 1},
    # 6: P2 to move facing P1 four-in-a-row (must block)
    {"name": "p2_block_four", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 0], 1], [[3, 0], 1], [[0, 2], -1], [[1, 2], -1]], "current_player": -1, "moves_remaining": 2},
    # 7: blocked line (P2 stone splits P1 line)
    {"name": "blocked_line", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 0], -1], [[3, 0], 1], [[4, 0], 1]], "current_player": 1, "moves_remaining": 1},
    # 8: multi-axis P1 structure
    {"name": "p1_multi_axis", "stones": [[[0, 0], 1], [[1, 0], 1], [[0, 1], 1], [[0, 2], 1], [[2, -1], -1], [[3, -1], -1]], "current_player": 1, "moves_remaining": 2},
    # 9: P2 has its own threat (axis (0,1))
    {"name": "p2_three_row_axis1", "stones": [[[0, 0], 1], [[3, 0], 1], [[0, 1], -1], [[0, 2], -1], [[0, 3], -1]], "current_player": -1, "moves_remaining": 1},
    # 10: diagonal axis (1,-1) structure
    {"name": "diag_axis", "stones": [[[0, 0], 1], [[1, -1], 1], [[2, -2], 1], [[0, 1], -1], [[1, 1], -1]], "current_player": 1, "moves_remaining": 2},
    # 11: denser midgame, P2 to move
    {"name": "dense_mid", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 0], -1], [[0, 1], -1], [[1, 1], 1], [[2, 1], -1], [[0, 2], 1]], "current_player": -1, "moves_remaining": 2},
    # 12: near-win P1 (five with a gap)
    {"name": "p1_five_gap", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 0], 1], [[4, 0], 1], [[5, 0], 1], [[0, 1], -1], [[1, 1], -1], [[2, 1], -1]], "current_player": 1, "moves_remaining": 1},
    # 13: spread-out stones (large centroid spread)
    {"name": "spread_out", "stones": [[[0, 0], 1], [[5, 0], -1], [[0, 5], 1], [[3, -3], -1]], "current_player": 1, "moves_remaining": 2},
    # 14: two adjacent stones each player, P2 mr=1
    {"name": "two_each_mr1", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 1], -1], [[3, 1], -1]], "current_player": -1, "moves_remaining": 1},
    # 15: L-shape P1
    {"name": "l_shape", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 0], 1], [[2, 1], 1], [[3, -1], -1], [[4, -1], -1]], "current_player": 1, "moves_remaining": 2},
    # 16: opponent double threat
    {"name": "opp_double_threat", "stones": [[[0, 0], -1], [[1, 0], -1], [[2, 0], -1], [[0, 1], -1], [[0, 2], -1], [[3, 3], 1], [[4, 3], 1]], "current_player": 1, "moves_remaining": 1},
    # 17: negative coordinates
    {"name": "neg_coords", "stones": [[[-2, 0], 1], [[-1, 0], 1], [[0, 0], 1], [[-2, 1], -1], [[-1, 1], -1]], "current_player": 1, "moves_remaining": 2},
    # 18: single far stone pair
    {"name": "far_pair", "stones": [[[0, 0], 1], [[6, -3], -1]], "current_player": -1, "moves_remaining": 2},
    # 19: symmetric ring-ish
    {"name": "ring_ish", "stones": [[[1, 0], 1], [[-1, 0], -1], [[0, 1], 1], [[0, -1], -1], [[1, -1], 1], [[-1, 1], -1]], "current_player": 1, "moves_remaining": 1},
    # 20: long P1 line about to win, P2 must respond
    {"name": "p2_must_respond", "stones": [[[0, 0], 1], [[1, 0], 1], [[2, 0], 1], [[3, 0], 1], [[4, 0], 1], [[0, 2], -1], [[1, 2], -1], [[2, 2], -1]], "current_player": -1, "moves_remaining": 2},
]

assert len(POSITIONS) == 20
