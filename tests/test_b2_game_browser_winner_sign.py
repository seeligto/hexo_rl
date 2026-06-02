"""B2 / SCATTER-4 display bug — bot-game P2 wins must index as "p2_win".

`_index_bot_games` reads bot_games/*.json whose `winner` is `int(board.winner())`
in {1, -1} (P1 / P2). The outcome mapping tested `winner == 2` — a DEAD branch
(board.winner() never returns 2), so P2 wins fell through to "unknown" in the
game browser. Display-only; route the P2 case to the actual `-1` code.
"""
import json

from hexo_rl.monitoring.game_browser import _index_bot_games, SOURCE_BOT_FAST


def test_bot_game_p2_win_indexed_not_unknown(tmp_path):
    d = tmp_path / "bot_games" / "sealbot_fast"
    d.mkdir(parents=True)
    (d / "g_p2.json").write_text(json.dumps({"winner": -1, "moves": [], "plies": 10}))
    (d / "g_p1.json").write_text(json.dumps({"winner": 1, "moves": [], "plies": 12}))

    outcomes = {e.game_id: e.outcome
                for e in _index_bot_games(tmp_path, "sealbot_fast", SOURCE_BOT_FAST, {})}

    assert outcomes["g_p2"] == "p2_win", "P2 win (winner=-1) must not show as 'unknown'"
    assert outcomes["g_p1"] == "p1_win"
