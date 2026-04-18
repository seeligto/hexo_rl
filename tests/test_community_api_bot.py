"""Regression test for C-002: CommunityAPIBot uses rust_board.get_stones()."""
from unittest.mock import MagicMock, patch
import json


def _make_rust_board(stones=None):
    """Minimal rust_board mock supporting get_stones()."""
    rb = MagicMock()
    rb.get_stones.return_value = stones or [(0, 0, 1), (1, 0, 2)]
    return rb


def _make_state():
    state = MagicMock()
    state.current_player = 1
    state.moves_remaining = 2
    state.ply = 3
    # Deliberately do NOT set state.board — confirms we never access it.
    del state.board
    return state


def test_bot_constructs_without_attribute_error():
    """get_move must not access state.board; must call rust_board.get_stones()."""
    from hexo_rl.bootstrap.bots.community_api_bot import CommunityAPIBot

    bot = CommunityAPIBot(url="http://fake-bot.local", name_id="test")
    rust_board = _make_rust_board()
    state = _make_state()

    response_body = json.dumps({"q": 2, "r": 3}).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        move = bot.get_move(state, rust_board)

    assert move == (2, 3)
    rust_board.get_stones.assert_called_once()


def test_stones_payload_uses_get_stones_coords():
    """Stones in HTTP payload must match get_stones() absolute coords, not window math."""
    import urllib.request
    from hexo_rl.bootstrap.bots.community_api_bot import CommunityAPIBot

    bot = CommunityAPIBot(url="http://fake-bot.local")
    rust_board = _make_rust_board(stones=[(5, -3, 1), (-2, 7, 2)])
    state = _make_state()

    captured = {}

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"q": 0, "r": 0}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        bot.get_move(state, rust_board)

    stones = captured["body"]["stones"]
    assert {"q": 5, "r": -3, "player": 1} in stones
    assert {"q": -2, "r": 7, "player": 2} in stones
