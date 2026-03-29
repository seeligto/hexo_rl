"""CommunityAPIBot — wraps any community bot that implements bot-api-v1 over HTTP.

Any bot hosted at a known URL can be plugged into corpus generation or
evaluation with zero extra code. The API contract (bot-api-v1.yaml) is the
ground truth; fetch it with:

    curl -L https://raw.githubusercontent.com/hex-tic-tac-toe/htttx-bot-api/main/definitions/bot-api-v1.yaml

The wrapper sends the current board as a POST request and returns the chosen
move. Move format follows the BKE notation standard.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

from python.bootstrap.bot_protocol import BotProtocol
from python.env import GameState


class CommunityAPIBot(BotProtocol):
    """Wraps a bot-api-v1 HTTP endpoint.

    Args:
        url:     Base URL of the bot endpoint (e.g. "https://explore.htttx.io/bots/mybot").
        name_id: Human-readable name for logging (defaults to the URL).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        name_id: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._name_id = name_id or url
        self._timeout = timeout

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        cq, cr = rust_board.window_center()
        board_arr = state.board

        stones: list[dict[str, Any]] = []
        for wq_idx in range(19):
            for wr_idx in range(19):
                cell = int(board_arr[wq_idx, wr_idx])
                if cell == 0:
                    continue
                q = wq_idx - 9 + cq
                r = wr_idx - 9 + cr
                stones.append({"q": q, "r": r, "player": 1 if cell == 1 else 2})

        payload = {
            "current_player": 1 if state.current_player == 1 else 2,
            "moves_remaining": state.moves_remaining,
            "ply": state.ply,
            "stones": stones,
        }

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._url}/move",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"CommunityAPIBot HTTP error from {self._url}: {exc}") from exc

        q = int(body["q"])
        r = int(body["r"])
        return q, r

    def name(self) -> str:
        return self._name_id
