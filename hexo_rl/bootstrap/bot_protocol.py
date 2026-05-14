"""BotProtocol — abstract interface that every game source must implement.

All bots are interchangeable: corpus generation, evaluation, and self-play
benchmarking accept any BotProtocol and swap them via config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hexo_rl.env import GameState


class ResignationException(Exception):
    """Raised when a bot identifies a forced loss and wants to resign early."""
    def __init__(self, winner: int):
        self.winner = winner
        super().__init__(f"Bot resigns. Winner: {winner}")

class BotProtocol(ABC):
    """Every bot that can play Hex Tac Toe must implement this interface.

    Contract stability (§176 P38): the `get_move` argument set is
    intentionally pinned. `rust_board` is the authoritative game state
    (always present, always current); `state` is a pre-computed
    `GameState` convenience snapshot the caller maintains in lockstep
    with `rust_board`. Bots that need only one of the two are free to
    ignore the other — no underscore-prefix renaming required.
    """

    @abstractmethod
    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        """Return a legal move (q, r) for the current position.

        Args:
            state:      Pre-computed immutable GameState snapshot, kept
                        in sync with `rust_board` by the caller. Provided
                        as a convenience so bots that need GameState
                        fields (current_player, moves_remaining, ply,
                        history, …) do not each re-derive them. Bots
                        that only need raw board geometry may ignore it.
            rust_board: Authoritative live engine.Board. Caller owns it
                        and has NOT yet applied the returned move; bots
                        MUST treat it as read-only (use accessors like
                        legal_moves(), get_stones(), check_win()).

        Returns:
            (q, r) axial coordinates of the chosen move. Must be a legal
            move on `rust_board`; callers will apply it after this call.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable bot identifier (used in logging and Elo tables)."""
        ...

    def reset(self) -> None:
        """Clear any per-game bot state. Called before starting a new game.

        Default is a no-op; wrappers that cache compound-move second stones or
        any other across-call state must override this to null that cache.
        """
        return

    def __str__(self) -> str:
        return self.name()
