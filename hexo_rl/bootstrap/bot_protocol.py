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
    """Every bot that can play Hex Tac Toe must implement this interface."""

    @abstractmethod
    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        """Return a legal move (q, r) for the current position.

        Args:
            state:      Immutable GameState snapshot.
            rust_board: Live engine.Board — caller owns it and has NOT
                        yet applied the returned move.

        Returns:
            (q, r) axial coordinates of the chosen move.
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
