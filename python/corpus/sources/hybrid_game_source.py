"""HybridGameSource — human opening + SealBot continuation game generator."""

from __future__ import annotations

import random
from typing import Iterator

import structlog

from native_core import Board
from python.bootstrap.bot_protocol import BotProtocol
from python.env.game_state import GameState
from python.corpus.sources.base import CorpusSource, GameRecord

log = structlog.get_logger()


class HybridGameSource(CorpusSource):
    """Generates hybrid games: human opening replayed to a turn boundary, then
    a bot plays both sides to completion.

    For each seed game from *seed_source*, ``games_per_seed`` continuations are
    generated. Continuations differ by:

    1. **Handoff player alternation** (avoids systematic P2-only bias):
       - i=0, i=2 (even): natural snap → bot enters as P2
       - i=1 (odd): extend one extra P2 turn → bot enters as P1
    2. **Entropy injection** (Q3): one random legal move applied at the handoff
       position before the bot takes over, seeded by ``rng_seed + i``.

    Args:
        seed_source:      Source of human game seeds (typically HumanGameSource).
        bot:              BotProtocol implementation used for the continuation.
                          Must be re-entrant across games (a fresh instance is
                          NOT created per game — the bot's internal state is
                          reset by calling it from ``moves_remaining == 2``).
        n_opening_moves:  Target stone-placement threshold before handoff.
                          Default 8. The turn-boundary snap ensures the bot
                          always enters at ``moves_remaining == 2``.
        games_per_seed:   Number of continuations per seed game. Default 3.
        min_bot_plies:    Minimum post-handoff plies (entropy move + bot moves).
                          Games shorter than this are discarded. Default 10.
        rng_seed:         Base RNG seed. Continuation i uses ``rng_seed + i``.
    """

    def __init__(
        self,
        seed_source: CorpusSource,
        bot: BotProtocol,
        n_opening_moves: int = 8,
        games_per_seed: int = 3,
        min_bot_plies: int = 10,
        rng_seed: int = 42,
    ) -> None:
        self._seed_source    = seed_source
        self._bot            = bot
        self._n_opening      = n_opening_moves
        self._games_per_seed = games_per_seed
        self._min_bot_plies  = min_bot_plies
        self._rng_seed       = rng_seed

    def name(self) -> str:
        return "hybrid"

    def __len__(self) -> int | None:
        # Call __len__ directly rather than len() — the base class signals
        # "unknown length" with None, which builtin len() cannot accept.
        seed_len = self._seed_source.__len__()
        if seed_len is None:
            return None
        return seed_len * self._games_per_seed

    def __iter__(self) -> Iterator[GameRecord]:
        for seed_record in self._seed_source:
            yield from self._continuations(seed_record)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _continuations(self, seed: GameRecord) -> Iterator[GameRecord]:
        """Generate up to games_per_seed continuations from one seed game."""
        human_moves = seed.moves

        # --- Step 1: Replay to base handoff (turn-boundary snap, bot=P2) ---
        base_board, base_state, base_moves = self._replay_to_base_handoff(
            human_moves, seed.game_id_str
        )
        if base_board is None:
            return  # seed too short — already logged

        cluster_count = len(base_state.centers)
        colony_bug    = cluster_count > 4

        for i in range(self._games_per_seed):
            # Re-replay from scratch to base handoff for each continuation
            # (avoids mutating a shared board state).
            board, state, opening_moves = self._replay_to_base_handoff(
                human_moves, seed.game_id_str
            )

            # --- Step 2: Extend one extra P2 turn for odd continuations ---
            if i % 2 == 1:
                board, state, opening_moves = self._extend_one_p2_turn(
                    board, state, opening_moves, human_moves
                )
            bot_plays_as = "P2" if i % 2 == 0 else "P1"
            handoff_ply  = board.ply

            # --- Step 3: Entropy injection (Q3) ---
            rng = random.Random(self._rng_seed + i)
            legal = board.legal_moves()
            if not legal:
                log.warning("hybrid_no_legal_at_handoff",
                            seed_id=seed.game_id_str, continuation=i)
                continue
            eq, er = rng.choice(legal)
            state = state.apply_move(board, eq, er)
            entropy_move = (eq, er)

            # --- Step 4: Bot continuation (both sides) ---
            bot_moves: list[tuple[int, int]] = []
            try:
                while not board.check_win() and board.legal_move_count() > 0:
                    q, r = self._bot.get_move(state, board)
                    state = state.apply_move(board, q, r)
                    bot_moves.append((q, r))
            except Exception as exc:
                log.warning("hybrid_bot_error",
                            seed_id=seed.game_id_str, continuation=i, error=str(exc))
                continue

            # --- Step 5: Minimum bot plies filter (Q5) ---
            post_handoff_plies = 1 + len(bot_moves)  # entropy move counts
            if post_handoff_plies < self._min_bot_plies:
                log.warning("hybrid_game_too_short",
                            seed_id=seed.game_id_str, continuation=i,
                            post_handoff_plies=post_handoff_plies,
                            min_required=self._min_bot_plies)
                continue

            winner = board.winner()
            if winner is None:
                log.warning("hybrid_game_no_winner",
                            seed_id=seed.game_id_str, continuation=i)
                continue

            full_moves = opening_moves + [entropy_move] + bot_moves

            yield GameRecord(
                game_id_str=f"{seed.game_id_str}:c{i}",
                moves=full_moves,
                winner=winner,
                source="hybrid",
                metadata={
                    "seed_game_id":            seed.game_id_str,
                    "continuation_idx":        i,
                    "human_winner":            seed.winner,
                    "handoff_ply":             handoff_ply,
                    "bot_plays_as":            bot_plays_as,
                    "cluster_count_at_handoff": cluster_count,
                    "colony_bug_at_handoff":   colony_bug,
                },
            )

    def _replay_to_base_handoff(
        self,
        human_moves: list[tuple[int, int]],
        game_id: str,
    ) -> tuple[Board | None, GameState | None, list[tuple[int, int]] | None]:
        """Replay human moves until we reach the turn-boundary snap point.

        Returns (board, state, moves_replayed) at the base handoff, or
        (None, None, None) if the game is too short.

        The base handoff is defined as:
        - After replaying >= n_opening_moves plies, advance one extra ply if
          board.moves_remaining < 2 (snap to turn boundary so bot enters fresh).
        - Bot enters as P2 (moves_remaining == 2, P2's turn) with N=8.
        """
        board = Board()
        state = GameState.from_board(board)
        played: list[tuple[int, int]] = []

        for idx, (q, r) in enumerate(human_moves):
            try:
                state = state.apply_move(board, q, r)
            except Exception as exc:
                log.warning("hybrid_replay_error",
                            seed_id=game_id, ply=idx, error=str(exc))
                return None, None, None
            played.append((q, r))

            if board.ply >= self._n_opening:
                if board.moves_remaining == 2:
                    # Clean turn boundary — hand off here.
                    return board, state, played
                # Mid-turn: need one more ply to complete the turn.
                # That ply is the next iteration.

        # Fell off the end of the human game without finding a clean boundary.
        log.warning("hybrid_seed_too_short",
                    seed_id=game_id,
                    human_plies=len(human_moves),
                    n_opening_moves=self._n_opening)
        return None, None, None

    def _extend_one_p2_turn(
        self,
        board: Board,
        state: GameState,
        opening_moves: list[tuple[int, int]],
        human_moves: list[tuple[int, int]],
    ) -> tuple[Board, GameState, list[tuple[int, int]]]:
        """Replay one additional full P2 turn (2 plies) past the base handoff.

        If the human game is exhausted, apply random legal moves instead.
        Returns the updated (board, state, moves_replayed).
        """
        moves_left = list(human_moves[len(opening_moves):])  # remaining human moves
        extra_plies = 2  # one full P2 turn = 2 stone placements
        rng_fallback = random.Random(self._rng_seed)

        for _ in range(extra_plies):
            if board.check_win() or board.legal_move_count() == 0:
                break
            if moves_left:
                q, r = moves_left.pop(0)
                try:
                    state = state.apply_move(board, q, r)
                    opening_moves = opening_moves + [(q, r)]
                    continue
                except Exception:
                    pass  # fall through to random
            # Human game exhausted or move illegal — use random.
            legal = board.legal_moves()
            if not legal:
                break
            q, r = rng_fallback.choice(legal)
            state = state.apply_move(board, q, r)
            opening_moves = opening_moves + [(q, r)]

        return board, state, opening_moves
