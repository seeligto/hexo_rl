"""Non-blocking game replay recorder for self-play training games."""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GameRecorder:
    """Record 1-in-N self-play games to daily-rotated jsonl files.

    Thread-safe. Writes happen on a background daemon thread so the
    stats loop is never blocked by I/O.

    Output format — one JSON object per line:
        moves:            [[q, r], ...] in play order
        outcome:          "x_win" | "o_win" | "draw"
        game_length:      int (number of compound moves / plies)
        timestamp:        ISO 8601 UTC
        checkpoint_step:  int (training step at which the weights were last updated)
    """

    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 50,
        enabled: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._sample_rate = sample_rate
        self._enabled = enabled and sample_rate > 0
        self._counter = 0
        self._checkpoint_step: int = 0
        # Queue items are (filepath, json_line) tuples, or None as a stop sentinel.
        self._queue: "queue.SimpleQueue[Optional[Tuple[str, str]]]" = queue.SimpleQueue()
        self._thread: Optional[threading.Thread] = None
        if self._enabled:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="game-recorder"
            )
            self._thread.start()

    def set_step(self, step: int) -> None:
        """Update the current training checkpoint step (called by the trainer)."""
        self._checkpoint_step = step

    def maybe_record(
        self,
        moves: List[Tuple[int, int]],
        winner_code: int,
        game_length: int,
    ) -> None:
        """Call after every game. Writes a record if this game hits the sample counter.

        Args:
            moves:        Sequence of (q, r) stone placements in play order.
            winner_code:  0 = draw, 1 = x_win, 2 = o_win (matches Rust encoding).
            game_length:  Number of plies (stone placements).
        """
        if not self._enabled:
            return
        self._counter += 1
        if self._counter % self._sample_rate != 0:
            return

        _outcome_map: Dict[int, str] = {0: "draw", 1: "x_win", 2: "o_win"}
        record: Dict[str, Any] = {
            "moves": [[q, r] for q, r in moves],
            "outcome": _outcome_map.get(winner_code, "unknown"),
            "game_length": game_length,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_step": self._checkpoint_step,
        }
        # Resolve the target file path now (caller's thread) so daily rotation is
        # determined at record time, not at write time — keeps the background thread simple.
        self._queue.put((str(self._current_path()), json.dumps(record)))

    def stop(self) -> None:
        """Flush pending writes and stop the background thread."""
        if self._thread is not None:
            self._queue.put(None)  # type: ignore[arg-type]  # sentinel
            self._thread.join(timeout=5.0)
            self._thread = None

    # ── private ──────────────────────────────────────────────────────────

    def _current_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._output_dir / f"games_{date_str}.jsonl"

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            path_str, line = item
            try:
                with open(path_str, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass  # never let I/O errors crash the background thread
