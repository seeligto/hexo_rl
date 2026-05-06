"""Signal handler registration for the self-play training loop.

Wraps SIGINT/SIGTERM → cooperative shutdown state. Two presses force-exit;
one press flips ``running=False`` and ``shutdown_save=True`` so the loop
saves a checkpoint before returning.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class ShutdownState:
    running: bool = True
    stop_count: int = 0
    shutdown_save: bool = False


def install_signal_handlers(state: ShutdownState) -> None:
    """Install SIGINT/SIGTERM handlers that flip ``state``.

    Two consecutive signals force-exit; one signal sets ``running=False``
    and ``shutdown_save=True``. The training loop is responsible for
    polling ``state`` between iterations.
    """

    def _stop(sig: int, frame: Any) -> None:
        state.stop_count += 1
        if state.stop_count >= 2:
            sys.exit(1)
        log.info(
            "shutdown_requested",
            msg="finishing current step… press Ctrl+C again to force",
        )
        state.shutdown_save = True
        state.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
