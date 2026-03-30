"""In-process self-play worker pool.

Phase 3.5 migration removed Python multiprocessing request/response queues.
Concurrency is now managed by Rust-owned worker threads via RustSelfPlayRunner.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from native_core import RustSelfPlayRunner  # type: ignore[attr-defined]

from python.model.network import HexTacToeNet
from python.selfplay.inference_server import InferenceServer
from python.training.replay_buffer import ReplayBuffer


class WorkerPool:
    """Runs concurrent self-play games on background threads."""

    def __init__(
        self,
        model: HexTacToeNet,
        config: Dict[str, Any],
        device: torch.device,
        replay_buffer: ReplayBuffer,
        n_workers: Optional[int] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.replay_buffer = replay_buffer

        sp = config.get("selfplay", config)
        self.n_workers = int(n_workers if n_workers is not None else sp.get("n_workers", 1))
        board_size = int(getattr(model, "board_size", 19))
        in_channels = int(config.get("in_channels", config.get("model", {}).get("in_channels", 18)))

        mcts_cfg = config.get("mcts", config)
        self.n_simulations = int(mcts_cfg.get("n_simulations", config.get("n_simulations", 50)))
        self.c_puct = float(mcts_cfg.get("c_puct", 1.5))
        leaf_batch_size = int(sp.get("leaf_batch_size", 8))

        self._runner = RustSelfPlayRunner(
            n_workers=self.n_workers,
            max_moves_per_game=int(sp.get("max_moves_per_game", 128)),
            n_simulations=self.n_simulations,
            leaf_batch_size=leaf_batch_size,
            c_puct=self.c_puct,
            feature_len=in_channels * board_size * board_size,
            policy_len=board_size * board_size + 1,
        )
        self._inference_server = InferenceServer(model, device, config, batcher=self._runner.batcher)

        self._stop_event = threading.Event()
        self._stats_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self.games_completed = 0
        self.positions_pushed = 0
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0

    @property
    def x_winrate(self) -> float:
        with self._lock:
            total = self.games_completed
            return (self.x_wins / total) if total > 0 else 0.0

    @property
    def o_winrate(self) -> float:
        with self._lock:
            total = self.games_completed
            return (self.o_wins / total) if total > 0 else 0.0

    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        with self._lock:
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def _stats_loop(self) -> None:
        while not self._stop_event.is_set():
            # Collect real data from Rust
            data = self._runner.collect_data()
            for feat, pol, outcome in data:
                feat_np = np.array(feat, dtype=np.float32).reshape(18, 19, 19)
                pol_np = np.array(pol, dtype=np.float32)
                self.replay_buffer.push(feat_np, pol_np, float(outcome))
                with self._lock:
                    self.positions_pushed += 1

            with self._lock:
                self.games_completed = int(self._runner.games_completed)
            time.sleep(0.1)

    def start(self) -> None:
        if self._runner.is_running():
            return

        self._stop_event.clear()
        self.model.eval()
        self._inference_server.start()
        self._runner.start()

        self._stats_thread = threading.Thread(
            target=self._stats_loop,
            daemon=True,
            name="selfplay-stats",
        )
        self._stats_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._runner.stop()
        self._inference_server.stop()
        self._inference_server.join(timeout=5.0)

        if self._stats_thread is not None:
            self._stats_thread.join(timeout=5.0)
            self._stats_thread = None
