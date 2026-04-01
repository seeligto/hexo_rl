"""In-process self-play worker pool with mixed opponent scheduling.

Phase 3.5 migration removed Python multiprocessing request/response queues.
Concurrency is now managed by Rust-owned worker threads via RustSelfPlayRunner.

Phase 4.0 adds configurable opponent mixing: a fraction of workers play
current model vs SealBot instead of model vs model, producing longer and
more tactical games that stabilise training.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
from native_core import RustSelfPlayRunner  # type: ignore[attr-defined]

from python.model.network import HexTacToeNet
from python.selfplay.inference_server import InferenceServer
from python.selfplay.worker import SelfPlayWorker
from python.selfplay.policy_projection import project_global_policy_to_local
from python.selfplay.tensor_buffer import TensorBuffer
from python.selfplay.utils import BOARD_SIZE, N_ACTIONS, get_temperature
from python.env.game_state import GameState
from native_core import Board, RustReplayBuffer

log = structlog.get_logger()


def _compute_worker_split(
    n_workers: int,
    sealbot_fraction: float,
) -> Tuple[int, int]:
    """Return (n_selfplay, n_sealbot) with integer rounding."""
    n_sealbot = int(round(n_workers * sealbot_fraction))
    n_sealbot = max(0, min(n_sealbot, n_workers))
    n_selfplay = n_workers - n_sealbot
    return n_selfplay, n_sealbot


class WorkerPool:
    """Runs concurrent self-play games on background threads."""

    def __init__(
        self,
        model: HexTacToeNet,
        config: Dict[str, Any],
        device: torch.device,
        replay_buffer: "RustReplayBuffer",
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

        pc = sp.get("playout_cap", config.get("playout_cap", {}))

        # Read opponent mix config (with reload support)
        self._config_lock = threading.Lock()
        self._opponent_mix = self._read_opponent_mix(config)

        # Compute worker split
        n_selfplay, n_sealbot = _compute_worker_split(
            self.n_workers, self._opponent_mix["sealbot_fraction"],
        )
        self._n_selfplay_workers = n_selfplay
        self._n_sealbot_workers = n_sealbot

        # Only create the Rust runner for self-play workers (may be 0 if all SealBot)
        self._runner: Optional[RustSelfPlayRunner] = None
        self._inference_server: Optional[InferenceServer] = None
        if n_selfplay > 0:
            self._runner = RustSelfPlayRunner(
                n_workers=n_selfplay,
                max_moves_per_game=int(sp.get("max_moves_per_game", 128)),
                n_simulations=self.n_simulations,
                leaf_batch_size=leaf_batch_size,
                c_puct=self.c_puct,
                feature_len=in_channels * board_size * board_size,
                policy_len=board_size * board_size + 1,
                fast_prob=float(pc.get("fast_prob", 0.0)),
                fast_sims=int(pc.get("fast_sims", 50)),
                standard_sims=int(pc.get("standard_sims", 0)),
                temp_threshold_compound_moves=int(pc.get("temperature_threshold_compound_moves", 15)),
            )
            self._inference_server = InferenceServer(model, device, config, batcher=self._runner.batcher)

        self._stop_event = threading.Event()
        self._stats_thread: Optional[threading.Thread] = None
        self._sealbot_threads: List[threading.Thread] = []

        self._lock = threading.Lock()
        self.games_completed = 0
        self.positions_pushed = 0
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0

        # SealBot worker stats (tracked separately for logging)
        self._sealbot_games = 0
        self._selfplay_games = 0

    @staticmethod
    def _read_opponent_mix(config: Dict[str, Any]) -> Dict[str, Any]:
        sp = config.get("selfplay", config)
        mix = sp.get("opponent_mix", {})
        return {
            "self_play_fraction": float(mix.get("self_play_fraction", 1.0)),
            "sealbot_fraction": float(mix.get("sealbot_fraction", 0.0)),
            "sealbot_time_limit_ms": float(mix.get("sealbot_time_limit_ms", 500)),
        }

    def reload_config(self, config: Dict[str, Any]) -> None:
        """Hot-reload opponent mix fractions from updated config.

        Only affects SealBot worker time limits immediately. Worker count
        changes take effect on next full restart.
        """
        with self._config_lock:
            self._opponent_mix = self._read_opponent_mix(config)
            self.config = config

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
            if self._runner is not None:
                # Collect real data from Rust
                data = self._runner.collect_data()
                for feat, pol, outcome in data:
                    feat_np = np.array(feat, dtype=np.float16).reshape(18, 19, 19)
                    pol_np = np.array(pol, dtype=np.float32)
                    self.replay_buffer.push(feat_np, pol_np, float(outcome))
                    with self._lock:
                        self.positions_pushed += 1

                with self._lock:
                    rust_games = int(self._runner.games_completed)
                    self._selfplay_games = rust_games
                    self.games_completed = rust_games + self._sealbot_games
            time.sleep(0.1)

    # ── SealBot worker thread ────────────────────────────────────────────────

    def _sealbot_worker_loop(self, worker_id: int) -> None:
        """Play model-vs-SealBot games in a loop on a Python thread."""
        # Lazy import — only when sealbot_fraction > 0
        from python.bootstrap.bots.sealbot_bot import SealBotBot

        worker = SelfPlayWorker(self.model, self.config, self.device)

        while not self._stop_event.is_set():
            # Re-read config each game for hot-reload support
            with self._config_lock:
                mix = self._opponent_mix
            base_ms = mix["sealbot_time_limit_ms"]
            time_limit_ms = base_ms + random.uniform(-200, 200)
            time_limit_ms = max(50, time_limit_ms)  # floor at 50ms
            time_limit_s = time_limit_ms / 1000.0

            bot = SealBotBot(time_limit=time_limit_s)

            # Alternate which side the model plays
            model_is_p1 = random.random() < 0.5

            try:
                n_positions, winner = self._play_game_vs_bot(
                    worker, bot, model_is_p1,
                )
            except Exception:
                log.exception("sealbot_worker_game_error", worker_id=worker_id)
                continue

            with self._lock:
                self._sealbot_games += 1
                self.games_completed = self._selfplay_games + self._sealbot_games
                self.positions_pushed += n_positions
                if winner == 1:
                    self.x_wins += 1
                elif winner == -1:
                    self.o_wins += 1
                else:
                    self.draws += 1

            log.info(
                "game_completed",
                source="sealbot",
                worker_id=worker_id,
                positions=n_positions,
                winner=winner,
                game_length_ply=n_positions,  # approximate
                sealbot_time_limit_ms=round(time_limit_ms),
                model_side="P1" if model_is_p1 else "P2",
            )

    def _play_game_vs_bot(
        self,
        worker: SelfPlayWorker,
        bot: "BotProtocol",
        model_is_p1: bool,
    ) -> Tuple[int, Optional[int]]:
        """Play one game: model (with MCTS) vs bot. Push all positions to buffer.

        Returns (n_positions, winner) where winner is 1, -1, or None.
        """
        rust_board = Board()
        state = GameState.from_board(rust_board)
        worker._buf.reset()

        records: List[Tuple[np.ndarray, np.ndarray, int]] = []
        fast_sims = np.random.randint(15, 26)

        while True:
            if rust_board.check_win() or rust_board.legal_move_count() == 0:
                break

            current_player = state.current_player  # 1 = P1, -1 = P2
            is_model_turn = (current_player == 1) == model_is_p1

            if is_model_turn:
                # Model move via MCTS
                current_n_sims = worker.n_sims
                if np.random.random() < 0.9:
                    current_n_sims = fast_sims

                mcts_policy = worker._run_mcts_with_sims(
                    rust_board, n_sims=current_n_sims, use_dirichlet=True,
                )

                # Record position for training
                full_tensor, centers = worker._buf.assemble(state)
                global_policy = np.array(mcts_policy, dtype=np.float32)
                for k, center in enumerate(centers):
                    state_tensor = full_tensor[k].copy()
                    policy_arr = project_global_policy_to_local(
                        rust_board, center, global_policy, board_size=BOARD_SIZE,
                    )
                    records.append((state_tensor, policy_arr, state.current_player))

                legal = rust_board.legal_moves()
                if not legal:
                    break
                q, r = SelfPlayWorker._sample_action(mcts_policy, legal, rust_board)
            else:
                # Bot move
                q, r = bot.get_move(state, rust_board)

            state = state.apply_move(rust_board, q, r)

        winner = rust_board.winner()

        for state_tensor, policy_arr, player in records:
            if winner is None:
                outcome = 0.0
            else:
                outcome = 1.0 if player == winner else -1.0
            self.replay_buffer.push(state_tensor, policy_arr, outcome)

        return len(records), winner

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._runner is not None and self._runner.is_running():
            return

        self._stop_event.clear()
        self.model.eval()

        # Start Rust self-play workers
        if self._runner is not None:
            self._inference_server.start()
            self._runner.start()

        # Start SealBot Python worker threads
        for i in range(self._n_sealbot_workers):
            t = threading.Thread(
                target=self._sealbot_worker_loop,
                args=(i,),
                daemon=True,
                name=f"sealbot-worker-{i}",
            )
            t.start()
            self._sealbot_threads.append(t)

        # Stats collection thread
        self._stats_thread = threading.Thread(
            target=self._stats_loop,
            daemon=True,
            name="selfplay-stats",
        )
        self._stats_thread.start()

        log.info(
            "worker_pool_started",
            total_workers=self.n_workers,
            selfplay_workers=self._n_selfplay_workers,
            sealbot_workers=self._n_sealbot_workers,
            sealbot_fraction=self._opponent_mix["sealbot_fraction"],
            sealbot_time_limit_ms=self._opponent_mix["sealbot_time_limit_ms"],
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._runner is not None:
            self._runner.stop()
        if self._inference_server is not None:
            self._inference_server.stop()
            self._inference_server.join(timeout=5.0)

        for t in self._sealbot_threads:
            t.join(timeout=5.0)
        self._sealbot_threads.clear()

        if self._stats_thread is not None:
            self._stats_thread.join(timeout=5.0)
            self._stats_thread = None
