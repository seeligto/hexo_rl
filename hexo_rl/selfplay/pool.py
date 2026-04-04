"""In-process self-play worker pool.

Phase 3.5 migration removed Python multiprocessing request/response queues.
Concurrency is now managed by Rust-owned worker threads via SelfPlayRunner.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import structlog
import torch
from engine import SelfPlayRunner  # type: ignore[attr-defined]

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.events import emit_event
from hexo_rl.monitoring.game_recorder import GameRecorder
from hexo_rl.selfplay.inference_server import InferenceServer
from engine import ReplayBuffer

log = structlog.get_logger()


class WorkerPool:
    """Runs concurrent self-play games on background threads."""

    def __init__(
        self,
        model: HexTacToeNet,
        config: Dict[str, Any],
        device: torch.device,
        replay_buffer: "ReplayBuffer",
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
        self.fpu_reduction = float(mcts_cfg.get("fpu_reduction", 0.25))
        self.quiescence_enabled = bool(mcts_cfg.get("quiescence_enabled", True))
        self.quiescence_blend_2 = float(mcts_cfg.get("quiescence_blend_2", 0.3))
        leaf_batch_size = int(sp.get("leaf_batch_size", 8))

        pc = sp.get("playout_cap", config.get("playout_cap", {}))

        training_cfg = config.get("training", config)
        self._runner = SelfPlayRunner(
            n_workers=self.n_workers,
            max_moves_per_game=int(sp.get("max_game_moves", sp.get("max_moves_per_game", 128))),
            n_simulations=self.n_simulations,
            leaf_batch_size=leaf_batch_size,
            c_puct=self.c_puct,
            fpu_reduction=self.fpu_reduction,
            feature_len=in_channels * board_size * board_size,
            policy_len=board_size * board_size + 1,
            fast_prob=float(pc.get("fast_prob", 0.0)),
            fast_sims=int(pc.get("fast_sims", 50)),
            standard_sims=int(pc.get("standard_sims", 0)),
            temp_threshold_compound_moves=int(pc.get("temperature_threshold_compound_moves", 15)),
            draw_reward=float(training_cfg.get("draw_reward", -0.1)),
            quiescence_enabled=self.quiescence_enabled,
            quiescence_blend_2=self.quiescence_blend_2,
            temp_min=float(pc.get("temp_min", 0.05)),
            zoi_enabled=bool(pc.get("zoi_enabled", False)),
            zoi_lookback=int(pc.get("zoi_lookback", 16)),
            zoi_margin=int(pc.get("zoi_margin", 5)),
        )
        self._inference_server = InferenceServer(model, device, config, batcher=self._runner.batcher)

        self._stop_event = threading.Event()
        self._stats_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self.games_completed = 0
        self.positions_pushed = 0
        self.self_play_positions_pushed = 0
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0
        self._sims_per_sec: float = 0.0
        self._last_drain_time: float = time.monotonic()
        self._total_sims: int = 0
        self._game_lengths: deque[int] = deque(maxlen=200)
        self._avg_game_length: float = 0.0

        gr_cfg = config.get("game_replay", {})
        self._recorder = GameRecorder(
            output_dir=gr_cfg.get("output_dir", "logs/replays"),
            sample_rate=int(gr_cfg.get("sample_rate", 50)),
            enabled=bool(gr_cfg.get("enabled", True)),
        )

        # Optional recent buffer for recency-weighted sampling.
        # Set by the training loop after construction; None = disabled.
        self.recent_buffer: Optional[Any] = None

        # Ring buffers of recent game auxiliary targets for ownership/threat heads.
        # Each entry: (19, 19) float32 ndarray projected to the final game window.
        self._ownership_ring: deque[np.ndarray] = deque(maxlen=200)
        self._threat_ring: deque[np.ndarray] = deque(maxlen=200)
        self._board_size = board_size

    @property
    def batch_fill_pct(self) -> float:
        srv = self._inference_server
        fwd  = getattr(srv, "_forward_count", 0)
        reqs = getattr(srv, "_total_requests", 0)
        bs   = getattr(srv, "_batch_size", 1)
        if fwd == 0:
            return 0.0
        return min((reqs / (fwd * max(bs, 1))) * 100.0, 100.0)

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

    @property
    def sims_per_sec(self) -> float:
        return self._sims_per_sec

    @property
    def avg_game_length(self) -> float:
        return self._avg_game_length

    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        with self._lock:
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def update_checkpoint_step(self, step: int) -> None:
        """Forward the current training step to the game recorder."""
        self._recorder.set_step(step)

    _WINNER_NAMES = ("draw", "x", "o")

    def _stats_loop(self) -> None:
        while not self._stop_event.is_set():
            data = self._runner.collect_data()
            for feat, pol, outcome, plies in data:
                feat_np = np.array(feat, dtype=np.float16).reshape(18, 19, 19)
                pol_np = np.array(pol, dtype=np.float32)
                game_length = (plies + 1) // 2  # compound moves
                self.replay_buffer.push(feat_np, pol_np, float(outcome), game_length=game_length)
                if self.recent_buffer is not None:
                    self.recent_buffer.push(feat_np, pol_np, float(outcome))
                with self._lock:
                    self.positions_pushed += 1
                    self.self_play_positions_pushed += 1

            with self._lock:
                self.games_completed = int(self._runner.games_completed)
                self.x_wins = int(self._runner.x_wins)
                self.o_wins = int(self._runner.o_wins)
                self.draws = int(self._runner.draws)

            games_batch = self._runner.drain_game_results()

            # Compute sims/sec from elapsed time and known n_simulations per game.
            now = time.monotonic()
            elapsed = now - self._last_drain_time
            self._last_drain_time = now
            if games_batch:
                sims = self.n_simulations * len(games_batch)
                self._total_sims += sims
                if elapsed > 0:
                    self._sims_per_sec = sims / elapsed

            for plies, winner_code, move_history, worker_id, ownership_flat, winning_line_flat in games_batch:
                # Accumulate ownership/threat targets for auxiliary head training.
                if ownership_flat:
                    self._ownership_ring.append(
                        np.array(ownership_flat, dtype=np.float32).reshape(self._board_size, self._board_size)
                    )
                if winning_line_flat:
                    self._threat_ring.append(
                        np.array(winning_line_flat, dtype=np.float32).reshape(self._board_size, self._board_size)
                    )
                winner = self._WINNER_NAMES[winner_code] if winner_code < 3 else "unknown"
                game_length = (plies + 1) // 2  # compound moves
                self._game_lengths.append(game_length)
                self._avg_game_length = sum(self._game_lengths) / len(self._game_lengths)

                # Map winner_code to spec: 0=P0, 1=P1, -1=draw
                winner_int = {0: -1, 1: 0, 2: 1}.get(winner_code, -1)

                # Format moves as axial coordinate strings
                moves_list = [f"({q},{r})" for q, r in move_history] if move_history else []

                emit_event({
                    "event": "game_complete",
                    "game_id": uuid.uuid4().hex,
                    "winner": winner_int,
                    "moves": plies,
                    "moves_list": moves_list,
                    "worker_id": worker_id,
                    # Per-move MCTS detail: None until Rust game_runner stores
                    # top_visits/root_value per move in drain_game_results().
                    "moves_detail": None,
                    "value_trace": None,
                })

                log.info(
                    "game_complete",
                    plies=plies,
                    winner=winner,
                    game_length=game_length,
                    sims_per_sec=self._sims_per_sec,
                )
                self._recorder.maybe_record(
                    moves=move_history,
                    winner_code=winner_code,
                    game_length=plies,
                )

            time.sleep(0.1)

    def get_aux_targets(
        self, batch_size: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample random ownership and threat targets from recently completed games.

        Returns (ownership_targets, threat_targets) as (B, H, W) float32 arrays,
        or (None, None) if not enough data has accumulated yet.
        """
        if len(self._ownership_ring) < 1 or len(self._threat_ring) < 1:
            return None, None
        own_list = list(self._ownership_ring)
        thr_list = list(self._threat_ring)
        own_idx = np.random.randint(0, len(own_list), size=batch_size)
        thr_idx = np.random.randint(0, len(thr_list), size=batch_size)
        own = np.stack([own_list[i] for i in own_idx], axis=0)   # (B, H, W)
        thr = np.stack([thr_list[i] for i in thr_idx], axis=0)   # (B, H, W)
        return own, thr

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

        log.info(
            "worker_pool_started",
            n_workers=self.n_workers,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._runner.stop()
        self._inference_server.stop()
        self._inference_server.join(timeout=5.0)

        if self._stats_thread is not None:
            self._stats_thread.join(timeout=5.0)
            self._stats_thread = None

        self._recorder.stop()
