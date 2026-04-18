"""In-process self-play worker pool.

Phase 3.5 migration removed Python multiprocessing request/response queues.
Concurrency is now managed by Rust-owned worker threads via SelfPlayRunner.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, Optional

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
        if "fast_sims" not in pc:
            raise ValueError(
                "playout_cap.fast_sims must be set in selfplay.yaml — no silent defaults"
            )

        # Move-level and game-level playout caps are mutually exclusive: move-level
        # (full_search_prob) overrides the game-level (fast_prob/fast_sims) sim selection
        # inside the worker loop, so running both at once silently ignores the latter.
        fast_prob_cfg = float(pc.get("fast_prob", 0.0))
        full_search_prob_cfg = float(pc.get("full_search_prob", 0.0))
        n_sims_quick_cfg = int(pc.get("n_sims_quick", 0))
        n_sims_full_cfg = int(pc.get("n_sims_full", 0))
        if full_search_prob_cfg > 0.0 and fast_prob_cfg > 0.0:
            raise ValueError(
                "playout_cap: fast_prob and full_search_prob are mutually exclusive — "
                "move-level cap (full_search_prob) overrides game-level cap (fast_prob). "
                f"Got fast_prob={fast_prob_cfg}, full_search_prob={full_search_prob_cfg}. "
                "Set one of them to 0 in selfplay.yaml."
            )
        if full_search_prob_cfg > 0.0 and (n_sims_quick_cfg <= 0 or n_sims_full_cfg <= 0):
            raise ValueError(
                "playout_cap: full_search_prob > 0 requires n_sims_quick > 0 AND "
                "n_sims_full > 0. "
                f"Got full_search_prob={full_search_prob_cfg}, "
                f"n_sims_quick={n_sims_quick_cfg}, n_sims_full={n_sims_full_cfg}."
            )

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
            fast_sims=int(pc["fast_sims"]),
            standard_sims=int(pc.get("standard_sims", 0)),
            temp_threshold_compound_moves=int(pc.get("temperature_threshold_compound_moves", 15)),
            draw_reward=float(training_cfg.get("draw_value", -0.5)),
            quiescence_enabled=self.quiescence_enabled,
            quiescence_blend_2=self.quiescence_blend_2,
            temp_min=float(pc.get("temp_min", 0.05)),
            zoi_enabled=bool(pc.get("zoi_enabled", False)),
            zoi_lookback=int(pc.get("zoi_lookback", 16)),
            zoi_margin=int(pc.get("zoi_margin", 5)),
            completed_q_values=bool(sp.get("completed_q_values", False)),
            c_visit=float(sp.get("c_visit", 50.0)),
            c_scale=float(sp.get("c_scale", 1.0)),
            gumbel_mcts=bool(sp.get("gumbel_mcts", False)),
            gumbel_m=int(sp.get("gumbel_m", 16)),
            gumbel_explore_moves=int(sp.get("gumbel_explore_moves", 10)),
            dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(mcts_cfg.get("epsilon", 0.25)),
            dirichlet_enabled=bool(mcts_cfg.get("dirichlet_enabled", True)),
            results_queue_cap=int(sp.get("results_queue_cap", 10_000)),
            full_search_prob=full_search_prob_cfg,
            n_sims_quick=n_sims_quick_cfg,
            n_sims_full=n_sims_full_cfg,
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

        self._board_size = board_size
        self._feat_len = in_channels * board_size * board_size  # 18*19*19 = 6498 for 18-plane layout
        self._pol_len = board_size * board_size + 1              # e.g. 19*19+1 = 362
        self._chain_len = 6 * board_size * board_size            # 6*19*19 = 2166

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

    def update_checkpoint_step(self, step: int) -> None:
        """Forward the current training step to the game recorder."""
        self._recorder.set_step(step)

    _WINNER_NAMES = ("draw", "x", "o")

    def _stats_loop(self) -> None:
        # Pre-allocate single-position receive buffers once.
        # replay_buffer.push() and recent_buffer.push() copy data into Rust memory,
        # so reusing these buffers across loop iterations is safe.
        _feat_buf  = np.empty(self._feat_len,  dtype=np.float16)
        _chain_buf = np.empty(self._chain_len, dtype=np.float16)
        _pol_buf   = np.empty(self._pol_len,   dtype=np.float32)
        _in_ch = self._feat_len // (self._board_size * self._board_size)
        _feat_2d  = _feat_buf.reshape(_in_ch, self._board_size, self._board_size)
        _chain_3d = _chain_buf.reshape(6, self._board_size, self._board_size)
        _last_buf_emit = time.monotonic()
        while not self._stop_event.is_set():
            # collect_data() returns 8-tuple from Rust — no Python list allocation.
            # feats_np: (N, feat_len) f32, chain_np: (N, chain_len) f32,
            # pols_np: (N, pol_len) f32, vals_np/plies_np: (N,),
            # own_np/wl_np: (N, 361) u8 — per-row aux projected to cluster window.
            # ifs_np: (N,) u8 — 1 = full-search, 0 = quick-search.
            feats_np, chain_np, pols_np, vals_np, plies_np, own_np, wl_np, ifs_np = self._runner.collect_data()
            n = len(vals_np)
            for i in range(n):
                # feats_np[i] is a numpy view (no copy); _feat_buf[:] = view does
                # the f32→f16 dtype conversion at C speed with no Python objects.
                _feat_buf[:]  = feats_np[i]
                _chain_buf[:] = chain_np[i]
                _pol_buf[:]   = pols_np[i]
                feat_np  = _feat_2d    # shaped view into _feat_buf
                chain_np_pos = _chain_3d  # shaped view into _chain_buf
                pol_np   = _pol_buf
                own_row  = own_np[i]
                wl_row   = wl_np[i]
                plies    = int(plies_np[i])
                outcome  = float(vals_np[i])
                game_length = (plies + 1) // 2  # compound moves
                is_full_search = bool(ifs_np[i])
                self.replay_buffer.push(
                    feat_np, chain_np_pos, pol_np, outcome, own_row, wl_row,
                    game_length=game_length,
                    is_full_search=is_full_search,
                )
                if self.recent_buffer is not None:
                    self.recent_buffer.push(
                        feat_np, chain_planes=chain_np_pos, policy=pol_np,
                        outcome=outcome, ownership=own_row, winning_line=wl_row,
                        is_full_search=is_full_search,
                    )
                with self._lock:
                    self.positions_pushed += 1
                    self.self_play_positions_pushed += 1

            with self._lock:
                self.games_completed = int(self._runner.games_completed)
                self.x_wins = int(self._runner.x_wins)
                self.o_wins = int(self._runner.o_wins)
                self.draws = int(self._runner.draws)

            # Local variable — fully consumed each iteration; no unbounded accumulation.
            # drain_game_results now returns metadata-only 4-tuples; spatial aux
            # targets flow per-row via collect_data() above.
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

            for plies, winner_code, move_history, worker_id in games_batch:
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

            # Emit buffer stats at ~5s resolution so dashboard stays fresh
            # between iteration_complete events.
            _now_buf = time.monotonic()
            if _now_buf - _last_buf_emit >= 5.0:
                _last_buf_emit = _now_buf
                emit_event({
                    "event": "system_stats",
                    "buffer_size": self.replay_buffer.size,
                    "buffer_capacity": self.replay_buffer.capacity,
                })

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
