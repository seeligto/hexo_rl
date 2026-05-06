"""In-process self-play worker pool.

Phase 3.5 migration removed Python multiprocessing request/response queues.
Concurrency is now managed by Rust-owned worker threads via SelfPlayRunner.

§162: per-game telemetry extracted to instrumentation.py (PoolInstrumentation).
This module handles orchestration: lifecycle, result drain, buffer push, and
delegations to PoolInstrumentation for all per-game telemetry state.
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

from hexo_rl.model.network import HexTacToeNet, WIRE_CHANNELS
from hexo_rl.utils.constants import BUFFER_CHANNELS
from hexo_rl.monitoring.events import emit_event
from hexo_rl.monitoring.game_recorder import GameRecorder
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.selfplay.instrumentation import PoolInstrumentation
from engine import ReplayBuffer

# Phase B' Class-4 stride-5 detector (§152).  Targets the q-axis stride-5
# spam pattern (mixed-color stones at distance-5 spacing along a single hex
# row).  Stride 5 is the inclusive boundary of LEGAL_MOVE_RADIUS = 5 (§146)
# and CLUSTER_THRESHOLD = 5 (§151 δ.c); existing macro detectors
# (colony_extension_fraction at hex_dist > 6, axis_distribution at
# distance-1 adjacency) miss it by construction.
_STRIDE5_STEP = 5


def _compute_stride5_metrics(
    move_history: list[tuple[int, int]],
) -> tuple[int, int]:
    """Phase B' Class-4 detector — return (stride5_run_max, row_max_density).

    Scans all hex rows in all three axes (matching ``_HEX_AXES``):

      axis_q (E-W,    dq=+1, dr= 0): row keyed by r,        position = q
      axis_r (NW-SE,  dq= 0, dr=+1): row keyed by q,        position = r
      axis_s (NE-SW,  dq=+1, dr=-1): row keyed by s=-q-r,   position = q

    ``stride5_run_max`` is the longest chain of stones lying on a single hex
    row whose along-row coordinates are consecutive at step 5 (e.g. q ∈
    {3, 8, 13, 18} on the same r-row is a chain of length 4).  Color-blind:
    we measure stone-on-row geometry, not same-color sub-runs.

    ``row_max_density`` is the maximum stone count on any single hex row in
    any of the three axes (densest row in the densest direction).

    Per-game cost: O(|stones|).  Budget << 1 ms at typical game length.
    """
    if not move_history:
        return (0, 0)

    bucket_r: dict[int, set[int]] = {}
    bucket_q: dict[int, set[int]] = {}
    bucket_s: dict[int, set[int]] = {}
    for q, r in move_history:
        s = -q - r
        bucket_r.setdefault(r, set()).add(q)
        bucket_q.setdefault(q, set()).add(r)
        bucket_s.setdefault(s, set()).add(q)

    row_max = 0
    stride5_max = 0
    step = _STRIDE5_STEP
    for buckets in (bucket_r, bucket_q, bucket_s):
        for posset in buckets.values():
            n = len(posset)
            if n > row_max:
                row_max = n
            if n < 2:
                continue
            for p in posset:
                if (p - step) in posset:
                    continue  # not start of chain
                length = 1
                cur = p
                while (cur + step) in posset:
                    length += 1
                    cur += step
                if length > stride5_max:
                    stride5_max = length
    return (stride5_max, row_max)


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
        # Rust inference batcher uses WIRE_CHANNELS (18) planes; game runner slices
        # to BUFFER_CHANNELS (8) before pushing to results queue (HEXB v6).

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
            feature_len=WIRE_CHANNELS * board_size * board_size,
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
            random_opening_plies=int(sp.get("random_opening_plies", 0)),
            # §130: per-game rotation port (closes §121 C1). Default true at
            # the WorkerPool layer so the training loop opts in by config;
            # eval/bot paths construct SelfPlayRunner directly without
            # passing this kwarg, picking up the Rust default of false.
            selfplay_rotation_enabled=bool(sp.get("rotation_enabled", True)),
            # Phase B' v8 §152 Q2 — per-game legal-move radius jitter ∈
            # {4, 5, 6}. Default off so eval/bot/test paths and any
            # pre-§152 variant stay at the canonical radius 5.
            legal_move_radius_jitter=bool(sp.get("legal_move_radius_jitter", False)),
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
        self._feat_len = BUFFER_CHANNELS * board_size * board_size  # 8*19*19 = 2888
        self._pol_len = board_size * board_size + 1              # e.g. 19*19+1 = 362
        self._chain_len = 6 * board_size * board_size            # 6*19*19 = 2166

        _mon = config.get("monitoring", config)
        self._log_investigation_metrics = bool(_mon.get(
            "log_investigation_metrics",
            config.get("log_investigation_metrics", True),
        ))

        instr_cfg = config.get("instrumentation", {}) or {}
        self._instrumentation_enabled = bool(instr_cfg.get("enabled", False))
        self._instrumentation = PoolInstrumentation(
            log_investigation_metrics=self._log_investigation_metrics,
        )

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

    @property
    def recent_move_histories(self) -> list[list[tuple[int, int]]]:
        """Snapshot of the last ≤100 self-play game move histories (thread-safe copy)."""
        return self._instrumentation.recent_move_histories(self._lock)

    @property
    def instrumentation_enabled(self) -> bool:
        return self._instrumentation_enabled

    def per_worker_draw_rates(self) -> dict[int, float]:
        """Phase B' Class-1: rolling last-50-game draw rate per worker."""
        return self._instrumentation.per_worker_draw_rates(self._lock)

    def terminal_reason_counts(self) -> dict[str, int]:
        """Phase B' Class-3: cumulative terminal-reason counts since pool start."""
        return self._instrumentation.terminal_reason_counts(self._lock)

    def model_version_summary(self) -> dict[str, float]:
        """Phase B' Class-1: distribution stats over per-game version ranges."""
        return self._instrumentation.model_version_summary(self._lock)

    def buffer_composition(self) -> dict[str, float]:
        """Phase B' Class-3 — composition snapshot of the live replay buffer.

        Reads:
          - ``corpus_fraction``: 1 − self_play_pushed / size (corpus = preload)
          - ``draw_target_fraction``: outcomes ∈ [-0.6, -0.4) over size
          - terminal-reason fractions over cumulative pushes since start

        Falls back gracefully when the engine wheel pre-dates
        ``outcome_in_range_count`` (returns NaN for that field).
        """
        size = max(1, int(self.replay_buffer.size))
        sp_pushed = int(self.self_play_positions_pushed)
        corpus_fraction = max(0.0, 1.0 - (sp_pushed / size))
        try:
            draws_in_buf = int(
                self.replay_buffer.outcome_in_range_count(-0.6, -0.4)
            )
            draw_target_fraction = draws_in_buf / size
        except (AttributeError, TypeError):
            draw_target_fraction = float("nan")
        tr = self.terminal_reason_counts()
        total_games = max(1, sum(tr.values()))
        return {
            "buffer_size": int(self.replay_buffer.size),
            "buffer_capacity": int(self.replay_buffer.capacity),
            "corpus_fraction":      round(corpus_fraction, 6),
            "draw_target_fraction": (
                round(draw_target_fraction, 6)
                if draw_target_fraction == draw_target_fraction
                else float("nan")
            ),
            "six_terminal_fraction":    tr["six_in_a_row"] / total_games,
            "colony_terminal_fraction": tr["colony"]       / total_games,
            "cap_terminal_fraction":    tr["ply_cap"]      / total_games,
            "other_draw_fraction":      tr["other_draw"]   / total_games,
            "n_games_observed": sum(tr.values()),
        }

    def update_checkpoint_step(self, step: int) -> None:
        """Forward the current training step to the game recorder."""
        self._recorder.set_step(step)

    _WINNER_NAMES = ("draw", "x", "o")

    def _stats_loop(self) -> None:
        _in_ch = self._feat_len // (self._board_size * self._board_size)
        _last_buf_emit = time.monotonic()
        while not self._stop_event.is_set():
            # collect_data() returns 8-tuple from Rust — no Python list allocation.
            # feats_np: (N, feat_len) f32, chain_np: (N, chain_len) f32,
            # pols_np: (N, pol_len) f32, vals_np/plies_np: (N,),
            # own_np/wl_np: (N, 361) u8 — per-row aux projected to cluster window.
            # ifs_np: (N,) u8 — 1 = full-search, 0 = quick-search.
            feats_np, chain_np, pols_np, vals_np, plies_np, own_np, wl_np, ifs_np = self._runner.collect_data()
            n = len(vals_np)
            if n > 0:
                # Bulk push: one PyO3 call instead of N per-row pushes (Bucket 5 #2).
                # Vectorised dtype cast + reshape is much cheaper than the per-row
                # _feat_buf[:] = feats_np[i] pattern that preceded this block.
                feats_f16 = feats_np.astype(np.float16).reshape(
                    n, _in_ch, self._board_size, self._board_size,
                )
                chain_f16 = chain_np.astype(np.float16).reshape(
                    n, 6, self._board_size, self._board_size,
                )
                # Per-row compound-move count; clamp into u16 range.
                game_lengths = np.minimum(
                    (plies_np.astype(np.int64) + 1) // 2, 65535,
                ).astype(np.uint16)
                self.replay_buffer.push_many(
                    feats_f16, chain_f16, pols_np, vals_np, own_np, wl_np,
                    game_lengths, ifs_np,
                )

                # Recent buffer still requires per-row push (Python Lock semantics).
                # Scope of item #2 is Rust ReplayBuffer only; recent_buffer bulk
                # push is a separate lever (not on supply critical path).
                if self.recent_buffer is not None:
                    for i in range(n):
                        self.recent_buffer.push(
                            feats_f16[i],
                            chain_planes=chain_f16[i],
                            policy=pols_np[i],
                            outcome=float(vals_np[i]),
                            ownership=own_np[i],
                            winning_line=wl_np[i],
                            is_full_search=bool(ifs_np[i]),
                        )

                with self._lock:
                    self.positions_pushed += n
                    self.self_play_positions_pushed += n

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

            for entry in games_batch:
                # Phase B' instrumentation: drain returns 8-tuples
                # (plies, winner_code, move_history, worker_id, terminal_reason,
                #  mv_min, mv_max, mv_distinct).
                (plies, winner_code, move_history, worker_id,
                 terminal_reason, mv_min, mv_max, mv_distinct) = entry
                winner = self._WINNER_NAMES[winner_code] if winner_code < 3 else "unknown"
                game_length = (plies + 1) // 2  # compound moves
                self._game_lengths.append(game_length)
                self._avg_game_length = sum(self._game_lengths) / len(self._game_lengths)
                # stride-5 per-game detection (pure function; pool computes,
                # instrumentation owns the rolling window and P90).
                if move_history:
                    _stride5_run, _ = _compute_stride5_metrics(move_history)
                else:
                    _stride5_run = 0

                _ext_count, _ext_total, _ext_frac, _stride5_p90 = (
                    self._instrumentation.on_game_complete(
                        self._lock, winner_code, move_history, worker_id,
                        terminal_reason, mv_min, mv_max, mv_distinct, _stride5_run,
                    )
                )

                # Map winner_code to spec: 0=P0, 1=P1, -1=draw
                winner_int = {0: -1, 1: 0, 2: 1}.get(winner_code, -1)

                # Format moves as axial coordinate strings
                moves_list = [f"({q},{r})" for q, r in move_history] if move_history else []

                # Phase B': map the Rust terminal_reason u8 to the dashboard
                # string convention used by reports/phase_b/.
                _TR_NAMES = {0: "six_in_a_row", 1: "colony", 2: "ply_cap", 3: "other_draw"}
                terminal_reason_name = _TR_NAMES.get(int(terminal_reason), "unknown")
                game_complete_payload: dict[str, Any] = {
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
                    # I1 — colony-extension: count/total/fraction of stones
                    # placed at hex-distance > 6 from any opponent stone.
                    "colony_extension_stone_count": _ext_count,
                    "colony_extension_stone_total": _ext_total,
                    "colony_extension_fraction":    _ext_frac,
                    # Phase B' Class-1/3 instrumentation. Always emitted (cheap),
                    # so dashboards/post-hoc analysis can pick them up without
                    # re-running with the flag set.
                    "terminal_reason":          terminal_reason_name,
                    "model_version_min":        int(mv_min),
                    "model_version_max":        int(mv_max),
                    "model_version_distinct":   int(mv_distinct),
                    "model_version_range_size": int(mv_max - mv_min),
                    # Phase B' — stride-5 P90 retained as passive metric (§162).
                    "stride5_run_p90":   int(_stride5_p90),
                }
                emit_event(game_complete_payload)

                log.info(
                    "game_complete",
                    plies=plies,
                    winner=winner,
                    game_length=game_length,
                    sims_per_sec=self._sims_per_sec,
                    colony_extension_stone_count=_ext_count,
                    colony_extension_stone_total=_ext_total,
                    colony_extension_fraction=_ext_frac,
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
