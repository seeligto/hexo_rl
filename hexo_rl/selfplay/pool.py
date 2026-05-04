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

from hexo_rl.model.network import HexTacToeNet, WIRE_CHANNELS
from hexo_rl.utils.constants import BUFFER_CHANNELS
from hexo_rl.monitoring.events import emit_event
from hexo_rl.monitoring.game_recorder import GameRecorder
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.utils.coordinates import axial_distance
from engine import ReplayBuffer

# I1 colony-extension detector (§107).  Hex distance threshold above which a
# stone is counted as "colony extension" — disjoint cluster spam behaviour
# flagged as the primary residual mechanism for pre-W1 fast-game draw collapse
# (W1 forensics R1, 2026-04-19).
_COLONY_EXT_HEX_DIST = 6


def _compute_colony_extension(move_history: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (colony_extension_count, classified_total) for a finished game.

    A stone counts as "colony extension" if its minimum hex distance to ANY
    opponent stone at game end is > ``_COLONY_EXT_HEX_DIST``.  Stones of a
    player with no opponent stones on the board (only possible with a one-move
    game) are excluded from both numerator and denominator.

    Ply→player rule: ply 0 is P1; thereafter each player places 2 stones before
    the turn passes (``compound_idx = (ply - 1) // 2``; P2 when even, else P1).
    Per-game cost: O(|stones|^2); budget <1ms at typical game length.
    """
    if not move_history:
        return (0, 0)
    p1: list[tuple[int, int]] = []
    p2: list[tuple[int, int]] = []
    for ply, (q, r) in enumerate(move_history):
        own_is_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
        (p1 if own_is_p1 else p2).append((q, r))
    ext = 0
    total = 0
    for own, opp in ((p1, p2), (p2, p1)):
        if not opp:
            continue
        for s in own:
            total += 1
            if min(axial_distance(s, o) for o in opp) > _COLONY_EXT_HEX_DIST:
                ext += 1
    return (ext, total)

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

        # Rolling window of the last N completed game move histories for
        # axis-distribution monitoring (§axis_dist).  Guarded by _lock.
        self._recent_move_histories: deque[list[tuple[int, int]]] = deque(maxlen=100)

        # ── Phase B' instrumentation (default off) ──────────────────────────
        # When `instrumentation.enabled=true`, the stats loop tracks per-worker
        # draw_rate over the last 50 games, terminal-reason counts, and
        # model-version range stats. Used to discriminate Class 1 (stale
        # dispatch) / Class 2 (value-head feedback) / Class 3 (buffer
        # composition) before touching production knobs.
        instr_cfg = config.get("instrumentation", {}) or {}
        self._instrumentation_enabled = bool(instr_cfg.get("enabled", False))
        # Sample-rate guard: skip per-game joint logs when ratio < 1.0
        # to keep instrumentation overhead under 5% of throughput at
        # n_workers=18. Default 1.0 = log every game.
        self._instr_log_ratio = float(instr_cfg.get("log_ratio", 1.0))
        # Per-worker rolling last-50-game outcomes (1 = draw, 0 = decisive).
        # Default to 4 lanes; auto-grows if worker_id ≥ len().
        self._per_worker_draws: dict[int, deque[int]] = {}
        # Cumulative terminal-reason counts (Phase B' Class-3 buffer
        # composition, reason axis). 0=six 1=colony 2=cap 3=other_draw.
        self._terminal_reason_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        # Per-game model-version range archive — last 200 games. (min, max,
        # distinct, terminal_reason, winner_code).
        self._mv_range_history: deque[tuple[int, int, int, int, int]] = deque(maxlen=200)

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
        with self._lock:
            return list(self._recent_move_histories)

    @property
    def instrumentation_enabled(self) -> bool:
        return self._instrumentation_enabled

    def per_worker_draw_rates(self) -> dict[int, float]:
        """Phase B' Class-1: rolling last-50-game draw rate per worker.

        Returned snapshot is a fresh dict; safe to read without holding the
        pool lock further. Empty workers (no completed games yet) are
        excluded so the dashboard can show 'pending'.
        """
        with self._lock:
            out: dict[int, float] = {}
            for wid, dq in self._per_worker_draws.items():
                if len(dq) > 0:
                    out[wid] = sum(dq) / len(dq)
            return out

    def terminal_reason_counts(self) -> dict[str, int]:
        """Phase B' Class-3: cumulative terminal-reason counts since pool start.

        Keys: ``six_in_a_row``, ``colony``, ``ply_cap``, ``other_draw``.
        """
        with self._lock:
            return {
                "six_in_a_row": self._terminal_reason_counts.get(0, 0),
                "colony":       self._terminal_reason_counts.get(1, 0),
                "ply_cap":      self._terminal_reason_counts.get(2, 0),
                "other_draw":   self._terminal_reason_counts.get(3, 0),
            }

    def model_version_summary(self) -> dict[str, float]:
        """Phase B' Class-1: distribution stats over per-game version ranges.

        Returns median / P90 / max of (mv_max - mv_min) across the last 200
        games, plus the per-game distinct-version median. ``correlation_p``
        is the Spearman ρ between range_size and is_draw on the same window
        (when n ≥ 10), used by the diagnosis report.
        """
        with self._lock:
            hist = list(self._mv_range_history)
        if not hist:
            return {"n": 0}
        ranges = [b - a for (a, b, _, _, _) in hist]
        distincts = [d for (_, _, d, _, _) in hist]
        is_draw  = [1 if wc == 0 else 0 for (_, _, _, _, wc) in hist]
        n = len(ranges)
        ranges_sorted = sorted(ranges)
        distincts_sorted = sorted(distincts)
        median_range = ranges_sorted[n // 2]
        p90_range = ranges_sorted[max(0, int(n * 0.9) - 1)]
        max_range = ranges_sorted[-1]
        median_distinct = distincts_sorted[n // 2]
        rho = None
        if n >= 10:
            try:
                # Spearman ρ via rankdata; fall back to None if scipy missing.
                from scipy.stats import spearmanr
                rho_val, p_val = spearmanr(ranges, is_draw)
                rho = float(rho_val) if rho_val == rho_val else None  # NaN guard
                _ = p_val
            except Exception:
                rho = None
        return {
            "n": n,
            "median_range": int(median_range),
            "p90_range": int(p90_range),
            "max_range": int(max_range),
            "median_distinct": int(median_distinct),
            "spearman_rho_range_vs_draw": rho,
        }

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
                if move_history:
                    with self._lock:
                        self._recent_move_histories.append(list(move_history))

                # Phase B' Class-1/3 telemetry. Always update counters (cheap);
                # event emission is gated on `instrumentation.enabled`.
                with self._lock:
                    self._terminal_reason_counts[int(terminal_reason)] = (
                        self._terminal_reason_counts.get(int(terminal_reason), 0) + 1
                    )
                    is_draw_outcome = 1 if winner_code == 0 else 0
                    dq = self._per_worker_draws.setdefault(int(worker_id), deque(maxlen=50))
                    dq.append(is_draw_outcome)
                    self._mv_range_history.append(
                        (int(mv_min), int(mv_max), int(mv_distinct),
                         int(terminal_reason), int(winner_code))
                    )

                # Map winner_code to spec: 0=P0, 1=P1, -1=draw
                winner_int = {0: -1, 1: 0, 2: 1}.get(winner_code, -1)

                # Format moves as axial coordinate strings
                moves_list = [f"({q},{r})" for q, r in move_history] if move_history else []

                # I1 colony-extension detector (§107) — pure Python, reads
                # move_history already in the drain tuple. Guarded by config so
                # bench runs stay quiet.
                if self._log_investigation_metrics and move_history:
                    _ext_count, _ext_total = _compute_colony_extension(move_history)
                    _ext_frac = float(_ext_count / _ext_total) if _ext_total > 0 else 0.0
                else:
                    _ext_count, _ext_total, _ext_frac = 0, 0, 0.0

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
