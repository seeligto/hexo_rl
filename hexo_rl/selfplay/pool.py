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
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import structlog
import torch
from engine import SelfPlayRunner, SelfPlayRunnerConfig  # type: ignore[attr-defined]

from hexo_rl.encoding import EncodingSpec as RegistrySpec
from hexo_rl.encoding import resolve_from_config as registry_resolve_from_config
from hexo_rl.model.network import HexTacToeNet, WIRE_CHANNELS
from hexo_rl.monitoring.events import emit_event
from hexo_rl.monitoring.game_recorder import GameRecorder
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.selfplay.instrumentation import (
    PoolInstrumentation,
    _compute_stride5_metrics,
)
from engine import ReplayBuffer

log = structlog.get_logger()


# §176 P9 — typed snapshot dataclasses replace ad-hoc reaches into the
# private ``_runner`` / ``_inference_server`` attributes. Pure read-only
# snapshots; no behaviour change. Consumers (events.py, step_coordinator,
# tests) call ``pool.runner_stats()`` / ``pool.inference_stats()`` instead.
@dataclass(frozen=True)
class RunnerStats:
    """Snapshot of Rust ``SelfPlayRunner`` counters / scalars."""

    games_completed: int
    positions_generated: int
    x_wins: int
    o_wins: int
    draws: int
    model_version: int
    mcts_quiescence_fires: int
    mcts_mean_depth: float
    mcts_mean_root_concentration: float
    cluster_value_std_mean: float
    cluster_policy_disagreement_mean: float
    cluster_variance_sample_count: int
    # §P3.2: ``runner_encoding`` retired alongside the legacy 4-field
    # PyEncodingSpec PyO3 class.  Field kept as a vestigial ``None``-valued
    # slot to avoid breaking external callers that construct ``RunnerStats``
    # by kwarg.  Tests that previously cross-checked ``cluster_window_size``
    # / ``cluster_threshold`` / ``legal_move_radius`` on this object now
    # cross-check ``pool.encoding_spec`` (registry-form spec, threaded
    # through to the Rust runner via ``encoding_spec=``).
    runner_encoding: Any = None


@dataclass(frozen=True)
class InferenceStats:
    """Snapshot of ``InferenceServer`` counters + bound encoding spec."""

    forward_count: int
    total_requests: int
    encoding_spec: Any  # hexo_rl.encoding.EncodingSpec


@dataclass(frozen=True)
class ResolvedPoolEncoding:
    """§176 P20 — encoding-resolve result for ``WorkerPool.__init__``.

    Bundles every encoding-derived value the WorkerPool wires through the
    Rust ``SelfPlayRunner`` and the Python NN-input/buffer dims:

      - ``registry_spec`` — canonical ``hexo_rl.encoding.EncodingSpec``
                            (== ``engine.RegistrySpec`` since cycle 3
                            Wave 8 Batch A FF.2).
      - ``encoding_name`` — registry name (e.g. "v6", "v6w25"). Wired to
                            the Rust ``SelfPlayRunner`` via
                            ``encoding_name=`` (cycle 3 Wave 8 Batch C
                            FF.10 — collapsed the WireFormatSpec +
                            ``encoding_spec=PyRegistrySpec`` round-trip
                            into a single string lookup at the Rust
                            boundary).
      - ``board_size`` / ``trunk_size`` / ``n_kept_planes`` — scalar
                                                              dims reused
                                                              for buffer
                                                              + reshape.
    """

    registry_spec: Any  # RegistrySpec (full schema)
    encoding_name: str  # registry name; wired to Rust runner via encoding_name kwarg
    board_size: int
    trunk_size: int
    n_kept_planes: int


def _resolve_encoding_for_pool(
    config: Dict[str, Any], model: Any | None = None
) -> ResolvedPoolEncoding:
    """Resolve every encoding-derived value ``WorkerPool.__init__`` needs.

    §172 A4.2 — resolves via the new ``hexo_rl.encoding`` registry.
    §173 A8' — multi-window guard lifted; v8 selfplay still loud-fails.
    §176 P20 — extracted from ``WorkerPool.__init__`` for unit testability.
    Cycle 3 Wave 8 Batch C (FF.10, 2026-05-17) — collapsed the
        WireFormatSpec + PyO3 ``encoding_spec=`` round-trip into a single
        string lookup at the Rust boundary; the runner now takes
        ``encoding_name: Optional[str]`` and resolves the registry record
        once on the Rust side.

    Validation:
      - v8 / v8_canvas_realness → ``NotImplementedError`` (Rust runner
        path absent; pretrain via ``dataset_v8.py`` instead).
      - When ``model`` is supplied, cross-checks ``model.board_size``
        against the resolved canvas geometry. Mis-paired model+config
        loud-fails with ``ValueError`` before any Rust runner is built.

    Returns a frozen ``ResolvedPoolEncoding`` dataclass; the caller wires
    the fields into ``SelfPlayRunner(...)`` and the Python buffer dims.
    """
    registry_spec: RegistrySpec = registry_resolve_from_config(config)

    # v8 selfplay guard (§172 A4.2). v8 has no Rust selfplay runner
    # path today; failing loud here beats letting downstream PyO3
    # construction raise an obscure ValueError later in __init__.
    if registry_spec.name in ("v8", "v8_canvas_realness"):
        raise NotImplementedError(
            f"v8 selfplay (encoding={registry_spec.name!r}) Rust runner "
            f"path not implemented; use v6 / v7full for selfplay or run "
            f"pretrain via dataset_v8.py + encode_position_v8."
        )

    # spec.board_size supersedes the legacy `getattr(model, 'board_size', 19)`
    # v6 fallback. Cross-check against the model's declared board_size so a
    # mis-paired model+config fails fast instead of silently routing planes
    # through wrong-shaped buffers.
    #
    # §173 A8' geometry split:
    #   - `board_size` is canvas geometry (physical hex grid extent).
    #   - `trunk_size` is the per-cluster NN-input window (== board_size
    #     for single-window encodings; == cluster_window_size for
    #     multi-window). All NN-input / buffer / reshape dims below use
    #     `trunk_size`; only the model.board_size cross-check uses the
    #     canvas value.
    spec = registry_spec
    if model is not None:
        model_board_size = int(getattr(model, "board_size", spec.board_size))
        if model_board_size != spec.board_size:
            raise ValueError(
                f"WorkerPool: model.board_size={model_board_size} disagrees "
                f"with resolved encoding {spec.name!r} (board_size="
                f"{spec.board_size}). Fix the variant `encoding.version` or "
                f"the checkpoint hparam mismatch before re-launching."
            )

    return ResolvedPoolEncoding(
        registry_spec=registry_spec,
        encoding_name=spec.name,
        board_size=spec.board_size,
        trunk_size=spec.trunk_size,
        n_kept_planes=len(spec.kept_plane_indices),
    )


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
        # §176 P20 — encoding-resolve extracted to `_resolve_encoding_for_pool`.
        # Returns a frozen ResolvedPoolEncoding bundling registry_spec,
        # wire_format_spec, runner_encoding, runner_registry_spec, plus the
        # board_size/trunk_size/n_kept_planes scalars. v8 selfplay still
        # loud-fails inside the helper; model.board_size cross-check is
        # delegated through `model=model`.
        #
        # Multi-window α (§173 A8') unblocked. K-cluster fan happens in
        # worker_loop.rs (get_cluster_views + aggregate_policy + min-pool
        # value); see engine/src/game_runner/worker_loop.rs:340-470 + :716-754.
        # NN-input geometry below uses spec.trunk_size (per-cluster window
        # dim) instead of spec.board_size (canvas geometry) so any
        # multi-window encoding with cluster_window_size != board_size
        # routes correctly. v6w25 (trunk_size == board_size == 25)
        # coincides today; the refactor unblocks future arches.
        _resolved = _resolve_encoding_for_pool(config, model=model)
        registry_spec: RegistrySpec = _resolved.registry_spec
        self.encoding_spec: RegistrySpec = registry_spec
        spec = registry_spec
        board_size = _resolved.board_size
        trunk_size = _resolved.trunk_size
        n_kept_planes = _resolved.n_kept_planes

        # Rust inference batcher uses WIRE_CHANNELS (8) planes wide; the
        # game runner slices to len(spec.kept_plane_indices) before pushing
        # to the results queue (HEXB v6 / §173 A3 spec-driven schema).

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
        # Cycle 3 Wave 8 Batch C (FF.10, 2026-05-17): the legacy
        # `encoding_spec=PyRegistrySpec` round-trip retired alongside
        # the WireFormatSpec shim. The Rust runner now takes the
        # registry name as a string and resolves the record once on
        # the Rust side (worker_loop reads `spec_static` for sym_tables
        # / n_cells / kept_planes / policy_stride / agg_trunk_sz and
        # for per-game `Board::with_registry_spec(spec)`).
        encoding_name = _resolved.encoding_name
        # §P55 / Wave 5a operator follow-up — opt-in `inference_pool_size`
        # forwarded from `selfplay.inference_pool_size`. Default = None
        # preserves cycle-1 behavior (InferenceBatcher uses its fixed 512
        # feature-buffer prefill / 1024 channel capacity). Operator opts in
        # via YAML for v6w25 16-worker high-K runs where the per-leaf K_avg
        # working set (~768 with n_workers=16, leaf_batch=8, K_avg≈6) exceeds
        # the 512 default; recommended size is
        # `n_workers * leaf_batch_size * K_max * 2` per the §P55 commit body.
        inference_pool_size = sp.get("inference_pool_size", None)
        if inference_pool_size is not None:
            inference_pool_size = int(inference_pool_size)
        # cycle 3 Wave 7 Batch A (P79): kwargs now ride on the
        # `SelfPlayRunnerConfig` builder; `SelfPlayRunner(config)` takes the
        # config struct. Breaking PyO3 API change — `!`-marked commit.
        self._runner = SelfPlayRunner(SelfPlayRunnerConfig(
            n_workers=self.n_workers,
            max_moves_per_game=int(sp.get("max_game_moves", sp.get("max_moves_per_game", 128))),
            n_simulations=self.n_simulations,
            leaf_batch_size=leaf_batch_size,
            c_puct=self.c_puct,
            fpu_reduction=self.fpu_reduction,
            # §173 A8' — NN-input geometry uses trunk_size (per-cluster
            # window) + policy_logit_count (handles has_pass_slot per
            # encoding). For v6w25 trunk_size == board_size == 25 so v6w25
            # coincides numerically; future arches with cluster_window_size
            # != board_size route correctly without further edits.
            feature_len=WIRE_CHANNELS * trunk_size * trunk_size,
            policy_len=spec.policy_logit_count,
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
            encoding_name=encoding_name,
            inference_pool_size=inference_pool_size,
        ))
        self._inference_server = InferenceServer(
            model, device, config, batcher=self._runner.batcher,
            encoding_spec=spec,
        )

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

        self._board_size = board_size            # canvas geometry (physical hex grid)
        self._trunk_size = trunk_size            # per-cluster NN-input geometry
        # §173 A8' — feat_len / chain_len keyed on trunk_size (per-cluster
        # window) and n_kept_planes (== len(spec.kept_plane_indices)) so the
        # schema is symmetric with engine/src/game_runner/worker_loop.rs:722
        # (`kept_planes.len() * n_cells`). For v6 / v7full this collapses to
        # 8*19*19=2888 and for v6w25 expands to 8*25*25=5000 — the same
        # values the Rust side computes from the spec.
        self._feat_len = n_kept_planes * trunk_size * trunk_size
        self._pol_len = spec.policy_logit_count
        self._chain_len = 6 * trunk_size * trunk_size

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

    # §176 P9 — typed accessors. Read-only snapshots; pure-move discipline
    # (no new computation, no behaviour change vs prior ``pool._runner.X``
    # / ``pool._inference_server.X`` reaches).
    def runner_stats(self) -> RunnerStats:
        """Snapshot of ``SelfPlayRunner`` counters / scalars.

        Defaults via ``getattr`` cover pre-Tier-1.A engine wheels that
        predate any individual counter — matches the legacy
        ``getattr(_runner, "field", 0.0)`` reaches in ``events.py``.
        """
        r = self._runner
        return RunnerStats(
            games_completed=int(getattr(r, "games_completed", 0)),
            positions_generated=int(getattr(r, "positions_generated", 0)),
            x_wins=int(getattr(r, "x_wins", 0)),
            o_wins=int(getattr(r, "o_wins", 0)),
            draws=int(getattr(r, "draws", 0)),
            model_version=int(getattr(r, "model_version", 0)),
            mcts_quiescence_fires=int(getattr(r, "mcts_quiescence_fires", 0)),
            mcts_mean_depth=float(getattr(r, "mcts_mean_depth", 0.0)),
            mcts_mean_root_concentration=float(
                getattr(r, "mcts_mean_root_concentration", 0.0)
            ),
            cluster_value_std_mean=float(getattr(r, "cluster_value_std_mean", 0.0)),
            cluster_policy_disagreement_mean=float(
                getattr(r, "cluster_policy_disagreement_mean", 0.0)
            ),
            cluster_variance_sample_count=int(
                getattr(r, "cluster_variance_sample_count", 0)
            ),
            # §P3.2: legacy `r.encoding` getter retired alongside the
            # Rust `SelfPlayRunner.encoding` field; runner_encoding stays
            # at the field default (`None`) for vestigial-field callers.
        )

    def inference_stats(self) -> InferenceStats:
        """Snapshot of ``InferenceServer`` counters + bound encoding spec."""
        s = self._inference_server
        return InferenceStats(
            forward_count=int(getattr(s, "_forward_count", 0)),
            total_requests=int(getattr(s, "_total_requests", 0)),
            encoding_spec=getattr(s, "encoding_spec", None),
        )

    def sync_inference_weights(self, state_dict: Dict[str, Any]) -> None:
        """Forward a fresh state_dict to the bound inference server.

        Promotion path (eval_drain) used to reach into
        ``pool._inference_server.load_state_dict_safe(...)`` — this is a
        mutating action, not a stat, so it gets its own forwarder rather
        than living on the snapshot dataclasses.
        """
        self._inference_server.load_state_dict_safe(state_dict)

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
        # §173 A8' — reshape NN-input tensors using trunk_size (per-cluster
        # window), not board_size (canvas). For single-window encodings the
        # two coincide; under multi-window the buffer holds per-cluster rows.
        _in_ch = self._feat_len // (self._trunk_size * self._trunk_size)
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
                    n, _in_ch, self._trunk_size, self._trunk_size,
                )
                chain_f16 = chain_np.astype(np.float16).reshape(
                    n, 6, self._trunk_size, self._trunk_size,
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
                    _stride5_run, _row_max_density = _compute_stride5_metrics(move_history)
                else:
                    _stride5_run = 0
                    _row_max_density = 0

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
                    # §176 P23 — densest hex-row stone count (any of 3 axes).
                    "row_max_density":   int(_row_max_density),
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

    def set_radius_override(self, radius: int | None) -> None:
        """§174 — update the per-game legal-move radius override live.

        ``None`` clears the override (use encoding default).  Propagated to
        the Rust ``SelfPlayRunner`` atomic; workers read it at the start of
        each game.
        """
        self._runner.set_radius_override(radius)

    def stop(self) -> None:
        self._stop_event.set()
        self._runner.stop()
        self._inference_server.stop()
        self._inference_server.join(timeout=5.0)

        if self._stats_thread is not None:
            self._stats_thread.join(timeout=5.0)
            self._stats_thread = None

        self._recorder.stop()
