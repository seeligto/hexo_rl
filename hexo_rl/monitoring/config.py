"""Centralized defaults for monitoring renderers.

`MonitoringConfig` is the single source of truth for static default values
that were previously duplicated across `terminal_dashboard.py`,
`web_dashboard.py`, and the `/api/monitoring-config` endpoint.

Renderers should instantiate `MonitoringConfig.from_dict(cfg)` once during
`__init__` and read fields via attribute access instead of calling
`cfg.get("monitoring", cfg).get("foo", DEFAULT_FOO)` at every site.

Back-compat: `from_dict` accepts either the top-level config dict
(with an optional `"monitoring"` subsection) or an already-unwrapped
monitoring dict. Unknown keys are ignored so callers can pass dicts that
mix monitoring and non-monitoring entries.

Note: `num_actions_for_entropy_norm` is intentionally NOT folded in here
— its default depends on the runtime `board_size` (board_size ** 2 + 1)
which is not a static constant. The 2 remaining inline-default sites
(terminal_dashboard.py + web_dashboard.py monitoring-config endpoint)
are addressed by a separate refactor (§176 P56) that wires the encoding
spec through directly.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MonitoringConfig:
    """Static default values shared by monitoring renderers."""

    # Alert thresholds (terminal + web)
    alert_entropy_min: float = 1.0
    alert_entropy_warn: float = 2.0
    alert_grad_norm_max: float = 10.0
    alert_loss_increase_window: int = 3
    collapse_threshold_nats: float = 1.5

    # §S181-AUDIT Wave 3 Stage 2B — sliding-window SealBot WR hard-abort (L50).
    # Wave 2 evidence: alt V_spread sustained +0.18-+0.30 across 46k steps
    # while wr_sealbot collapsed 33% → 5%. The held-out V_spread canary
    # failed to track actual eval performance. L50 mandates a SealBot WR
    # sliding-window gate as the PRIMARY abort trigger.
    # Operator can disable for debug (NEVER on real runs) via wr_hard_abort_enabled.
    wr_hard_abort_enabled: bool = True
    wr_rolling_consecutive_evals: int = 2
    wr_rolling_threshold: float = 0.10
    wr_rolling_min_step: int = 20000
    wr_collapse_from_peak_ratio: float = 0.5
    wr_collapse_min_step: int = 25000
    # Require N consecutive eval rounds below the floor before HARD-ABORT fires
    # for triggers B (collapse-from-peak) and C (early death). The colony
    # attractor causes transient SealBot-WR dips that self-correct (§175/L34);
    # a single 5% dip at step 75k aborted a RECOVERING golong run at 87.5k
    # (2026-06-07 — 87.5k re-eval recovered to ~0.23). 3 consec @ 12.5k cadence
    # = ~37.5k of sustained collapse, distinguishing a real collapse from an
    # oscillation. (Trigger A already has its own wr_rolling_consecutive_evals.)
    wr_collapse_consecutive_evals: int = 3
    wr_early_death_threshold: float = 0.05
    wr_early_death_min_step: int = 15000

    # Web dashboard server
    web_port: int = 5001
    web_host: str = "127.0.0.1"
    socketio_async_mode: str = "threading"

    # Event log / history retention
    event_log_maxlen: int = 500
    training_step_history: int = 2000
    game_history: int = 500
    emit_queue_maxsize: int = 200

    # Viewer game index
    viewer_max_memory_games: int = 50
    viewer_max_disk_games: int = 1000
    viewer_games_dir: str = "runs"

    # Client-side EMA / win-rate target band (web /api/monitoring-config)
    ema_alpha: float = 0.06
    p0_win_rate_target_low: float = 54.0
    p0_win_rate_target_high: float = 58.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "MonitoringConfig":
        """Build a MonitoringConfig from a (possibly nested) config dict.

        Accepts either the full training config dict (in which case the
        `"monitoring"` subsection is consulted if present, falling back to
        the outer dict for back-compat) or an already-unwrapped monitoring
        dict. Unknown keys are ignored.
        """
        if not cfg:
            return cls()
        mon = cfg.get("monitoring", cfg) if isinstance(cfg, dict) else cfg
        if not isinstance(mon, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: mon[k] for k in known if k in mon}
        return cls(**kwargs)
