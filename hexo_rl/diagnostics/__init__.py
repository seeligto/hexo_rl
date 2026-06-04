"""Diagnostics — shared, production-importable readouts derived from already-recorded
games.  No hot-path passes; consumers (eval gate, dashboard, structured logs, probes)
import ONE definition so a metric cannot drift between copies (§EVALGATE-B)."""

from hexo_rl.diagnostics.forced_win_detector import (
    ForcedWinTrend,
    GameForcedWinSummary,
    analyze_recorded_game,
    analyze_replay_file,
    emit_forced_win_trend,
    update_trend_from_file_incremental,
)

__all__ = [
    "ForcedWinTrend",
    "GameForcedWinSummary",
    "analyze_recorded_game",
    "analyze_replay_file",
    "emit_forced_win_trend",
    "update_trend_from_file_incremental",
]
