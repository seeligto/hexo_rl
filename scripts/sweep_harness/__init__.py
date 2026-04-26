"""Hardware-agnostic, knob-registry-driven throughput sweep harness.

Replaces the host-specific scripts/sweep_epyc4080.sh. Each knob in
``knobs.KNOBS`` declares its own search strategy (ternary/grid/bisect/fixed);
the runner orchestrates per-knob search with IQR-aware comparison and
bimodality detection. Designed so a freshly rented vast.ai box converges to
an optimal config in under 90 minutes without editing the script per host.

See docs/sweep_harness.md for the knob-recipe and __main__.py for the CLI.
"""

from .compare import CellResult, bimodal_from_raw, compare_iqr
from .knobs import KNOBS, knob_registry, resolve_auto_bounds
from .strategies import (
    bisect_search,
    grid_coarse_refine,
    grid_search,
    ternary_search_int,
)

__all__ = [
    "CellResult",
    "KNOBS",
    "bimodal_from_raw",
    "bisect_search",
    "compare_iqr",
    "grid_coarse_refine",
    "grid_search",
    "knob_registry",
    "resolve_auto_bounds",
    "ternary_search_int",
]
