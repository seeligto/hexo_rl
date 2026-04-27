"""Knob registry — the only host-specific config in the harness.

Each knob entry declares:

* ``strategy`` — one of ``ternary``, ``grid_coarse_refine``, ``grid``,
  ``bisect``, ``fixed``.
* Strategy-specific config (bounds, values, iterations, ...).
* ``param_path`` — dotted path inside the variant YAML where the chosen
  value is written. The runner converts the path + value into a nested
  dict and merges it onto ``configs/variants/_sweep_template.yaml``.
* Optional ``auto_bounds_fn(host)`` that turns detected hardware into
  concrete bounds when ``bounds == "auto"``.
* Optional ``constraint`` — predicate on candidate values; values that
  fail the constraint are filtered before eval.

Adding a knob: see ``docs/sweep_harness.md``.
"""

from __future__ import annotations

from typing import Any, Callable


# Knob registry. Pure config — importing this module must not run anything.
KNOBS: dict[str, dict[str, Any]] = {
    "n_workers": {
        "strategy": "ternary",
        "bounds": "auto",
        "auto_bounds_fn": lambda host: (
            max(8, host["cpu_threads"] // 4),
            min(64, host["cpu_threads"]),
        ),
        "param_path": "selfplay.n_workers",
        "iterations": 4,
        "tolerance": 2,
        "doc": (
            "Worker count is the dominant lever (§125). Unimodal: rises to "
            "GPU saturation, plateaus, degrades from cache contention. "
            "Search FIRST so downstream knobs see the right batch-fill regime."
        ),
    },
    "inference_batch_size": {
        "strategy": "grid_coarse_refine",
        "coarse": [256, 320, 384, 448, 512],
        "refine_window": 1,
        "refine_step": 32,
        "param_path": "selfplay.inference_batch_size",
        "constraint": "must_be_>=_n_workers_x2",
        "doc": (
            "Coarse-then-refine since the optimum tends to land on or near "
            "a power-of-two. Constraint: batch >= n_workers * 2 — below that "
            "the InferenceServer never assembles a full batch."
        ),
    },
    "inference_max_wait_ms": {
        "strategy": "grid",
        "values": [2.0, 4.0, 8.0],
        "param_path": "selfplay.inference_max_wait_ms",
        "doc": (
            "Three values, exhaustive grid. At fill ≥ 96 % wait barely "
            "matters; the search confirms which side of saturation we are on."
        ),
    },
    "max_train_burst": {
        "strategy": "bisect",
        "bounds": (8, 64),
        "iterations": 3,
        "param_path": "training.max_train_burst",
        "doc": (
            "Lowest burst that doesn't starve the trainer — pos/hr is "
            "near-flat above the threshold; bisect for the cheap value."
        ),
    },
    "leaf_burst": {
        "strategy": "fixed",
        "value": 8,
        "param_path": "selfplay.leaf_burst",
        "skip_reason": "§125 marginal effect, fixed at validate winner",
    },
}


# Strategy ordering — n_workers MUST be searched first (§125: it's the
# binding lever; downstream knobs depend on the right worker regime).
KNOB_ORDER: tuple[str, ...] = (
    "n_workers",
    "inference_batch_size",
    "inference_max_wait_ms",
    "max_train_burst",
    "leaf_burst",
)


def knob_registry() -> dict[str, dict[str, Any]]:
    """Return a defensive copy of the registry."""
    return {k: dict(v) for k, v in KNOBS.items()}


def resolve_auto_bounds(knob_name: str, host: dict[str, Any]) -> tuple[int, int]:
    """Resolve ``bounds == "auto"`` to concrete (low, high) using host info."""
    knob = KNOBS[knob_name]
    bounds = knob.get("bounds")
    if bounds == "auto":
        fn = knob["auto_bounds_fn"]
        return fn(host)
    if isinstance(bounds, tuple):
        return bounds
    raise ValueError(f"knob {knob_name}: bounds must be 'auto' or tuple, got {bounds!r}")


def resolve_constraint(spec: Any, fixed: dict[str, Any]) -> Callable[[int], bool] | None:
    """Turn a constraint spec from the registry into a predicate.

    Today only ``"must_be_>=_n_workers_x2"`` is supported. ``fixed`` is the
    dict of already-resolved knob winners (so a downstream knob can
    constrain on an upstream choice).
    """
    if spec is None:
        return None
    if spec == "must_be_>=_n_workers_x2":
        nw = int(fixed.get("n_workers", 1))
        return lambda v: v >= nw * 2
    raise ValueError(f"unknown constraint spec: {spec!r}")


def param_path_to_yaml(param_path: str, value: Any) -> dict[str, Any]:
    """``"selfplay.n_workers", 24 → {"selfplay": {"n_workers": 24}}``."""
    parts = param_path.split(".")
    out: dict[str, Any] = {}
    cur = out
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return out


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge (later wins). Used to layer knob overrides
    on top of the sweep template."""
    out: dict[str, Any] = {}

    def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    for d in dicts:
        if d:
            _merge(out, d)
    return out
