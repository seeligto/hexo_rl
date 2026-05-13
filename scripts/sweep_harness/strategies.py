"""Per-knob search strategies.

Each strategy takes ``eval_fn: value -> CellResult`` and a ``compare_fn``
(see :mod:`compare`). Returns ``(best_value, trace)``. ``trace`` is a list
of dicts the reporter renders into the per-knob trace table.

Strategies do NOT know about subprocess, YAML, or the bench harness — they
operate on ``eval_fn`` callbacks only. This keeps strategy unit tests pure
(see ``tests/test_sweep_harness.py``).

The four strategies model different knob shapes:

* ``ternary_search_int`` — unimodal integer knob (n_workers, where pos/hr
  rises to GPU saturation, plateaus, degrades from cache contention).
  Eval cache means iter k re-uses iter k-1 evals when the bracket
  collapses; total evals ≈ ``2 + iterations``.
* ``grid_coarse_refine`` — knob with a known reasonable coarse grid plus
  a tighter refine around the winner. Used for ``inference_batch_size``
  where the optimum tends to align with a power-of-two coarse grid.
* ``grid_search`` — exhaustive over a small set (≤ 4 values).
* ``bisect_search`` — "lowest value that doesn't degrade": eval ``mid``
  and ``mid+1``, descend toward the higher of the two.
"""

from __future__ import annotations

from typing import Callable, Iterable

from .compare import CellResult


EvalFn = Callable[[int | float], CellResult]
CompareFn = Callable[[CellResult, CellResult], int]


def _cached(eval_fn: EvalFn) -> tuple[Callable[[int | float], CellResult], dict]:
    cache: dict = {}

    def wrapped(x: int | float) -> CellResult:
        if x not in cache:
            cache[x] = eval_fn(x)
        return cache[x]

    return wrapped, cache


def ternary_search_int(
    eval_fn: EvalFn,
    low: int,
    high: int,
    iterations: int,
    tolerance: int,
    compare_fn: CompareFn,
) -> tuple[int, list[dict]]:
    """Ternary search over an integer-valued unimodal function.

    Each iteration evaluates two interior points ``m1, m2`` and uses
    ``compare_fn`` to decide which third to drop. On TIE (``compare_fn``
    returns 0) the bracket shrinks symmetrically — equivalent to declaring
    the maximum is between m1 and m2.

    Stops when ``high - low <= tolerance`` OR ``iterations`` exhausted.
    Returns the cached arg-max (also evaluating the final endpoints).
    """
    if low > high:
        raise ValueError(f"ternary low={low} > high={high}")
    cached, cache = _cached(eval_fn)
    trace: list[dict] = []

    for it in range(iterations):
        if high - low <= tolerance:
            break
        third = max(1, (high - low) // 3)
        m1 = low + third
        m2 = high - third
        if m1 == m2:
            # bracket collapsed onto a single point — eval and stop
            cached(m1)
            trace.append({"iter": it, "low": low, "high": high, "m1": m1, "m2": m2,
                          "decision": "collapsed", "f_m1": cached(m1).median, "f_m2": cached(m2).median})
            break
        r1 = cached(m1)
        r2 = cached(m2)
        cmp = compare_fn(r1, r2)
        decision: str
        if cmp == 0:
            low, high = m1, m2
            decision = "tie-shrink"
        elif cmp > 0:
            high = m2
            decision = "left wins"
        else:
            low = m1
            decision = "right wins"
        trace.append({
            "iter": it, "low_in": low, "high_in": high,
            "m1": m1, "f_m1": r1.median, "iqr_m1": r1.iqr,
            "m2": m2, "f_m2": r2.median, "iqr_m2": r2.iqr,
            "decision": decision,
        })

    # final endpoint sweep so the arg-max sees a full bracket
    for x in (low, high):
        cached(x)

    best_val = max(cache, key=lambda x: cache[x].median)
    return best_val, trace


def grid_coarse_refine(
    eval_fn: EvalFn,
    coarse: list[int],
    refine_window: int,
    refine_step: int,
    compare_fn: CompareFn,
    constraint: Callable[[int], bool] | None = None,
) -> tuple[int, list[dict]]:
    """Coarse-grid winner, then refine ±``refine_window * refine_step``.

    ``constraint(value) -> bool`` filters out values that violate the rule
    (e.g. ``inference_batch_size >= n_workers * 2``). Filtered candidates
    do not consume an eval and never appear in the trace.

    Total evals: ``len(coarse) + (2 * refine_window)`` minus refine values
    already in ``coarse`` (cached).
    """
    cached, cache = _cached(eval_fn)
    trace: list[dict] = []

    valid = [v for v in coarse if constraint is None or constraint(v)]
    if not valid:
        raise ValueError(f"grid_coarse_refine: no candidate in {coarse} passes constraint")
    for v in valid:
        r = cached(v)
        trace.append({"phase": "coarse", "value": v, "median": r.median, "iqr": r.iqr})

    coarse_winner = max(valid, key=lambda v: cache[v].median)

    refine: list[int] = []
    for k in range(1, refine_window + 1):
        for delta in (-k * refine_step, +k * refine_step):
            cand = coarse_winner + delta
            if cand <= 0 or cand in cache:
                continue
            if constraint is not None and not constraint(cand):
                continue
            refine.append(cand)
    for v in refine:
        r = cached(v)
        trace.append({"phase": "refine", "value": v, "median": r.median, "iqr": r.iqr})

    best_val = max(cache, key=lambda x: cache[x].median)
    return best_val, trace


def grid_search(
    eval_fn: EvalFn,
    values: list[int | float],
    compare_fn: CompareFn,
) -> tuple[int | float, list[dict]]:
    """Exhaustive eval over ``values``. Used when ``len(values) <= 4``."""
    cached, cache = _cached(eval_fn)
    trace: list[dict] = []
    for v in values:
        r = cached(v)
        trace.append({"value": v, "median": r.median, "iqr": r.iqr})
    best_val = max(cache, key=lambda x: cache[x].median)
    return best_val, trace


def bisect_search(
    eval_fn: EvalFn,
    low: int,
    high: int,
    iterations: int,
    compare_fn: CompareFn,
) -> tuple[int, list[dict]]:
    """"Lowest-value-that-doesn't-degrade" descent.

    Each iter evaluates ``mid`` and ``mid + 1``. If ``f(mid+1) > f(mid)``
    (compare_fn says right wins) the optimum is to the right → ``low = mid``.
    Otherwise (left wins or tie) the optimum is at or below mid → ``high = mid``.
    Tie collapses leftward (favor cheaper).
    """
    if low > high:
        raise ValueError(f"bisect low={low} > high={high}")
    cached, cache = _cached(eval_fn)
    trace: list[dict] = []

    for it in range(iterations):
        if high - low <= 1:
            break
        mid = (low + high) // 2
        nxt = mid + 1
        r_mid = cached(mid)
        r_nxt = cached(nxt)
        cmp = compare_fn(r_nxt, r_mid)
        if cmp > 0:
            low = mid
            decision = "right wins"
        else:
            high = mid
            decision = "left wins" if cmp < 0 else "tie-favor-left"
        trace.append({
            "iter": it, "mid": mid, "f_mid": r_mid.median, "iqr_mid": r_mid.iqr,
            "next": nxt, "f_next": r_nxt.median, "iqr_next": r_nxt.iqr,
            "decision": decision,
        })

    for x in (low, high):
        cached(x)

    best_val = max(cache, key=lambda x: cache[x].median)
    return best_val, trace
