"""Regression guard for the GPUMonitor `Thread._stop` shadowing bug.

`threading.Thread` defines a private `_stop()` method that `join()`
invokes via `_wait_for_tstate_lock`. If a Thread subclass shadows
`self._stop` with anything that is not callable, `join()` raises
`TypeError: 'Event' object is not callable` on clean teardown.

The first instance of the bug surfaced at the end of the §S181-AUDIT
Wave 1 Track B B4 instrumented run (lifecycle.py:177 `gpu_monitor.join`
call after 3000 training steps completed cleanly).
"""
from __future__ import annotations

import threading

import pytest

from hexo_rl.monitoring.gpu_monitor import GPUMonitor


def test_gpu_monitor_stop_event_is_not_named_underscore_stop():
    """The Event member must NOT be named `_stop` (Thread internals reserve it).

    On Python 3.12 `threading.Thread` defines a private `_stop()` method
    that `join()` invokes via `_wait_for_tstate_lock`; shadowing it with
    an Event instance raises `TypeError: 'Event' object is not callable`
    on clean teardown (the original symptom in the §S181-AUDIT Wave 1
    Track B B4 run on the remote Python 3.12 runtime). Python 3.14
    refactored away the method, so the bug is 3.12-specific — but the
    invariant must survive future Python upgrades that may reintroduce
    the same naming. Keep the Event on `_stop_event` regardless.
    """
    mon = GPUMonitor(interval_sec=0.01)
    # Whatever `mon._stop` resolves to (a bound method on Python 3.12, or
    # AttributeError on Python 3.14), it must NOT be our Event instance.
    stop_attr = getattr(mon, "_stop", None)
    assert not isinstance(stop_attr, threading.Event), (
        "GPUMonitor._stop must not shadow Thread's internal _stop; "
        "the stop-signal Event belongs on `_stop_event`."
    )
    assert isinstance(mon._stop_event, threading.Event), (
        "GPUMonitor._stop_event must be a threading.Event"
    )


def test_gpu_monitor_clean_start_stop_join_does_not_raise():
    """Full lifecycle (start → stop → join) must not raise.

    The bug presented at lifecycle.teardown's `gpu_monitor.join(timeout=2.0)`
    call after the iteration limit was reached. Reproduce with a minimal
    monitor: start (no NVML, no-op), stop, join.
    """
    mon = GPUMonitor(interval_sec=0.01)
    mon.start()
    mon.stop()
    # join must NOT raise even if pynvml isn't installed (start sets
    # self._handle = None and run() returns early in that branch).
    mon.join(timeout=2.0)
    assert not mon.is_alive()
