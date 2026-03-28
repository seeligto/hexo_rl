"""
GPU monitoring daemon thread.

Polls pynvml every `interval_sec` seconds and emits structured log events
with GPU utilisation, VRAM usage, and temperature.

Usage:
    monitor = GPUMonitor(interval_sec=5)
    monitor.start()
    # ... training ...
    monitor.stop()
    monitor.join()
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import structlog

log = structlog.get_logger()

# Lazy import so the module can be imported even if pynvml is not installed
# (will raise at start() time, not at import time).
_pynvml = None


def _get_pynvml():
    global _pynvml
    if _pynvml is None:
        import pynvml
        _pynvml = pynvml
    return _pynvml


class GPUMonitor(threading.Thread):
    """Daemon thread that periodically logs GPU stats via structlog.

    Emits ``gpu_stats`` events with the following fields:
      - ``gpu_util_pct``:   GPU compute utilisation (0–100).
      - ``mem_util_pct``:   Memory controller utilisation (0–100).
      - ``vram_used_gb``:   VRAM used in gigabytes.
      - ``vram_total_gb``:  Total VRAM in gigabytes.
      - ``temp_c``:         GPU core temperature in °C.

    Args:
        interval_sec:  Poll interval in seconds (default 5).
        device_index:  NVML device index (default 0 for the first GPU).
    """

    def __init__(self, interval_sec: float = 5.0, device_index: int = 0) -> None:
        super().__init__(daemon=True, name="gpu-monitor")
        self.interval     = interval_sec
        self.device_index = device_index
        self._stop        = threading.Event()
        self._handle      = None  # nvml device handle, set in start()

        # Latest stats (read-only from other threads; approximate, no lock).
        self.gpu_util_pct:  float = 0.0
        self.vram_used_gb:  float = 0.0
        self.vram_total_gb: float = 0.0
        self.temp_c:        float = 0.0

    def start(self) -> None:
        pynvml = _get_pynvml()
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            log.warning("gpu_monitor_init_failed", error=str(exc))
            self._handle = None
        super().start()

    def run(self) -> None:
        if self._handle is None:
            return  # no GPU or nvml init failed — silently exit
        pynvml = _get_pynvml()
        while not self._stop.wait(self.interval):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
                self.gpu_util_pct  = float(util.gpu)
                self.vram_used_gb  = mem.used  / 1e9
                self.vram_total_gb = mem.total / 1e9
                self.temp_c        = float(temp)

                log.info(
                    "gpu_stats",
                    gpu_util_pct  = self.gpu_util_pct,
                    mem_util_pct  = float(util.memory),
                    vram_used_gb  = self.vram_used_gb,
                    vram_total_gb = self.vram_total_gb,
                    temp_c        = self.temp_c,
                )
            except Exception as exc:
                log.warning("gpu_monitor_poll_error", error=str(exc))

    def stop(self) -> None:
        """Signal the monitor to stop after the current sleep."""
        self._stop.set()
