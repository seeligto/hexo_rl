"""
Per-host TF32 configuration.

Rationale — see reports/investigations/tf32_channels_last_20260423/report.md
and sprint log §117. Short version: TF32 matmul is a near-free latency win on
some GPU architectures and a net regression on others. This module resolves
the `gpu.tf32_matmul` and `gpu.tf32_cudnn` config entries against the active
CUDA compute capability and applies the corresponding PyTorch flags once at
process startup.

Config schema (configs/training.yaml → `gpu:`):
    tf32_matmul: auto | on | off    # default: auto
    tf32_cudnn:  auto | on | off    # default: auto

`auto` resolves to True for architectures that the probe (or architectural
reasoning) established as TF32-positive, False otherwise. Unmeasured arches
resolve to True with a warning-log flag — re-probe on first run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

log = logging.getLogger(__name__)

# Architectures where TF32 has been measured (2026-04-23 probe) or inferred
# from architectural reasoning (see §117):
#   sm_86 (Ampere consumer, RTX 3070): probe LOSES 5.9% latency → False
#   sm_89 (Ada consumer, RTX 4060): probe WINS 5.8% latency → True
#   sm_80 (Ampere datacenter, A100): inferred True (HBM bandwidth + TF32
#         hardware delivery matches Ada better than consumer Ampere);
#         UNMEASURED — re-probe on first cloud run.
#   sm_90+ (Hopper, H100 and up): inferred True (next-gen tensor cores);
#          UNMEASURED — re-probe on first cloud run.
_TF32_MEASURED = {
    (8, 6): False,
    (8, 9): True,
}


def _arch_default(cc: tuple) -> tuple[bool, bool]:
    """Return (resolved_on, measured) for a given compute capability."""
    if cc in _TF32_MEASURED:
        return _TF32_MEASURED[cc], True
    # Unmeasured heuristic: A100 (8, 0) and Hopper+ (9, x+) default on.
    # Consumer Ampere variants (other 8, x) default off (sm_86 analogy).
    if cc == (8, 0):
        return True, False
    if cc >= (9, 0):
        return True, False
    if cc[0] == 8:
        return False, False
    # Pre-Ampere (sm_75 and below) — no TF32 hardware.
    return False, True


def _resolve_one(setting: str, cc: tuple, knob_name: str) -> bool:
    if setting == "on":
        return True
    if setting == "off":
        return False
    if setting != "auto":
        raise ValueError(
            f"gpu.{knob_name} must be one of 'on'|'off'|'auto', got {setting!r}"
        )
    on, measured = _arch_default(cc)
    if not measured and torch.cuda.is_available():
        log.warning(
            "tf32_auto_unmeasured_arch",
            extra={
                "knob": knob_name,
                "compute_capability": f"sm_{cc[0]}{cc[1]}",
                "resolved": on,
                "remedy": "run scripts/probe_tf32_channels_last.py on this host and update hexo_rl/model/tf32.py::_TF32_MEASURED",
            },
        )
    return on


def resolve_and_apply(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read gpu.* TF32 settings from config, resolve, apply, return resolved dict.

    Call once at process entrypoint (scripts/train.py, scripts/benchmark.py,
    scripts/eval*.py). Safe to call on CPU-only hosts: resolves both knobs to
    False and returns without touching any backend flag.
    """
    gpu_cfg = config.get("gpu") if isinstance(config.get("gpu"), dict) else {}
    matmul_setting = str(gpu_cfg.get("tf32_matmul", "auto"))
    cudnn_setting  = str(gpu_cfg.get("tf32_cudnn",  "auto"))

    if not torch.cuda.is_available():
        resolved = {
            "tf32_matmul": False,
            "tf32_cudnn":  False,
            "compute_capability": None,
            "source":       {"tf32_matmul": matmul_setting, "tf32_cudnn": cudnn_setting},
            "applied":      False,
        }
        return resolved

    cc = torch.cuda.get_device_capability()
    matmul_on = _resolve_one(matmul_setting, cc, "tf32_matmul")
    cudnn_on  = _resolve_one(cudnn_setting,  cc, "tf32_cudnn")

    # Belt-and-suspenders: set both the backend flag and the high-level
    # set_float32_matmul_precision routing. The latter picks tf32 vs fp32
    # decomposition paths inside PT; the former is the cuBLAS-level gate.
    torch.backends.cuda.matmul.allow_tf32 = matmul_on
    torch.backends.cudnn.allow_tf32       = cudnn_on
    torch.set_float32_matmul_precision("high" if matmul_on else "highest")

    resolved = {
        "tf32_matmul": matmul_on,
        "tf32_cudnn":  cudnn_on,
        "compute_capability": f"sm_{cc[0]}{cc[1]}",
        "source":       {"tf32_matmul": matmul_setting, "tf32_cudnn": cudnn_setting},
        "applied":      True,
    }
    log.info("tf32_resolved", extra=resolved)
    return resolved
