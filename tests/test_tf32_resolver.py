"""TF32 resolver — auto / on / off / bad-input behaviour.

Probe background: reports/investigations/tf32_channels_last_20260423/report.md
"""

from __future__ import annotations

import pytest

from hexo_rl.model.tf32 import _arch_default, _resolve_one


# ── _arch_default ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cc,expected_on,expected_measured",
    [
        # Measured arches — probe data lives in hexo_rl/model/tf32.py::_TF32_MEASURED.
        ((8, 6), False, True),   # RTX 3070: probe loses 5.9% latency
        ((8, 9), True,  True),   # RTX 4060 Laptop: probe wins 5.8% latency
        # Inferred on: A100 (datacenter Ampere) + Hopper+ (H100 and up).
        ((8, 0), True,  False),
        ((9, 0), True,  False),
        ((9, 5), True,  False),
        ((10, 0), True, False),
        # Inferred off: consumer Ampere variants not measured.
        ((8, 2), False, False),  # hypothetical consumer Ampere
        # Pre-Ampere: no TF32 hardware.
        ((7, 5), False, True),   # Turing RTX 2080
        ((6, 1), False, True),   # Pascal
    ],
)
def test_arch_default(cc, expected_on, expected_measured):
    on, measured = _arch_default(cc)
    assert on is expected_on
    assert measured is expected_measured


# ── _resolve_one ──────────────────────────────────────────────────────────────

def test_explicit_on_overrides_arch():
    # Even on 3070 (measured off), explicit 'on' wins.
    assert _resolve_one("on", (8, 6), "tf32_matmul") is True


def test_explicit_off_overrides_arch():
    # Even on 4060 (measured on), explicit 'off' wins.
    assert _resolve_one("off", (8, 9), "tf32_matmul") is False


@pytest.mark.parametrize(
    "cc,expected",
    [((8, 6), False), ((8, 9), True), ((8, 0), True), ((9, 0), True)],
)
def test_auto_resolves_per_arch(cc, expected):
    assert _resolve_one("auto", cc, "tf32_matmul") is expected


def test_invalid_setting_raises():
    with pytest.raises(ValueError, match="must be one of 'on'|'off'|'auto'"):
        _resolve_one("yes", (8, 9), "tf32_matmul")


def test_invalid_setting_names_knob():
    with pytest.raises(ValueError, match="gpu.tf32_cudnn"):
        _resolve_one("true", (8, 9), "tf32_cudnn")


# ── resolve_and_apply ──────────────────────────────────────────────────────────

def test_resolve_and_apply_cpu_is_noop(monkeypatch):
    """On a CPU-only host, resolve_and_apply returns applied=False and touches
    no backend flag."""
    import torch
    from hexo_rl.model import tf32 as m
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = m.resolve_and_apply({"gpu": {"tf32_matmul": "on", "tf32_cudnn": "on"}})
    assert out["applied"] is False
    assert out["tf32_matmul"] is False
    assert out["tf32_cudnn"] is False
    assert out["compute_capability"] is None


def test_resolve_and_apply_missing_gpu_section_defaults_auto(monkeypatch):
    """No `gpu:` key at all — resolver defaults to 'auto' for both knobs."""
    import torch
    from hexo_rl.model import tf32 as m
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = m.resolve_and_apply({})
    assert out["source"] == {"tf32_matmul": "auto", "tf32_cudnn": "auto"}


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="CUDA required for backend-flag verification",
)
def test_resolve_and_apply_sets_backend_flags():
    """On CUDA, resolved matmul/cudnn flags mirror the backend state."""
    import torch
    from hexo_rl.model import tf32 as m
    out = m.resolve_and_apply({"gpu": {"tf32_matmul": "on", "tf32_cudnn": "off"}})
    assert out["applied"] is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32       is False
    # Restore default for the rest of the test session (pt default: True for cudnn,
    # device-dependent for matmul — re-apply auto).
    m.resolve_and_apply({"gpu": {"tf32_matmul": "auto", "tf32_cudnn": "auto"}})
