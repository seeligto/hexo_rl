"""§P5-CT P0-1 — orchestrator fresh-run in_channels routes through the registry.

Pre-fix: `int(combined_config.get("in_channels", 18))` baked the literal 18 —
neither the wire width (8) nor any encoding's plane count (4/8/10/11). A variant
YAML that omits `in_channels` built an 18-channel trunk against an n-plane wire.
Fix: fall back to resolve_arch(encoding).in_channels.
"""
from __future__ import annotations

import pytest

from hexo_rl.training.orchestrator import _resolve_fresh_in_channels


@pytest.mark.parametrize(
    "encoding,expected",
    [
        ("v6", 8),
        ("v6tp", 10),
        ("v6_live2", 4),
        ("v8", 11),
        (None, 8),  # absent encoding key → v6 default (8), NOT the legacy 18
    ],
)
def test_fresh_in_channels_falls_back_to_encoding_planes(encoding, expected):
    cfg: dict = {} if encoding is None else {"encoding": encoding}
    in_ch, input_channels = _resolve_fresh_in_channels(cfg)
    assert in_ch == expected
    assert input_channels is None


def test_fresh_in_channels_never_defaults_to_18():
    """The legacy literal 18 must never be produced for a 4-plane encoding."""
    in_ch, _ = _resolve_fresh_in_channels({"encoding": "v6_live2"})
    assert in_ch == 4 and in_ch != 18


def test_explicit_in_channels_consistent_with_encoding_wins():
    """An explicit in_channels (consistent with the encoding) is honoured."""
    in_ch, _ = _resolve_fresh_in_channels({"encoding": "v6tp", "in_channels": 10})
    assert in_ch == 10


def test_input_channels_list_overrides():
    """input_channels list still derives in_channels from its length."""
    cfg: dict = {"encoding": "v6", "input_channels": [0, 8, 16]}
    in_ch, input_channels = _resolve_fresh_in_channels(cfg)
    assert in_ch == 3
    assert input_channels == [0, 8, 16]
    assert cfg["in_channels"] == 3  # mutated in-place (preserved behaviour)
