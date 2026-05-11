"""§172 A4.5 — scattered-key rejection in resolve_from_config.

Strict path: when a config has an explicit `encoding` declaration, every
scattered scalar (board_size, n_planes, in_channels, cluster_*,
legal_move_radius) must agree with the registry spec.

Lenient path: legacy configs without an `encoding:` key downgrade
disagreement to a DeprecationWarning and still resolve as v6.
"""
from __future__ import annotations

import warnings

import pytest

from hexo_rl.encoding import EncodingRegistryError, resolve_from_config


def test_v6_explicit_with_consistent_board_size_accepted():
    cfg = {"encoding": "v6", "board_size": 19, "in_channels": 8}
    spec = resolve_from_config(cfg)
    assert spec.name == "v6"


def test_v6w25_explicit_with_consistent_keys_accepted():
    cfg = {
        "encoding": "v6w25",
        "board_size": 25,
        "cluster_window_size": 25,
        "cluster_threshold": 8,
        "legal_move_radius": 8,
    }
    spec = resolve_from_config(cfg)
    assert spec.name == "v6w25"


def test_v6w25_explicit_with_disagreeing_board_size_raises():
    cfg = {"encoding": "v6w25", "board_size": 19}
    with pytest.raises(EncodingRegistryError, match=r"board_size=19.*v6w25"):
        resolve_from_config(cfg)


def test_v6_explicit_with_disagreeing_in_channels_raises():
    cfg = {"encoding": "v6", "in_channels": 11}  # v6 has n_planes=8
    with pytest.raises(EncodingRegistryError, match=r"in_channels.*8"):
        resolve_from_config(cfg)


def test_legacy_no_encoding_section_with_consistent_v6_keys_accepted():
    """Legacy configs without `encoding:` key — v6 default — silent if consistent."""
    cfg = {"board_size": 19, "in_channels": 8}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        spec = resolve_from_config(cfg)
    assert spec.name == "v6"


def test_legacy_no_encoding_section_with_disagreeing_keys_warns():
    """Legacy config without explicit encoding but disagreeing keys — DeprecationWarning."""
    cfg = {"board_size": 25}  # disagrees with v6 default
    with pytest.warns(DeprecationWarning, match=r"scattered keys"):
        spec = resolve_from_config(cfg)
    assert spec.name == "v6"  # still resolves to v6 (silent default)


def test_mapping_form_with_consistent_keys_accepted():
    cfg = {"encoding": {"version": "v6w25"}, "board_size": 25}
    spec = resolve_from_config(cfg)
    assert spec.name == "v6w25"


def test_mapping_form_with_disagreeing_keys_raises():
    cfg = {"encoding": {"version": "v6"}, "board_size": 25}
    with pytest.raises(EncodingRegistryError, match=r"board_size=25.*v6"):
        resolve_from_config(cfg)


def test_no_cfg_returns_v6():
    assert resolve_from_config(None).name == "v6"
    assert resolve_from_config({}).name == "v6"


def test_unknown_encoding_name_still_raises():
    with pytest.raises(EncodingRegistryError, match=r"unknown encoding"):
        resolve_from_config({"encoding": "v999"})
