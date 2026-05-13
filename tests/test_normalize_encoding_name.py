"""Tests for hexo_rl.encoding.normalize_encoding_name (§175 eval-fix)."""
from __future__ import annotations

import pytest

from hexo_rl.encoding import (
    EncodingRegistryError,
    lookup,
    normalize_encoding_name,
)


def test_string_passthrough_v6() -> None:
    assert normalize_encoding_name("v6") == "v6"


def test_string_passthrough_v6w25() -> None:
    assert normalize_encoding_name("v6w25") == "v6w25"


def test_dict_version_key() -> None:
    assert normalize_encoding_name({"version": "v6"}) == "v6"


def test_dict_version_key_extra_fields() -> None:
    assert normalize_encoding_name({"version": "v7full", "board_size": 19}) == "v7full"


def test_dict_name_key_fallback() -> None:
    assert normalize_encoding_name({"name": "v6w25"}) == "v6w25"


def test_dict_name_preferred_over_version() -> None:
    assert normalize_encoding_name({"name": "v6", "version": "v8"}) == "v6"


def test_spec_object() -> None:
    spec = lookup("v6")
    assert normalize_encoding_name(spec) == "v6"


def test_none_default_v6() -> None:
    assert normalize_encoding_name(None) == "v6"


def test_dict_empty_defaults_v6() -> None:
    assert normalize_encoding_name({}) == "v6"


def test_dict_non_string_version_rejected() -> None:
    with pytest.raises(EncodingRegistryError):
        normalize_encoding_name({"version": 6})


def test_unsupported_type_rejected() -> None:
    with pytest.raises(EncodingRegistryError):
        normalize_encoding_name(42)


def test_lookup_after_normalize_succeeds_for_dict_form() -> None:
    """End-to-end: §175 crash repro path."""
    enc = {"version": "v6"}
    spec = lookup(normalize_encoding_name(enc))
    assert spec.name == "v6"
