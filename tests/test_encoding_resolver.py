"""Tests for hexo_rl.utils.encoding.resolve_encoding."""
from __future__ import annotations

import pytest

from hexo_rl.utils.encoding import (
    EncodingSpec,
    resolve_encoding,
    v6_spec,
    v8_spec,
)


def test_default_resolves_to_v6() -> None:
    spec = resolve_encoding({})
    assert spec.version == "v6"
    assert spec.board_size == 19
    assert spec.n_cells == 361
    assert spec.n_actions == 362
    assert spec.n_planes == 8
    assert spec.legal_move_radius == 5
    assert spec.cluster_threshold == 5
    assert spec.state_stride == 8 * 361
    assert spec.policy_stride == 362
    assert spec.has_pass_slot is True


def test_missing_encoding_section_resolves_to_v6() -> None:
    spec = resolve_encoding({"unrelated_key": 42})
    assert spec.version == "v6"


def test_explicit_v6() -> None:
    spec = resolve_encoding({"encoding": {"version": "v6"}})
    assert spec == v6_spec()


def test_explicit_v8() -> None:
    spec = resolve_encoding({"encoding": {"version": "v8"}})
    assert spec.version == "v8"
    assert spec.board_size == 25
    assert spec.half == 12
    assert spec.n_cells == 625
    assert spec.n_actions == 625  # no pass slot under v8
    assert spec.n_planes == 11
    assert spec.legal_move_radius == 8
    assert spec.cluster_threshold is None
    assert spec.state_stride == 11 * 625
    assert spec.policy_stride == 625
    assert spec.aux_stride == 625
    assert spec.has_pass_slot is False
    assert spec == v8_spec()


def test_unknown_version_raises() -> None:
    with pytest.raises(ValueError, match="unknown encoding.version"):
        resolve_encoding({"encoding": {"version": "v7"}})


def test_non_mapping_section_raises() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_encoding({"encoding": "v8"})


def test_v6_v8_specs_distinct() -> None:
    v6 = v6_spec()
    v8 = v8_spec()
    assert v6 != v8
    # Spatial extent differs in both directions
    assert v8.board_size > v6.board_size
    assert v8.n_planes > v6.n_planes
    # v8 drops pass slot — strictly fewer actions per cell
    assert v8.n_actions == v8.n_cells
    assert v6.n_actions == v6.n_cells + 1


def test_specs_are_namedtuples() -> None:
    """EncodingSpec should be a NamedTuple (immutable, hashable, comparable)."""
    spec = v6_spec()
    assert isinstance(spec, EncodingSpec)
    # Hashable
    {spec}
    # Frozen (cannot reassign field)
    with pytest.raises(AttributeError):
        spec.board_size = 99  # type: ignore[misc]


def test_real_model_yaml_loads_v6() -> None:
    """Wire test: configs/model.yaml default loads as v6."""
    import yaml
    import pathlib

    cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "model.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    spec = resolve_encoding(cfg)
    assert spec.version == "v6"
