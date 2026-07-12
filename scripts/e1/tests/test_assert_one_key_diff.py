"""T8 D5 — one-key-diff assertion tests.

The E1 pair MUST differ in EXACTLY `model.value_head_type`. run_pair.sh calls
this helper and REFUSES to launch if the resolved diff != that single key.

TDD: authored BEFORE the implementation.
"""
from __future__ import annotations

import pytest

from scripts.e1.assert_one_key_diff import (
    EXPECTED_DIFF_KEY,
    assert_one_key_diff,
    diff_variant_configs,
)


def test_real_pair_diffs_exactly_value_head_type():
    """The real e1_scalar vs e1_dist65 diff == {model.value_head_type}."""
    diff = diff_variant_configs("e1_scalar", "e1_dist65")
    assert set(diff) == {EXPECTED_DIFF_KEY}
    assert diff[EXPECTED_DIFF_KEY] == ("scalar", "dist65")


def test_assert_passes_on_real_pair():
    """assert_one_key_diff returns the diff (no raise) on the real pair."""
    diff = assert_one_key_diff("e1_scalar", "e1_dist65")
    assert set(diff) == {EXPECTED_DIFF_KEY}


def test_assert_rejects_two_key_perturbation(tmp_path, monkeypatch):
    """A 2-key perturbation (a 2nd differing key injected) is REFUSED."""
    import scripts.e1.assert_one_key_diff as mod

    # Perturb: monkeypatch the resolver so the 'b' variant carries an extra
    # differing key beyond model.value_head_type.
    real_resolve = mod._resolve_variant_config

    def fake_resolve(name):
        cfg = real_resolve(name)
        if name == "e1_dist65":
            cfg = dict(cfg)
            cfg["seed"] = 9999  # 2nd differing key (real pair shares seed 1234)
        return cfg

    monkeypatch.setattr(mod, "_resolve_variant_config", fake_resolve)
    with pytest.raises((ValueError, SystemExit), match="(?i)diff|key|seed|value_head_type"):
        assert_one_key_diff("e1_scalar", "e1_dist65")


def test_assert_rejects_zero_key_diff(monkeypatch):
    """Identical configs (0-key diff) are ALSO refused — the arms MUST differ in
    the value head, an identical pair is a mis-clone."""
    import scripts.e1.assert_one_key_diff as mod
    real_resolve = mod._resolve_variant_config

    def fake_resolve(name):
        # Both arms resolve to the scalar config -> 0-key diff.
        return real_resolve("e1_scalar")

    monkeypatch.setattr(mod, "_resolve_variant_config", fake_resolve)
    with pytest.raises((ValueError, SystemExit), match="(?i)diff|key|exactly|value_head_type"):
        assert_one_key_diff("e1_scalar", "e1_dist65")


def test_assert_rejects_wrong_single_key(monkeypatch):
    """A single-key diff on the WRONG key (not model.value_head_type) is refused
    — the invariant is the value head specifically, not 'any one key'."""
    import scripts.e1.assert_one_key_diff as mod
    real_resolve = mod._resolve_variant_config

    def fake_resolve(name):
        cfg = dict(real_resolve("e1_scalar"))  # both start scalar (0-key)
        if name == "e1_dist65":
            cfg["lr"] = 5e-5  # single WRONG-key diff
        return cfg

    monkeypatch.setattr(mod, "_resolve_variant_config", fake_resolve)
    with pytest.raises((ValueError, SystemExit), match="(?i)value_head_type|invariant|exactly"):
        assert_one_key_diff("e1_scalar", "e1_dist65")
