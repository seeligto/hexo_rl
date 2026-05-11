"""Tests: per-function DeprecationWarning on legacy hexo_rl.utils.encoding shim.

§172 A10 — warnings emit at call site (stacklevel=2), not at import time,
so hot-path importers don't spam at startup.
"""
import warnings
import pytest


def test_resolve_encoding_deprecation_warning():
    from hexo_rl.utils.encoding import resolve_encoding
    cfg = {"encoding": "v6"}
    with pytest.warns(DeprecationWarning, match="hexo_rl.encoding.resolve_from_config"):
        resolve_encoding(cfg)


def test_v6_spec_deprecation_warning():
    from hexo_rl.utils.encoding import v6_spec
    with pytest.warns(DeprecationWarning, match="hexo_rl.encoding.lookup"):
        v6_spec()


def test_v6w25_spec_deprecation_warning():
    from hexo_rl.utils.encoding import v6w25_spec
    with pytest.warns(DeprecationWarning, match="hexo_rl.encoding.lookup"):
        v6w25_spec()


def test_v8_spec_deprecation_warning():
    from hexo_rl.utils.encoding import v8_spec
    with pytest.warns(DeprecationWarning, match="hexo_rl.encoding.lookup"):
        v8_spec()
