"""Regression: `_binomial_ci` is a Wilson score interval.

Pins the three canonical cases named in the review (D-005):
- p=0.55 n=200 — the §101.a promotion boundary
- p=0.5  n=200 — null-hypothesis center
- p=1.0 n=10  — boundary case where Wald collapses to zero width

Reference values from `scipy.stats.binomtest(...).proportion_ci(method='wilson')`.
"""

from hexo_rl.eval.eval_pipeline import _binomial_ci


def test_wilson_ci_at_promotion_boundary_p055_n200() -> None:
    lo, hi = _binomial_ci(110, 200)
    assert abs(lo - 0.480756) < 1e-5
    assert abs(hi - 0.617359) < 1e-5


def test_wilson_ci_at_null_p05_n200() -> None:
    lo, hi = _binomial_ci(100, 200)
    assert abs(lo - 0.431361) < 1e-5
    assert abs(hi - 0.568639) < 1e-5


def test_wilson_ci_perfect_small_n_has_positive_spread() -> None:
    lo, hi = _binomial_ci(10, 10)
    assert abs(lo - 0.722467) < 1e-5
    assert hi == 1.0
