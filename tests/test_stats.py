"""The statistics layer is where a wrong answer turns into a wrong diagnosis."""

from __future__ import annotations

import numpy as np
import pytest

from rldoctor import stats


def test_theil_sen_recovers_a_known_slope():
    x = np.arange(100, dtype=float)
    y = 3.0 * x + 7.0
    slope, intercept = stats.theil_sen(x, y)
    assert slope == pytest.approx(3.0, abs=1e-9)
    assert intercept == pytest.approx(7.0, abs=1e-6)


def test_theil_sen_survives_catastrophic_outliers():
    """The whole point of a median-of-slopes estimator: a few insane steps
    must not flip the reported trend."""
    x = np.arange(100, dtype=float)
    y = 3.0 * x + 7.0
    y[[10, 20, 30, 40]] = 1e6  # four exploded steps
    slope, _ = stats.theil_sen(x, y)
    assert slope == pytest.approx(3.0, rel=0.05)

    ols_slope = np.polyfit(x, y, 1)[0]
    assert abs(ols_slope - 3.0) > 100, "OLS should be wrecked; that is why we do not use it"


def test_mann_kendall_detects_monotone_trend_and_ignores_noise():
    rng = np.random.default_rng(0)
    trending = np.arange(60, dtype=float) + rng.normal(0, 0.5, 60)
    p_trend, tau_trend = stats.mann_kendall(trending)
    assert p_trend < 0.001
    assert tau_trend > 0.8

    flat = rng.normal(0, 1, 60)
    p_flat, _ = stats.mann_kendall(flat)
    assert p_flat > 0.05


def test_mann_kendall_handles_ties_without_dividing_by_zero():
    constant = np.zeros(50)
    p, tau = stats.mann_kendall(constant)
    assert p == 1.0
    assert tau == 0.0


def test_trend_direction_requires_significance():
    """A slope with no statistical support must report 'flat', not a direction."""
    rng = np.random.default_rng(1)
    x = np.arange(40, dtype=float)
    y = rng.normal(0, 1, 40)
    assert stats.trend(x, y).direction == "flat"


def test_spearman_matches_known_values():
    a = np.array([1.0, 2, 3, 4, 5])
    assert stats.spearman(a, a) == pytest.approx(1.0)
    assert stats.spearman(a, a[::-1]) == pytest.approx(-1.0)
    # Monotone but very nonlinear: Pearson would under-report, Spearman must not.
    assert stats.spearman(a, a**5) == pytest.approx(1.0)


def test_rankdata_averages_ties():
    ranks = stats._rankdata(np.array([10.0, 20.0, 20.0, 30.0]))
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_mad_is_a_consistent_sigma_estimator():
    rng = np.random.default_rng(2)
    sample = rng.normal(0, 3.0, 20000)
    assert stats.mad(sample) == pytest.approx(3.0, rel=0.05)


def test_cusum_finds_an_injected_level_shift():
    x = np.arange(200, dtype=float)
    y = np.concatenate([np.ones(100), np.full(100, 5.0)])
    y += np.random.default_rng(3).normal(0, 0.05, 200)
    change = stats.cusum_changepoint(x, y)
    assert change is not None
    assert 90 <= change.step <= 110
    assert change.is_drop is False


def test_cusum_stays_quiet_on_a_smooth_series():
    x = np.arange(200, dtype=float)
    y = np.linspace(0, 1, 200) + np.random.default_rng(4).normal(0, 0.01, 200)
    assert stats.cusum_changepoint(x, y, min_sigma=3.0) is None


def test_project_crossing_refuses_to_extrapolate_the_wrong_way():
    x = np.arange(50, dtype=float)
    falling = stats.trend(x, 10.0 - 0.1 * x)
    # Heading down towards 1.0: a crossing exists.
    assert stats.project_crossing(falling, 49.0, 5.0, 1.0) == pytest.approx(89.0, abs=1.0)
    # Heading down, threshold above: no crossing, and we must not invent one.
    assert stats.project_crossing(falling, 49.0, 5.0, 9.0) is None


def test_fmt_p_never_prints_zero():
    assert stats.fmt_p(0.0) == "<1e-16"
    assert stats.fmt_p(0.03).startswith("=")


def test_ewma_tolerates_nans():
    out = stats.ewma(np.array([1.0, np.nan, 3.0]))
    assert np.isfinite(out[0]) and np.isfinite(out[2])
