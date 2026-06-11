"""Tests for v5 metrics: Beta vs benchmark, Up/Down capture, bootstrap Sharpe CI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.backtest.metrics import (
    beta, up_capture, down_capture, bootstrap_sharpe_ci, metrics_summary, sharpe,
)


@pytest.fixture
def aligned_returns():
    """Construct (strategy, benchmark) where strategy = 0.7×benchmark + idiosyncratic noise.
    True beta should be ~0.7."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    bench = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    ours = 0.7 * bench + pd.Series(rng.normal(0.0001, 0.005, n), index=idx)
    return ours, bench


# ---------- Beta ----------

def test_beta_recovers_known_factor(aligned_returns):
    ours, bench = aligned_returns
    b = beta(ours, bench)
    # True beta is 0.7; sample noise + small N typically lands in [0.5, 0.9]
    assert 0.5 < b < 0.9


def test_beta_nan_on_too_short_series():
    s = pd.Series([0.01, -0.01, 0.02])  # < 30 obs
    assert np.isnan(beta(s, s))


def test_beta_nan_on_zero_variance_benchmark():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    ours = pd.Series(rng.normal(0, 0.01, 100), index=idx)
    flat = pd.Series(0.0, index=idx)
    assert np.isnan(beta(ours, flat))


def test_beta_one_for_self():
    """A series regressed on itself has beta = 1."""
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    s = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    assert beta(s, s) == pytest.approx(1.0, abs=1e-9)


# ---------- Up / Down capture ----------

def test_up_capture_in_sensible_range(aligned_returns):
    ours, bench = aligned_returns
    uc = up_capture(ours, bench)
    assert 0.3 < uc < 1.5


def test_down_capture_in_sensible_range(aligned_returns):
    ours, bench = aligned_returns
    dc = down_capture(ours, bench)
    assert 0.3 < dc < 1.5


def test_capture_returns_nan_on_short_series():
    s = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1, freq="B"))
    assert np.isnan(up_capture(s, s))
    assert np.isnan(down_capture(s, s))


# ---------- Bootstrap CI ----------

def test_bootstrap_ci_brackets_point_estimate(aligned_returns):
    """A 95% CI on the Sharpe should contain the point-estimate Sharpe (almost always)."""
    ours, _ = aligned_returns
    point = sharpe(ours)
    lo, hi = bootstrap_sharpe_ci(ours, n_iter=400, seed=1)
    assert lo <= point <= hi
    # CI shouldn't be absurdly wide
    assert (hi - lo) < 5.0


def test_bootstrap_ci_returns_nan_on_short_series():
    s = pd.Series([0.01, -0.01, 0.02])
    lo, hi = bootstrap_sharpe_ci(s)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_lo_lt_hi(aligned_returns):
    ours, _ = aligned_returns
    lo, hi = bootstrap_sharpe_ci(ours, n_iter=200, seed=2)
    assert lo < hi


# ---------- metrics_summary integration ----------

def test_metrics_summary_includes_v5_keys_with_benchmark(aligned_returns):
    ours, bench = aligned_returns
    equity = (1 + ours).cumprod() * 100_000
    turnover = pd.Series(0.05, index=ours.index)
    m = metrics_summary(ours, equity, turnover,
                        benchmark_returns=bench, bootstrap_n_iter=200)
    for k in ("beta_vs_bench", "up_capture", "down_capture",
              "sharpe_ci_low", "sharpe_ci_high"):
        assert k in m


def test_metrics_summary_omits_relative_keys_without_benchmark(aligned_returns):
    ours, _ = aligned_returns
    equity = (1 + ours).cumprod() * 100_000
    turnover = pd.Series(0.05, index=ours.index)
    m = metrics_summary(ours, equity, turnover,
                        benchmark_returns=None, bootstrap_n_iter=200)
    for k in ("beta_vs_bench", "up_capture", "down_capture"):
        assert k not in m
    # CI keys are always present (don't need a benchmark)
    assert "sharpe_ci_low" in m
    assert "sharpe_ci_high" in m
