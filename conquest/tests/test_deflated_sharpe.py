"""Tests for the per-trade Deflated-Sharpe / multiple-testing haircut.

Pure-Python module, so these run on hand-built lists with no numpy/pandas/QC
dependency. Samples are constructed deterministically (two-point symmetric
patterns) so the mean / std / Sharpe / t-stat are exact and the pass/fail
thresholds are predictable rather than fitted to whatever the code emits.
"""
from __future__ import annotations

import math

import pytest

from conquest.backtest.deflated_sharpe import (
    OPTIONS_TRIAL_CLUSTERS,
    DeflatedResult,
    TradeSampleStats,
    cluster_aware_n_eff,
    deflated_sharpe_per_trade,
    expected_max_sr,
    min_expectancy_to_clear,
    n_eff_vif,
    norm_cdf,
    psr_per_trade,
    total_options_trials,
)


# ── normal helpers ──────────────────────────────────────────────────────────

def test_norm_cdf_anchors():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-6)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)
    assert norm_cdf(float("inf")) == 1.0
    assert norm_cdf(float("-inf")) == 0.0


# ── expected_max_sr: the deflation z-benchmark ───────────────────────────────

def test_expected_max_sr_zero_below_two_trials():
    assert expected_max_sr(1) == 0.0
    assert expected_max_sr(0) == 0.0
    assert expected_max_sr(None) == 0.0


def test_expected_max_sr_monotonic_increasing():
    seq = [expected_max_sr(n) for n in (2, 5, 20, 77, 300, 1000)]
    assert all(b > a for a, b in zip(seq, seq[1:])), seq
    # More trials => higher bar to clear under the global null.
    assert expected_max_sr(300) > expected_max_sr(77) > 0.0


# ── n_eff_vif: correlation haircut ───────────────────────────────────────────

def test_n_eff_vif_no_haircut_when_uncorrelated():
    assert n_eff_vif(50, 0.0) == 50
    assert n_eff_vif(50, -0.1) == 50  # negative rho ignored


def test_n_eff_vif_reduces_correlated_trials():
    # 20 param-sweep trials at rho=0.5 collapse to ~1.9 independent trials.
    assert n_eff_vif(20, 0.5) == pytest.approx(20 / (1 + 0.5 * 19), rel=1e-9)
    assert n_eff_vif(20, 0.5) < 20
    # Floor at 1.0 — never below a single trial.
    assert n_eff_vif(1000, 0.99) >= 1.0


def test_cluster_aware_n_eff_two_stage():
    clusters = {"a": 20, "b": 22, "c": 30}
    n_eff_total, per_cluster = cluster_aware_n_eff(clusters, rho_within=0.7, rho_across=0.2)
    assert set(per_cluster) == {"a", "b", "c"}
    # Each cluster collapses well below its raw trial count at high within-rho.
    for name, raw in clusters.items():
        assert per_cluster[name] < raw
    # Total effective count is positive and below the raw sum (72).
    assert 0.0 < n_eff_total < sum(clusters.values())


def test_cluster_aware_n_eff_empty():
    n_eff_total, per_cluster = cluster_aware_n_eff({}, 0.5, 0.2)
    assert n_eff_total == 0.0
    assert per_cluster == {}


# ── TradeSampleStats.from_returns: moments ───────────────────────────────────

def test_from_returns_basic_moments():
    # Two-point symmetric pattern: mean exact, population std exact, skew 0.
    stats = TradeSampleStats.from_returns([0.6, -0.2] * 50)
    assert stats.n == 100
    assert stats.mean == pytest.approx(0.2, abs=1e-12)
    assert stats.std == pytest.approx(0.4, abs=1e-12)  # population std (÷n)
    assert stats.skew == pytest.approx(0.0, abs=1e-12)
    assert stats.sharpe == pytest.approx(0.5, abs=1e-12)
    # t-stat = sharpe * sqrt(n-1)
    assert stats.t_stat == pytest.approx(0.5 * math.sqrt(99), rel=1e-9)


def test_from_returns_filters_nonfinite_and_none():
    stats = TradeSampleStats.from_returns([0.1, None, float("nan"), float("inf"), 0.3])
    assert stats.n == 2
    assert stats.mean == pytest.approx(0.2, abs=1e-12)


def test_from_returns_empty_and_constant():
    empty = TradeSampleStats.from_returns([])
    assert empty.n == 0 and empty.std == 0.0 and empty.kurt_raw == 3.0
    assert empty.sharpe == 0.0 and empty.t_stat == 0.0

    const = TradeSampleStats.from_returns([5.0, 5.0, 5.0])
    assert const.n == 3 and const.std == 0.0
    assert const.sharpe == 0.0 and const.t_stat == 0.0


# ── psr_per_trade ────────────────────────────────────────────────────────────

def test_psr_half_for_zero_mean_sample():
    # Symmetric, mean exactly zero => Sharpe 0 => PSR vs 0 == 0.5.
    stats = TradeSampleStats.from_returns([0.4, -0.4] * 50)
    assert stats.sharpe == pytest.approx(0.0, abs=1e-12)
    assert psr_per_trade(stats, 0.0) == pytest.approx(0.5, abs=1e-6)


def test_psr_high_for_strong_positive_edge():
    stats = TradeSampleStats.from_returns([0.6, -0.2] * 50)  # Sharpe 0.5, n=100
    assert psr_per_trade(stats, 0.0) > 0.99


def test_psr_nan_for_degenerate_sample():
    assert math.isnan(psr_per_trade(TradeSampleStats.from_returns([5.0, 5.0]), 0.0))
    assert math.isnan(psr_per_trade(TradeSampleStats.from_returns([1.0]), 0.0))


# ── deflated_sharpe_per_trade: the headline haircut ──────────────────────────

def test_deflated_strong_edge_passes_both():
    # Sharpe 0.5, n=100, searched against 77 trials.
    stats = TradeSampleStats.from_returns([0.6, -0.2] * 50)
    res = deflated_sharpe_per_trade(stats, n_trials=77)
    assert isinstance(res, DeflatedResult)
    assert res.n_eff == 77  # rho=0 => no VIF haircut
    assert res.z_threshold == pytest.approx(expected_max_sr(77), rel=1e-9)
    # t-stat ~4.97 must clear both the DSR bar and Bonferroni.
    assert res.passes_dsr_95 is True
    assert res.passes_bonferroni is True
    assert res.dsr >= 0.95


def test_deflated_weak_edge_fails_both():
    # Sharpe 0.05, n=60 — a real but tiny edge that should NOT survive 77 trials.
    stats = TradeSampleStats.from_returns([0.42, -0.38] * 30)
    assert stats.sharpe == pytest.approx(0.05, abs=1e-12)
    res = deflated_sharpe_per_trade(stats, n_trials=77)
    assert res.passes_dsr_95 is False
    assert res.passes_bonferroni is False
    assert res.dsr < 0.95


def test_deflated_psr_vs_zero_dominates_dsr():
    # PSR against benchmark 0 is always >= PSR against the (positive) deflated bar.
    stats = TradeSampleStats.from_returns([0.6, -0.2] * 50)
    res = deflated_sharpe_per_trade(stats, n_trials=300)
    assert res.psr_vs_zero >= res.dsr
    assert res.sr_star_per_trade > 0.0


def test_deflated_more_trials_is_stricter():
    stats = TradeSampleStats.from_returns([0.5, -0.2] * 40)  # moderate edge
    lenient = deflated_sharpe_per_trade(stats, n_trials=10)
    strict = deflated_sharpe_per_trade(stats, n_trials=1000)
    assert strict.z_threshold > lenient.z_threshold
    assert strict.dsr <= lenient.dsr
    assert strict.bonferroni_t > lenient.bonferroni_t


def test_deflated_n_eff_override_beats_n_trials():
    stats = TradeSampleStats.from_returns([0.6, -0.2] * 50)
    explicit = deflated_sharpe_per_trade(stats, n_trials=300, n_eff=5.0)
    assert explicit.n_eff == 5.0
    assert explicit.z_threshold == pytest.approx(expected_max_sr(5.0), rel=1e-9)


def test_deflated_rho_haircut_lowers_bar():
    stats = TradeSampleStats.from_returns([0.6, -0.2] * 50)
    no_corr = deflated_sharpe_per_trade(stats, n_trials=50, rho=0.0)
    correlated = deflated_sharpe_per_trade(stats, n_trials=50, rho=0.6)
    assert correlated.n_eff < no_corr.n_eff
    assert correlated.z_threshold < no_corr.z_threshold


# ── min_expectancy_to_clear: pre-journal sizing of the bar ───────────────────

def test_min_expectancy_positive_and_finite():
    bar = min_expectancy_to_clear(std=0.4, n=60, n_trials=77)
    assert math.isfinite(bar) and bar > 0.0


def test_min_expectancy_increases_with_trials():
    bar_few = min_expectancy_to_clear(std=0.4, n=60, n_trials=30)
    bar_many = min_expectancy_to_clear(std=0.4, n=60, n_trials=300)
    assert bar_many > bar_few


def test_min_expectancy_infinite_for_degenerate_inputs():
    assert min_expectancy_to_clear(std=0.4, n=1, n_trials=77) == float("inf")
    assert min_expectancy_to_clear(std=0.0, n=60, n_trials=77) == float("inf")


# ── trial-count registry: the "log every variant" artifact ───────────────────

def test_trial_registry_total_is_documented_77():
    assert total_options_trials() == 77
    assert sum(int(c["trials"]) for c in OPTIONS_TRIAL_CLUSTERS.values()) == 77


def test_trial_registry_entries_carry_source_provenance():
    for name, cluster in OPTIONS_TRIAL_CLUSTERS.items():
        assert cluster["trials"] > 0, name
        assert cluster["source"], f"{name} missing provenance source"
        assert cluster["label"], f"{name} missing label"


def test_total_options_trials_accepts_custom_clusters():
    custom = {"x": {"trials": 3}, "y": {"trials": 4}}
    assert total_options_trials(custom) == 7
