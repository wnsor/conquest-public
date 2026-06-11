"""Multiple-testing / Deflated-Sharpe haircut, adapted to per-trade options metrics.

Why this module exists
----------------------
The conquest_options promotion gate (WR ≥ 35% / Expectancy ≥ +15% / PF ≥ 2.0 /
R-mean ≥ +0.5 / Sortino ≥ 2.0 / n ≥ 50) is *necessary but not sufficient*. It says
nothing about whether an edge is real or an artifact of how many strategy variants we
searched. We ran ~40 strategies × many variants (momentum_otm alone went v1→v28); with
that many configs, *something* clears the raw gate by luck. This module implements the
Day-2 handover directive (bias #2, multiple-comparisons):

    "Require the per-trade expectancy t-stat to survive the haircut for N trials.
     Log every variant ever run so N is honest."

It mirrors the fund-level machinery in ``scripts/cross_fund_dsr_v2.py``
(``expected_max_sr``, ``n_eff_vif``) but operates on a **per-trade** return sample
(R-multiples or pnl_pct), not a daily equity curve — options strategies sit in cash
most days, so NAV Sharpe understates the edge and the per-trade sample is the natural
unit of inference.

Method (Bailey & López de Prado 2012/2014)
------------------------------------------
* Per-trade Sharpe  SR = mean(R) / std(R)  (dimensionless; NOT annualized).
* Estimated-Sharpe std under the higher-moment correction:
      sigma(SR) = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (n - 1) )
  where ``kurt`` is the RAW (non-excess) kurtosis (=3 for a Gaussian). This matches
  ``cross_fund_dsr_v2.psr_against`` exactly. (Note: ``conquest.backtest.csa`` uses
  *excess* kurtosis with a ``kurt/4`` term, which drops the Gaussian baseline — that
  is a separate, daily-curve code path and is not reused here.)
* PSR(SR*) = Phi[ (SR - SR*) / sigma(SR) ]  — probability the true per-trade Sharpe
  exceeds the benchmark SR*.
* Deflated benchmark for N trials: under the global null (every trial has zero edge),
  the expected MAXIMUM per-trade Sharpe across N (correlation-adjusted) trials is
      SR* = expected_max_sr(N_eff) / sqrt(n - 1)
  so the DSR reduces to requiring the per-trade t-stat  SR*sqrt(n-1)  to beat the
  expected-max z-score for N_eff trials — precisely the handover's "expectancy t-stat
  must survive the haircut for N trials".

Pure-Python, no pandas/numpy dependency in the core — fully unit-testable on lists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Normal-distribution helpers (Acklam inverse-CDF + Hart CDF; copied verbatim from
# scripts/cross_fund_dsr_v2.py so the two haircuts are numerically identical).
# ──────────────────────────────────────────────────────────────────────────────

def norm_cdf(z: float) -> float:
    if not math.isfinite(z):
        return 1.0 if z > 0 else 0.0
    t = 1 / (1 + 0.2316419 * abs(z))
    d = 0.3989422804 * math.exp(-z * z / 2)
    p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1 - p if z >= 0 else p


def inv_norm(p: float) -> float:
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
    c = [-7.78489400243029e-3, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d_ = [7.78469570904146e-3, 0.32246712907004, 2.445134137143, 3.75440866190742]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1)


# ──────────────────────────────────────────────────────────────────────────────
# Multiple-testing primitives (López de Prado: expected max under the null + VIF).
# ──────────────────────────────────────────────────────────────────────────────

def expected_max_sr(n: float) -> float:
    """Expected maximum of ``n`` i.i.d. standard-normal Sharpe estimates under H0.

    E[max] ≈ (1-γ)·Φ⁻¹(1 - 1/n) + γ·Φ⁻¹(1 - 1/(n·e)), γ = Euler-Mascheroni.
    This is the deflation benchmark (in standard-normal/z units) for ``n`` trials.
    """
    if n is None or n < 2:
        return 0.0
    gamma = 0.5772156649
    return (1 - gamma) * inv_norm(1 - 1 / n) + gamma * inv_norm(1 - 1 / (n * math.e))


def n_eff_vif(n: float, rho: float) -> float:
    """Effective trial count after a variance-inflation-factor correlation haircut.

    ``rho`` is the average pairwise correlation among the ``n`` trials. Highly
    correlated trials (param sweeps of one strategy) count for far less than ``n``.
    """
    if n <= 1 or rho <= 0:
        return n
    return max(1.0, n / (1 + rho * (n - 1)))


def cluster_aware_n_eff(
    clusters: Mapping[str, int],
    rho_within: float,
    rho_across: float,
) -> tuple[float, dict[str, float]]:
    """Two-stage VIF: collapse each cluster's trials to a within-cluster N_eff
    (param sweeps are highly correlated → ``rho_within`` high), then collapse the
    clusters themselves under ``rho_across`` (different architectures → low).

    ``clusters`` maps cluster-name → raw trial count. Returns (n_eff_total, per_cluster).
    """
    per_cluster = {name: n_eff_vif(trials, rho_within) for name, trials in clusters.items()}
    n_clusters = len(clusters)
    if n_clusters == 0:
        return 0.0, per_cluster
    sum_within = sum(per_cluster.values())
    n_clusters_eff = n_eff_vif(n_clusters, rho_across)
    avg_within = sum_within / n_clusters
    return n_clusters_eff * avg_within, per_cluster


# ──────────────────────────────────────────────────────────────────────────────
# Per-trade sample statistics + per-trade PSR / Deflated Sharpe.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeSampleStats:
    """Moments of a per-trade return sample (R-multiples or fractional pnl)."""
    n: int
    mean: float
    std: float          # population std (÷n), matches cross_fund_dsr_v2 convention
    skew: float
    kurt_raw: float     # RAW kurtosis (Gaussian = 3.0)

    @classmethod
    def from_returns(cls, returns: Sequence[float]) -> "TradeSampleStats":
        vals = [float(r) for r in returns if r is not None and math.isfinite(float(r))]
        n = len(vals)
        if n == 0:
            return cls(0, 0.0, 0.0, 0.0, 3.0)
        m = sum(vals) / n
        var = sum((r - m) ** 2 for r in vals) / n
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd <= 0:
            return cls(n, m, 0.0, 0.0, 3.0)
        s3 = sd ** 3
        s4 = sd ** 4
        skew = sum((r - m) ** 3 for r in vals) / n / s3
        kurt_raw = sum((r - m) ** 4 for r in vals) / n / s4
        return cls(n, m, sd, skew, kurt_raw)

    @property
    def sharpe(self) -> float:
        """Per-trade Sharpe = mean / std (dimensionless, NOT annualized)."""
        return self.mean / self.std if self.std > 0 else 0.0

    @property
    def t_stat(self) -> float:
        """Expectancy t-statistic = per-trade Sharpe · sqrt(n-1)."""
        if self.n < 2 or self.std <= 0:
            return 0.0
        return self.sharpe * math.sqrt(self.n - 1)


def _sharpe_std(sr: float, skew: float, kurt_raw: float, n: int) -> float:
    """Bailey-LdP standard error of the per-trade Sharpe estimate."""
    if n < 2:
        return float("inf")
    var_factor = 1.0 - skew * sr + ((kurt_raw - 1.0) / 4.0) * sr * sr
    var_factor = max(1e-12, var_factor)
    return math.sqrt(var_factor / (n - 1))


def psr_per_trade(stats: TradeSampleStats, sr_benchmark: float = 0.0) -> float:
    """Probability the true per-trade Sharpe exceeds ``sr_benchmark`` (per-trade units).

    Bailey-LdP PSR with skew + RAW-kurtosis correction. No annualization — the unit
    of observation is one trade.
    """
    if stats.n < 2 or stats.std <= 0:
        return float("nan")
    sigma = _sharpe_std(stats.sharpe, stats.skew, stats.kurt_raw, stats.n)
    if not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    z = (stats.sharpe - sr_benchmark) / sigma
    return max(0.0, min(1.0, norm_cdf(z)))


@dataclass
class DeflatedResult:
    n_trades: int
    sharpe_per_trade: float
    t_stat: float
    n_trials_raw: float
    n_eff: float
    z_threshold: float          # expected_max_sr(n_eff): the bar the t-stat must beat
    sr_star_per_trade: float    # z_threshold / sqrt(n-1)
    psr_vs_zero: float          # PSR against SR*=0 (no multiple-testing correction)
    dsr: float                  # PSR against the deflated benchmark
    bonferroni_t: float         # t the per-trade Sharpe must beat for p<alpha after N trials
    passes_dsr_95: bool
    passes_bonferroni: bool


def deflated_sharpe_per_trade(
    stats: TradeSampleStats,
    n_trials: float,
    *,
    rho: float = 0.0,
    n_eff: float | None = None,
    alpha: float = 0.05,
    dsr_threshold: float = 0.95,
) -> DeflatedResult:
    """Deflated per-trade Sharpe given ``n_trials`` searched configs.

    Provide either ``n_eff`` directly, or ``n_trials`` + ``rho`` (single-stage VIF).
    """
    if n_eff is None:
        n_eff = n_eff_vif(n_trials, rho) if rho > 0 else n_trials
    z_thr = expected_max_sr(n_eff)
    denom = math.sqrt(stats.n - 1) if stats.n >= 2 else float("inf")
    sr_star = z_thr / denom if math.isfinite(denom) and denom > 0 else float("inf")
    psr0 = psr_per_trade(stats, 0.0)
    dsr = psr_per_trade(stats, sr_star)
    # Bonferroni: per-trade t needed for p < alpha/N_trials (one-sided).
    p_corr = alpha / max(1.0, n_trials)
    bonf_t = inv_norm(1 - p_corr)
    return DeflatedResult(
        n_trades=stats.n,
        sharpe_per_trade=stats.sharpe,
        t_stat=stats.t_stat,
        n_trials_raw=n_trials,
        n_eff=n_eff,
        z_threshold=z_thr,
        sr_star_per_trade=sr_star,
        psr_vs_zero=psr0,
        dsr=dsr,
        bonferroni_t=bonf_t,
        passes_dsr_95=(math.isfinite(dsr) and dsr >= dsr_threshold),
        passes_bonferroni=(stats.t_stat >= bonf_t),
    )


def min_expectancy_to_clear(
    std: float,
    n: int,
    n_trials: float,
    *,
    rho: float = 0.0,
    n_eff: float | None = None,
    target_psr: float = 0.95,
) -> float:
    """Minimum mean per-trade return needed to clear the deflated bar.

    Useful BEFORE a trade journal exists: plug in a plausible per-trade return std and
    sample size to see what expectancy a strategy must show to survive N trials.
    Uses the Gaussian approximation (var_factor ≈ 1) for the threshold, which is
    conservative for the fat-tailed, positive-skew payoff of long OTM calls.
    """
    if n < 2 or std <= 0:
        return float("inf")
    if n_eff is None:
        n_eff = n_eff_vif(n_trials, rho) if rho > 0 else n_trials
    z_thr = expected_max_sr(n_eff)
    sr_min = (z_thr + inv_norm(target_psr)) / math.sqrt(n - 1)
    return sr_min * std


# ──────────────────────────────────────────────────────────────────────────────
# Honest trial-count registry for conquest_options.
#
# This IS the "log every variant ever run so N is honest" artifact the handover asks
# for. Counts are a DOCUMENTED LOWER BOUND sourced from code comments + the reject
# list; the true N is almost certainly higher (the handover says "~40 strategies ×
# many variants"). Under-counting N makes the haircut too LENIENT, so when in doubt
# these err toward the conservative (higher) side. Update as new variants are run.
# ──────────────────────────────────────────────────────────────────────────────

OPTIONS_TRIAL_CLUSTERS: dict[str, dict] = {
    "momentum_otm_iters": {
        "trials": 20,
        "label": "momentum_otm v1→v28 gate-config iterations",
        "source": "momentum_otm_calls.py comments: v1,v3,v4,v8,v8w,v9,v10,v16,v17 + v22 iter v2/v5/v6/v7/v8/v9/v10 + v28(v15d)",
    },
    "dollar1m_overlays_standalone_leaps": {
        "trials": 22,
        "label": "$1M-era options trials (overlays + standalone + LEAPS), deduplicated",
        "source": "CONQUEST_OPTIONS_DO_NOT_RETEST.md §A(9 overlays)+§B(5 standalone)+§C(8 LEAPS)",
    },
    "phase5_framework_invalidated": {
        "trials": 5,
        "label": "Phase-5 minute-noise-invalidated distinct-architecture BTs",
        "source": "DO_NOT_RETEST §E invalidated list (A_GEX v10e, etc.); excludes runs folded into other clusters",
    },
    "other_daily_screened_strategies": {
        "trials": 30,
        "label": "remaining distinct options strategies screened at daily ≥1×",
        "source": "conquest_options/strategies/ (~37 files): im_divergence, dealer_opex, gdelt, insider_cluster, uoa, volume_spike, reflex_ignition v1/v2, ~16 leading-indicator standby, no_catalyst_baseline, etc.",
    },
}


def total_options_trials(clusters: Mapping[str, dict] | None = None) -> int:
    src = clusters if clusters is not None else OPTIONS_TRIAL_CLUSTERS
    return sum(int(c["trials"]) for c in src.values())
