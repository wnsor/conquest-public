"""Performance metrics over the backtest engine's returns / equity / turnover series."""
from __future__ import annotations

import numpy as np
import pandas as pd


_TRADING_DAYS = 252


def annual_return(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return float("nan")
    total = (1 + returns).prod()
    return total ** (_TRADING_DAYS / n) - 1


def annual_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(_TRADING_DAYS)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / _TRADING_DAYS
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return excess.mean() / vol * np.sqrt(_TRADING_DAYS)


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / _TRADING_DAYS
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    dvol = downside.std()
    if dvol == 0 or np.isnan(dvol):
        return float("nan")
    return excess.mean() / dvol * np.sqrt(_TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough drawdown of the equity curve. Returns a non-positive number."""
    if equity.empty:
        return float("nan")
    cummax = equity.cummax()
    return ((equity - cummax) / cummax).min()


def calmar(returns: pd.Series, equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return annual_return(returns) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def turnover_annual_one_way(turnover_per_bar: pd.Series) -> float:
    """Annualized one-way turnover. `turnover_per_bar` is sum of |Δw|."""
    if turnover_per_bar.empty:
        return float("nan")
    avg_per_bar = turnover_per_bar.sum() / len(turnover_per_bar)
    return float(avg_per_bar * _TRADING_DAYS / 2)


# ---------- v4 distribution metrics ----------
#
# Returns are not normal — they're left-skewed and fat-tailed. Sharpe and
# Sortino assume normality, so they overstate strategy quality on heavy left
# tails. CVaR / Omega / skew / kurtosis describe the *shape* of the return
# distribution and surface what those moment-based metrics hide.


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional VaR (Expected Shortfall) at confidence level alpha.

    Average of the worst alpha-fraction of daily returns. Negative number for
    losing tails. Tells you what a "bad day" actually looks like, not just
    its standard deviation."""
    if returns.empty or alpha <= 0 or alpha >= 1:
        return float("nan")
    cutoff = returns.quantile(alpha)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return float("nan")
    return float(tail.mean())


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: sum of returns above `threshold` / sum of |returns| below.

    Equivalent to integrating the survival function around `threshold`. A
    threshold-aware version of Sortino — captures the *whole* distribution
    shape in one number, not just the lower partial moment."""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def skewness(returns: pd.Series) -> float:
    """Sample skewness. Negative = left-skew (bad: rare big losses dominate);
    positive = right-skew (good: rare big gains dominate). Momentum strategies
    typically run negative-skew (catch falling knives during regime shifts)."""
    if returns.empty or returns.std() == 0:
        return float("nan")
    return float(returns.skew())


def excess_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (Fisher definition; normal distribution = 0).
    Positive = fat tails (extreme events more likely than normal predicts)."""
    if returns.empty or returns.std() == 0:
        return float("nan")
    return float(returns.kurtosis())   # pandas kurtosis() returns *excess* kurtosis


# ---------- v4.5 drawdown-shape metrics ----------
#
# Max-drawdown alone is a single point. Two strategies with the same -20% max
# DD can have very different drawdown *experience*: one chronically underwater,
# one occasional shock + fast recovery. These metrics characterise the shape.


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Per-bar drawdown as a fraction (non-positive). 0 = at all-time high."""
    cummax = equity.cummax()
    return (equity - cummax) / cummax


def ulcer_index(equity: pd.Series) -> float:
    """RMS of drawdowns. Penalises deep drawdowns more than shallow ones (squared);
    smoother strategies and lumpy ones with the same max DD will differ here."""
    if equity.empty:
        return float("nan")
    dd = _drawdown_series(equity)
    return float(np.sqrt((dd ** 2).mean()))


def avg_drawdown(equity: pd.Series) -> float:
    """Mean drawdown depth over all bars. Returns a non-positive number."""
    if equity.empty:
        return float("nan")
    dd = _drawdown_series(equity)
    return float(dd.mean())


def time_in_drawdown_pct(equity: pd.Series) -> float:
    """Fraction of bars below the all-time high (i.e., drawdown < 0). Range [0, 1]."""
    if equity.empty:
        return float("nan")
    dd = _drawdown_series(equity)
    return float((dd < 0).mean())


def recovery_factor(equity: pd.Series) -> float:
    """Net profit / |max drawdown|. How many max-DDs of cumulative profit you've made.
    Different angle from Calmar (which is annualized rather than total)."""
    if equity.empty:
        return float("nan")
    net_profit = equity.iloc[-1] - equity.iloc[0]
    mdd = max_drawdown(equity)
    if mdd is None or mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(net_profit / equity.iloc[0] / abs(mdd))


# ---------- v5 benchmark-relative metrics ----------
#
# A strategy can have a great Sharpe just by being levered beta. These metrics
# decompose: how much of our return is just "beta vs SPY"? How much do we
# capture of SPY's up moves vs eat its down moves?


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """OLS regression slope of `returns` on `benchmark_returns`. Beta = 1 means
    we move 1:1 with the benchmark; beta = 0.5 means we get half of its swings."""
    if returns.empty or benchmark_returns.empty:
        return float("nan")
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return float("nan")
    var = aligned.iloc[:, 1].var()
    if var == 0 or np.isnan(var):
        return float("nan")
    cov = aligned.cov().iloc[0, 1]
    return float(cov / var)


def up_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Average monthly return when benchmark is up, divided by benchmark's avg
    up-month return. Higher is better. 1.0 = matching benchmark on up months."""
    monthly = _monthly_pair(returns, benchmark_returns)
    if monthly is None:
        return float("nan")
    up = monthly[monthly["bench"] > 0]
    if up.empty:
        return float("nan")
    bench_avg = up["bench"].mean()
    if bench_avg == 0:
        return float("nan")
    return float(up["us"].mean() / bench_avg)


def down_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Average monthly return when benchmark is down, divided by benchmark's avg
    down-month return. Lower (closer to 0) is better — we eat less of the losses."""
    monthly = _monthly_pair(returns, benchmark_returns)
    if monthly is None:
        return float("nan")
    down = monthly[monthly["bench"] < 0]
    if down.empty:
        return float("nan")
    bench_avg = down["bench"].mean()
    if bench_avg == 0:
        return float("nan")
    return float(down["us"].mean() / bench_avg)


def _monthly_pair(returns: pd.Series, benchmark_returns: pd.Series):
    """Helper: aggregate both daily series to monthly compound returns and align."""
    if returns.empty or benchmark_returns.empty:
        return None
    us_m = (1 + returns).resample("ME").prod() - 1
    bench_m = (1 + benchmark_returns).resample("ME").prod() - 1
    df = pd.concat([us_m, bench_m], axis=1, join="inner").dropna()
    df.columns = ["us", "bench"]
    if len(df) < 6:
        return None
    return df


# ---------- v5 bootstrap confidence intervals ----------
#
# Most of our bake-off has only ~7 years of monthly data — many pairwise
# Sharpe differences in the leaderboard are not statistically distinguishable
# from noise. Bootstrap by resampling daily returns with replacement gives
# error bars so we know which improvements are real.


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_iter: int = 500,
    alpha: float = 0.05,
    seed: int | None = 0,
    block_size: int = 1,
) -> tuple[float, float]:
    """Return (lower, upper) bound of a (1-alpha) bootstrap CI on the Sharpe ratio.

    Args:
        returns: daily returns series.
        n_iter: number of bootstrap iterations (default 500).
        alpha: 1 - confidence level (default 0.05 = 95% CI).
        seed: RNG seed.
        block_size: 1 = iid bootstrap (default, backward-compat). >1 = moving-block
            bootstrap (Politis-Romano), recommended for daily returns to capture
            serial correlation. Typical values: 5 (weekly), 21 (monthly).

    iid bootstrap underestimates Sharpe-ratio uncertainty when daily returns
    have vol clustering / autocorrelation. Use block_size=21 for v8+ work."""
    return bootstrap_metric_ci(
        returns=returns,
        equity=None,
        metric_fn=lambda r, e: sharpe(r),
        n_iter=n_iter,
        alpha=alpha,
        seed=seed,
        block_size=block_size,
    )


def bootstrap_metric_ci(
    returns: pd.Series,
    equity: pd.Series | None,
    metric_fn,
    n_iter: int = 500,
    alpha: float = 0.05,
    seed: int | None = 0,
    block_size: int = 1,
) -> tuple[float, float]:
    """Generalized bootstrap CI for any return-based metric.

    Args:
        returns: daily returns series (required).
        equity: optional equity curve. If the metric needs it (e.g. Calmar,
            MaxDD), the bootstrap *recomputes* equity from the resampled
            returns: equity_resampled = (1 + returns_resampled).cumprod().
            This means the metric_fn signature is `(returns, equity) -> float`.
        metric_fn: callable accepting (resampled_returns: pd.Series,
            resampled_equity: pd.Series | None) returning a float.
        n_iter, alpha, seed, block_size: see `bootstrap_sharpe_ci`.

    Returns:
        (lower, upper) percentile bounds at confidence (1 - alpha).
    """
    if returns.empty or len(returns) < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(returns)
    arr = returns.to_numpy()
    initial_capital = float(equity.iloc[0]) if equity is not None and len(equity) > 0 else 1.0

    vals: list[float] = []
    if block_size <= 1:
        idx_template = np.arange(n)
        for _ in range(n_iter):
            sel = rng.integers(0, n, size=n)
            sample = pd.Series(arr[sel], index=returns.index)
            sample_eq = (1 + sample).cumprod() * initial_capital if equity is not None else None
            try:
                v = float(metric_fn(sample, sample_eq))
            except Exception:
                continue
            if not np.isnan(v) and not np.isinf(v):
                vals.append(v)
    else:
        n_blocks = (n + block_size - 1) // block_size
        max_start = n - block_size + 1
        if max_start <= 0:
            return (float("nan"), float("nan"))
        for _ in range(n_iter):
            starts = rng.integers(0, max_start, size=n_blocks)
            sample_arr = np.concatenate([arr[s:s + block_size] for s in starts])[:n]
            sample = pd.Series(sample_arr, index=returns.index)
            sample_eq = (1 + sample).cumprod() * initial_capital if equity is not None else None
            try:
                v = float(metric_fn(sample, sample_eq))
            except Exception:
                continue
            if not np.isnan(v) and not np.isinf(v):
                vals.append(v)

    if not vals:
        return (float("nan"), float("nan"))
    a = np.array(vals)
    return (
        float(np.percentile(a, 100 * alpha / 2)),
        float(np.percentile(a, 100 * (1 - alpha / 2))),
    )


def metrics_summary(
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    benchmark_returns: pd.Series | None = None,
    bootstrap_n_iter: int = 500,
    weights: pd.DataFrame | None = None,
    bootstrap_block_size: int = 1,
) -> dict:
    out: dict = {
        "annual_return":    annual_return(returns),
        "annual_vol":       annual_vol(returns),
        "sharpe":           sharpe(returns),
        "sortino":          sortino(returns),
        "max_drawdown":     max_drawdown(equity),
        "calmar":           calmar(returns, equity),
        "hit_rate":         hit_rate(returns),
        "profit_factor":    profit_factor(returns),
        "turnover_annual":  turnover_annual_one_way(turnover),
        # v4 distribution metrics
        "cvar_5pct":        cvar(returns, alpha=0.05),
        "omega_ratio":      omega_ratio(returns, threshold=0.0),
        "skewness":         skewness(returns),
        "excess_kurtosis":  excess_kurtosis(returns),
        # v4.5 drawdown-shape metrics
        "ulcer_index":      ulcer_index(equity),
        "avg_drawdown":     avg_drawdown(equity),
        "time_in_dd_pct":   time_in_drawdown_pct(equity),
        "recovery_factor":  recovery_factor(equity),
    }
    # v5 — benchmark-relative (only computed if benchmark provided)
    if benchmark_returns is not None:
        out["beta_vs_bench"] = beta(returns, benchmark_returns)
        out["up_capture"] = up_capture(returns, benchmark_returns)
        out["down_capture"] = down_capture(returns, benchmark_returns)
    # v5 — bootstrap CI on Sharpe (always computed; ~500 iters is fast).
    # v8 — block_size>1 enables moving-block bootstrap to capture daily-return
    # autocorrelation (vol clustering); use 21 (one trading month) for new work.
    sharpe_lo, sharpe_hi = bootstrap_sharpe_ci(
        returns, n_iter=bootstrap_n_iter, block_size=bootstrap_block_size,
    )
    out["sharpe_ci_low"] = sharpe_lo
    out["sharpe_ci_high"] = sharpe_hi
    # v5.5 — concentration (only if weights provided)
    if weights is not None:
        from conquest.backtest.concentration import (
            avg_hhi, max_hhi, avg_effective_n, max_single_name_weight,
        )
        out["hhi_avg"] = avg_hhi(weights)
        out["hhi_max"] = max_hhi(weights)
        out["effective_n_avg"] = avg_effective_n(weights)
        out["max_single_weight"] = max_single_name_weight(weights)
    return out
