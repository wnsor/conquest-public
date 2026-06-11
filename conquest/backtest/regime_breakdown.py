"""Per-regime decomposition of strategy returns.

Given a regime label time series (from `conquest.regime.RegimeClassifier`),
slice strategy returns by regime and recompute key metrics on each subset.

This answers: *where* does the strategy actually make its money? A strategy
that overall has Sharpe 0.8 might have Sharpe 1.2 in Disinflation and -0.3
in Stagflation. That asymmetry matters for sizing and for diagnosing
overfitting (a strategy that wins by crushing one regime is brittle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.backtest.metrics import (
    sharpe, max_drawdown, hit_rate, annual_return, _drawdown_series,
)


def per_regime_stats(
    returns: pd.Series,
    regime: pd.Series,
    equity: pd.Series | None = None,
) -> dict[str, dict[str, float]]:
    """Decompose returns by regime label.

    Returns:
        dict[regime_label, {n_days, sharpe, hit_rate, annual_return, max_dd}]
        Regimes with fewer than 30 observations get NaN metrics (too noisy).
    """
    aligned = pd.concat(
        [returns.rename("ret"), regime.reindex(returns.index, method="ffill").rename("reg")],
        axis=1,
    ).dropna()

    out: dict[str, dict[str, float]] = {}
    for label, group in aligned.groupby("reg"):
        sub_ret = group["ret"]
        n = len(sub_ret)
        if n < 30:
            out[label] = {
                "n_days": n,
                "sharpe": float("nan"),
                "hit_rate": float("nan"),
                "annual_return": float("nan"),
                "max_dd": float("nan"),
            }
            continue
        # Build the regime-restricted equity curve from compounded returns
        sub_equity = (1 + sub_ret).cumprod()
        out[label] = {
            "n_days": int(n),
            "sharpe": sharpe(sub_ret),
            "hit_rate": hit_rate(sub_ret),
            "annual_return": annual_return(sub_ret),
            "max_dd": max_drawdown(sub_equity),
        }
    return out


def regime_returns(
    returns: pd.Series,
    periods: dict[str, tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Compute period-restricted return stats for explicit calendar windows.

    Used for hedge-overlay anti-overfit gates that target specific stress
    windows (e.g. 2020 COVID 2020-02-19 → 2020-04-30, 2022 bear 2022-01 → 2022-10).
    Different from `per_regime_stats` (which slices by a regime label series);
    here we slice by hand-picked date ranges.

    Args:
        returns: daily returns series.
        periods: dict[label, (start_date, end_date)] where dates are ISO strings
            interpretable by pd.Timestamp. Inclusive on both ends.

    Returns:
        dict[label, {n_days, total_return, sharpe, max_dd}].
    """
    out: dict[str, dict[str, float]] = {}
    for label, (start, end) in periods.items():
        mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        sub = returns[mask]
        n = len(sub)
        if n < 5:
            out[label] = {
                "n_days": n,
                "total_return": float("nan"),
                "sharpe": float("nan"),
                "max_dd": float("nan"),
            }
            continue
        sub_eq = (1 + sub).cumprod()
        out[label] = {
            "n_days": int(n),
            "total_return": float(sub_eq.iloc[-1] - 1.0),
            "sharpe": sharpe(sub),
            "max_dd": max_drawdown(sub_eq),
        }
    return out


def regime_breakdown_table(
    per_model_results: dict[str, "BacktestResult"],   # noqa: F821 (forward ref)
    regime: pd.Series,
) -> pd.DataFrame:
    """Build a long-form DataFrame of per-model × per-regime stats for the bake-off.

    Columns: model, regime, n_days, sharpe, hit_rate, annual_return, max_dd.
    Sorted by (model, regime).
    """
    rows: list[dict] = []
    for model_name, result in per_model_results.items():
        stats = per_regime_stats(result.returns, regime, result.equity)
        for reg_label, vals in stats.items():
            rows.append({"model": model_name, "regime": reg_label, **vals})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["model", "regime"]).reset_index(drop=True)
    return df
