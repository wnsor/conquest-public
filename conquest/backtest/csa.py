"""Conquest Standard Allocation (CSA) — pandas-side portfolio combiner.

Inputs:
    - cstability + cgrowth equity curves (JSON: {"values": [{"date","equity"}, ...]})
    - auxiliary asset price parquets (close-only, daily): GLD, BIL, GBTC, BITO, IBIT
    - target weights and a crypto bridge spec (which proxy is active in each window)

Output:
    - Daily CSA portfolio equity curve, per-sleeve dollar tracks, and metrics.

Rebalance rule:
    Per-sleeve dollar trackers compounded by their own daily return. On the first
    NYSE trading day of each month, redistribute the total to target weights at
    that day's open. Day order is: apply the day's return through end-of-day,
    THEN if the *next* day is a rebalance trigger, reallocate at next-open.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from conquest.backtest.metrics import (
    annual_return,
    max_drawdown,
    sharpe,
)


@dataclass
class CryptoBridgeSegment:
    start: str
    end_exclusive: str
    proxy: str


@dataclass
class CSAConfig:
    fund_curves: dict[str, Path]
    aux_prices: dict[str, Path]
    weights: dict[str, float]
    crypto_bridge: list[CryptoBridgeSegment]
    start_capital: float = 50_000.0
    window: tuple[str, str] = ("2008-01-01", "2026-05-04")
    rebalance: str = "monthly_first_trading_day"

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1.0; got {total}")
        required_sleeves = {"cgrowth", "cstability", "crypto", "GLD", "BIL"}
        missing = required_sleeves - self.weights.keys()
        if missing:
            raise ValueError(f"weights missing sleeves: {sorted(missing)}")


@dataclass
class CSAResult:
    equity: pd.Series
    sleeve_values: pd.DataFrame
    rebalance_dates: list[pd.Timestamp]
    crypto_proxy_map: list[dict]
    metrics: dict
    config: CSAConfig = field(repr=False)

    def to_jsonable(self) -> dict:
        return {
            "config": {
                "weights": self.config.weights,
                "start_capital": self.config.start_capital,
                "window": list(self.config.window),
                "rebalance": self.config.rebalance,
                "crypto_bridge": [
                    {"start": s.start, "end_exclusive": s.end_exclusive, "proxy": s.proxy}
                    for s in self.config.crypto_bridge
                ],
            },
            "equity": [
                {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
                for d, v in self.equity.items()
            ],
            "sleeves": {
                col: [
                    {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
                    for d, v in self.sleeve_values[col].items()
                ]
                for col in self.sleeve_values.columns
            },
            "crypto_proxy_map": self.crypto_proxy_map,
            "rebalance_dates": [d.strftime("%Y-%m-%d") for d in self.rebalance_dates],
            "metrics": self.metrics,
        }


def load_fund_returns(curve_path: Path) -> pd.Series:
    """Load a *_lean_equity.json file and return daily returns.

    The exported equity curves carry one record per trading day (latest equity
    of that UTC day). pct_change yields daily returns; first row is NaN.
    """
    data = json.loads(Path(curve_path).read_text())
    rows = data["values"]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(
        {pd.Timestamp(r["date"]): float(r["equity"]) for r in rows}
    ).sort_index()
    rets = s.pct_change().dropna()
    rets.name = data.get("fund", curve_path.stem)
    return rets


def _load_close_parquet(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "close" in df.columns:
        s = df["close"]
    elif "Close" in df.columns:
        s = df["Close"]
    else:
        # single-column parquet
        s = df.iloc[:, 0]
    if not isinstance(s.index, pd.DatetimeIndex):
        if "date" in df.columns:
            s.index = pd.to_datetime(df["date"])
        else:
            s.index = pd.to_datetime(s.index)
    return s.sort_index().astype(float)


def load_aux_returns(
    aux_prices: dict[str, Path],
    crypto_bridge: list[CryptoBridgeSegment],
) -> tuple[pd.DataFrame, list[dict]]:
    """Build a returns DataFrame with columns [GLD, BIL, crypto].

    `crypto` is the bridge-active proxy's daily return on each date. Outside any
    bridge segment (e.g., before the first segment's start), crypto returns are
    NaN — the caller is expected to align this against a window where bridges
    cover every date.

    Returns:
        (returns_df, crypto_proxy_map)
        crypto_proxy_map records {"date": "...", "proxy": "BIL|GBTC|BITO|IBIT"}
        for the first date each proxy becomes active (compact — boundaries only).
    """
    gld_close = _load_close_parquet(aux_prices["GLD"])
    bil_close = _load_close_parquet(aux_prices["BIL"])

    gld_ret = gld_close.pct_change()
    bil_ret = bil_close.pct_change()

    proxy_returns: dict[str, pd.Series] = {}
    for seg in crypto_bridge:
        proxy = seg.proxy
        if proxy in proxy_returns:
            continue
        if proxy == "BIL":
            proxy_returns["BIL"] = bil_ret
        else:
            close = _load_close_parquet(aux_prices[proxy])
            proxy_returns[proxy] = close.pct_change()

    # Build the crypto column by stitching bridge segments.
    crypto_pieces: list[pd.Series] = []
    crypto_proxy_map: list[dict] = []
    for seg in crypto_bridge:
        start = pd.Timestamp(seg.start)
        end_exclusive = pd.Timestamp(seg.end_exclusive)
        proxy_ret = proxy_returns[seg.proxy]
        seg_slice = proxy_ret[(proxy_ret.index >= start) & (proxy_ret.index < end_exclusive)]
        crypto_pieces.append(seg_slice)
        if not seg_slice.empty:
            crypto_proxy_map.append(
                {"date": seg_slice.index[0].strftime("%Y-%m-%d"), "proxy": seg.proxy}
            )

    crypto_ret = pd.concat(crypto_pieces).sort_index()
    crypto_ret = crypto_ret[~crypto_ret.index.duplicated(keep="first")]

    # Common date index from union of all three.
    idx = gld_ret.index.union(bil_ret.index).union(crypto_ret.index)
    df = pd.DataFrame(index=idx)
    df["GLD"] = gld_ret.reindex(idx)
    df["BIL"] = bil_ret.reindex(idx)
    df["crypto"] = crypto_ret.reindex(idx)
    return df, crypto_proxy_map


def align_returns(
    fund_returns: dict[str, pd.Series],
    aux_returns: pd.DataFrame,
    window: tuple[str, str],
) -> pd.DataFrame:
    """Outer-join fund + aux returns, restrict to window, fill missing aux days
    with 0.0 (no-op return on weekends/holidays where one source has data).
    Funds drive the trading calendar; aux returns are aligned to fund dates."""
    start = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1])
    fund_df = pd.DataFrame(fund_returns)
    df = fund_df.join(aux_returns, how="left")
    df = df[(df.index >= start) & (df.index <= end)]
    # Aux columns may be NaN if fund index has dates the aux source skipped or
    # if the bridge segment is misaligned — treat as 0.0 (no return that day).
    for col in ("GLD", "BIL", "crypto"):
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    df[list(fund_returns.keys())] = df[list(fund_returns.keys())].fillna(0.0)
    return df


def first_trading_days_of_month(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First date in `index` for each (year, month) pair, in order."""
    if len(index) == 0:
        return []
    seen: set[tuple[int, int]] = set()
    out: list[pd.Timestamp] = []
    for ts in index:
        ym = (ts.year, ts.month)
        if ym not in seen:
            seen.add(ym)
            out.append(ts)
    return out


def simulate(returns: pd.DataFrame, cfg: CSAConfig) -> CSAResult:
    """Run the daily-return × monthly-rebalance simulation.

    The returns DataFrame is expected to have one row per trading day in the
    window, with columns matching the sleeve names in cfg.weights.
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty")
    sleeves = list(cfg.weights.keys())
    missing = [s for s in sleeves if s not in returns.columns]
    if missing:
        raise ValueError(f"returns DataFrame missing sleeve columns: {missing}")

    capital = cfg.start_capital
    sleeve_values = {s: capital * cfg.weights[s] for s in sleeves}

    rebalance_set = set(first_trading_days_of_month(returns.index))
    # First trading day in the window is implicitly the seed allocation day, not a
    # rebalance trigger (capital is already at target weights).
    first_day = returns.index[0]
    rebalance_set.discard(first_day)

    equity_records: list[tuple[pd.Timestamp, float]] = []
    sleeve_records: dict[str, list[float]] = {s: [] for s in sleeves}
    rebalance_log: list[pd.Timestamp] = []

    for ts in returns.index:
        # 1. Apply this day's return to each sleeve.
        for s in sleeves:
            r = returns.at[ts, s]
            sleeve_values[s] *= (1.0 + r)
        portfolio_value = sum(sleeve_values.values())
        equity_records.append((ts, portfolio_value))
        for s in sleeves:
            sleeve_records[s].append(sleeve_values[s])

        # 2. If THIS day is a first-trading-day-of-month and it's not the seed
        #    day, redistribute at end-of-day for the next day forward. (Matches
        #    the convention: monthly rebalance "as of" the first trading day.)
        if ts in rebalance_set:
            for s in sleeves:
                sleeve_values[s] = portfolio_value * cfg.weights[s]
            rebalance_log.append(ts)

    equity = pd.Series({d: v for d, v in equity_records}, name="csa_equity")
    sleeve_df = pd.DataFrame(sleeve_records, index=equity.index)

    metrics = compute_metrics(equity)
    # Rebuild the crypto proxy map from the bridge spec for downstream archival.
    proxy_map: list[dict] = []
    for seg in cfg.crypto_bridge:
        proxy_map.append({"start": seg.start, "end_exclusive": seg.end_exclusive, "proxy": seg.proxy})

    return CSAResult(
        equity=equity,
        sleeve_values=sleeve_df,
        rebalance_dates=rebalance_log,
        crypto_proxy_map=proxy_map,
        metrics=metrics,
        config=cfg,
    )


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & López de Prado 2012): probability the true Sharpe exceeds
    `sr_benchmark`, given the observed Sharpe and the higher moments of returns.

    Returns a probability in [0, 1]. PSR > 0.95 is the standard 'edge is real'
    bar; the conquest project loosens this to 0.05 (5%) for promotion gates.
    """
    if returns.empty or len(returns) < 30:
        return float("nan")
    sr = sharpe(returns)
    if np.isnan(sr):
        return float("nan")
    n = len(returns)
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())  # excess kurtosis (Fisher)
    # Constant-return series have undefined higher moments; assume normality.
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    sr_periodic = sr / np.sqrt(252.0)
    # Annualized SR benchmark -> per-period
    sr_b = sr_benchmark / np.sqrt(252.0)
    discriminant = 1.0 - skew * sr_periodic + (kurt / 4.0) * sr_periodic ** 2
    if discriminant <= 0:
        return float("nan")
    denom = np.sqrt(discriminant)
    z = (sr_periodic - sr_b) * np.sqrt(n - 1) / denom
    # PSR = standard normal CDF of z
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def compute_metrics(equity: pd.Series) -> dict:
    """CAGR, Sharpe, max DD, PSR, end equity. Mirrors the LIVE-pin metric set."""
    if equity.empty:
        return {}
    rets = equity.pct_change().dropna()
    return {
        "cagr": annual_return(rets),
        "sharpe": sharpe(rets),
        "max_dd": max_drawdown(equity),
        "psr": probabilistic_sharpe_ratio(rets),
        "end_equity": float(equity.iloc[-1]),
        "start_equity": float(equity.iloc[0]),
        "n_days": int(len(equity)),
    }


def default_crypto_bridge() -> list[CryptoBridgeSegment]:
    """The conquest CSA bridge: BIL pre-GBTC, then GBTC -> BITO -> IBIT.

    Boundaries chosen at each vehicle's first trading day per public records:
        - GBTC: 2013-09-25 (FINRA OTC quotation start)
        - BITO: 2021-10-19 (NYSE Arca listing)
        - IBIT: 2024-01-11 (NASDAQ listing alongside spot ETF approval)

    Tail end_exclusive set well past the project's window so a 2026-05-04 CSA
    backtest doesn't fall off the end of the bridge.
    """
    return [
        CryptoBridgeSegment("2008-01-01", "2013-09-25", "BIL"),
        CryptoBridgeSegment("2013-09-25", "2021-10-19", "GBTC"),
        CryptoBridgeSegment("2021-10-19", "2024-01-11", "BITO"),
        CryptoBridgeSegment("2024-01-11", "2099-12-31", "IBIT"),
    ]


def default_weights() -> dict[str, float]:
    """50/30/10/5/5 — cgrowth/cstability/crypto/GLD/BIL."""
    return {
        "cgrowth":    0.50,
        "cstability": 0.30,
        "crypto":     0.10,
        "GLD":        0.05,
        "BIL":        0.05,
    }


def run(cfg: CSAConfig) -> CSAResult:
    """End-to-end: load fund + aux returns, align, simulate, compute metrics."""
    fund_rets = {
        name: load_fund_returns(path) for name, path in cfg.fund_curves.items()
    }
    aux_rets, _proxy_map = load_aux_returns(cfg.aux_prices, cfg.crypto_bridge)
    aligned = align_returns(fund_rets, aux_rets, cfg.window)
    return simulate(aligned, cfg)
