"""Dynamic stock picker — daily top-N ranking for momentum strategies.

Purpose
-------
Replace the static WSB_UNIVERSE list (22 hand-picked tickers + IPO guards)
with a daily-computed top-N ranking of candidates from a broader universe
(S&P 500 + supplemental high-vol single-names). The picker scores each
candidate on a composite of factors and returns the top N for momentum
strategies to consume.

Architecture inspiration
------------------------
cgrowth's Q+M composite (180-day momentum + 60-day -vol z-score, VOL_WEIGHT=4.0)
is the proven template. PSR 35.29% / CAGR 20.22% / DD -23% on PIT
2008-2026. This module ports that template + adds factors specific to
options trading:
  - Liquidity gate (option chain availability + ADV minimum)
  - News momentum (GDELT story growth)
  - Recent breakout score (price > 252d high * 0.95)

Why we want this
----------------
Static universe is fragile to:
  1. Survivorship bias (we cherry-pick names that worked in hindsight)
  2. Time-dependence (PLTR 2020 ≠ PLTR 2024 ≠ PLTR 2026 narrative)
  3. Regime change (NVDA-style names don't always exist; what replaces them?)

Dynamic picker addresses all three: scans a broad universe daily, picks
the names with the active momentum profile RIGHT NOW.

API
---
    picker = StockPicker(universe_csv=...)
    top_20 = picker.rank(today, hist_prices, hist_news, option_chains)
    # → list[ScoredTicker], sorted by score descending

Pure-function design — no QC dependency, fully unit-testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping


# Score weights — calibrated to cgrowth Q+M template + options-specific factors.
# Sum doesn't need to equal 1.0; we min-max scale within each factor before
# multiplying by weight.
DEFAULT_WEIGHTS = {
    "momentum_180d": 1.0,      # 6-month price momentum (cgrowth Q+M anchor)
    "neg_vol_60d_z": 4.0,      # negative vol-z (cgrowth's VOL_WEIGHT — anchors quality)
    "news_growth_5d": 0.5,     # recent narrative acceleration
    "breakout_score": 0.8,     # near-52w high (fish_-style momentum)
    "iv_rank": 0.3,            # lower IV = cheaper options = better RR
    "liquidity_gate": 0.0,     # HARD GATE — multiplier 1.0 or 0.0
}


@dataclass
class ScoredTicker:
    ticker: str
    composite: float
    momentum_180d: float
    neg_vol_60d_z: float
    news_growth_5d: float
    breakout_score: float
    iv_rank: float
    passed_liquidity: bool

    def __repr__(self) -> str:
        return (f"{self.ticker:<6s} comp={self.composite:+.2f} "
                f"mom={self.momentum_180d:+.2f} "
                f"vol_z={self.neg_vol_60d_z:+.2f} "
                f"news={self.news_growth_5d:+.2f} "
                f"brk={self.breakout_score:+.2f} "
                f"liq={self.passed_liquidity}")


def _momentum_180d(prices_252d: list[float]) -> float | None:
    """Return 180-day price momentum as a ratio (e.g., 1.25 = +25%).

    Requires ≥180 prices. Returns None if insufficient history.
    """
    if not prices_252d or len(prices_252d) < 180:
        return None
    p_now = prices_252d[-1]
    p_180 = prices_252d[-180]
    if p_180 <= 0:
        return None
    return p_now / p_180


def _vol_60d(prices_252d: list[float]) -> float | None:
    """Return 60-day realized vol (std of daily log returns)."""
    if not prices_252d or len(prices_252d) < 61:
        return None
    rets = []
    for i in range(-60, 0):
        p0 = prices_252d[i - 1]
        p1 = prices_252d[i]
        if p0 <= 0 or p1 <= 0:
            continue
        rets.append(math.log(p1 / p0))
    if len(rets) < 10:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var) if var > 0 else 0.0


def _vol_z_score(vol_now: float, vol_history_252d: list[float]) -> float | None:
    """Z-score of current vol vs trailing 252d vol distribution.

    Negative z = vol is BELOW historical median (quality momentum profile).
    Positive z = vol is ABOVE historical median (chop / risk-off).
    """
    if vol_now is None or not vol_history_252d:
        return None
    hist = [v for v in vol_history_252d if v is not None and v > 0]
    if len(hist) < 30:
        return None
    mean = sum(hist) / len(hist)
    var = sum((v - mean) ** 2 for v in hist) / len(hist)
    if var <= 0:
        return None
    return (vol_now - mean) / math.sqrt(var)


def _news_growth_5d(news_count_10d: list[int]) -> float | None:
    """Last 5 days news count vs prior 5 days. >1.5 = acceleration.

    Uses GDELT-style daily story counts. Caller supplies a 10-day series.
    """
    if not news_count_10d or len(news_count_10d) < 10:
        return None
    recent = sum(news_count_10d[-5:])
    prior = sum(news_count_10d[-10:-5])
    if prior <= 0:
        return None
    return recent / prior


def _breakout_score(prices_252d: list[float]) -> float | None:
    """How close is current price to 252d high? 1.0 = at high, 0.0 = at low.

    Score = (current - 252d_low) / (252d_high - 252d_low).
    """
    if not prices_252d or len(prices_252d) < 30:
        return None
    hi = max(prices_252d)
    lo = min(prices_252d)
    if hi <= lo:
        return None
    return (prices_252d[-1] - lo) / (hi - lo)


def _liquidity_gate(
    adv_dollars: float | None,
    option_chain_size: int | None,
    *,
    min_adv: float = 50_000_000,
    min_chain: int = 50,
) -> bool:
    """Pass if average daily $-volume ≥ min_adv AND chain has ≥ min_chain contracts."""
    if adv_dollars is None or adv_dollars < min_adv:
        return False
    if option_chain_size is None or option_chain_size < min_chain:
        return False
    return True


def _minmax_scale(values: dict[str, float]) -> dict[str, float]:
    """Min-max scale a dict of {ticker: value} to [0, 1]. Tickers with None values
    are excluded from output."""
    real = {k: v for k, v in values.items() if v is not None and math.isfinite(v)}
    if not real:
        return {}
    mn = min(real.values())
    mx = max(real.values())
    if mx == mn:
        return {k: 0.5 for k in real}
    return {k: (v - mn) / (mx - mn) for k, v in real.items()}


@dataclass
class TickerInputs:
    """Per-ticker daily snapshot fed to the picker."""
    prices_252d: list[float] = None       # noqa: RUF013 — None default ok for dataclass
    news_count_10d: list[int] = None
    adv_dollars: float = None
    option_chain_size: int = None
    iv_rank: float = None


class StockPicker:
    """Compose factor scores → composite → rank → top-N selection.

    Use:
        picker = StockPicker(weights=DEFAULT_WEIGHTS)
        scored = picker.rank({
            "NVDA": TickerInputs(prices_252d=[...], news_count_10d=[...], ...),
            "AMD":  TickerInputs(...),
            ...
        })
        top_20 = [s.ticker for s in scored[:20]]
    """

    def __init__(self, weights: Mapping[str, float] | None = None):
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)

    def _per_ticker_scores(self, t: TickerInputs) -> dict[str, float | None]:
        mom = _momentum_180d(t.prices_252d) if t.prices_252d else None
        vol_now = _vol_60d(t.prices_252d) if t.prices_252d else None
        # Vol history is the rolling 60-day std taken at every point; for the
        # z-score we use the trailing-252d distribution of those rolling stds.
        # Caller-provided shortcut: pass prices_252d, we compute vol history
        # for each window. Approximate: just use today's vol minus mean vol.
        vol_history = []
        if t.prices_252d and len(t.prices_252d) >= 121:
            for end in range(60, len(t.prices_252d)):
                window = t.prices_252d[max(0, end - 60):end]
                v = _vol_60d(window + [t.prices_252d[end]])
                if v is not None:
                    vol_history.append(v)
        vol_z = _vol_z_score(vol_now, vol_history) if vol_now is not None else None
        # We want NEGATIVE vol z (low vol = quality momentum) — flip sign
        neg_vol_z = -vol_z if vol_z is not None else None
        news = _news_growth_5d(t.news_count_10d) if t.news_count_10d else None
        breakout = _breakout_score(t.prices_252d) if t.prices_252d else None
        liq = _liquidity_gate(t.adv_dollars, t.option_chain_size)
        return {
            "momentum_180d": mom,
            "neg_vol_60d_z": neg_vol_z,
            "news_growth_5d": news,
            "breakout_score": breakout,
            "iv_rank": t.iv_rank,
            "liquidity_gate": liq,
        }

    def rank(self, inputs: dict[str, TickerInputs]) -> list[ScoredTicker]:
        """Rank tickers by composite score. Liquidity-failures excluded."""
        per_ticker = {tk: self._per_ticker_scores(ti) for tk, ti in inputs.items()}
        # Filter liquidity-failures FIRST
        eligible = {tk: scores for tk, scores in per_ticker.items()
                    if scores["liquidity_gate"]}
        if not eligible:
            return []
        # Min-max scale each factor across eligible tickers
        scaled: dict[str, dict[str, float]] = {tk: {} for tk in eligible}
        for factor in ("momentum_180d", "neg_vol_60d_z", "news_growth_5d",
                       "breakout_score", "iv_rank"):
            raw = {tk: per_ticker[tk][factor] for tk in eligible}
            scaled_factor = _minmax_scale(raw)
            for tk in eligible:
                scaled[tk][factor] = scaled_factor.get(tk, 0.5)
        # Composite: weighted sum of scaled factors
        composites = {}
        for tk in eligible:
            comp = 0.0
            for factor, weight in self.weights.items():
                if factor == "liquidity_gate":
                    continue
                comp += weight * scaled[tk].get(factor, 0.5)
            composites[tk] = comp
        # Build ScoredTicker objects
        results = []
        for tk in eligible:
            results.append(ScoredTicker(
                ticker=tk,
                composite=composites[tk],
                momentum_180d=per_ticker[tk]["momentum_180d"] or 0.0,
                neg_vol_60d_z=per_ticker[tk]["neg_vol_60d_z"] or 0.0,
                news_growth_5d=per_ticker[tk]["news_growth_5d"] or 0.0,
                breakout_score=per_ticker[tk]["breakout_score"] or 0.0,
                iv_rank=per_ticker[tk]["iv_rank"] or 0.0,
                passed_liquidity=True,
            ))
        results.sort(key=lambda s: s.composite, reverse=True)
        return results

    def top_n(self, inputs: dict[str, TickerInputs], n: int = 20) -> list[str]:
        """Convenience: just the ticker symbols for the top-N."""
        return [s.ticker for s in self.rank(inputs)[:n]]
