"""IV rank / percentile per underlying.

IV rank vs IV percentile (industry conventions):
  iv_rank      = 100 * (iv_today - iv_min_1y) / (iv_max_1y - iv_min_1y)
  iv_percentile = 100 * P(iv_history < iv_today)

Both ∈ [0, 100]. "Low IV" = rank/percentile below ~30; "high IV" = above
~70. Strategies use these as entry filters (e.g. A1 WSB calls only fire
when iv_rank<30 — buy options cheap).

Inputs are daily ATM IV estimates per ticker; we use the algorithm's
implied-vol proxy (VIX-to-stock via beta or chain-derived ATM IV).
"""
from __future__ import annotations

from collections import deque


class IVRankTracker:
    """One rolling window per ticker. Insert new daily IV samples; query
    rank/percentile against the trailing 252-day history."""

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._history: dict[str, deque[float]] = {}

    def update(self, ticker: str, iv_today: float) -> None:
        if iv_today is None or iv_today <= 0:
            return
        dq = self._history.setdefault(ticker, deque(maxlen=self.lookback_days))
        dq.append(float(iv_today))

    def has_warmup(self, ticker: str, min_samples: int = 60) -> bool:
        return len(self._history.get(ticker, ())) >= min_samples

    def rank(self, ticker: str, iv_today: float) -> float | None:
        h = self._history.get(ticker)
        if not h or iv_today is None or iv_today <= 0:
            return None
        lo, hi = min(h), max(h)
        if hi <= lo:
            return 50.0
        return 100.0 * (iv_today - lo) / (hi - lo)

    def percentile(self, ticker: str, iv_today: float) -> float | None:
        h = self._history.get(ticker)
        if not h or iv_today is None or iv_today <= 0:
            return None
        if not h:
            return None
        below = sum(1 for v in h if v < iv_today)
        return 100.0 * below / len(h)
