"""DIY equity put/call ratio from Lean chain volume.

The CBOE-published equity P/C series ended 2019-10. For 2019+ we compute
the same statistic ourselves from QC's `US Equity Option Universe` chain
data: each day, sum the put-leg traded volume divided by the call-leg
traded volume across all subscribed underlyings' chains. Methodology
matches Vasquez/Xiao 2024.

Smooths with a 10-day EMA to dampen single-day noise; strategies inspect
ctx.pc_ratio_equity.
"""
from __future__ import annotations

from collections import deque


class PutCallRatioTracker:
    def __init__(self, ema_span_days: int = 10):
        self.ema_span_days = ema_span_days
        self._alpha = 2.0 / (ema_span_days + 1)
        self._ema: float | None = None
        self._daily_history: deque[float] = deque(maxlen=252)

    def consume_day(self, total_put_volume: int, total_call_volume: int) -> float | None:
        """Update with one day of aggregated chain volume. Returns the
        smoothed P/C ratio (EMA). Returns None until first non-zero day."""
        if total_call_volume <= 0:
            return self._ema
        raw = total_put_volume / total_call_volume
        self._daily_history.append(raw)
        if self._ema is None:
            self._ema = raw
        else:
            self._ema = self._alpha * raw + (1 - self._alpha) * self._ema
        return self._ema

    @property
    def current(self) -> float | None:
        return self._ema

    def percentile(self, value: float | None = None) -> float | None:
        v = self._ema if value is None else value
        h = list(self._daily_history)
        if v is None or not h:
            return None
        below = sum(1 for x in h if x < v)
        return 100.0 * below / len(h)
