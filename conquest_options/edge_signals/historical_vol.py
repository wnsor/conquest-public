"""Realized (historical) volatility from price history.

Cheap to compute (no chain dependency, unlike IV). Useful three ways:

  1. IV/HV ratio as an "options expensive vs cheap" gate:
       iv_hv_ratio < 1.0 → market under-pricing vol → favors LONG options
       iv_hv_ratio > 1.2 → market over-pricing vol → avoid buying options
     Closely related to the Goyal-Saretto vol-risk-premium signal.

  2. Premium estimation for sizing:
       expected_premium ≈ spot × HV × sqrt(DTE/365) × 0.4   (BS-ATM approx)
     Lets the sizer predict what a contract will cost and skip if too expensive.

  3. Position-quality gate:
       High-HV names with low-IV setups = best long-options environment.
       Low-HV names = options don't move enough to pay theta.
"""
from __future__ import annotations

import math
from collections import deque


def compute_realized_vol(prices, periods: int = 30, annualize: float = 252.0) -> float | None:
    """Annualized stdev of log returns over the last `periods` price points.

    Returns None if insufficient history.
    """
    if prices is None:
        return None
    p = list(prices)
    if len(p) < periods + 1:
        return None
    recent = p[-(periods + 1):]
    rets = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))
            if recent[i - 1] > 0 and recent[i] > 0]
    if len(rets) < periods // 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var) * math.sqrt(annualize)


class HVTracker:
    """Per-ticker rolling-window HV computer over a price history deque."""

    def __init__(self, window_30d: int = 30, window_60d: int = 60):
        self.w30 = window_30d
        self.w60 = window_60d

    def hv_30(self, prices) -> float | None:
        return compute_realized_vol(prices, periods=self.w30)

    def hv_60(self, prices) -> float | None:
        return compute_realized_vol(prices, periods=self.w60)
