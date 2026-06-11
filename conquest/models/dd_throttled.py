"""DrawdownThrottled - cut gross exposure when in active drawdown.

Wraps a base model. Computes an approximate strategy equity curve from the
base weights x daily price returns, then on rows where the running drawdown
is below `threshold` (e.g., -0.12), scales the row's weights by
`throttle_factor` (e.g., 0.5). Restores 1.0x scale once the equity curve
makes a new high.

The strategy-equity proxy uses signal-weights x lagged returns (engine
convention: weights set end-of-day t-1 hold through day t). It is daily-
rebalanced, uncosted, and uses signals before the engine's monthly
resampling - so it slightly overstates DD vs the realized backtest. The
direction is right; the throttle reaction may be a bit early in practice,
which is conservative for the goal (cap DD).
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class DrawdownThrottled(Model):
    def __init__(
        self,
        base_model: Model,
        threshold: float = -0.12,
        throttle_factor: float = 0.5,
    ):
        if not (-1.0 < threshold < 0):
            raise ValueError("threshold must be a negative drawdown level (e.g. -0.12)")
        if not (0 < throttle_factor < 1):
            raise ValueError("throttle_factor must be in (0, 1)")
        self.base = base_model
        self.threshold = threshold
        self.throttle_factor = throttle_factor
        self.name = f"{base_model.name}_dd_throttled"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        rets = prices.pct_change().fillna(0)
        strat_rets = (weights.shift(1).fillna(0) * rets).sum(axis=1)
        equity = (1.0 + strat_rets).cumprod()
        high_water = equity.cummax()
        dd = equity.div(high_water) - 1.0
        in_dd = dd < self.threshold
        scale = pd.Series(1.0, index=weights.index)
        scale[in_dd] = self.throttle_factor
        return weights.mul(scale, axis=0)
