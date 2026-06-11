"""KellySized — re-weight any base model's selected names via half-Kelly sizing.

Takes whichever names the base model said to hold (any non-zero weight) and
replaces equal-weight with::

    fᵢ = fraction × μ̂ᵢ / σ̂ᵢ²

where μ̂ and σ̂ are trailing ``lookback``-day annualized log-return statistics.
Then the row is normalised to a ``leverage_cap`` and clipped long-only.

Why use ½-Kelly here?
    Full-Kelly amplifies μ̂ noise into wild leverage. ½-Kelly keeps most of the
    asymptotic growth advantage with much smoother drawdowns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model
from conquest.vol.kelly import kelly_weights


class KellySized(Model):
    def __init__(
        self,
        base_model: Model,
        lookback: int = 60,
        fraction: float = 0.5,
        leverage_cap: float = 1.5,
    ):
        self.base = base_model
        self.lookback = lookback
        self.fraction = fraction
        self.leverage_cap = leverage_cap
        self.name = f"{base_model.name}_kelly"

    def signal(self, prices, regime=None, vol=None):
        base_w = self.base.signal(prices, regime, vol)
        selected = base_w > 0

        log_rets = np.log(prices / prices.shift(1))
        mu = log_rets.rolling(self.lookback).mean() * 252
        sigma = log_rets.rolling(self.lookback).std() * np.sqrt(252)

        # Mask μ̂ to selected names; σ̂ stays full so the kelly formula is well-defined.
        mu_active = mu.where(selected, 0)
        kw = kelly_weights(
            mu_active, sigma,
            fraction=self.fraction,
            leverage_cap=self.leverage_cap,
            long_only=True,
        )
        return kw.where(selected, 0)
