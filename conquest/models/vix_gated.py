"""VixGated — wrap any base model with an asymmetric VIX-level filter.

Goes risk-off (scale weights × ``risk_off_factor``) when VIX rises through
``vix_high``, and only exits risk-off when VIX drops back below ``vix_low``.

Asymmetric thresholds are the standard practitioner pattern: avoids flapping
when VIX hovers near a single threshold. Common values: enter at 25, exit at 15.
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class VixGated(Model):
    def __init__(
        self,
        base_model: Model,
        vix_series: pd.Series | None = None,
        vix_high: float = 25.0,
        vix_low: float = 15.0,
        risk_off_factor: float = 0.5,
    ):
        if vix_low >= vix_high:
            raise ValueError("vix_low must be < vix_high (asymmetric gate)")
        self.base = base_model
        self.vix = vix_series
        self.vix_high = vix_high
        self.vix_low = vix_low
        self.risk_off_factor = risk_off_factor
        self.name = f"{base_model.name}_vix_gated"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        if self.vix is None:
            return weights

        vix_aligned = self.vix.reindex(prices.index, method="ffill")
        scale = pd.Series(1.0, index=prices.index)
        risk_off = False
        for i, v in enumerate(vix_aligned):
            if pd.isna(v):
                if risk_off:
                    scale.iloc[i] = self.risk_off_factor
                continue
            if not risk_off and v > self.vix_high:
                risk_off = True
            elif risk_off and v < self.vix_low:
                risk_off = False
            if risk_off:
                scale.iloc[i] = self.risk_off_factor
        return weights.mul(scale, axis=0)
