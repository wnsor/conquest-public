"""RegimeGated — wrap any base model with a 4-quadrant macro regime filter.

On Stagflation / Deflation regimes (below-trend growth), scale the base model's
weights down by ``risk_off_factor`` (default 0.5). On Inflation / Disinflation
(above-trend growth), pass through unchanged.

The Bridgewater 4-quadrant labels come from `conquest.regime.RegimeClassifier`
and live at ``storage/conquest/regime/daily.csv`` (Lean Object Store path).
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class RegimeGated(Model):
    def __init__(
        self,
        base_model: Model,
        regime_series: pd.Series | None = None,
        risk_off_regimes: tuple[str, ...] = ("Stagflation", "Deflation"),
        risk_off_factor: float = 0.5,
    ):
        self.base = base_model
        self.regime = regime_series
        self.risk_off_regimes = set(risk_off_regimes)
        self.risk_off_factor = risk_off_factor
        self.name = f"{base_model.name}_regime_gated"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        regime_to_use = regime if regime is not None else self.regime
        if regime_to_use is None:
            return weights  # no regime context → pass through

        regime_aligned = regime_to_use.reindex(prices.index, method="ffill")
        is_risk_off = regime_aligned.isin(self.risk_off_regimes)
        scale = pd.Series(1.0, index=prices.index)
        scale.loc[is_risk_off] = self.risk_off_factor
        return weights.mul(scale, axis=0)
