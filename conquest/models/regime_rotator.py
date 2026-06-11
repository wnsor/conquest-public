"""RegimeRotator - in risk-off regimes, ROTATE to a defensive basket at full gross.

The existing RegimeGated wrapper halves gross in risk-off regimes. That's the
wrong shape: it cedes upside in Deflation (the best regime for equities per
2014-2024 data) and only modestly cushions Stagflation (the worst regime).

This wrapper instead REPLACES the base model's selection in specified regimes
with a fixed defensive basket at equal weight. In normal regimes (default:
Inflation, Disinflation), the base model passes through unchanged. In risk-off
regimes (default: Stagflation), the basket holds gold + TIPS + treasuries at
equal weight.

Since the rotation is to alternative asset classes (not cash) at full gross,
total exposure stays at 100% in every regime - return drag is replaced by
regime-appropriate diversifier exposure. Stagflation gold/TIPS rallies
historically offset much of the equity drawdown.
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class RegimeRotator(Model):
    def __init__(
        self,
        base_model: Model,
        regime_series: pd.Series | None = None,
        regime_baskets: dict[str, tuple[str, ...]] | None = None,
    ):
        """
        Args:
            base_model: model used in regimes NOT covered by `regime_baskets`.
            regime_series: optional regime label series (else passed via signal()).
            regime_baskets: dict mapping regime name -> tuple of tickers. In each
                listed regime, the base model is replaced by an equal-weight hold
                of the basket. Default: {"Stagflation": ("GLD", "TIP", "TLT")}.
        """
        self.base = base_model
        self.regime = regime_series
        self.regime_baskets = regime_baskets or {
            "Stagflation": ("GLD", "TIP", "TLT"),
        }
        self.name = f"{base_model.name}_regime_rotated"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        regime_to_use = regime if regime is not None else self.regime
        if regime_to_use is None:
            return weights

        regime_aligned = regime_to_use.reindex(prices.index, method="ffill")

        for regime_name, basket in self.regime_baskets.items():
            available = [t for t in basket if t in prices.columns]
            if not available:
                continue
            mask = regime_aligned == regime_name
            if not mask.any():
                continue
            n = len(available)
            # Replace this regime's rows with equal-weight basket
            for col in weights.columns:
                weights.loc[mask, col] = (1.0 / n) if col in available else 0.0
        return weights
