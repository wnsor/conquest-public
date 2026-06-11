"""VixRotated - rotate to a defensive basket when VIX enters risk-off state.

Mirrors RegimeRotator's pattern but driven by VIX (not the macro regime
classifier). Uses asymmetric hysteresis: enter risk-off when VIX > vix_high
(default 25.0), exit when VIX < vix_low (default 15.0). During risk-off,
hold the basket at full gross (1.0x, no scaling). Otherwise, pass through
to the base model unchanged.

The reason for two separate signals (regime classifier + VIX): the macro
classifier uses ALFRED vintage GDP/CPI data with ~1 month publication lag.
A rapid market shock (e.g., COVID March 2020) can crash 30%+ before the
classifier even flips to Stagflation. VIX moves intraday; this wrapper
catches the shock days before macro data would.

This is the v8.1 cstability addition (Option B in the user-facing bake-off).
The simpler Option A wraps with VixGated (halves gross to 0.5x without
rotating) — see scripts/rank_models.py for the comparison.
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class VixRotated(Model):
    def __init__(
        self,
        base_model: Model,
        vix_series: pd.Series | None = None,
        vix_high: float = 25.0,
        vix_low: float = 15.0,
        risk_off_basket: tuple[str, ...] = ("GLD", "TIP", "TLT"),
    ):
        if vix_low >= vix_high:
            raise ValueError("vix_low must be < vix_high (asymmetric hysteresis)")
        self.base = base_model
        self.vix = vix_series
        self.vix_high = vix_high
        self.vix_low = vix_low
        self.risk_off_basket = risk_off_basket
        self.name = f"{base_model.name}_vix_rotated"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        if self.vix is None:
            return weights

        vix_aligned = self.vix.reindex(prices.index, method="ffill")
        # Build per-row risk_off state via asymmetric hysteresis
        risk_off = pd.Series(False, index=prices.index)
        state = False
        for i, v in enumerate(vix_aligned):
            if pd.isna(v):
                risk_off.iloc[i] = state
                continue
            if not state and v > self.vix_high:
                state = True
            elif state and v < self.vix_low:
                state = False
            risk_off.iloc[i] = state

        if not risk_off.any():
            return weights

        # In risk-off rows, replace base weights with equal-weight basket at full gross
        available = [t for t in self.risk_off_basket if t in prices.columns]
        if not available:
            return weights
        n = len(available)
        for col in weights.columns:
            mask = risk_off
            if col in available:
                weights.loc[mask, col] = 1.0 / n
            else:
                weights.loc[mask, col] = 0.0
        return weights
