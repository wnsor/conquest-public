"""RegimeProbabilityRotated - rotate to defensive basket when P(Stagflation) > threshold.

Where `RegimeRotator` reacts to TODAY's deterministic regime label, this wrapper
reacts to NEXT-MONTH's regime PROBABILITY — a leading signal. When the
probability of Stagflation crosses a threshold, rotate to GLD/TIP/TLT
PRE-EMPTIVELY, before the deterministic classifier flips.

This is the main answer to "predict, don't react."

Soft blending: between `low_threshold` and `high_threshold`, linearly blend
between base model and basket weights. Above `high_threshold`, full rotation.
Below `low_threshold`, pure base model.

Default thresholds (low=0.15, high=0.30) are conservative — only de-risk when
the model is meaningfully confident the next month will be Stagflation. Tune
in the bake-off if needed.
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class RegimeProbabilityRotated(Model):
    def __init__(
        self,
        base_model: Model,
        prob_df: pd.DataFrame,
        risk_off_basket: tuple[str, ...] = ("GLD", "TIP", "TLT"),
        low_threshold: float = 0.15,
        high_threshold: float = 0.30,
        prob_col: str = "p_stagflation",
    ):
        if not (0 <= low_threshold < high_threshold <= 1):
            raise ValueError("require 0 <= low_threshold < high_threshold <= 1")
        self.base = base_model
        self.prob_df = prob_df
        self.risk_off_basket = risk_off_basket
        self.low = low_threshold
        self.high = high_threshold
        self.prob_col = prob_col
        self.name = f"{base_model.name}_prob_rotated"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)

        prob_aligned = self.prob_df[self.prob_col].reindex(prices.index, method="ffill")
        # Blending factor: 0 below low, 1 above high, linear in between.
        # 0 -> pure base; 1 -> pure basket.
        alpha = ((prob_aligned - self.low) / (self.high - self.low)).clip(lower=0, upper=1)

        available = [t for t in self.risk_off_basket if t in prices.columns]
        if not available:
            return weights
        n = len(available)
        basket_w = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        for t in available:
            basket_w[t] = 1.0 / n

        # blended = (1 - alpha) * base + alpha * basket
        out = weights.mul(1 - alpha, axis=0) + basket_w.mul(alpha, axis=0)
        return out
