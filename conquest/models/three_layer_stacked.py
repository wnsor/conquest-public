"""ThreeLayerStacked — combine slow + medium + fast stress signals via voting.

Layered leading-indicator system:
- Layer 1 (slow, structural):  macro regime classifier flags Stagflation
- Layer 2 (medium, leading):   HY credit-stress proxy < threshold (HYG-IEF 60d log spread)
- Layer 3 (fast, confirmation): VIX/VIX3M term ratio > threshold (backwardation)

Voting logic with hysteresis-style thresholds:
- 0 votes -> pass through to base model
- 1 vote (early warning) -> blend 80% base + 20% defensive basket
- 2 votes (confirmed risk-off) -> blend 30% base + 70% defensive basket
- 3 votes (consensus) -> 100% defensive basket

Why three layers? Each is independent and has different lead times. A single
signal (e.g. VIX) has too many false positives (saw this with v8.1). A 2-of-3
vote filters out noise: only de-risk when AT LEAST TWO independent measures
agree something is wrong. This was the design recommended in the leading-
indicator analysis after v8.1 failed.

Defaults tuned conservatively from inspection of the 2008 / 2018 / 2020 /
2022 episodes. Not in-sample optimized — see anti-overfit memory.
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class ThreeLayerStacked(Model):
    def __init__(
        self,
        base_model: Model,
        regime_series: pd.Series,
        credit_stress_series: pd.Series,
        vix_term_series: pd.Series,
        defensive_basket: tuple[str, ...] = ("GLD", "TIP", "TLT"),
        risk_off_regimes: tuple[str, ...] = ("Stagflation",),
        credit_threshold: float = -0.05,    # HY-IEF 60d log spread <= -5% = stress
        vix_term_threshold: float = 1.05,   # VIX/VIX3M > 1.05 = backwardation
        blend_weights: tuple[float, ...] = (0.0, 0.20, 0.70, 1.0),  # alpha for 0/1/2/3 votes
    ):
        if len(blend_weights) != 4:
            raise ValueError("blend_weights must have 4 entries (for 0/1/2/3 votes)")
        if not all(0 <= w <= 1 for w in blend_weights):
            raise ValueError("blend_weights must each be in [0, 1]")
        self.base = base_model
        self.regime = regime_series
        self.credit = credit_stress_series
        self.vix_term = vix_term_series
        self.defensive_basket = defensive_basket
        self.risk_off_regimes = set(risk_off_regimes)
        self.credit_threshold = credit_threshold
        self.vix_term_threshold = vix_term_threshold
        self.blend_weights = blend_weights
        self.name = f"{base_model.name}_three_layer"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)

        regime_aligned = self.regime.reindex(prices.index, method="ffill")
        credit_aligned = self.credit.reindex(prices.index, method="ffill")
        vix_aligned = self.vix_term.reindex(prices.index, method="ffill")

        # Votes (boolean Series)
        v_regime = regime_aligned.isin(self.risk_off_regimes).fillna(False)
        v_credit = (credit_aligned < self.credit_threshold).fillna(False)
        v_vix = (vix_aligned > self.vix_term_threshold).fillna(False)
        votes = v_regime.astype(int) + v_credit.astype(int) + v_vix.astype(int)

        # Per-row blend factor: 0 = pure base, 1 = pure basket
        alpha = votes.map(lambda n: self.blend_weights[int(n)]).astype(float)

        # Build the basket weights df
        available = [t for t in self.defensive_basket if t in prices.columns]
        if not available:
            return weights
        n_basket = len(available)
        basket_w = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        for t in available:
            basket_w[t] = 1.0 / n_basket

        out = weights.mul(1 - alpha, axis=0) + basket_w.mul(alpha, axis=0)
        return out
