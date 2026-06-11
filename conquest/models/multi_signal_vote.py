"""MultiSignalVote — generic N-signal voting ensemble.

Generalizes ThreeLayerStacked to support any number of independent stress
signals. Each signal is a boolean pd.Series indexed by date; the wrapper
counts votes per row and uses a blend-weight lookup to mix base model with
defensive basket.

Why generic: v9.2 uses 3 votes (regime + credit + VIX-term). v10 adds the
P(Stagflation) probability forecast as a 4th vote. Future versions may add
more (yield-curve, breadth, dollar-strength, etc.). Better to have one
parameterizable wrapper than a separate class per vote count.

Default blend weights when not provided: linear from 0 (no votes) to 1
(all votes). For 4-vote: [0.0, 0.25, 0.50, 0.75, 1.0]. For 3-vote
(equivalent to ThreeLayerStacked): [0.0, 0.33, 0.67, 1.0]. Caller can
override for non-linear blending (e.g., heavy weight on consensus-only).

Defensive basket weighting: 'equal' (default; 1/N) or 'inverse_vol' (weight
each defensive ETF by 1/realized_vol over `vol_lookback` days, normalized to
sum to 1). Inverse-vol prevents implicit leverage into the highest-vol
defensive (TLT ≈ 18% annualized vs TIP ≈ 7% — equal-weight overweights TLT
relative to its risk contribution).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model
from conquest.vol.realized import realized_vol


class MultiSignalVote(Model):
    def __init__(
        self,
        base_model: Model,
        vote_signals: list[pd.Series],
        defensive_basket: tuple[str, ...] = ("GLD", "TIP", "TLT"),
        blend_weights: list[float] | None = None,
        defensive_weighting: str = "equal",
        vol_lookback: int = 126,
        name_suffix: str = "multi_vote",
    ):
        if not vote_signals:
            raise ValueError("at least one vote signal required")
        n_votes = len(vote_signals)
        if blend_weights is None:
            blend_weights = [i / n_votes for i in range(n_votes + 1)]
        if len(blend_weights) != n_votes + 1:
            raise ValueError(
                f"blend_weights must have {n_votes + 1} entries for {n_votes} votes "
                f"(one per possible vote count 0..{n_votes})"
            )
        if not all(0 <= w <= 1 for w in blend_weights):
            raise ValueError("each blend_weight must be in [0, 1]")
        if defensive_weighting not in ("equal", "inverse_vol"):
            raise ValueError("defensive_weighting must be 'equal' or 'inverse_vol'")
        self.base = base_model
        self.vote_signals = vote_signals
        self.defensive_basket = defensive_basket
        self.blend_weights = blend_weights
        self.defensive_weighting = defensive_weighting
        self.vol_lookback = vol_lookback
        weighting_tag = "_invvol" if defensive_weighting == "inverse_vol" else ""
        self.name = f"{base_model.name}_{name_suffix}_{n_votes}{weighting_tag}"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)

        # Sum boolean votes per row, aligned to prices index
        votes = pd.Series(0, index=prices.index, dtype=int)
        for s in self.vote_signals:
            aligned = s.reindex(prices.index, method="ffill").fillna(False)
            votes = votes + aligned.astype(int)

        # Per-row blend factor based on vote count
        alpha = votes.map(lambda n: self.blend_weights[int(n)]).astype(float)

        # Defensive basket weights — equal or inverse-vol per row
        available = [t for t in self.defensive_basket if t in prices.columns]
        if not available:
            return weights
        basket_w = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        if self.defensive_weighting == "equal":
            n_basket = len(available)
            for t in available:
                basket_w[t] = 1.0 / n_basket
        else:  # inverse_vol
            basket_vol = realized_vol(prices[available], lookback=self.vol_lookback)
            inv = (1.0 / basket_vol).replace([np.inf, -np.inf], np.nan)
            row_sum = inv.sum(axis=1)
            iv_weights = inv.div(row_sum.replace(0, np.nan), axis=0)
            # Fallback to equal-weight on rows where vol is not yet available (warmup)
            warmup_mask = row_sum.isna() | (row_sum == 0)
            n_basket = len(available)
            for t in available:
                basket_w[t] = iv_weights[t].fillna(1.0 / n_basket).where(~warmup_mask, 1.0 / n_basket)

        return weights.mul(1 - alpha, axis=0) + basket_w.mul(alpha, axis=0)
