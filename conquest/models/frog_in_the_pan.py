"""Frog-in-the-Pan (FIP) filter — wrap any momentum model to favor stocks with
'continuous information' (steady price drift) over 'discrete information' (gappy).

Reference: Da, Gurun, Warachka (2014) "Frog in the Pan: Continuous Information
and Momentum", Review of Financial Studies. Key empirical finding: 6-mo
holding-period return for high-FIP-quality (continuous info) momentum stocks
was 8.86% vs 2.91% for low-FIP-quality (discrete info), and persisted ~8 months
vs ~2 months.

FIP measure
-----------
For each stock and date t, look at the last `lookback` days of returns:

    FIP_t = sign(cumulative_return_t) × (% negative days − % positive days)

Lower FIP = MORE continuous information (steady drift in same direction).
Higher FIP = MORE discrete (jumpy, news-driven).

This wrapper FILTERS the base model's picks by keeping only the top-K by
"continuity" — i.e., stocks whose momentum came from steady drift, not
news shocks. The hypothesis: continuous-info momentum has longer runway.

Usage
-----
    base = DualMomentum(top_n=10, lookback=180)
    fip = FrogInPanFilter(base, top_k_filter=5, lookback=180)
    weights = fip.signal(prices)

The base model picks `top_n` candidates (by momentum); the FIP filter then
trims to `top_k_filter` by continuity (lowest FIP value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model


class FrogInPanFilter(Model):
    """Filter a base model's picks by FIP continuity score (lower = better).

    Args:
        base_model: any Model whose signal() returns weights DataFrame.
        top_k_filter: keep this many of the base's picks (must be ≤ base.top_n).
        lookback: days to evaluate FIP over (default 180 to match v8.5).
        cum_return_min: minimum cumulative return required for "continuous"
            interpretation. If a stock's cumulative return is near zero,
            the sign is noisy; default 0.01 (1%) keeps near-flat stocks out.
    """

    def __init__(
        self,
        base_model: Model,
        top_k_filter: int = 5,
        lookback: int = 180,
        cum_return_min: float = 0.01,
    ):
        self.base = base_model
        self.top_k_filter = top_k_filter
        self.lookback = lookback
        self.cum_return_min = cum_return_min
        self.name = f"{base_model.name}_fip"

    def _fip_score(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return FIP score per (date, stock). Lower = more continuous."""
        returns = prices.pct_change()
        # Cumulative return over lookback: sign tells us "is this an up- or down-mover?"
        cum_ret = (1 + returns).rolling(self.lookback, min_periods=self.lookback).apply(
            lambda x: x.prod() - 1, raw=True
        )
        # %positive and %negative days over lookback
        pct_pos = (returns > 0).rolling(self.lookback, min_periods=self.lookback).mean()
        pct_neg = (returns < 0).rolling(self.lookback, min_periods=self.lookback).mean()

        # FIP = sign(cum_ret) × (pct_neg − pct_pos)
        # For an UP mover (cum_ret > 0): more pct_pos days → continuous (FIP negative);
        #     more pct_neg days but still up → discrete jump (FIP positive).
        # For a DOWN mover (cum_ret < 0): mirror image.
        fip = np.sign(cum_ret) * (pct_neg - pct_pos)
        # Mask out stocks with near-zero cumulative move (sign is noise)
        fip = fip.where(cum_ret.abs() > self.cum_return_min)
        return fip

    def signal(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        vol: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        base_w = self.base.signal(prices, regime, vol)
        fip = self._fip_score(prices)

        # For each row, restrict base's non-zero picks to the lowest-FIP top_k_filter
        out = pd.DataFrame(0.0, index=base_w.index, columns=base_w.columns)
        for date in base_w.index:
            row = base_w.loc[date]
            picks = row[row > 0]
            if picks.empty:
                continue
            # Get FIP scores for these picks
            fip_row = fip.loc[date].reindex(picks.index)
            # Keep picks with valid FIP (not NaN); among those, take lowest-FIP top-K
            valid = fip_row.dropna()
            if valid.empty:
                # Fallback: keep base's picks unchanged (FIP not yet warmed up)
                out.loc[date] = row
                continue
            keep = valid.nsmallest(self.top_k_filter).index
            # Equal-weight the kept picks
            n = len(keep)
            if n > 0:
                w = 1.0 / n
                for sym in keep:
                    out.at[date, sym] = w
        return out
