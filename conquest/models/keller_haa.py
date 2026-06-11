"""Keller HAA (Hybrid Asset Allocation) — Wouter Keller & JW Keuning, 2023.

Reference: SSRN 4346906, "Dual and Canary Momentum with Rising Yields/Inflation".
Allocate Smartly tracks this with reported ~15.8% CAGR / -10% MaxDD / Sharpe 1.25
on the 1971-2023 backtest window.

Rules
-----
- "Canary" universe: just TIP (TIPS ETF). Computed as the unweighted average
  of TIP's 1, 3, 6, 12 month returns. If positive, treat regime as "risk-on"
  (~86% of months historically). If negative, "risk-off".
- Offensive universe (8 assets): SPY, IWM, EFA, EEM, VNQ, PDBC, IEF, TLT.
- Defensive: IEF (intermediate Treasuries) — also used as a fallback when an
  offensive asset has negative momentum.

Monthly process:
    1. Risk-on (canary > 0):
       - Compute 1/3/6/12-month avg momentum for all 8 offensive assets.
       - Pick top-4 by momentum.
       - For each of the top-4: if its momentum > 0, allocate 25%; else
         allocate that 25% slice to IEF.
    2. Risk-off (canary <= 0):
       - 100% IEF (or cash, but our universe doesn't include cash so IEF).

Anti-overfit notes
------------------
- Canary universe is intentionally MINIMAL (1 asset = TIP). Keller explicitly
  designed HAA to be simpler than BAA after BAA was criticized for overfit risk.
- The 1/3/6/12 unweighted average is robust to lookback choice — it's the
  "diversify across reasonable lookbacks" pattern, not a single curve-fit number.
- No threshold tuning, no asset-weight-by-momentum (just equal 25% each top-4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model


HAA_OFFENSIVE = ["SPY", "IWM", "EFA", "EEM", "VNQ", "PDBC", "IEF", "TLT"]
HAA_CANARY = "TIP"
HAA_DEFENSIVE = "IEF"


def _avg_1_3_6_12_momentum(prices: pd.Series) -> pd.Series:
    """Unweighted avg of 1/3/6/12 calendar-month returns.

    1mo ≈ 21bd, 3mo ≈ 63bd, 6mo ≈ 126bd, 12mo ≈ 252bd.
    """
    r1 = prices.pct_change(21)
    r3 = prices.pct_change(63)
    r6 = prices.pct_change(126)
    r12 = prices.pct_change(252)
    return (r1 + r3 + r6 + r12) / 4


class KellerHAA(Model):
    """Keller Hybrid Asset Allocation.

    Args:
        offensive: list of 8 offensive tickers (default: SPY/IWM/EFA/EEM/VNQ/PDBC/IEF/TLT).
        canary: single canary ticker (default: TIP).
        defensive: defensive ticker for risk-off / negative-momentum slices (default: IEF).
        top_n: number of offensive picks when risk-on (default: 4 per HAA spec).
    """
    name = "keller_haa"

    def __init__(
        self,
        offensive: list[str] | None = None,
        canary: str = HAA_CANARY,
        defensive: str = HAA_DEFENSIVE,
        top_n: int = 4,
    ):
        self.offensive = offensive or list(HAA_OFFENSIVE)
        self.canary = canary
        self.defensive = defensive
        self.top_n = top_n

    def signal(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        vol: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # All assets must be present.
        needed = set(self.offensive) | {self.canary, self.defensive}
        missing = needed - set(prices.columns)
        if missing:
            raise ValueError(f"KellerHAA: missing required tickers in prices: {missing}")

        # Compute momentum scores for canary + offensive
        canary_mom = _avg_1_3_6_12_momentum(prices[self.canary])
        offensive_moms = pd.DataFrame({
            t: _avg_1_3_6_12_momentum(prices[t]) for t in self.offensive
        })

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        slice_weight = 1.0 / self.top_n  # 0.25 for top_n=4

        for date in prices.index:
            cm = canary_mom.loc[date]
            if pd.isna(cm):
                continue  # warmup
            if cm <= 0:
                # Risk-off: 100% defensive
                weights.at[date, self.defensive] = 1.0
                continue
            # Risk-on: pick top-N by momentum
            row = offensive_moms.loc[date].dropna()
            if row.empty:
                continue
            top = row.nlargest(self.top_n)
            for ticker, mom in top.items():
                if mom > 0:
                    weights.at[date, ticker] += slice_weight
                else:
                    # Slice goes to defensive instead
                    weights.at[date, self.defensive] += slice_weight

        return weights
