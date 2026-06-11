"""Transaction-cost model approximating Interactive Brokers tiered pricing on US equities.

The exact IB cost on a single equity trade is approximately:

    commission = max($1, $0.005 * shares)
    slippage   ~ half_spread_pct * trade_dollars
    total      = commission + slippage

For typical ETF rebalances (price $50–$500/share, $5k–$100k/leg), this rolls up to
roughly 1–3 bps on dollar turnover. Default here is **2 bps**; override as needed.

Phase 2 can replace this with a per-trade model that counts shares per name and
applies the commission floor properly. For Phase 1 model ranking the simple
turnover-based form is fine — it's directionally correct and fast.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class IBCostModel:
    bps_per_turnover: float = 2.0
    """Cost in basis points per unit of dollar turnover (sum of |Δ weight|).
    A round-trip rebalance (sell A, buy B at equal weight) has turnover = 2 in this units."""

    def cost_fraction(self, turnover: pd.Series) -> pd.Series:
        """Per-bar cost as fraction of portfolio NAV."""
        return turnover * self.bps_per_turnover * 1e-4
