"""SectorCapped — wrap any base model and enforce a per-sector weight cap.

Why
---
Stock-picking on a 500-name universe by 12-month momentum tends to crowd into
whichever sector ran hardest. Top-10 by 252d return in late 2024 = mostly AI/
semis. Without a sector cap, the "diversified" stock-picker is secretly a
concentrated tech bet — exactly the failure mode HHI was added in v5.5 to
detect, and that this wrapper is meant to prevent at construction time.

Behaviour
---------
For each rebalance row, if any GICS sector's total weight exceeds
``max_per_sector``, the over-weight sector's individual ticker weights are
scaled down proportionally to bring the sector total to the cap. The freed
weight is NOT redistributed — total gross exposure simply decreases, which is
the conservative thing to do (matches the spirit of the regime gate's risk-off
behaviour).
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class SectorCapped(Model):
    def __init__(
        self,
        base_model: Model,
        sector_map: dict[str, str] | None = None,
        max_per_sector: float = 0.30,
    ):
        if not (0 < max_per_sector <= 1):
            raise ValueError("max_per_sector must be in (0, 1]")
        self.base = base_model
        self.sector_map = sector_map or {}
        self.max_per_sector = max_per_sector
        self.name = f"{base_model.name}_sector_capped"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        if not self.sector_map:
            return weights

        # Map each column (ticker) to its sector
        sectors = pd.Series(
            [self.sector_map.get(t, "Unknown") for t in weights.columns],
            index=weights.columns,
            name="sector",
        )

        capped = weights.copy()
        for sector in sectors.unique():
            cols = sectors[sectors == sector].index
            if len(cols) == 0:
                continue
            sector_total = weights[cols].abs().sum(axis=1)
            over_cap = sector_total > self.max_per_sector
            if not over_cap.any():
                continue
            # Scale down the over-cap rows' weights in this sector to the cap
            scale = (self.max_per_sector / sector_total[over_cap])
            capped.loc[over_cap, cols] = (
                weights.loc[over_cap, cols].mul(scale, axis=0)
            )
        return capped
