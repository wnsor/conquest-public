"""Equal Weight — naive baseline. Equal weight across every symbol that has a price."""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class EqualWeight(Model):
    name = "equal_weight"

    def signal(self, prices, regime=None, vol=None):
        n_active_per_row = prices.notna().sum(axis=1)
        w = prices.notna().div(n_active_per_row.replace(0, pd.NA), axis=0).astype(float)
        return w.fillna(0)
