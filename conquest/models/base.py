"""Strategy interface for the research bake-off."""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    """Map prices (and optional regime/vol context) to per-symbol target weights.

    Each subclass sets ``name`` and implements ``signal``. The result is a DataFrame
    with the same index/columns as ``prices``: each row is a vector of target
    weights summing in absolute value to <= the model's leverage cap (default 1.0).
    Zero or NaN means "flat" in that name on that date.
    """

    name: str = "base"

    @abstractmethod
    def signal(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        vol: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...
