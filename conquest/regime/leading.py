"""OECD CLI / CCI / BCI augmentation hooks.

Phase 1 stub: the regime classifier in `classifier.py` runs on GDP and CPI alone.
Phase 2 wires forward-leading indicators here as additional features that can
adjust regime confidence or shift the boundary (e.g. detect turning points
earlier than coincident GDP/CPI alone).

This file is intentionally minimal until OECD CSVs are loaded.
"""
from __future__ import annotations

import pandas as pd


def cli_filter(regime_df: pd.DataFrame, cli: pd.Series) -> pd.DataFrame:
    """Placeholder. Phase 2 will use OECD CLI direction to bias the regime.

    Argument shapes (when implemented):
        regime_df: output of RegimeClassifier.classify (monthly)
        cli:       OECD Composite Leading Indicator (monthly)
    """
    return regime_df  # noop in Phase 1
