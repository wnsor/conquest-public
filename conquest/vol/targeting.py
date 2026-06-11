"""Inverse-vol weighting + portfolio vol-targeting + leverage cap.

Two functions form the v1 sizer stack:

- ``inverse_vol_weights``  — per-row weights ∝ 1/σᵢ across qualified symbols,
  normalized to sum to 1.
- ``vol_target_scale``     — scale a weights matrix so each row's portfolio vol
  approximates ``target_vol``, subject to a leverage cap.

Approximation: portfolio vol = sqrt(Σ wᵢ²σᵢ²) (zero-correlation assumption).
This is the standard linear-models-first sizer; v2+ may swap in a covariance
estimator if the linear baseline shows obvious failure modes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(
    vol: pd.DataFrame,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-row weight ∝ 1/σᵢ across symbols in `mask`. Result rows sum to 1
    (or 0 if no qualified symbol)."""
    if mask is None:
        mask = pd.DataFrame(True, index=vol.index, columns=vol.columns)
    inv = (1.0 / vol).where(mask, 0).replace([np.inf, -np.inf], np.nan).fillna(0)
    row_sum = inv.sum(axis=1)
    return inv.div(row_sum.replace(0, np.nan), axis=0).fillna(0)


def vol_target_scale(
    weights: pd.DataFrame,
    vol: pd.DataFrame,
    target_vol: float = 0.10,
    leverage_cap: float = 1.5,
) -> pd.DataFrame:
    """Scale `weights` so each row's portfolio vol ≈ `target_vol`; cap leverage.

    portfolio_vol(row_t) ≈ sqrt(Σᵢ wᵢ² σᵢ²)   (zero-correlation simplification)

    Returns:
        Scaled weights with the same shape; rows where vol stack is degenerate
        (all-zero portfolio vol) are returned unchanged at zero.
    """
    pv = np.sqrt(((weights ** 2) * (vol ** 2)).sum(axis=1))
    scale = (target_vol / pv).where(pv > 0, 0).clip(upper=leverage_cap)
    return weights.mul(scale, axis=0)
