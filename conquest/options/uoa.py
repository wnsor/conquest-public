"""Unusual Options Activity (UOA) detector.

Per-contract heuristic from Vasquez & Xiao (2024) and CheddarFlow practitioner
lore: a contract is "unusual" when today's volume is materially above its
recent baseline AND above recent open interest. We default to the brief A8
rule — vol > 5x 20-day mean vol AND vol > 3x 5-day mean OI — but the
multipliers and lookbacks are parameters so Phase 2 can tune them.

Two entry points:

    uoa_flag(...)            -> bool   single contract, scalar inputs
    uoa_flag_series(...)     -> bool   single contract, pandas series inputs

Inputs use plain numpy/pandas so this is callable from a Lean Algorithm
(per-OnData snapshot) or from an offline pandas DataFrame screen. No
QC dependency.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _mean_safe(arr: Iterable[float]) -> float:
    a = np.asarray(list(arr), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(a.mean())


def uoa_flag(
    current_volume: float,
    hist_volume_20d: Iterable[float],
    hist_oi_5d: Iterable[float],
    vol_multiplier: float = 5.0,
    oi_multiplier: float = 3.0,
    min_baseline_volume: float = 10.0,
) -> bool:
    """Return True if today's contract volume is "unusual" per the rule
    `vol > vol_multiplier * mean(hist_volume) AND vol > oi_multiplier * mean(hist_oi)`.

    Args:
        current_volume: today's contract volume.
        hist_volume_20d: prior N-day daily volumes (defaults to 20 in caller).
        hist_oi_5d: prior N-day daily open interest values (defaults to 5).
        vol_multiplier: vol-spike threshold (default 5x).
        oi_multiplier: OI-spike threshold (default 3x).
        min_baseline_volume: dead-contract guard. If the 20d mean volume is
            below this floor, return False (avoid false positives on
            illiquid strikes where 0 -> 50 looks like ∞x).

    Returns:
        True iff both spike conditions are met and the baseline is healthy.
    """
    if not np.isfinite(current_volume) or current_volume <= 0:
        return False
    vol_baseline = _mean_safe(hist_volume_20d)
    oi_baseline = _mean_safe(hist_oi_5d)
    if vol_baseline < min_baseline_volume or oi_baseline <= 0:
        return False
    return (
        current_volume > vol_multiplier * vol_baseline
        and current_volume > oi_multiplier * oi_baseline
    )


def uoa_flag_series(
    volume: pd.Series,
    open_interest: pd.Series,
    vol_window: int = 20,
    oi_window: int = 5,
    vol_multiplier: float = 5.0,
    oi_multiplier: float = 3.0,
    min_baseline_volume: float = 10.0,
) -> pd.Series:
    """Vectorized version for offline screening of one contract over time.

    For each date t, flag = volume[t] is unusual given the trailing
    `vol_window` volumes and trailing `oi_window` OI values up to and
    including t-1 (no look-ahead).
    """
    if len(volume) != len(open_interest):
        raise ValueError("volume and open_interest must align")
    vol = pd.Series(volume).astype(float)
    oi = pd.Series(open_interest).astype(float)
    vol_baseline = vol.shift(1).rolling(vol_window, min_periods=1).mean()
    oi_baseline = oi.shift(1).rolling(oi_window, min_periods=1).mean()
    cond_vol = vol > vol_multiplier * vol_baseline
    cond_oi = vol > oi_multiplier * oi_baseline
    cond_baseline = (vol_baseline >= min_baseline_volume) & (oi_baseline > 0)
    cond_alive = vol > 0
    return (cond_vol & cond_oi & cond_baseline & cond_alive).fillna(False)
