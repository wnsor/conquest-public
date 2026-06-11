"""Gamma exposure (GEX) — dealer positioning estimate from a chain.

GEX = sum across all contracts of (Γ × OI × 100 × spot² / 1e9), with calls
*positive* (dealers are net short calls to retail buyers, so positive Γ
when spot rises means they buy underlying) and puts *negative* (dealers are
net long puts to retail buyers).

Output units are billions of dollars per 1% spot move — the standard
SpotGamma convention. Sign indicates regime:

    GEX > 0   long-gamma   dealers dampen vol; spot mean-reverts
    GEX < 0   short-gamma  dealers amplify vol; trends accelerate

For a daily-resolution backtest we use GEX as a regime tag (confluence
input for entries + sizing scalar), not as an intraday level-trade signal.

Pure functions for unit testing; the live wiring lives in main.py and
calls these on the SPY chain each tick.
"""
from __future__ import annotations

from typing import Iterable


def compute_gex_contributions(
    chain_items: Iterable,
    *,
    spot: float,
    multiplier: int = 100,
) -> dict:
    """Sum dealer gamma exposure across the chain.

    Each item is a duck-typed contract with:
      .Strike (float)
      .Right (0=call or 1=put per Lean enum; tolerates str alternatives)
      .OpenInterest (int)
      .Greeks.Gamma (float)

    Returns dict {gex_total, gex_calls, gex_puts, count_used} in $bn / 1% move.
    Returns zeros + count_used=0 if no usable contracts.
    """
    if spot <= 0:
        return {"gex_total": 0.0, "gex_calls": 0.0, "gex_puts": 0.0, "count_used": 0}

    gex_calls = 0.0
    gex_puts = 0.0
    used = 0
    spot_sq = spot * spot
    for c in chain_items:
        oi = int(getattr(c, "OpenInterest", None) or getattr(c, "open_interest", 0) or 0)
        if oi <= 0:
            continue
        greeks = getattr(c, "Greeks", None) or getattr(c, "greeks", None)
        gamma = None
        if greeks is not None:
            gamma = getattr(greeks, "Gamma", None) or getattr(greeks, "gamma", None)
        if gamma is None or gamma <= 0:
            continue
        right = getattr(c, "Right", None)
        if right is None:
            right = getattr(c, "right", None)
        is_call = (right == 0) or (str(right).lower() == "call")
        contribution = gamma * oi * multiplier * spot_sq / 1e9  # $bn per 1% move
        if is_call:
            gex_calls += contribution
        else:
            gex_puts -= contribution  # dealers long puts → negative GEX
        used += 1

    return {
        "gex_total": gex_calls + gex_puts,
        "gex_calls": gex_calls,
        "gex_puts": gex_puts,
        "count_used": used,
    }


def classify_gex_regime(gex_total: float, flip_threshold_bn: float = 0.5) -> str:
    """Map GEX scalar to a regime label.

    flip_threshold_bn defines the band around zero where we call it
    "flip_zone" (conditions are mixed; treat as short-gamma defensively).
    """
    if gex_total > flip_threshold_bn:
        return "long_gamma"
    if gex_total < -flip_threshold_bn:
        return "short_gamma"
    return "flip_zone"
