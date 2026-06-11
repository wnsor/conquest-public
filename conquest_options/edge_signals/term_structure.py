"""VIX term-structure regime classifier.

cstability already uses VIX/VIX3M backwardation as one of its 3 votes. We
add the same signal here so options strategies can hard-gate entries
during stress regimes (backwardation = expectation of near-term realized
vol > forward vol = panic).

Definitions:
    contango        VIX < VIX3M           normal; calm forward expectation
    flat            |VIX - VIX3M| < 1     ambiguous
    backwardation   VIX > VIX3M           stress; near-term fear elevated

VIX9D/VIX < 1 is an even stronger backwardation tell (sub-monthly stress).
"""
from __future__ import annotations


def compute_term_ratio(vix: float | None, vix3m: float | None) -> float | None:
    """VIX/VIX3M ratio; >1 = backwardation, <1 = contango."""
    if vix is None or vix3m is None or vix3m <= 0:
        return None
    return vix / vix3m


def classify_term_regime(
    vix: float | None,
    vix3m: float | None,
    *,
    flat_band: float = 0.02,
) -> str:
    """Return 'contango' | 'flat' | 'backwardation' | 'unknown'.

    flat_band: |ratio - 1| < flat_band → flat. Default 2% so daily noise
    doesn't toggle the regime.
    """
    ratio = compute_term_ratio(vix, vix3m)
    if ratio is None:
        return "unknown"
    if ratio > 1 + flat_band:
        return "backwardation"
    if ratio < 1 - flat_band:
        return "contango"
    return "flat"


def is_acute_stress(
    vix9d: float | None,
    vix: float | None,
    threshold: float = 1.0,
) -> bool:
    """VIX9D < VIX is the classic 'panic now' signal — even shorter-dated
    expectation > monthly. Used to hard-gate new directional entries."""
    if vix9d is None or vix is None or vix <= 0:
        return False
    return (vix9d / vix) > threshold
