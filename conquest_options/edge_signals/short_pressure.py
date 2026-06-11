"""QC ShortableProvider wrapper for short-squeeze pressure signals.

Replaces the deprecated FINRA bi-weekly short-interest scraper. Reads
per-ticker `fee_rate` (annualized borrow cost in %) and `shortable_quantity`
(available shares to borrow) via the InteractiveBrokersShortableProvider
that QC exposes through `security.shortable_provider`.

For Strategy A7 (short-squeeze calls), the pressure signal is:
  - high fee_rate (P90+ across the universe today)
  - AND low shortable_quantity (P10-)
  - AND positive 5-day momentum on the underlying

This module just exposes the read interface — universe percentiles and
momentum are computed at strategy level.
"""
from __future__ import annotations


def read_short_metrics(algo, symbol) -> dict:
    """Best-effort read of QC short-availability metrics for one symbol.

    Returns dict with keys `fee_rate` (float, % annualized), `shortable_qty`
    (int), `rebate_rate` (float). Any of these may be None if the data
    point isn't available for that day/ticker.
    """
    security = algo.Securities.get(symbol) if hasattr(algo, "Securities") else None
    if security is None:
        return {"fee_rate": None, "shortable_qty": None, "rebate_rate": None}
    provider = getattr(security, "ShortableProvider", None) or getattr(security, "shortable_provider", None)
    if provider is None:
        return {"fee_rate": None, "shortable_qty": None, "rebate_rate": None}
    now = algo.Time
    try:
        fee = provider.FeeRate(symbol, now)
    except Exception:
        fee = None
    try:
        qty = provider.ShortableQuantity(symbol, now)
    except Exception:
        qty = None
    try:
        rebate = provider.RebateRate(symbol, now)
    except Exception:
        rebate = None
    return {
        "fee_rate": float(fee) if fee is not None else None,
        "shortable_qty": int(qty) if qty is not None else None,
        "rebate_rate": float(rebate) if rebate is not None else None,
    }
