"""Pick a specific OptionContract from a Lean OptionChain given a signal.

A StrategySignal specifies intent (30d ATM call on SPY) but not a contract.
This module turns intent into a contract by scoring all chain candidates
along (dte_distance, target_match, mid_price_validity, liquidity) and
returning the best.

Two strike-target modes (mutually exclusive in StrategySignal):
  - target_delta: pick the contract whose delta is closest to target.
    Sign matters: +0.50 = ~ATM call; -0.20 = OTM put.
  - target_otm_pct: pick the contract whose strike is target_otm_pct
    away from spot in the direction implied by side.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from strategies.base import Side, StrategySignal


@dataclass(frozen=True)
class PickedContract:
    """Lean-agnostic projection of the chosen contract."""
    contract: object         # raw QC OptionContract; we don't import here
    symbol_str: str
    strike: float
    expiry: date
    side_is_call: bool
    delta: float | None
    bid: float
    ask: float
    mid: float
    open_interest: int
    volume: int


def _normalize_side(side: Side) -> tuple[bool, bool]:
    """Return (is_call, is_two_leg). Two-leg sides aren't yet selected here —
    main algorithm handles straddle/strangle by calling selector twice."""
    if side in ("call", "leaps_call"):
        return True, False
    if side in ("put", "leaps_put"):
        return False, False
    if side in ("straddle", "strangle"):
        # Single-call leg returned; caller invokes again for the put leg.
        return True, True
    raise ValueError(f"unknown side {side}")


def pick_contract(
    chain,                          # QC OptionChain (iterable of OptionContract)
    signal: StrategySignal,
    *,
    spot: float,
    now: date,
    dte_tolerance: int | None = None,
    min_open_interest: int | None = None,
    min_volume: int = 0,
    max_spread_pct: float | None = None,
) -> PickedContract | None:
    """Iterate the chain, filter to side + DTE window, pick the best by score.

    Filter defaults scale with target_dte — LEAPS (180+ DTE) have wider
    spreads and lower OI than short-dated options, so the original
    "30% max spread + 25 min OI + 7-day tolerance" defaults rejected
    every candidate for D2 Tepper / CrisisReboundBasket in March 2020.
    v22 fix scales these inversely with DTE.

    Returns None if no contract qualifies (e.g. chain empty around target).
    """
    is_call, _ = _normalize_side(signal.side)

    # v22 fix: tier filters by target_dte. Short-dated options have tight
    # spreads + deep OI; LEAPS have neither. Use the strictest filter the
    # contract is likely to clear.
    tdte = signal.target_dte
    if dte_tolerance is None:
        dte_tolerance = 7 if tdte <= 45 else (14 if tdte <= 120 else 21)
    if min_open_interest is None:
        min_open_interest = 25 if tdte <= 45 else (10 if tdte <= 120 else 5)
    if max_spread_pct is None:
        max_spread_pct = 0.30 if tdte <= 45 else (0.50 if tdte <= 120 else 0.70)

    candidates = []
    for c in chain:
        # Side filter
        contract_is_call = (getattr(c, "Right", None) == 0 or
                            str(getattr(c, "Right", "")).lower() == "call")
        if contract_is_call != is_call:
            continue

        expiry = c.Expiry.date() if hasattr(c.Expiry, "date") else c.Expiry
        dte = (expiry - now).days
        if dte <= 0:
            continue
        if abs(dte - signal.target_dte) > dte_tolerance and signal.target_dte > 0:
            continue

        bid = float(getattr(c, "BidPrice", 0) or 0)
        ask = float(getattr(c, "AskPrice", 0) or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            continue

        oi = int(getattr(c, "OpenInterest", 0) or 0)
        if oi < min_open_interest:
            continue
        # v3: require recent volume so market-orders don't fill on stale chain
        vol = int(getattr(c, "Volume", 0) or 0)
        if vol < min_volume:
            continue
        # v22: tier max_spread by DTE; LEAPS naturally have wider spreads
        if mid > 0 and (ask - bid) / mid > max_spread_pct:
            continue

        candidates.append((c, expiry, dte, bid, ask, mid, oi))

    if not candidates:
        return None

    if signal.target_delta is not None:
        target_d = signal.target_delta
        def score(item):
            c = item[0]
            d = _safe_delta(c)
            if d is None:
                return float("inf")
            return abs(d - target_d) + 0.01 * abs(item[2] - signal.target_dte)
    elif signal.target_otm_pct is not None:
        otm_dir = 1.0 if is_call else -1.0
        target_strike = spot * (1.0 + otm_dir * signal.target_otm_pct)
        def score(item):
            c = item[0]
            return abs(float(c.Strike) - target_strike) + 0.01 * abs(item[2] - signal.target_dte)
    else:
        # ATM fallback
        def score(item):
            c = item[0]
            return abs(float(c.Strike) - spot) + 0.01 * abs(item[2] - signal.target_dte)

    best = min(candidates, key=score)
    c, expiry, dte, bid, ask, mid, oi = best
    return PickedContract(
        contract=c,
        symbol_str=str(c.Symbol),
        strike=float(c.Strike),
        expiry=expiry,
        side_is_call=is_call,
        delta=_safe_delta(c),
        bid=bid,
        ask=ask,
        mid=mid,
        open_interest=oi,
        volume=int(getattr(c, "Volume", 0) or 0),
    )


def _safe_delta(contract) -> float | None:
    """Lean exposes Greeks as `contract.Greeks.Delta` when option model is
    enabled; can be None when the chain doesn't carry greeks (older slices)."""
    try:
        greeks = getattr(contract, "Greeks", None)
        if greeks is None:
            return None
        d = getattr(greeks, "Delta", None)
        return None if d is None else float(d)
    except Exception:
        return None
