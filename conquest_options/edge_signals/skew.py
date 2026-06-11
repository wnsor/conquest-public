"""25-delta put-call IV spread (vol skew).

Skew = IV(25-delta put) - IV(25-delta call) at the front-month expiry.
Positive skew = puts expensive vs calls = downside hedging demand.

We surface raw skew per ticker and a z-score vs trailing 252d. Extreme
positive skew z (≥+2) suggests crowded hedging — contrarian setup to BUY
calls (everyone else is too defensive). Extreme negative skew z (≤-2)
suggests complacency / call buying frenzy — caveat A1/A8 entries.

Per-ticker rolling history lives in SkewTracker for the live algorithm.
Pure compute functions are testable on synthetic chains.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable


def _best_match_iv(
    chain_items: Iterable,
    *,
    is_call: bool,
    target_delta: float,
    front_dte_max: int = 45,
    now_date,
    spot: float | None = None,
) -> float | None:
    """Find front-month contract closest to target_delta and return its IV.

    target_delta: signed; +0.25 for 25-delta call, -0.25 for 25-delta put.
    Returns None when no suitable contract.

    2026-05-27: at QC daily resolution, greeks.ImpliedVolatility populates 0%
    (probe-confirmed). Without greeks-IV, the original delta-matching path
    returned None for every contract. New fallback path:
      1) If greeks.delta + greeks.iv available → original behavior.
      2) Else compute delta proxy from strike-distance to spot
         (25-delta ≈ ~8% OTM for ~30-DTE), and compute IV via BS-inverse
         from market price.
      3) Pick the contract with the smallest strike-distance score AND
         a valid BS-inverse IV → return that IV.
    """
    items = list(chain_items)
    # Pass 1: try the original greeks-based matching
    best_iv_greeks = None
    best_score_greeks = float("inf")
    for c in items:
        right = getattr(c, "Right", None)
        if right is None:
            right = getattr(c, "right", None)
        contract_is_call = (right == 0) or (str(right).lower() == "call")
        if contract_is_call != is_call:
            continue
        expiry = c.Expiry.date() if hasattr(c.Expiry, "date") else c.Expiry
        dte = (expiry - now_date).days
        if dte <= 0 or dte > front_dte_max:
            continue
        greeks = getattr(c, "Greeks", None) or getattr(c, "greeks", None)
        delta = None
        iv = None
        if greeks is not None:
            delta = getattr(greeks, "Delta", None) or getattr(greeks, "delta", None)
            iv = getattr(greeks, "ImpliedVolatility", None) or getattr(greeks, "implied_volatility", None)
        if delta is not None and iv is not None and iv > 0:
            score = abs(float(delta) - target_delta) + 0.01 * dte
            if score < best_score_greeks:
                best_score_greeks = score
                best_iv_greeks = float(iv)
    if best_iv_greeks is not None:
        return best_iv_greeks

    # Pass 2: BS-inverse fallback. Use strike-distance to spot as a delta proxy.
    if spot is None or spot <= 0:
        return None
    try:
        from .iv_inverse import solve_iv_cached
    except ImportError:
        return None
    # 25-delta puts are ~5-10% OTM, 25-delta calls similar. Pick contracts
    # whose strike is in the OTM region matching target_delta sign:
    #   call (target_delta > 0): strike > spot
    #   put  (target_delta < 0): strike < spot
    is_otm_side = (target_delta > 0)   # True for call OTM, False for put OTM
    best_iv = None
    best_score = float("inf")
    for c in items:
        right = getattr(c, "Right", None)
        if right is None:
            right = getattr(c, "right", None)
        contract_is_call = (right == 0) or (str(right).lower() == "call")
        if contract_is_call != is_call:
            continue
        K = float(getattr(c, "Strike", 0) or getattr(c, "strike", 0) or 0)
        if K <= 0:
            continue
        moneyness = (K - spot) / spot   # positive = OTM call / ITM put
        if is_otm_side and moneyness <= 0:
            continue
        if (not is_otm_side) and moneyness >= 0:
            continue
        expiry = c.Expiry.date() if hasattr(c.Expiry, "date") else c.Expiry
        dte = (expiry - now_date).days
        if dte <= 0 or dte > front_dte_max:
            continue
        # Pick price (mid > last > price)
        bid = float(getattr(c, "BidPrice", 0) or getattr(c, "bid_price", 0) or 0)
        ask = float(getattr(c, "AskPrice", 0) or getattr(c, "ask_price", 0) or 0)
        if bid > 0 and ask > 0:
            mkt = 0.5 * (bid + ask)
        else:
            mkt = float(getattr(c, "LastPrice", 0) or getattr(c, "last_price", 0)
                        or getattr(c, "Price", 0) or getattr(c, "price", 0) or 0)
        if mkt <= 0:
            continue
        # Score: how close is moneyness to the typical 25-delta level (~8% OTM)?
        target_moneyness = 0.08 if is_call else -0.08
        score = abs(moneyness - target_moneyness) + 0.01 * abs(dte - 30)
        if score >= best_score:
            continue
        # Solve IV
        T = max(1.0 / 365.0, dte / 365.0)
        side = "call" if is_call else "put"
        iv = solve_iv_cached(mkt, spot, K, T, r=0.04, q=0.0, side=side)
        if iv is not None and iv > 0:
            best_score = score
            best_iv = iv
    return best_iv


def compute_skew(
    chain_items,
    *,
    now_date,
    put_target_delta: float = -0.25,
    call_target_delta: float = 0.25,
    spot: float | None = None,
) -> float | None:
    """Return skew = IV(25Δ put) - IV(25Δ call) in vol points. None if any leg missing.

    spot: required for BS-inverse fallback when greeks-IV isn't populated
    (the daily-resolution case). Caller (main.py) should pass current price.
    """
    items = list(chain_items)
    put_iv = _best_match_iv(items, is_call=False, target_delta=put_target_delta,
                             now_date=now_date, spot=spot)
    call_iv = _best_match_iv(items, is_call=True, target_delta=call_target_delta,
                              now_date=now_date, spot=spot)
    if put_iv is None or call_iv is None:
        return None
    return put_iv - call_iv


class SkewTracker:
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._history: dict[str, deque[float]] = {}

    def update(self, ticker: str, skew_value: float | None) -> None:
        if skew_value is None:
            return
        dq = self._history.setdefault(ticker, deque(maxlen=self.lookback_days))
        dq.append(float(skew_value))

    def z_score(self, ticker: str, value: float | None = None) -> float | None:
        h = self._history.get(ticker)
        if not h or len(h) < 5:
            return None
        v = value if value is not None else h[-1]
        n = len(h)
        mean = sum(h) / n
        var = sum((x - mean) ** 2 for x in h) / n
        if var <= 0:
            return 0.0
        return (v - mean) / (var ** 0.5)

    def current(self, ticker: str) -> float | None:
        h = self._history.get(ticker)
        return h[-1] if h else None
