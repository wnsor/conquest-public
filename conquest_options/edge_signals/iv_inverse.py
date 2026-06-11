"""BS-inverse implied volatility solver.

Why this exists
---------------
At QC daily resolution, `Greeks.implied_volatility` on individual option
contracts populates 0% of the time (the v28 probe BT confirmed this:
`probe_t_iv_raw` = 0%, `probe_t_iv_hv_ratio` = 0%). Without IV, every
strategy that gates on IV-rank or IV/HV ratio is dead-on-arrival.

This module inverts Black-Scholes: given a contract's market price + the
underlying spot + strike + days-to-expiry + rates, it solves for the
implied volatility sigma via Brent's method on the scalar BS pricer.

Inputs we already have at runtime:
  - underlying spot (`self.securities[ticker].price`)
  - option strike (`contract.strike`)
  - option DTE (`(contract.expiry - now).days`)
  - option price (`contract.last_price` / `bid+ask)/2` / `contract.price`)
  - risk-free rate (constant 0.04 fallback; SOFR/FRED if available)

Performance
-----------
Brent's method on a single contract: ~0.5-2 ms. To avoid blowing up BT
runtime, we:
  1. Solve IV only for the ATM call + ATM put per ticker per day (2 calls,
     not the full chain — gives us 1 ticker-level IV measure)
  2. Cache by (price, S_rounded, K, dte) so identical inputs reuse the
     result (rare at daily, but helps on EOD-bar repeats)

Coverage expected post-fix
--------------------------
Tickers with ANY ATM contract: should be ~85-95% per day for liquid
underlyings (SPY, QQQ, AAPL, etc.). Sparse for some WSB names. The
populate rate is bounded by chain density, not the solver.

Public API
----------
  solve_iv(market_price, S, K, T_years, r=0.04, q=0.0, side="call") -> float | None
      Returns sigma in [0.01, 5.0], or None if no solution exists
      (e.g., market price violates no-arbitrage bounds).

  ticker_atm_iv(chain, spot, today_date, r=0.04) -> float | None
      Convenience: given an option chain + current spot, picks the
      closest-to-ATM call (or any contract if no exact ATM), solves IV,
      returns it. Returns None if chain too sparse.
"""
from __future__ import annotations

import math
from functools import lru_cache

# scipy.optimize.brentq is the robust root-finder for monotone functions.
# Available in the QC runtime (verified via conquest/regime/probability.py).
try:
    from scipy.optimize import brentq
except ImportError:
    brentq = None   # caller handles None-return via solve_iv


SQRT_2PI = math.sqrt(2.0 * math.pi)
_IV_MIN = 0.01    # 1% annual vol floor
_IV_MAX = 5.00    # 500% annual vol ceiling
_BRENT_TOL = 1e-4 # solver tolerance — vol precision to ~1bp


def _norm_cdf(x: float) -> float:
    """Standard normal CDF without numpy (faster for scalar inner loops)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes call price (scalar, no-array fast path)."""
    if T <= 0.0 or sigma <= 0.0:
        return max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes put price via put-call parity."""
    if T <= 0.0 or sigma <= 0.0:
        return max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def solve_iv(
    market_price: float,
    S: float,
    K: float,
    T_years: float,
    r: float = 0.04,
    q: float = 0.0,
    side: str = "call",
) -> float | None:
    """Solve for implied volatility given a market option price.

    Returns sigma in [_IV_MIN, _IV_MAX], or None if no valid solution
    exists in that range (typically means market price violates
    no-arbitrage bounds — stale quote, fat-fingered fill, or zero-bid).
    """
    if brentq is None:
        return None
    if market_price <= 0.0 or S <= 0.0 or K <= 0.0 or T_years <= 0.0:
        return None

    pricer = _bs_call if side == "call" else _bs_put

    # No-arbitrage check: option price must lie within the intrinsic-extrinsic bounds.
    # For a call: max(0, S-K*e^-rT) <= C <= S; for a put: max(0, K*e^-rT - S) <= P <= K*e^-rT.
    if side == "call":
        intrinsic = max(0.0, S * math.exp(-q * T_years) - K * math.exp(-r * T_years))
        upper_bound = S * math.exp(-q * T_years)
    else:
        intrinsic = max(0.0, K * math.exp(-r * T_years) - S * math.exp(-q * T_years))
        upper_bound = K * math.exp(-r * T_years)

    if market_price < intrinsic * 0.99 or market_price > upper_bound * 1.01:
        return None   # outside no-arb bounds — bad quote

    # Define f(sigma) = pricer(S,K,T,r,sigma,q) - market_price; solve f=0.
    def f(sigma: float) -> float:
        return pricer(S, K, T_years, r, sigma, q) - market_price

    # Probe endpoints to ensure they bracket a root (f sign-change).
    try:
        f_lo = f(_IV_MIN)
        f_hi = f(_IV_MAX)
    except Exception:
        return None
    if f_lo * f_hi > 0.0:
        # No sign change → no root in [_IV_MIN, _IV_MAX]
        return None
    try:
        sigma = brentq(f, _IV_MIN, _IV_MAX, xtol=_BRENT_TOL, maxiter=64)
    except (ValueError, RuntimeError):
        return None
    return float(sigma) if _IV_MIN <= sigma <= _IV_MAX else None


def _round_key(x: float, ndigits: int = 2) -> float:
    """Round for cache-key stability (so adjacent ticks share cache hits)."""
    return round(float(x), ndigits)


@lru_cache(maxsize=2048)
def _solve_iv_cached(
    market_price_r: float, S_r: float, K_r: float, T_years_r: float,
    r_r: float, q_r: float, side: str,
) -> float | None:
    """Cached solver. Inputs are pre-rounded to avoid cache-key churn."""
    return solve_iv(market_price_r, S_r, K_r, T_years_r, r_r, q_r, side)


def solve_iv_cached(
    market_price: float, S: float, K: float, T_years: float,
    r: float = 0.04, q: float = 0.0, side: str = "call",
) -> float | None:
    """Cached variant — same as solve_iv but memoized by rounded inputs."""
    return _solve_iv_cached(
        _round_key(market_price, 2), _round_key(S, 2), _round_key(K, 2),
        _round_key(T_years, 4), _round_key(r, 4), _round_key(q, 4), side,
    )


def ticker_atm_iv(
    chain: list,
    spot: float,
    today_date,
    r: float = 0.04,
    prefer_call: bool = True,
    min_dte: int = 7,
    max_dte: int = 60,
) -> float | None:
    """Convenience: pick the closest-to-ATM contract within DTE band + solve IV.

    Args:
        chain: list of QC OptionContract (or any object with .strike, .expiry,
               .right, .last_price / .bid_price / .ask_price)
        spot: current underlying price
        today_date: today (used to compute DTE from contract.expiry)
        r: risk-free rate (default 0.04 = 4% — SOFR-ish)
        prefer_call: which side to invert (call is conventional for IV-rank)
        min_dte / max_dte: filter contracts to a sensible DTE band

    Returns:
        sigma (e.g., 0.18 = 18% annual vol), or None if no usable contract.
    """
    if not chain or spot <= 0.0:
        return None
    side_target = "call" if prefer_call else "put"
    side_code = 0 if prefer_call else 1   # QC OptionRight: Call=0, Put=1

    best = None
    best_dist = float("inf")
    for c in chain:
        try:
            right = getattr(c, "right", None)
            if right is None:
                # Some QC chain objects use string
                right_str = str(getattr(c, "Right", "")).lower()
                if (side_target == "call" and "call" not in right_str) or \
                   (side_target == "put" and "put" not in right_str):
                    continue
            elif right != side_code:
                continue
            K = float(getattr(c, "strike", 0) or 0)
            if K <= 0:
                continue
            expiry = getattr(c, "expiry", None) or getattr(c, "Expiry", None)
            if expiry is None:
                continue
            # expiry can be datetime or date
            expiry_d = expiry.date() if hasattr(expiry, "date") else expiry
            dte = (expiry_d - today_date).days
            if dte < min_dte or dte > max_dte:
                continue
            # Prefer ATM
            dist = abs(K - spot)
            if dist < best_dist:
                # Best-quote price
                price = 0.0
                bid = float(getattr(c, "bid_price", 0) or 0)
                ask = float(getattr(c, "ask_price", 0) or 0)
                if bid > 0 and ask > 0:
                    price = 0.5 * (bid + ask)
                else:
                    price = float(getattr(c, "last_price", 0) or 0) or \
                            float(getattr(c, "price", 0) or 0)
                if price > 0:
                    best = (price, K, dte)
                    best_dist = dist
        except (AttributeError, TypeError, ValueError):
            continue

    if best is None:
        return None
    price, K, dte = best
    T = max(1.0 / 365.0, dte / 365.0)
    return solve_iv_cached(price, spot, K, T, r, 0.0, side_target)
