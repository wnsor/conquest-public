"""Black-Scholes pricing for European puts (+ Greeks), vectorized.

Convention
----------
- All inputs may be scalars, 1-D arrays, or pandas Series. Outputs follow the
  broadcast shape. Time `T` is in years (use trading-day count / 252 for
  research consistency with the rest of conquest).
- We price European puts directly via Black-Scholes (no early-exercise
  premium). SPY puts are American but for OTM puts on a non-dividend-paying
  ETF the early-exercise value is ~0 over normal regimes; the bias is small
  and conservative for our use (slight under-pricing of premium drag).
- Risk-free rate `r` is annualized, continuously compounded. Dividend yield
  `q` defaults to 0 (we handle SPY's dividend on the equity sleeve, not in
  the put pricing — slightly under-prices puts by ~q*T*S, ~0.3% of premium
  for 3M tenor).
- At T=0 we return intrinsic value; Greeks degenerate cleanly (delta = -1
  for ITM, 0 for OTM; gamma/vega = 0 in both cases).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def _as_array(x):
    """Cast scalars/Series/arrays to a numpy array, remember the original shape."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float), x.index
    arr = np.asarray(x, dtype=float)
    return arr, None


def _wrap(out: np.ndarray, index) -> np.ndarray | pd.Series:
    if index is not None:
        return pd.Series(out, index=index)
    return out


def _d1_d2(S, K, T, r, sigma, q):
    # Avoid divide-by-zero. Caller masks T==0 separately.
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_put_price(S, K, T, r=0.0, sigma=0.20, q=0.0):
    """European put price. Args broadcast.

    At T <= 0: returns intrinsic max(K - S, 0).
    At sigma <= 0: returns intrinsic discounted cash flow max(K*e^{-rT} - S, 0)
    (zero-vol limit).
    """
    S_a, idx = _as_array(S)
    K_a, _ = _as_array(K)
    T_a, _ = _as_array(T)
    r_a, _ = _as_array(r)
    sig_a, _ = _as_array(sigma)
    q_a, _ = _as_array(q)

    expired = T_a <= 0
    zero_vol = (sig_a <= 0) & (~expired)
    live = ~expired & ~zero_vol

    out = np.zeros(np.broadcast(S_a, K_a, T_a, r_a, sig_a, q_a).shape, dtype=float)
    # Intrinsic at expiry.
    if expired.any() or out.size == 0:
        intrinsic = np.maximum(K_a - S_a, 0.0)
        out = np.where(expired, np.broadcast_to(intrinsic, out.shape), out)
    # Zero-vol degenerate.
    if zero_vol.any():
        zv = np.maximum(K_a * np.exp(-r_a * T_a) - S_a * np.exp(-q_a * T_a), 0.0)
        out = np.where(zero_vol, np.broadcast_to(zv, out.shape), out)
    if live.any():
        d1, d2 = _d1_d2(S_a, K_a, T_a, r_a, sig_a, q_a)
        bs = K_a * np.exp(-r_a * T_a) * norm.cdf(-d2) - S_a * np.exp(-q_a * T_a) * norm.cdf(-d1)
        out = np.where(live, np.broadcast_to(bs, out.shape), out)
    return _wrap(out, idx)


def bs_put_delta(S, K, T, r=0.0, sigma=0.20, q=0.0):
    """∂P/∂S. Negative; range (-1, 0). At expiry: -1 ITM, 0 OTM."""
    S_a, idx = _as_array(S)
    K_a, _ = _as_array(K)
    T_a, _ = _as_array(T)
    r_a, _ = _as_array(r)
    sig_a, _ = _as_array(sigma)
    q_a, _ = _as_array(q)
    expired = T_a <= 0
    out = np.zeros(np.broadcast(S_a, K_a, T_a, r_a, sig_a, q_a).shape, dtype=float)
    if expired.any():
        intr_delta = np.where(K_a > S_a, -1.0, 0.0)
        out = np.where(expired, np.broadcast_to(intr_delta, out.shape), out)
    live = ~expired
    if live.any():
        d1, _ = _d1_d2(S_a, K_a, T_a, r_a, sig_a, q_a)
        delta_live = np.exp(-q_a * T_a) * (norm.cdf(d1) - 1.0)
        out = np.where(live, np.broadcast_to(delta_live, out.shape), out)
    return _wrap(out, idx)


def bs_put_gamma(S, K, T, r=0.0, sigma=0.20, q=0.0):
    """∂²P/∂S². Same as call gamma. Zero at expiry."""
    S_a, idx = _as_array(S)
    K_a, _ = _as_array(K)
    T_a, _ = _as_array(T)
    r_a, _ = _as_array(r)
    sig_a, _ = _as_array(sigma)
    q_a, _ = _as_array(q)
    expired = T_a <= 0
    out = np.zeros(np.broadcast(S_a, K_a, T_a, r_a, sig_a, q_a).shape, dtype=float)
    live = ~expired
    if live.any():
        d1, _ = _d1_d2(S_a, K_a, T_a, r_a, sig_a, q_a)
        sqrtT = np.sqrt(np.maximum(T_a, 1e-12))
        g_live = np.exp(-q_a * T_a) * norm.pdf(d1) / (S_a * sig_a * sqrtT)
        out = np.where(live, np.broadcast_to(g_live, out.shape), out)
    return _wrap(out, idx)


def bs_put_vega(S, K, T, r=0.0, sigma=0.20, q=0.0):
    """∂P/∂σ per 1.0 vol-point change (i.e. 100 vol pts → multiply by 100).
    Zero at expiry."""
    S_a, idx = _as_array(S)
    K_a, _ = _as_array(K)
    T_a, _ = _as_array(T)
    r_a, _ = _as_array(r)
    sig_a, _ = _as_array(sigma)
    q_a, _ = _as_array(q)
    expired = T_a <= 0
    out = np.zeros(np.broadcast(S_a, K_a, T_a, r_a, sig_a, q_a).shape, dtype=float)
    live = ~expired
    if live.any():
        d1, _ = _d1_d2(S_a, K_a, T_a, r_a, sig_a, q_a)
        sqrtT = np.sqrt(np.maximum(T_a, 1e-12))
        v_live = S_a * np.exp(-q_a * T_a) * norm.pdf(d1) * sqrtT
        out = np.where(live, np.broadcast_to(v_live, out.shape), out)
    return _wrap(out, idx)


@dataclass
class BlackScholes:
    """Thin ergonomic wrapper for notebook use; carries r/q defaults."""
    r: float = 0.0
    q: float = 0.0

    def put_price(self, S, K, T, sigma):
        return bs_put_price(S, K, T, self.r, sigma, self.q)

    def put_delta(self, S, K, T, sigma):
        return bs_put_delta(S, K, T, self.r, sigma, self.q)

    def put_gamma(self, S, K, T, sigma):
        return bs_put_gamma(S, K, T, self.r, sigma, self.q)

    def put_vega(self, S, K, T, sigma):
        return bs_put_vega(S, K, T, self.r, sigma, self.q)
