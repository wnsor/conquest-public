"""Black-Scholes put pricing + Greeks correctness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.options.pricing import (
    BlackScholes, bs_put_price, bs_put_delta, bs_put_gamma, bs_put_vega,
)


def _bs_call_price(S, K, T, r, sigma, q=0.0):
    """Reference call price (we don't ship one, but parity needs it)."""
    from scipy.stats import norm
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def test_put_call_parity():
    """C - P = S*e^{-qT} - K*e^{-rT}"""
    S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.05, 0.20, 0.0
    P = float(bs_put_price(S, K, T, r, sigma, q))
    C = _bs_call_price(S, K, T, r, sigma, q)
    parity_lhs = C - P
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(parity_lhs - parity_rhs) < 1e-6


def test_atm_put_delta_in_range():
    """Delta of ATM put should be near -0.5 (slightly above due to drift)."""
    delta = float(bs_put_delta(100.0, 100.0, 0.25, 0.0, 0.20))
    assert -0.55 <= delta <= -0.45


def test_deep_otm_put_delta_to_zero():
    delta = float(bs_put_delta(100.0, 50.0, 0.25, 0.0, 0.20))
    assert -0.01 < delta <= 0.0


def test_deep_otm_put_gamma_to_zero():
    gamma = float(bs_put_gamma(100.0, 50.0, 0.25, 0.0, 0.20))
    assert 0.0 <= gamma < 1e-3


def test_vega_zero_at_expiry():
    v = float(bs_put_vega(100.0, 100.0, 0.0, 0.0, 0.20))
    assert v == 0.0


def test_intrinsic_at_expiry():
    """At T=0, put price is max(K - S, 0)."""
    assert float(bs_put_price(90.0, 100.0, 0.0, 0.0, 0.20)) == pytest.approx(10.0)
    assert float(bs_put_price(110.0, 100.0, 0.0, 0.0, 0.20)) == pytest.approx(0.0)
    assert float(bs_put_price(100.0, 100.0, 0.0, 0.0, 0.20)) == pytest.approx(0.0)


def test_monotone_in_sigma():
    """Put price strictly increases in sigma (vega > 0 for live options)."""
    S, K, T, r = 100.0, 95.0, 0.25, 0.0
    p_low = float(bs_put_price(S, K, T, r, 0.10))
    p_mid = float(bs_put_price(S, K, T, r, 0.20))
    p_high = float(bs_put_price(S, K, T, r, 0.40))
    assert p_low < p_mid < p_high


def test_vectorized_input_returns_array():
    S = np.array([90.0, 100.0, 110.0])
    p = bs_put_price(S, 100.0, 0.25, 0.0, 0.20)
    assert hasattr(p, "shape")
    assert p.shape == (3,)
    # Sanity: ITM > ATM > OTM
    assert p[0] > p[1] > p[2]


def test_pandas_series_input_returns_series():
    S = pd.Series([90.0, 100.0, 110.0], index=pd.date_range("2024-01-01", periods=3))
    p = bs_put_price(S, 100.0, 0.25, 0.0, 0.20)
    assert isinstance(p, pd.Series)
    assert p.index.equals(S.index)


def test_blackscholes_dataclass_consistency():
    """The BlackScholes wrapper must agree with the bare functions."""
    bs = BlackScholes(r=0.04, q=0.0)
    args = (100.0, 95.0, 0.25, 0.20)
    assert float(bs.put_price(*args)) == pytest.approx(float(bs_put_price(100.0, 95.0, 0.25, 0.04, 0.20)))
    assert float(bs.put_delta(*args)) == pytest.approx(float(bs_put_delta(100.0, 95.0, 0.25, 0.04, 0.20)))
