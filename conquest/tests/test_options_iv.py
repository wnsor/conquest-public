"""VIX-to-SPX-IV interpolation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.options.implied_vol import vix_to_spx_iv, VIX_TENOR_DAYS, VIX3M_TENOR_DAYS


@pytest.fixture
def vix_pair():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    vix = pd.Series([18.0] * 10, index=idx)
    vix3m = pd.Series([22.0] * 10, index=idx)
    return vix, vix3m


def test_atm_at_vix_tenor_returns_vix(vix_pair):
    vix, vix3m = vix_pair
    iv = vix_to_spx_iv(vix, vix3m, tenor_days=VIX_TENOR_DAYS, strike_offset=0.0, skew_per_5pct_otm=0.0)
    assert (iv == 18.0).all()


def test_atm_at_vix3m_tenor_returns_vix3m(vix_pair):
    vix, vix3m = vix_pair
    iv = vix_to_spx_iv(vix, vix3m, tenor_days=VIX3M_TENOR_DAYS, strike_offset=0.0, skew_per_5pct_otm=0.0)
    assert (iv == 22.0).all()


def test_interpolation_between_anchors_is_monotone(vix_pair):
    vix, vix3m = vix_pair
    # vix3m > vix in this fixture, so longer tenor → higher IV
    iv_short = vix_to_spx_iv(vix, vix3m, VIX_TENOR_DAYS + 5, 0.0, 0.0).iloc[0]
    iv_mid = vix_to_spx_iv(vix, vix3m, (VIX_TENOR_DAYS + VIX3M_TENOR_DAYS) // 2, 0.0, 0.0).iloc[0]
    iv_long = vix_to_spx_iv(vix, vix3m, VIX3M_TENOR_DAYS - 5, 0.0, 0.0).iloc[0]
    assert iv_short < iv_mid < iv_long


def test_clamp_above_vix3m(vix_pair):
    """For tenors > VIX3M anchor, clamp to VIX3M."""
    vix, vix3m = vix_pair
    iv = vix_to_spx_iv(vix, vix3m, tenor_days=180, strike_offset=0.0, skew_per_5pct_otm=0.0)
    assert (iv == 22.0).all()


def test_clamp_below_vix_tenor(vix_pair):
    """For tenors < VIX anchor (e.g. weekly), clamp to VIX."""
    vix, vix3m = vix_pair
    iv = vix_to_spx_iv(vix, vix3m, tenor_days=5, strike_offset=0.0, skew_per_5pct_otm=0.0)
    assert (iv == 18.0).all()


def test_skew_premium_adds_at_otm(vix_pair):
    vix, vix3m = vix_pair
    atm = vix_to_spx_iv(vix, vix3m, tenor_days=63, strike_offset=0.0, skew_per_5pct_otm=2.0).iloc[0]
    otm5 = vix_to_spx_iv(vix, vix3m, tenor_days=63, strike_offset=-0.05, skew_per_5pct_otm=2.0).iloc[0]
    otm10 = vix_to_spx_iv(vix, vix3m, tenor_days=63, strike_offset=-0.10, skew_per_5pct_otm=2.0).iloc[0]
    assert otm5 == pytest.approx(atm + 2.0)
    assert otm10 == pytest.approx(atm + 4.0)


def test_misaligned_indices_take_intersection():
    idx_v = pd.date_range("2024-01-01", periods=10, freq="B")
    idx_v3 = pd.date_range("2024-01-03", periods=10, freq="B")
    vix = pd.Series(18.0, index=idx_v)
    vix3m = pd.Series(22.0, index=idx_v3)
    iv = vix_to_spx_iv(vix, vix3m, 63, 0.0, 0.0)
    # intersection has 8 days
    assert len(iv) == 8
