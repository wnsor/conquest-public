"""FrogInPanFilter behavior tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.models.frog_in_the_pan import FrogInPanFilter
from conquest.models.dual_momentum import DualMomentum


@pytest.fixture
def synth_universe():
    """4 stocks: 2 with steady drift (continuous info), 2 with jumpy drift (discrete)."""
    rng = np.random.default_rng(17)
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    # Steady up drifters: small daily +drift, low vol
    a = (1 + rng.normal(0.0010, 0.005, len(idx))).cumprod() * 100
    b = (1 + rng.normal(0.0008, 0.005, len(idx))).cumprod() * 100
    # Jumpy up drifters: same final return, but via rare big jumps
    c_rets = np.zeros(len(idx))
    c_rets[::50] = 0.10  # big up moves every 50 days
    c_rets[1:] += rng.normal(-0.0005, 0.003, len(idx) - 1)  # small down drift between jumps
    c = (1 + c_rets).cumprod() * 100
    d_rets = np.zeros(len(idx))
    d_rets[::40] = 0.08  # big up moves every 40 days
    d_rets[1:] += rng.normal(-0.0003, 0.003, len(idx) - 1)
    d = (1 + d_rets).cumprod() * 100
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=idx)


def test_steady_drifters_preferred_over_jumpy(synth_universe):
    """Among 4 picks (all positive momentum), FIP should keep the steady ones (A, B)."""
    base = DualMomentum(top_n=4, lookback=180)
    fip = FrogInPanFilter(base, top_k_filter=2, lookback=180)
    weights = fip.signal(synth_universe)
    post = weights.iloc[300:]
    a_avg = post["A"].mean()
    b_avg = post["B"].mean()
    c_avg = post["C"].mean()
    d_avg = post["D"].mean()
    # Steady drifters should be selected more
    assert a_avg + b_avg > c_avg + d_avg, \
        f"Steady drifters A+B selected {a_avg+b_avg:.2f} vs jumpy C+D {c_avg+d_avg:.2f}"


def test_top_k_filter_respected(synth_universe):
    """If we restrict to top-2, exactly 2 names should be held when warm."""
    base = DualMomentum(top_n=4, lookback=180)
    fip = FrogInPanFilter(base, top_k_filter=2, lookback=180)
    weights = fip.signal(synth_universe)
    post = weights.iloc[300:]
    n_held = (post > 0).sum(axis=1)
    # Should hold ≤ 2 names per row (could be fewer if base picks fewer)
    assert (n_held <= 2).all()


def test_warmup_returns_zero_or_passthrough(synth_universe):
    """Before lookback days of FIP data, should fall back to base's picks
    (warmup behavior; test that nothing crashes)."""
    base = DualMomentum(top_n=2, lookback=180)
    fip = FrogInPanFilter(base, top_k_filter=2, lookback=180)
    weights = fip.signal(synth_universe)
    # No NaN in weights
    assert not weights.isna().any().any()
    # Sums are bounded [0, 1] within tiny float tolerance
    assert (weights.sum(axis=1).max() <= 1.0 + 1e-9)


def test_passthrough_when_top_k_equals_base(synth_universe):
    """If top_k_filter == base.top_n, FIP shouldn't drop any picks."""
    base = DualMomentum(top_n=2, lookback=180)
    fip = FrogInPanFilter(base, top_k_filter=2, lookback=180)
    base_w = base.signal(synth_universe)
    fip_w = fip.signal(synth_universe)
    post = base_w.iloc[300:]
    fip_post = fip_w.iloc[300:]
    # Number of held names per row should match (filter is a no-op when top_k = base.top_n)
    n_base = (post > 0).sum(axis=1)
    n_fip = (fip_post > 0).sum(axis=1)
    # In post-warmup region they should match exactly (allow 1 difference for FIP-NaN cases)
    diff = (n_base - n_fip).abs().max()
    assert diff <= 1, f"Filter dropped picks unexpectedly: max diff {diff}"
