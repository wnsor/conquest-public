"""Tests for v6: SectorCapped wrapper + S&P 500 universe loader (cache only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.models.sector_cap import SectorCapped
from conquest.models.dual_momentum import DualMomentum
from conquest.models.equal_weight import EqualWeight


# ---------- SectorCapped wrapper ----------

@pytest.fixture
def stock_universe():
    """Synthetic 6-stock universe across 3 sectors."""
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = ["TECH_A", "TECH_B", "TECH_C", "FIN_A", "FIN_B", "ENRGY"]
    drifts = [0.001, 0.0009, 0.0008, 0.0003, 0.0002, 0.0001]  # tech > finance > energy
    prices = pd.DataFrame(
        {c: (1 + d + rng.normal(0, 0.012, n)).cumprod() * 100
         for c, d in zip(cols, drifts)},
        index=idx,
    )
    sector_map = {
        "TECH_A": "Information Technology",
        "TECH_B": "Information Technology",
        "TECH_C": "Information Technology",
        "FIN_A": "Financials",
        "FIN_B": "Financials",
        "ENRGY": "Energy",
    }
    return prices, sector_map


def test_sector_capped_no_map_passes_through(stock_universe):
    prices, _ = stock_universe
    base = DualMomentum(top_n=4, lookback=60)
    capped = SectorCapped(base, sector_map=None, max_per_sector=0.30)
    pd.testing.assert_frame_equal(capped.signal(prices), base.signal(prices))


def test_sector_capped_enforces_per_sector_max(stock_universe):
    """With drifts favouring tech, top-3 dual_momentum picks all 3 tech names.
    Equal-weight = 0.333 each, sector_total = 1.0 in tech. Cap at 0.30 must scale down."""
    prices, smap = stock_universe
    base = DualMomentum(top_n=3, lookback=60)
    capped = SectorCapped(base, sector_map=smap, max_per_sector=0.30)

    base_w = base.signal(prices).iloc[-1]
    capped_w = capped.signal(prices).iloc[-1]

    # Sum across tech in capped should be ≤ 0.30 + tiny float tolerance
    tech_cols = [c for c, s in smap.items() if s == "Information Technology"]
    capped_tech_total = capped_w[tech_cols].sum()
    assert capped_tech_total <= 0.30 + 1e-9
    # Base has tech total close to 1.0 (since all 3 picks are tech)
    base_tech_total = base_w[tech_cols].sum()
    assert base_tech_total > 0.30


def test_sector_capped_does_not_inflate_uncapped_sectors(stock_universe):
    """Conservative behaviour: freed weight is NOT redistributed; gross may decrease."""
    prices, smap = stock_universe
    base = DualMomentum(top_n=3, lookback=60)
    capped = SectorCapped(base, sector_map=smap, max_per_sector=0.30)
    base_w = base.signal(prices).iloc[-1]
    capped_w = capped.signal(prices).iloc[-1]
    # Capped gross should be <= base gross
    assert capped_w.abs().sum() <= base_w.abs().sum() + 1e-9


def test_sector_capped_rejects_invalid_max():
    base = EqualWeight()
    with pytest.raises(ValueError):
        SectorCapped(base, max_per_sector=0.0)
    with pytest.raises(ValueError):
        SectorCapped(base, max_per_sector=1.5)


def test_sector_capped_name_pattern():
    base = DualMomentum(top_n=10)
    wrapped = SectorCapped(base, sector_map={"A": "Tech"})
    assert wrapped.name == "dual_momentum_sector_capped"


# ---------- S&P 500 loader (cache-only smoke test; no network) ----------

def test_sp500_cache_loads_if_present(tmp_path, monkeypatch):
    """If cache exists, fetch_sp500 should NOT hit the network."""
    import conquest.data.sp500 as sp500_mod
    fake_cache = tmp_path / "fake_sp500.parquet"
    fake_df = pd.DataFrame({
        "ticker":            ["AAA", "BBB", "CCC"],
        "security_name":     ["Alpha", "Beta", "Gamma"],
        "gics_sector":       ["Tech", "Tech", "Finance"],
        "gics_sub_industry": ["X", "Y", "Z"],
        "hq_location":       ["NYC", "SFO", "BOS"],
    })
    fake_df.to_parquet(fake_cache)
    monkeypatch.setattr(sp500_mod, "CACHE", fake_cache)
    # Should hit cache, not network
    out = sp500_mod.fetch_sp500()
    assert len(out) == 3
    assert "ticker" in out.columns
    sm = sp500_mod.sector_map()
    assert sm == {"AAA": "Tech", "BBB": "Tech", "CCC": "Finance"}
