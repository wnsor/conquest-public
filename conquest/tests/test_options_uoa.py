"""Unit tests for conquest.options.uoa — the UOA detector pure function."""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.options.uoa import uoa_flag, uoa_flag_series


class TestUoaFlagScalar:
    def test_clear_spike_triggers(self):
        # Baseline ~50, today 500 → 10x vol; OI baseline ~100, today 500 → 5x.
        assert uoa_flag(
            current_volume=500,
            hist_volume_20d=[50] * 20,
            hist_oi_5d=[100] * 5,
        ) is True

    def test_below_vol_multiplier_no_trigger(self):
        # Today 200 vs baseline 50 → 4x < 5x default
        assert uoa_flag(
            current_volume=200,
            hist_volume_20d=[50] * 20,
            hist_oi_5d=[1] * 5,
        ) is False

    def test_below_oi_multiplier_no_trigger(self):
        # Vol-spike passes (10x) but OI baseline = 200 so 500 / 200 = 2.5x < 3x
        assert uoa_flag(
            current_volume=500,
            hist_volume_20d=[50] * 20,
            hist_oi_5d=[200] * 5,
        ) is False

    def test_dead_contract_guard(self):
        # Baseline below min_baseline_volume → no trigger even if multiplier passes
        assert uoa_flag(
            current_volume=50,
            hist_volume_20d=[1] * 20,
            hist_oi_5d=[1] * 5,
        ) is False

    def test_zero_volume_today(self):
        assert uoa_flag(
            current_volume=0,
            hist_volume_20d=[50] * 20,
            hist_oi_5d=[100] * 5,
        ) is False

    def test_zero_oi_baseline(self):
        assert uoa_flag(
            current_volume=500,
            hist_volume_20d=[50] * 20,
            hist_oi_5d=[0] * 5,
        ) is False

    def test_nan_in_history_handled(self):
        # NaNs should be ignored, not propagate
        assert uoa_flag(
            current_volume=500,
            hist_volume_20d=[50] * 18 + [np.nan, np.nan],
            hist_oi_5d=[100, np.nan, 100, 100, 100],
        ) is True

    def test_custom_multipliers(self):
        # 4x passes when multiplier is lowered to 3
        assert uoa_flag(
            current_volume=200,
            hist_volume_20d=[50] * 20,
            hist_oi_5d=[20] * 5,
            vol_multiplier=3.0,
            oi_multiplier=3.0,
        ) is True


class TestUoaFlagSeries:
    def test_basic_series_no_lookahead(self):
        # Build a series with one clear spike at index 25
        n = 40
        volume = pd.Series([50.0] * n)
        oi = pd.Series([100.0] * n)
        volume.iloc[25] = 1000.0  # 20x spike
        flags = uoa_flag_series(volume, oi)
        assert flags.iloc[25] is np.True_ or flags.iloc[25] == True
        assert flags.iloc[24] == False
        assert flags.iloc[26] == False  # next day already counts the spike in baseline

    def test_aligns_length(self):
        volume = pd.Series([1.0, 2.0])
        oi = pd.Series([1.0])
        try:
            uoa_flag_series(volume, oi)
        except ValueError:
            return
        raise AssertionError("expected ValueError on mismatched lengths")

    def test_warm_up_no_false_positive(self):
        # First few entries: rolling baseline is small/NaN → no false positive
        volume = pd.Series([100, 100, 100, 100, 100, 100])
        oi = pd.Series([100, 100, 100, 100, 100, 100])
        flags = uoa_flag_series(volume, oi, vol_window=5, oi_window=3)
        # Steady-state, no spike anywhere
        assert not flags.any()
