"""Strategy implementations + a uniform Model interface for cross-comparison.

The factory ``all_models`` returns the bake-off lineup (the head-to-head
comparison set in ``scripts/rank_models.py``). v0 is the original 6 models;
``include_v2=True`` extends with regime/VIX/Kelly wrappers; ``include_v4=True``
adds the ADX-gated variants (requires ``ohlc`` to be passed since ADX uses
High/Low/Close, not just Close).
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model
from conquest.models.equal_weight import EqualWeight
from conquest.models.momentum_consensus import MomentumConsensus
from conquest.models.mean_reversion import MeanReversion
from conquest.models.trend_follow import TrendFollow
from conquest.models.dual_momentum import DualMomentum
from conquest.models.ensemble import Ensemble
from conquest.models.regime_gated import RegimeGated
from conquest.models.vix_gated import VixGated
from conquest.models.kelly_sized import KellySized
from conquest.models.adx_gated import AdxGated
from conquest.models.sector_cap import SectorCapped
from conquest.models.vol_targeted import VolTargeted
from conquest.models.dd_throttled import DrawdownThrottled
from conquest.models.multi_horizon_momentum import MultiHorizonMomentum
from conquest.models.low_vol_filtered import LowVolFiltered
from conquest.models.regime_rotator import RegimeRotator
from conquest.models.quality_filtered import QualityFiltered
from conquest.models.vix_rotated import VixRotated
from conquest.models.regime_probability_rotated import RegimeProbabilityRotated
from conquest.models.three_layer_stacked import ThreeLayerStacked
from conquest.models.multi_signal_vote import MultiSignalVote
from conquest.models.keller_haa import KellerHAA
from conquest.models.faber_gtaa import FaberGTAA
from conquest.models.residual_momentum import ResidualMomentum
from conquest.models.frog_in_the_pan import FrogInPanFilter
from conquest.models.multi_beta_residual import MultiBetaResidualMomentum
from conquest.models.adaptive_residual import (
    VolAdaptiveResidualMomentum,
    TwoStageMomentum,
    RegimeAdaptiveResidualMomentum,
)


def all_models(
    *,
    include_v2: bool = False,
    include_v4: bool = False,
    regime: pd.Series | None = None,
    vix: pd.Series | None = None,
    ohlc: dict[str, pd.DataFrame] | None = None,
) -> list[Model]:
    """Default bake-off lineup.

    Args:
        include_v2: True adds 6 regime/vix/kelly-gated wrappers around momentum_consensus
                    and dual_momentum.
        include_v4: True adds 3 ADX-gated wrappers (requires ``ohlc``).
        regime: pd.Series of regime labels (from storage/conquest/regime/daily.csv).
        vix:    pd.Series of VIX closes (from yfinance ^VIX cache).
        ohlc:   dict[ticker, OHLC DataFrame] (from yfinance group_by='ticker' cache).

    Variants without their required context fall back to pass-through (no gating).
    """
    models: list[Model] = [
        EqualWeight(),
        MomentumConsensus(top_n=5, min_score=2),
        MeanReversion(max_positions=5, oversold=30, exit_level=50),
        TrendFollow(top_n=3, momp_period=90),
        DualMomentum(top_n=5, lookback=252),
        Ensemble(top_n=5),
    ]
    if include_v2:
        models.extend([
            RegimeGated(MomentumConsensus(top_n=5, min_score=2), regime_series=regime),
            RegimeGated(DualMomentum(top_n=5, lookback=252),     regime_series=regime),
            VixGated(MomentumConsensus(top_n=5, min_score=2),    vix_series=vix),
            VixGated(DualMomentum(top_n=5, lookback=252),        vix_series=vix),
            KellySized(MomentumConsensus(top_n=5, min_score=2)),
            KellySized(DualMomentum(top_n=5, lookback=252)),
        ])
    if include_v4:
        # ADX gate filters out names that aren't currently trending (ADX < 25).
        # Most useful on trend-style strategies; included on all three momentum
        # families so the bake-off can show which benefits most.
        models.extend([
            AdxGated(TrendFollow(top_n=3, momp_period=90),         ohlc=ohlc),
            AdxGated(MomentumConsensus(top_n=5, min_score=2),       ohlc=ohlc),
            AdxGated(DualMomentum(top_n=5, lookback=252),           ohlc=ohlc),
        ])
    return models


__all__ = [
    "Model",
    "EqualWeight",
    "MomentumConsensus",
    "MeanReversion",
    "TrendFollow",
    "DualMomentum",
    "Ensemble",
    "RegimeGated",
    "VixGated",
    "KellySized",
    "AdxGated",
    "SectorCapped",
    "VolTargeted",
    "DrawdownThrottled",
    "MultiHorizonMomentum",
    "LowVolFiltered",
    "RegimeRotator",
    "QualityFiltered",
    "VixRotated",
    "RegimeProbabilityRotated",
    "ThreeLayerStacked",
    "MultiSignalVote",
    "KellerHAA",
    "FaberGTAA",
    "ResidualMomentum",
    "FrogInPanFilter",
    "MultiBetaResidualMomentum",
    "VolAdaptiveResidualMomentum",
    "TwoStageMomentum",
    "RegimeAdaptiveResidualMomentum",
    "all_models",
]
