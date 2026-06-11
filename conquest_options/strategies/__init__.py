"""Strategy plug-in registry for conquest_options/.

A strategy is a plain class (NOT a QCAlgorithm subclass) that:
  - declares its required underlying universe
  - inspects a StrategyContext on each tick
  - emits zero or more StrategySignal objects

The main algorithm composes the registered strategies. Each strategy is
evaluated independently; sizing / contract selection / exit management are
done centrally by the main algorithm via the shared OptionSelector,
PositionSizer, ExitManager components.

Add a new strategy:
  1. Create strategies/<id>.py implementing Strategy protocol.
  2. Import + append to ENABLED_STRATEGIES below.

v22 (2026-05-24): renamed all strategies from academic-paper notation
(A1/A2/A5a/B1/C1/D2) to semantic English names matching the webapp_v2
fund-name convention (cgrowth, cstability, voltgt_v2, combined_v2).
"""
from __future__ import annotations

from strategies.base import (
    Strategy,
    StrategyContext,
    StrategySignal,
    Side,
)
from strategies.momentum_otm_calls import MomentumOtmCalls
from strategies.spy_atm_calls import SpyAtmCalls
from strategies.pead_megacap import PeadMegacap
from strategies.pead_midcap import PeadMidcap
from strategies.insider_buy_calls import InsiderBuyCalls
from strategies.uoa_following_calls import UoaFollowingCalls
from strategies.earnings_straddle import EarningsStraddle
from strategies.earnings_strangle import EarningsStrangle
from strategies.gex_spy_selective import GexSpySelective
from strategies.gex_spy_baseline import GexSpyBaseline
from strategies.cgrowth_leaps import CgrowthLeaps
from strategies.tepper_vbottom_leaps import TepperVbottomLeaps
from strategies.crisis_rebound_basket import CrisisReboundBasket
from strategies.spy_crisis_put import SpyCrisisPut
from strategies.momentum_failure_put import MomentumFailurePut
from strategies.crisis_dual_directional import CrisisDualDirectional
from strategies.momentum_no_catalyst_baseline import MomentumNoCatalystBaseline
# Abstract / forward-looking strategies (2026-05-26):
#   reflex_ignition       — narrative-price feedback loop activation detector
#   dealer_opex_squeeze   — calendar-anchored dealer forced-flow trade
from strategies.reflex_ignition import ReflexIgnition
from strategies.reflex_ignition_v2 import ReflexIgnitionV2
from strategies.dealer_opex_squeeze import DealerOpexSqueeze
# Leading-indicator-only strategies (2026-05-26, per user directive
# "lagging indicators not sufficient for options"):
from strategies.short_squeeze_pure import ShortSqueezePure
from strategies.implied_move_divergence import ImpliedMoveDivergence
from strategies.triple_confluence import TripleConfluence
from strategies.vix_term_recovery import VixTermRecovery
from strategies.network_propagation import NetworkPropagation
# Mix-and-match standby strategies (2026-05-26) — drafted ahead of data feeds
# (VVIX/SKEW/WSB/Google Trends/13D/8-K). Loaded but won't fire until main.py
# populates the relevant context fields from refreshed Object Store CSVs.
from strategies.tail_hedge_regime import TailHedgeRegime
from strategies.vvix_divergence import VvixDivergence
from strategies.retail_attention_cascade import RetailAttentionCascade
from strategies.earnings_revision_momentum import EarningsRevisionMomentum
from strategies.quad_confluence import QuadConfluence
from strategies.activist_drift import ActivistDrift
from strategies.eightk_burst import EightKBurst

# Registry of built strategies. Cloud-backtest activation is controlled
# separately via the ACTIVE_STRATEGY_IDS config parameter in main.py —
# default empty means no strategies fire even though they're all registered
# here. This protects cloud runs from accidentally enabling every strategy.
ENABLED_STRATEGIES: list[Strategy] = [
    MomentumOtmCalls(),
    # SpyAtmCalls(),       # v3: disabled — never triggered in v2.
    PeadMegacap(),
    PeadMidcap(),
    InsiderBuyCalls(),
    UoaFollowingCalls(),
    EarningsStraddle(),
    EarningsStrangle(),
    GexSpySelective(),
    GexSpyBaseline(),
    CgrowthLeaps(),
    TepperVbottomLeaps(),
    CrisisReboundBasket(),
    SpyCrisisPut(),
    MomentumFailurePut(),
    CrisisDualDirectional(),
    MomentumNoCatalystBaseline(),
    ReflexIgnition(),
    ReflexIgnitionV2(),
    DealerOpexSqueeze(),
    ShortSqueezePure(),
    ImpliedMoveDivergence(),
    TripleConfluence(),
    VixTermRecovery(),
    NetworkPropagation(),
    # Standby — wait for new data ingesters
    TailHedgeRegime(),
    VvixDivergence(),
    RetailAttentionCascade(),
    EarningsRevisionMomentum(),
    QuadConfluence(),
    ActivistDrift(),
    EightKBurst(),
]

__all__ = [
    "Strategy",
    "StrategyContext",
    "StrategySignal",
    "Side",
    "ENABLED_STRATEGIES",
]
