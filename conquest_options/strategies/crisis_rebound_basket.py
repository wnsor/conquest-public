"""Crisis Rebound Basket — fires AGGRESSIVELY once per crisis when the
V-bottom recovery is confirming. Tepper-style playbook expanded across
multiple vehicles to maximize asymmetric capture.

Trigger: ctx.crisis_state == 'rebound' (CrisisDetector signal)
         AND haven't fired this crisis yet

Deploys simultaneously:
  - SPY 180-DTE delta-0.60 call (broad recovery beta)
  - QQQ 180-DTE delta-0.60 call (tech leverage)
  - 5 individual high-momentum names (90-DTE 5% OTM calls each)

Sizing: each leg at 5-8% NAV — total basket = 30-50% NAV deployed
Exit: TP+200% per leg, SL-50%, max_hold 120 days
Lock: won't re-fire for 180 days after entry

Historical fires expected:
  - 2009-03 GFC bottom (SPY +65% over 9 months)
  - 2020-04 COVID bottom (SPY +50% in 11 weeks)
  - 2022-10 mid-cycle bear bottom (SPY +17% in Q4)
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal

REBOUND_BASKET_TICKERS = [
    # Broad indexes
    "SPY", "QQQ",
    # High-beta tech recovery names (AI/semis leaders)
    "NVDA", "AMD", "META", "GOOGL", "MSFT",
]


class CrisisReboundBasket:
    id = "crisis_rebound_basket"
    enabled = True
    universe = REBOUND_BASKET_TICKERS

    def __init__(self):
        self._fired_dates: list = []   # log all fires for debugging
        self._last_fire_date = None
        self._fire_cooldown_days = 180

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Only fire when crisis detector signals rebound
        if ctx.crisis_state != "rebound":
            return []

        # Once-per-crisis lock
        if self._last_fire_date is not None and \
           (today - self._last_fire_date).days < self._fire_cooldown_days:
            return []

        signals = []
        # v22: 7-leg basket = 50% NAV total during a crisis fire.
        # Split: SPY/QQQ index LEAPS get larger per-leg caps (10% each =
        # 20% total) since they're 180-DTE delta-0.60 = $1.5-3k contracts;
        # individual name calls (90-DTE 5% OTM) get 6% each (= 30% total
        # for 5 names). Total: ~50% NAV at fire (CrisisDetector triggers
        # at most once per 180d so concentration is bounded).
        # SPY/QQQ broad-index calls (longer DTE for compounding)
        for ticker in ("SPY", "QQQ"):
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="leaps_call",
                target_dte=180,
                edge_score=1.0,
                target_delta=0.60,
                take_profit_pct=2.0,
                stop_loss_pct=-0.5,
                max_hold_days=120,
                max_per_trade_pct_nav=0.10,   # 10% per index leg
                notes=f"crisis_rebound basket-index vix_peak={ctx.crisis_vix_peak}",
            ))
        # Individual high-beta tech names (shorter DTE for leverage)
        for ticker in ("NVDA", "AMD", "META", "GOOGL", "MSFT"):
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=90,
                edge_score=1.0,
                target_otm_pct=0.05,
                take_profit_pct=2.5,
                stop_loss_pct=-0.6,
                max_hold_days=60,
                max_per_trade_pct_nav=0.06,   # 6% per name leg (5 names = 30% total)
                notes=f"crisis_rebound basket-name vix_peak={ctx.crisis_vix_peak}",
            ))

        self._last_fire_date = today
        self._fired_dates.append(today)
        return signals
