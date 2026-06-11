"""v_TAIL_HEDGE_REGIME — Spitznagel-style tail insurance.

Buy deep-OTM SPY puts when the market is pricing CHEAP tail risk (low VIX
percentile + complacent skew). Captures the asymmetric payoff in crisis
events — Spitznagel/Universa's documented playbook adapted for daily
resolution + retail.

Trigger thesis (leading):
  - CBOE SKEW > 130 = market priced for tail risk (but spot VIX still low)
  - VIX 1y percentile < 25 = market complacent vs trailing year
  - VIX9D / VIX < 1.05 = no acute stress yet (we want to enter BEFORE)

This is the "buy insurance when it's cheap" trade. Most fires lose 100% of
premium (asymmetric payoff). The handful of fires that hit crises return
500-2000%. Trailing-SL ladder + low NAV per fire makes the bet survivable.

Universe = ["SPY"]. Single-index trade; not for single-stocks.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class TailHedgeRegime:
    id = "tail_hedge_regime"
    enabled = True
    universe = ["SPY"]

    SKEW_THRESHOLD = 130.0       # market paying up for tail risk
    VIX_PCT_THRESHOLD = 0.25     # < 25th percentile = complacent
    MAX_VIX9D_VIX_RATIO = 1.05   # no acute stress yet
    COOLDOWN_DAYS = 21

    def __init__(self):
        self._last_fired: dict[str, object] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        if ctx.cboe_skew is None or ctx.cboe_skew < self.SKEW_THRESHOLD:
            return []
        if ctx.vix_percentile_1y is None or ctx.vix_percentile_1y > self.VIX_PCT_THRESHOLD:
            return []
        if (ctx.vix9d_vix_ratio is not None
                and ctx.vix9d_vix_ratio > self.MAX_VIX9D_VIX_RATIO):
            return []

        last = self._last_fired.get("SPY")
        if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
            return []

        self._last_fired["SPY"] = today
        return [StrategySignal(
            strategy_id=self.id,
            underlying="SPY",
            side="put",
            target_dte=60,
            edge_score=0.85,
            target_otm_pct=0.10,         # 10% OTM puts
            take_profit_pct=None,         # trailing SL ladder captures asymmetry
            stop_loss_pct=-0.60,          # accept higher SL — tail bets need time
            max_hold_days=60,
            max_per_trade_pct_nav=0.03,   # SMALL — most fires lose 100%
            notes=f"TAIL_HEDGE skew={ctx.cboe_skew:.1f} vix_pct={ctx.vix_percentile_1y:.2f}",
        )]
