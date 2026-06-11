"""A_GEX_SPY — GEX-driven SPY directional call.

v12 SELECTIVE (2026-05-23): tightened gates per user directive
"extremely selective + high ROI per trade". Trade count drops ~30-50%
in exchange for higher win rate + bigger winners (further OTM = more
leverage when GEX-thesis pays off). Goal: target fish_-style 200-400%
per-trade winners.

Edge thesis (unchanged): dealer short-gamma + technical up-cross + cheap
vol environment → SPY rally accelerates → OTM calls capture leverage.

v12 changes vs v1:
  - require edge_score >= 0.6 (5+ of 8 confluences) — only highest-conviction setups
  - 3 NEW confluences: IV rank<50, IV/HV<1.2, SPY drawdown<5% from 252d high
  - 7% OTM (was 5%) — cheaper premium, more leverage if SPY moves
  - TP +300% (was +200%) — let winners run further
  - SL -60% (was -50%) — accept deeper drawdown on high-conviction
  - max_hold 20d (was 15d) — give the move time
  - cooldown 20d (was 15d) — be patient between entries
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class GexSpySelective:
    id = "gex_spy_selective"
    enabled = True
    universe = ["SPY"]

    def __init__(self):
        self._last_fired = None
        self._cooldown_days = 20    # v12: was 15

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts
        if self._last_fired is not None and (today - self._last_fired) < timedelta(days=self._cooldown_days):
            return []

        # Hard gates (same as v1)
        if ctx.gex_regime != "short_gamma":
            return []
        if not ctx.underlying_5ma_above_20ma.get("SPY", False):
            return []
        if ctx.vix is None or ctx.vix >= 25:
            return []
        if ctx.term_regime == "backwardation":
            return []

        # v12: count expanded confluences (8 possible vs 5 in v1)
        confluences = 3  # base: GEX + 5MA + VIX gates
        # Existing v1 confluences
        if (m30 := ctx.underlying_momentum_30d.get("SPY")) is not None and m30 > 1.0:
            confluences += 1
        if ctx.cstability_vote_count is not None and ctx.cstability_vote_count == 0:
            confluences += 1
        # v12 NEW confluences
        iv = ctx.iv_rank.get("SPY")
        if iv is not None and iv < 50:
            confluences += 1    # low IV rank = cheap options
        iv_hv = ctx.iv_hv_ratio.get("SPY")
        if iv_hv is not None and iv_hv < 1.2:
            confluences += 1    # options not overpriced vs realized vol
        dd = ctx.underlying_drawdown_from_252d_high.get("SPY")
        if dd is not None and dd < 0.05:
            confluences += 1    # not buying near the top
        edge = min(1.0, confluences / 8.0)

        # v12: REQUIRE edge_score >= 0.6 (≥5 of 8 confluences) — selective
        if edge < 0.6:
            return []

        self._last_fired = today
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="call",
                target_dte=35,
                edge_score=edge,
                target_otm_pct=0.07,        # v12: 7% OTM (was 5%) — cheaper premium, more leverage
                take_profit_pct=3.0,         # v12: +300% (was +200%) — let winners run
                stop_loss_pct=-0.6,          # v12: -60% (was -50%)
                max_hold_days=20,            # v12: 20d (was 15)
                notes=f"v12 gex_short 5MA>20 vix={ctx.vix:.1f} iv_rank={iv} iv_hv={iv_hv} conf={confluences}/8",
            )
        ]
