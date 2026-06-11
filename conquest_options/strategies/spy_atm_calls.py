"""A2 — SPY ATM calls.

Brief: 30-DTE ATM SPY call, monthly, triggered by cgrowth signal + IV rank<30.

Phase 2 implementation: cgrowth's full Q+M signal isn't published to Object
Store yet. We use a proxy that captures the same intent:
    SPY 60-day momentum > 1.0  (cgrowth-positive market regime)
    AND VIX < 25               (no acute stress)
    AND IV rank < 30           (options reasonably priced)

edge_score scales with the number of additional confluences (cstability
vote_count == 0 = +0.2; SPY 30d momentum > 30d momentum 30 days ago = +0.1).

Cadence: at most once per calendar month. Sized at base 1.5% NAV @ edge=1.
"""
from __future__ import annotations

from strategies.base import StrategyContext, StrategySignal


class SpyAtmCalls:
    id = "spy_atm_calls"
    enabled = True
    universe = ["SPY"]

    def __init__(self):
        self._last_fired_month: tuple[int, int] | None = None

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        ym = (ts.year, ts.month)
        if self._last_fired_month == ym:
            return []

        vix = ctx.vix
        mom60 = ctx.underlying_momentum_60d.get("SPY")
        iv_rank = ctx.iv_rank.get("SPY")
        if vix is None or mom60 is None or iv_rank is None:
            return []

        # Hard gates
        if vix >= 25:
            return []
        if mom60 <= 1.0:
            return []
        if iv_rank >= 30:
            return []
        if ctx.term_regime == "backwardation":
            return []

        # Confluence-based edge score
        confluences = 3   # all three hard gates above
        if ctx.cstability_vote_count is not None and ctx.cstability_vote_count == 0:
            confluences += 1
        if (m30 := ctx.underlying_momentum_30d.get("SPY")) is not None and m30 > 1.0:
            confluences += 1
        # GEX regime: long_gamma supportive of directional drift; short_gamma
        # means trends accelerate (which we want) but realized vol is
        # elevated (which we don't, since we're buying premium).
        # Net neutral — log it but don't shift edge_score either direction.
        if ctx.gex_regime == "long_gamma":
            confluences += 1
        edge = min(1.0, confluences / 6.0)

        self._last_fired_month = ym
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="call",
                target_dte=30,
                edge_score=edge,
                target_delta=0.50,
                take_profit_pct=1.0,
                stop_loss_pct=-0.5,
                time_stop_dte=5,
                notes=f"vix={vix:.1f}, mom60={mom60:.2f}, iv_rank={iv_rank:.1f}, conf={confluences}",
            )
        ]
