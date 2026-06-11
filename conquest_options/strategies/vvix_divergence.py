"""v_VVIX_DIVERGENCE — vol-of-vol leading vol-regime shifts.

VVIX measures the vol of VIX itself. When VVIX is elevated while VIX is
still calm, vol traders are positioning for a regime change BEFORE it
shows up in spot vol. Classic leading indicator for vol-spike risk.

Trigger thesis (leading):
  - VVIX > 130 (1-yr lookback typical: 80-120; 130+ is elevated)
  - VIX < 25 (still in normal/calm regime)
  - VIX 1y percentile < 50 (spot vol below median)

Position: VIX calls (if available) — but VIX options have settlement
quirks. Cleaner: SPY 30 DTE 5% OTM puts that benefit if vol-spike causes
a downside move. Smaller position than tail_hedge_regime because the
mechanism is shorter-duration.

Universe = ["SPY"]. Index-level vol-regime trade.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class VvixDivergence:
    id = "vvix_divergence"
    enabled = True
    universe = ["SPY"]

    VVIX_THRESHOLD = 130.0
    MAX_VIX = 25.0
    MAX_VIX_PCT = 0.50           # spot vol below median
    COOLDOWN_DAYS = 14

    def __init__(self):
        self._last_fired: dict[str, object] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        if ctx.vvix is None or ctx.vvix < self.VVIX_THRESHOLD:
            return []
        if ctx.vix is None or ctx.vix >= self.MAX_VIX:
            return []
        if ctx.vix_percentile_1y is None or ctx.vix_percentile_1y > self.MAX_VIX_PCT:
            return []

        last = self._last_fired.get("SPY")
        if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
            return []

        self._last_fired["SPY"] = today
        return [StrategySignal(
            strategy_id=self.id,
            underlying="SPY",
            side="put",
            target_dte=30,
            edge_score=0.75,
            target_otm_pct=0.05,
            take_profit_pct=None,
            stop_loss_pct=-0.50,
            max_hold_days=30,
            max_per_trade_pct_nav=0.04,
            notes=f"VVIX_DIV vvix={ctx.vvix:.1f} vix={ctx.vix:.1f}",
        )]
