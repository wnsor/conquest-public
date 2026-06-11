"""Smoke-test strategy — proves the Phase 1 framework wiring end-to-end.

Logic: on the first trading day of each month, emit a buy signal for a
30-DTE ATM SPY call with TP=+100%, SL=-50%, time_stop=5 DTE. No real edge
signal — this exists purely to exercise the pipeline (signal → contract
pick → sizing → entry → exit → trade log).

Enabled only when config parameter ENABLE_SMOKE=true. Phase 2 deletes
this file in favor of real A-category strategies.
"""
from __future__ import annotations

from strategies.base import StrategyContext, StrategySignal


class SmokeA2:
    id = "smoke_a2_spy_monthly_call"
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
        self._last_fired_month = ym
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="call",
                target_dte=30,
                edge_score=0.5,
                target_delta=0.50,
                take_profit_pct=1.0,
                stop_loss_pct=-0.5,
                time_stop_dte=5,
                notes="Phase 1 framework smoke test",
            )
        ]
