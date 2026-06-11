"""B3 — Single-stock momentum-failure long put.

Thesis: High-momentum names that BREAK trend (5MA crosses below 20MA) while
options are still cheap (IV-rank < 50, i.e. market hasn't repriced vol yet)
offer asymmetric short setups. Catch the early breakdown before the panic.

Why long puts on cgrowth-style names: cgrowth's Q+M momentum signal RIDES
trend; this strategy fades names that EXIT trend before consensus catches
on. Hedged exposure: cgrowth is long the names while in trend; we're short
the names once trend breaks. Don't double-up by also being long via D1
LEAPS on the same name (D1's `_last_fired` cooldown is 180d; B3's is 30d,
so timing usually deconflicts).

Universe: boom-bust momentum stocks — NVDA/TSLA/AMD/META/GOOGL/MSFT/AAPL
          /NFLX/AMZN/COIN. Skewed toward "growth at any price" names that
          have severe drawdown patterns when trend breaks.

Trigger:
  - 5MA < 20MA (trend break confirmation)
  - IV-rank < 50 (options cheap — vol not yet repriced)
  - Drawdown from 252d high < 8% (still near peaks — early in breakdown)
  - No recent insider buy in last 5 days (those are bullish; don't fade)

Position: 45-DTE 5% OTM put
Exit: TP+150%, SL-60%, max_hold 30d
Cooldown: 30 days per stock
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


# Boom-bust momentum names — high beta, severe drawdowns when trend breaks
B3_UNIVERSE = [
    "NVDA", "TSLA", "AMD", "META", "GOOGL", "MSFT", "AAPL",
    "NFLX", "AMZN", "COIN",
]


class MomentumFailurePut:
    id = "momentum_failure_put"
    enabled = True
    universe = B3_UNIVERSE

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        self._cooldown_days = 30

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        signals: list[StrategySignal] = []
        for ticker in B3_UNIVERSE:
            last = self._last_fired.get(ticker)
            if (last is not None and
                    (today - last) < timedelta(days=self._cooldown_days)):
                continue

            # Trend-break confirmation: 5MA must be BELOW 20MA
            if ctx.underlying_5ma_above_20ma.get(ticker, True):
                continue  # still in uptrend, no setup

            # Options must still be CHEAP — IV not yet repriced for breakdown
            iv_rank = ctx.iv_rank.get(ticker)
            if iv_rank is None or iv_rank >= 50:
                continue  # too late, vol already priced in

            # Early in breakdown: drawdown from 252d high still small
            dd = ctx.underlying_drawdown_from_252d_high.get(ticker)
            if dd is None or dd > 0.08:
                continue  # already too deep into the breakdown

            # Don't fade insider buying (they're calling the bottom — fade them
            # only if we have stronger conviction, not for typical breakdowns)
            if ticker in ctx.insider_recent_buys:
                continue

            confluences = 3  # cross + cheap_iv + early_dd
            if (m60 := ctx.underlying_momentum_60d.get(ticker)) is not None and m60 < 1.0:
                confluences += 1  # 60d momentum also negative — stronger setup
            if ctx.vix is not None and ctx.vix > 20:
                confluences += 1  # market also stressed
            edge = min(1.0, confluences / 5.0)

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="put",
                target_dte=45,
                edge_score=edge,
                target_otm_pct=0.05,        # 5% OTM
                take_profit_pct=1.5,         # +150%
                stop_loss_pct=-0.6,          # -60%
                max_hold_days=30,
                notes=(f"5MA<20MA iv_rank={iv_rank:.0f} dd={dd:.2%} "
                       f"conf={confluences}/5"),
            ))
        return signals
