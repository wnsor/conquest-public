"""C1 — Pre-earnings ATM straddle (long).

Goyal & Saretto (2009) "Cross-section of option returns and volatility":
implied vol expands 5-15% in the 2-5 days BEFORE earnings as the market
anticipates the event-day vol jump. A long ATM straddle entered ~2 days
pre-earnings and exited ~1 day pre-earnings captures this IV expansion
without taking the post-announce IV crush. Documented 10-25% alpha.

This strategy emits TWO independent signals per setup — one ATM call and
one ATM put with same DTE — that the framework treats as separate trades
sharing strategy_id. Per-trade metrics aggregate both legs.

Mechanics:
  Trigger: days_until_next_earnings ∈ [2, 3]
  Entry: ATM (delta ±0.50) call + put, 14-30 DTE
  Exit: max_hold_days=1 → exits ~1 day after entry, ~1 day before earnings
  TP: +100%, SL: -50%
"""
from __future__ import annotations

from strategies.base import StrategyContext, StrategySignal

# Universe selected 2026-05-23 per user directive (no leveraged ETFs, prefer
# stocks and SPY). Selection criteria:
#   - Liquid options (tight spreads, OI > 1k at front-month ATM)
#   - Documented high pre-earnings IV cycle (Goyal-Saretto edge biggest here)
#   - $20+ stock price (avoid the NOK-style $4 disaster from Phase 2)
#   - Diverse sectors (no single-sector concentration risk)
#   - Reliable earnings cadence + Yahoo coverage back to 2018
#   - Pre-2020 listing (so 2022-2026 BT window has full history)
STRADDLE_UNIVERSE = [
    # Index (per user directive)
    "SPY",
    # Mega-cap tech (highest IV cycle reliability)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    # Semiconductors (earnings-vol heavy)
    "AMD", "MU", "QCOM", "TSM",
    # Streaming/Media (NFLX in particular has historically massive post-earnings gaps)
    "NFLX", "DIS",
    # Big banks (quarterly earnings cycle is the canonical PEAD setup)
    "JPM", "GS",
    # Consumer (COST/NKE consistently have earnings-day surprises)
    "COST", "WMT", "NKE", "LULU",
    # Healthcare (LLY recent vol especially; UNH/JNJ are PEAD classics)
    "UNH", "LLY",
    # Cloud/SaaS (high IV pumps pre-earnings)
    "PLTR", "CRWD", "SNOW",
    # Fintech / payments
    "V", "MA", "PYPL",
]


class EarningsStraddle:
    id = "earnings_straddle"
    enabled = True
    universe = STRADDLE_UNIVERSE

    def __init__(self):
        # Track which earnings event we've already entered (avoid double-fire)
        self._fired_event: dict[str, int] = {}  # ticker → days_until_earnings at entry

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        signals: list[StrategySignal] = []
        for ticker in STRADDLE_UNIVERSE:
            d = ctx.days_until_next_earnings.get(ticker)
            if d is None:
                continue
            # Enter window: 2-3 days before earnings (calendar days, not trading)
            if d not in (2, 3):
                # Clear stale fire-marker when we're well past the prior event
                if d > 7 and ticker in self._fired_event:
                    del self._fired_event[ticker]
                continue
            # De-dup: don't re-enter the same earnings event
            if ticker in self._fired_event:
                continue
            self._fired_event[ticker] = d

            # Edge confluences (above the base earnings-cycle setup)
            confluences = 1   # base: pre-earnings IV expansion edge
            iv = ctx.iv_rank.get(ticker)
            if iv is not None and iv < 60:
                confluences += 1   # IV not already elevated → more room to expand
            if ctx.vix is not None and ctx.vix < 25:
                confluences += 1   # market not in stress
            if ctx.term_regime in ("contango", "flat"):
                confluences += 1
            edge = min(1.0, confluences / 4.0)

            common = dict(
                strategy_id=self.id,
                underlying=ticker,
                target_dte=21,
                edge_score=edge,
                take_profit_pct=1.0,
                stop_loss_pct=-0.5,
                max_hold_days=1,
                notes=f"earnings T-{d}, iv_rank={iv}, conf={confluences}",
            )
            # Two-leg straddle: ATM call + ATM put
            signals.append(StrategySignal(side="call", target_delta=0.50, **common))
            signals.append(StrategySignal(side="put", target_delta=-0.50, **common))
        return signals
