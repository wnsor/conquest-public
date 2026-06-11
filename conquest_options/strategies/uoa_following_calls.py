"""A8 — Unusual Options Activity following calls.

Brief: 14-30 DTE calls on a ticker when its option chain shows UOA today
(per Vasquez/Xiao 2024: today's vol > 5× 20d avg AND > 3× 5d OI).

Phase 2 implementation: main.py's UOATracker per-contract baselines feed
ctx.uoa_active (set of tickers with at least one UOA-flagged call contract
today). Strategy fires when the ticker enters the active set.

Per-ticker cooldown: 10 days (UOA flow bursts are short; multiple entries
on the same name same week are usually the same flow).

Universe: union of S&P 100 mega-caps + WSB list. Conservative starting
point; Phase 3+ expansion expected.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal

UOA_UNIVERSE = [
    # Mega-cap (UOA flow often hits these)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "WMT", "COST", "UNH", "JNJ", "PG",
    "DIS", "BA", "AMD", "INTC", "QCOM", "ORCL",
    # WSB-archetype small / mid caps
    "CRDO", "MU", "NOK", "NBIS", "PL", "MX", "DRAM",
]


class UoaFollowingCalls:
    id = "uoa_following_calls"
    enabled = True
    universe = UOA_UNIVERSE

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        self._cooldown_days = 10

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts
        signals: list[StrategySignal] = []
        for ticker in UOA_UNIVERSE:
            if ticker not in ctx.uoa_active:
                continue
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self._cooldown_days):
                continue
            # Edge confluences — UOA alone is 1 signal; supplement with
            # VIX, term-structure, IV-rank, skew-z, GEX regime.
            # 2026-05-25: dropped mom30 confluence — lagging signal misaligns
            # with UOA's leading-indicator thesis (UOA = early positioning;
            # requiring already-positive 30d momentum delays entries).
            confluences = 1
            if ctx.vix is not None and ctx.vix < 25:
                confluences += 1
            if ctx.term_regime in ("contango", "flat"):
                confluences += 1
            iv = ctx.iv_rank.get(ticker)
            if iv is not None and iv < 50:
                confluences += 1
            sz = ctx.skew_z.get(ticker)
            if sz is not None and sz >= 1.0:
                confluences += 1   # crowded hedging → contrarian
            if ctx.gex_regime == "long_gamma":
                confluences += 1
            edge = min(1.0, confluences / 6.0)

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=21,
                edge_score=edge,
                target_otm_pct=0.05,
                take_profit_pct=1.0,
                stop_loss_pct=-0.5,
                time_stop_dte=5,
                notes=f"uoa+ vix={ctx.vix}, iv={iv}, conf={confluences}",
            ))
        return signals
