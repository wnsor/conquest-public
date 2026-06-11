"""A6 — Insider Form 4 opportunistic-buy calls.

Brief: 30-60 DTE calls within 5 trading days of an Officer/Director
opportunistic buy.

Empirical basis: Cohen, Malloy, Pomorski (2012) "Decoding Inside
Information" identifies opportunistic insider buys as predictive of
abnormal future returns (~5-15% alpha). arXiv 2025 (Insider Purchase
Signals in Microcap Equities) achieves 0.70 AUC out-of-sample on 2024
data using a gradient-boosted classifier on filing features — confirming
the signal still works post-Reg-FD.

Universe: tickers that BOTH (a) appear in the Phase 0 Form 4 dataset AND
(b) have option-chain liquidity. We declare a fixed candidate universe at
init (Tier 1 from chain-coverage map + WSB list) and filter dynamically
to those with insider buys per OnData.

Signal: ticker has ≥1 opportunistic buy with dollar_value ≥ $25k by an
Officer/Director in the last 5 trading days. Edge_score scales with
log(dollar_value) — larger buys = stronger conviction.

Exit: 45 DTE entry → time_stop at 10 DTE. TP=+150%, SL=-50%. Per-ticker
cooldown 30d.
"""
from __future__ import annotations

import math
from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal

INSIDER_UNIVERSE = [
    # Megacaps with chain liquidity
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "WMT", "COST", "UNH", "JNJ", "PG",
    "BAC", "WFC", "GS", "MS",
    # Midcaps
    "RH", "WSM", "DKS", "CMG", "FTNT", "ZS", "DDOG",
    "REGN", "VRTX", "DVN", "OXY",
    # WSB / volatile names where insider buys are signal-rich
    "CRDO", "MU", "PLTR", "SOFI", "AMD", "INTC", "RIVN",
]


class InsiderBuyCalls:
    id = "insider_buy_calls"
    enabled = True
    universe = INSIDER_UNIVERSE

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        self._cooldown_days = 30

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts
        signals: list[StrategySignal] = []
        for ticker in INSIDER_UNIVERSE:
            dollar = ctx.insider_recent_buys.get(ticker)
            if dollar is None or dollar <= 0:
                continue
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self._cooldown_days):
                continue

            # Confluences (over the base insider signal): VIX < 25, term contango,
            # IV rank < 60 (don't chase blow-off names).
            # 2026-05-25: dropped mom30 confluence — lagging signal misaligns
            # with the leading-indicator thesis (insider buys are the leading
            # gate; requiring already-positive 30d momentum just delays entries).
            confluences = 1  # insider buy itself
            if ctx.vix is not None and ctx.vix < 25:
                confluences += 1
            if ctx.term_regime in ("contango", "flat"):
                confluences += 1
            iv = ctx.iv_rank.get(ticker)
            if iv is not None and iv < 60:
                confluences += 1

            # Dollar-size edge component: log scale, $25k → 0, $1M → 1.0
            size_edge = min(1.0, max(0.0, (math.log10(max(dollar, 1)) - math.log10(25_000)) /
                                          (math.log10(1_000_000) - math.log10(25_000))))
            edge = 0.5 * (confluences / 4.0) + 0.5 * size_edge

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=45,
                edge_score=edge,
                target_otm_pct=0.07,
                take_profit_pct=1.5,
                stop_loss_pct=-0.5,
                time_stop_dte=10,
                notes=f"insider buy ${dollar/1000:.0f}k, conf={confluences}, size_edge={size_edge:.2f}",
            ))
        return signals
