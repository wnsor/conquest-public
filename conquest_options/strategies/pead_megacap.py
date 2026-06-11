"""A5a — PEAD on megacap names (baseline; alpha decay documented in 2025
literature; expected to fail per-trade gate).

Identical mechanics to A5b but on a megacap universe where the SUE-based
PEAD anomaly has been arbitraged away in modern markets (Beyond the last
surprise: Reviving PEAD with ML, ScienceDirect 2025). We keep A5a as the
**baseline**: if it passes the per-trade gate at retail scale we'll be
genuinely surprised; if it fails we have a clean reference point against
which A5b (midcap) gains can be measured.

Logic: 14-30 DTE 5% OTM call when days_since_last_earnings ∈ [1, 3] AND
last_surprise_pct > 0. De-dup'd per (ticker, ds, surprise) tuple.
"""
from __future__ import annotations

from strategies.base import StrategyContext, StrategySignal

MEGACAP_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "WMT", "COST", "UNH", "JNJ", "PG",
]


class PeadMegacap:
    id = "pead_megacap"
    enabled = True
    universe = MEGACAP_UNIVERSE

    def __init__(self):
        self._last_event_fired: dict[str, str] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        signals: list[StrategySignal] = []
        for ticker in MEGACAP_UNIVERSE:
            ds = ctx.days_since_last_earnings.get(ticker)
            sp = ctx.last_earnings_surprise_pct.get(ticker)
            if ds is None or sp is None:
                continue
            if not (1 <= ds <= 3):
                continue
            if sp <= 0:
                continue
            event_fp = f"{ticker}:{ds}:{round(sp, 1)}"
            if self._last_event_fired.get(ticker) == event_fp:
                continue
            self._last_event_fired[ticker] = event_fp

            edge = max(0.3, min(1.0, sp / 20.0))
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
                # v22 iter v2_bigcap: PEAD megacap contracts on AVGO/MSFT/NVDA
                # in 2024 cost $500-$3000+ each at 5% OTM. Default 10% cap of
                # $10k NAV = $1000 → rejected most setups (fail_size=46/66).
                # Bump to 20% to allow more contracts AND fit AVGO-style names.
                # Per-trade-gate strategies (frequent fires) typically use 10%
                # but PEAD fires ~14×/year so concentration is bounded.
                max_per_trade_pct_nav=0.20,
                notes=f"megacap, ds={ds}, surprise={sp:.1f}%",
            ))
        return signals
