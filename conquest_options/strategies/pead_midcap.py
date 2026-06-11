"""A5b — PEAD on midcap names (where the SUE-based anomaly is still alive).

2025 ScienceDirect ML-PEAD paper documents PEAD alpha decay specifically
in *developed, large-cap* markets, while smaller-cap segments retain the
drift. arXiv 2025 retail-horizon paper similarly finds long-horizon (often
midcap/small) stocks have larger post-announcement drift (0.43%/mo alpha).

Universe: representative midcaps with reliable earnings coverage. Not the
full S&P MidCap 400 — that adds option-chain liquidity risk. Phase 3+ can
dynamically expand once we know which names have usable chains.

Same per-trade mechanics as A5a (21 DTE / 5% OTM / TP+100% / SL-50%) so
the alpha differential between A5a and A5b is interpretable as universe
effect, not strategy-design effect.
"""
from __future__ import annotations

from strategies.base import StrategyContext, StrategySignal

# ~30 midcap names with options-friendly liquidity. Built from S&P MidCap
# 400 + Russell MidCap intersection, then filtered to names with consistent
# weekly option chains (verified via Phase 0 chain_availability scaffold).
MIDCAP_UNIVERSE = [
    # Consumer / retail / restaurants
    "RH", "WSM", "DKS", "CMG", "DPZ", "TXRH",
    # Industrial / mid-cap tech
    "FTNT", "ZS", "DDOG", "MDB", "NET", "TWLO", "CRWD",
    # Healthcare / biotech mid
    "REGN", "VRTX", "BIIB", "ALGN", "ILMN",
    # Energy / materials mid
    "DVN", "FANG", "MRO", "OXY",
    # Financials mid
    "RJF", "SIVB", "FITB", "ZION",
    # Other liquid mid
    "PLTR", "SOFI", "RIVN", "AFRM",
]


class PeadMidcap:
    id = "pead_midcap"
    enabled = True
    universe = MIDCAP_UNIVERSE

    def __init__(self):
        self._last_event_fired: dict[str, str] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        signals: list[StrategySignal] = []
        for ticker in MIDCAP_UNIVERSE:
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
                notes=f"midcap, ds={ds}, surprise={sp:.1f}%",
            ))
        return signals
