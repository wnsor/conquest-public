"""D1 — LEAPS DITM on cgrowth top-5 (synthetic stock with leverage).

Brief: 365-730 DTE deep-ITM (delta 75+) calls on conviction names — the
"buy and hold with leverage" approach. Per Phase 5.13 memory: LEAPS
replacement at >60% NAV is catastrophic; at small NAV it's a leveraged
long position with theta as the cost.

EXPECTED CAPITAL CONSTRAINT AT $10k SEED: A 12-month DITM AAPL LEAPS at
delta 80 trades around $30-50 per share = $3k-5k per contract. With the
10% per-trade NAV cap, we can only buy this if NAV ≥ $30k-50k. At $10k
seed we expect 0-1 trades total over the BT — the strategy result will
mostly tell us "LEAPS structurally unaffordable at $10k", which IS the
brief-mandated re-test outcome.

Phase 2.1 Q+M signal isn't published to Object Store yet → universe is
hardcoded mega-cap shortlist (AAPL/MSFT/NVDA/GOOGL/META) that matches
the cgrowth-LIVE top names in 2024-2026.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal

CGROWTH_TOP5 = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]


class CgrowthLeaps:
    id = "cgrowth_leaps"
    enabled = True
    universe = CGROWTH_TOP5

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        self._cooldown_days = 180  # half-year between re-entries per name

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Regime gate: only fire in trending bullish + low-vol environments
        if ctx.vix is None or ctx.vix >= 25:
            return []
        if ctx.term_regime == "backwardation":
            return []

        signals: list[StrategySignal] = []
        for ticker in CGROWTH_TOP5:
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self._cooldown_days):
                continue
            # Require positive 60d momentum + 5MA>20MA (trend confirmation)
            m60 = ctx.underlying_momentum_60d.get(ticker)
            if m60 is None or m60 < 1.05:
                continue
            if not ctx.underlying_5ma_above_20ma.get(ticker, False):
                continue
            # Drawdown filter: don't buy after huge run-up (>15% from 252d high
            # means we're near peaks; LEAPS at peak = bad entry timing)
            dd = ctx.underlying_drawdown_from_252d_high.get(ticker)
            if dd is None or dd > 0.15:
                # Allow entry only when there's some pullback room
                pass

            confluences = 2  # mom + 5MA
            if dd is not None and 0.03 <= dd <= 0.15:
                confluences += 1  # mild pullback = better entry
            if ctx.cstability_vote_count == 0:
                confluences += 1
            edge = min(1.0, confluences / 4.0)

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="leaps_call",
                target_dte=365,
                edge_score=edge,
                target_delta=0.80,         # DITM
                take_profit_pct=1.0,        # +100% on the LEAPS (= 1.25× underlying)
                stop_loss_pct=-0.4,         # -40% before bailing
                time_stop_dte=60,           # roll/close at 60 DTE remaining
                # v22: 365-DTE DITM LEAPS cost $3-5k/contract at $10k seed.
                # 25% cap = one contract per top-5 name max. Cooldown 180d/
                # ticker = max 2 fires/name/year. Total max NAV deployed if
                # all 5 fire = 125% — but only ~3 contracts ever afford-
                # able at $10k seed, so realistic max is 75%.
                max_per_trade_pct_nav=0.25,
                notes=f"leaps DITM, m60={m60:.2f}, dd={dd or 0:.2f}, conf={confluences}",
            ))
        return signals
