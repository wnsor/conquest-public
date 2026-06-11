"""v_IMPLIED_MOVE_DIVERGENCE — buy when options price a move that hasn't shown up.

Trigger thesis (leading-only)
=============================
Options market makers continuously update IV based on order flow.
When IM_30d (forward-implied 30-day move) is meaningfully larger than
HV_30d (realized 30-day move), the options market sees information the
spot market hasn't priced yet. This is informed positioning ahead of:
  - a known catalyst (earnings, FDA approval, product launch)
  - cross-market arbitrage (sector ETF flows incoming)
  - insider-level info leaking through dealer positioning

The signal is LEADING: it fires BEFORE the move shows up in realized
volatility, by definition.

Entry condition:
  1. LEADING: implied_move_vs_realized > 2.5
     IM/HV ratio threshold. 2.5x means options pricing in 2.5x the
     realized volatility — significant divergence.
  2. CONFIRM: either earnings_within_5d OR uoa_active
     We want a REASON the options market is repricing. Pure IM>HV
     without a catalyst could be a temp arb that mean-reverts.
  3. REGIME: vix < 30, term_regime not backwardation

Position structure:
  - 30 DTE 5% OTM CALL (slight upside bias since calls cheaper than
    straddle; if move is symmetric, ATM straddle would be better but
    we're avoiding spending 2× premium)
  - Time-stop at earnings date OR 14 days (whichever first)
  - 6% NAV per trade (smaller — less directional conviction)

Why we don't use straddle:
  Straddles are 2× premium for 1 win. At $10k seed × 6% = $600,
  splitting into both call+put cuts each leg too small to be liquid.
  Single-leg call captures most of the asymmetric tail when the move
  IS to the upside (which earnings beats are 60%+ of the time at $10
  beat-rates).

Expected fire rate:
  At 2.5× threshold + earnings/UOA confirmation: ~5-15 per ticker per
  year (earnings hit ~4×/yr, UOA hit ~2-5×/yr). Across WSB universe:
  ~80-240 fires/year.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class ImpliedMoveDivergence:
    id = "implied_move_divergence"
    enabled = True
    universe = WSB_UNIVERSE

    # 2026-05-27 RECALIBRATION: original 2.5x threshold produced 0 fires across
    # 6 windows. IV/HV ratio rarely exceeds 2.0 even pre-earnings (typical
    # earnings vol premium is 1.3-1.7x). Lowered to 1.4x — captures real
    # repricing events without requiring a unicorn ratio. Will validate
    # via diagnostic counters that the IM-HV gate is now passable.
    IM_HV_RATIO_THRESHOLD = 1.4
    MAX_VIX = 30.0
    COOLDOWN_DAYS = 14   # short — re-trigger if signal repeats post-cooldown

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        self._diag = {
            "vix_block": 0, "term_block": 0, "cooldown": 0,
            "im_hv_missing": 0, "im_hv_below_thr": 0,
            "no_confirmation": 0, "fired": 0, "ticks_total": 0,
        }

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts
        self._diag["ticks_total"] += 1

        if ctx.vix is None or ctx.vix >= self.MAX_VIX:
            self._diag["vix_block"] += 1
            return []
        if ctx.term_regime == "backwardation":
            self._diag["term_block"] += 1
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                self._diag["cooldown"] += 1
                continue

            im_r = ctx.implied_move_vs_realized.get(ticker)
            if im_r is None:
                self._diag["im_hv_missing"] += 1
                continue
            if im_r < self.IM_HV_RATIO_THRESHOLD:
                self._diag["im_hv_below_thr"] += 1
                continue

            # Confirmation: a REASON for the IM repricing
            earnings_soon = (
                ticker in ctx.earnings_within_5d
                or ctx.days_until_next_earnings.get(ticker, 999) <= 5
            )
            uoa = ticker in ctx.uoa_active
            if not (earnings_soon or uoa):
                self._diag["no_confirmation"] += 1
                continue

            # Tighter time-stop if earnings is the trigger
            d_earn = ctx.days_until_next_earnings.get(ticker, 999)
            max_hold = min(14, max(5, d_earn + 2)) if earnings_soon else 14

            self._last_fired[ticker] = today
            self._diag["fired"] += 1
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=30,
                edge_score=0.75,
                target_otm_pct=0.05,   # near-ATM for the implied move capture
                take_profit_pct=None,
                stop_loss_pct=-0.50,
                max_hold_days=max_hold,
                max_per_trade_pct_nav=0.06,
                notes=f"IM_DIV im_hv={im_r:.2f} earn_soon={earnings_soon} uoa={uoa}",
            ))
        return signals
