"""v_NETWORK_PROPAGATION — pure attention-acceleration play.

Trigger thesis (leading-only)
=============================
When attention SPREADS across distinct sources/days for a ticker, the
narrative is consolidating BEFORE the price reacts. This is the
"information cascade" pre-stage: institutions are starting to write
about it, retail is starting to discuss it, and the price hasn't fully
caught up yet.

GDELT count series is a proxy. Better would be unique-source count
(how many distinct outlets cover the ticker), which requires GDELT
ingester enhancement — current implementation tracks aggregate counts
only. With aggregate counts, news_propagation_5d (this 5d / prior 5d)
is our best proxy.

Entry condition:
  1. LEADING: news_propagation_5d ≥ 2.0
     STRICTER than v_REFLEX_v2's 1.5 — we want CLEAR acceleration,
     not just a small bump
  2. CONFIRM: volume_spike ≥ 2.5
     Sanity check — narrative consolidation should show in trading too
  3. REGIME: VIX < 25, term_regime in (contango, flat)

Position structure:
  - 30 DTE 10% OTM call
  - 6% NAV — narrower signal, smaller bet
  - max_hold 30 days — narrative-driven moves resolve within DTE window
  - Trailing-SL ladder

Expected fire rate:
  ~3-8 per ticker per year at 2.0 threshold (rare strong acceleration).
  Across 16 names: ~50-130 fires/year. Higher than v_TRIPLE_CONFLUENCE,
  lower than v_REFLEX_v2.

When this is most useful:
  As a complement to v_REFLEX_v2 — captures the cases where attention
  is FIRST signal (SI hasn't moved yet, insiders haven't filed).
  Acts as an early-warning satellite around the main reflexivity setup.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE, WSB_TICKER_INCEPTION


class NetworkPropagation:
    id = "network_propagation"
    enabled = True
    universe = WSB_UNIVERSE

    NEWS_PROPAGATION_THRESHOLD = 2.0
    VOLUME_SPIKE_THRESHOLD = 2.5
    # v2 (2026-05-27): added directional momentum confirm. Prior version fired
    # often (4-of-5 windows had trades) but lost most (0% WR in 4 of those 5).
    # Root cause: "narrative + volume" alone doesn't guarantee directional
    # follow-through. ~10% of those narratives go DOWN despite buzz.
    # Adding mom5d > 0 ensures price IS moving in the call-favorable direction
    # at entry. Also tightened OTM from 10% → 5% (easier strike to reach).
    MIN_MOM_30D = 1.00          # ticker non-negative 30d return
    MAX_VIX = 25.0
    COOLDOWN_DAYS = 21
    TARGET_OTM_PCT = 0.05       # tighter — 5% reach more achievable in 30d

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        self._diag = {
            "vix_block": 0, "term_block": 0, "cooldown": 0,
            "np5_missing": 0, "np5_below_thr": 0,
            "vol_below_thr": 0, "mom_neg": 0, "fired": 0, "ticks_total": 0,
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
            # v17 IPO-date guard (see momentum_otm_calls.py docstring)
            min_date = WSB_TICKER_INCEPTION.get(ticker)
            if min_date is not None and today < min_date:
                continue
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                self._diag["cooldown"] += 1
                continue

            np5 = ctx.news_propagation_5d.get(ticker)
            if np5 is None:
                self._diag["np5_missing"] += 1
                continue
            if np5 < self.NEWS_PROPAGATION_THRESHOLD:
                self._diag["np5_below_thr"] += 1
                continue

            vs = ctx.volume_spike.get(ticker, 1.0)
            if vs < self.VOLUME_SPIKE_THRESHOLD:
                self._diag["vol_below_thr"] += 1
                continue

            # v2: directional momentum confirm — don't buy calls on declining tickers
            mom30 = ctx.underlying_momentum_30d.get(ticker, 1.0)
            if mom30 < self.MIN_MOM_30D:
                self._diag["mom_neg"] += 1
                continue

            self._last_fired[ticker] = today
            self._diag["fired"] += 1
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=30,
                edge_score=0.70,
                target_otm_pct=self.TARGET_OTM_PCT,
                take_profit_pct=None,
                stop_loss_pct=-0.50,
                max_hold_days=30,
                max_per_trade_pct_nav=0.06,
                notes=f"NETPROP_v2 np5d={np5:.2f} vol={vs:.1f} mom30={mom30:.2f}",
            ))
        return signals
