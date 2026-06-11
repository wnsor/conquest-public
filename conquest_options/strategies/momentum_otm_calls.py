"""A1 — WSB single-stock OTM calls.

Brief: 20-30 DTE 10% OTM call on WSB-archetype stocks (CRDO, MU, NOK, NBIS,
PL, MX, DRAM). Confluence required: positive 30d momentum + IV rank<30 +
sector strength (proxied as VIX<25 for market regime).

Per-ticker cooldown: once a position fires on a ticker, wait 30 days before
firing again on the same ticker (independent of whether the position is
still open — the gate is on the strategy's re-entry, not the position).
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal

# Phase 4 v4 (2026-05-23): universe re-expanded based on:
#   (a) fish_ WSB-trader analysis: NOK/MX/PL/DRAM names were his BIG winners
#       (+200%, +189%, +230%, +200% — exactly what we dropped in v3)
#   (b) Conquest universe screen output: NBIS (+135% 60d), CRDO (+94%),
#       DRAM (+90%), MU (+82%) ALL pass ADV > $25M, price > $20 filter
#   (c) Lesson learned: execution fix (limit orders + min OI/spread filters
#       in option_selector.py) is the right response to illiquidity, not
#       dropping the names that generate the alpha
from datetime import date


# v17 (2026-05-28): user-curated to fish_'s confirmed names + parallel-profile
# AI-narrative + speculative single-name plays. CRITICALLY: each ticker now
# has a min_start_date in WSB_TICKER_INCEPTION below — without that guard,
# the BT would trade post-IPO names back to 2018 (impossible — security
# didn't exist) on QC's data layer which silently substitutes a predecessor
# security thread (e.g., NBIS BT entries in 2018-2024 are actually trading
# the delisted YNDX/Yandex ADR — same security ID, different fundamental
# profile). The IPO-date filter in on_data ensures we only enter trades
# AFTER the ticker existed as the security we calibrated for.
WSB_UNIVERSE = [
    # Core fish_-confirmed names (Dec 2025 → May 2026 book)
    "PLTR", "SMCI", "NVDA", "AMD", "MU", "RKLB", "META", "AMZN",
    "MSTR", "CRDO", "COIN",
    # High-confidence AI-narrative parallels (same momentum + retail profile)
    "ARM",   # AI chip licensing IPO Sept 2023
    "AVGO",  # AI infrastructure pure play
    "ANET",  # AI/data-center networking
    "TSM",   # foundry leader
    "VRT",   # AI cooling/power
    # Speculative single-name plays (retail attention profile)
    "HOOD",  # Robinhood — retail trading proxy
    "MARA",  # Bitcoin miner
    "RIOT",  # Bitcoin miner
    "APP",   # AppLovin — mobile ad-tech 5x in 2024-25
    "TSLA",  # EV/AI/robotics persistent meme
    # 2024-25 AI-cloud spinoffs (most-aligned with current fish_ rotation)
    "NBIS",  # Yandex spinoff Sept 2024 — modern Nebius profile
    "CRWV",  # CoreWeave IPO March 2025 — pure-play AI cloud
]


# Per-ticker IPO / re-listing dates. Strategies that gate on this dict
# refuse to trade a ticker before its inception date — prevents survivorship
# leakage where QC's data layer fills in pre-IPO history from a predecessor
# security thread (e.g., NBIS → YNDX, RKLB → SPAC pre-merger).
#
# Sources: SEC EDGAR S-1 effective dates / first-trade dates on NYSE/NASDAQ.
WSB_TICKER_INCEPTION: dict[str, date] = {
    "PLTR": date(2020, 9, 30),       # direct listing
    "RKLB": date(2021, 8, 25),       # SPAC merger close
    "COIN": date(2021, 4, 14),       # direct listing
    "CRDO": date(2022, 1, 27),       # IPO
    "ARM":  date(2023, 9, 14),       # IPO
    "HOOD": date(2021, 7, 29),       # IPO
    "APP":  date(2021, 4, 14),       # IPO
    "NBIS": date(2024, 9, 1),        # Yandex spinoff (post-suspension)
    "CRWV": date(2025, 3, 28),       # IPO
    # MSTR: pre-existing, but Bitcoin pivot ~Aug 2020 changed the fundamental
    # profile. Earlier years are an enterprise-software MSTR not the crypto
    # proxy fish_ trades. Guard at 2020-08-01.
    "MSTR": date(2020, 8, 1),
    # VRT: post-SPAC. Pre-2020 was Vertiv private; SPAC closed Feb 2020.
    "VRT":  date(2020, 2, 10),
    # AVGO: pre-existing as Broadcom but AI-narrative repricing ~mid-2023.
    # Strategy gates on momentum so this is a soft guard — but pre-2023
    # AVGO had a different signal regime. Conservative: 2023-01-01.
    "AVGO": date(2023, 1, 1),
    # Others (NVDA, AMD, MU, META, AMZN, ANET, TSM, MARA, RIOT, TSLA, SMCI)
    # all have meaningful trading history back to 2018+ → no guard needed.
}


class MomentumOtmCalls:
    id = "momentum_otm_calls"
    enabled = True
    universe = WSB_UNIVERSE

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        # v22 iter v5: 21d cooldown — let real winners re-enter quarterly,
        # but space wider than v3's 20d (which contributed to overtrading).
        self._cooldown_days = 21

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        vix = ctx.vix
        if vix is None or vix >= 25:
            return []
        # Hard-gate: refuse new entries during term-backwardation or acute
        # stress (cstability uses the same logic for risk-off).
        if ctx.term_regime == "backwardation":
            return []
        if ctx.vix9d_vix_ratio is not None and ctx.vix9d_vix_ratio > 1.0:
            return []

        signals: list[StrategySignal] = []
        today = ts.date() if hasattr(ts, "date") else ts
        for ticker in WSB_UNIVERSE:
            # v17 (2026-05-28): IPO-date guard. Skip tickers that didn't
            # exist as the security we calibrated for at this BT date.
            # Prevents trading e.g. NBIS in 2018 (which QC data resolves to
            # the predecessor YNDX/Yandex thread — different fundamentals).
            min_date = WSB_TICKER_INCEPTION.get(ticker)
            if min_date is not None and today < min_date:
                continue
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self._cooldown_days):
                continue
            iv_rank = ctx.iv_rank.get(ticker)
            mom30 = ctx.underlying_momentum_30d.get(ticker)
            # v22 iter v2: treat iv_rank=None as 50 (neutral) so small-caps
            # and recent IPOs (RKLB/IONQ/APP/SMCI) with no warmup history
            # can still trigger. Raise threshold 60 → 80 to admit high-vol
            # momentum names that are common in fish_-style trades.
            # v22 iter v7: PURE LEADING INDICATORS — pre-position BEFORE
            # the move, not in response to volume/news already happening.
            # Removed lagging signals (volume_spike, news_volume_spike,
            # mom30, 5MA>20MA). Kept only what foreshadows price action.
            #
            # v22 iter v8 (2026-05-25): DROPPED iv_rank<40 hard gate. v7
            # produced 0 fires in 3-month Feb-Apr 2024 NVDA supercycle window.
            # Pre-earnings IV typically spikes to 50-70 on momentum names —
            # the <40 cap excluded exactly the setups fish_ would target.
            # fish_'s edge wasn't paying-up-front but the asymmetric outcome:
            # a 200% expected move dominates a 30% IV-overpay.
            #
            # v22 iter v10 (2026-05-25): v9 result IDENTICAL to v8 — iv_rank
            # gate is a no-op in 3-month BTs (IVRankTracker has 252d lookback,
            # tracker returns None until warmup completes, default=50 passes
            # any <=60 gate trivially). DROPPED iv_rank gate as structurally
            # broken in this BT framework.
            #
            # Real lever per v8w (n=8, WR 12.5%): catalyst OR-gate fires too
            # broadly. v10 = DECISIVE CATALYSTS ONLY:
            #   - earnings <=7d (was 14d — half the window)
            #   - skew_z <= -1.5 (was -1.0 — only on STRONG positioning)
            #   - insider_cluster >= 1.5 (unchanged)
            #   - uoa_active (unchanged)
            iv_rank_eff = 50.0 if iv_rank is None else iv_rank
            # iv_rank gate dropped — see v10 note above
            # 2. LEADING catalyst — at least ONE of (TIGHTENED in v10):
            #    - insider cluster (institutional buying)
            #    - earnings within 7 days (NEAR-TERM known catalyst)
            #    - skew_z <= -1.5 (STRONG call-side positioning)
            #    - uoa_active (early options flow)
            i_cluster = ctx.insider_cluster_score.get(ticker, 0.0)
            d_earn = ctx.days_until_next_earnings.get(ticker, 999)
            sz = ctx.skew_z.get(ticker)
            uoa_flagged = ticker in ctx.uoa_active
            d_earn_dummy = d_earn  # keep variable name for later
            catalyst = (
                i_cluster >= 1.5                    # insiders accumulating
                or d_earn <= 7                      # near-term known catalyst (v10)
                or (sz is not None and sz <= -1.5)  # strong call-side (v10)
                or uoa_flagged                       # early options flow
            )
            # Placeholders for variables no longer used (avoid undefined refs):
            v_spike = None
            n_spike = 1.0
            if not catalyst:
                continue
            mom30_eff = 1.0 if mom30 is None else mom30
            # v22 iter v2: DROPPED the iv_hv gate (was filtering out >50% of
            # high-vol momentum setups that are exactly fish_'s territory).
            # Original: if ctx.iv_hv_ratio.get(ticker) > 1.5: continue
            # Confluence score — 3 hard gates passed = baseline 0.6
            confluences = 3
            if (m60 := ctx.underlying_momentum_60d.get(ticker)) is not None and m60 > 1.0:
                confluences += 1
            if ctx.cstability_vote_count is not None and ctx.cstability_vote_count == 0:
                confluences += 1
            if ctx.gex_regime == "long_gamma":
                confluences += 1  # sticky range favors directional small caps
            # Skew z: extreme positive (puts crowded) → contrarian +1; extreme
            # negative (calls crowded) → -1 (likely top, fade entries)
            sz = ctx.skew_z.get(ticker)
            if sz is not None and sz >= 1.5:
                confluences += 1
            elif sz is not None and sz <= -1.5:
                confluences -= 1
            # Tier1 Signal 2: volume breakout — institutional confirmation of move
            vs = ctx.volume_spike.get(ticker)
            if vs is not None:
                if vs > 5.0:
                    confluences += 2
                elif vs > 3.0:
                    confluences += 1
            # Tier1 Signal 3: insider cluster — multiple distinct insiders buying
            cl = ctx.insider_cluster_score.get(ticker, 0.0)
            if cl >= 3.0:
                confluences += 2
            elif cl >= 1.5:
                confluences += 1
            # Tier1 Signal 1: GDELT news sentiment + article volume
            if ctx.news_sentiment_24h.get(ticker, 0.0) > 0.3:
                confluences += 1
            if ctx.news_volume_spike.get(ticker, 1.0) > 3.0:
                confluences += 1
            # Edge normalized to [0.1, 1.0] across the now-13-confluence range
            # (3 hard gates + up to 4 original boosts + up to 6 Tier1 boosts).
            edge = min(1.0, max(0.1, confluences / 13.0))

            self._last_fired[ticker] = today
            # v4: 35 DTE (was 25), 12% OTM (was 10%), looser SL -70% (was -60%),
            # max_hold 25d (was time_stop only). Matches fish_ trader holding
            # behavior — let big moves develop, accept deeper drawdowns on
            # high-confluence setups.
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=28,   # v16: matches fish_'s 4-week DTE preference
                edge_score=edge,
                # v22 iter v6: 15% OTM still (cheap leverage). TP+200%/SL-40%
                # asymmetric (tighter SL preserves capital on failed catalysts).
                # max_hold 45d — catalyst plays need time. 8% NAV per trade —
                # smaller bets since signals less certain.
                target_otm_pct=0.15,
                # v28 (v15d): trailing SL handles asymmetric upside (no fixed TP).
                # v16 (2026-05-26): aligned to fish_'s actual trading behavior
                # (from his screenshots, Dec 2025 → May 2026):
                #   - Most trades held DAYS TO WEEKS, not months → max_hold 21d
                #   - 4-week DTE typical → target_dte 28 (set in strategy_args)
                # The asymmetric upside via trailing SL ladder is preserved.
                take_profit_pct=None,
                stop_loss_pct=-0.4,
                max_hold_days=21,   # v16: matches fish_'s actual hold periods
                max_per_trade_pct_nav=0.08,
                notes=f"iv={iv_rank_eff:.1f}, mom30={mom30_eff:.2f}, conf={confluences}",
            ))
        return signals
