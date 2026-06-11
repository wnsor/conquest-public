"""Strategy plug-in ABC + signal types.

A Strategy emits StrategySignal objects describing INTENT (buy a 30d ATM
call on SPY because Q+M signal triggered with edge_score=0.7). The main
algorithm then picks the actual contract (OptionSelector), sizes it
(PositionSizer), opens the position, and tracks exit rules (ExitManager).

This separation lets Phases 2-8 add strategies as small files that don't
know anything about Lean internals beyond reading the context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Side = Literal["call", "put", "straddle", "strangle", "leaps_call", "leaps_put"]
ExitReason = Literal[
    "take_profit", "stop_loss", "time_stop", "expiry",
    "regime_exit", "signal_exit", "manual",
]


@dataclass(frozen=True)
class StrategyContext:
    """Per-tick snapshot of everything strategies might inspect.

    The main algorithm populates this in OnData before iterating over
    strategies, so each strategy sees the same context within a tick.

    Fields are intentionally read-only; strategies should not mutate.
    """
    # Wall-clock time of the tick (algo.Time)
    timestamp: object  # datetime; loose typed to avoid Lean import here

    # Market data
    underlying_prices: dict[str, float] = field(default_factory=dict)

    # Dynamic PIT-momentum path (main.py DYNAMIC_PIT_MOMENTUM=1): the monthly
    # top-N point-in-time S&P-500 momentum selection currently eligible for new
    # entries. Empty on the static hand-picked-universe path; ignored by every
    # existing strategy (they iterate their own .universe instead).
    active_universe: list[str] = field(default_factory=list)
    # ticker → date of its most recent ENTRY (order placed), maintained by main.py.
    # Lets a strategy cool down on actual entries instead of on emits — an emit
    # that never fills (chain absent that bar) must NOT burn the cooldown, or the
    # name is locked out for the whole cycle. Empty on the static path.
    last_entry_date: dict[str, object] = field(default_factory=dict)

    vix: float | None = None
    vix3m: float | None = None
    vix9d: float | None = None
    vix_term_ratio: float | None = None     # VIX/VIX3M; >1 = backwardation
    vix9d_vix_ratio: float | None = None    # VIX9D/VIX; >1 = acute stress
    term_regime: str | None = None          # "contango" | "flat" | "backwardation"

    # Dealer positioning (computed from SPY chain in main.py)
    gex_total: float | None = None          # $bn / 1% spot move
    gex_regime: str | None = None           # "long_gamma" | "short_gamma" | "flip_zone"

    # Per-ticker 25Δ put-call IV spread
    skew: dict[str, float] = field(default_factory=dict)
    skew_z: dict[str, float] = field(default_factory=dict)

    # Insider Form 4 — tickers with opportunistic Officer/Director buys
    # in last N trading days (N configured in main.py; default 5).
    insider_recent_buys: dict[str, float] = field(default_factory=dict)  # ticker → most-recent dollar_value

    # Crisis state machine (Phase 6): normal | warning | crash | capitulation | rebound | recovery
    crisis_state: str | None = None
    crisis_vix_peak: float | None = None     # highest VIX seen in current crisis cycle

    # Conquest cross-fund signals (read from ObjectStore once per day)
    cstability_vote_count: int | None = None
    cgrowth_q_m_top5: list[str] = field(default_factory=list)
    regime: str | None = None  # "Inflation" | "Disinflation" | "Stagflation" | "Deflation"

    # Options-derived signals (computed by edge_signals/ modules)
    iv_rank: dict[str, float] = field(default_factory=dict)   # ticker → 0..100 percentile
    earnings_today: set[str] = field(default_factory=set)
    earnings_within_5d: set[str] = field(default_factory=set)
    last_earnings_surprise_pct: dict[str, float] = field(default_factory=dict)  # ticker → most recent surprise
    days_since_last_earnings: dict[str, int] = field(default_factory=dict)
    days_until_next_earnings: dict[str, int] = field(default_factory=dict)  # for pre-earnings strategies
    pc_ratio_equity: float | None = None
    short_pressure_fee_rate: dict[str, float] = field(default_factory=dict)
    uoa_active: set[str] = field(default_factory=set)        # tickers where UOA was detected today

    # Underlying momentum (ratio to N days ago; >1 = up, <1 = down)
    underlying_momentum_30d: dict[str, float] = field(default_factory=dict)
    underlying_momentum_60d: dict[str, float] = field(default_factory=dict)
    # v6: extended trend signals for GEX/LEAPS/Tepper strategies
    underlying_5ma_above_20ma: dict[str, bool] = field(default_factory=dict)
    underlying_drawdown_from_252d_high: dict[str, float] = field(default_factory=dict)  # 0.10 = 10% below high
    # v9: historical (realized) vol + IV/HV ratio — the missing options-pricing signals
    historical_vol_30d: dict[str, float] = field(default_factory=dict)  # annualized e.g. 0.35 = 35%
    historical_vol_60d: dict[str, float] = field(default_factory=dict)
    iv_raw: dict[str, float] = field(default_factory=dict)               # absolute ATM IV e.g. 0.28
    iv_hv_ratio: dict[str, float] = field(default_factory=dict)         # iv/hv30; <1 = options cheap

    # Tier 1 signals (price-derived + insider cluster + news sentiment)
    # volume_spike: today's $-volume / mean(prior 20d $-volume). >3 = institutional confirmation.
    volume_spike: dict[str, float] = field(default_factory=dict)
    # insider_cluster_score: weighted distinct insiders with buys in last 5d (Officer 2x, Director 1.5x, 10pct 1x)
    insider_cluster_score: dict[str, float] = field(default_factory=dict)
    # news_sentiment_24h: GDELT mean tone last 24h, rescaled to [-1.0, +1.0]
    news_sentiment_24h: dict[str, float] = field(default_factory=dict)
    # news_volume_spike: articles today / 30d avg article count. >3 = breaking-news intensity
    news_volume_spike: dict[str, float] = field(default_factory=dict)

    # ── LEADING signals (2026-05-26, for v_REFLEX_v2 and successors) ───────
    # These are computed in main.py from QC Object Store CSVs that are
    # refreshed daily by .github/workflows/refresh_data.yml. Empty dicts
    # mean the underlying data hasn't been backfilled yet — strategies
    # should fail-loud or skip the accelerator, not silently treat as 0.

    # short_interest_velocity: (latest_si_shares - prior_si_shares) / prior.
    # > +0.20 indicates accelerating short-side conviction (squeeze precursor).
    # Source: FINRA biweekly (storage/conquest/options/finra_si_biweekly.csv).
    short_interest_velocity: dict[str, float] = field(default_factory=dict)

    # insider_count_5d: count of DISTINCT insiders (CIK) who bought in last 5
    # trading days. 3+ from same company = active accumulation cluster.
    # Source: Form 4 daily (storage/conquest/insider/form4_opportunistic_buys_daily.csv).
    insider_count_5d: dict[str, int] = field(default_factory=dict)

    # news_propagation_5d: (mean article count last 5d) / (mean prior 5d).
    # > 1.50 means attention is accelerating — leading indicator of
    # institutional + retail focus consolidating BEFORE the price move.
    # Computed from GDELT count series.
    news_propagation_5d: dict[str, float] = field(default_factory=dict)

    # implied_move_vs_realized: ATM straddle IM_30d / HV_30d. >2.0 means
    # options pricing in a move that hasn't shown up in realized vol yet
    # (informed positioning ahead of a known catalyst).
    implied_move_vs_realized: dict[str, float] = field(default_factory=dict)

    # ─── Mix-and-match signals (2026-05-26 standby pool) ────────────────────
    # All optional — strategies handle None / missing gracefully so the
    # mix-and-match design lets multiple strategies share these fields.

    # CBOE SKEW index (tail-risk pricing, ^SKEW yfinance). > 130 = market
    # paying up for tail-risk insurance (leading for crash setups).
    cboe_skew: float | None = None

    # VVIX (vol-of-vol, ^VVIX yfinance). > 130 = vol regime potentially
    # shifting (leading for vol-spike risk).
    vvix: float | None = None

    # VIX percentile in 1y/5y lookback (computed from existing VIX series
    # in main.py). Low pct = complacent → leading for tail hedge.
    vix_percentile_1y: float | None = None
    vix_percentile_5y: float | None = None

    # Retail attention velocity (5d-mean / prior-5d-mean ratio).
    # Source: PRAW Reddit /r/wallstreetbets submission/comment counts.
    wsb_mention_velocity: dict[str, float] = field(default_factory=dict)

    # Google Trends search-volume velocity (5d/5d ratio).
    # Source: pytrends free Google Trends API.
    google_trends_velocity: dict[str, float] = field(default_factory=dict)

    # Earnings revision velocity (30d % change in consensus EPS estimate).
    # Source: yfinance `earningsTrend`. > +5% = analyst upgrades flowing.
    earnings_revision_velocity: dict[str, float] = field(default_factory=dict)

    # Recent 13D/G filings (tickers with new >5% stake filed in last 14d).
    # Source: EDGAR 13D/13G scrape.
    recent_13d_filing: set[str] = field(default_factory=set)

    # Recent 8-K filing burst count (14-day rolling).
    # Source: EDGAR 8-K. > 3 in 14d for a single ticker = material disclosures.
    recent_8k_count: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategySignal:
    """An intent emitted by a strategy. The framework converts this to
    actual order(s) by picking a contract from the chain.

    Either target_delta or target_otm_pct must be set (not both). Negative
    delta means put; positive means call.

    edge_score (0.0 .. 1.0): how confluent the entry signal is. Used by
    PositionSizer to scale into stronger setups (e.g. 1 confluence → 1%
    NAV, 3 confluences → 3% NAV).
    """
    strategy_id: str
    underlying: str
    side: Side
    target_dte: int                        # picker tolerates ±5d
    edge_score: float                      # 0.0 .. 1.0
    target_delta: float | None = None      # signed; +0.50 = ATM call, -0.20 = OTM put
    target_otm_pct: float | None = None    # 0.05 = 5% OTM
    take_profit_pct: float | None = None   # 2.00 = exit at +200%
    stop_loss_pct: float | None = None     # -0.50 = exit at -50%
    time_stop_dte: int | None = None       # exit if DTE drops below this
    max_hold_days: int | None = None       # exit N calendar days after entry (calendar-event timing)
    max_concurrent_per_underlying: int = 1
    # v22: per-strategy override of position_sizer's base_pct_nav cap.
    # Default None → use the sizer's global cap (typically 0.10 = 10%).
    # Set explicitly to allow rare/high-conviction strategies (e.g.
    # tepper_vbottom_leaps, crisis_rebound_basket, cgrowth_leaps) to take
    # larger positions than the per-trade-gate strategies. The strategy is
    # responsible for setting this only when the trade thesis justifies
    # concentration (e.g. 1-3 fires per decade with asymmetric V-bottom
    # payoff = bounded concentration risk; OK to use 25% NAV).
    max_per_trade_pct_nav: float | None = None
    notes: str = ""


@runtime_checkable
class Strategy(Protocol):
    """Duck-typed protocol — any class with these attrs/methods is a Strategy."""

    id: str
    enabled: bool
    universe: list[str]  # underlying tickers this strategy needs subscribed

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        """Return zero or more signals for this tick. May be called every
        OnData, but strategies are responsible for their own throttling
        (e.g. monthly rebalance, only on event triggers)."""
        ...
