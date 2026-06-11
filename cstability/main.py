# region imports
from AlgorithmImports import *
from datetime import timedelta
from io import StringIO
import pandas as pd
# endregion


class Cstability(QCAlgorithm):
    """
    Conquest Stability v10 — Top-3 Dual Momentum + 4-Vote MultiSignalVote Ensemble.

    Mandate
    -------
    Equity-only, drawdown-focused. No leverage. Mandate: preserve capital with
    smooth equity curves AND match buy-and-hold SPY return as a hard floor.
    cstability is the DD-priority side of the two-fund pair (cgrowth handles
    max-return). v10 supersedes v8's deterministic regime rotation with a
    4-vote ensemble that catches stress earlier through independent signals,
    using consensus voting to avoid v8.1's 37.5% false-positive rate.

    Strategy
    --------
    Universe: 28 ETFs spanning equity (broad indices, sectors, low-vol),
    fixed income (treasuries, IG, HY, TIPS), real assets (gold, broad
    commodities, REITs), and international (developed + EM).

    Daily check (scheduled 09:35 ET, 5min after SPY market open):
        Refresh the 4 vote signals from Object Store. Update self.vote_count.

    Each rebalance (scheduled 10:00 ET, 30min after SPY open, 1st trading day):
        1. Compute BASE allocation:
            IF regime is Stagflation:
                base = equal-weight {GLD, TIP, TLT}  (defensive rotation)
            ELSE:
                base = top-3 by 252d momentum, dual-momentum filter
        2. Compute alpha = blend_weights[vote_count]
            blend_weights = [0.0, 0.15, 0.40, 0.75, 1.0]   # for 0/1/2/3/4 votes
        3. final_weights = (1 - alpha) * base + alpha * STAGFLATION_BASKET
        4. Liquidate names not in final_weights, set holdings per weight.

    The 3 votes (v11.2 — P(Stag) ablated 2026-05-04)
    -----------
        1. Macro regime classifier flags Stagflation     (direct macro; ~1mo lag)
        2. HY-IEF 60d log spread < -0.05                 (medium; credit-market lead)
        3. VIX/VIX3M > 1.05                              (fast; vol curve backwardation)

    Each is independent — different drivers, different lead times. Consensus
    voting avoids the false-positive trap (single-signal VIX > 25 fired on
    37.5% of trading days in v8.1).

    P(Stagflation) > 0.30 was vote 2 of v11 LIVE. Ablation backtest
    (cstability id df2949e3f831c0d9bbba0b71f4e9013d, INCLUDE_PROB_VOTE=0,
    2008-01-01 → 2026-02-04) returned identical metrics to v11 LIVE:
    CAGR 12.33%, Sharpe 0.656, DD -19.9%, PSR 15.42%, end $205,366. The GBM
    forecast was redundant — same GDP/CPI inputs as the regime classifier,
    saturated by credit + VIX-term in real stress. Default flipped to
    INCLUDE_PROB_VOTE=0 in v11.2; set to 1 to revert to 4-vote v11 LIVE.

    Why this works (per-regime + per-vote-count breakdown 2014-2024)
    --------------
    Per-regime equity Sharpe on this universe:
        Disinflation  Sharpe 1.32  ← BEST regime
        Deflation     Sharpe 1.20  ← second-best (v3 wrongly halved here)
        Inflation     Sharpe 0.60
        Stagflation   Sharpe 0.35  ← entire DD source — gate must avoid this only

    Per-vote-count blend (vote_count ranges 0..3 in v11.2; index-4 unreachable):
        0 votes -> 0%   defensive (pure momentum or regime-rotated base)
        1 vote  -> 15%  defensive (early warning — single signal isolated)
        2 votes -> 40%  defensive (warning — consensus emerging)
        3 votes -> 75%  defensive (confirmed risk-off — max defensive in v11.2)
        [4 votes -> 100% defensive] — UNREACHABLE in v11.2 (was full rotation in v11 LIVE)

    v10 vs v8 + v9.2 (2014-2024 pandas bake-off, 28-ETF universe, monthly, IB 2bps)
    --------------------------------------------------
        v10  4-vote ensemble                CAGR 10.60%  Sharpe 0.79  DD -17.7%  Calmar 0.60  ← BEST CALMAR
        v9.2 3-vote stack (no prob)         CAGR 11.03%  Sharpe 0.80  DD -19.8%  Calmar 0.56
        v8   regime-rotated only            CAGR 10.86%  Sharpe 0.76  DD -22.5%  Calmar 0.48
        v3   LIVE (RSI/MACD/TRIX + halve)   CAGR  3.08%  Sharpe 0.40  DD -24.7%  Calmar 0.12
        SPY benchmark                       CAGR 13.19%  Sharpe 0.81  DD -33.7%  Calmar 0.39

    v10 trades 0.4pp CAGR for 2.1pp DD reduction vs v9.2 — the highest Calmar
    in the entire cstability lineup. Strict win for the DD-priority mandate.

    Cost model
    ----------
    SetBrokerageModel(INTERACTIVE_BROKERS_BROKERAGE) applies IB tiered
    commissions plus standard fill / slippage. No hand-coded costs.
    """

    # 28-ETF universe (added GLD/DBC/TIP/VNQ/EFA/IEMG/HYG/LQD/BIL to v3's 19)
    UNIVERSE_FULL = [
        "SPY", "QQQ", "IWM",                                 # broad indices
        "XLK", "XLF", "XLV", "XLE", "XLY", "XLP",            # SPDR sector ETFs
        "XLI", "XLB", "XLU", "XLRE", "XLC",
        "SPLV", "USMV",                                      # low-vol
        "AGG", "TLT", "IEF",                                 # treasuries / aggregate bonds
        "GLD",                                               # gold (Stagflation hedge)
        "DBC",                                               # broad commodities
        "TIP",                                               # TIPS
        "VNQ",                                               # REITs
        "EFA", "IEMG",                                       # developed + emerging international
        "HYG", "LQD", "BIL",                                 # credit + cash equivalent
    ]

    # 23-ETF trimmed universe — drops late-inception names so backtest can extend
    # back to 2008 (covers full GFC peak Oct-2007 → trough Mar-2009).
    # Dropped: XLRE (2015), XLC (2018), SPLV (2011), USMV (2011), IEMG (2012).
    # Activate via --parameter TRIMMED_UNIVERSE 1.
    UNIVERSE_TRIMMED = [t for t in UNIVERSE_FULL if t not in ("XLRE", "XLC", "SPLV", "USMV", "IEMG")]

    # Active universe — set by initialize() based on TRIMMED_UNIVERSE parameter.
    # Default UNIVERSE = full 28-ETF list (preserves v10 LIVE behavior for default backtest window 2018-2024).
    UNIVERSE = UNIVERSE_FULL

    TOP_N                  = 3
    MOMENTUM_LOOKBACK_DAYS = 252
    WARMUP_BARS            = 252 + 20  # MOMP(252) + small buffer
    STAGFLATION_BASKET     = ("GLD", "TIP", "TLT")

    # Object Store keys for the 4 vote signals (regime is the existing classifier)
    REGIME_STORE_KEY        = "conquest/regime/daily.csv"
    PROBABILITY_STORE_KEY   = "conquest/regime/probability.csv"
    CREDIT_STRESS_STORE_KEY = "conquest/credit/hyg_ief_spread.csv"
    VIX_TERM_STORE_KEY      = "conquest/vix/term_ratio.csv"
    # v10.5a candidate: T10Y2Y yield curve inversion as substitute for P(Stagflation)
    T10Y2Y_STORE_KEY        = "conquest/yield_curve/t10y2y.csv"
    UNRATE_STORE_KEY        = "conquest/macro/unrate.csv"
    # v11.3 candidate: GMM regime probabilities (4-component Gaussian Mixture
    # on (gdp_yoy_z, cpi_yoy_z)). When REGIME_USE_GMM=1 the regime vote uses
    # P(Stagflation) >= GMM_STAG_THRESHOLD instead of the deterministic
    # quadrant-of-z-score label. Default REGIME_USE_GMM=0 = LIVE behavior.
    GMM_REGIME_STORE_KEY    = "conquest/regime/gmm_daily.csv"

    # v10 vote thresholds (chosen by inspection of 2008/2018/2020/2022 stress episodes,
    # NOT optimized in-sample). Match the pandas v10 candidate in scripts/rank_models.py.
    PROB_STAG_THRESHOLD     = 0.30
    CREDIT_STRESS_THRESHOLD = -0.05
    VIX_TERM_THRESHOLD      = 1.05
    # v10.5a: T10Y2Y is "inverted" (recession-signaling) when the spread is below 0.
    # Inversion has historically led recessions by 6-18 months; combined with the
    # other 3 votes, gives an early-warning consensus that the noisy P(Stagflation)
    # GBM forecast was supposed to provide but didn't (2007-2024 extended bake-off
    # showed P(Stagflation) is the source of v10's residual false-positive sensitivity).
    T10Y2Y_INVERSION_THRESHOLD = 0.0

    # Blend weights: vote_count → defensive basket fraction.
    # Index 0..4 = 0..4 votes. Match the pandas v10 candidate.
    BLEND_WEIGHTS = [0.0, 0.15, 0.40, 0.75, 1.0]

    # ---------- v10.5a candidate: T10Y2Y replaces P(Stagflation) (default OFF — LIVE behavior) ----------
    # Vote-set selection: "v10" (LIVE pin: regime + prob + credit + vix backwardation)
    # or "v10.5a" (regime + T10Y2Y inversion + credit + vix backwardation, dropping the
    # noisy P(Stagflation) GBM forecast). Default "v10" preserves LIVE pin behavior.
    # Override via --parameter VOTE_MODE v10.5a.
    VOTE_MODE                  = "v10"

    # ---------- v11.2 LIVE: P(Stagflation) vote dropped (INCLUDE_PROB_VOTE=0 default) ----------
    # Promoted 2026-05-04. Ablation backtest df2949e3f831c0d9bbba0b71f4e9013d
    # (INCLUDE_PROB_VOTE=0, PIT 2008-2026) returned identical metrics to v11 LIVE
    # 4-vote: CAGR 12.33%, Sharpe 0.656, DD -19.9%, PSR 15.42%, end $205,366.
    # Cause: P(Stag) reads the same GDP/CPI z-scores as the regime classifier (vote 1),
    # so the "forecast leads deterministic label" hypothesis didn't hold; in real
    # stress the credit + VIX-term votes saturated the defensive trigger and the
    # SJSU vote contributed nothing. Set INCLUDE_PROB_VOTE=1 to revert to v11 LIVE
    # 4-vote behavior. Different from VOTE_MODE="v10.5a" which REPLACES prob with
    # T10Y2Y; v11.2 simply drops the vote.
    INCLUDE_PROB_VOTE          = 0

    # ---------- v11.3 candidate: GMM probabilistic regime classifier (default OFF — LIVE behavior) ----------
    # When REGIME_USE_GMM=1, the regime vote becomes
    #     p_stagflation >= GMM_STAG_THRESHOLD
    # instead of the deterministic
    #     gdp_yoy_z < 0 AND cpi_yoy_z > 0  (i.e. regime label == "Stagflation")
    # The GMM is fit offline by scripts/export_gmm_regime.py on the same
    # (gdp_yoy_z, cpi_yoy_z) z-scores the deterministic classifier uses, and
    # serialized as conquest/regime/gmm_daily.csv. The GMM candidate was
    # tested in 2026-05-04 paired backtests and FAILED strict-Pareto vs
    # v11.2 LIVE (CAGR −0.22 pp, end equity −3.4%); kept as gated parameter
    # for future revisits. See LEARNINGS.md for full results.
    REGIME_USE_GMM             = 0
    GMM_STAG_THRESHOLD         = 0.50

    # ---------- v11.3 candidates: ADDITIONAL votes (default OFF — LIVE behavior) ----------
    # Add UNRATE (Sahm-rule unemployment) and/or T10Y2Y (yield-curve inversion) as
    # ADDITIONAL votes on top of the base v10/v10.5a vote set. These restore the
    # 4th-vote symmetry v11.2 dropped (when INCLUDE_PROB_VOTE went 1→0) without
    # re-introducing the regime/prob redundancy that motivated the drop.
    #
    # INCLUDE_T10Y2Y_VOTE only takes effect when VOTE_MODE=="v10"; in v10.5a mode
    # T10Y2Y is already a replacement vote and toggling this would double-count.
    #
    # vote_count is clamped to len(BLEND_WEIGHTS)-1 = 4 to avoid blend-weight
    # overflow (votes ≥ 4 all map to fully defensive 100% basket).
    #
    # Test plan: paired backtest each candidate vs v11.2 LIVE on PIT 2008-2026,
    # promote only if strict-Pareto better on every metric. See LEARNINGS.md
    # for prior v11.3 GMM rejection rationale.
    INCLUDE_UNRATE_VOTE        = 0
    INCLUDE_T10Y2Y_VOTE        = 0   # ADD as 4th/5th vote (distinct from VOTE_MODE=v10.5a which REPLACES prob)
    UNRATE_RISE_THRESHOLD      = 0.3 # Sahm-rule-like: unrate ≥ 12mo-min + 0.3pp triggers vote

    # ---------- DD circuit-breaker (default OFF — LIVE behavior) ----------
    # Strategy-level DD circuit-breaker for cstability's DD-priority mandate.
    # Tracks rolling 30d high-water mark of NAV; reduces gross by scale factor
    # when DD exceeds thresholds. cstability is the right home for a DD breaker
    # (DD reduction is its mandate; cgrowth/cgrowth_options/cgrowth_calls are
    # max-return funds where a DD breaker would clip V-recoveries).
    # Default OFF for clean comparison vs v10 LIVE; enable via --parameter
    # DD_CIRCUIT_BREAKER_ENABLED 1.
    # Thresholds calibrated for cstability's tighter DD profile (LIVE Lean DD -17.1%):
    # -8% halves; -12% quarters. Looser than the cgrowth-side defaults would have
    # been, because cstability's natural DD floor is shallower.
    DD_CIRCUIT_BREAKER_ENABLED = 0
    DD_HIGH_WATER_WINDOW       = 30
    DD_LEVEL_1_THRESHOLD       = -0.08
    DD_LEVEL_1_SCALE           = 0.50
    DD_LEVEL_2_THRESHOLD       = -0.12
    DD_LEVEL_2_SCALE           = 0.25

    # ---------- v11 candidate: SPY floor at 0 votes + BIL cash at max stress ----------
    # v10 LIVE has been observed to underperform buy-and-hold SPY by ~4-5pp annual
    # CAGR (cloud Lean 2018-2024: v10 ~17.2% vs SPY ~21.7%). Cause: top-3 ETF
    # momentum concentration in calm-bull regimes (when 0 votes fire) holds whatever
    # has been highest-momentum over 252d, often defensive sectors that lag a tech-led
    # rally. v11 fixes this by replacing the calm-regime base allocation with an
    # SPY/QQQ floor: when 0 votes fire AND not in Stagflation regime, hold SPY/QQQ
    # equal-weight instead of top-3 momentum. Plus an optional BIL-cash path when
    # 4 votes fire (true crisis), since 2022 demonstrated GLD/TIP/TLT can lose
    # alongside equities in rate-shock regimes — cash beats both in those windows.
    # Both parameters default OFF (preserves v10 LIVE behavior); enable via
    # --parameter USE_SPY_FLOOR_AT_ZERO_VOTES 1 (and/or USE_BIL_CASH_AT_MAX_STRESS 1).
    USE_SPY_FLOOR_AT_ZERO_VOTES = 0       # 0 = top-3 momentum (v10 LIVE), 1 = floor basket
    # FLOOR_MODE selects which floor basket (when USE_SPY_FLOOR_AT_ZERO_VOTES=1):
    #   "spy_qqq" — SPY/QQQ equal-weight (matches SPY closely; DD ≈ SPY -33%)
    #   "splv_usmv" — low-vol equity ETFs SPLV/USMV (captures ~70-80% of SPY upside
    #                with ~70% of the DD; better DD profile but lower CAGR floor)
    #   "spy_splv" — 50/50 SPY + SPLV (compromise between max-return and DD)
    FLOOR_MODE                  = "spy_qqq"
    USE_BIL_CASH_AT_MAX_STRESS  = 0       # 0 = full defensive basket (LIVE), 1 = 100% BIL at 4 votes
    CASH_TICKER                 = "BIL"   # short-term treasuries; already in UNIVERSE

    # ---------- v11 Layer 1: VIX-based fast cash trigger (v11 LIVE — defaults ON) ----------
    # Daily handler that bypasses the slow vote ensemble for spot-vol shocks.
    # When spot VIX > LAYER1_VIX_HIGH, override portfolio to 100% BIL immediately.
    # Hysteresis: enter at HIGH, exit at LOW. 90-day guardrail forces re-eval if
    # hysteresis hasn't resolved (avoids "stuck in cash forever").
    # Calibrated for COVID Feb-Mar 2020 (VIX > 30 from Feb 27 at SPY $297) which
    # v10's monthly-cadence gate caught 5 weeks late. Layered cash saves ~23pp DD
    # on PIT 2008-2024. Override via --parameter LAYER1_CASH_ENABLED 0 to revert.
    LAYER1_CASH_ENABLED       = 1
    LAYER1_VIX_HIGH           = 30.0
    LAYER1_VIX_LOW            = 25.0
    LAYER1_MAX_DAYS_IN_CASH   = 90

    # ---------- v11 slow layer: defensive basket mode (v11 LIVE — defaults to bil_cash) ----------
    # "gld_tip_tlt" — 33/33/33 GLD/TIP/TLT (legacy v10 Stagflation hedge)
    # "bil_cash"    — 100% BIL whenever defensive blend is active (vote_count >= 1)
    #                 Eliminates 2022-style basket crashes (TLT -29% in rate-shock).
    # Override via --parameter DEFENSIVE_BASKET_MODE gld_tip_tlt to revert.
    DEFENSIVE_BASKET_MODE     = "bil_cash"

    # ---------- v11 slow layer: intra-month edge trigger (v11 LIVE — defaults ON) ----------
    # When vote_count changes intra-day, force an immediate rebalance instead of
    # waiting for next month_start. Required so fast crashes (Lehman week, COVID
    # onset) don't wait 2-4 weeks for the slow-layer blend to update. Same pattern
    # as cgrowth v11 crisis-gate (cgrowth/main.py:402-411) but fires on any
    # vote_count change since cstability's BLEND_WEIGHTS is 5-level granular vs
    # cgrowth's binary gate. Override via --parameter INTRA_MONTH_EDGE_ENABLED 0.
    INTRA_MONTH_EDGE_ENABLED  = 1

    # Spot VIX object store key (shared with cgrowth/cgrowth_options).
    # Populated by scripts/export_vix.py from yfinance ^VIX, 2008-01-01 onwards.
    VIX_SPOT_STORE_KEY        = "conquest/vix/daily.csv"

    # ---------- v11.1 re-entry trigger (v11 LIVE — defaults ON, VIX_MAX=30) ----------
    # Forces 100% equity from any defensive state when both vol AND price momentum
    # confirm a regime change: VIX < REENTRY_VIX_MAX AND SPY 20d ROC > REENTRY_SPY_ROC_MIN.
    # After the trigger fires, rebalance() skips the defensive alpha-blend for
    # REENTRY_HOLD_DAYS (default 5) so the next monthly rebalance doesn't
    # immediately undo the override. Also serves as a whipsaw dampener.
    #
    # **Empirical calibration (2026-05-04):** the original handoff spec called
    # for VIX_MAX=25, but PIT 2008-2024 sweep showed VIX<25 fires too late post-
    # stress (COVID 2020: VIX hit 25 only on Jun 5, ~3mo after Mar bottom; GFC
    # 2009: VIX hit 25 only in Aug, ~5mo after Mar bottom). VIX_MAX=30 fires
    # earlier and captures the late-rebound phase where slow-layer is still
    # defensive (BIL) but momentum has confirmed. PIT 17yr: VIX_MAX=30 gives
    # CAGR 11.22% / Sharpe 0.607 / DD -19.9% / PSR 9.48% (vs VIX_MAX=25 at
    # 11.01% / 0.595 / -21.7% / 8.59% — strict Pareto). Dropping VIX entirely
    # (VIX_MAX=99) tested worse (CAGR 10.56% / PSR 6.35%) — too many false
    # positives during ongoing stress. Override via --parameter REENTRY_ENABLED 0.
    REENTRY_ENABLED            = 1
    REENTRY_VIX_MAX            = 30.0
    REENTRY_SPY_ROC_DAYS       = 20
    REENTRY_SPY_ROC_MIN        = 0.05    # +5% over 20d
    REENTRY_HOLD_DAYS          = 5       # min hold; also blocks defensive blend during window

    def initialize(self):
        # Default: 2018-2025 (matches v10 LIVE Lean window). Overridable via --parameter.
        start_year = int(self.get_parameter("BACKTEST_START_YEAR") or 2018)
        end_year   = int(self.get_parameter("BACKTEST_END_YEAR")   or 2025)
        self.set_start_date(start_year, 1, 1)
        self.set_end_date(end_year, 12, 31)
        self.set_cash(25000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE)
        self.set_benchmark("SPY")

        # Slippage model — IB-realistic VolumeShareSlippageModel scales price
        # impact with order size relative to bar volume. Disable via SLIPPAGE_MODEL=none.
        slip_mode = (self.get_parameter("SLIPPAGE_MODEL") or "volume_share").lower()
        if slip_mode == "volume_share":
            def _init_security(security):
                try: security.set_slippage_model(VolumeShareSlippageModel(0.025, 0.1))
                except Exception: pass
            self.set_security_initializer(_init_security)
            self.log("[cstability] slippage: VolumeShareSlippageModel(0.025, 0.1)")
        elif slip_mode != "none":
            self.log(f"[cstability] slippage: NONE (Lean default)")

        # Parameter overrides for v10.5a vote-mode + DD breaker + v11 SPY-floor / BIL-cash.
        for attr, key, cast in [
            ("VOTE_MODE",                  "VOTE_MODE",                  str),
            ("INCLUDE_PROB_VOTE",          "INCLUDE_PROB_VOTE",          int),
            ("REGIME_USE_GMM",             "REGIME_USE_GMM",             int),
            ("GMM_STAG_THRESHOLD",         "GMM_STAG_THRESHOLD",         float),
            # v11.3 candidate ADDITIONAL votes (default OFF — preserves v11.2 LIVE)
            ("INCLUDE_UNRATE_VOTE",        "INCLUDE_UNRATE_VOTE",        int),
            ("INCLUDE_T10Y2Y_VOTE",        "INCLUDE_T10Y2Y_VOTE",        int),
            ("UNRATE_RISE_THRESHOLD",      "UNRATE_RISE_THRESHOLD",      float),
            ("DD_CIRCUIT_BREAKER_ENABLED", "DD_CIRCUIT_BREAKER_ENABLED", int),
            ("DD_HIGH_WATER_WINDOW",       "DD_HIGH_WATER_WINDOW",       int),
            ("DD_LEVEL_1_THRESHOLD",       "DD_LEVEL_1_THRESHOLD",       float),
            ("DD_LEVEL_1_SCALE",           "DD_LEVEL_1_SCALE",           float),
            ("DD_LEVEL_2_THRESHOLD",       "DD_LEVEL_2_THRESHOLD",       float),
            ("DD_LEVEL_2_SCALE",           "DD_LEVEL_2_SCALE",           float),
            ("USE_SPY_FLOOR_AT_ZERO_VOTES","USE_SPY_FLOOR_AT_ZERO_VOTES",int),
            ("FLOOR_MODE",                 "FLOOR_MODE",                 str),
            ("USE_BIL_CASH_AT_MAX_STRESS", "USE_BIL_CASH_AT_MAX_STRESS", int),
            # v11 Layer 1 + slow-layer additions
            ("LAYER1_CASH_ENABLED",        "LAYER1_CASH_ENABLED",        int),
            ("LAYER1_VIX_HIGH",            "LAYER1_VIX_HIGH",            float),
            ("LAYER1_VIX_LOW",             "LAYER1_VIX_LOW",             float),
            ("LAYER1_MAX_DAYS_IN_CASH",    "LAYER1_MAX_DAYS_IN_CASH",    int),
            ("DEFENSIVE_BASKET_MODE",      "DEFENSIVE_BASKET_MODE",      str),
            ("INTRA_MONTH_EDGE_ENABLED",   "INTRA_MONTH_EDGE_ENABLED",   int),
            # v11.1 re-entry trigger
            ("REENTRY_ENABLED",            "REENTRY_ENABLED",            int),
            ("REENTRY_VIX_MAX",            "REENTRY_VIX_MAX",            float),
            ("REENTRY_SPY_ROC_DAYS",       "REENTRY_SPY_ROC_DAYS",       int),
            ("REENTRY_SPY_ROC_MIN",        "REENTRY_SPY_ROC_MIN",        float),
            ("REENTRY_HOLD_DAYS",          "REENTRY_HOLD_DAYS",          int),
        ]:
            v = self.get_parameter(key)
            if v:
                try:
                    setattr(self, attr, cast(v))
                except ValueError:
                    pass

        # TRIMMED_UNIVERSE override — drops late-inception ETFs so we can backtest
        # back to 2008-01-01 (covers full GFC + EU crisis + taper tantrum + Volmageddon
        # + COVID + 2022 inflation). Default OFF preserves v10 LIVE 28-ETF behavior.
        if int(self.get_parameter("TRIMMED_UNIVERSE") or 0):
            self.UNIVERSE = self.UNIVERSE_TRIMMED
            self.log(f"TRIMMED_UNIVERSE active: {len(self.UNIVERSE)} ETFs (dropped XLRE/XLC/SPLV/USMV/IEMG)")
        else:
            self.UNIVERSE = self.UNIVERSE_FULL

        self.signals: dict = {}                  # Symbol -> {"momp"}
        self.symbols_by_ticker: dict = {}        # str -> Symbol
        self.symbol_to_ticker: dict = {}         # Symbol -> str
        for ticker in self.UNIVERSE:
            equity = self.add_equity(ticker, Resolution.DAILY)
            sym = equity.symbol
            self.symbols_by_ticker[ticker] = sym
            self.symbol_to_ticker[sym] = ticker
            self.signals[sym] = {
                "momp": self.MOMP(sym, self.MOMENTUM_LOOKBACK_DAYS, Resolution.DAILY),
            }

        # v11.1 SPY 20d ROC for re-entry trigger (consulted when REENTRY_ENABLED=1).
        # SPY is guaranteed to be in UNIVERSE (broad-index anchor). Built-in ROC
        # indicator updates daily; ready after REENTRY_SPY_ROC_DAYS bars.
        spy_sym = self.symbols_by_ticker.get("SPY")
        self.spy_roc = self.ROC(spy_sym, self.REENTRY_SPY_ROC_DAYS, Resolution.DAILY) if spy_sym is not None else None

        self.set_warm_up(self.WARMUP_BARS, Resolution.DAILY)

        # Production hardening — shared across all Conquest projects.
        # Wires email alerts, slippage, state persistence, freshness alerts,
        # and in-QC FRED refresh. Idempotent if any of these are already set.
        try:
            from conquest.production import harden
            harden(self)
        except Exception as e:
            self.log(f"[cstability] production hardening skipped: {e}")

        # ---- Schedule: monthly rebalance + daily vote refresh ----
        self.schedule.on(
            self.date_rules.month_start("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.rebalance,
        )
        # Daily check — refreshes vote_count, then evaluates v11 Layer 1 cash
        # hysteresis and slow-layer intra-month edge trigger. Both gated behind
        # parameter flags (default OFF preserves v10 LIVE behavior).
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 5),
            self._daily_check,
        )

        # Load all signal feeds at startup. Probability + T10Y2Y are conditionally
        # used per VOTE_MODE — both loaded eagerly so vote_mode toggle doesn't need
        # late-binding I/O.
        self.regime_df       = self._load_csv(self.REGIME_STORE_KEY,        "regime")
        self.probability_df  = self._load_csv(self.PROBABILITY_STORE_KEY,   "probability")
        self.credit_df       = self._load_csv(self.CREDIT_STRESS_STORE_KEY, "credit stress")
        self.vix_term_df     = self._load_csv(self.VIX_TERM_STORE_KEY,      "VIX term ratio")
        # v10.5a feed (only consulted when VOTE_MODE == "v10.5a")
        # T10Y2Y feed loaded when used as either replacement (VOTE_MODE=v10.5a) OR additional vote (INCLUDE_T10Y2Y_VOTE=1)
        self.t10y2y_df       = self._load_csv(self.T10Y2Y_STORE_KEY,        "T10Y2Y yield curve") if (self.VOTE_MODE != "v10" or self.INCLUDE_T10Y2Y_VOTE) else None
        self.unrate_df       = self._load_csv(self.UNRATE_STORE_KEY,        "UNRATE unemployment") if self.INCLUDE_UNRATE_VOTE else None
        # v11.3 candidate feed (only consulted when REGIME_USE_GMM == 1)
        self.gmm_regime_df   = self._load_csv(self.GMM_REGIME_STORE_KEY,    "GMM regime") if self.REGIME_USE_GMM else None
        # v11 Layer 1 spot VIX feed (consulted when LAYER1_CASH_ENABLED=1)
        self.vix_spot_df     = self._load_csv(self.VIX_SPOT_STORE_KEY,      "VIX spot")

        self.stag_symbols = [
            self.symbols_by_ticker[t]
            for t in self.STAGFLATION_BASKET
            if t in self.symbols_by_ticker
        ]
        if len(self.stag_symbols) != len(self.STAGFLATION_BASKET):
            self.error(
                f"Stagflation basket missing tickers; got {len(self.stag_symbols)} of "
                f"{len(self.STAGFLATION_BASKET)}. Check UNIVERSE includes "
                f"{self.STAGFLATION_BASKET}."
            )

        # Vote state (refreshed daily; logged at rebalance)
        self.vote_count = 0
        self.last_votes = (False, False, False, False)

        # DD circuit-breaker state (only used if DD_CIRCUIT_BREAKER_ENABLED=1)
        self.nav_history: list[float] = []
        self.last_base_weights: dict = {}
        self.last_dd_scale: float = 1.0

        # v11 Layer 1 state (only used if LAYER1_CASH_ENABLED=1)
        self.in_layer1_cash: bool = False
        self.layer1_entered_at = None
        # v11 intra-month edge tracking (only used if INTRA_MONTH_EDGE_ENABLED=1)
        self.last_vote_count: int = 0
        # v11.1 re-entry state (only used if REENTRY_ENABLED=1)
        self.last_reentry_at = None

    # ---------- Object Store loaders ----------

    def _load_csv(self, key: str, label: str):
        try:
            csv_text = self.object_store.read(key)
        except Exception as e:
            self.error(f"Object Store read of {key} failed: {e}")
            return None
        if not csv_text:
            self.log(f"WARN: {key} not in Object Store; {label} vote will return False.")
            return None
        try:
            df = pd.read_csv(StringIO(csv_text), index_col=0, parse_dates=True)
            self.log(
                f"{label} feed: {len(df)} rows, range "
                f"{df.index[0].date()} -> {df.index[-1].date()}"
            )
            return df
        except Exception as e:
            self.error(f"Failed to parse {label} CSV: {e}")
            return None

    # ---------- vote signal lookups (latest value at-or-before self.time) ----------

    def _latest_value(self, df, column: str | None = None):
        if df is None:
            return None
        valid = df.index <= self.time
        if not valid.any():
            return None
        row = df.loc[valid].iloc[-1]
        if column is None:
            return row.iloc[0] if hasattr(row, "iloc") else row
        return row[column] if column in row else None

    def _current_regime(self):
        return self._latest_value(self.regime_df, column="regime")

    def _current_vix_spot(self):
        """Latest spot VIX value at-or-before self.time, or None if unavailable.

        Reads from VIX_SPOT_STORE_KEY (conquest/vix/daily.csv); used by v11
        Layer 1 cash trigger and v11.1 re-entry trigger.
        """
        if self.vix_spot_df is None:
            return None
        valid = self.vix_spot_df.index <= self.time
        if not valid.any():
            return None
        return float(self.vix_spot_df.loc[valid].iloc[-1, 0])

    def _vote_regime_stagflation(self) -> bool:
        # v11.3 candidate: GMM probabilistic classifier. When REGIME_USE_GMM=1,
        # the vote fires when P(Stagflation) from the GMM >= GMM_STAG_THRESHOLD.
        # When REGIME_USE_GMM=0 (default, LIVE behavior), use the deterministic
        # quadrant-of-z-score regime label.
        if self.REGIME_USE_GMM and self.gmm_regime_df is not None:
            p = self._latest_value(self.gmm_regime_df, column="p_stagflation")
            return p is not None and float(p) >= self.GMM_STAG_THRESHOLD
        return self._current_regime() == "Stagflation"

    def _vote_probability_high(self) -> bool:
        # SJSU ablation hook: when INCLUDE_PROB_VOTE=0, force the vote False regardless
        # of the actual P(Stag) feed value. Default 1 preserves v11 LIVE behavior.
        if not self.INCLUDE_PROB_VOTE:
            return False
        p = self._latest_value(self.probability_df, column="p_stagflation")
        return p is not None and float(p) > self.PROB_STAG_THRESHOLD

    def _vote_credit_stress(self) -> bool:
        c = self._latest_value(self.credit_df, column="hy_stress_proxy")
        return c is not None and float(c) < self.CREDIT_STRESS_THRESHOLD

    def _vote_vix_backwardation(self) -> bool:
        v = self._latest_value(self.vix_term_df, column="vix_term_ratio")
        return v is not None and float(v) > self.VIX_TERM_THRESHOLD

    def _vote_yield_curve_inverted(self) -> bool:
        """v10.5a vote: T10Y2Y < 0 (yield-curve inversion) — recession lead indicator.

        Inversion historically leads recessions by 6-18 months. Combined with
        regime, credit stress, and VIX term backwardation, gives a 4-driver
        consensus that's more independent than v10's prob+regime overlap.
        Returns False if T10Y2Y feed is missing (falls back to no signal).
        """
        if self.t10y2y_df is None:
            return False
        # Try common column names; FRED ALFRED export uses "value" but conquest's
        # custom export may use "t10y2y" or "spread"
        for col in ("t10y2y", "spread", "value"):
            v = self._latest_value(self.t10y2y_df, column=col)
            if v is not None:
                return float(v) < self.T10Y2Y_INVERSION_THRESHOLD
        return False

    def _vote_unemployment_rising(self) -> bool:
        """v11.3 candidate vote: Sahm-rule-like unemployment trigger.

        Sahm rule (Sahm 2019, used by Federal Reserve real-time recession indicator):
        when 3mo-MA of unrate exceeds its 12mo minimum by ≥0.5pp, a recession is
        imminent. We use UNRATE_RISE_THRESHOLD (default 0.3pp) for an earlier
        warning. Vote fires when current unrate ≥ 12mo-min + threshold (within
        the trailing-12-month window AS OF algorithm time, not CSV end).

        Independent of GDP/CPI z-scores (the regime classifier's inputs), so
        avoids the redundancy that motivated dropping the SJSU/P(Stagflation)
        vote in v11.2. Late-cycle by construction — leads recessions ~3-6mo
        but trails the very-fastest stress signals (VIX term backwardation).

        PIT discipline: filters to rows with index ≤ self.time before the
        12-month window — otherwise backtest would peek at end-of-CSV values.

        Returns False if UNRATE feed is missing or insufficient history.
        """
        if self.unrate_df is None:
            return False
        # PIT filter: only rows with index timestamp ≤ algorithm's current time.
        valid = self.unrate_df.index <= self.time
        if not valid.any():
            return False
        df_pit = self.unrate_df.loc[valid]
        if len(df_pit) < 252:
            return False
        window = df_pit.iloc[-252:]
        # CSV column is "unrate" (per scripts/export_unrate.py); fall back gracefully.
        col = "unrate" if "unrate" in window.columns else window.columns[0]
        try:
            current = float(window[col].iloc[-1])
            min_12mo = float(window[col].min())
        except (ValueError, TypeError):
            return False
        return current >= min_12mo + self.UNRATE_RISE_THRESHOLD

    def _update_votes(self) -> None:
        if self.is_warming_up:
            return
        # Base vote set per VOTE_MODE.
        # v10 mode: regime + P(Stagflation) + credit + VIX backwardation (LIVE pin behavior)
        # v10.5a mode: regime + T10Y2Y inversion + credit + VIX backwardation
        # (drops the noisy P(Stagflation) GBM forecast for an independent macro lead)
        if self.VOTE_MODE == "v10":
            base = [
                ("regime",   self._vote_regime_stagflation()),
                ("prob",     self._vote_probability_high()),
                ("credit",   self._vote_credit_stress()),
                ("vix_term", self._vote_vix_backwardation()),
            ]
        else:  # v10.5a
            base = [
                ("regime",   self._vote_regime_stagflation()),
                ("t10y2y",   self._vote_yield_curve_inverted()),
                ("credit",   self._vote_credit_stress()),
                ("vix_term", self._vote_vix_backwardation()),
            ]

        # v11.3 candidate ADDITIONAL votes (default OFF). INCLUDE_T10Y2Y_VOTE only
        # adds T10Y2Y in v10 mode (in v10.5a it's already a base replacement vote
        # — toggling here would double-count the same signal).
        additional = []
        if self.INCLUDE_UNRATE_VOTE:
            additional.append(("unrate", self._vote_unemployment_rising()))
        if self.INCLUDE_T10Y2Y_VOTE and self.VOTE_MODE == "v10":
            additional.append(("t10y2y_add", self._vote_yield_curve_inverted()))

        all_votes = base + additional
        raw_count = sum(int(v) for _, v in all_votes)
        # Clamp to len(BLEND_WEIGHTS)-1 to avoid array-index overflow when 5+ votes fire.
        new_count = min(raw_count, len(self.BLEND_WEIGHTS) - 1)

        if new_count != self.vote_count or raw_count != getattr(self, "_last_raw_vote_count", -1):
            labels = " ".join(f"{name}={int(val)}" for name, val in all_votes)
            extra = f" raw={raw_count}→clamped" if raw_count != new_count else ""
            self.log(
                f"{self.time:%Y-%m-%d}: vote_count {self.vote_count} -> {new_count} "
                f"mode={self.VOTE_MODE}{extra} ({labels})"
            )
        self.vote_count = new_count
        self._last_raw_vote_count = raw_count
        # last_votes preserves backward compatibility (4-tuple of base votes only;
        # callers that read this don't see the additional v11.3 votes by name)
        self.last_votes = tuple(v for _, v in base)
        # DD breaker daily check (no-op when DD_CIRCUIT_BREAKER_ENABLED=0)
        self._update_nav_history()
        self._intramonth_dd_rescale()

    # ---------- v11 daily handler + Layer 1 + intra-month edge ----------

    def _daily_check(self) -> None:
        """Daily 09:35 ET schedule entry point. Refresh votes; evaluate v11
        Layer 1 cash hysteresis, v11.1 re-entry trigger, and v11 slow-layer
        intra-month edge trigger. All v11/v11.1 components no-op when their
        parameter flags are 0. Order matters: votes first (so re-entry can
        check vote_count), then Layer 1 (most authoritative), then re-entry
        (overrides defensive blend), then edge trigger (slow-layer drift)."""
        self._update_votes()
        self._update_layer1_cash()
        self._check_reentry()
        self._check_edge_trigger()

    def _update_layer1_cash(self) -> None:
        """v11 Layer 1: VIX-based fast cash trigger with hysteresis + 90d guardrail.

        - Enter cash when spot VIX > LAYER1_VIX_HIGH (default 30).
        - Exit cash when spot VIX < LAYER1_VIX_LOW (default 25).
        - 90-day guardrail forces re-evaluation if hysteresis hasn't resolved.

        Each transition fires self.rebalance() so portfolio updates intra-day
        without waiting for next month_start. No-op when LAYER1_CASH_ENABLED=0.
        """
        if not self.LAYER1_CASH_ENABLED or self.is_warming_up:
            return
        vix = self._current_vix_spot()
        if vix is None:
            return
        if not self.in_layer1_cash and vix > self.LAYER1_VIX_HIGH:
            self.in_layer1_cash = True
            self.layer1_entered_at = self.time
            self.log(
                f"{self.time:%Y-%m-%d}: LAYER1 ENTER cash, "
                f"VIX={vix:.1f} > {self.LAYER1_VIX_HIGH}"
            )
            self.rebalance()
            return
        if self.in_layer1_cash and vix < self.LAYER1_VIX_LOW:
            self.in_layer1_cash = False
            self.layer1_entered_at = None
            self.log(
                f"{self.time:%Y-%m-%d}: LAYER1 EXIT cash, "
                f"VIX={vix:.1f} < {self.LAYER1_VIX_LOW}"
            )
            self.rebalance()
            return
        if self.in_layer1_cash and self.layer1_entered_at is not None:
            days = (self.time - self.layer1_entered_at).days
            if days > self.LAYER1_MAX_DAYS_IN_CASH:
                self.log(
                    f"{self.time:%Y-%m-%d}: LAYER1 GUARDRAIL "
                    f"{days}d > {self.LAYER1_MAX_DAYS_IN_CASH}d in cash "
                    f"(VIX={vix:.1f}); forcing re-eval"
                )
                self.in_layer1_cash = False
                self.layer1_entered_at = None
                self.rebalance()

    def _check_reentry(self) -> None:
        """v11.1 re-entry trigger: VIX < REENTRY_VIX_MAX AND SPY 20d ROC >
        REENTRY_SPY_ROC_MIN. Fires from any defensive state (Layer 1 cash OR
        slow-layer alpha > 0). Forces an immediate rebalance with the alpha-blend
        suppressed for REENTRY_HOLD_DAYS (the cooldown also prevents whipsaw
        re-fires). Resets in_layer1_cash if active. No-op when REENTRY_ENABLED=0.
        """
        if not self.REENTRY_ENABLED or self.is_warming_up:
            return
        in_defensive = self.in_layer1_cash or self.vote_count >= 1
        if not in_defensive:
            return
        if self.last_reentry_at is not None:
            if (self.time - self.last_reentry_at).days < self.REENTRY_HOLD_DAYS:
                return
        vix = self._current_vix_spot()
        if vix is None or vix >= self.REENTRY_VIX_MAX:
            return
        if self.spy_roc is None or not self.spy_roc.is_ready:
            return
        roc = float(self.spy_roc.current.value)
        if roc < self.REENTRY_SPY_ROC_MIN:
            return
        self.log(
            f"{self.time:%Y-%m-%d}: REENTRY triggered "
            f"(VIX={vix:.1f}<{self.REENTRY_VIX_MAX}, "
            f"SPY 20d ROC={roc:.2%}>{self.REENTRY_SPY_ROC_MIN:.2%}, "
            f"vote={self.vote_count}, layer1={self.in_layer1_cash}); "
            f"forcing 100% equity"
        )
        self.in_layer1_cash = False
        self.layer1_entered_at = None
        self.last_reentry_at = self.time
        self.rebalance()

    def _check_edge_trigger(self) -> None:
        """v11 slow-layer intra-month edge: rebalance when vote_count changes.

        cstability's BLEND_WEIGHTS is 5-level granular [0.0, 0.15, 0.40, 0.75, 1.0]
        per vote_count 0..4. Without this trigger, a vote count change between
        monthly rebalances waits 2-4 weeks to take effect — too slow for fast
        crashes (Lehman week, COVID onset). Mirrors cgrowth/main.py:402-411
        edge-detect pattern but fires on any change instead of binary threshold
        crossing. No-op when INTRA_MONTH_EDGE_ENABLED=0.
        """
        if not self.INTRA_MONTH_EDGE_ENABLED or self.is_warming_up:
            return
        if self.vote_count != self.last_vote_count:
            self.log(
                f"{self.time:%Y-%m-%d}: intra-month edge "
                f"vote_count {self.last_vote_count} -> {self.vote_count}; rebalance"
            )
            self.last_vote_count = self.vote_count
            self.rebalance()

    # ---------- DD circuit-breaker (default OFF; opt-in via DD_CIRCUIT_BREAKER_ENABLED=1) ----------

    def _update_nav_history(self):
        """Append current NAV to rolling history; trim to DD_HIGH_WATER_WINDOW."""
        nav = self.portfolio.total_portfolio_value
        if nav <= 0:
            return
        self.nav_history.append(nav)
        if len(self.nav_history) > self.DD_HIGH_WATER_WINDOW:
            self.nav_history = self.nav_history[-self.DD_HIGH_WATER_WINDOW:]

    def _compute_dd_scale(self) -> float:
        """Return current DD-breaker scale factor (1.0 if disabled or DD shallow).

        DD ≤ DD_LEVEL_2_THRESHOLD → DD_LEVEL_2_SCALE (default 0.25, deep DD).
        DD ≤ DD_LEVEL_1_THRESHOLD → DD_LEVEL_1_SCALE (default 0.50, moderate DD).
        Otherwise 1.0 (full exposure).
        """
        if not self.DD_CIRCUIT_BREAKER_ENABLED:
            return 1.0
        if not self.nav_history:
            return 1.0
        peak = max(self.nav_history)
        if peak <= 0:
            return 1.0
        nav = self.portfolio.total_portfolio_value
        dd = (nav - peak) / peak
        if dd <= self.DD_LEVEL_2_THRESHOLD:
            return self.DD_LEVEL_2_SCALE
        if dd <= self.DD_LEVEL_1_THRESHOLD:
            return self.DD_LEVEL_1_SCALE
        return 1.0

    def _intramonth_dd_rescale(self):
        """Daily check: if DD scale has changed since last rebalance, re-apply weights."""
        if not self.DD_CIRCUIT_BREAKER_ENABLED or self.is_warming_up:
            return
        if not self.last_base_weights:
            return
        new_dd_scale = self._compute_dd_scale()
        if abs(new_dd_scale - self.last_dd_scale) < 1e-6:
            return
        self.log(
            f"{self.time:%Y-%m-%d}: DD breaker transition "
            f"{self.last_dd_scale:.2f} → {new_dd_scale:.2f}; rescaling holdings."
        )
        for sym, w in self.last_base_weights.items():
            self.set_holdings(sym, w * new_dd_scale)
        self.last_dd_scale = new_dd_scale

    # ---------- selection + sizing ----------

    def _base_allocation(self) -> dict:
        """Return base weights dict (Symbol -> weight, summing to 1.0).

        Allocation logic:
            1. If regime is Stagflation, base = equal-weight defensive basket
               (matches v8 RegimeRotator behavior).
            2. v11: if USE_SPY_FLOOR_AT_ZERO_VOTES=1 AND 0 votes fire AND not
               in Stagflation, base = equal-weight SPY/QQQ. Removes the top-3
               ETF momentum concentration drag in calm-bull regimes; matches
               buy-and-hold SPY as a hard floor.
            3. Otherwise, base = top-N by 252d momentum (v10 LIVE behavior).
            4. Returns empty dict if none of the above applies.
        """
        regime = self._current_regime()
        if regime == "Stagflation" and self.stag_symbols:
            w = 1.0 / len(self.stag_symbols)
            return {sym: w for sym in self.stag_symbols}

        # v11 floor basket: when 0 votes fire and not in Stagflation, hold a
        # broad-market basket instead of top-N ETF momentum. FLOOR_MODE selects:
        #   "spy_qqq"   → SPY/QQQ (matches SPY closely; SPY-like DD)
        #   "splv_usmv" → SPLV/USMV (low-vol equity; ~80% upside, ~70% DD of SPY)
        #   "spy_splv"  → SPY/SPLV (compromise: more return than splv_usmv, less DD than spy_qqq)
        if self.USE_SPY_FLOOR_AT_ZERO_VOTES and self.vote_count == 0:
            mode = (self.FLOOR_MODE or "spy_qqq").lower()
            if mode == "splv_usmv":
                floor_tickers = ("SPLV", "USMV")
            elif mode == "spy_splv":
                floor_tickers = ("SPY", "SPLV")
            else:
                floor_tickers = ("SPY", "QQQ")
            floor_syms = [
                self.symbols_by_ticker[t]
                for t in floor_tickers
                if t in self.symbols_by_ticker
            ]
            if floor_syms:
                w = 1.0 / len(floor_syms)
                return {sym: w for sym in floor_syms}

        scored = [
            (sym, self.signals[sym]["momp"].current.value)
            for sym in self.signals
            if self.signals[sym]["momp"].is_ready
        ]
        eligible = [(sym, m) for sym, m in scored if m > 0]
        eligible.sort(key=lambda p: p[1], reverse=True)
        chosen = [sym for sym, _ in eligible[: self.TOP_N]]
        if not chosen:
            return {}
        w = 1.0 / len(chosen)
        return {sym: w for sym in chosen}

    def _defensive_basket_weights(self) -> dict:
        # v11 DEFENSIVE_BASKET_MODE: when "bil_cash", entire defensive component
        # is BIL whenever active (vote_count >= 1). Eliminates 2022-style basket
        # crashes (TLT -29%); short-duration treasuries earn ~5% with negligible
        # duration risk. v10 LIVE preserved by the default "gld_tip_tlt" mode.
        if self.DEFENSIVE_BASKET_MODE == "bil_cash":
            cash_sym = self.symbols_by_ticker.get(self.CASH_TICKER)
            if cash_sym is not None:
                return {cash_sym: 1.0}
        # v10.5 USE_BIL_CASH_AT_MAX_STRESS path: BIL only at vote_count == 4
        # (kept as backstop; orthogonal to DEFENSIVE_BASKET_MODE).
        if self.USE_BIL_CASH_AT_MAX_STRESS and self.vote_count == 4:
            cash_sym = self.symbols_by_ticker.get(self.CASH_TICKER)
            if cash_sym is not None:
                return {cash_sym: 1.0}
        if not self.stag_symbols:
            return {}
        w = 1.0 / len(self.stag_symbols)
        return {sym: w for sym in self.stag_symbols}

    def _blend_weights(self, base: dict, alpha: float) -> dict:
        """Blend base allocation with defensive basket: (1-alpha)*base + alpha*defensive."""
        defensive = self._defensive_basket_weights()
        out: dict = {}
        for sym, w in base.items():
            out[sym] = (1.0 - alpha) * w
        for sym, w in defensive.items():
            out[sym] = out.get(sym, 0.0) + alpha * w
        # Drop near-zero weights to avoid 0% set_holdings churn
        return {sym: w for sym, w in out.items() if abs(w) > 1e-4}

    def rebalance(self) -> None:
        if self.is_warming_up:
            return

        # v11 Layer 1 override: when active, short-circuit the slow-layer blend
        # entirely and route 100% to BIL. Triggered by spot VIX > LAYER1_VIX_HIGH
        # via _update_layer1_cash() and persists until VIX falls below
        # LAYER1_VIX_LOW (or 90-day guardrail expires).
        if self.LAYER1_CASH_ENABLED and self.in_layer1_cash:
            cash_sym = self.symbols_by_ticker.get(self.CASH_TICKER)
            if cash_sym is not None:
                for sym in self.signals:
                    if sym != cash_sym and self.portfolio[sym].invested:
                        self.liquidate(sym)
                self.set_holdings(cash_sym, 1.0)
                vix = self._current_vix_spot()
                vix_str = f"{vix:.1f}" if vix is not None else "n/a"
                days_in = (self.time - self.layer1_entered_at).days if self.layer1_entered_at else 0
                self.log(
                    f"{self.time:%Y-%m-%d}: LAYER1_CASH active "
                    f"(VIX={vix_str}, days_in_cash={days_in}); 100% BIL"
                )
                self.last_base_weights = {cash_sym: 1.0}
                self.last_dd_scale = 1.0
                return

        # Refresh votes one more time at rebalance for logging accuracy
        self._update_votes()

        base = self._base_allocation()

        # v11.1 re-entry window: skip the defensive alpha-blend for
        # REENTRY_HOLD_DAYS after a re-entry trigger fires. Forces 100%
        # base allocation so the override survives until conditions
        # naturally normalize (vote_count drops on its own) or another
        # trigger (Layer 1 cash, edge trigger) overrides.
        in_reentry_window = False
        if (
            self.REENTRY_ENABLED
            and self.last_reentry_at is not None
            and (self.time - self.last_reentry_at).days < self.REENTRY_HOLD_DAYS
        ):
            in_reentry_window = True

        alpha = 0.0 if in_reentry_window else self.BLEND_WEIGHTS[self.vote_count]
        weights = self._blend_weights(base, alpha)

        # Liquidate anything not in final weights
        keep_set = set(weights.keys())
        for sym in self.signals:
            if sym not in keep_set and self.portfolio[sym].invested:
                self.liquidate(sym)

        if not weights:
            self.log(
                f"{self.time:%Y-%m-%d}: regime={self._current_regime() or 'unknown'} "
                f"votes={self.vote_count}/4 — no eligible momentum and not in Stagflation; "
                f"staying flat (cash)."
            )
            return

        # Apply DD circuit-breaker scaling on top of vote-based blend (no-op when disabled).
        dd_scale = self._compute_dd_scale()
        # Save base weights so the daily DD check can intra-month-rescale if the
        # breaker level transitions. Stored pre-DD-scale so re-applying is idempotent.
        self.last_base_weights = dict(weights)
        self.last_dd_scale = dd_scale

        for sym, w in weights.items():
            self.set_holdings(sym, w * dd_scale)

        gross = sum(weights.values()) * dd_scale
        chosen_tickers = [self.symbol_to_ticker.get(s, str(s)) for s in weights.keys()]
        regime = self._current_regime() or "unknown"
        # Vote-2 label depends on mode: "prob" for v10, "t10y2y" for v10.5a
        label2 = "prob" if self.VOTE_MODE == "v10" else "t10y2y"
        dd_str = f" dd_scale={dd_scale:.2f}" if self.DD_CIRCUIT_BREAKER_ENABLED else ""
        self.log(
            f"{self.time:%Y-%m-%d}: mode={self.VOTE_MODE} regime={regime} votes={self.vote_count}/4 "
            f"(reg={int(self.last_votes[0])} {label2}={int(self.last_votes[1])} "
            f"credit={int(self.last_votes[2])} vix={int(self.last_votes[3])}){dd_str} "
            f"alpha={alpha:.2f} gross={gross:.2%} holdings={chosen_tickers}"
        )

    def on_data(self, data: Slice) -> None:
        # Decisions happen in scheduled monthly rebalance + daily vote refresh.
        pass
