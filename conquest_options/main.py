# region imports
from AlgorithmImports import *
from collections import defaultdict, deque
from datetime import date
# endregion

# Local package imports — `conquest_options/conquest/` is sync'd from workspace
# `conquest/` by scripts/sync_to_lean.py prior to `lean cloud push`.
from strategies import ENABLED_STRATEGIES
from strategies.base import StrategyContext
from option_selector import pick_contract
from position_sizer import SizerConfig, size_position
from exit_manager import ExitManager
from metrics_per_trade import TradeLogger
from edge_signals.iv_rank import IVRankTracker
from edge_signals.put_call_ratio_lean import PutCallRatioTracker
from edge_signals.earnings_lookup import EarningsCalendar
from edge_signals.uoa_lean import UOATracker
from edge_signals.short_pressure import read_short_metrics
from edge_signals.gex import compute_gex_contributions, classify_gex_regime
from edge_signals.term_structure import (
    compute_term_ratio, classify_term_regime, is_acute_stress,
)
from edge_signals.skew import compute_skew, SkewTracker
from edge_signals.insider_lookup import InsiderForm4Calendar
from edge_signals.gdelt_sentiment import GDELTSentimentLoader
from edge_signals.finra_short_interest import FinraShortInterestLoader
from edge_signals.historical_vol import HVTracker
from edge_signals.crisis_detector import CrisisDetector
from edge_signals.pit_universe import PitUniverse
# Dynamic PIT-momentum universe ops live in dyn_universe.py (kept out of this
# file to stay under QC's 64k-char/file push limit).
from dyn_universe import setup_dynamic_universe
from dyn_universe import on_securities_changed as _dyn_on_securities_changed


class ConquestOptions(QCAlgorithm):
    """Long-only options framework. Phase 1 ships zero registered
    strategies; the framework is exercised by `smoke_a2` when run with
    config parameter ENABLE_SMOKE=true."""

    def initialize(self):
        # ---- date + cash ----
        start_year = int(self.get_parameter("BACKTEST_START_YEAR") or 2008)
        start_month = int(self.get_parameter("BACKTEST_START_MONTH") or 1)
        start_day = int(self.get_parameter("BACKTEST_START_DAY") or 1)
        end_year = int(self.get_parameter("BACKTEST_END_YEAR") or 2026)
        end_month = int(self.get_parameter("BACKTEST_END_MONTH") or 2)
        end_day = int(self.get_parameter("BACKTEST_END_DAY") or 4)
        seed_cash = int(self.get_parameter("SEED_CASH") or 10_000)
        # v22: month/day params allow focused-window BTs (e.g. COVID Q1-Q2 2020).
        # Defaults preserve the old behaviour (Jan 1 → Feb 4 next year).
        self.set_start_date(start_year, start_month, start_day)
        self.set_end_date(end_year, end_month, end_day)
        self.set_cash(seed_cash)

        # ---- dynamic PIT-momentum path (survivorship-bias #1 kill; default OFF) ----
        # DYNAMIC_PIT_MOMENTUM=1 replaces the fixed WSB hand-pick with a monthly
        # top-N S&P-500-by-180d-momentum selection gated to point-in-time index
        # members. Default OFF → existing BTs byte-for-byte identical (see dyn_universe.py).
        self._dynamic_pit = (self.get_parameter("DYNAMIC_PIT_MOMENTUM") or "0").strip().lower() in ("1", "true", "yes")
        self._dyn_top_n = int(self.get_parameter("DYN_TOP_N") or 10)
        self._dyn_mom_lookback = int(self.get_parameter("DYN_MOM_LOOKBACK") or 180)
        # Escalation lever: union equities + MOMP always DAILY; the ≤N active option
        # chains (+ their underlyings) escalate daily→hour→minute via DYN_RESOLUTION
        # if daily chains deliver too few tradeable contracts.
        _res_param = (self.get_parameter("DYN_RESOLUTION") or "daily").strip().lower()
        self._dyn_option_res = {
            "minute": Resolution.MINUTE, "min": Resolution.MINUTE,
            "hour": Resolution.HOUR, "hourly": Resolution.HOUR,
            "daily": Resolution.DAILY, "day": Resolution.DAILY,
        }.get(_res_param, Resolution.DAILY)
        # Dynamic-path state — ALWAYS defined so the static path never AttributeErrors.
        self._dyn_pit: PitUniverse | None = None
        self._dyn_equity_sym: dict[str, Symbol] = {}     # union ticker → equity Symbol
        self._dyn_momp: dict[Symbol, object] = {}        # equity Symbol → MOMP indicator
        self._dyn_active: set[str] = set()               # entry-eligible (this month's top-N)
        self._dyn_draining: set[str] = set()             # held-but-left; chain kept, NOT eligible
        self._dyn_pending_entry: dict[str, object] = {}  # ticker → date its chain was added (no-trade bar)
        self._last_entry_by_ticker: dict[str, object] = {}  # ticker → date of last ENTRY (dynamic-path cooldown)

        # ---- benchmark + warmup ----
        self.set_benchmark("SPY")
        # v22: 365 cal days (~252 trading bars) at DAILY, sampled during warm-up,
        # so 252d-high / HV60 / momentum_60d are non-degenerate from BT day 0.
        if self._dynamic_pit:
            # Need DYN_MOM_LOOKBACK daily bars to ready every MOMP, plus a margin
            # so the first month_start rebalance lands AFTER warm-up. Bar-count form.
            self.set_warm_up(self._dyn_mom_lookback + 40, Resolution.DAILY)
        else:
            self.set_warm_up(timedelta(days=365), Resolution.DAILY)

        # ---- strategies ----
        # ENABLED_STRATEGIES is the *registry* (all built strategies).
        # ACTIVE_STRATEGY_IDS is the *per-run activation list* (comma-separated
        # IDs). Default empty → no strategies fire (safe boot). This protects
        # cloud backtests from accidentally enabling every strategy at once.
        active_ids_param = (self.get_parameter("ACTIVE_STRATEGY_IDS") or "").strip()
        active_ids = {s.strip() for s in active_ids_param.split(",") if s.strip()}
        self._strategies = [s for s in ENABLED_STRATEGIES
                            if not active_ids or s.id in active_ids]
        if (self.get_parameter("ENABLE_SMOKE") or "false").lower() == "true":
            from strategies.smoke_a2 import SmokeA2
            self._strategies.append(SmokeA2())

        # Dynamic-PIT path runs exactly ONE strategy (replaces the static list) so
        # the honest BT isn't polluted by hand-picked-universe strategies.
        if self._dynamic_pit:
            from strategies.dynamic_pit_momentum_calls import DynamicPitMomentumCalls
            self._strategies = [DynamicPitMomentumCalls()]

        if not self._strategies:
            self.debug("ENABLED_STRATEGIES empty (or ACTIVE_STRATEGY_IDS filtered all out); "
                       "ENABLE_SMOKE=false. Framework boots but emits no trades.")

        # ---- universe (static path) ----
        # v10: QC hosts options at MINUTE only; daily chains were sparse (root
        # cause of prior 0-trade BTs). Greeks need a VolatilityModel on the
        # underlying or delta-filter / IV-rank / GEX all fail.
        self._option_symbols: dict[str, Symbol] = {}
        if self._dynamic_pit:
            # Dynamic path subscribes the full S&P union + rotates option chains
            # monthly (see dyn_universe.py). The static loop below is then a no-op
            # (empty underlyings) so it stays byte-for-byte unchanged.
            setup_dynamic_universe(self)
            underlyings = []
        else:
            underlyings = sorted({u for s in self._strategies for u in s.universe})
        for ticker in underlyings:
            equity = self.add_equity(ticker, Resolution.MINUTE)
            # v10: set VolatilityModel so Greeks populate on the chain
            self.securities[equity.symbol].volatility_model = \
                StandardDeviationOfReturnsVolatilityModel(30)
            option = self.add_option(ticker, Resolution.MINUTE)
            # v10d: strikes(-N,N) = strike UNITS not %; ±30 gives the picker real
            # OTM reach. v22: expiration(14,400) (was (14,60)) so LEAPS-style
            # contracts (Tepper 180d, CrisisRebound 90/180, D1 365) have candidates.
            option.set_filter(lambda u: u.strikes(-30, 30).expiration(14, 400))
            # v10: set PriceModel so theoretical price + IV calc work.
            # BlackScholes is the canonical default; if Greeks still come back
            # as 0/None we'll switch to BinomialTian per QC forum workaround.
            option.price_model = OptionPriceModels.black_scholes()
            self._option_symbols[ticker] = option.symbol

        # VIX + VIX3M + VIX9D — term-structure regime context.
        def _add_vix_like(symbol_str: str) -> Symbol | None:
            try:
                return self.add_index(symbol_str, Resolution.DAILY).symbol
            except Exception:
                try:
                    return self.add_equity(symbol_str, Resolution.DAILY).symbol
                except Exception:
                    return None
        self._vix_symbol = _add_vix_like("VIX")
        self._vix3m_symbol = _add_vix_like("VIX3M")
        self._vix9d_symbol = _add_vix_like("VIX9D")

        # v6: 260-day price history (covers 252d high for Tepper V-bottom)
        self._price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=260))
        # Tier1 Signal 2: parallel $-volume deque, populated in the same
        # per-day sampler as _price_history. Stores price * sec.volume.
        # At MINUTE resolution sec.volume is the last 1-min bar's volume —
        # the cross-day RATIO at the same 15:00 daily sample time is a
        # useful spike proxy even if absolute values are noisy.
        self._volume_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=260))
        # v22 fix: at MINUTE resolution on_data fires ~390×/day. Without a
        # per-day guard, the 260-element deque holds <1 trading day of data,
        # breaking 5MA/20MA, 252d-high drawdown, momentum_30d/60d, HV30/60
        # for D2 Tepper, CrisisDetector, A_GEX gates, D1 LEAPS, etc.
        # Track the last calendar date sampled per ticker and append at most
        # one price per day.
        self._last_price_sample_date: dict[str, object] = {}

        # ---- supporting components ----
        self._sizer_config = SizerConfig(mode="flat", base_pct_nav=0.015,
                                          portfolio_cap_pct_nav=0.10,
                                          min_premium_dollars=50.0)
        self._exit_mgr = ExitManager()
        self._trade_log = TradeLogger()
        self._iv_rank = IVRankTracker(lookback_days=252)
        self._pc_ratio = PutCallRatioTracker(ema_span_days=10)
        # 2026-05-27: at daily res, contract volume rarely hits the old 5x/3x
        # (intraday-designed) multipliers — probe BT showed UOA=0%. Loosened to
        # 2.5x/1.5x to catch real daily-volume spikes. Tunable per BT.
        self._uoa = UOATracker(
            vol_multiplier=2.5, oi_multiplier=1.5,
        )
        self._skew = SkewTracker(lookback_days=252)
        self._gex_total: float | None = None
        self._gex_regime: str | None = None
        self._hv = HVTracker()

        # Data-availability probe (2026-05-27). See edge_signals/data_probe.py.
        from edge_signals.data_probe import DataProbe
        self._probe = DataProbe()
        try:
            self._data_probe_only = (self.get_parameter("DATA_PROBE_ONLY") == "1")
        except Exception:
            self._data_probe_only = False
        # v6 Phase 6: crisis state machine for aggressive V-bottom deploys
        self._crisis = CrisisDetector()

        # Earnings calendar from Object Store (best-effort; Phase 0 fetcher
        # is the source). Empty calendar if key missing.
        self._earnings = EarningsCalendar()
        try:
            if self.object_store.contains_key(EarningsCalendar.OBJECT_STORE_KEY):
                txt = self.object_store.read(EarningsCalendar.OBJECT_STORE_KEY)
                self._earnings = EarningsCalendar.from_csv_text(txt)
                self.debug(f"Earnings calendar: loaded "
                           f"({sum(1 for _ in self._earnings._by_ticker)} tickers)")
        except Exception as e:
            self.debug(f"Earnings calendar load skipped: {e}")

        # Insider Form 4 opportunistic buys (Object Store, Phase 0 source)
        self._insider = InsiderForm4Calendar()
        try:
            if self.object_store.contains_key(InsiderForm4Calendar.OBJECT_STORE_KEY):
                txt = self.object_store.read(InsiderForm4Calendar.OBJECT_STORE_KEY)
                self._insider = InsiderForm4Calendar.from_csv_text(txt)
                self.debug(f"Insider Form 4: loaded "
                           f"({sum(1 for _ in self._insider._by_ticker)} tickers)")
        except Exception as e:
            self.debug(f"Insider Form 4 load skipped: {e}")

        # Tier1 Signal 1 — GDELT news sentiment + volume (Object Store).
        # Populated by scripts/ingest_gdelt_sentiment.py and pushed via
        # `lean object-store set --key conquest/sentiment/gdelt_daily.csv ...`
        self._gdelt = GDELTSentimentLoader()
        try:
            if self.object_store.contains_key(GDELTSentimentLoader.OBJECT_STORE_KEY):
                txt = self.object_store.read(GDELTSentimentLoader.OBJECT_STORE_KEY)
                self._gdelt = GDELTSentimentLoader.from_csv_text(txt)
                self.debug(f"GDELT sentiment: loaded "
                           f"({sum(1 for _ in self._gdelt._by_ticker)} tickers)")
        except Exception as e:
            self.debug(f"GDELT sentiment load skipped: {e}")

        # FINRA short interest biweekly (LEADING signal for v_REFLEX_v2,
        # v_SHORT_SQUEEZE_PURE, v_TRIPLE_CONFLUENCE). Loader returns None
        # if key not yet bootstrapped — strategies handle missing data.
        self._finra: FinraShortInterestLoader | None = FinraShortInterestLoader()
        try:
            if self.object_store.contains_key(FinraShortInterestLoader.OBJECT_STORE_KEY):
                txt = self.object_store.read(FinraShortInterestLoader.OBJECT_STORE_KEY)
                self._finra = FinraShortInterestLoader.from_csv_text(txt)
                self.debug(f"FINRA SI: loaded "
                           f"({sum(1 for _ in self._finra._by_ticker)} tickers)")
        except Exception as e:
            self.debug(f"FINRA SI load skipped: {e}")

        # Pending exits — populated each on_data, consumed in on_order_event.
        # Maps order_ticket_id → (exit_reason, contract_symbol, position).
        self._pending_exits: dict[int, dict] = {}
        # Pending entries — order_ticket_id → signal dict, used to populate
        # TradeLogger on fill.
        self._pending_entries: dict[int, dict] = {}

        # cstability vote_count — read from Object Store; refreshed daily.
        self._cstability_vote_count: int | None = None
        self._cstability_vote_csv: dict[str, int] = {}
        try:
            key = "conquest/votes/cstability_4vote_daily.csv"
            if self.object_store.contains_key(key):
                txt = self.object_store.read(key)
                # Schema: date,vote_count[,...other cols]
                for i, line in enumerate(txt.splitlines()):
                    if i == 0 or not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            self._cstability_vote_csv[parts[0].strip()] = int(float(parts[1]))
                        except Exception:
                            continue
                self.debug(f"cstability vote_count loaded: {len(self._cstability_vote_csv)} rows")
        except Exception as e:
            self.debug(f"cstability vote_count load skipped: {e}")

        # ---- PIT universe gate on the STATIC universe (default OFF) ----------
        # Bias #1: trade a static name on date T only if it was an index member
        # as of T. Reads sp500_pit_monthly.csv; fails CLOSED (blocks all entries)
        # if enabled but absent. (Distinct from DYNAMIC_PIT_MOMENTUM, which also
        # picks the universe; this only gates the hand-picked list.)
        self._pit_gate = (self.get_parameter("PIT_UNIVERSE_GATE") or "0").strip().lower() in ("1", "true", "yes")
        self._pit: PitUniverse | None = None
        if self._pit_gate:
            try:
                key = PitUniverse.OBJECT_STORE_KEY
                if self.object_store.contains_key(key):
                    self._pit = PitUniverse.from_csv_text(self.object_store.read(key))
                    self.debug(f"PIT universe gate ON: {len(self._pit)} monthly snapshots ({key})")
                else:
                    self.error(f"PIT_UNIVERSE_GATE=1 but {key} missing from Object Store; "
                               f"failing CLOSED — all entries blocked until the CSV is pushed.")
            except Exception as e:
                self.error(f"PIT universe load failed (failing CLOSED): {e}")

        # UOA-active tickers for the current day (computed per OnData)
        self._uoa_tickers_today: set[str] = set()

        # v5: starting NAV for drawdown-aware sizing
        self._starting_nav: float = float(seed_cash)

        # v7: DIAGNOSTIC — count signals emitted per strategy (regardless of
        # whether they translate to actual orders). Lets us distinguish
        # "gates never passed" from "gates pass but picker rejects contract".
        self._signals_emitted: dict[str, int] = defaultdict(int)
        self._signals_entered: dict[str, int] = defaultdict(int)
        # v8: finer-grained _try_enter failure counters
        self._fail_chain: dict[str, int] = defaultdict(int)
        self._fail_picker: dict[str, int] = defaultdict(int)
        self._fail_sizer: dict[str, int] = defaultdict(int)
        self._fail_dup: dict[str, int] = defaultdict(int)
        self._fail_cap: dict[str, int] = defaultdict(int)
        # v23: per-strategy exit-reason counters (TP/SL/time/expiry/manual).
        self._exit_reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # v26: which price-discovery path succeeded for deep-OTM held contracts.
        self._exit_feed_diag: dict[str, int] = defaultdict(int)

        # v22: per-day marker gating the once-per-day samplers. At MINUTE res
        # on_data fires ~390×/day; without this a 252-elem deque fills in <1 day
        # and "252-day percentile" reads intraday noise. Gate = first tick of a
        # day with local hour >= 15 (chain-volume aggregates mostly settled).
        self._last_daily_pass: object = None

    # Dynamic PIT-momentum universe (DYNAMIC_PIT_MOMENTUM=1) lives in
    # dyn_universe.py; Lean calls this override automatically (no-op when static).
    def on_securities_changed(self, changes) -> None:
        _dyn_on_securities_changed(self, changes)

    # ------------------------------------------------------------------
    #  Per-tick orchestration
    # ------------------------------------------------------------------

    def on_data(self, data: Slice) -> None:
        today = self.time.date()

        # v22: warm-up runs at DAILY res; populate price_history each tick so
        # 5MA/20MA/HV/252d-high are non-empty post-warmup (else multi-day gates
        # can't fire for 20-252 days into the BT).
        if self.is_warming_up:
            if self._last_daily_pass != today:
                self._last_daily_pass = today
                self._update_price_history()
                if "SPY" in self._option_symbols:
                    hist_len = len(self._price_history.get("SPY", []))
                    if hist_len % 30 == 0 and hist_len > 0:
                        self.debug(f"[WARMUP] {today} SPY price_history len={hist_len}")
            return  # skip strategy/exit logic during warm-up

        # 1a. PER-TICK state computations (instantaneous "right now" reads).
        # GEX uses the latest chain snapshot; UOA-active flags compare
        # current chain volume vs. (now-correctly-daily) UOA history.
        self._compute_gex(data)                # market regime
        self._detect_uoa_active_tickers(data)  # per-ticker UOA flag for ctx

        # 1b. ONCE-PER-DAY samplers (history-mutating). First tick of the day at
        # hour >= 15 so chain-volume aggregates are mostly settled.
        if self._last_daily_pass != today and self.time.hour >= 15:
            self._last_daily_pass = today
            self._update_price_history()
            self._record_uoa_history(data)
            self._update_pc_ratio(data)
            self._sample_atm_iv(data)              # feeds IVRankTracker
            self._compute_skew(data)               # per-ticker skew + z-score
            # v22 DIAG: daily per-ticker signal snapshot (diagnose 0-trade BTs).
            vix_v = (float(self.securities.get(self._vix_symbol).price)
                     if self._vix_symbol and self.securities.get(self._vix_symbol)
                        and self.securities.get(self._vix_symbol).has_data else None)
            for ticker in sorted(self._option_symbols.keys()):
                hist = list(self._price_history.get(ticker, []))
                sec = self.securities.get(self.symbol(ticker))
                if sec is None or not sec.has_data:
                    self.debug(f"[DIAG] {today} {ticker} NO_SEC_DATA")
                    continue
                spot = float(sec.price)
                fm = sum(hist[-5:]) / 5 if len(hist) >= 5 else None
                tm = sum(hist[-20:]) / 20 if len(hist) >= 20 else None
                look = hist[-252:] if len(hist) >= 252 else hist
                high = max(look) if look else None
                dd = max(0.0, (high - spot) / high) if high and high > 0 else None
                mom30 = (spot / hist[-31]) if len(hist) >= 31 and hist[-31] > 0 else None
                # IVRank percentile needs iv_today arg — show sample count
                # instead (proves whether _sample_atm_iv populates this ticker)
                iv_samples = len(self._iv_rank._history.get(ticker, []))
                self.debug(
                    f"[DIAG] {today} {ticker} spot={spot:.2f} hist={len(hist)} "
                    f"5MA={fm if fm is None else f'{fm:.2f}'} "
                    f"20MA={tm if tm is None else f'{tm:.2f}'} "
                    f"dd={dd if dd is None else f'{dd:.3f}'} "
                    f"mom30={mom30 if mom30 is None else f'{mom30:.3f}'} "
                    f"iv_samples={iv_samples} "
                    f"vix={vix_v if vix_v is None else f'{vix_v:.1f}'} "
                    f"crisis={self._crisis.state}"
                )

        # Monthly: evict stale option contracts from UOA history to bound memory.
        if self.time.day == 1:
            active_contract_syms: set[str] = set()
            for opt_sym in self._option_symbols.values():
                chain = data.option_chains.get(opt_sym)
                if chain is None:
                    continue
                for c in chain:
                    active_contract_syms.add(str(c.symbol))
            if active_contract_syms:
                evicted = self._uoa.prune(active_contract_syms)
                if evicted > 0:
                    self.debug(f"UOA prune: evicted {evicted} stale contracts")

        # 2. Build context for strategies
        ctx = self._build_context(data)
        self._probe.record(ctx, n_underlyings=len(self._option_symbols))

        # 3. Run exit manager FIRST — frees capital before new entries
        self._process_exits(data)

        # 4. Dispatch strategies → emit signals → enter positions.
        # Skip entirely in data-probe mode to keep the BT cheap.
        if self._data_probe_only:
            return

        # v5: count signals per strategy_id in this tick for multi-leg sizing
        from collections import Counter as _C
        for strat in self._strategies:
            if not getattr(strat, "enabled", True):
                continue
            try:
                signals = strat.on_data(ctx)
            except Exception as e:
                self.debug(f"strategy {strat.id} raised: {e}")
                continue
            # Count signals per strategy_id (handles straddle/strangle's 2-leg case)
            leg_counts = _C(s.strategy_id for s in signals)
            for sig in signals:
                self._signals_emitted[sig.strategy_id] += 1   # v7 diag
                self._try_enter(sig, data, n_legs=leg_counts[sig.strategy_id])

    def on_order_event(self, order_event):
        if order_event.status != OrderStatus.FILLED:
            return
        oid = order_event.order_id
        # Entry?
        if oid in self._pending_entries:
            info = self._pending_entries.pop(oid)
            self._trade_log.log_entry(
                strategy_id=info["strategy_id"],
                underlying=info["underlying"],
                contract_symbol=info["contract_symbol"],
                side=info["side"],
                edge_score=info["edge_score"],
                entry_time=self.time,
                entry_premium_per_share=float(order_event.fill_price),
                contracts=int(order_event.fill_quantity),
                entry_underlying_price=info["underlying_price_at_entry"],
                strike=info["strike"],
                expiry=info["expiry"],
            )
            self._exit_mgr.register(
                info["contract_symbol"],
                info["signal"],
                entry_time=self.time,
                expiry=info["expiry"],
                entry_premium_per_share=float(order_event.fill_price),
                contracts=int(order_event.fill_quantity),
                symbol=info.get("contract_symbol_obj"),   # v25: pass Symbol for securities[] lookup
            )
            return
        # Exit? — Two cases:
        #   (a) order_id matches our pending_exits  → our limit/market exit filled
        #   (b) order_id does NOT match but the contract IS tracked → Lean
        #       initiated the exit (auto-assignment at expiry, margin call).
        #       This was the silent gap in v4/Phase 3: 34 of 51 trades never
        #       routed through log_exit. v5 catches them.
        sym_str_evt = str(order_event.symbol)
        if oid in self._pending_exits:
            info = self._pending_exits.pop(oid)
            commissions = abs(float(order_event.order_fee.value.amount or 0))
            self._trade_log.log_exit(
                contract_symbol=info["contract_symbol"],
                exit_time=self.time,
                exit_premium_per_share=float(order_event.fill_price),
                exit_underlying_price=info["underlying_price_at_exit"],
                exit_reason=info["reason"],
                commissions_dollars=commissions,
            )
            # v23: bump per-strategy exit-reason counter for diagnostics
            tracked = self._exit_mgr.get(info["contract_symbol"])
            if tracked is not None:
                self._exit_reasons[tracked.strategy_id][info["reason"]] += 1
            self._exit_mgr.unregister(info["contract_symbol"])
            if hasattr(self, "_exit_attempt_date"):
                self._exit_attempt_date.pop(info["contract_symbol"], None)
        elif self._exit_mgr.is_tracked(sym_str_evt):
            # v5: Lean-initiated exit (assignment / margin call / liquidation)
            commissions = abs(float(order_event.order_fee.value.amount or 0))
            self._trade_log.log_exit(
                contract_symbol=sym_str_evt,
                exit_time=self.time,
                exit_premium_per_share=float(order_event.fill_price),
                exit_underlying_price=0.0,
                exit_reason="manual",   # Lean's intervention is closest to manual
                commissions_dollars=commissions,
            )
            # v23: bump per-strategy exit-reason counter (manual = Lean-initiated)
            tracked = self._exit_mgr.get(sym_str_evt)
            if tracked is not None:
                self._exit_reasons[tracked.strategy_id]["manual"] += 1
            self._exit_mgr.unregister(sym_str_evt)
            if hasattr(self, "_exit_attempt_date"):
                self._exit_attempt_date.pop(sym_str_evt, None)

    def on_end_of_algorithm(self) -> None:
        # Force-close any open positions to canonicalize the journal
        for pos in self._exit_mgr.positions():
            self._exit_mgr.force_close(pos.symbol_str, "manual")
        # NOTE: we don't synthesize fake exits — the algorithm exits with
        # whatever's still open; the journal includes only closed trades.
        journal_json = self._trade_log.to_json()
        self.object_store.save("conquest/options/trade_journal.json", journal_json)
        self.debug(f"trade_journal: {len(self._trade_log.closed_records())} closed trades persisted")
        if self._trade_log.open_count():
            self.debug(f"WARN: {self._trade_log.open_count()} positions still open at end of algorithm")

        # v3: emit per-strategy stats as runtime statistics so they show up
        # in the API's runtimeStatistics dict (no Object Store export needed —
        # QC blocks that below Institutional). Format: A1_wsb_n / _wr / _e / _pf.
        per_strat: dict[str, list] = defaultdict(list)
        for tr in self._trade_log.closed_records():
            if tr.pnl_pct is None:
                continue
            per_strat[tr.strategy_id].append(tr.pnl_pct)
        for sid, pnls in per_strat.items():
            n = len(pnls)
            if n == 0:
                continue
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            wr = len(wins) / n * 100
            exp_pct = sum(pnls) / n * 100
            pf = (sum(wins) / abs(sum(losses))) if sum(losses) < 0 else 99.0
            short = sid.replace("_call", "").replace("_otm", "")[:14]
            self.set_runtime_statistic(f"{short}_n", n)
            self.set_runtime_statistic(f"{short}_wr_pct", f"{wr:.1f}")
            self.set_runtime_statistic(f"{short}_exp_pct", f"{exp_pct:+.1f}")
            self.set_runtime_statistic(f"{short}_pf", f"{pf:.2f}")
            self.log(f"PER_STRATEGY {sid}: n={n} wr={wr:.1f}% exp={exp_pct:+.2f}% pf={pf:.2f}")

        # v7+v8: emit fine-grained diagnostics — distinguishes between gate
        # failure, chain absence, picker rejection, sizer rejection, and caps.
        all_sids = (set(self._signals_emitted) | set(self._signals_entered) |
                    set(self._fail_chain) | set(self._fail_picker) |
                    set(self._fail_sizer) | set(self._fail_dup) | set(self._fail_cap))
        for sid in all_sids:
            short = sid.replace("_call", "").replace("_otm", "")[:12]
            self.set_runtime_statistic(f"{short}_emit", self._signals_emitted.get(sid, 0))
            self.set_runtime_statistic(f"{short}_enter", self._signals_entered.get(sid, 0))
            self.set_runtime_statistic(f"{short}_fail_chain", self._fail_chain.get(sid, 0))
            self.set_runtime_statistic(f"{short}_fail_pick", self._fail_picker.get(sid, 0))
            self.set_runtime_statistic(f"{short}_fail_size", self._fail_sizer.get(sid, 0))
            self.set_runtime_statistic(f"{short}_fail_cap", self._fail_cap.get(sid, 0))

        # v23 (2026-05-25): per-strategy exit-reason counters. Tells us which
        # exit fired on each closed trade — losses from SL hits vs time decay
        # vs end-of-BT mark-to-market all look the same in WR/E aggregate.
        # Keys: take_profit, stop_loss, time_stop, expiry, regime_exit,
        # signal_exit, manual. Manual = Lean-initiated (assignment / EOB MTM).
        for sid, reasons_dict in self._exit_reasons.items():
            short = sid.replace("_call", "").replace("_otm", "")[:10]
            for reason in ("take_profit", "stop_loss", "time_stop",
                           "expiry", "manual", "regime_exit", "signal_exit"):
                count = reasons_dict.get(reason, 0)
                if count > 0:
                    self.set_runtime_statistic(f"{short}_X_{reason[:4]}", count)

        # v26: cumulative exit-feed price-discovery diagnostics (which lookup
        # path populated current_prices for tracked positions).
        for k, v in self._exit_feed_diag.items():
            if v > 0:
                self.set_runtime_statistic(f"exit_feed_{k}", v)

        # v27 (2026-05-27): per-strategy internal gate diagnostics. Strategies
        # that expose a `_diag` dict get their gate-rejection counters surfaced
        # as runtime stats. Tells us WHICH gate is the bottleneck on 0-fire BTs
        # (e.g., dealer_opex_squeeze: gex_block vs uoa_miss vs cooldown).
        for strat in self._strategies:
            diag = getattr(strat, "_diag", None)
            if not diag:
                continue
            short = strat.id.replace("_call", "").replace("_otm", "")[:10]
            for gate, n in diag.items():
                if n > 0:
                    self.set_runtime_statistic(f"{short}_G_{gate[:7]}", n)

        # v28: data-availability probe (see edge_signals/data_probe.py).
        for k, v in self._probe.summary().items():
            self.set_runtime_statistic(k, v)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, data: Slice) -> StrategyContext:
        underlying_prices: dict[str, float] = {}
        momentum_30d: dict[str, float] = {}
        momentum_60d: dict[str, float] = {}
        five_above_twenty: dict[str, bool] = {}
        dd_from_252_high: dict[str, float] = {}
        # v9: HV per ticker
        hv30: dict[str, float] = {}
        hv60: dict[str, float] = {}
        for ticker, opt_sym in self._option_symbols.items():
            sec = self.securities.get(self.symbol(ticker))
            if sec is not None and sec.has_data:
                cur = float(sec.price)
                underlying_prices[ticker] = cur
                hist = self._price_history.get(ticker)
                if hist and len(hist) >= 31 and hist[-31] > 0:
                    momentum_30d[ticker] = cur / hist[-31]
                if hist and len(hist) >= 61 and hist[-61] > 0:
                    momentum_60d[ticker] = cur / hist[-61]
                # v6: 5MA vs 20MA cross
                if hist and len(hist) >= 20:
                    five_ma = sum(list(hist)[-5:]) / 5
                    twenty_ma = sum(list(hist)[-20:]) / 20
                    five_above_twenty[ticker] = five_ma > twenty_ma
                # v6: drawdown from 252d high (Tepper V-bottom signal)
                if hist and len(hist) >= 60:
                    lookback = list(hist)[-252:] if len(hist) >= 252 else list(hist)
                    high = max(lookback)
                    if high > 0:
                        dd_from_252_high[ticker] = max(0.0, (high - cur) / high)
                # v9: realized vol (HV)
                if hist:
                    v30 = self._hv.hv_30(hist)
                    v60 = self._hv.hv_60(hist)
                    if v30 is not None:
                        hv30[ticker] = v30
                    if v60 is not None:
                        hv60[ticker] = v60

        today = self.time.date()
        iso = today.isoformat()
        if iso in self._cstability_vote_csv:
            self._cstability_vote_count = self._cstability_vote_csv[iso]

        # Tier1 Signal 2: volume spike — today $-vol / mean(prior 20d $-vol).
        # >3 = institutional confirmation. Requires 21 days of history (today + 20 baseline).
        volume_spike_today: dict[str, float] = {}
        for ticker in self._option_symbols:
            vh = self._volume_history.get(ticker)
            if vh is None or len(vh) < 21:
                continue
            today_dv = vh[-1]
            if today_dv <= 0:
                continue
            baseline = list(vh)[-21:-1]
            avg = sum(baseline) / len(baseline)
            if avg <= 0:
                continue
            volume_spike_today[ticker] = today_dv / avg

        # Insider Form 4 — tickers with recent opportunistic buys
        insider_recent: dict[str, float] = {}
        insider_cluster: dict[str, float] = {}
        for ticker in self._option_symbols:
            buys = self._insider.buys_within_n_days(ticker, today, n=5, min_dollar=25_000)
            if buys:
                # Use the most-recent buy's dollar value
                insider_recent[ticker] = max(b[2] for b in buys)
            # Tier1 Signal 3: cluster score over last 5 days
            cs = self._insider.cluster_score(ticker, today, n_days=5, min_dollar=25_000)
            if cs > 0:
                insider_cluster[ticker] = cs

        # Tier1 Signal 1 — GDELT news sentiment + article volume
        # tone: rescale from GDELT's [-100, +100] to [-1.0, +1.0] for downstream
        # confluence thresholds.
        news_sentiment_24h: dict[str, float] = {}
        news_volume_spike: dict[str, float] = {}
        for ticker in self._option_symbols:
            tone = self._gdelt.tone(ticker, today)
            if tone is not None:
                news_sentiment_24h[ticker] = tone / 100.0
            vs = self._gdelt.volume_spike(ticker, today)
            if vs is not None:
                news_volume_spike[ticker] = vs

        earnings_today = {t for t in self._option_symbols
                          if self._earnings.next_earnings(t, today) == today}
        earnings_5d = {t for t in self._option_symbols
                       if self._earnings.within_n_days(t, today, 5)}
        last_surprise: dict[str, float] = {}
        days_since_earnings: dict[str, int] = {}
        days_until_earnings: dict[str, int] = {}
        for t in self._option_symbols:
            sp = self._earnings.last_surprise(t, today)
            if sp is not None:
                last_surprise[t] = sp
            ds = self._earnings.days_since_last_earnings(t, today)
            if ds is not None:
                days_since_earnings[t] = ds
            nxt = self._earnings.next_earnings(t, today)
            if nxt is not None:
                days_until_earnings[t] = (nxt - today).days

        # VIX / VIX3M / VIX9D
        def _read_price(sym) -> float | None:
            if sym is None:
                return None
            sec = self.securities.get(sym)
            if sec is None or not getattr(sec, "has_data", False):
                return None
            try:
                return float(sec.price)
            except Exception:
                return None
        vix_val = _read_price(self._vix_symbol)
        vix3m_val = _read_price(self._vix3m_symbol)
        vix9d_val = _read_price(self._vix9d_symbol)
        term_ratio = compute_term_ratio(vix_val, vix3m_val)
        term_regime = classify_term_regime(vix_val, vix3m_val)
        vix9d_vix_ratio = (vix9d_val / vix_val) if (vix9d_val and vix_val and vix_val > 0) else None

        # Per-ticker IV rank + raw IV + IV/HV ratio (v9)
        iv_rank: dict[str, float] = {}
        iv_raw: dict[str, float] = {}
        iv_hv_ratio: dict[str, float] = {}
        for ticker in self._option_symbols:
            hist = list(self._iv_rank._history.get(ticker, []))
            if not hist:
                continue
            today_iv = hist[-1]
            iv_raw[ticker] = today_iv
            # IV rank requires warmup
            if self._iv_rank.has_warmup(ticker, min_samples=60):
                r = self._iv_rank.rank(ticker, today_iv)
                if r is not None:
                    iv_rank[ticker] = r
            # IV/HV ratio — v9's missing-signal addition
            hv = hv30.get(ticker)
            if hv is not None and hv > 0:
                iv_hv_ratio[ticker] = today_iv / hv

        # Per-ticker skew + z-score from SkewTracker
        skew_now: dict[str, float] = {}
        skew_z: dict[str, float] = {}
        for ticker in self._option_symbols:
            cur = self._skew.current(ticker)
            if cur is not None:
                skew_now[ticker] = cur
                z = self._skew.z_score(ticker)
                if z is not None:
                    skew_z[ticker] = z

        # ─── LEADING signals for v_REFLEX_v2 and successors (2026-05-26) ────
        # All four fields default to empty dicts if their source data is not
        # yet populated in the Object Store — strategies are responsible for
        # the gate-closed semantics (don't fire on missing data).
        si_velocity: dict[str, float] = {}
        ins_count_5d: dict[str, int] = {}
        news_propagation: dict[str, float] = {}
        im_vs_realized: dict[str, float] = {}
        for ticker in self._option_symbols:
            # Short interest velocity (FINRA biweekly)
            if self._finra is not None:
                v = self._finra.velocity(ticker, today)
                if v is not None:
                    si_velocity[ticker] = v
            # Insider count over last 5 trading days (Form 4)
            ic = self._insider.distinct_insider_count(ticker, today, n_days=5)
            if ic > 0:
                ins_count_5d[ticker] = ic
            # News propagation 5d/5d ratio (GDELT)
            np = self._gdelt.propagation_5d(ticker, today)
            if np is not None:
                news_propagation[ticker] = np
            # Implied move vs realized vol
            iv = iv_raw.get(ticker)
            hv = hv30.get(ticker)
            if iv is not None and hv is not None and hv > 0:
                # IM_30d = IV × sqrt(30/365) ≈ implied 30d move pct
                # HV_30d is annualized; we scale similarly for ratio
                im_vs_realized[ticker] = iv / hv

        # Crisis state update (Phase 6)
        crisis_state = self._crisis.update(
            today=today,
            vix=vix_val,
            vix_term_ratio=term_ratio,
            term_regime=term_regime,
            spy_drawdown_from_252d_high=dd_from_252_high.get("SPY"),
            spy_5ma_above_20ma=five_above_twenty.get("SPY", False),
        )

        # Dynamic path exposes the current month's entry-eligible top-N so
        # DynamicPitMomentumCalls can range over it; empty on the static path.
        active_universe = sorted(self._dyn_active) if self._dynamic_pit else []

        return StrategyContext(
            timestamp=self.time,
            underlying_prices=underlying_prices,
            active_universe=active_universe,
            last_entry_date=self._last_entry_by_ticker,
            vix=vix_val,
            vix3m=vix3m_val,
            vix9d=vix9d_val,
            vix_term_ratio=term_ratio,
            vix9d_vix_ratio=vix9d_vix_ratio,
            term_regime=term_regime,
            gex_total=self._gex_total,
            gex_regime=self._gex_regime,
            skew=skew_now,
            skew_z=skew_z,
            cstability_vote_count=self._cstability_vote_count,
            cgrowth_q_m_top5=[],         # Phase 3: cgrowth Q+M publishing
            regime=None,                 # Phase 3: regime classifier read
            iv_rank=iv_rank,
            earnings_today=earnings_today,
            earnings_within_5d=earnings_5d,
            last_earnings_surprise_pct=last_surprise,
            days_since_last_earnings=days_since_earnings,
            days_until_next_earnings=days_until_earnings,
            pc_ratio_equity=self._pc_ratio.current,
            short_pressure_fee_rate={},  # populated by strategies on-demand
            uoa_active=self._uoa_tickers_today,
            underlying_momentum_30d=momentum_30d,
            underlying_momentum_60d=momentum_60d,
            underlying_5ma_above_20ma=five_above_twenty,
            underlying_drawdown_from_252d_high=dd_from_252_high,
            historical_vol_30d=hv30,
            historical_vol_60d=hv60,
            iv_raw=iv_raw,
            iv_hv_ratio=iv_hv_ratio,
            insider_recent_buys=insider_recent,
            crisis_state=crisis_state,
            crisis_vix_peak=self._crisis.vix_peak,
            # Tier 1 signals
            volume_spike=volume_spike_today,
            insider_cluster_score=insider_cluster,
            news_sentiment_24h=news_sentiment_24h,
            news_volume_spike=news_volume_spike,
            # Leading signals (2026-05-26, for v_REFLEX_v2 and successors)
            short_interest_velocity=si_velocity,
            insider_count_5d=ins_count_5d,
            news_propagation_5d=news_propagation,
            implied_move_vs_realized=im_vs_realized,
        )

    def _compute_gex(self, data: Slice) -> None:
        """Compute aggregate GEX from the SPY chain (market regime proxy)."""
        spy_opt = self._option_symbols.get("SPY")
        if spy_opt is None:
            self._gex_total = None
            self._gex_regime = None
            return
        chain = data.option_chains.get(spy_opt)
        spy_sec = self.securities.get(self.symbol("SPY")) if "SPY" in self._option_symbols else None
        if chain is None or spy_sec is None or not spy_sec.has_data:
            return
        spot = float(spy_sec.price)
        result = compute_gex_contributions(list(chain), spot=spot)
        if result["count_used"] >= 10:
            self._gex_total = result["gex_total"]
            self._gex_regime = classify_gex_regime(result["gex_total"])

    def _compute_skew(self, data: Slice) -> None:
        """Per-ticker 25Δ put-call IV spread, fed into SkewTracker.

        2026-05-27: pass `spot` so BS-inverse fallback in skew.py can compute
        IV when greeks-IV is missing (the daily-resolution case).
        """
        for ticker, opt_sym in self._option_symbols.items():
            chain = data.option_chains.get(opt_sym)
            if chain is None:
                continue
            sec = self.securities.get(self.symbol(ticker))
            spot = float(sec.price) if (sec is not None and sec.has_data) else None
            skew_val = compute_skew(list(chain), now_date=self.time.date(), spot=spot)
            self._skew.update(ticker, skew_val)

    def _update_price_history(self) -> None:
        """Append exactly one closing price per calendar day per ticker.

        Critical at MINUTE resolution: on_data fires ~390×/trading day, so
        an unguarded append would fill the 260-slot deque in <1 day and
        every "multi-day" indicator (5MA/20MA, 252d high, momentum_Nd,
        HV30/60) would be reading intraday noise. The per-day guard makes
        these signals behave the way the strategies and CrisisDetector
        assume — one bar per day.

        Tier1 Signal 2 also samples $-volume (price * sec.volume) into a
        parallel _volume_history deque for the volume_spike signal.
        """
        today = self.time.date()
        for ticker in self._option_symbols:
            if self._last_price_sample_date.get(ticker) == today:
                continue
            sec = self.securities.get(self.symbol(ticker))
            if sec is not None and sec.has_data:
                price = float(sec.price)
                vol = float(getattr(sec, "volume", 0) or 0)
                self._price_history[ticker].append(price)
                self._volume_history[ticker].append(price * vol)
                self._last_price_sample_date[ticker] = today

    def _sample_atm_iv(self, data: Slice) -> None:
        """Per-ticker daily IV sample: take Greeks.ImpliedVolatility from the
        ~ATM ~30-DTE call. Feeds IVRankTracker."""
        for ticker, opt_sym in self._option_symbols.items():
            chain = data.option_chains.get(opt_sym)
            if chain is None:
                continue
            sec = self.securities.get(self.symbol(ticker))
            if sec is None or not sec.has_data:
                continue
            spot = float(sec.price)
            best = None
            best_score = float("inf")
            for c in chain:
                right_is_call = (getattr(c, "right", None) == 0 or
                                  str(getattr(c, "right", "")).lower() == "call")
                if not right_is_call:
                    continue
                exp = c.expiry.date() if hasattr(c.expiry, "date") else c.expiry
                dte = (exp - self.time.date()).days
                if dte <= 0 or dte > 60:
                    continue
                strike_dist = abs(float(c.strike) - spot)
                dte_dist = abs(dte - 30)
                score = strike_dist + 0.01 * dte_dist
                if score < best_score:
                    best_score = score
                    best = c
            if best is None:
                continue
            greeks = getattr(best, "greeks", None) or getattr(best, "Greeks", None)
            iv = getattr(greeks, "implied_volatility", None) if greeks else None
            if iv is None:
                iv = getattr(greeks, "ImpliedVolatility", None) if greeks else None
            # 2026-05-27 BS-inverse fallback (Track 2). The v28 probe BT showed
            # Greeks.implied_volatility populates 0% at QC daily resolution.
            # When greeks-iv is missing, solve IV from the contract's market
            # price via Brent on Black-Scholes. Brings probe_t_iv_raw,
            # probe_t_iv_rank, probe_t_iv_hv_ratio from 0% → ~90% for liquid
            # underlyings, unlocking implied_move_divergence + triple_confluence.
            if iv is None or iv <= 0:
                try:
                    from edge_signals.iv_inverse import solve_iv_cached
                    # Contract price preference: mid > last > price
                    bid = float(getattr(best, "bid_price", 0) or 0)
                    ask = float(getattr(best, "ask_price", 0) or 0)
                    if bid > 0 and ask > 0:
                        mkt = 0.5 * (bid + ask)
                    else:
                        mkt = (float(getattr(best, "last_price", 0) or 0) or
                               float(getattr(best, "price", 0) or 0))
                    if mkt > 0:
                        exp_d = best.expiry.date() if hasattr(best.expiry, "date") else best.expiry
                        T = max(1.0 / 365.0, (exp_d - self.time.date()).days / 365.0)
                        solved = solve_iv_cached(mkt, spot, float(best.strike), T,
                                                  r=0.04, q=0.0, side="call")
                        if solved is not None and solved > 0:
                            iv = solved
                except Exception:
                    pass
            if iv is not None and iv > 0:
                self._iv_rank.update(ticker, float(iv))

    def _detect_uoa_active_tickers(self, data: Slice) -> None:
        """Per-ticker UOA flag: any call contract in the ticker's chain that
        the UOATracker flags. Reset each tick."""
        flagged: set[str] = set()
        for ticker, opt_sym in self._option_symbols.items():
            chain = data.option_chains.get(opt_sym)
            if chain is None:
                continue
            for c in chain:
                right_is_call = (getattr(c, "right", None) == 0 or
                                  str(getattr(c, "right", "")).lower() == "call")
                if not right_is_call:
                    continue
                vol_today = float(getattr(c, "volume", 0) or 0)
                if vol_today <= 0:
                    continue
                if self._uoa.is_uoa(str(c.symbol), vol_today):
                    flagged.add(ticker)
                    break  # one is enough per ticker
        self._uoa_tickers_today = flagged

    def _record_uoa_history(self, data: Slice) -> None:
        for ticker, opt_sym in self._option_symbols.items():
            chain = data.option_chains.get(opt_sym)
            if chain is None:
                continue
            for c in chain:
                self._uoa.record(str(c.symbol),
                                 float(getattr(c, "volume", 0) or 0),
                                 float(getattr(c, "open_interest", 0) or 0))

    def _update_pc_ratio(self, data: Slice) -> None:
        put_vol = 0
        call_vol = 0
        for opt_sym in self._option_symbols.values():
            chain = data.option_chains.get(opt_sym)
            if chain is None:
                continue
            for c in chain:
                v = int(getattr(c, "volume", 0) or 0)
                right_is_call = (getattr(c, "right", None) == 0 or
                                  str(getattr(c, "right", "")).lower() == "call")
                if right_is_call:
                    call_vol += v
                else:
                    put_vol += v
        if call_vol > 0:
            self._pc_ratio.consume_day(put_vol, call_vol)

    def _try_enter(self, signal, data: Slice, n_legs: int = 1) -> None:
        # PIT survivorship gate (default OFF; configured in initialize). When ON,
        # only names that were index members as of the current date may be traded;
        # if the PIT data failed to load we fail CLOSED (block everything).
        if self._pit_gate and (
            self._pit is None
            or signal.underlying not in self._pit.members_asof(self.time)
        ):
            return
        # Dynamic-PIT path: a freshly-rotated-in chain isn't in this bar's Slice
        # yet (it was added at the month_start rebalance). Refuse entry on the add
        # date; the next daily bar carries it — immaterial given monthly rotation.
        if self._dynamic_pit:
            pend = self._dyn_pending_entry.get(signal.underlying)
            if pend is not None and pend == self.time.date():
                self._fail_chain[signal.strategy_id] += 1
                return
        opt_sym = self._option_symbols.get(signal.underlying)
        if opt_sym is None:
            self._fail_chain[signal.strategy_id] += 1   # v8 diag
            return
        chain = data.option_chains.get(opt_sym)
        if chain is None:
            self._fail_chain[signal.strategy_id] += 1   # v8 diag
            return
        equity_sym = self.symbol(signal.underlying)
        sec = self.securities.get(equity_sym)
        if sec is None or not sec.has_data:
            self._fail_chain[signal.strategy_id] += 1   # v8 diag
            return
        spot = float(sec.price)

        picked = pick_contract(
            chain, signal,
            spot=spot,
            now=self.time.date(),
        )
        if picked is None:
            self._fail_picker[signal.strategy_id] += 1   # v8 diag
            return

        nav = float(self.portfolio.total_portfolio_value)
        sizing = size_position(
            signal,
            contract_mid_price=picked.mid,
            nav=nav,
            config=self._sizer_config,
            n_legs=n_legs,
            starting_nav=self._starting_nav,
        )
        if sizing.contracts <= 0:
            self._fail_sizer[signal.strategy_id] += 1   # v8 diag
            return

        # Guard against duplicate open positions per strategy/underlying
        already_open = sum(
            1 for p in self._exit_mgr.positions()
            if p.strategy_id == signal.strategy_id and p.symbol_str.startswith(signal.underlying)
        )
        if already_open >= signal.max_concurrent_per_underlying:
            self._fail_dup[signal.strategy_id] += 1   # v8 diag
            return

        # v5: Global open-position cap — refuse new entries if total open
        # premium (this trade + already-open) would exceed cap_pct_nav of NAV.
        proposed_capital = sizing.capital_committed_dollars
        cur_open_capital = sum(
            p.entry_premium_per_share * 100.0 * p.contracts
            for p in self._exit_mgr.positions()
        )
        global_cap = nav * self._sizer_config.global_open_cap_pct_nav
        if cur_open_capital + proposed_capital > global_cap:
            self._fail_cap[signal.strategy_id] += 1   # v8 diag
            return

        # v5: Per-strategy cap — same logic, scoped to signal.strategy_id.
        cur_strategy_capital = sum(
            p.entry_premium_per_share * 100.0 * p.contracts
            for p in self._exit_mgr.positions()
            if p.strategy_id == signal.strategy_id
        )
        strategy_cap = nav * self._sizer_config.per_strategy_cap_pct_nav
        if cur_strategy_capital + proposed_capital > strategy_cap:
            self._fail_cap[signal.strategy_id] += 1   # v8 diag
            return

        # v4: REMOVED is_market_open guard — on daily resolution, algo.time
        # is EOD/after-hours so the guard always returns False (zero trades
        # in v3). Slippage control comes from limit_order instead.
        # Limit order at mid + 1.5% (small premium over mid for fill prob).
        limit_px = round(picked.mid * 1.015, 2)
        ticket = self.limit_order(picked.contract.symbol, sizing.contracts, limit_px)
        if ticket is None or ticket.order_id <= 0:
            return

        self._signals_entered[signal.strategy_id] += 1   # v7 diag
        if self._dynamic_pit:
            # cooldown keys on entry, not emit (see DynamicPitMomentumCalls)
            self._last_entry_by_ticker[signal.underlying] = self.time.date()
        self._pending_entries[ticket.order_id] = {
            "signal": signal,
            "strategy_id": signal.strategy_id,
            "underlying": signal.underlying,
            "contract_symbol": picked.symbol_str,
            "contract_symbol_obj": picked.contract.symbol,   # v25: store Symbol for securities[sym] lookup
            "side": signal.side,
            "edge_score": signal.edge_score,
            "underlying_price_at_entry": spot,
            "strike": picked.strike,
            "expiry": picked.expiry,
        }

    def _process_exits(self, data: Slice) -> None:
        # v25: iterate tracked positions DIRECTLY (not the filtered chain — a
        # contract <14 DTE or >30 strikes OTM drops out of data.option_chains,
        # so SL was never evaluated and positions rode to expiry as "manual").
        # Price/greeks come from the persistent securities[]/portfolio subscription.
        from exit_manager import compute_current_prices
        # v27: portfolio.holdings.price PRIMARY, securities backup + greeks source.
        current_prices, current_deltas, diag = compute_current_prices(
            self._exit_mgr.positions(),
            lambda sym: self.securities.get(sym),
            portfolio_lookup=lambda sym: self.portfolio.get(sym),
        )
        for k, v in diag.items():
            self._exit_feed_diag[k] += v

        # v15b DIAG: count cycles where SL math SHOULD trigger
        # (pnl <= stop_loss_pct). Independent of whether order then fires.
        # If this counter > 0 AND momentums_X_stop = 0 → exit_manager bug.
        # If this counter = 0 → SL math never triggered (price too stable).
        for pos in self._exit_mgr.positions():
            cur = current_prices.get(pos.symbol_str)
            if cur is None or cur <= 0 or pos.stop_loss_pct is None:
                continue
            pnl = pos.pnl_pct(cur)
            if pnl <= pos.stop_loss_pct:
                self._exit_feed_diag['n_sl_math_triggered'] += 1
            if pnl <= -0.20:
                self._exit_feed_diag['n_pnl_below_minus20'] += 1
            if pnl <= -0.50:
                self._exit_feed_diag['n_pnl_below_minus50'] += 1

        today = self.time.date()
        # v25 DIAG (kept for log-level granularity)
        n_tracked = self._exit_mgr.n_open
        if n_tracked > 0:
            self.debug(
                f"[V25_DIAG] {today} n_tracked={n_tracked} "
                f"n_priced={len(current_prices)} n_delta={len(current_deltas)}"
            )

        # v5: APPROACHING-EXPIRY SWEEP — for any tracked position with DTE ≤ 3,
        # force-close NOW via market_order to prevent ITM assignment + margin
        # calls (PLTR-style failure in v4/Phase 3). Independent of strategy
        # exit rules — this is a hard backstop on the framework level.
        for pos in list(self._exit_mgr.positions()):
            dte = pos.dte_remaining(today)
            if dte <= 3:
                self._exit_mgr.force_close(pos.symbol_str, "expiry")
            # v25: force-close on delta-died (option mechanically near-
            # worthless; safety net independent of price-based SL trigger)
            elif current_deltas.get(pos.symbol_str, 1.0) < 0.05:
                self._exit_mgr.force_close(pos.symbol_str, "stop_loss")

        # v5: STALE-LIMIT FALLBACK — track per-symbol "exit-attempt date".
        # If a position has been flagged for exit > 2 days but limit hasn't
        # filled (still tracked), escalate to market_order.
        if not hasattr(self, "_exit_attempt_date"):
            self._exit_attempt_date: dict[str, date] = {}

        to_close = self._exit_mgr.positions_to_close(current_prices, today)
        for sym_str, reason in to_close:
            pos = self._exit_mgr.get(sym_str)
            if pos is None:
                continue
            # Underlying price for logging
            equity_sym = self.symbol(pos.symbol_str.split(" ")[0]) if " " in pos.symbol_str else None
            underlying_price = float(self.securities[equity_sym].price) if equity_sym else 0.0

            # v5: pick order type — market for hard backstops, limit for normal exits
            # v22 Fix 1: stop_loss is now market-IMMEDIATELY (was lumped with
            # take_profit waiting 2 days before escalation). SL is a risk-
            # control exit — never let limit drift through a falling premium.
            first_attempt = sym_str not in self._exit_attempt_date
            days_attempting = 0 if first_attempt else (today - self._exit_attempt_date[sym_str]).days
            is_hard_exit = (
                reason in ("expiry", "stop_loss") or
                days_attempting >= 2
            )

            # v27: order on the Symbol OBJECT (pos.symbol), not sym_str — string
            # resolution fails for contracts outside the active chain filter, so
            # the exit silently no-ops and the position rides to expiry ("manual").
            order_target = pos.symbol if pos.symbol is not None else sym_str

            if is_hard_exit:
                ticket = self.market_order(order_target, -pos.contracts)
            else:
                cur_mid = current_prices.get(sym_str, 0.01)
                # v5: wider concession (mid × 0.95) for higher fill probability.
                limit_px = max(0.01, round(cur_mid * 0.95, 2))
                ticket = self.limit_order(order_target, -pos.contracts, limit_px)

            if ticket is None or ticket.order_id <= 0:
                # v27 DIAG: count failures so we can see if order routing
                # itself is the remaining bottleneck.
                self._exit_feed_diag['exit_order_failed'] += 1
                continue
            self._exit_feed_diag['exit_order_placed'] += 1
            if first_attempt:
                self._exit_attempt_date[sym_str] = today
            self._pending_exits[ticket.order_id] = {
                "contract_symbol": sym_str,
                "reason": reason,
                "underlying_price_at_exit": underlying_price,
            }
