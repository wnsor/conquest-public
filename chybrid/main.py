# region imports
from AlgorithmImports import *
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
# endregion


# ============================================================================
# CstabilitySleeve — non-QC helper holding cstability v11 LIVE state.
# Body mirrors cstability/main.py (LIVE pin: backtest 40d68ddea257c3d551f2c9ab04c2c254).
# QC handles (self.object_store, self.time, self.log, self.is_warming_up, etc.)
# are accessed via self.algo. Sleeve-private state stays on self.
# ============================================================================
class CstabilitySleeve:
    UNIVERSE_FULL = [
        "SPY", "QQQ", "IWM",
        "XLK", "XLF", "XLV", "XLE", "XLY", "XLP",
        "XLI", "XLB", "XLU", "XLRE", "XLC",
        "SPLV", "USMV",
        "AGG", "TLT", "IEF",
        "GLD", "DBC", "TIP", "VNQ",
        "EFA", "IEMG",
        "HYG", "LQD", "BIL",
    ]
    UNIVERSE_TRIMMED = [t for t in UNIVERSE_FULL if t not in ("XLRE", "XLC", "SPLV", "USMV", "IEMG")]

    TOP_N                  = 3
    MOMENTUM_LOOKBACK_DAYS = 252
    STAGFLATION_BASKET     = ("GLD", "TIP", "TLT")

    REGIME_STORE_KEY        = "conquest/regime/daily.csv"
    PROBABILITY_STORE_KEY   = "conquest/regime/probability.csv"
    CREDIT_STRESS_STORE_KEY = "conquest/credit/hyg_ief_spread.csv"
    VIX_TERM_STORE_KEY      = "conquest/vix/term_ratio.csv"
    T10Y2Y_STORE_KEY        = "conquest/yield_curve/t10y2y.csv"
    VIX_SPOT_STORE_KEY      = "conquest/vix/daily.csv"

    PROB_STAG_THRESHOLD        = 0.30
    CREDIT_STRESS_THRESHOLD    = -0.05
    VIX_TERM_THRESHOLD         = 1.05
    T10Y2Y_INVERSION_THRESHOLD = 0.0
    BLEND_WEIGHTS              = [0.0, 0.15, 0.40, 0.75, 1.0]

    # v11 LIVE defaults (mirror cstability/main.py)
    VOTE_MODE                  = "v10"
    LAYER1_CASH_ENABLED        = 1
    LAYER1_VIX_HIGH            = 30.0
    LAYER1_VIX_LOW             = 25.0
    LAYER1_MAX_DAYS_IN_CASH    = 90
    DEFENSIVE_BASKET_MODE      = "bil_cash"
    INTRA_MONTH_EDGE_ENABLED   = 1
    REENTRY_ENABLED            = 1
    REENTRY_VIX_MAX            = 30.0
    REENTRY_SPY_ROC_DAYS       = 20
    REENTRY_SPY_ROC_MIN        = 0.05
    REENTRY_HOLD_DAYS          = 5
    CASH_TICKER                = "BIL"
    USE_SPY_FLOOR_AT_ZERO_VOTES = 0
    FLOOR_MODE                 = "spy_qqq"
    USE_BIL_CASH_AT_MAX_STRESS = 0

    def __init__(self, algo, universe_tickers, signals, symbols_by_ticker, symbol_to_ticker, spy_roc):
        self.algo = algo
        self.UNIVERSE = universe_tickers
        self.signals = signals
        self.symbols_by_ticker = symbols_by_ticker
        self.symbol_to_ticker = symbol_to_ticker
        self.spy_roc = spy_roc
        self.vote_count = 0
        self.last_votes = (False, False, False, False)
        self.last_vote_count = 0
        self.in_layer1_cash = False
        self.layer1_entered_at = None
        self.last_reentry_at = None
        self.regime_df = None
        self.probability_df = None
        self.credit_df = None
        self.vix_term_df = None
        self.t10y2y_df = None
        self.vix_spot_df = None
        self.stag_symbols = []
        self.cash_sym = None

    def initialize_store(self):
        self.regime_df       = self._load_csv(self.REGIME_STORE_KEY,        "regime")
        self.probability_df  = self._load_csv(self.PROBABILITY_STORE_KEY,   "probability")
        self.credit_df       = self._load_csv(self.CREDIT_STRESS_STORE_KEY, "credit stress")
        self.vix_term_df     = self._load_csv(self.VIX_TERM_STORE_KEY,      "VIX term ratio")
        self.t10y2y_df       = self._load_csv(self.T10Y2Y_STORE_KEY,        "T10Y2Y") if self.VOTE_MODE != "v10" else None
        self.vix_spot_df     = self._load_csv(self.VIX_SPOT_STORE_KEY,      "VIX spot")
        self.stag_symbols = [
            self.symbols_by_ticker[t]
            for t in self.STAGFLATION_BASKET
            if t in self.symbols_by_ticker
        ]
        if len(self.stag_symbols) != len(self.STAGFLATION_BASKET):
            self.algo.error(
                f"[cstab] Stagflation basket missing tickers; got {len(self.stag_symbols)}"
                f" of {len(self.STAGFLATION_BASKET)}."
            )
        self.cash_sym = self.symbols_by_ticker.get(self.CASH_TICKER)

    # ----- Object Store loaders -----
    def _load_csv(self, key, label):
        try:
            csv_text = self.algo.object_store.read(key)
        except Exception as e:
            self.algo.error(f"[cstab] Object Store read of {key} failed: {e}")
            return None
        if not csv_text:
            self.algo.log(f"[cstab] WARN: {key} not in Object Store; {label} vote will return False.")
            return None
        try:
            df = pd.read_csv(StringIO(csv_text), index_col=0, parse_dates=True)
            self.algo.log(f"[cstab] {label} feed: {len(df)} rows, "
                          f"{df.index[0].date()} -> {df.index[-1].date()}")
            return df
        except Exception as e:
            self.algo.error(f"[cstab] Failed to parse {label} CSV: {e}")
            return None

    # ----- vote signal lookups -----
    def _latest_value(self, df, column=None):
        if df is None:
            return None
        valid = df.index <= self.algo.time
        if not valid.any():
            return None
        row = df.loc[valid].iloc[-1]
        if column is None:
            return row.iloc[0] if hasattr(row, "iloc") else row
        return row[column] if column in row else None

    def _current_regime(self):
        return self._latest_value(self.regime_df, column="regime")

    def _current_vix_spot(self):
        if self.vix_spot_df is None:
            return None
        valid = self.vix_spot_df.index <= self.algo.time
        if not valid.any():
            return None
        return float(self.vix_spot_df.loc[valid].iloc[-1, 0])

    def _vote_regime_stagflation(self):
        return self._current_regime() == "Stagflation"

    def _vote_probability_high(self):
        p = self._latest_value(self.probability_df, column="p_stagflation")
        return p is not None and float(p) > self.PROB_STAG_THRESHOLD

    def _vote_credit_stress(self):
        c = self._latest_value(self.credit_df, column="hy_stress_proxy")
        return c is not None and float(c) < self.CREDIT_STRESS_THRESHOLD

    def _vote_vix_backwardation(self):
        v = self._latest_value(self.vix_term_df, column="vix_term_ratio")
        return v is not None and float(v) > self.VIX_TERM_THRESHOLD

    def _vote_yield_curve_inverted(self):
        if self.t10y2y_df is None:
            return False
        for col in ("t10y2y", "spread", "value"):
            v = self._latest_value(self.t10y2y_df, column=col)
            if v is not None:
                return float(v) < self.T10Y2Y_INVERSION_THRESHOLD
        return False

    def _update_votes(self):
        if self.algo.is_warming_up:
            return
        if self.VOTE_MODE == "v10":
            votes = (
                self._vote_regime_stagflation(),
                self._vote_probability_high(),
                self._vote_credit_stress(),
                self._vote_vix_backwardation(),
            )
            label2 = "prob"
        else:
            votes = (
                self._vote_regime_stagflation(),
                self._vote_yield_curve_inverted(),
                self._vote_credit_stress(),
                self._vote_vix_backwardation(),
            )
            label2 = "t10y2y"
        new_count = sum(int(v) for v in votes)
        if new_count != self.vote_count:
            self.algo.log(
                f"[cstab] {self.algo.time:%Y-%m-%d}: vote {self.vote_count} -> {new_count} "
                f"mode={self.VOTE_MODE} (regime={int(votes[0])} {label2}={int(votes[1])} "
                f"credit={int(votes[2])} vix_term={int(votes[3])})"
            )
        self.vote_count = new_count
        self.last_votes = votes

    # ----- daily state updates (return dirty bool for intra-month edges) -----
    def update_daily(self):
        """Run daily state updates. Returns True if any intra-month edge fired."""
        if self.algo.is_warming_up:
            return False
        self._update_votes()
        dirty = False
        if self._update_layer1_cash():
            dirty = True
        if self._check_reentry():
            dirty = True
        if self._check_edge_trigger():
            dirty = True
        return dirty

    def _update_layer1_cash(self):
        """v11 Layer 1 VIX cash trigger. Returns True on enter/exit/guardrail transition."""
        if not self.LAYER1_CASH_ENABLED or self.algo.is_warming_up:
            return False
        vix = self._current_vix_spot()
        if vix is None:
            return False
        if not self.in_layer1_cash and vix > self.LAYER1_VIX_HIGH:
            self.in_layer1_cash = True
            self.layer1_entered_at = self.algo.time
            self.algo.log(f"[cstab] {self.algo.time:%Y-%m-%d}: LAYER1 ENTER cash, "
                          f"VIX={vix:.1f} > {self.LAYER1_VIX_HIGH}")
            return True
        if self.in_layer1_cash and vix < self.LAYER1_VIX_LOW:
            self.in_layer1_cash = False
            self.layer1_entered_at = None
            self.algo.log(f"[cstab] {self.algo.time:%Y-%m-%d}: LAYER1 EXIT cash, "
                          f"VIX={vix:.1f} < {self.LAYER1_VIX_LOW}")
            return True
        if self.in_layer1_cash and self.layer1_entered_at is not None:
            days = (self.algo.time - self.layer1_entered_at).days
            if days > self.LAYER1_MAX_DAYS_IN_CASH:
                self.algo.log(f"[cstab] {self.algo.time:%Y-%m-%d}: LAYER1 GUARDRAIL "
                              f"{days}d > {self.LAYER1_MAX_DAYS_IN_CASH}d (VIX={vix:.1f}); "
                              f"forcing re-eval")
                self.in_layer1_cash = False
                self.layer1_entered_at = None
                return True
        return False

    def _check_reentry(self):
        """v11.1 re-entry trigger. Returns True when fired."""
        if not self.REENTRY_ENABLED or self.algo.is_warming_up:
            return False
        in_defensive = self.in_layer1_cash or self.vote_count >= 1
        if not in_defensive:
            return False
        if self.last_reentry_at is not None:
            if (self.algo.time - self.last_reentry_at).days < self.REENTRY_HOLD_DAYS:
                return False
        vix = self._current_vix_spot()
        if vix is None or vix >= self.REENTRY_VIX_MAX:
            return False
        if self.spy_roc is None or not self.spy_roc.is_ready:
            return False
        roc = float(self.spy_roc.current.value)
        if roc < self.REENTRY_SPY_ROC_MIN:
            return False
        self.algo.log(
            f"[cstab] {self.algo.time:%Y-%m-%d}: REENTRY fired "
            f"(VIX={vix:.1f}<{self.REENTRY_VIX_MAX}, ROC={roc:.2%}>{self.REENTRY_SPY_ROC_MIN:.2%}, "
            f"vote={self.vote_count}, layer1={self.in_layer1_cash}); 100% equity"
        )
        self.in_layer1_cash = False
        self.layer1_entered_at = None
        self.last_reentry_at = self.algo.time
        return True

    def _check_edge_trigger(self):
        """v11 slow-layer intra-month edge: rebalance on vote_count change."""
        if not self.INTRA_MONTH_EDGE_ENABLED or self.algo.is_warming_up:
            return False
        if self.vote_count != self.last_vote_count:
            self.algo.log(f"[cstab] {self.algo.time:%Y-%m-%d}: edge "
                          f"vote {self.last_vote_count} -> {self.vote_count}")
            self.last_vote_count = self.vote_count
            return True
        return False

    # ----- selection + sizing -----
    def _base_allocation(self):
        regime = self._current_regime()
        if regime == "Stagflation" and self.stag_symbols:
            w = 1.0 / len(self.stag_symbols)
            return {sym: w for sym in self.stag_symbols}

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

    def _defensive_basket_weights(self):
        if self.DEFENSIVE_BASKET_MODE == "bil_cash":
            if self.cash_sym is not None:
                return {self.cash_sym: 1.0}
        if self.USE_BIL_CASH_AT_MAX_STRESS and self.vote_count == 4:
            if self.cash_sym is not None:
                return {self.cash_sym: 1.0}
        if not self.stag_symbols:
            return {}
        w = 1.0 / len(self.stag_symbols)
        return {sym: w for sym in self.stag_symbols}

    def _blend_weights(self, base, alpha):
        defensive = self._defensive_basket_weights()
        out = {}
        for sym, w in base.items():
            out[sym] = (1.0 - alpha) * w
        for sym, w in defensive.items():
            out[sym] = out.get(sym, 0.0) + alpha * w
        return {sym: w for sym, w in out.items() if abs(w) > 1e-4}

    def target_weights(self):
        """Return dict[Symbol, float] summing ≤ 1.0 (the cstability sleeve's 30% slot)."""
        if self.algo.is_warming_up:
            return {}
        # Layer 1 override: 100% cash inside the sleeve
        if self.LAYER1_CASH_ENABLED and self.in_layer1_cash and self.cash_sym is not None:
            return {self.cash_sym: 1.0}
        self._update_votes()
        base = self._base_allocation()
        in_reentry_window = (
            self.REENTRY_ENABLED
            and self.last_reentry_at is not None
            and (self.algo.time - self.last_reentry_at).days < self.REENTRY_HOLD_DAYS
        )
        alpha = 0.0 if in_reentry_window else self.BLEND_WEIGHTS[self.vote_count]
        return self._blend_weights(base, alpha)


# ============================================================================
# CgrowthSleeve — non-QC helper holding cgrowth v11 LIVE state.
# Body mirrors cgrowth/main.py (LIVE pin: backtest e1d3c9e43f6a4feeddf99ecafa49146b).
# Crisis-gate signal is read IN-MEMORY from cstab.vote_count (no Object Store CSV).
# ============================================================================
class CgrowthSleeve:
    TOP_N                  = 5
    MOMENTUM_LOOKBACK_DAYS = 180
    VOL_LOOKBACK_DAYS      = 60
    SAFE_TICKER            = "IEF"
    MAX_PER_SECTOR         = 0.30
    VIX_HIGH               = 25.0
    VIX_LOW                = 15.0
    RISK_OFF_FACTOR        = 0.5
    UNIVERSE_STORE_KEY     = "conquest/universe/sp500.csv"
    VIX_STORE_KEY          = "conquest/vix/daily.csv"
    PIT_UNION_KEY          = "conquest/universe/sp500_union_2008_2024.csv"
    PIT_MONTHLY_KEY        = "conquest/universe/sp500_pit_monthly.csv"

    SIGNAL_MODE            = "qm_composite"
    VOL_WEIGHT             = 4.0
    WEIGHTING_MODE         = "rank"
    USE_PIT_UNIVERSE       = True   # default ON for CF — matches v11 2008-2026 backtest

    CRISIS_GATE_ENABLED    = True
    CRISIS_GATE_FACTOR     = 0.25
    CRISIS_GATE_THRESHOLD  = 2

    def __init__(self, algo, sector_map, signals, symbols_by_ticker, symbol_to_ticker,
                 safe_symbol, pit_monthly, cstab_sleeve):
        self.algo = algo
        self.sector_map = sector_map
        self.signals = signals
        self.symbols_by_ticker = symbols_by_ticker
        self.symbol_to_ticker = symbol_to_ticker
        self.safe_symbol = safe_symbol
        self.pit_monthly = pit_monthly
        self.cstab = cstab_sleeve
        self.risk_off = False
        self.last_vote_was_above_threshold = False
        self.vix_df = None

    def initialize_store(self):
        self.vix_df = self._load_vix()

    def _load_vix(self):
        try:
            csv_text = self.algo.object_store.read(self.VIX_STORE_KEY)
        except Exception as e:
            self.algo.error(f"[cgrowth] Object Store read of {self.VIX_STORE_KEY} failed: {e}")
            return None
        if not csv_text:
            return None
        try:
            df = pd.read_csv(StringIO(csv_text), index_col=0, parse_dates=True)
            self.algo.log(f"[cgrowth] VIX feed: {len(df)} rows, "
                          f"{df.index[0].date()} -> {df.index[-1].date()}")
            return df
        except Exception as e:
            self.algo.error(f"[cgrowth] Failed to parse VIX CSV: {e}")
            return None

    def _current_vix(self):
        if self.vix_df is None:
            return None
        valid = self.vix_df.index <= self.algo.time
        if not valid.any():
            return None
        return float(self.vix_df.loc[valid].iloc[-1, 0])

    def update_daily(self):
        """Run daily state updates. Returns True on intra-month crisis-gate edge."""
        if self.algo.is_warming_up:
            return False
        # VIX hysteresis (state-machine only; doesn't fire intra-month rebalance)
        vix = self._current_vix()
        if vix is not None:
            if not self.risk_off and vix > self.VIX_HIGH:
                self.risk_off = True
                self.algo.log(f"[cgrowth] {self.algo.time:%Y-%m-%d}: VIX={vix:.1f} > "
                              f"{self.VIX_HIGH}, risk-off")
            elif self.risk_off and vix < self.VIX_LOW:
                self.risk_off = False
                self.algo.log(f"[cgrowth] {self.algo.time:%Y-%m-%d}: VIX={vix:.1f} < "
                              f"{self.VIX_LOW}, risk-on")
        # Crisis-gate edge: read in-memory from cstab.vote_count (no Object Store CSV).
        # Deviation from cgrowth-LIVE behavior, which reads
        # conquest/votes/cstability_4vote_daily.csv. In CF, the cstability sleeve runs
        # in-process and computes vote_count daily, so the CSV would be redundant.
        if self.CRISIS_GATE_ENABLED:
            vote_above = self.cstab.vote_count >= self.CRISIS_GATE_THRESHOLD
            if vote_above != self.last_vote_was_above_threshold:
                self.algo.log(f"[cgrowth] {self.algo.time:%Y-%m-%d}: crisis edge "
                              f"{self.last_vote_was_above_threshold}->{vote_above} "
                              f"(vote={self.cstab.vote_count})")
                self.last_vote_was_above_threshold = vote_above
                return True
        return False

    def _pit_active_set(self):
        if not self.USE_PIT_UNIVERSE or not self.pit_monthly:
            return None
        ym = self.algo.time.strftime("%Y-%m")
        if ym in self.pit_monthly:
            return self.pit_monthly[ym]
        candidates = [m for m in self.pit_monthly if m <= ym]
        if not candidates:
            return set()
        return self.pit_monthly[max(candidates)]

    def _select_momentum_only(self):
        pit_active = self._pit_active_set()
        scored = []
        for sym in self.signals:
            if pit_active is not None:
                ticker = self.symbol_to_ticker.get(sym)
                if ticker not in pit_active:
                    continue
            if not self.signals[sym]["momp"].is_ready:
                continue
            scored.append((sym, float(self.signals[sym]["momp"].current.value)))
        eligible = [(sym, m) for sym, m in scored if m > 0]
        eligible.sort(key=lambda p: p[1], reverse=True)
        return [sym for sym, _ in eligible[: self.TOP_N]]

    def _select_qm_composite(self):
        pit_active = self._pit_active_set()
        eligible = []
        for sym, sigs in self.signals.items():
            if pit_active is not None:
                ticker = self.symbol_to_ticker.get(sym)
                if ticker not in pit_active:
                    continue
            momp_ind = sigs["momp"]
            vol_ind = sigs["vol"]
            if not momp_ind.is_ready or not vol_ind.is_ready:
                continue
            mom_value = float(momp_ind.current.value)
            vol_value = float(vol_ind.current.value)
            if mom_value <= 0 or vol_value <= 0:
                continue
            eligible.append((sym, mom_value, vol_value))
        if not eligible:
            return []
        moms = [m for _, m, _ in eligible]
        vols = [v for _, _, v in eligible]
        n = len(eligible)
        mom_mean = sum(moms) / n
        vol_mean = sum(vols) / n
        mom_std = (sum((m - mom_mean) ** 2 for m in moms) / n) ** 0.5 or 1.0
        vol_std = (sum((v - vol_mean) ** 2 for v in vols) / n) ** 0.5 or 1.0
        scored = []
        for sym, mom_value, vol_value in eligible:
            z_mom = (mom_value - mom_mean) / mom_std
            z_qual = -((vol_value - vol_mean) / vol_std)
            composite = z_mom + self.VOL_WEIGHT * z_qual
            scored.append((sym, composite))
        scored.sort(key=lambda p: p[1], reverse=True)
        return [sym for sym, _ in scored[: self.TOP_N]]

    def _apply_sector_cap(self, weights):
        sector_totals = {}
        sector_symbols = {}
        for sym, w in weights.items():
            ticker = self.symbol_to_ticker.get(sym)
            sector = self.sector_map.get(ticker, "Unknown") if ticker else "Unknown"
            sector_totals[sector] = sector_totals.get(sector, 0.0) + abs(w)
            sector_symbols.setdefault(sector, []).append(sym)
        out = dict(weights)
        for sector, total in sector_totals.items():
            if total > self.MAX_PER_SECTOR:
                scale = self.MAX_PER_SECTOR / total
                for sym in sector_symbols[sector]:
                    out[sym] = weights[sym] * scale
        return out

    def target_weights(self):
        """Return dict[Symbol, float] summing ≤ 1.0 (the cgrowth sleeve's 50% slot)."""
        if self.algo.is_warming_up:
            return {}
        if self.SIGNAL_MODE == "qm_composite":
            chosen = self._select_qm_composite()
        else:
            chosen = self._select_momentum_only()
        if not chosen:
            return {self.safe_symbol: 1.0}
        n = len(chosen)
        if self.WEIGHTING_MODE == "rank":
            denom = n * (n + 1) / 2.0
            weights = {sym: (n - i) / denom for i, sym in enumerate(chosen)}
        else:
            equal_weight = 1.0 / n
            weights = {sym: equal_weight for sym in chosen}
        weights = self._apply_sector_cap(weights)
        scale = self.RISK_OFF_FACTOR if self.risk_off else 1.0
        if self.CRISIS_GATE_ENABLED and self.cstab.vote_count >= self.CRISIS_GATE_THRESHOLD:
            scale *= self.CRISIS_GATE_FACTOR
        return {sym: w * scale for sym, w in weights.items()}


# ============================================================================
# CF (Conquest Fund) — combined 50/30/10/5/5 sleeve portfolio.
# 50% cgrowth v11, 30% cstability v11, 10% crypto chain (BIL/GBTC/BITO/IBIT),
# 5% GLD, 5% BIL. Monthly rebalance to target sleeve weights, daily edge
# triggers from either sleeve fire intra-month rebalance.
# ============================================================================
class CF(QCAlgorithm):
    SLEEVE_W_CGROWTH = 0.50
    SLEEVE_W_CSTAB   = 0.30
    SLEEVE_W_CRYPTO  = 0.10
    SLEEVE_W_GLD     = 0.05
    SLEEVE_W_BIL     = 0.05

    # ---------- Stagflation tilt candidate (default OFF — preserves CF v1 LIVE) ----------
    # When STAGFLATION_TILT_ENABLED=1 AND cstab's current regime is "Stagflation",
    # the rebalance uses STAG_W_* instead of the base SLEEVE_W_* weights. Other
    # regimes (Disinflation/Inflation/Deflation) continue to use SLEEVE_W_*.
    #
    # Hypothesis (2026-05-05): cgrowth loses ~−1.7% annualized in Stagflation per
    # per-regime attribution; the 10% crypto chain crashed −65% in 2022. Two
    # independent leaks contribute most of cstag's −22.4% backtest max DD.
    # Removing cgrowth + crypto in Stagflation only — and reallocating to GLD +
    # BIL — should tighten the 2022 trough without affecting the 82% of days
    # when not in Stagflation.
    #
    # Default tilt = Config A (0% cgrowth + 30% cstab + 0% crypto + 30% GLD + 40% BIL).
    # Tunable via --parameter STAG_W_<sleeve> at backtest time.
    # PROMOTED to LIVE 2026-05-05 (cf v1.1). Strict-Pareto cleared on every metric:
    # +0.255 pp CAGR, +0.025 Sharpe, +0.038 Sortino, TIED DD, +5.21 pp PSR,
    # +$23,872 end equity vs cf v1 (post-CSV-fix). 2008-2019 sub-window DD
    # IMPROVED (−12.4% vs baseline −13.2%) — the late-2008 risk that rejected
    # cstag v2 does NOT apply to cf because cf is always in CF blend (5% GLD
    # baseline, no zero-GLD discontinuity across regime boundaries).
    # Backtest: `9ec557afb26380a11ae47fe77d11de57`. See LEARNINGS.md.
    STAGFLATION_TILT_ENABLED = 1
    STAG_W_CGROWTH = 0.0
    STAG_W_CSTAB   = 0.30
    STAG_W_CRYPTO  = 0.0
    STAG_W_GLD     = 0.30
    STAG_W_BIL     = 0.40

    # ---------- Stag-tilt V-recovery override (default ON when tilt enabled) ----------
    # Mirrors cstability's REENTRY rule: when VIX < threshold AND SPY 20d ROC >
    # threshold, skip the Stagflation tilt and use base SLEEVE_W_* weights, even
    # if the regime classifier still says Stagflation. This catches sharp
    # V-recoveries (e.g. March 2009) that the regime classifier is too slow to
    # detect because GDP/CPI YoY data lags the equity bottom.
    #
    # 2008-2019 sub-window backtest showed Config A's tilt added 0.6pp DD vs
    # baseline almost entirely from a 2008-2009 transition where the algorithm
    # stayed defensive through the March 2009 V-recovery. This override is
    # designed to fix that specific failure mode without touching 2022 behavior
    # (no V-recovery from 2022 was that fast — the 2022 trough recovered slowly,
    # so the override correctly stays inactive in 2022).
    STAG_TILT_REENTRY_ENABLED = 1
    STAG_TILT_REENTRY_VIX_MAX = 30.0
    STAG_TILT_REENTRY_ROC_MIN = 0.05

    GBTC_INCEPTION = datetime(2013, 9, 25)
    BITO_INCEPTION = datetime(2021, 10, 19)
    IBIT_INCEPTION = datetime(2024, 1, 11)

    GLD_TICKER     = "GLD"
    BIL_TICKER     = "BIL"
    CRYPTO_TICKERS = ("GBTC", "BITO", "IBIT")

    def initialize(self):
        # ---- Backtest window ----
        start_year = int(self.get_parameter("BACKTEST_START_YEAR") or 2008)
        end_year   = int(self.get_parameter("BACKTEST_END_YEAR")   or 2026)
        self.set_start_date(start_year, 1, 1)
        if end_year == 2026:
            self.set_end_date(2026, 2, 4)  # match v11 LIVE pin window
        else:
            self.set_end_date(end_year, 12, 31)
        # STARTING_CAPITAL override (default $50k matches CF v1 LIVE pin).
        # Used for the merged single-node deploy variant ($75k = original
        # CF allocation + extra $25k cgrowth-tilted slice — see
        # BILLIONAIRE_PATH.md "Why dual not just CF" section).
        starting_capital = 50000
        sc = self.get_parameter("STARTING_CAPITAL")
        if sc:
            try:
                starting_capital = int(float(sc))
            except ValueError:
                pass
        self.set_cash(starting_capital)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE)
        self.set_benchmark("SPY")

        # Slippage model — DEFAULT IS NONE for cf to preserve historical LIVE-pin
        # BT ids (v1: 32563355ae41089ccc1c46c2674d3d70; v1.1: 9ec557afb26380a11ae47fe77d11de57).
        # Live trading uses real fills; this only affects backtest. Pass
        # --parameter SLIPPAGE_MODEL volume_share for the slippage-corrected metric.
        slip_mode = (self.get_parameter("SLIPPAGE_MODEL") or "none").lower()
        if slip_mode == "volume_share":
            def _init_security(security):
                try: security.set_slippage_model(VolumeShareSlippageModel(0.025, 0.1))
                except Exception: pass
            self.set_security_initializer(_init_security)
            self.log("[cf] slippage: VolumeShareSlippageModel(0.025, 0.1)")
        elif slip_mode == "constant_10bp":
            def _init_security(security):
                try: security.set_slippage_model(ConstantSlippageModel(0.0010))
                except Exception: pass
            self.set_security_initializer(_init_security)
            self.log("[cf] slippage: ConstantSlippageModel(0.0010)")
        else:
            self.log("[cf] slippage: NONE (preserves historical LIVE-pin behavior)")

        # ---- Sleeve weight overrides ----
        for attr in ["SLEEVE_W_CGROWTH", "SLEEVE_W_CSTAB", "SLEEVE_W_CRYPTO",
                     "SLEEVE_W_GLD", "SLEEVE_W_BIL"]:
            v = self.get_parameter(attr)
            if v:
                try:
                    setattr(self, attr, float(v))
                except ValueError:
                    pass
        total_w = (self.SLEEVE_W_CGROWTH + self.SLEEVE_W_CSTAB + self.SLEEVE_W_CRYPTO
                   + self.SLEEVE_W_GLD + self.SLEEVE_W_BIL)
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"Sleeve weights must sum to 1.0; got {total_w:.4f}")

        # ---- Stagflation-tilt overrides (default OFF preserves LIVE behavior) ----
        for attr, cast in [
            ("STAGFLATION_TILT_ENABLED", int),
            ("STAG_W_CGROWTH", float), ("STAG_W_CSTAB", float),
            ("STAG_W_CRYPTO", float),  ("STAG_W_GLD", float), ("STAG_W_BIL", float),
            ("STAG_TILT_REENTRY_ENABLED", int),
            ("STAG_TILT_REENTRY_VIX_MAX", float),
            ("STAG_TILT_REENTRY_ROC_MIN", float),
        ]:
            v = self.get_parameter(attr)
            if v:
                try:
                    setattr(self, attr, cast(v))
                except ValueError:
                    pass
        if self.STAGFLATION_TILT_ENABLED:
            stag_total = (self.STAG_W_CGROWTH + self.STAG_W_CSTAB + self.STAG_W_CRYPTO
                          + self.STAG_W_GLD + self.STAG_W_BIL)
            if abs(stag_total - 1.0) > 1e-4:
                raise ValueError(f"STAG_W_* weights must sum to 1.0; got {stag_total:.4f}")
            self.log(f"[cf] Stagflation tilt ENABLED — STAG weights: "
                     f"cgrowth={self.STAG_W_CGROWTH} cstab={self.STAG_W_CSTAB} "
                     f"crypto={self.STAG_W_CRYPTO} gld={self.STAG_W_GLD} bil={self.STAG_W_BIL}")

        # ---- Sleeve-class-level toggle overrides (apply to class attrs before instantiation) ----
        for attr, cast in [
            ("VOTE_MODE", str), ("LAYER1_CASH_ENABLED", int),
            ("LAYER1_VIX_HIGH", float), ("LAYER1_VIX_LOW", float),
            ("LAYER1_MAX_DAYS_IN_CASH", int), ("DEFENSIVE_BASKET_MODE", str),
            ("INTRA_MONTH_EDGE_ENABLED", int), ("REENTRY_ENABLED", int),
            ("REENTRY_VIX_MAX", float), ("REENTRY_SPY_ROC_DAYS", int),
            ("REENTRY_SPY_ROC_MIN", float), ("REENTRY_HOLD_DAYS", int),
            ("USE_SPY_FLOOR_AT_ZERO_VOTES", int), ("FLOOR_MODE", str),
            ("USE_BIL_CASH_AT_MAX_STRESS", int),
        ]:
            v = self.get_parameter(f"CSTAB_{attr}")
            if v:
                try:
                    setattr(CstabilitySleeve, attr, cast(v))
                except ValueError:
                    pass
        for attr, cast in [
            ("VOL_WEIGHT", float), ("WEIGHTING_MODE", str), ("SIGNAL_MODE", str),
            ("USE_PIT_UNIVERSE", int), ("SAFE_TICKER", str),
            ("CRISIS_GATE_ENABLED", int), ("CRISIS_GATE_FACTOR", float),
            ("CRISIS_GATE_THRESHOLD", int),
        ]:
            v = self.get_parameter(f"CGROWTH_{attr}")
            if v:
                try:
                    val = cast(v)
                    if attr in ("USE_PIT_UNIVERSE", "CRISIS_GATE_ENABLED"):
                        val = bool(val)
                    setattr(CgrowthSleeve, attr, val)
                except ValueError:
                    pass

        # ---- Universe build (dedupe-aware) ----
        # cstability TRIMMED universe (drops post-2007 inception ETFs) — default ON
        # since CF backtests start 2008.
        use_trimmed = bool(int(self.get_parameter("CSTAB_TRIMMED_UNIVERSE") or 1))
        cstab_tickers = (CstabilitySleeve.UNIVERSE_TRIMMED if use_trimmed
                         else CstabilitySleeve.UNIVERSE_FULL)

        ticker_set = set()
        symbols_by_ticker = {}
        symbol_to_ticker = {}
        cstab_signals = {}
        cgrowth_signals = {}

        # Add cstability ETFs + their MOMP indicators
        for t in cstab_tickers:
            if t in ticker_set:
                continue
            equity = self.add_equity(t, Resolution.DAILY)
            sym = equity.symbol
            ticker_set.add(t)
            symbols_by_ticker[t] = sym
            symbol_to_ticker[sym] = t
            cstab_signals[sym] = {
                "momp": self.MOMP(sym, CstabilitySleeve.MOMENTUM_LOOKBACK_DAYS, Resolution.DAILY),
            }

        # SPY 20d ROC for cstab re-entry trigger (SPY is always in cstab universe)
        spy_sym = symbols_by_ticker.get("SPY")
        spy_roc = (self.ROC(spy_sym, CstabilitySleeve.REENTRY_SPY_ROC_DAYS, Resolution.DAILY)
                   if spy_sym else None)

        # ---- cgrowth universe load ----
        use_pit = CgrowthSleeve.USE_PIT_UNIVERSE
        pit_monthly = {}
        if use_pit:
            try:
                union_text = self.object_store.read(CgrowthSleeve.PIT_UNION_KEY)
                pit_text   = self.object_store.read(CgrowthSleeve.PIT_MONTHLY_KEY)
            except Exception as e:
                raise RuntimeError(f"PIT object-store read failed: {e}. "
                                   f"Push storage/conquest/universe/sp500_union_2008_2024.csv "
                                   f"and sp500_pit_monthly.csv first.")
            universe_df = pd.read_csv(StringIO(union_text))
            pit_df      = pd.read_csv(StringIO(pit_text))
            for as_of, group in pit_df.groupby("as_of"):
                ym = str(as_of)[:7]
                pit_monthly[ym] = set(group["ticker"].astype(str).tolist())
            self.log(f"[cf] PIT universe: union={len(universe_df)} tickers, "
                     f"snapshots={len(pit_monthly)}")
        else:
            try:
                csv_text = self.object_store.read(CgrowthSleeve.UNIVERSE_STORE_KEY)
            except Exception as e:
                raise RuntimeError(f"Universe load failed: {e}")
            universe_df = pd.read_csv(StringIO(csv_text))
        sector_map = dict(zip(universe_df["ticker"], universe_df["sector"]))

        # Add IEF safe-haven (cgrowth)
        safe_t = CgrowthSleeve.SAFE_TICKER
        if safe_t not in ticker_set:
            equity = self.add_equity(safe_t, Resolution.DAILY)
            sym = equity.symbol
            ticker_set.add(safe_t)
            symbols_by_ticker[safe_t] = sym
            symbol_to_ticker[sym] = safe_t
        safe_symbol = symbols_by_ticker[safe_t]

        # Add cgrowth S&P 500 stocks + their MOMP+STD indicators
        for t in universe_df["ticker"].tolist():
            if t in ticker_set:
                continue
            equity = self.add_equity(t, Resolution.DAILY)
            sym = equity.symbol
            ticker_set.add(t)
            symbols_by_ticker[t] = sym
            symbol_to_ticker[sym] = t
            cgrowth_signals[sym] = {
                "momp": self.MOMP(sym, CgrowthSleeve.MOMENTUM_LOOKBACK_DAYS, Resolution.DAILY),
                "vol":  self.STD(sym, CgrowthSleeve.VOL_LOOKBACK_DAYS, Resolution.DAILY),
            }

        # Add crypto chain (GBTC/BITO/IBIT — BIL already in cstab universe)
        for t in self.CRYPTO_TICKERS:
            if t in ticker_set:
                continue
            equity = self.add_equity(t, Resolution.DAILY)
            sym = equity.symbol
            ticker_set.add(t)
            symbols_by_ticker[t] = sym
            symbol_to_ticker[sym] = t

        # GLD + BIL must be in the cstab universe; verify
        if self.GLD_TICKER not in ticker_set:
            self.error(f"[cf] GLD missing from universe — should be in cstability set")
        if self.BIL_TICKER not in ticker_set:
            self.error(f"[cf] BIL missing from universe — should be in cstability set")

        self.log(f"[cf] Universe: {len(ticker_set)} tickers "
                 f"(cstab={len(cstab_tickers)}, cgrowth={len(universe_df)}, "
                 f"+IEF, +3 crypto)")

        # ---- Sleeve instantiation ----
        cstab_symbols_by_ticker = {t: symbols_by_ticker[t] for t in cstab_tickers}
        cstab_symbol_to_ticker = {symbols_by_ticker[t]: t for t in cstab_tickers}
        self.cstab = CstabilitySleeve(
            algo=self,
            universe_tickers=cstab_tickers,
            signals=cstab_signals,
            symbols_by_ticker=cstab_symbols_by_ticker,
            symbol_to_ticker=cstab_symbol_to_ticker,
            spy_roc=spy_roc,
        )
        cgrowth_tickers = universe_df["ticker"].tolist()
        cgrowth_symbols_by_ticker = {
            t: symbols_by_ticker[t] for t in cgrowth_tickers if t in symbols_by_ticker
        }
        cgrowth_symbols_by_ticker[safe_t] = safe_symbol
        cgrowth_symbol_to_ticker = {sym: t for t, sym in cgrowth_symbols_by_ticker.items()}
        self.cgrowth = CgrowthSleeve(
            algo=self,
            sector_map=sector_map,
            signals=cgrowth_signals,
            symbols_by_ticker=cgrowth_symbols_by_ticker,
            symbol_to_ticker=cgrowth_symbol_to_ticker,
            safe_symbol=safe_symbol,
            pit_monthly=pit_monthly,
            cstab_sleeve=self.cstab,
        )

        # CF-level symbol shortcuts
        self.gld_sym = symbols_by_ticker[self.GLD_TICKER]
        self.bil_sym = symbols_by_ticker[self.BIL_TICKER]
        self.crypto_syms = {t: symbols_by_ticker[t] for t in self.CRYPTO_TICKERS}
        self.all_symbols = set(symbols_by_ticker.values())

        # Object Store load (after add_equity completes; uses self.algo.object_store)
        self.cstab.initialize_store()
        self.cgrowth.initialize_store()

        # Warmup: max(252, 180+60) + 20 = 272
        self.set_warm_up(272, Resolution.DAILY)

        # Production hardening — shared across all Conquest projects.
        try:
            from conquest.production import harden
            harden(self)
        except Exception as e:
            self.log(f"[cf] production hardening skipped: {e}")

        # Schedules — single daily handler + monthly rebalance
        self.schedule.on(
            self.date_rules.month_start("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self._rebalance_cf,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 5),
            self._cf_daily_check,
        )

    def _active_crypto_symbol(self, t):
        """Date-gated crypto chain.
            t < 2013-09-25 → BIL (placeholder, GBTC inception)
            2013-09-25 ≤ t < 2021-10-19 → GBTC
            2021-10-19 ≤ t < 2024-01-11 → BITO
            2024-01-11 ≤ t → IBIT
        """
        if t < self.GBTC_INCEPTION:
            return self.bil_sym
        if t < self.BITO_INCEPTION:
            return self.crypto_syms["GBTC"]
        if t < self.IBIT_INCEPTION:
            return self.crypto_syms["BITO"]
        return self.crypto_syms["IBIT"]

    def _cf_daily_check(self):
        if self.is_warming_up:
            return
        dirty_cstab = self.cstab.update_daily()
        dirty_cgrowth = self.cgrowth.update_daily()
        if dirty_cstab or dirty_cgrowth:
            self._rebalance_cf()

    def _stag_tilt_reentry_active(self) -> bool:
        """V-recovery override for the Stagflation tilt.

        When VIX < STAG_TILT_REENTRY_VIX_MAX AND SPY 20d ROC > STAG_TILT_REENTRY_ROC_MIN,
        skip the tilt and use base sleeve weights even if regime is still Stagflation.
        Mirrors cstability's REENTRY rule. Catches V-recoveries (March 2009 pattern)
        that the regime classifier is too slow to detect.

        Returns False when:
        - The override is disabled
        - VIX feed is missing or VIX >= threshold
        - SPY ROC indicator is not ready or ROC < threshold
        """
        if not self.STAG_TILT_REENTRY_ENABLED:
            return False
        vix = self.cstab._current_vix_spot()
        if vix is None or vix >= self.STAG_TILT_REENTRY_VIX_MAX:
            return False
        if self.cstab.spy_roc is None or not self.cstab.spy_roc.is_ready:
            return False
        roc = float(self.cstab.spy_roc.current.value)
        if roc < self.STAG_TILT_REENTRY_ROC_MIN:
            return False
        return True

    def _rebalance_cf(self):
        if self.is_warming_up:
            return

        # Stagflation tilt: when enabled AND cstab's current regime is Stagflation,
        # use STAG_W_* weights (skip cgrowth + crypto, load GLD + BIL). All other
        # regimes use SLEEVE_W_* base weights (LIVE behavior).
        # V-recovery override: if SPY ROC > threshold AND VIX < threshold, skip
        # the tilt (catches March-2009-style V-recoveries that the regime
        # classifier mistimes due to GDP/CPI YoY data lag).
        reentry_override = self._stag_tilt_reentry_active()
        in_stag_tilt = (
            self.STAGFLATION_TILT_ENABLED
            and self.cstab._current_regime() == "Stagflation"
            and not reentry_override
        )
        if in_stag_tilt:
            w_cgrowth, w_cstab, w_crypto, w_gld, w_bil = (
                self.STAG_W_CGROWTH, self.STAG_W_CSTAB, self.STAG_W_CRYPTO,
                self.STAG_W_GLD, self.STAG_W_BIL,
            )
        else:
            w_cgrowth, w_cstab, w_crypto, w_gld, w_bil = (
                self.SLEEVE_W_CGROWTH, self.SLEEVE_W_CSTAB, self.SLEEVE_W_CRYPTO,
                self.SLEEVE_W_GLD, self.SLEEVE_W_BIL,
            )

        combined = {}

        # cstability sleeve
        if w_cstab > 0:
            cstab_w = self.cstab.target_weights()
            for sym, w in cstab_w.items():
                combined[sym] = combined.get(sym, 0.0) + w_cstab * w

        # cgrowth sleeve
        if w_cgrowth > 0:
            cgrowth_w = self.cgrowth.target_weights()
            for sym, w in cgrowth_w.items():
                combined[sym] = combined.get(sym, 0.0) + w_cgrowth * w

        # crypto sleeve (active ticker per date)
        crypto_sym = self._active_crypto_symbol(self.time)
        if w_crypto > 0:
            combined[crypto_sym] = combined.get(crypto_sym, 0.0) + w_crypto

        # static GLD
        if w_gld > 0:
            combined[self.gld_sym] = combined.get(self.gld_sym, 0.0) + w_gld

        # static BIL
        if w_bil > 0:
            combined[self.bil_sym] = combined.get(self.bil_sym, 0.0) + w_bil

        # Drop near-zero (avoid 0% set_holdings churn)
        combined = {sym: w for sym, w in combined.items() if abs(w) > 1e-4}

        # Gross-exposure clamp: 1.0× cap is a hard constraint.
        gross = sum(combined.values())
        if gross > 1.001:
            self.log(f"[cf] WARN: gross={gross:.4f} > 1.001; clamping to 1.0")
            scale = 1.0 / gross
            combined = {sym: w * scale for sym, w in combined.items()}
            gross = 1.0

        # Liquidate currently-held symbols not in combined
        keep_set = set(combined.keys())
        for sym in self.all_symbols:
            if sym not in keep_set and self.portfolio[sym].invested:
                self.liquidate(sym)

        # Set holdings
        for sym, w in combined.items():
            self.set_holdings(sym, w)

        crypto_label = next(
            (t for t, s in self.crypto_syms.items() if s == crypto_sym),
            "BIL" if crypto_sym == self.bil_sym else "?"
        )
        if in_stag_tilt:
            tilt_tag = "STAG-TILT"
        elif reentry_override and self.cstab._current_regime() == "Stagflation":
            tilt_tag = "STAG-TILT-OVERRIDE(V-recovery)"
        else:
            tilt_tag = "base"
        self.log(
            f"[cf] {self.time:%Y-%m-%d}: rebalance gross={gross:.2%} mode={tilt_tag} "
            f"cstab(vote={self.cstab.vote_count}/4 layer1={self.cstab.in_layer1_cash}) "
            f"cgrowth(riskoff={self.cgrowth.risk_off} crisis="
            f"{self.cstab.vote_count >= self.cgrowth.CRISIS_GATE_THRESHOLD}) "
            f"crypto={crypto_label} n={len(combined)}"
        )

    def on_data(self, data):
        # All decisions in scheduled monthly rebalance + daily edge handler.
        pass
