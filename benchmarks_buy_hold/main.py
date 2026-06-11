# region imports
from AlgorithmImports import *
# endregion


class BenchmarksBuyHold(QCAlgorithm):
    """
    Lean-native buy-hold benchmarks.

    Single backtest that subscribes to QQQ / IWM / EFA / GLD daily and
    maintains an in-memory $25k tracker for each. No real trades happen —
    each tracker is just NAV * (1 + daily_return) on every bar. The four
    trackers are emitted as series under the custom chart "Buy-Hold
    Benchmarks", which scripts/fetch_cloud_backtest.py extracts and saves
    to storage/conquest/lean/{qqq,iwm,efa,gld}_buyhold_lean.json.

    The webapp's chart toggles for QQQ / IWM / EFA / GLD auto-enable when
    those JSONs appear (the loader in webapp/app.js degrades gracefully
    when they're missing).

    Why a single Lean project for 4 trackers (vs 4 separate projects):
    one cloud backtest, one push, one fetch — and the QC compute spend
    is the same as one backtest of any single ticker.

    Backtest window 2008-01-01 → 2026-02-04 matches the v11 LIVE pin
    window so the buy-hold curves can overlay cstability / cgrowth / CF
    cleanly on the webapp performance chart.
    """

    # Buy-hold trackers. Includes original benchmarks (QQQ/IWM/EFA/GLD) plus
    # high-AUM sector/asset ETFs the webapp surfaces for comparison: VTI
    # (total US stock market), VGT/XLK (tech sector), SMH (semis), VWO
    # (emerging markets), VNQ (US REITs), TLT (long Treasury).
    TICKERS = (
        # Curated set most relevant to Surge + ctactical comparison:
        "QQQ",   # Nasdaq-100 — growth/tech baseline
        "IWM",   # Russell 2000 — small-cap (Surge holds TNA = 3x small-cap)
        "GLD",   # gold — ctactical's 10% sleeve + classic diversifier
        "TLT",   # 20yr+ Treasury — the defensive / long-bond peer
        "EFA",   # intl developed — coverage
        "TQQQ",  # 3x Nasdaq — Surge's flagship leveraged peer (naive 3x buy-hold: huge upside + brutal DD/decay)
        "UVXY",  # 1.5x VIX-futures ETP — Surge's overlay instrument (contango decay + backwardation spikes)
    )
    SEED_PER_TICKER = 25_000.0
    CHART_NAME = "Buy-Hold Benchmarks"

    def initialize(self):
        start_year = int(self.get_parameter("BACKTEST_START_YEAR") or 2008)
        end_year = int(self.get_parameter("BACKTEST_END_YEAR") or 2026)
        self.set_start_date(start_year, 1, 1)
        if end_year == 2026:
            self.set_end_date(2026, 2, 4)  # matches v11 LIVE pin window
        else:
            self.set_end_date(end_year, 12, 31)
        # Scratch capital — no trades will execute. SetCash is required by Lean
        # but the algorithm never calls SetHoldings or PlaceOrder.
        self.set_cash(100_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE)

        self.symbols = {}
        self.trackers = {}
        self.last_prices = {}
        for t in self.TICKERS:
            sym = self.add_equity(t, Resolution.DAILY).symbol
            self.symbols[t] = sym
            self.trackers[t] = self.SEED_PER_TICKER
            self.last_prices[t] = None

    def on_data(self, data: Slice) -> None:
        for t in self.TICKERS:
            sym = self.symbols[t]
            if sym not in data.bars:
                continue
            price = float(data.bars[sym].close)
            if price <= 0:
                continue
            if self.last_prices[t] is not None and self.last_prices[t] > 0:
                ret = price / self.last_prices[t] - 1.0
                self.trackers[t] *= (1.0 + ret)
            self.last_prices[t] = price
            self.plot(self.CHART_NAME, t, self.trackers[t])
