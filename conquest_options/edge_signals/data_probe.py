"""Data-availability probe instrumentation.

Counts which context fields are populated at QC runtime so we know BEFORE
running expensive strategy BTs whether the data signals the gates depend
on actually exist at daily resolution. Surfaces as runtime stats with
prefixes:
  probe_g_<field> — global (non-per-ticker) field populate % per tick
  probe_t_<field> — per-ticker field populate % per (ticker × tick)
"""
from __future__ import annotations

from collections import defaultdict


# Global fields probed (read from StrategyContext top-level attrs).
GLOBAL_FIELDS = (
    "vix", "vix3m", "vix9d", "vix_term_ratio", "vix9d_vix_ratio",
    "term_regime", "gex_total", "gex_regime", "crisis_state",
    "pc_ratio_equity", "cstability_vote_count",
)

# Per-ticker dict fields (counted as #tickers with non-None values).
PER_TICKER_DICT_FIELDS = (
    "underlying_prices", "underlying_momentum_30d", "underlying_momentum_60d",
    "underlying_5ma_above_20ma", "underlying_drawdown_from_252d_high",
    "historical_vol_30d", "historical_vol_60d",
    "iv_rank", "iv_raw", "iv_hv_ratio", "implied_move_vs_realized",
    "skew_z", "days_until_next_earnings",
    "volume_spike", "insider_cluster_score",
    "news_propagation_5d", "short_interest_velocity", "insider_count_5d",
)

# Per-ticker set fields (counted as len of set).
PER_TICKER_SET_FIELDS = ("uoa_active", "earnings_within_5d")


class DataProbe:
    """Lightweight per-tick + per-(ticker, tick) populate-rate tracker."""

    def __init__(self):
        self.ticks_total = 0
        self.ticker_ticks_total = 0
        self.per_tick: dict[str, int] = defaultdict(int)
        self.per_ticker: dict[str, int] = defaultdict(int)

    def record(self, ctx, n_underlyings: int) -> None:
        self.ticks_total += 1
        self.ticker_ticks_total += n_underlyings
        for f in GLOBAL_FIELDS:
            if getattr(ctx, f, None) is not None:
                self.per_tick[f] += 1
        for f in PER_TICKER_DICT_FIELDS:
            d = getattr(ctx, f, None) or {}
            self.per_ticker[f] += sum(1 for v in d.values() if v is not None)
        for f in PER_TICKER_SET_FIELDS:
            s = getattr(ctx, f, None) or set()
            self.per_ticker[f] += len(s)

    def summary(self) -> dict[str, str]:
        """Return {stat_name: percent_string} for runtime stat emission."""
        out: dict[str, str] = {}
        ticks = max(1, self.ticks_total)
        ticker_ticks = max(1, self.ticker_ticks_total)
        out["probe_ticks_total"] = str(self.ticks_total)
        out["probe_ticker_ticks_total"] = str(self.ticker_ticks_total)
        for f, n in self.per_tick.items():
            out[f"probe_g_{f[:14]}"] = f"{100.0*n/ticks:.1f}%"
        for f, n in self.per_ticker.items():
            out[f"probe_t_{f[:14]}"] = f"{100.0*n/ticker_ticks:.1f}%"
        return out
