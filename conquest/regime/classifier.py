"""Bridgewater-style 4-quadrant macro regime classifier.

Inputs
------
- gdp_yoy: GDP year-over-year growth (%), quarterly
- cpi_yoy: CPI year-over-year inflation (%), monthly

Output
------
A monthly DataFrame with z-scores, regime label, and a confidence score.

Quadrants (axes: growth vs trend, inflation vs trend)
-----------------------------------------------------
- Inflation:    growth > trend AND inflation > trend
- Disinflation: growth > trend AND inflation < trend
- Stagflation:  growth < trend AND inflation > trend
- Deflation:    growth < trend AND inflation < trend

Mechanics
---------
- Z-score each YoY series over a rolling `lookback_months` window (default 60).
- Hysteresis: only switch sides of zero when `|z|` exceeds `hysteresis`. Default
  is 0 (any sign change flips), but raising it to 0.25 or 0.5 dampens noise.
- Min-dwell: a candidate new regime must persist for `min_dwell_months`
  consecutive months before being reported; otherwise the previous stable
  regime sticks.

This is a v1, rule-based, linear-models-first implementation. Phase 5+ may swap
in HMM / clustering once the linear baseline's failure modes are characterised.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REGIME_LABELS: list[str] = ["Inflation", "Disinflation", "Stagflation", "Deflation"]


def _quadrant(g_above_trend: bool, c_above_trend: bool) -> str:
    if g_above_trend and c_above_trend:
        return "Inflation"
    if g_above_trend and not c_above_trend:
        return "Disinflation"
    if not g_above_trend and c_above_trend:
        return "Stagflation"
    return "Deflation"


@dataclass
class RegimeClassifier:
    lookback_months: int = 60
    hysteresis: float = 0.0           # z-units around zero where the regime "sticks"
    min_dwell_months: int = 2

    def classify(self, gdp_yoy: pd.Series, cpi_yoy: pd.Series) -> pd.DataFrame:
        """Classify regime at month-end frequency."""
        gdp_yoy = self._ensure_dt_index(gdp_yoy)
        cpi_yoy = self._ensure_dt_index(cpi_yoy)

        start = max(gdp_yoy.index.min(), cpi_yoy.index.min())
        # END FIX 2026-05-24: was `min(gdp.max, cpi.max)` capped by GDP's
        # start-of-quarter date convention (FRED returns 2026-01-01 for Q1 2026),
        # which combined with `freq="ME"` produced a last month-end of 2025-12-31
        # — cutting off Q1/Q2 2026 regime classification by 4 months. CPI is
        # monthly (always fresher); use its max rounded UP to month-end. GDP
        # forward-fills via reindex+ffill below, so the Q1 2026 reading applies
        # through Q2 release (~July 2026).
        end = cpi_yoy.index.max() + pd.offsets.MonthEnd(0)
        idx = pd.date_range(start=start, end=end, freq="ME")

        g = gdp_yoy.resample("ME").last().reindex(idx).ffill()
        c = cpi_yoy.resample("ME").last().reindex(idx).ffill()

        gz = self._rolling_z(g)
        cz = self._rolling_z(c)

        # Raw quadrant assignment with hysteresis around zero
        regimes_raw: list[str | None] = []
        prev_g_above: bool | None = None
        prev_c_above: bool | None = None
        for gv, cv in zip(gz, cz):
            if pd.isna(gv) or pd.isna(cv):
                regimes_raw.append(None)
                prev_g_above = None
                prev_c_above = None
                continue
            if prev_g_above is None or abs(gv) > self.hysteresis:
                g_above = bool(gv > 0)
            else:
                g_above = prev_g_above
            if prev_c_above is None or abs(cv) > self.hysteresis:
                c_above = bool(cv > 0)
            else:
                c_above = prev_c_above
            regimes_raw.append(_quadrant(g_above, c_above))
            prev_g_above, prev_c_above = g_above, c_above

        regime = pd.Series(regimes_raw, index=idx)
        if self.min_dwell_months > 1:
            regime = self._apply_min_dwell(regime, self.min_dwell_months)

        confidence = (gz.abs() + cz.abs()) / 2.0

        return pd.DataFrame({
            "gdp_yoy": g,
            "cpi_yoy": c,
            "gdp_yoy_z": gz,
            "cpi_yoy_z": cz,
            "regime": regime,
            "confidence": confidence,
        }, index=idx)

    def classify_to_daily(
        self,
        gdp_yoy: pd.Series,
        cpi_yoy: pd.Series,
        daily_index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Like `classify`, but ffilled to a business-day daily index — the
        format the Lean Algorithms read from the Object Store."""
        monthly = self.classify(gdp_yoy, cpi_yoy)
        if daily_index is None:
            daily_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="B")
        return monthly.reindex(daily_index, method="ffill")

    # ---------- internals ----------

    @staticmethod
    def _ensure_dt_index(s: pd.Series) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex):
            return s
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s

    def _rolling_z(self, s: pd.Series) -> pd.Series:
        mean = s.rolling(self.lookback_months, min_periods=12).mean()
        std = s.rolling(self.lookback_months, min_periods=12).std()
        return (s - mean) / std

    @staticmethod
    def _apply_min_dwell(regime: pd.Series, min_dwell: int) -> pd.Series:
        """A new regime is reported only after it has persisted for `min_dwell`
        consecutive months; otherwise the previous stable regime is kept."""
        out: list[str | None] = []
        current: str | None = None
        tentative: str | None = None
        run = 0
        for r in regime:
            if r is None:
                out.append(current)
                tentative, run = None, 0
                continue
            if r == current:
                out.append(current)
                tentative, run = None, 0
                continue
            if tentative == r:
                run += 1
            else:
                tentative, run = r, 1
            if run >= min_dwell:
                current = r
                tentative, run = None, 0
            out.append(current)
        return pd.Series(out, index=regime.index)
