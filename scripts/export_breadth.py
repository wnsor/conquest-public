"""S&P-500 market breadth (% of members above their 200-day SMA) — the BULL
options-gate signal.

The conquest_options BULL book (`leaps_coreai_cheapiv` + PREMIUM_CAP_MODE=gate)
deploys cheap-IV LEAPS calls ONLY when the broad market is in a confirmed
uptrend (breadth > 0.50) and sits in CASH otherwise. That gate is what lifted
the book from 2.5% CAGR / -78% DD (always-on) to 37.2% / -54% (2017-2026). The
gate reads `conquest/leading/confidence.csv` from the QC Object Store, whose
`leading_confidence` column IS this breadth series (see compute_leading_confidence.py).

WHY THIS SCRIPT EXISTS: the breadth source `storage/conquest/breadth/sp500_above_200d_ma.csv`
was originally built by a one-off scratch script and left FROZEN at 2025-12-30.
A frozen gate signal is the single worst failure mode for this model — a stale
breadth read makes the live gate decorative (it can't tell bull from bear), so
the book either rides the full -78% always-on drawdown or sits in cash forever.
This script makes the signal auto-refreshable so it can be wired into the daily/
weekly data automation (scripts/data/refresh_all.py).

DATA SOURCE: yfinance member closes. This matches the established repo pattern
for signal inputs (screen_options_universe.py already uses yfinance in a refresh
handler); the "no yfinance" rule in CLAUDE.md governs PRIMARY backtest curves,
not refreshed signal CSVs. Fails CLOSED: if too few members download, the script
exits non-zero so the refresh handler marks it failed and the monitor flags the
staleness — the algo's fail-safe gate (premium_governor) then parks in cash.

METHOD (matches the frozen series' definition):
  breadth(t) = count(members with close(t) > SMA200(t)) / count(members priced at t)

DEFAULT mode EXTENDS the frozen CSV: existing rows (<= last frozen date) are kept
verbatim; only newer dates are appended. The seam vs the frozen series is printed
for validation (current-membership breadth ~ then-membership breadth within a few
pp; immaterial at the 0.50 gate threshold since breadth is rarely near 0.50).
`--full` regenerates the whole computable window from scratch (current members).

Usage:
  python scripts/export_breadth.py                 # extend frozen CSV with new dates
  python scripts/export_breadth.py --full          # regenerate from scratch (~2y)
  python scripts/export_breadth.py --lookback 3y   # longer download window
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "storage" / "conquest"
MEMBERS_CSV = S / "universe" / "sp500.csv"

# Two breadth universes, identical %>200DMA method:
#   sp500   — the whole market → confidence.csv (the existing BULL gate signal).
#   ai_tech — the AI/tech complex (S&P Information Technology + Communication
#             Services, ~96 names incl. 7/8 core-AI bull names) → ai_breadth.csv.
#             Lets a DUAL gate catch a narrow AI/tech bust where broad S&P breadth
#             still holds > 0.50 (the SPY-only gate's documented blind spot). This
#             is a TRUE 200DMA breadth — NOT the 252d-high proxy that ratcheted and
#             sank the prior ai_gate (see project_options_leaps_mu_arc "AI-UNIVERSE GATE").
# Each universe writes a source breadth CSV + an algo-facing signal CSV whose FIRST
# value column is what main.py reads (it parses date + column[1] only).
UNIVERSES = {
    "sp500": {
        "sectors": None,
        "source_csv": S / "breadth" / "sp500_above_200d_ma.csv",
        "signal_csv": S / "leading" / "confidence.csv",
        "signal_col": "leading_confidence",
    },
    "ai_tech": {
        "sectors": ["Information Technology", "Communication Services"],
        "source_csv": S / "breadth" / "ai_tech_above_200d_ma.csv",
        "signal_csv": S / "leading" / "ai_breadth.csv",
        "signal_col": "ai_breadth",
    },
}

SMA_WINDOW = 200          # 200-day SMA = the gate's trend definition
MIN_MEMBER_FRACTION = 0.80  # fail CLOSED if < 80% of members download

# Fallback large-cap S&P proxy if the membership CSV is missing (keeps the script
# runnable in a bare CI env; the real run reads the refreshed sp500.csv).
_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "V",
    "LLY", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV", "MRK",
]


def load_members(sectors: list | None = None) -> list[str]:
    """S&P tickers, optionally filtered to the given GICS sectors (for ai_tech)."""
    if MEMBERS_CSV.exists():
        df = pd.read_csv(MEMBERS_CSV)
        if sectors is not None and "sector" in df.columns:
            df = df[df["sector"].isin(sectors)]
        tics = sorted({str(t).strip().upper() for t in df["ticker"] if str(t).strip()})
        if len(tics) >= 20:
            return tics
        print(f"WARN: {MEMBERS_CSV.name} has only {len(tics)} tickers (sectors={sectors}); using fallback.",
              file=sys.stderr)
    else:
        print(f"WARN: {MEMBERS_CSV} missing; using {len(_FALLBACK)}-name fallback.",
              file=sys.stderr)
    return list(_FALLBACK)


def fetch_closes(tickers: list[str], lookback: str) -> pd.DataFrame:
    """Daily adjusted closes (rows=dates, cols=tickers). Drops empty columns."""
    import yfinance as yf
    # yfinance handles batching; auto_adjust=True → split/div-adjusted close.
    raw = yf.download(tickers, period=lookback, interval="1d",
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or len(raw) == 0:
        print("ERROR: yfinance returned no data.", file=sys.stderr)
        sys.exit(1)
    # MultiIndex columns ('Close', TICKER) for multi-ticker; flat for single.
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw[["Close"]].copy()
        closes.columns = tickers[:1]
    closes = closes.dropna(axis=1, how="all")
    got = closes.shape[1]
    frac = got / max(1, len(tickers))
    print(f"  fetched {got}/{len(tickers)} members ({frac:.0%}) over {lookback}")
    if frac < MIN_MEMBER_FRACTION:
        print(f"ERROR: only {frac:.0%} of members downloaded (< {MIN_MEMBER_FRACTION:.0%} "
              f"floor) — FAILING CLOSED so the refresh marks this stale, not silently "
              f"writing a biased partial-universe breadth.", file=sys.stderr)
        sys.exit(1)
    return closes


def compute_breadth(closes: pd.DataFrame) -> pd.DataFrame:
    """% of members with close > their trailing 200d SMA, per date."""
    sma = closes.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    above = closes > sma                      # bool; NaN (warmup / missing) → False
    valid = sma.notna() & closes.notna()      # member counts toward denom only when priced + warmed
    num = above.where(valid).sum(axis=1)
    den = valid.sum(axis=1)
    breadth = num / den.where(den > 0)   # zero denom → NaN (avoids int ZeroDivisionError on warmup)
    out = breadth.dropna().rename("pct_above_200d_ma").to_frame()
    out.index.name = "date"
    # keep only fully-warmed dates (>= SMA_WINDOW members typically priced)
    out = out[den.reindex(out.index) >= 0.5 * closes.shape[1]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500", choices=list(UNIVERSES.keys()),
                    help="sp500 (default → confidence.csv) or ai_tech (→ ai_breadth.csv)")
    ap.add_argument("--full", action="store_true",
                    help="regenerate the whole computable window (don't merge with existing)")
    ap.add_argument("--lookback", default="2y",
                    help="yfinance download period (default 2y → 200d SMA valid ~1.5y back)")
    args = ap.parse_args()

    cfg = UNIVERSES[args.universe]
    out_csv, signal_csv, signal_col = cfg["source_csv"], cfg["signal_csv"], cfg["signal_col"]

    tickers = load_members(cfg["sectors"])
    print(f"  universe={args.universe}: {len(tickers)} members")
    closes = fetch_closes(tickers, args.lookback)
    fresh = compute_breadth(closes)
    if fresh.empty:
        print("ERROR: computed breadth is empty (insufficient history).", file=sys.stderr)
        return 1
    fresh.index = pd.to_datetime(fresh.index).normalize()
    print(f"  computed breadth: {len(fresh)} dates, "
          f"{fresh.index[0].date()} → {fresh.index[-1].date()}, "
          f"latest = {fresh['pct_above_200d_ma'].iloc[-1]:.4f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.full or not out_csv.exists():
        merged = fresh
        print("  mode: FULL regenerate" if args.full else "  mode: fresh (no existing CSV)")
    else:
        existing = pd.read_csv(out_csv, parse_dates=["date"]).set_index("date").sort_index()
        seam_end = existing.index.max()
        # validation: where the two overlap, how far apart are they?
        overlap = fresh.index.intersection(existing.index)
        if len(overlap) >= 10:
            diff = (fresh.loc[overlap, "pct_above_200d_ma"]
                    - existing.loc[overlap, "pct_above_200d_ma"]).abs()
            print(f"  SEAM CHECK: {len(overlap)} overlapping dates, "
                  f"mean|Δ|={diff.mean():.4f}, max|Δ|={diff.max():.4f} "
                  f"(current-membership vs existing; immaterial at the 0.50 gate)")
        # extend: keep existing verbatim, append only NEW dates
        new_rows = fresh[fresh.index > seam_end]
        merged = pd.concat([existing, new_rows]).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        print(f"  mode: EXTEND — existing through {seam_end.date()}, "
              f"appended {len(new_rows)} new dates "
              f"({'→ ' + str(new_rows.index[-1].date()) if len(new_rows) else 'none'})")

    merged.index.name = "date"
    merged["pct_above_200d_ma"].round(6).to_csv(out_csv)
    print(f"  → {out_csv.relative_to(ROOT)}  "
          f"({len(merged)} rows, {merged.index.min().date()} → {merged.index.max().date()})")

    # Algo-facing gate signal (date,<signal_col>); the value IS the breadth (gate
    # compares to 0.50). main.py reads date+column[1] only; monitor reads vcol=1.
    signal_csv.parent.mkdir(parents=True, exist_ok=True)
    sig = merged["pct_above_200d_ma"].round(6).rename(signal_col)
    sig.to_csv(signal_csv)
    print(f"  → {signal_csv.relative_to(ROOT)}  (gate signal; latest "
          f"{signal_col}={sig.iloc[-1]:.4f} → {'DEPLOY' if sig.iloc[-1] > 0.50 else 'CASH'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
