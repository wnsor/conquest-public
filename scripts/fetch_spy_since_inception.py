"""Fetch SPY daily prices since inception (1993-01-22) and rebase to $25k seed.

Output: storage/conquest/lean/spy_since_inception.json — same schema as the
other Lean equity JSON sidecars (fund / version_label / engine / period /
start_equity / values[{date, equity}]). The webapp loads this when the
"SPY since 1993" toggle is on to extend the chart x-axis back to SPY's
ETF inception.

Note: this is a YAHOO total-return curve (adjusted close), not a Lean
cloud backtest. We're using it purely as a long-history reference series
to visualize the deeper context vs the 2008-start canonical backtest
window. Quality should be sufficient for that — Yahoo adj-close is
total-return (splits + dividends) and matches Lean's daily within
~0.1%/yr drift.
"""
from __future__ import annotations
import json
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "storage" / "conquest" / "lean" / "spy_since_inception.json"

SEED = 25_000.0
TICKER = "SPY"
START = "1993-01-22"   # SPY inception date

def main() -> None:
    print(f"Fetching {TICKER} daily from {START} ...")
    df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise SystemExit("yfinance returned no data")
    # auto_adjust=True yields a single "Close" column that's already total-return adjusted.
    close = df["Close"].dropna()
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    start_px = float(close.iloc[0])
    values = []
    for ts, px in close.items():
        values.append({
            "date":   ts.strftime("%Y-%m-%d"),
            "equity": float(px) / start_px * SEED,
        })
    payload = {
        "fund": "SPY",
        "version_label": f"SPY since inception ({START}), $25k seed (Yahoo adj-close)",
        "engine": "yahoo (total-return adj-close)",
        "source_backtest": "yahoo-finance",
        "period": {"start": values[0]["date"], "end": values[-1]["date"]},
        "start_equity": SEED,
        "values": values,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(values)} daily pts, {values[0]['date']} -> {values[-1]['date']})")
    print(f"  start ${values[0]['equity']:,.0f} -> end ${values[-1]['equity']:,.0f}")

if __name__ == "__main__":
    main()
