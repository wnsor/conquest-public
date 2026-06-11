"""Convert cached benchmark price parquets into a single JSON the webapp can read.

The webapp's chart needs benchmark series (SPY, QQQ, IWM, EFA, GLD) normalized
to $25k seed equity. Browsers can't parse parquet directly, so we pre-export.

Output:
    storage/conquest/lean/benchmarks.json

Schema:
    {
      "seed_equity": 25000,
      "as_of": "2026-05-04",
      "series": {
        "SPY": {"label": "Buy-and-Hold SPY", "values": [{"date":"2007-01-03","equity":25000.0}, ...]},
        ...
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = ROOT / "data" / "alternative" / "conquest" / "raw" / "prices"
OUT_PATH = ROOT / "storage" / "conquest" / "lean" / "benchmarks.json"
SEED = 25_000.0

# (ticker, label, primary parquet, optional fallback parquet for older history)
SOURCES = [
    ("SPY", "Buy-and-Hold SPY (S&P 500)", "spy_close.parquet", None),
    ("QQQ", "Buy-and-Hold QQQ (NASDAQ 100)", "qqq_close.parquet", None),
    ("IWM", "Buy-and-Hold IWM (Russell 2000)", "iwm_close.parquet", None),
    ("EFA", "Buy-and-Hold EFA (MSCI EAFE)", "efa_close.parquet", "etf_basket_daily.parquet"),
    ("GLD", "Buy-and-Hold GLD (Gold)", "gld_close.parquet", "etf_basket_daily.parquet"),
]


def load_close_series(ticker: str, primary: str, fallback: str | None) -> pd.Series:
    primary_path = PRICE_DIR / primary
    df = pd.read_parquet(primary_path)
    if ticker in df.columns:
        s = df[ticker]
    else:
        # _close.parquet files use the ticker as their single column
        s = df.iloc[:, 0]
    s = s.astype(float).sort_index()

    if fallback is not None:
        fb_path = PRICE_DIR / fallback
        if fb_path.exists():
            fb = pd.read_parquet(fb_path)
            if ticker in fb.columns:
                fb_s = fb[ticker].astype(float).sort_index()
                # Use fallback for dates BEFORE primary's start.
                primary_start = s.index.min()
                fb_pre = fb_s[fb_s.index < primary_start]
                if not fb_pre.empty:
                    s = pd.concat([fb_pre, s]).sort_index()
                    s = s[~s.index.duplicated(keep="last")]
    return s


def normalize_to_seed(close: pd.Series, seed: float) -> pd.Series:
    if close.empty:
        return close
    return seed * (close / close.iloc[0])


def main() -> None:
    out = {
        "seed_equity": SEED,
        "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "series": {},
    }
    for ticker, label, primary, fallback in SOURCES:
        try:
            close = load_close_series(ticker, primary, fallback)
        except FileNotFoundError as e:
            print(f"[skip] {ticker}: {e}")
            continue
        equity = normalize_to_seed(close, SEED)
        records = [
            {"date": d.strftime("%Y-%m-%d"), "equity": float(v)}
            for d, v in equity.items()
        ]
        out["series"][ticker] = {"label": label, "values": records}
        print(f"[ok] {ticker:4s} {label:35s} {records[0]['date']} -> {records[-1]['date']} "
              f"({len(records)} rows; end ${records[-1]['equity']:,.0f})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out) + "\n")
    print(f"\n[ok] wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
