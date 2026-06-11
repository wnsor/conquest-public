"""Fetch CFTC Commitments of Traders (COT) weekly positioning data.

COT reports show how futures positions are distributed across trader
categories: Commercial (hedgers), Non-Commercial (hedge funds/speculators),
and Non-Reportable (small traders).

Key contracts we track:
  E-mini S&P 500    — equity index institutional positioning
  VIX futures        — vol regime expectations
  10-year Treasury   — rate direction conviction
  Eurodollar         — short rate expectations
  US Dollar Index    — DXY positioning

Output: storage/conquest/macro/cftc_cot_weekly.csv
Schema: date, contract, noncomm_net, comm_net, noncomm_long_pct, noncomm_short_pct

Source: CFTC.gov public CSVs (no auth needed). They publish "Disaggregated
Futures Only" data every Friday for the prior Tuesday.

Leading-indicator value:
  Net-long extreme (>+95th pct) for non-commercials in SPX/VIX = sell signal
  Net-short extreme (<5th pct) = buy signal
  Δ positioning week-over-week = leading 1-4 weeks
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "macro" / "cftc_cot_weekly.csv"

# Contract codes — see CFTC.gov for full list
TRACKED_CONTRACTS = {
    "13874A": "es_emini_sp500",      # E-mini S&P 500
    "1170E1": "vx_vix_futures",      # VIX futures
    "043602": "ty_10yr_treasury",    # 10-Year T-Note
    "132741": "ed_eurodollar",       # Eurodollar (legacy SOFR is replacing)
    "098662": "dx_usd_index",        # US Dollar Index
}

COT_BASE = "https://www.cftc.gov/files/dea/history"


def _fetch_zip_csv(url: str) -> pd.DataFrame:
    """Download zip + read the first .txt as CSV."""
    print(f"  fetching {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".txt")]
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as f:
            return pd.read_csv(f, low_memory=False)


def fetch_year(year: int) -> pd.DataFrame:
    """Download + parse one year's worth of CFTC COT data.

    2026-05-27 REWRITE v2: tracked contracts live in TWO CFTC reports:
       fut_disagg_txt_YYYY.zip  — commodity futures (oil, gold, corn, ...)
       fut_fin_txt_YYYY.zip     — financial futures (S&P, VIX, Treasuries, USD)
    All our targets (E-mini S&P, VIX, 10y Treasury, USD Index) are FINANCIAL
    so we now fetch from fut_fin. We also fetch disagg in case commodity
    tracking is added later. Concat both. Original code only hit the
    commodity URL — that's why 'no data fetched' happened.
    """
    parts = []
    for kind in ("fut_fin", "fut_disagg"):
        url = f"{COT_BASE}/{kind}_txt_{year}.zip"
        try:
            df = _fetch_zip_csv(url)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  WARN {kind}_{year}: {e}")
    if not parts:
        return pd.DataFrame()
    # Schemas overlap (both have CFTC_Contract_Market_Code, position cols, etc.)
    # so we can concat — extra columns become NaN in the union.
    return pd.concat(parts, ignore_index=True)


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract just the tracked contracts + standardize columns.

    2026-05-27 REWRITE v3: handle BOTH CFTC report schemas:
       fin_fut (financial futures — VIX, S&P, Treasuries, USD):
         Lev_Money_*  (≈ non-commercial speculators)
         Asset_Mgr_*  (≈ commercial / long-only institutionals)
         Dealer_*     (sell-side market makers)
       fut_disagg (commodity futures — oil, gold, ...):
         M_Money_*    (≈ non-commercial)
         Prod_Merc_*  (≈ commercial hedgers)
         Swap_*
       Legacy fut (deprecated):
         Noncommercial_* / Commercial_*

    Output canonical schema: noncomm_long/short + comm_long/short + nets.
    """
    if raw.empty:
        return raw
    col_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in raw.columns}
    raw = raw.rename(columns=col_map)
    code_col = next((c for c in raw.columns if "contract_market_code" in c), None)
    if code_col is None:
        return pd.DataFrame()
    raw = raw[raw[code_col].astype(str).isin(TRACKED_CONTRACTS.keys())].copy()
    if raw.empty:
        return raw
    raw["contract"] = raw[code_col].astype(str).map(TRACKED_CONTRACTS)

    def col(prefix: str) -> str | None:
        return next((c for c in raw.columns if c.startswith(prefix)), None)

    # Try fin_fut schema first (financial — our primary targets)
    lm_long = col("lev_money_positions_long")
    lm_short = col("lev_money_positions_short")
    am_long = col("asset_mgr_positions_long")
    am_short = col("asset_mgr_positions_short")
    dl_long = col("dealer_positions_long")
    dl_short = col("dealer_positions_short")
    # Disaggregated (commodity)
    mm_long = col("m_money_positions_long")
    mm_short = col("m_money_positions_short")
    pm_long = col("prod_merc_positions_long")
    pm_short = col("prod_merc_positions_short")
    sw_long = col("swap_positions_long") or col("swap__positions_long")
    sw_short = col("swap__positions_short") or col("swap_positions_short")
    # Legacy
    nc_long = col("noncommercial_positions_long")
    nc_short = col("noncommercial_positions_short")
    comm_long = col("commercial_positions_long")
    comm_short = col("commercial_positions_short")
    # Prefer the proper YYYY-MM-DD report_date column over the YYMMDD int form
    date_col = next((c for c in raw.columns if "report_date_as_yyyy_mm_dd" in c), None) \
        or next((c for c in raw.columns if "report_date" in c), None)

    zeros = pd.Series([0] * len(raw), index=raw.index)
    if lm_long and lm_short:
        # Financial-futures path
        noncomm_long_series = raw[lm_long]
        noncomm_short_series = raw[lm_short]
        comm_long_series = (raw[am_long] if am_long else zeros) + (raw[dl_long] if dl_long else zeros)
        comm_short_series = (raw[am_short] if am_short else zeros) + (raw[dl_short] if dl_short else zeros)
    elif mm_long and mm_short:
        # Disaggregated path
        noncomm_long_series = raw[mm_long]
        noncomm_short_series = raw[mm_short]
        comm_long_series = (raw[pm_long] if pm_long else zeros) + (raw[sw_long] if sw_long else zeros)
        comm_short_series = (raw[pm_short] if pm_short else zeros) + (raw[sw_short] if sw_short else zeros)
    elif nc_long and nc_short:
        noncomm_long_series = raw[nc_long]
        noncomm_short_series = raw[nc_short]
        comm_long_series = raw[comm_long] if comm_long else zeros
        comm_short_series = raw[comm_short] if comm_short else zeros
    else:
        print(f"  WARN: no recognized trader-class columns; "
              f"cols={list(raw.columns)[:15]}")
        return pd.DataFrame()

    # date parse — prefer ISO string column
    if date_col:
        dates = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        dates = pd.Series([pd.NaT] * len(raw), index=raw.index)

    out = pd.DataFrame({
        "date": dates.values,
        "contract": raw["contract"].values,
        "noncomm_long": pd.to_numeric(noncomm_long_series, errors="coerce").values,
        "noncomm_short": pd.to_numeric(noncomm_short_series, errors="coerce").values,
        "comm_long": pd.to_numeric(comm_long_series, errors="coerce").values,
        "comm_short": pd.to_numeric(comm_short_series, errors="coerce").values,
    })
    out["noncomm_net"] = out["noncomm_long"] - out["noncomm_short"]
    out["comm_net"] = out["comm_long"] - out["comm_short"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=pd.Timestamp.today().year)
    args = ap.parse_args()

    print(f"Fetching CFTC COT for years {args.start_year}-{args.end_year}")
    parts = []
    for y in range(args.start_year, args.end_year + 1):
        try:
            df = transform(fetch_year(y))
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  WARN {y}: {e}")
    if not parts:
        print("ERROR: no data fetched")
        return 1
    combined = pd.concat(parts, ignore_index=True).sort_values(["date", "contract"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(combined)} rows -> {OUT_CSV.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
