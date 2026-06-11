"""Build storage/conquest/universe/acwx_top.csv from a curated ADR list.

This is the source-of-truth for the P1 international momentum universe
(`cgrowth_international` sleeve). ~38 high-liquidity ADRs from MSCI ACWI
ex-US, curated for full or near-full 2008-2026 PIT coverage.

Post-2008 IPOs have explicit inception dates; CgrowthInternationalSleeve
honors these per-ticker gates so QC's pre-inception $0 bars don't inflate
phantom alpha (mirrors VoltgtSleeve.INCEPTION_DATES pattern).

Universe known biases (see BIAS_LEDGER.md):
- Survivorship: curated list of present-day-listed ADRs. Names that
  delisted between 2008-2026 are excluded. Comparable in magnitude to
  sp500.csv before sp500_union_2008_2024.csv landed.
- Geographic concentration: Asia (~40%), Europe (~30%), LatAm (~20%),
  India (~10%) by ticker count.
- Sector concentration: Financials and Information Technology dominate;
  partially mitigated by CgrowthInternationalSleeve's 30% sector cap.
- Ticker rename risk: SHEL (Royal Dutch Shell rebrand 2022) and TTE
  (TotalEnergies 2021) excluded due to QC continuous-data uncertainty.

Run: python scripts/build_acwx_top_universe.py
Then push to QC: lean cloud object-store set \\
    "conquest/universe/acwx_top.csv" \\
    "storage/conquest/universe/acwx_top.csv"
"""
import csv
from pathlib import Path

# (ticker, sector, inception_iso_or_empty)
ADRS = [
    ("ASML", "Information Technology", ""),
    ("TSM",  "Information Technology", ""),
    ("SAP",  "Information Technology", ""),
    ("INFY", "Information Technology", ""),
    ("WIT",  "Information Technology", ""),
    ("NVO",  "Health Care", ""),
    ("AZN",  "Health Care", ""),
    ("SNY",  "Health Care", ""),
    ("NVS",  "Health Care", ""),
    ("GSK",  "Health Care", ""),
    ("TM",   "Consumer Discretionary", ""),
    ("HMC",  "Consumer Discretionary", ""),
    ("SONY", "Consumer Discretionary", ""),
    ("DEO",  "Consumer Staples", ""),
    ("UL",   "Consumer Staples", ""),
    ("BUD",  "Consumer Staples", "2009-07-13"),
    ("TEF",  "Communication Services", ""),
    ("VOD",  "Communication Services", ""),
    ("BCS",  "Financials", ""),
    ("HSBC", "Financials", ""),
    ("ING",  "Financials", ""),
    ("SAN",  "Financials", ""),
    ("UBS",  "Financials", ""),
    ("MUFG", "Financials", ""),
    ("IBN",  "Financials", ""),
    ("HDB",  "Financials", ""),
    ("ITUB", "Financials", ""),
    ("RIO",  "Materials", ""),
    ("BHP",  "Materials", ""),
    ("VALE", "Materials", ""),
    ("PBR",  "Energy", ""),
    ("E",    "Energy", ""),
    ("BP",   "Energy", ""),
    ("MELI", "Consumer Discretionary", "2007-08-10"),
    ("BABA", "Consumer Discretionary", "2014-09-19"),
    ("JD",   "Consumer Discretionary", "2014-05-22"),
    ("SHOP", "Information Technology", "2015-05-21"),
    ("SE",   "Communication Services", "2017-10-20"),
    ("BIDU", "Communication Services", "2005-08-05"),
]


def main():
    out = Path(__file__).resolve().parent.parent / "storage" / "conquest" / "universe" / "acwx_top.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "sector", "inception"])
        for row in ADRS:
            w.writerow(row)
    print(f"wrote {len(ADRS)} ADRs to {out}")


if __name__ == "__main__":
    main()
