"""Ingest GDELT GKG v2 daily news sentiment + article volume per ticker.

GDELT v2 GKG (Global Knowledge Graph) files contain ticker-tagged articles with
GCAM sentiment scores. We download ONE GKG file per day (the 23:00 close-of-day
file) for the requested date range, filter for the configured tickers, and
emit a daily-aggregated CSV with two columns per ticker: mean tone and article
count.

Output:
    storage/conquest/sentiment/gdelt_daily.csv
        columns: date, <TICKER>_tone, <TICKER>_count, ... × N tickers
        values:  daily mean tone (-100..+100) and article count (int) per ticker

Universe (Tier1 Signal 1 — conquest_options A1 WSB consumers):
    16 single-stock tickers + 7 ETF basket entries (preserved from prior cspec
    consumer). Short tickers (MX, MU, PL, NOK) use full-company-name aliases
    only to avoid false positives on common English words ("Mexico", "Mutual",
    "platinum", "Nokia/NOK country code").

GDELT GKG file format:
    Tab-separated, columns include V2.1Tone (mean of 6 emotion scores, -100..+100),
    V2EnhancedThemes, V2EnhancedPersons, V2EnhancedOrganizations, V2.1AllNames,
    V2.1Counts, V2GCAM (1500+ score dimensions), and a SourceCommonName.

Usage:
    # Test run for a specific week (small sample, ~7 GKG downloads)
    python scripts/ingest_gdelt_sentiment.py --start 2024-12-01 --end 2024-12-07

    # Full ingestion (multi-day wall-clock; resumable)
    python scripts/ingest_gdelt_sentiment.py --start 2015-02-18 --end 2026-05-07

Notes
-----
* GDELT v2 starts 2015-02-18. Pre-2015 = no data; columns = 0 (neutral / no articles).
* GKG files are ~10MB compressed; ~365 files/year × 11 yr ≈ 40GB for full backfill.
* Script is incremental — checks existing CSV and only downloads missing dates.
* Sentiment per ticker = mean V2.1Tone across articles where any company-name
  alias appears in V2.1AllNames.
* Article count per ticker = number of articles where any alias matched.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile

# GDELT GKG rows can have multi-MB fields (V2EnhancedThemes, V2GCAM are
# long-flat-string blobs). Default Python csv field limit (131072 bytes)
# rejects them. Raise to maxsize — these rows ARE legitimately huge.
csv.field_size_limit(sys.maxsize)
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

GKG_BASE_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"

# Ticker → company-name aliases used for substring matching against V2.1AllNames.
# Two groups:
#   (a) 7 ETF basket entries — preserved from the original cspec_v20_news consumer
#   (b) 16 WSB single-stock entries (Tier1 Signal 1 for conquest_options A1)
# Conservative aliases for short tickers (MX, MU, PL, NOK, APP, AMD) — full
# canonical company names only, no bare-ticker substring matching.
TICKER_NAMES = {
    # --- ETF baskets (basket-level sentiment; broad-term matching) ---
    "SPY":  ["S&P 500", "S&P500", "Standard & Poor's 500", "SPDR S&P 500", "SPX index"],
    "QQQ":  ["Nasdaq", "NASDAQ 100", "Invesco QQQ"],
    "IWM":  ["Russell 2000", "small-cap"],
    "SOXX": ["semiconductor", "Philadelphia Semiconductor", "SOX index"],
    "DIA":  ["Dow Jones", "Dow 30", "Dow Industrial"],
    "GLD":  ["gold", "gold price", "SPDR Gold"],
    "TLT":  ["20-year Treasury", "long Treasury", "20+ year bond"],
    # --- WSB universe (Tier1 Signal 1) ---
    "CRDO": ["Credo Technology"],
    "MU":   ["Micron Technology"],            # NOT "Micron" (matches "Microsoft")
    "NBIS": ["Nebius Group"],
    "NOK":  ["Nokia Corporation", "Nokia Oyj", "Nokia"],
    "MX":   ["MagnaChip Semiconductor"],
    "PL":   ["Planet Labs"],
    "DRAM": ["Pono Capital", "DRAM ETF"],     # ambiguous — accept low recall
    "AVGO": ["Broadcom Inc", "Broadcom"],
    "AMD":  ["Advanced Micro Devices"],       # NOT "AMD" (matches "AMDocs")
    "PLTR": ["Palantir Technologies", "Palantir"],
    "MSTR": ["MicroStrategy"],
    "COIN": ["Coinbase"],
    "RKLB": ["Rocket Lab"],
    "IONQ": ["IonQ"],
    "APP":  ["AppLovin"],                     # NOT "APP" (any 3-letter app word)
    "SMCI": ["Super Micro Computer", "Supermicro"],
}

OUT_PATH = Path(__file__).parent.parent / "storage/conquest/sentiment/gdelt_daily.csv"


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def fetch_gkg_for_day(d: date, hour: int = 23, minute: int = 0):
    """Download one GKG file for the given date+time; returns DataFrame or None."""
    ts = d.strftime("%Y%m%d") + f"{hour:02d}{minute:02d}00"
    url = GKG_BASE_URL.format(ts=ts)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as f:
                text = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  fetch error {ts}: {e}", file=sys.stderr)
        return None
    return text


def parse_gkg_text(text: str) -> dict:
    """Parse GKG TSV; return dict ticker -> list of tone scores from matching articles.

    The article count per ticker is len(scores[ticker]); aggregation in
    process_day uses both the mean tone and the count.
    """
    scores = {t: [] for t in TICKER_NAMES}
    if not text:
        return scores
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 16:
            continue
        # GKG v2 columns: 0=GKGRECORDID, 1=DATE, 2=SourceCollectionIdentifier,
        # 3=SourceCommonName, 4=DocumentIdentifier, 5=Counts, 6=V2Counts,
        # 7=Themes, 8=V2Themes, 9=Locations, 10=V2Locations, 11=Persons,
        # 12=V2Persons, 13=Organizations, 14=V2Organizations,
        # 15=V2.1Tone (mean,positive,negative,polarity,activity,grouprefdensity,
        #              wordcount), ... up to V2.1AllNames=23
        try:
            tone_field = row[15]
            if not tone_field:
                continue
            tone_parts = tone_field.split(",")
            mean_tone = float(tone_parts[0])
        except (ValueError, IndexError):
            continue
        # V2.1AllNames at column 23 (or V2EnhancedOrganizations at col 14 as fallback)
        try:
            all_names = row[23] if len(row) > 23 else row[14]
        except IndexError:
            continue
        if not all_names:
            continue
        all_names_lower = all_names.lower()
        for ticker, names in TICKER_NAMES.items():
            for n in names:
                if n.lower() in all_names_lower:
                    scores[ticker].append(mean_tone)
                    break
    return scores


def process_day(d: date) -> dict:
    """Return dict ticker -> {"tone": float, "count": int} for the day, or {} if no data.

    "tone"  = mean V2.1Tone across matched articles (0.0 if no articles).
    "count" = number of articles matched.
    """
    text = fetch_gkg_for_day(d, hour=23, minute=0)
    if text is None:
        # Try 23:30 as fallback (15-min granularity files)
        text = fetch_gkg_for_day(d, hour=23, minute=30)
    if text is None:
        return {}
    scores = parse_gkg_text(text)
    aggregated = {}
    for t, vals in scores.items():
        if vals:
            aggregated[t] = {"tone": sum(vals) / len(vals), "count": len(vals)}
        else:
            aggregated[t] = {"tone": 0.0, "count": 0}
    return aggregated


def _fieldnames() -> list[str]:
    """Schema: date, <TICKER>_tone, <TICKER>_count, ... × N tickers."""
    out = ["date"]
    for t in TICKER_NAMES:
        out.append(f"{t}_tone")
        out.append(f"{t}_count")
    return out


def load_existing(path: Path) -> dict:
    """Read existing CSV (if any) into dict[date_str] -> dict[ticker -> {tone,count}].

    Tolerates old single-column-per-ticker schemas by treating missing _count
    columns as 0.
    """
    if not path.exists():
        return {}
    existing = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = {}
            for t in TICKER_NAMES:
                tone_col = f"{t}_tone"
                count_col = f"{t}_count"
                # New schema first; fall back to bare ticker (old schema) for tone.
                tone_raw = row.get(tone_col, row.get(t, "0"))
                count_raw = row.get(count_col, "0")
                try:
                    tone = float(tone_raw) if tone_raw not in (None, "") else 0.0
                except ValueError:
                    tone = 0.0
                try:
                    count = int(float(count_raw)) if count_raw not in (None, "") else 0
                except ValueError:
                    count = 0
                day[t] = {"tone": tone, "count": count}
            existing[row["date"]] = day
    return existing


def write_csv(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in sorted(data.keys()):
            row = {"date": d}
            day_data = data[d]
            for t in TICKER_NAMES:
                entry = day_data.get(t, {"tone": 0.0, "count": 0})
                row[f"{t}_tone"] = entry["tone"]
                row[f"{t}_count"] = entry["count"]
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--out",   type=Path, default=OUT_PATH)
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    existing = load_existing(args.out)
    print(f"Existing CSV: {len(existing)} days")
    print(f"Date range requested: {start} → {end}")

    new_count = 0
    empty_day = {t: {"tone": 0.0, "count": 0} for t in TICKER_NAMES}
    for d in daterange(start, end):
        ds = d.isoformat()
        if ds in existing:
            continue
        # Skip weekends (markets closed, news still matters but signal weaker)
        if d.weekday() >= 5:
            existing[ds] = {t: {"tone": 0.0, "count": 0} for t in TICKER_NAMES}
            continue
        scores = process_day(d)
        if not scores:
            print(f"  [{ds}] no data")
            continue
        existing[ds] = scores
        new_count += 1
        if new_count % 10 == 0:
            # Print a compact summary of the just-processed day
            summary = {t: (v["tone"], v["count"]) for t, v in scores.items() if v["count"] > 0}
            print(f"  [{ds}] processed {new_count} new days; nonzero tickers: {len(summary)}")
            # Incremental save so progress persists
            write_csv(args.out, existing)

    write_csv(args.out, existing)
    print(f"Done. {len(existing)} total days in {args.out}")


if __name__ == "__main__":
    main()
