"""Fetch r/wallstreetbets mention velocity per ticker via PRAW.

WSB mention frequency is a LEADING signal for retail-driven moves
(GME-Jan21, AMC-May21, MSTR-Dec23, RKLB-2024 all preceded by WSB chatter
spikes 3-7 days before the price reaction).

Output: storage/conquest/sentiment/reddit_wsb_daily.csv
Schema: date, ticker, mention_count, distinct_users, top_score

Used by: retail_attention_cascade, quad_confluence strategies

Setup (one-time):
  1. Create a Reddit app: https://www.reddit.com/prefs/apps (script-type)
  2. Add to GH Action secrets:
       REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
  3. PRAW handles OAuth via these env vars

Strategy:
  - Pull all submissions + top comments from r/wallstreetbets for the day
  - Regex-extract ticker mentions ($MSTR, $GME, MSTR, etc.) with the
    universe filter (avoid common-English false positives like "PL", "MU")
  - Count mentions per ticker per day
  - Track distinct authors to filter spam/sockpuppet activity
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "sentiment" / "reddit_wsb_daily.csv"

WSB_UNIVERSE = {
    "CRDO", "MU", "NBIS", "NOK", "MX", "PL", "DRAM", "LPTH", "SOUN", "RDDT",
    "MSTR", "COIN", "PLTR", "RKLB", "APP", "SMCI",
    # Common WSB names not in our trading universe — track for context
    "SPY", "QQQ", "TSLA", "NVDA", "GME", "AMC", "BB", "META", "AMZN",
}

# Tickers that have common-English false positives — require $-prefix to count
AMBIGUOUS = {"PL", "MU", "MX", "APP", "BB", "GME", "AMC"}

TICKER_RX = re.compile(r"\$?([A-Z]{2,5})\b")


def extract_mentions(text: str, universe: set[str]) -> list[str]:
    """Return list of ticker mentions in text. Handles $-prefix carefully."""
    if not text:
        return []
    found = []
    for m in TICKER_RX.finditer(text):
        had_dollar = m.group(0).startswith("$")
        sym = m.group(1)
        if sym not in universe:
            continue
        if sym in AMBIGUOUS and not had_dollar:
            continue   # avoid false positives
        found.append(sym)
    return found


def fetch_wsb_day(target_date: date, days_back: int = 1) -> dict:
    """Pull r/wsb submissions + top comments for the past N days,
    return {ticker: {count, distinct_users, top_score}}."""
    import praw
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "conquest-research-bot/0.1")
    if not (client_id and client_secret):
        raise SystemExit("Set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET env vars")
    reddit = praw.Reddit(
        client_id=client_id, client_secret=client_secret, user_agent=user_agent,
    )
    sub = reddit.subreddit("wallstreetbets")
    cutoff = datetime.combine(target_date - timedelta(days=days_back),
                              datetime.min.time()).timestamp()
    end_cutoff = datetime.combine(target_date + timedelta(days=1),
                                   datetime.min.time()).timestamp()

    counts: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "users": set(), "top_score": 0,
    })
    # PRAW: pull "new" submissions (chronological); for backfill we'd need
    # PushShift API (now restricted) or Reddit's recent-archive API.
    # For daily incremental, .new() with limit=1000 is sufficient.
    for sub_post in sub.new(limit=1000):
        if sub_post.created_utc < cutoff:
            break
        if sub_post.created_utc > end_cutoff:
            continue
        text = (sub_post.title or "") + " " + (sub_post.selftext or "")
        for sym in extract_mentions(text, WSB_UNIVERSE):
            d = counts[sym]
            d["count"] += 1
            if sub_post.author:
                d["users"].add(str(sub_post.author))
            d["top_score"] = max(d["top_score"], int(sub_post.score or 0))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=1)
    args = ap.parse_args()

    today = date.today()
    print(f"Fetching r/wallstreetbets mentions for {today} (back {args.days_back}d)")
    counts = fetch_wsb_day(today, days_back=args.days_back)
    if not counts:
        print("WARN: no mentions found")
        return 0
    rows = []
    for ticker, d in counts.items():
        rows.append({
            "date": today.isoformat(),
            "ticker": ticker,
            "mention_count": d["count"],
            "distinct_users": len(d["users"]),
            "top_score": d["top_score"],
        })
    new_df = pd.DataFrame(rows)

    # Append + dedupe
    if OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV)
            df = pd.concat([existing, new_df], ignore_index=True)
            df.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=True)
        except Exception:
            df = new_df
    else:
        df = new_df
    df.sort_values(["date", "ticker"], inplace=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} (date, ticker) rows -> {OUT_CSV.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
