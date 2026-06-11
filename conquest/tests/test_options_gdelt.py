"""Tier1 Signal 1 — GDELTSentimentLoader unit tests.

Validates the loader's CSV parsing, tone lookup, and 30d-prior article-count
mean for `news_volume_spike`.

Production data is produced by scripts/ingest_gdelt_sentiment.py and pushed to
QC Object Store under key `conquest/sentiment/gdelt_daily.csv`. These tests
synthesize CSV text directly — no live GDELT downloads.
"""
from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

# Bare-sibling import path per memory feedback_lean_bare_imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conquest_options"))

from edge_signals.gdelt_sentiment import GDELTSentimentLoader  # noqa: E402


def _csv_text(tickers: list[str], rows: list[dict]) -> str:
    """Render rows to GDELT CSV schema: date, <T>_tone, <T>_count, ..."""
    cols = ["date"]
    for t in tickers:
        cols.append(f"{t}_tone")
        cols.append(f"{t}_count")
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def test_loader_empty_csv_returns_no_data():
    loader = GDELTSentimentLoader.from_csv_text("")
    assert loader.tone("PLTR", date(2024, 6, 3)) is None
    assert loader.volume_spike("PLTR", date(2024, 6, 3)) is None


def test_loader_tone_returns_raw_scale():
    """tone() returns raw GDELT scale [-100, +100]; caller rescales."""
    text = _csv_text(["PLTR"], [
        {"date": "2024-06-01", "PLTR_tone": "5.5", "PLTR_count": "12"},
        {"date": "2024-06-02", "PLTR_tone": "-3.2", "PLTR_count": "8"},
    ])
    loader = GDELTSentimentLoader.from_csv_text(text)
    assert loader.tone("PLTR", date(2024, 6, 2)) == -3.2
    assert loader.tone("PLTR", date(2024, 6, 1)) == 5.5


def test_loader_tone_returns_most_recent_before_or_on_date():
    """If today has no row, return the last row before today."""
    text = _csv_text(["PLTR"], [
        {"date": "2024-06-01", "PLTR_tone": "5.5", "PLTR_count": "12"},
        {"date": "2024-06-03", "PLTR_tone": "1.0", "PLTR_count": "20"},
    ])
    loader = GDELTSentimentLoader.from_csv_text(text)
    # 6/2 has no row → returns 6/1's tone
    assert loader.tone("PLTR", date(2024, 6, 2)) == 5.5


def test_loader_tone_missing_ticker_returns_none():
    text = _csv_text(["PLTR"], [
        {"date": "2024-06-01", "PLTR_tone": "5.5", "PLTR_count": "12"},
    ])
    loader = GDELTSentimentLoader.from_csv_text(text)
    assert loader.tone("AAPL", date(2024, 6, 3)) is None


def test_volume_spike_requires_30d_history():
    """<31 entries → no baseline → volume_spike returns None."""
    rows = []
    for i in range(20):
        d = date(2024, 5, 1) + timedelta(days=i)
        rows.append({"date": d.isoformat(), "PLTR_tone": "0", "PLTR_count": "10"})
    text = _csv_text(["PLTR"], rows)
    loader = GDELTSentimentLoader.from_csv_text(text)
    assert loader.volume_spike("PLTR", date(2024, 5, 20)) is None


def test_volume_spike_3x_baseline():
    """30 days @ count=10, day 31 @ count=30 → spike = 30 / 10 = 3.0."""
    rows = []
    for i in range(30):
        d = date(2024, 5, 1) + timedelta(days=i)
        rows.append({"date": d.isoformat(), "PLTR_tone": "0", "PLTR_count": "10"})
    rows.append({"date": "2024-05-31", "PLTR_tone": "0", "PLTR_count": "30"})
    text = _csv_text(["PLTR"], rows)
    loader = GDELTSentimentLoader.from_csv_text(text)
    spike = loader.volume_spike("PLTR", date(2024, 5, 31))
    assert spike == 3.0


def test_volume_spike_zero_baseline_returns_none():
    """30 days of count=0 + today=5 — baseline is 0, can't divide → None."""
    rows = []
    for i in range(30):
        d = date(2024, 5, 1) + timedelta(days=i)
        rows.append({"date": d.isoformat(), "PLTR_tone": "0", "PLTR_count": "0"})
    rows.append({"date": "2024-05-31", "PLTR_tone": "0", "PLTR_count": "5"})
    text = _csv_text(["PLTR"], rows)
    loader = GDELTSentimentLoader.from_csv_text(text)
    assert loader.volume_spike("PLTR", date(2024, 5, 31)) is None


def test_loader_multi_ticker_parse():
    """CSV with multiple tickers — each parses independently."""
    text = _csv_text(["PLTR", "RKLB"], [
        {"date": "2024-06-01", "PLTR_tone": "2.0", "PLTR_count": "5",
         "RKLB_tone": "-1.5", "RKLB_count": "3"},
    ])
    loader = GDELTSentimentLoader.from_csv_text(text)
    assert loader.tone("PLTR", date(2024, 6, 1)) == 2.0
    assert loader.tone("RKLB", date(2024, 6, 1)) == -1.5


def test_loader_handles_blank_count_field():
    """Empty count field → treated as 0, not crash."""
    text = _csv_text(["PLTR"], [
        {"date": "2024-06-01", "PLTR_tone": "1.0", "PLTR_count": ""},
    ])
    loader = GDELTSentimentLoader.from_csv_text(text)
    # No exception is the test
    assert loader.tone("PLTR", date(2024, 6, 1)) == 1.0


def test_strategy_context_has_news_fields():
    """Schema regression: StrategyContext has news_sentiment_24h + news_volume_spike."""
    from datetime import datetime

    from strategies.base import StrategyContext  # noqa: E402

    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        news_sentiment_24h={"PLTR": 0.45},
        news_volume_spike={"PLTR": 3.5},
    )
    assert ctx.news_sentiment_24h == {"PLTR": 0.45}
    assert ctx.news_volume_spike == {"PLTR": 3.5}
    ctx2 = StrategyContext(timestamp=datetime(2024, 6, 3, 15, 0))
    assert ctx2.news_sentiment_24h == {}
    assert ctx2.news_volume_spike == {}
