"""Schema + content validation for data ingester outputs.

Why this exists
---------------
Silent ingester failures (status=ok but rows=0, or right column names but
wrong dtypes, or yfinance API changed and now returns empty dataframes
without errors) caused multiple conquest_options leading-indicator
strategies to fire ZERO trades across many backtests.

Diagnosis took half a day. Most cases were preventable IF the daily refresh
had validated the output schema and contents instead of just trusting that
"subprocess exit 0" means "data is good".

This module declares a `SourceSpec` per data source — the schema the
downstream algorithms assume — and `validate()` produces a `ValidationResult`
that turns silent-pass failures into explicit failures in the workflow
report. Add a new SourceSpec entry whenever a new ingester ships.

Validation layers caught here
-----------------------------
  • file exists
  • CSV is parseable
  • row count meets the minimum (catches "ran fine but produced empty CSV")
  • required columns are present (catches schema drift)
  • date column parses (catches "wrote dates as 1970-epoch integers")
  • max_date is fresh enough (catches "static stale data")
  • numeric columns are actually numeric and not 100% NaN
  • per-column null rate stays below threshold

Adding a new source
-------------------
    REGISTRY["my_new_source"] = SourceSpec(
        name="my_new_source",
        output_path=STORAGE / "options" / "my_new_source.csv",
        required_columns=["date", "ticker", "value"],
        date_column="date",
        min_rows=10,
        max_lag_days=7,
        numeric_columns=["value"],
        null_rate_threshold={"value": 0.30},
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
STORAGE = WORKSPACE / "storage" / "conquest"


@dataclass
class SourceSpec:
    """Declares the contract that an ingester's output CSV must satisfy.

    Fields
    ------
    name : str
        Source key (matches HANDLERS / DEFAULT_SOURCES in refresh_all.py).
    output_path : Path
        Absolute path to the CSV the ingester writes.
    required_columns : list[str]
        Column names that MUST be present. Extra columns are allowed.
    date_column : str | None
        Which column is the timeline anchor. None = no recency check (for
        static reference data like sp500 membership).
    min_rows : int
        Minimum rows. Output with fewer rows is treated as `empty`.
    max_lag_days : int | None
        Allowed gap between `date_column.max()` and today. None disables.
    numeric_columns : list[str]
        Columns that must parse as numeric (pd.to_numeric). 100% non-numeric
        marks the source as `invalid`.
    null_rate_threshold : dict[str, float]
        Per-column max null rate (0.5 = up to 50% null is acceptable).
        Columns not listed default to 1.0 (any null rate OK).
    """
    name: str
    output_path: Path
    required_columns: list[str] = field(default_factory=list)
    date_column: str | None = None
    min_rows: int = 1
    max_lag_days: int | None = None
    numeric_columns: list[str] = field(default_factory=list)
    null_rate_threshold: dict[str, float] = field(default_factory=dict)
    # Per-column (min, max) plausibility bounds. A non-null value outside the band
    # (a 0 VIX, a negative spread, a decimal-shift spike) marks the source `invalid`.
    # Catches "syntactically valid but semantically wrong" values that the
    # null/numeric/staleness checks miss.
    value_range: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class ValidationResult:
    name: str
    status: str = "ok"   # ok | stale | empty | schema_mismatch | invalid | missing_spec
    rows: int = 0
    max_date: str | None = None
    issues: list[str] = field(default_factory=list)


def validate(spec: SourceSpec, today: date) -> ValidationResult:
    """Apply all checks for `spec`. Returns the worst-status ValidationResult.

    Status precedence (worst first):
      invalid > schema_mismatch > empty > stale > ok
    """
    res = ValidationResult(name=spec.name)

    if not spec.output_path.exists():
        res.status = "empty"
        res.issues.append(f"output file missing: {spec.output_path}")
        return res

    try:
        df = pd.read_csv(spec.output_path, low_memory=False)
    except Exception as e:
        res.status = "invalid"
        res.issues.append(f"CSV parse error: {type(e).__name__}: {e}")
        return res

    res.rows = len(df)
    if res.rows < spec.min_rows:
        res.status = "empty"
        res.issues.append(f"row count {res.rows} below min {spec.min_rows}")
        return res

    # Column lookups are CASE-INSENSITIVE. The canonical Object Store CSVs use
    # mixed casing (vix/daily.csv ships `Date,VIX` — what every algorithm
    # consumes, by name in cstability/cgrowth/cf and positionally in surge),
    # while specs are written lowercase. Matching on lowercase means a pure
    # casing difference is never a false schema_mismatch AND the date/numeric
    # checks below actually run instead of silently skipping. (2026-06-07)
    colmap = {str(c).lower(): c for c in df.columns}

    # Schema check
    missing = [c for c in spec.required_columns if c.lower() not in colmap]
    if missing:
        res.status = "schema_mismatch"
        res.issues.append(
            f"missing required columns: {missing}; got: {list(df.columns)[:10]}"
        )
        # Continue to surface other issues too — schema mismatch is fatal
        # but other checks may still produce useful diagnostics.

    # Date column check
    date_col = colmap.get(spec.date_column.lower()) if spec.date_column else None
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        n_bad = int(dates.isna().sum())
        if n_bad == res.rows:
            res.issues.append(
                f"date column `{spec.date_column}` is 100% unparseable"
            )
            if res.status == "ok":
                res.status = "invalid"
        elif n_bad > 0:
            res.issues.append(
                f"{n_bad}/{res.rows} rows have unparseable {spec.date_column}"
            )
        max_date_ts = dates.max()
        if pd.notna(max_date_ts):
            res.max_date = max_date_ts.strftime("%Y-%m-%d")
            if spec.max_lag_days is not None:
                age = (today - max_date_ts.date()).days
                if age > spec.max_lag_days:
                    if res.status == "ok":
                        res.status = "stale"
                    res.issues.append(
                        f"max_date {res.max_date} is {age}d old "
                        f"(threshold {spec.max_lag_days}d)"
                    )

    # Numeric column check
    for col in spec.numeric_columns:
        actual = colmap.get(col.lower())
        if actual is None:
            continue
        numeric = pd.to_numeric(df[actual], errors="coerce")
        n_nan = int(numeric.isna().sum())
        if n_nan == res.rows:
            res.issues.append(f"numeric column `{col}` is 100% non-numeric/null")
            if res.status == "ok":
                res.status = "invalid"
            continue
        threshold = spec.null_rate_threshold.get(col, 1.0)
        actual_rate = n_nan / res.rows
        if actual_rate > threshold:
            res.issues.append(
                f"column `{col}` null rate {actual_rate:.1%} > "
                f"threshold {threshold:.1%}"
            )
            if res.status == "ok":
                res.status = "stale"   # advisory; not blocking unless threshold tight

        # Value-range / plausibility check: a non-null value outside the declared
        # band is corrupt (0 VIX, negative spread, decimal-shift spike) — fatal.
        lo_hi = spec.value_range.get(col)
        if lo_hi is not None:
            lo, hi = lo_hi
            vals = numeric.dropna()
            oob = vals[(vals < lo) | (vals > hi)]
            if len(oob):
                examples = [round(float(x), 4) for x in oob.head(3).tolist()]
                res.issues.append(
                    f"column `{col}` has {len(oob)}/{res.rows} value(s) outside "
                    f"[{lo}, {hi}] (e.g. {examples}) — implausible/corrupt"
                )
                res.status = "invalid"   # top of precedence; a bad value is always fatal

    return res


# ─── Source registry ─────────────────────────────────────────────────────
# Every ingester listed in scripts/data/refresh_all.py HANDLERS should have
# an entry here. New sources MUST add a spec or validate() reports the
# source as `missing_spec` (informational; doesn't fail the workflow but
# warns that schema drift will go undetected).

REGISTRY: dict[str, SourceSpec] = {
    # ── FRED / BLS macro ─────────────────────────────────────────────────
    "fred_macro": SourceSpec(
        name="fred_macro",
        output_path=WORKSPACE / "data" / "alternative" / "conquest" / "raw"
                              / "fred_combined_daily.csv",
        required_columns=["date"],
        date_column="date",
        min_rows=1000,
        max_lag_days=7,
    ),
    # ── VIX family ───────────────────────────────────────────────────────
    "vix": SourceSpec(
        name="vix",
        output_path=STORAGE / "vix" / "daily.csv",
        required_columns=["date", "vix"],
        date_column="date",
        min_rows=4000,
        max_lag_days=4,
        numeric_columns=["vix"],
        null_rate_threshold={"vix": 0.05},
        value_range={"vix": (3.0, 150.0)},   # VIX historically ~9–90; <3 or >150 = corrupt
    ),
    "t10y2y": SourceSpec(
        name="t10y2y",
        output_path=STORAGE / "yield_curve" / "t10y2y.csv",
        required_columns=["date", "t10y2y"],
        date_column="date",
        min_rows=1000,
        max_lag_days=4,
        numeric_columns=["t10y2y"],
        value_range={"t10y2y": (-5.0, 5.0)},   # 10Y–2Y spread historically ~ -3..+3
    ),
    "unrate": SourceSpec(
        name="unrate",
        output_path=STORAGE / "macro" / "unrate.csv",
        required_columns=["date", "unrate"],
        date_column="date",
        min_rows=200,
        max_lag_days=45,   # monthly release
        numeric_columns=["unrate"],
        value_range={"unrate": (0.5, 30.0)},   # unemployment % never ~0 or >30
    ),
    "credit_vix_term": SourceSpec(
        name="credit_vix_term",
        output_path=STORAGE / "credit" / "hyg_ief_spread.csv",
        required_columns=["date"],
        date_column="date",
        min_rows=1000,
        max_lag_days=4,
    ),
    # ── Insider + activist signals ───────────────────────────────────────
    "form4": SourceSpec(
        name="form4",
        output_path=STORAGE / "insider" / "form4_opportunistic_buys_daily.csv",
        required_columns=["filing_date", "ticker"],
        date_column="filing_date",
        min_rows=100,
        max_lag_days=10,
    ),
    "edgar_13d": SourceSpec(
        name="edgar_13d",
        output_path=STORAGE / "insider" / "edgar_13d_filings.csv",
        required_columns=["filing_date", "ticker"],
        date_column="filing_date",
        min_rows=10,           # 13D filings are rare — 10/90d is reasonable
        max_lag_days=90,
    ),
    "edgar_8k_count": SourceSpec(
        name="edgar_8k_count",
        output_path=STORAGE / "insider" / "edgar_8k_count_daily.csv",
        required_columns=["date", "ticker"],
        date_column="date",
        min_rows=10,
        max_lag_days=90,
    ),
    # ── News / sentiment ─────────────────────────────────────────────────
    "gdelt": SourceSpec(
        name="gdelt",
        output_path=STORAGE / "sentiment" / "gdelt_daily.csv",
        required_columns=["date", "ticker"],
        date_column="date",
        min_rows=1000,
        max_lag_days=4,
    ),
    "google_trends": SourceSpec(
        name="google_trends",
        output_path=STORAGE / "sentiment" / "google_trends_daily.csv",
        required_columns=["date"],
        date_column="date",
        min_rows=10,
        max_lag_days=14,
    ),
    "reddit_wsb": SourceSpec(
        name="reddit_wsb",
        output_path=STORAGE / "sentiment" / "reddit_wsb_daily.csv",
        required_columns=["date", "ticker"],
        date_column="date",
        min_rows=1,
        max_lag_days=4,
    ),
    # ── Options / earnings ───────────────────────────────────────────────
    "earnings": SourceSpec(
        name="earnings",
        output_path=STORAGE / "options" / "earnings_calendar.csv",
        required_columns=["ticker", "earnings_date"],
        date_column="earnings_date",
        min_rows=50,
        max_lag_days=None,   # forward-looking calendar — no lag check
    ),
    "earnings_revisions": SourceSpec(
        name="earnings_revisions",
        output_path=STORAGE / "options" / "earnings_revisions_daily.csv",
        required_columns=["date", "ticker", "current_eps"],
        date_column="date",
        min_rows=10,
        max_lag_days=3,
        numeric_columns=["current_eps"],
    ),
    "finra_si": SourceSpec(
        name="finra_si",
        output_path=STORAGE / "options" / "finra_si_biweekly.csv",
        required_columns=["settlement_date", "ticker"],
        date_column="settlement_date",
        min_rows=10,
        max_lag_days=20,
    ),
    "cboe_indices": SourceSpec(
        name="cboe_indices",
        output_path=STORAGE / "options" / "cboe_indices.csv",
        required_columns=["date"],
        date_column="date",
        min_rows=1000,
        max_lag_days=4,
    ),
    "nfci": SourceSpec(
        name="nfci",
        output_path=STORAGE / "macro" / "nfci.csv",
        required_columns=["date"],
        date_column="date",
        min_rows=500,
        max_lag_days=10,
    ),
    "cftc_cot": SourceSpec(
        name="cftc_cot",
        output_path=STORAGE / "macro" / "cftc_cot_weekly.csv",
        required_columns=["date", "contract", "noncomm_long", "noncomm_short"],
        date_column="date",
        min_rows=10,
        max_lag_days=14,   # weekly publication
        numeric_columns=["noncomm_long", "noncomm_short"],
    ),
    "ipo_lockup": SourceSpec(
        name="ipo_lockup",
        output_path=STORAGE / "options" / "ipo_lockup_calendar.csv",
        required_columns=["ticker", "filing_date"],
        date_column="filing_date",
        min_rows=10,
        max_lag_days=365,   # IPOs are infrequent for individual tickers
    ),
    # ── Tier 2: derived ──────────────────────────────────────────────────
    "regime": SourceSpec(
        name="regime",
        output_path=STORAGE / "regime" / "daily.csv",
        required_columns=["date"],
        date_column="date",
        min_rows=2000,
        max_lag_days=4,
    ),
    "votes": SourceSpec(
        name="votes",
        output_path=STORAGE / "votes" / "cstability_4vote_daily.csv",
        required_columns=["date", "vote_count"],
        date_column="date",
        min_rows=2000,
        max_lag_days=4,
        numeric_columns=["vote_count"],
    ),
    # ── Universes (static, no lag check) ─────────────────────────────────
    "sp500": SourceSpec(
        name="sp500",
        output_path=STORAGE / "universe" / "sp500.csv",
        required_columns=["ticker"],
        min_rows=400,
    ),
    "leveraged_etfs": SourceSpec(
        name="leveraged_etfs",
        output_path=STORAGE / "universe" / "leveraged_etfs.csv",
        required_columns=["ticker"],
        min_rows=5,
    ),
    "acwx_top": SourceSpec(
        name="acwx_top",
        output_path=STORAGE / "universe" / "acwx_top.csv",
        required_columns=["ticker"],
        min_rows=20,
    ),
    "options_screen": SourceSpec(
        name="options_screen",
        output_path=STORAGE / "universe" / "options_screen.csv",
        required_columns=["ticker"],
        min_rows=10,
    ),
    # BULL options gate signal: confidence.csv (leading_confidence = S&P breadth).
    # A frozen/empty file here makes the live gate decorative — validate it has a
    # numeric leading_confidence and is recent (weekly cadence → 8d lag tolerance).
    "breadth": SourceSpec(
        name="breadth",
        output_path=STORAGE / "leading" / "confidence.csv",
        required_columns=["date", "leading_confidence"],
        date_column="date",
        min_rows=1000,
        max_lag_days=8,
        numeric_columns=["leading_confidence"],
        null_rate_threshold={"leading_confidence": 0.0},
    ),
}


def validate_by_name(source_name: str, today: date) -> ValidationResult:
    """Validate one source by its registry name.

    Returns `missing_spec` status (informational) if the source has no
    SourceSpec registered yet — adding new ingesters without specs is
    allowed but surfaces a warning so we remember to register them.
    """
    spec = REGISTRY.get(source_name)
    if spec is None:
        return ValidationResult(
            name=source_name,
            status="missing_spec",
            issues=[
                f"no SourceSpec registered for {source_name!r} — "
                f"add one to scripts/data/_validators.py REGISTRY to enable "
                f"schema + content checks"
            ],
        )
    return validate(spec, today)
