"""Registry mapping logical names → source series IDs and transforms.

Used by scripts/refresh_data.py for pulls and by conquest/regime/* to assemble
macro inputs by logical name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    source: str          # 'fred' | 'bls' | 'oecd'
    series_id: str
    transform: str       # 'level' | 'yoy_pct' | 'mom_pct' | 'qoq_pct'
    publication_lag_days: int   # release timing relative to reference period end
    description: str = ""


REGISTRY: dict[str, SeriesSpec] = {
    "gdp_real": SeriesSpec(
        name="gdp_real",
        source="fred",
        series_id="GDPC1",
        transform="level",
        publication_lag_days=30,
        description="Real Gross Domestic Product, quarterly, seasonally adjusted annual rate.",
    ),
    "cpi_headline": SeriesSpec(
        name="cpi_headline",
        source="fred",
        series_id="CPIAUCSL",
        transform="level",
        publication_lag_days=14,
        description="CPI for All Urban Consumers, all items, seasonally adjusted, monthly.",
    ),
    "cpi_core": SeriesSpec(
        name="cpi_core",
        source="fred",
        series_id="CPILFESL",
        transform="level",
        publication_lag_days=14,
        description="Core CPI (less food and energy), seasonally adjusted, monthly.",
    ),
    "unemployment_rate": SeriesSpec(
        name="unemployment_rate",
        source="fred",
        series_id="UNRATE",
        transform="level",
        publication_lag_days=7,
        description="Civilian unemployment rate, seasonally adjusted, monthly.",
    ),
    "pce_headline": SeriesSpec(
        name="pce_headline",
        source="fred",
        series_id="PCEPI",
        transform="level",
        publication_lag_days=30,
        description="PCE price index, seasonally adjusted, monthly. Fed's preferred inflation gauge.",
    ),
    "fed_funds": SeriesSpec(
        name="fed_funds",
        source="fred",
        series_id="DFF",
        transform="level",
        publication_lag_days=1,
        description="Effective federal funds rate, daily.",
    ),
}
