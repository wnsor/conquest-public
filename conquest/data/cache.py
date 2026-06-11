"""Parquet-backed cache for raw API pulls under data/alternative/conquest/raw/{source}/{key}.parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetCache:
    """Per-source parquet cache. Keyed by (source, key)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, source: str, key: str) -> Path:
        return self.root / source / f"{key}.parquet"

    def path(self, source: str, key: str) -> Path:
        return self._path(source, key)

    def exists(self, source: str, key: str) -> bool:
        return self._path(source, key).exists()

    def read(self, source: str, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(source, key))

    def write(self, source: str, key: str, df: pd.DataFrame) -> None:
        path = self._path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
