"""Conquest — shared trading-system library.

Subpackages
-----------
conquest.data        — FRED/ALFRED, BLS, OECD ingestion + parquet cache
conquest.regime      — 4-quadrant macro regime classifier (Bridgewater-style)
conquest.vol         — realized vol + vol-targeted sizing
conquest.indicators  — pandas-vectorized RSI/MACD/TRIX/SMA/MOMP (Lean parity)
conquest.models      — strategy implementations for cross-comparison
conquest.backtest    — vectorized backtest engine + IB-realistic costs + metrics + ranker
conquest.signals     — signal exporter (writes Lean Object Store CSVs)
conquest.secrets     — secret.yaml loader
"""

__version__ = "0.1.0"
