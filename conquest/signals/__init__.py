"""Pandas-vectorized signal helpers — cross-sectional rankers, breakout
detectors, volume-spike scorers.

Use these in research / vectorized backtests. Inside Lean Algorithms run the
equivalent arithmetic on a `self.history(...)` slice (cspec/main.py does
this) — they share the math, not the implementation. Tests in
`conquest/tests/test_cspec_signals.py` enforce that the two paths produce
identical numbers on a fixture.
"""
from conquest.signals.breakout_proximity import breakout_proximity
from conquest.signals.composite import cspec_composite_score
from conquest.signals.volume_spike import dollar_volume_spike, volume_spike

__all__ = [
    "dollar_volume_spike",
    "volume_spike",
    "breakout_proximity",
    "cspec_composite_score",
]
