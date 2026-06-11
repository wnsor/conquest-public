"""Macro regime classification: 4-quadrant Bridgewater-style on GDP × CPI YoY.

Phase 1 v1 implementation: rule-based, hysteresis + min-dwell filtered. Phase 5+
may swap in HMM/clustering once the linear baseline's failure modes are known.
"""
from conquest.regime.classifier import RegimeClassifier, REGIME_LABELS

__all__ = ["RegimeClassifier", "REGIME_LABELS"]
