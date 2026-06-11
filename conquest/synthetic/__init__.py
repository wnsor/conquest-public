"""Synthetic data infrastructure for pre-inception BT windows.

For 1970s + 1987 stress-test BTs, the conquest universe needs synthesized data:
- Lev ETFs (TQQQ/UPRO/TNA/SOXL/UDOW/UGL/TMF) launched 2008-2010
- VIX launched 1990; VIX3M launched 2007-12
- HYG launched 2007-04; LQD launched 2002-07
- SPDR sectors launched 1998-12
- SPY launched 1993; SPX daily back to 1927

Each module here builds and validates a synthesizer against post-inception data
where ground truth exists, then exposes a function to back-cast pre-inception.

Validation discipline: every synthesizer must achieve ≤10% relative error on
held-out post-inception data before being used for back-cast. Validation scores
are saved alongside the synthetic CSVs.

See DEFERRED_RESEARCH.md #11 for the full project scope. This package is the
research-grade implementation; expect 3-5 days wall-clock end-to-end.
"""
from conquest.synthetic.lev_etf import synthesize_lev_etf, validate_lev_etf
from conquest.synthetic.vix import synthesize_vix, validate_vix

__all__ = [
    "synthesize_lev_etf",
    "validate_lev_etf",
    "synthesize_vix",
    "validate_vix",
]
