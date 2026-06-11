"""Shared edge-signal computations.

These run inside the Lean Algorithm in OnData / Scheduled events. They
populate the StrategyContext that all strategies inspect.

Modules:
  iv_rank           IV rank/percentile per ticker (rolling 252d)
  uoa_lean          Wrapper around conquest.options.uoa.uoa_flag
  earnings_lookup   Object Store-backed earnings calendar lookup
  short_pressure    Per-ticker borrow fee_rate / shortable_quantity
  put_call_ratio_lean  DIY equity P/C ratio from chain volume each day
"""
from __future__ import annotations
