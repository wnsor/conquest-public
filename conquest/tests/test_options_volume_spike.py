"""Tier1 Signal 2 — volume_spike unit tests.

Mirrors the inline computation in `conquest_options/main.py:_build_context()`:

    today_dv = vh[-1]
    baseline = list(vh)[-21:-1]
    avg = sum(baseline) / len(baseline)
    volume_spike = today_dv / avg

The test exercises the math directly against a synthesized 21-element deque,
since the live sampler reads from QC's `securities` API (not testable here
without booting Lean).
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

# Bare-sibling import path per memory feedback_lean_bare_imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conquest_options"))


def _compute_volume_spike(vh: deque) -> float | None:
    """Replicate the production computation for unit-testing."""
    if vh is None or len(vh) < 21:
        return None
    today_dv = vh[-1]
    if today_dv <= 0:
        return None
    baseline = list(vh)[-21:-1]
    avg = sum(baseline) / len(baseline)
    if avg <= 0:
        return None
    return today_dv / avg


def test_volume_spike_5x_baseline():
    """20 days of dollar-volume=1.0, today=5.0 → spike == 5.0."""
    vh = deque([1.0] * 20 + [5.0], maxlen=260)
    assert _compute_volume_spike(vh) == 5.0


def test_volume_spike_1x_baseline():
    """Uniform baseline + today same value → spike == 1.0."""
    vh = deque([100.0] * 21, maxlen=260)
    assert _compute_volume_spike(vh) == 1.0


def test_volume_spike_below_average():
    """Today's dollar-volume is half the 20d average → spike == 0.5."""
    vh = deque([2.0] * 20 + [1.0], maxlen=260)
    assert _compute_volume_spike(vh) == 0.5


def test_volume_spike_insufficient_history():
    """<21 entries returns None (no signal)."""
    vh = deque([1.0] * 20, maxlen=260)
    assert _compute_volume_spike(vh) is None


def test_volume_spike_empty_deque():
    """Empty deque returns None."""
    assert _compute_volume_spike(deque(maxlen=260)) is None


def test_volume_spike_zero_baseline_average():
    """All-zero baseline (e.g. holiday weeks for thinly-traded names) returns None."""
    vh = deque([0.0] * 20 + [100.0], maxlen=260)
    assert _compute_volume_spike(vh) is None


def test_volume_spike_zero_today():
    """Zero today (no trade) returns None — no false high-spike."""
    vh = deque([1.0] * 20 + [0.0], maxlen=260)
    assert _compute_volume_spike(vh) is None


def test_strategy_context_has_volume_spike_field():
    """Schema regression: StrategyContext must have volume_spike as a dict
    field. If this breaks, A1 confluence integration breaks."""
    from strategies.base import StrategyContext  # noqa: E402
    from datetime import datetime

    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        volume_spike={"PLTR": 4.2},
    )
    assert ctx.volume_spike == {"PLTR": 4.2}
    # Default factory: missing field returns empty dict
    ctx2 = StrategyContext(timestamp=datetime(2024, 6, 3, 15, 0))
    assert ctx2.volume_spike == {}
