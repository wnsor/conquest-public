"""OptionsCostModel: per-leg + per-roll cost arithmetic."""
from __future__ import annotations

import pytest

from conquest.options.costs import OptionsCostModel


def test_default_per_leg_cost():
    m = OptionsCostModel()
    # 0.85 commission + 5.00 slippage = 5.85
    assert m.per_leg_cost == pytest.approx(5.85)


def test_round_trip_cost_default():
    m = OptionsCostModel()
    # 10 contracts open + 10 contracts close = 20 legs * $5.85
    assert m.roll_cost_usd(10.0) == pytest.approx(20 * 5.85)


def test_partial_close():
    m = OptionsCostModel()
    # Open new 10, close existing 5 → 15 legs total
    assert m.roll_cost_usd(10.0, 5.0) == pytest.approx(15 * 5.85)


def test_zero_contracts_zero_cost():
    m = OptionsCostModel()
    assert m.roll_cost_usd(0.0) == 0.0


def test_negative_contracts_treated_as_abs():
    m = OptionsCostModel()
    assert m.roll_cost_usd(-5.0, -3.0) == pytest.approx(8 * 5.85)


def test_custom_commission_and_slippage():
    m = OptionsCostModel(commission_per_contract=1.0, slippage_per_contract=10.0)
    assert m.per_leg_cost == pytest.approx(11.0)
    assert m.roll_cost_usd(10.0) == pytest.approx(20 * 11.0)
