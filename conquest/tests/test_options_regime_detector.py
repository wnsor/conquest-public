"""Tests for Phase 8 regime_detector module.

Validates the classifier produces the expected label across regime
transitions, and the strategy allow-list correctly filters which
strategies can fire in which regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

# conquest_options/ uses bare sibling imports (per memory feedback_lean_bare_imports)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conquest_options"))

from edge_signals.regime_detector import (  # noqa: E402
    REGIME_LABELS,
    STRATEGY_REGIME_DEFAULTS,
    classify_regime,
    is_strategy_allowed_in_regime,
)


def test_crisis_state_takes_priority_over_vix():
    """If CrisisDetector says 'rebound', return 'rebound' even with high VIX."""
    assert classify_regime(vix=45, vote_count=3, crisis_state="rebound") == "rebound"
    assert classify_regime(vix=12, vote_count=0, crisis_state="rebound") == "rebound"


def test_crisis_state_recovery_passthrough():
    assert classify_regime(vix=18, vote_count=0, crisis_state="recovery") == "recovery"


def test_crisis_state_crash_maps_to_crisis():
    assert classify_regime(vix=40, vote_count=2, crisis_state="crash") == "crisis"
    assert classify_regime(vix=70, vote_count=3, crisis_state="capitulation") == "crisis"


def test_vix_above_30_or_vote_2_is_crisis():
    assert classify_regime(vix=31, vote_count=0) == "crisis"
    assert classify_regime(vix=20, vote_count=2) == "crisis"
    assert classify_regime(vix=50, vote_count=3) == "crisis"


def test_warning_thresholds():
    assert classify_regime(vix=27, vote_count=0) == "warning"
    assert classify_regime(vix=18, vote_count=1) == "warning"
    assert classify_regime(vix=18, vote_count=0, term_regime="backwardation") == "warning"


def test_neutral_band():
    assert classify_regime(vix=22, vote_count=0) == "neutral"
    assert classify_regime(vix=24.9, vote_count=0) == "neutral"


def test_bull_high_vol_band():
    assert classify_regime(vix=16, vote_count=0) == "bull_high_vol"
    assert classify_regime(vix=19.9, vote_count=0) == "bull_high_vol"


def test_bull_low_vol_band():
    assert classify_regime(vix=12, vote_count=0) == "bull_low_vol"
    assert classify_regime(vix=14.9, vote_count=0) == "bull_low_vol"


def test_missing_vix_defaults_to_neutral():
    assert classify_regime(vix=None, vote_count=0) == "neutral"


def test_returned_label_always_valid():
    """No matter the inputs, classifier returns a label from REGIME_LABELS."""
    import itertools
    vix_values = [None, 10, 18, 22, 28, 35, 60]
    vote_values = [None, 0, 1, 2, 3]
    term_values = [None, "contango", "flat", "backwardation"]
    crisis_values = [None, "warning", "crash", "capitulation", "rebound", "recovery"]
    for vix, vote, term, crisis in itertools.product(
            vix_values, vote_values, term_values, crisis_values):
        regime = classify_regime(
            vix=vix, vote_count=vote, term_regime=term, crisis_state=crisis)
        assert regime in REGIME_LABELS, f"got '{regime}' for vix={vix} vote={vote}"


def test_d2_tepper_only_fires_in_rebound():
    """D2 Tepper V-bottom LEAPS should be locked to rebound regime."""
    assert is_strategy_allowed_in_regime("tepper_vbottom_leaps", "rebound")
    assert not is_strategy_allowed_in_regime("tepper_vbottom_leaps", "crisis")
    assert not is_strategy_allowed_in_regime("tepper_vbottom_leaps", "bull_low_vol")


def test_b1_spy_crisis_put_only_fires_in_warning_or_crisis():
    """B1 only triggers in the early-warning / crisis bands."""
    assert is_strategy_allowed_in_regime("spy_crisis_put", "warning")
    assert is_strategy_allowed_in_regime("spy_crisis_put", "crisis")
    assert not is_strategy_allowed_in_regime("spy_crisis_put", "bull_low_vol")
    assert not is_strategy_allowed_in_regime("spy_crisis_put", "rebound")


def test_crisis_rebound_basket_only_fires_in_rebound():
    """CrisisReboundBasket should fire exclusively in rebound regime."""
    assert is_strategy_allowed_in_regime("crisis_rebound_basket", "rebound")
    assert not is_strategy_allowed_in_regime("crisis_rebound_basket", "crisis")
    assert not is_strategy_allowed_in_regime("crisis_rebound_basket", "recovery")


def test_unknown_strategy_id_defaults_to_allowed():
    """Strategies not in STRATEGY_REGIME_DEFAULTS fire in every regime."""
    for regime in REGIME_LABELS:
        assert is_strategy_allowed_in_regime("some_new_strategy_xyz", regime)


def test_a_gex_spy_call_allowed_in_recovery():
    """A_GEX selective call resumes during recovery (markets normalizing)."""
    assert is_strategy_allowed_in_regime("gex_spy_selective", "recovery")
    assert not is_strategy_allowed_in_regime("gex_spy_selective", "crisis")


def test_d1_leaps_only_in_quiet_bull():
    """D1 LEAPS — only fire in calmest regime + recovery."""
    assert is_strategy_allowed_in_regime("cgrowth_leaps", "bull_low_vol")
    assert is_strategy_allowed_in_regime("cgrowth_leaps", "recovery")
    assert not is_strategy_allowed_in_regime("cgrowth_leaps", "bull_high_vol")
    assert not is_strategy_allowed_in_regime("cgrowth_leaps", "warning")


def test_strategy_regime_defaults_keys_are_known_strategy_ids():
    """Every key in STRATEGY_REGIME_DEFAULTS should be a real strategy id
    that appears in ENABLED_STRATEGIES. Catches typos."""
    # Import lazily to avoid circular deps
    from strategies import ENABLED_STRATEGIES  # noqa
    known_ids = {s.id for s in ENABLED_STRATEGIES}
    for sid in STRATEGY_REGIME_DEFAULTS:
        assert sid in known_ids, (
            f"STRATEGY_REGIME_DEFAULTS has unknown id '{sid}' — "
            f"known: {sorted(known_ids)}"
        )
