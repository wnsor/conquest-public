"""Unit tests for scripts/data/_strategy_specs.py preflight logic.

Layer-4 (data availability) + Layer-5 (window-fit) checks. Each test
constructs a known input and asserts the PreflightResult status / issues.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from _strategy_specs import (  # type: ignore
    REGISTRY, StrategySpec, get_spec, preflight_check,
)


# ── Layer-4 (populate %) ──────────────────────────────────────────────────

def test_pass_when_all_fields_populate_above_threshold():
    # vix_term_recovery needs vix ≥ 95% + vix9d_vix_ratio ≥ 80%
    # Keys are field_name[:14] per DataProbe.summary() truncation.
    probe = {
        "probe_g_vix": "99.5%",
        "probe_g_vix9d_vix_rati": "85.0%",
    }
    res = preflight_check("vix_term_recovery",
                          "2012-01-01", "2026-01-01", probe)
    assert res.status == "pass"


def test_fail_when_required_field_below_threshold():
    # im_divergence requires implied_move_vs_realized ≥ 30%
    probe = {
        "probe_g_vix": "99%",
        "probe_t_implied_move_v": "5.0%",   # below 30% threshold
        "probe_t_earnings_withi": "3.0%",
        "probe_t_uoa_active": "2.0%",
    }
    res = preflight_check("implied_move_divergence",
                          "2024-01-01", "2024-08-31", probe)
    assert res.status == "fail"
    assert any("implied_move_vs_realized" in i and "5.0%" in i for i in res.issues)


def test_warn_when_probe_field_missing():
    # Probe runtime stats doesn't contain the required field at all
    probe = {"probe_g_vix": "99%"}   # no vix9d_vix_ratio
    res = preflight_check("vix_term_recovery",
                          "2012-01-01", "2026-01-01", probe)
    # Should warn (field not in probe results — probe is older / didn't track)
    assert res.status in ("warn", "fail")
    assert any("vix9d_vix_ratio" in i for i in res.issues)


def test_skips_layer_4_when_probe_omitted():
    # No probe stats → only Layer 5 (window-fit) applies
    res = preflight_check("vix_term_recovery",
                          "2012-01-01", "2026-01-01", probe_runtime_stats=None)
    # 14y × 5/yr = 70 fires expected — well above min 5
    assert res.status == "pass"


# ── Layer-5 (window-fit) ─────────────────────────────────────────────────

def test_warn_when_window_too_short_for_rare_fire_strategy():
    # vix_term_recovery: 5 fires/yr. 6-month window = ~2.5 expected fires.
    res = preflight_check("vix_term_recovery",
                          "2024-01-01", "2024-07-01")
    assert res.status == "warn"
    assert any("expected fires" in i.lower() for i in res.issues)
    assert res.expected_fires is not None
    assert res.expected_fires < 5


def test_pass_when_window_supports_n_fires():
    # network_propagation: 20 fires/yr × 1 year = 20 expected → above min 5
    res = preflight_check("network_propagation",
                          "2024-01-01", "2024-12-31")
    assert res.status == "pass"


def test_fail_supersedes_warn():
    # Window-fit fails (warn) AND data-pop fails (fail) → final is fail
    probe = {
        "probe_g_vix": "99%",
        "probe_t_implied_move_v": "5%",
        "probe_t_earnings_withi": "3%",
        "probe_t_uoa_active": "2%",
    }
    # Short window AND data fails
    res = preflight_check("implied_move_divergence",
                          "2024-01-01", "2024-03-01", probe)
    assert res.status == "fail"


# ── registry sanity ──────────────────────────────────────────────────────

def test_missing_spec_is_warn():
    res = preflight_check("totally_made_up_strat",
                          "2024-01-01", "2024-12-31", probe_runtime_stats={})
    assert res.status == "warn"
    assert any("no StrategySpec" in i for i in res.issues)


def test_registry_has_all_active_leading_only_strategies():
    """Every active leading-only / Tier C strategy should have a spec.

    If you add a new strategy and forget to register a spec, this test
    fails — forcing the developer to declare data dependencies + fire rate.
    """
    expected = {
        # Leading-indicator strategies
        "vix_term_recovery", "implied_move_divergence",
        "dealer_opex_squeeze", "network_propagation",
        "reflex_ignition_v2", "short_squeeze_pure",
        "triple_confluence", "momentum_otm_calls",
        # Crisis-cycle strategies
        "crisis_dual_directional", "spy_crisis_put",
        "tepper_vbottom_leaps",
    }
    missing = expected - REGISTRY.keys()
    assert not missing, (
        f"strategies without StrategySpec: {sorted(missing)}. "
        f"Add to scripts/data/_strategy_specs.py REGISTRY."
    )


def test_get_spec_returns_known():
    spec = get_spec("vix_term_recovery")
    assert spec is not None
    assert spec.expected_fires_per_year > 0
    assert "vix" in spec.required_context_fields
