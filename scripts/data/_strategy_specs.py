"""Strategy-spec registry for pre-launch backtest validation.

Why this exists
---------------
Layer 4 + 5 of the "no more 0-trade BTs" defense.

A 0-trade BT can have many root causes, but most fall into 3 buckets:
  1. **Data unavailable**: strategy gates on a ctx.* field that is empty
     at QC daily resolution (caught by `required_context_fields` + populate
     check against the latest probe BT runtime stats).
  2. **Wrong window**: strategy is rare-fire by design and the test window
     is too short for honest sample size (caught by `expected_fires_per_year`
     × window_years).
  3. **Code bug / gate calibration**: caught at Layer 1 by
     test_options_leading_indicator_strategies.py.

This registry covers buckets 1 and 2. Each StrategySpec declares:

  required_context_fields:
      list of context field names the strategy reads in on_data. Must be a
      subset of the fields the v28 DataProbe tracks. The pre-launch
      validator looks up each field's latest populate-% from a reference
      probe BT and refuses to launch if any drops below
      `min_populate_pct[field]`.

  min_populate_pct:
      per-field minimum populate %. Use 80% for fields the strategy gates
      MULTIPLE times (would never fire without it), 30% for fields that
      are filter-only confirmations.

  expected_fires_per_year:
      authoritative fire-rate estimate based on the strategy's gate
      tightness × universe size. Used to compute expected_fires for a
      given window. If `window_years * expected_fires_per_year <
      min_fires_to_test`, the validator warns (or fails with --strict).

  min_fires_to_test:
      Default 5. Below this, the BT can't meaningfully validate the
      strategy (sample size too small for per-trade gate evaluation).

Adding a strategy spec
----------------------
    REGISTRY["my_new_strategy"] = StrategySpec(
        strategy_id="my_new_strategy",
        required_context_fields=["vix", "iv_raw", "uoa_active"],
        min_populate_pct={"vix": 95.0, "iv_raw": 50.0, "uoa_active": 5.0},
        expected_fires_per_year=8.0,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategySpec:
    strategy_id: str
    required_context_fields: list[str] = field(default_factory=list)
    min_populate_pct: dict[str, float] = field(default_factory=dict)
    expected_fires_per_year: float = 0.0
    min_fires_to_test: int = 5
    notes: str = ""


@dataclass
class PreflightResult:
    strategy_id: str
    status: str   # "pass" | "warn" | "fail"
    issues: list[str] = field(default_factory=list)
    expected_fires: float | None = None
    populate_summary: dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# REGISTRY — one entry per strategy. Fire-rate estimates are heuristic and
# should be revised as we get actual BT data. Add an entry whenever a new
# strategy lands; a strategy without a spec defaults to PASS but emits a
# `missing_spec` advisory.
# ──────────────────────────────────────────────────────────────────────────

REGISTRY: dict[str, StrategySpec] = {
    # ── Leading-indicator strategies (Phase 2) ───────────────────────────
    "vix_term_recovery": StrategySpec(
        strategy_id="vix_term_recovery",
        required_context_fields=["vix", "vix9d_vix_ratio"],
        min_populate_pct={"vix": 95.0, "vix9d_vix_ratio": 80.0},
        # ~3-8 per year (one per crisis-recovery cycle). 1-ticker universe (SPY).
        expected_fires_per_year=5.0,
        notes=("VIX9D launched 2011-10. Pre-2011 windows are structurally "
               "0-fire — choose start ≥ 2012 OR accept the gap."),
    ),
    "implied_move_divergence": StrategySpec(
        strategy_id="implied_move_divergence",
        required_context_fields=[
            "vix", "implied_move_vs_realized",
            # OR-confirmation gate
            "earnings_within_5d", "uoa_active",
        ],
        min_populate_pct={
            "vix": 95.0,
            # IV-derived ratio MUST populate for this strategy to fire at all
            "implied_move_vs_realized": 30.0,
            "earnings_within_5d": 5.0,
            "uoa_active": 1.0,
        },
        # 16-ticker WSB universe × ~5-15 fires/yr ≈ 80-240/yr
        expected_fires_per_year=120.0,
        notes=("Data-blocked at daily resolution if Greeks.implied_volatility "
               "is not populated by QC. Requires BS-inverse IV tracker fix to "
               "unlock — defer until then."),
    ),
    "dealer_opex_squeeze": StrategySpec(
        strategy_id="dealer_opex_squeeze",
        required_context_fields=[
            "vix", "term_regime",
            "underlying_momentum_30d",
            # v2 OR-logic: needs uoa_active OR strong mom (not both)
            "uoa_active",
        ],
        min_populate_pct={
            "vix": 95.0,
            "underlying_momentum_30d": 80.0,
            # UOA can be sparse — v2 falls back to strong-mom branch
            "uoa_active": 1.0,
        },
        # 16-ticker × 12 OPEX/yr × ~25% pass rate ≈ 50/yr
        expected_fires_per_year=50.0,
        notes="v2 OR-logic: uoa OR mom30 ≥ 1.08. Tolerates low UOA populate.",
    ),
    "network_propagation": StrategySpec(
        strategy_id="network_propagation",
        required_context_fields=[
            "vix", "news_propagation_5d", "volume_spike",
            "underlying_momentum_30d",
        ],
        min_populate_pct={
            "vix": 95.0,
            "news_propagation_5d": 30.0,
            "volume_spike": 50.0,
            "underlying_momentum_30d": 80.0,
        },
        # 16-ticker × ~3-8/yr/ticker × strict threshold (np≥2.0+vol≥2.5+mom≥1.0)
        expected_fires_per_year=20.0,
    ),
    "reflex_ignition_v2": StrategySpec(
        strategy_id="reflex_ignition_v2",
        required_context_fields=[
            "vix", "short_interest_velocity", "insider_count_5d",
            "news_propagation_5d", "volume_spike",
        ],
        min_populate_pct={
            "vix": 95.0,
            "short_interest_velocity": 50.0,
            "insider_count_5d": 30.0,
            "news_propagation_5d": 30.0,
            "volume_spike": 50.0,
        },
        # Tighter than network_prop (5-accel confluence)
        expected_fires_per_year=12.0,
        notes=("Tier C — gated on form4 (insider_count) AND FINRA-SI "
               "(short_interest_velocity). Currently both data sources "
               "are failing in the workflow."),
    ),
    "short_squeeze_pure": StrategySpec(
        strategy_id="short_squeeze_pure",
        required_context_fields=[
            "vix", "short_interest_velocity", "insider_cluster_score",
            "volume_spike", "skew_z",
        ],
        min_populate_pct={
            "vix": 95.0,
            "short_interest_velocity": 50.0,
            "insider_cluster_score": 30.0,
            "volume_spike": 50.0,
            "skew_z": 30.0,
        },
        expected_fires_per_year=8.0,
        notes=("Tier C — needs FINRA-SI (currently failing) AND form4 "
               "(currently failing PR #6 backfill convergence)."),
    ),
    "triple_confluence": StrategySpec(
        strategy_id="triple_confluence",
        required_context_fields=[
            "vix", "short_interest_velocity", "insider_count_5d",
            "news_propagation_5d", "volume_spike",
            "implied_move_vs_realized", "skew_z",
        ],
        min_populate_pct={
            "vix": 95.0,
            "short_interest_velocity": 50.0,
            "insider_count_5d": 30.0,
            "news_propagation_5d": 30.0,
            "volume_spike": 50.0,
            "implied_move_vs_realized": 30.0,
            "skew_z": 30.0,
        },
        expected_fires_per_year=4.0,   # very strict — 7 gates
        notes="Tightest gate. Expect rare fires. Choose full-PIT window.",
    ),
    "momentum_otm_calls": StrategySpec(
        strategy_id="momentum_otm_calls",
        required_context_fields=["vix", "underlying_momentum_30d"],
        min_populate_pct={"vix": 95.0, "underlying_momentum_30d": 80.0},
        # WSB momentum — fires multiple times per name per year
        expected_fires_per_year=50.0,
    ),

    # ── Crisis-cycle strategies (puts → calls regime transition) ─────────
    "crisis_dual_directional": StrategySpec(
        strategy_id="crisis_dual_directional",
        required_context_fields=[
            "vix", "vix9d_vix_ratio", "term_regime",
            "crisis_state", "cstability_vote_count",
        ],
        min_populate_pct={
            "vix": 95.0, "vix9d_vix_ratio": 80.0,
            "term_regime": 95.0, "crisis_state": 95.0,
            "cstability_vote_count": 95.0,
        },
        # Puts: ~1-3 fires/yr (early warnings); Calls: ~1-3 fires/yr (rebounds)
        expected_fires_per_year=4.0,
        notes=("VIX9D launched Oct 2011 — pre-2011 windows are partially "
               "blocked (vix9d_vix_ratio is None). SPY-only universe; rare "
               "but high-conviction setups."),
    ),
    "spy_crisis_put": StrategySpec(
        strategy_id="spy_crisis_put",
        required_context_fields=["vix", "crisis_state"],
        min_populate_pct={"vix": 95.0, "crisis_state": 95.0},
        # Fires only on confirmed crisis_state ∈ (warning, crash) — rare
        expected_fires_per_year=2.0,
        notes="Reactive crisis-hedge (long SPY put). Pairs with crisis_dual.",
    ),
    "tepper_vbottom_leaps": StrategySpec(
        strategy_id="tepper_vbottom_leaps",
        required_context_fields=[
            "vix", "vix9d_vix_ratio", "term_regime",
            "underlying_drawdown_from_252d_high",
            "underlying_5ma_above_20ma",
        ],
        min_populate_pct={
            "vix": 95.0, "vix9d_vix_ratio": 80.0,
            "term_regime": 95.0,
            "underlying_drawdown_from_252d_high": 90.0,
            "underlying_5ma_above_20ma": 90.0,
        },
        # Only fires on V-bottom — extreme drawdown + 5MA cross-up.
        # Empirical: 3 fires across 2008-2024 (2009, 2020, 2022).
        expected_fires_per_year=0.5,
        notes=("MEMORY confirmed: 100% WR on 3 V-bottom captures (GFC, "
               "COVID, 2022 selloff) at $1M. Use full PIT for honest n."),
    ),
}


def get_spec(strategy_id: str) -> StrategySpec | None:
    """Return the StrategySpec for `strategy_id`, or None if not registered."""
    return REGISTRY.get(strategy_id)


def preflight_check(
    strategy_id: str,
    window_start: str,
    window_end: str,
    probe_runtime_stats: dict[str, str] | None = None,
    min_fires_to_test: int | None = None,
) -> PreflightResult:
    """Pre-launch validation for a strategy + window + probe-result combo.

    Args:
        strategy_id: maps into REGISTRY
        window_start, window_end: ISO date strings ("2024-01-01")
        probe_runtime_stats: dict from QC BT runtimeStatistics — should
            include keys like "probe_g_vix = 99.5%" or "probe_t_uoa_active = 2.1%".
            If None, skips Layer-4 populate check (Layer-5 still applies).
        min_fires_to_test: override per-call min fires (default: spec.min_fires_to_test).

    Returns:
        PreflightResult with status pass/warn/fail and human-readable issues.
        Caller (CLI / dispatcher) can refuse to launch on fail.
    """
    from datetime import date

    spec = get_spec(strategy_id)
    if spec is None:
        return PreflightResult(
            strategy_id=strategy_id,
            status="warn",
            issues=[
                f"no StrategySpec registered for {strategy_id!r} — "
                f"add one to scripts/data/_strategy_specs.py REGISTRY "
                f"to enable pre-launch validation"
            ],
        )

    res = PreflightResult(strategy_id=strategy_id, status="pass")

    # ── Layer 4: required-context-fields populate check ─────────────────
    if probe_runtime_stats is not None:
        for fld in spec.required_context_fields:
            # Look in both global (probe_g_) and per-ticker (probe_t_) buckets.
            # Per-ticker is the more common case so check it first.
            key_t = f"probe_t_{fld[:14]}"
            key_g = f"probe_g_{fld[:14]}"
            val = probe_runtime_stats.get(key_t) or probe_runtime_stats.get(key_g)
            if val is None:
                res.populate_summary[fld] = "missing-from-probe"
                res.issues.append(
                    f"field {fld!r} not present in probe runtime stats "
                    f"(probe may be old or didn't track this field)"
                )
                if res.status == "pass":
                    res.status = "warn"
                continue
            res.populate_summary[fld] = val
            # val is a string like "12.3%" — parse
            try:
                pct = float(str(val).rstrip("%"))
            except (ValueError, TypeError):
                continue
            min_required = spec.min_populate_pct.get(fld, 0.0)
            if pct < min_required:
                res.issues.append(
                    f"field {fld!r} populate {pct:.1f}% below required "
                    f"{min_required:.1f}% — strategy will rarely / never fire"
                )
                res.status = "fail"

    # ── Layer 5: expected-fires-vs-window estimator ─────────────────────
    try:
        d1 = date.fromisoformat(window_start)
        d2 = date.fromisoformat(window_end)
        years = max(0.001, (d2 - d1).days / 365.0)
    except Exception:
        years = 0.0
    expected = spec.expected_fires_per_year * years
    res.expected_fires = expected
    min_n = min_fires_to_test if min_fires_to_test is not None else spec.min_fires_to_test
    if expected < min_n:
        res.issues.append(
            f"window {window_start}→{window_end} ({years:.2f}y) × "
            f"{spec.expected_fires_per_year} fires/yr = ~{expected:.1f} "
            f"expected fires; below min {min_n}. Sample size will be too "
            f"small to validate per-trade gate. Use a longer window or accept."
        )
        if res.status != "fail":
            res.status = "warn"

    return res
