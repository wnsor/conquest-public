"""Conquest Options — long-only options-strategy backtest framework.

Per-trade promotion gate (NOT CAGR):
    Expectancy ≥ +15%, PF ≥ 2.0, WR ≥ 35%, R-mean ≥ +0.5, Sortino ≥ 2.0,
    sample ≥ 50, max losing streak ≤ 12, time-in-market ≤ 30% (sparse).

Phase 1 ships the framework only — zero strategies in ENABLED_STRATEGIES.
Strategies land in Phases 2-8.
"""
