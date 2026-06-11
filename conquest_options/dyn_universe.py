"""Dynamic PIT-momentum universe operations for ConquestOptions.

Split out of main.py purely to stay under QC's 64,000-char/file push limit.
These are free functions taking the algorithm instance as ``algo`` (not methods),
so the module imports cleanly and main.py calls them from its subscription branch
plus a thin ``on_securities_changed`` override. Behaviour is identical to the
former in-class version; the rotation invariant (a held leaver DRAINS — chain
kept for ExitManager — never force-liquidated) lives in dyn_rotation.plan_rotation.

Lean-only (imports AlgorithmImports), like main.py. Offline tests exercise the
pure decision logic via dyn_rotation.py, never this module.
"""
from __future__ import annotations

from AlgorithmImports import *

from edge_signals.pit_universe import PitUniverse
from dyn_rotation import plan_rotation

UNION_KEY = "conquest/universe/sp500_union_2008_2024.csv"


def union_tickers_from_csv(text: str) -> list[str]:
    """Parse the S&P union CSV ('ticker,sector' + header) → unique tickers."""
    out: list[str] = []
    seen: set[str] = set()
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        first = line.split(",")[0].strip().upper()
        if i == 0 and first in ("TICKER", "SYMBOL"):
            continue
        if first and first not in seen:
            seen.add(first)
            out.append(first)
    return out


def setup_dynamic_universe(algo) -> None:
    """Subscribe the full S&P union DAILY with a 180d MOMP each, load PIT
    membership, register the monthly rebalance. Fails CLOSED if a CSV is absent
    (no silent fallback to the survivorship-biased universe)."""
    pit_key = PitUniverse.OBJECT_STORE_KEY
    if not (algo.object_store.contains_key(UNION_KEY)
            and algo.object_store.contains_key(pit_key)):
        raise RuntimeError(
            f"DYNAMIC_PIT_MOMENTUM=1 but universe CSVs missing from Object Store "
            f"({UNION_KEY} / {pit_key}); failing CLOSED — push them first.")
    union = union_tickers_from_csv(algo.object_store.read(UNION_KEY))
    algo._dyn_pit = PitUniverse.from_csv_text(algo.object_store.read(pit_key))
    for ticker in union:
        sym = algo.add_equity(ticker, Resolution.DAILY).symbol
        algo._dyn_equity_sym[ticker] = sym
        algo._dyn_momp[sym] = algo.MOMP(sym, algo._dyn_mom_lookback, Resolution.DAILY)
    algo.debug(f"Dynamic PIT universe: {len(union)} union equities subscribed; "
               f"{len(algo._dyn_pit)} PIT snapshots; top_n={algo._dyn_top_n}.")
    algo.schedule.on(
        algo.date_rules.month_start("SPY"),
        algo.time_rules.after_market_open("SPY", 30),
        lambda: dyn_rebalance(algo),
    )


def dyn_rebalance(algo) -> None:
    """Monthly: PIT-gate → rank eligible by MOMP → rotate the top-N option chains.
    Held leavers drain (chain kept for ExitManager); see dyn_rotation.plan_rotation."""
    if algo.is_warming_up or algo._dyn_pit is None:
        return
    eligible = algo._dyn_pit.members_asof(algo.time)   # PIT gate BEFORE ranking
    scored: list[tuple[float, str]] = []
    for ticker in eligible:
        sym = algo._dyn_equity_sym.get(ticker)
        if sym is None:
            continue
        momp = algo._dyn_momp.get(sym)
        if momp is None or not momp.is_ready:
            continue
        val = float(momp.current.value)
        if val <= 0:
            continue
        scored.append((val, ticker))
    scored.sort(reverse=True)
    new_set = {t for _, t in scored[: algo._dyn_top_n]}

    plan = plan_rotation(
        current_active=algo._dyn_active,
        draining=algo._dyn_draining,
        new_set=new_set,
        subscribed=set(algo._option_symbols.keys()),
        has_position=lambda t: dyn_has_open_position(algo, t),
    )
    for ticker in plan.remove:                 # removals first (free budget)
        dyn_remove_option(algo, ticker)
    today = algo.time.date()
    for ticker in plan.add_chain:              # equity already in union
        eq_sym = algo._dyn_equity_sym.get(ticker)
        if algo._dyn_option_res != Resolution.DAILY:
            # Escalated path: match option resolution on the underlying so chain
            # data flows in sync (daily MOMP unaffected — same Symbol).
            eq_sym = algo.add_equity(ticker, algo._dyn_option_res).symbol
            algo._dyn_equity_sym[ticker] = eq_sym
        if eq_sym is not None and eq_sym in algo.securities:
            algo.securities[eq_sym].volatility_model = \
                StandardDeviationOfReturnsVolatilityModel(30)
        option = algo.add_option(ticker, algo._dyn_option_res)
        # Tight band around the 28-DTE / 15%-OTM target → far fewer contracts
        # than the static (-30,30)/(14,400) filter (≤10 chains active at once).
        option.set_filter(lambda u: u.strikes(-30, 30).expiration(20, 45))
        option.price_model = OptionPriceModels.black_scholes()
        algo._option_symbols[ticker] = option.symbol
        algo._dyn_pending_entry[ticker] = today   # can't trade on the add bar
    algo._dyn_active = plan.new_active
    algo._dyn_draining = plan.new_draining
    algo.debug(
        f"dyn_rebalance {today}: top{algo._dyn_top_n}={sorted(new_set)} "
        f"entrants={plan.entrants} add_chain={plan.add_chain} drain={plan.drain} "
        f"remove={plan.remove} active={len(algo._dyn_active)} "
        f"draining={len(algo._dyn_draining)}")


def dyn_has_open_position(algo, ticker: str) -> bool:
    """True iff we hold an open option position on this underlying — removing such
    a security would force an uncontrolled Lean liquidation."""
    return any(p.symbol_str.split(" ")[0] == ticker
               for p in algo._exit_mgr.positions())


def dyn_remove_option(algo, ticker: str) -> None:
    """Unsubscribe a ticker's OPTION chain only — the equity stays (feeds MOMP).
    Caller guarantees the name is flat."""
    opt_sym = algo._option_symbols.pop(ticker, None)
    algo._dyn_pending_entry.pop(ticker, None)
    if opt_sym is not None:
        algo.remove_security(opt_sym)


def on_securities_changed(algo, changes) -> None:
    """Drop a removed security's canonical option Symbol from the maps so a stale
    Symbol can't be re-traded. No selection logic — dyn_rebalance owns rotation.
    No-op on the static path."""
    if not getattr(algo, "_dynamic_pit", False):
        return
    for sec in changes.removed_securities:
        sym = sec.symbol
        for t, osym in list(algo._option_symbols.items()):
            if osym == sym:
                algo._option_symbols.pop(t, None)
                algo._dyn_pending_entry.pop(t, None)
                algo._dyn_active.discard(t)
