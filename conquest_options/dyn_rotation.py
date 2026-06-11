"""Pure-stdlib monthly rotation planner for the dynamic PIT-momentum path.

Why this is a separate module
-----------------------------
``main.py`` imports ``AlgorithmImports`` (Lean-only), so it cannot be imported
in offline pytest. The *decision* half of the monthly rebalance — given the
current active set, the draining set, the freshly-ranked top-N, what is already
subscribed, and which names we hold an open position on, work out who enters,
who gets a new option chain, who drains, and who is fully removed — is pure
bookkeeping with no Lean dependency. Extracting it here lets the rotation-diff
logic be unit-tested with no QC / network / data, while ``main.py`` keeps only
the thin Lean side-effects (``add_option`` / ``remove_security``).

The single invariant this enforces (the one QC gotcha that loses money):
**a name we hold an open position on is NEVER removed** — removing a held
security triggers an uncontrolled Lean market-order liquidation that bypasses
ExitManager. Such a leaver is instead moved to ``drain``: its chain stays
subscribed (so ExitManager can still close it via TP/SL/time/expiry) but it is
dropped from the entry-eligible active set. Once a drainer is flat, the next
rebalance removes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class RotationPlan:
    """The fully-resolved diff for one monthly rebalance.

    entrants     — names newly in the top-N (were not active before).
    add_chain    — entrants that still need an option chain subscribed
                   (equity already lives in the union, only the option is added).
    drain        — leavers we hold a position on → move to draining, do NOT remove.
    remove       — option chains to actually unsubscribe this cycle (flat leavers
                   + drainers that have since gone flat).
    new_active   — the resulting entry-eligible set after applying the diff.
    new_draining — the resulting draining set after applying the diff.
    """
    entrants: list[str]
    add_chain: list[str]
    drain: list[str]
    remove: list[str]
    new_active: set[str]
    new_draining: set[str]


def plan_rotation(
    current_active: Iterable[str],
    draining: Iterable[str],
    new_set: Iterable[str],
    subscribed: Iterable[str],
    has_position: Callable[[str], bool],
) -> RotationPlan:
    """Resolve the monthly rotation diff.

    Args:
        current_active: tickers currently entry-eligible.
        draining: tickers held-but-leaving from a prior cycle (chain still up).
        new_set: this month's freshly-ranked PIT-gated top-N.
        subscribed: tickers whose option chain is already subscribed (active or
            draining) — an entrant in this set needs no fresh ``add_option``.
        has_position: predicate — True iff we currently hold an open option
            position on the ticker (so removing it would force-liquidate).

    Order of operations matters: leavers are processed BEFORE entrants so a name
    that both leaves and re-enters in the same cycle (rare, but possible at the
    top-N boundary) is correctly reactivated rather than churned.
    """
    current_active = set(current_active)
    new_set = set(new_set)
    subscribed = set(subscribed)

    new_active: set[str] = set(current_active)
    new_draining: set[str] = set(draining)

    entrants = sorted(new_set - current_active)
    leavers = sorted(current_active - new_set)

    remove: list[str] = []
    drain: list[str] = []

    # ── leavers first (free budget / mark drains before adding) ──────────────
    for t in leavers:
        new_active.discard(t)
        if has_position(t):
            new_draining.add(t)      # held → keep chain, drop from eligible
            drain.append(t)
        else:
            remove.append(t)         # flat → safe to unsubscribe now

    # ── entrants ─────────────────────────────────────────────────────────────
    add_chain: list[str] = []
    for t in entrants:
        new_active.add(t)
        new_draining.discard(t)      # re-entering a draining name reactivates it
        if t not in subscribed and t not in remove:
            add_chain.append(t)

    # ── drainers that have since gone flat → remove this cycle ───────────────
    for t in sorted(new_draining):
        if not has_position(t):
            remove.append(t)
            new_draining.discard(t)

    return RotationPlan(
        entrants=entrants,
        add_chain=add_chain,
        drain=drain,
        remove=sorted(set(remove)),
        new_active=new_active,
        new_draining=new_draining,
    )
