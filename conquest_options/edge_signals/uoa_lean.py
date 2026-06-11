"""Per-OnData wrapper around conquest.options.uoa.uoa_flag for Lean.

Maintains a per-contract rolling history of (volume, open_interest) so the
pure-function uoa_flag has its 20-day and 5-day baselines available.

Memory model: a dict keyed by (symbol_str) → two deques (vol, oi). Trimmed
implicitly by deque maxlen so memory stays bounded.
"""
from __future__ import annotations

from collections import deque, defaultdict

from conquest.options.uoa import uoa_flag


class UOATracker:
    def __init__(self, vol_window: int = 20, oi_window: int = 5,
                 vol_multiplier: float = 5.0, oi_multiplier: float = 3.0,
                 min_baseline_volume: float = 10.0):
        self.vol_window = vol_window
        self.oi_window = oi_window
        self.vol_multiplier = vol_multiplier
        self.oi_multiplier = oi_multiplier
        self.min_baseline_volume = min_baseline_volume
        self._vol_hist: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=vol_window + 1))
        self._oi_hist: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=oi_window + 1))

    def record(self, contract_symbol: str, volume: float, open_interest: float) -> None:
        self._vol_hist[contract_symbol].append(float(volume))
        self._oi_hist[contract_symbol].append(float(open_interest))

    def prune(self, active_symbols: set[str]) -> int:
        """Drop history for contracts no longer in the active chain. Returns
        the count evicted. Call monthly to bound memory — over 6yr of daily
        chains, dropped expiries accumulate to 100k+ contracts otherwise."""
        stale = [s for s in self._vol_hist if s not in active_symbols]
        for s in stale:
            self._vol_hist.pop(s, None)
            self._oi_hist.pop(s, None)
        return len(stale)

    def is_uoa(self, contract_symbol: str, current_volume: float) -> bool:
        vol = self._vol_hist.get(contract_symbol)
        oi = self._oi_hist.get(contract_symbol)
        if not vol or not oi:
            return False
        # Exclude today's volume from baseline by taking up-to-last-N items
        # (current_volume is passed separately).
        hist_vol = list(vol)[:-1] if len(vol) > 1 else list(vol)
        hist_oi = list(oi)[-self.oi_window:]
        return uoa_flag(
            current_volume=current_volume,
            hist_volume_20d=hist_vol,
            hist_oi_5d=hist_oi,
            vol_multiplier=self.vol_multiplier,
            oi_multiplier=self.oi_multiplier,
            min_baseline_volume=self.min_baseline_volume,
        )
