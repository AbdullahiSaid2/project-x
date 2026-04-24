"""
Optional generic CFD restrictions for prop-style CFD accounts.

Place at:
    trading_system/src/strategies/manual/cfd_restrictions.py

Purpose:
- Keep additional CFD restrictions separate from the core prop profile file.
- Support optional news blackout and Friday/weekend new-entry blocks.
- The actual account limits should still come from prop_firm_profiles.py + prop_guard.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class RestrictedNewsCalendar:
    """CSV-backed restricted-news blackout checker.

    CSV columns accepted:
        time_utc, scope, description

    scope examples:
        USD, US_INDICES, GOLD, OIL, ALL, US100, US500, US30, XAUUSD, USOIL
    """

    def __init__(self, csv_path: Optional[str | Path] = None, blackout_minutes: int = 2):
        self.blackout_minutes = int(blackout_minutes)
        self.df = pd.DataFrame(columns=["time_utc", "scope", "description"])
        if csv_path:
            p = Path(csv_path)
            if p.exists():
                self.df = pd.read_csv(p)
                self.df["time_utc"] = pd.to_datetime(self.df["time_utc"], utc=True, errors="coerce")
                self.df["scope"] = self.df["scope"].astype(str).str.upper()
                self.df = self.df.dropna(subset=["time_utc"])

    @staticmethod
    def _symbol_scopes(symbol_key: str) -> set[str]:
        s = symbol_key.upper()
        scopes = {s, "ALL"}
        if s in {"US100", "US500", "US30"}:
            scopes.update({"USD", "US_INDICES"})
        if s == "XAUUSD":
            scopes.update({"USD", "GOLD"})
        if s == "USOIL":
            scopes.update({"OIL", "CRUDE", "USOIL"})
        return scopes

    def is_blackout(self, when, symbol_key: str) -> bool:
        if self.df.empty:
            return False
        ts = pd.Timestamp(when)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        delta = pd.Timedelta(minutes=self.blackout_minutes)
        scopes = self._symbol_scopes(symbol_key)
        candidates = self.df[self.df["scope"].isin(scopes)]
        if candidates.empty:
            return False
        return bool(((candidates["time_utc"] - ts).abs() <= delta).any())


def is_late_friday_cutoff(when, timezone: str = "Europe/Prague", cutoff_hour: int = 20) -> bool:
    """Conservative new-entry block late on Friday for standard prop-style accounts."""
    ts = pd.Timestamp(when)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(timezone)
    return local.weekday() == 4 and local.hour >= int(cutoff_hour)
