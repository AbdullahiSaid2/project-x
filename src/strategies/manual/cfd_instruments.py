"""
CFD instrument configuration for Algotec.

Place at:
    trading_system/src/strategies/manual/cfd_instruments.py

Purpose:
- Keep CFD symbol/config separate from futures configs.
- Map the current futures universe to CFD equivalents.
- Store broker/prop-platform symbol names, lot settings, and fallback $/point values.

Important:
- CFD symbols and contract specs are broker/platform specific.
- Always run the MT5 smoke test and export real symbol specs from the account you will trade.
- The defaults below are only safe placeholders until overridden by MT5 symbol specs JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional


@dataclass(frozen=True)
class CFDInstrumentConfig:
    key: str
    mt5_symbol: str
    futures_equivalent: str
    description: str
    timeframe: str = "5m"
    days_back: int = 365
    tail_rows: Optional[int] = None

    # Backtesting.py uses integer size. We model 1 internal unit as lot_unit lots.
    lot_unit: float = 0.01
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 10.0

    # Fallback USD value per 1.0 price unit per 1.0 lot.
    # Override from MT5 symbol specs whenever possible.
    usd_per_point_per_lot: float = 1.0

    # Optional assumptions for later modelling/reporting.
    spread_points: float = 0.0
    slippage_points: float = 0.0


# Futures -> CFD mapping.
# Adjust mt5_symbol values to match your exact prop/platform Market Watch names.
CFD_INSTRUMENTS: Dict[str, CFDInstrumentConfig] = {
    "US100": CFDInstrumentConfig(
        key="US100",
        mt5_symbol="US100.cash",
        futures_equivalent="MNQ/NQ",
        description="Nasdaq 100 CFD",
        usd_per_point_per_lot=1.0,
        max_lot=20.0,
    ),
    "US500": CFDInstrumentConfig(
        key="US500",
        mt5_symbol="US500.cash",
        futures_equivalent="MES/ES",
        description="S&P 500 CFD",
        usd_per_point_per_lot=1.0,
        max_lot=50.0,
    ),
    "US30": CFDInstrumentConfig(
        key="US30",
        mt5_symbol="US30.cash",
        futures_equivalent="MYM/YM",
        description="Dow Jones CFD",
        usd_per_point_per_lot=1.0,
        max_lot=20.0,
    ),
    "XAUUSD": CFDInstrumentConfig(
        key="XAUUSD",
        mt5_symbol="XAUUSD",
        futures_equivalent="MGC/GC",
        description="Gold CFD",
        usd_per_point_per_lot=100.0,
        max_lot=5.0,
    ),
    "USOIL": CFDInstrumentConfig(
        key="USOIL",
        mt5_symbol="USOIL.cash",
        futures_equivalent="MCL/CL",
        description="WTI Crude Oil CFD",
        usd_per_point_per_lot=100.0,
        max_lot=10.0,
    ),
}


def with_runtime_overrides(
    cfg: CFDInstrumentConfig,
    *,
    days_back: Optional[int] = None,
    tail_rows: Optional[int] = None,
    usd_per_point_per_lot: Optional[float] = None,
    min_lot: Optional[float] = None,
    lot_step: Optional[float] = None,
    max_lot: Optional[float] = None,
) -> CFDInstrumentConfig:
    return replace(
        cfg,
        days_back=cfg.days_back if days_back is None else int(days_back),
        tail_rows=tail_rows,
        usd_per_point_per_lot=cfg.usd_per_point_per_lot if usd_per_point_per_lot is None else float(usd_per_point_per_lot),
        min_lot=cfg.min_lot if min_lot is None else float(min_lot),
        lot_step=cfg.lot_step if lot_step is None else float(lot_step),
        max_lot=cfg.max_lot if max_lot is None else float(max_lot),
    )
