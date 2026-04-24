"""
MT5 CFD smoke test for Algotec.

Place at:
    trading_system/src/strategies/manual/mt5_cfd_smoke_test.py

Run:
    PYTHONPATH=. python -m src.strategies.manual.mt5_cfd_smoke_test \
        --symbols US100.cash,US500.cash,US30.cash,XAUUSD,USOIL.cash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2] if len(ROOT.parents) >= 3 else ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exchanges.mt5_cfd_connector import initialize, shutdown, get_symbol_spec, latest_closed_candles, export_symbol_specs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check MT5 connection, specs, and latest candles for CFD symbols.")
    parser.add_argument("--symbols", default="US100.cash,US500.cash,US30.cash,XAUUSD,USOIL.cash")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--bars", type=int, default=5)
    parser.add_argument("--mt5-path", default=None)
    parser.add_argument("--mt5-login", type=int, default=None)
    parser.add_argument("--mt5-password", default=None)
    parser.add_argument("--mt5-server", default=None)
    parser.add_argument("--export-json", default="src/data/mt5_cfd_cache/cfd_symbol_specs.json")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    initialize(path=args.mt5_path, login=args.mt5_login, password=args.mt5_password, server=args.mt5_server)
    try:
        for sym in symbols:
            spec = get_symbol_spec(sym)
            print("\n===", sym, "===")
            print("description:", spec.description)
            print("profit currency:", spec.currency_profit)
            print("tick_size:", spec.trade_tick_size)
            print("tick_value:", spec.trade_tick_value)
            print("usd_per_price_unit_per_lot:", spec.usd_per_price_unit_per_lot)
            print("volume min/step/max:", spec.volume_min, spec.volume_step, spec.volume_max)
            candles = latest_closed_candles(sym, args.timeframe, args.bars)
            print(candles.tail(args.bars))
        out = export_symbol_specs(symbols, args.export_json)
        print("\nWrote symbol specs ->", out)
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
