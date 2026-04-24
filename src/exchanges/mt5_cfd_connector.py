"""
MT5 CFD connector for Algotec.

Place at:
    trading_system/src/exchanges/mt5_cfd_connector.py

Use cases:
- Pull historical CFD candles from the exact MT5 account/server you plan to use.
- Read latest closed live candles/ticks for forward/live execution.
- Read symbol specifications for correct CFD $/point and lot sizing.
- Send market orders with SL/TP via MT5.

Requires:
    pip install MetaTrader5

MT5 terminal must be installed, open, logged in, and Algo Trading enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - lets repo import without MT5 installed
    mt5 = None


TIMEFRAME_MAP = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}


@dataclass(frozen=True)
class MT5SymbolSpec:
    symbol: str
    description: str
    currency_profit: str
    point: float
    trade_tick_size: float
    trade_tick_value: float
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    trade_stops_level: int
    digits: int

    @property
    def usd_per_price_unit_per_lot(self) -> float:
        if self.trade_tick_size <= 0:
            return 0.0
        return self.trade_tick_value / self.trade_tick_size

    @property
    def usd_per_point_per_lot(self) -> float:
        # In this codebase, "point" means one full price unit unless overridden.
        return self.usd_per_price_unit_per_lot


def _require_mt5():
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed. Run: pip install MetaTrader5")
    return mt5


def initialize(path: Optional[str] = None, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None) -> None:
    m = _require_mt5()
    kwargs = {}
    if path:
        kwargs["path"] = path
    if login is not None:
        kwargs["login"] = int(login)
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server
    ok = m.initialize(**kwargs)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {m.last_error()}")


def shutdown() -> None:
    if mt5 is not None:
        mt5.shutdown()


def ensure_symbol(symbol: str) -> None:
    m = _require_mt5()
    info = m.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"MT5 symbol not found: {symbol}")
    if not info.visible:
        if not m.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select MT5 symbol in Market Watch: {symbol}")


def get_timeframe(timeframe: str):
    m = _require_mt5()
    key = TIMEFRAME_MAP.get(timeframe.lower())
    if key is None:
        raise ValueError(f"Unsupported timeframe {timeframe}. Use one of: {sorted(TIMEFRAME_MAP)}")
    return getattr(m, key)


def get_symbol_spec(symbol: str) -> MT5SymbolSpec:
    m = _require_mt5()
    ensure_symbol(symbol)
    info = m.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"No MT5 symbol info for {symbol}")
    return MT5SymbolSpec(
        symbol=symbol,
        description=str(getattr(info, "description", "")),
        currency_profit=str(getattr(info, "currency_profit", "")),
        point=float(getattr(info, "point", 0.0)),
        trade_tick_size=float(getattr(info, "trade_tick_size", 0.0)),
        trade_tick_value=float(getattr(info, "trade_tick_value", 0.0)),
        contract_size=float(getattr(info, "trade_contract_size", 0.0)),
        volume_min=float(getattr(info, "volume_min", 0.0)),
        volume_step=float(getattr(info, "volume_step", 0.0)),
        volume_max=float(getattr(info, "volume_max", 0.0)),
        trade_stops_level=int(getattr(info, "trade_stops_level", 0)),
        digits=int(getattr(info, "digits", 0)),
    )


def export_symbol_specs(symbols: list[str], output_json: str | Path) -> Path:
    specs = [asdict(get_symbol_spec(s)) for s in symbols]
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(specs).to_json(out, orient="records", indent=2)
    return out


def load_symbol_specs_json(path: str | Path) -> Dict[str, MT5SymbolSpec]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_json(p)
    specs: Dict[str, MT5SymbolSpec] = {}
    for row in df.to_dict(orient="records"):
        spec = MT5SymbolSpec(**row)
        specs[spec.symbol] = spec
    return specs


def copy_rates_range(symbol: str, timeframe: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    m = _require_mt5()
    ensure_symbol(symbol)
    tf = get_timeframe(timeframe)
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)
    rates = m.copy_rates_range(symbol, tf, start_utc, end_utc)
    if rates is None:
        raise RuntimeError(f"MT5 copy_rates_range failed for {symbol}: {m.last_error()}")
    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    out = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"})
    return out[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def get_historical_candles(
    symbol: str,
    timeframe: str = "5m",
    days_back: int = 365,
    cache_dir: str | Path = "src/data/mt5_cfd_cache",
    refresh: bool = False,
) -> pd.DataFrame:
    safe_symbol = symbol.replace(".", "_").replace("/", "_")
    cache_path = Path(cache_dir) / f"{safe_symbol}_{timeframe}_{days_back}d.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        print(f"📦 Using MT5 CFD cache: {cache_path}")
        return pd.read_parquet(cache_path)

    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=int(days_back))
    print(f"📡 Fetching MT5 candles: symbol={symbol} timeframe={timeframe} start={start.isoformat()} end={end.isoformat()}")
    df = copy_rates_range(symbol, timeframe, start, end)
    df.to_parquet(cache_path)
    print(f"💾 Wrote MT5 CFD cache: {cache_path}")
    return df


def latest_closed_candles(symbol: str, timeframe: str = "5m", count: int = 200) -> pd.DataFrame:
    m = _require_mt5()
    ensure_symbol(symbol)
    tf = get_timeframe(timeframe)
    rates = m.copy_rates_from_pos(symbol, tf, 1, int(count))  # position 1 = last closed candle
    if rates is None:
        raise RuntimeError(f"MT5 copy_rates_from_pos failed for {symbol}: {m.last_error()}")
    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"})[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def account_info() -> dict:
    m = _require_mt5()
    info = m.account_info()
    if info is None:
        raise RuntimeError(f"MT5 account_info failed: {m.last_error()}")
    return info._asdict()


def send_market_order(
    *,
    symbol: str,
    side: str,
    lots: float,
    sl: float,
    tp: float,
    deviation: int = 20,
    magic: int = 240424,
    comment: str = "Algotec CFD",
):
    m = _require_mt5()
    ensure_symbol(symbol)
    tick = m.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No live tick for {symbol}")
    side_u = side.upper()
    if side_u == "BUY":
        order_type = m.ORDER_TYPE_BUY
        price = float(tick.ask)
    elif side_u == "SELL":
        order_type = m.ORDER_TYPE_SELL
        price = float(tick.bid)
    else:
        raise ValueError("side must be BUY or SELL")

    request = {
        "action": m.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": int(deviation),
        "magic": int(magic),
        "comment": comment,
        "type_time": m.ORDER_TIME_GTC,
        "type_filling": m.ORDER_FILLING_IOC,
    }
    result = m.order_send(request)
    if result is None:
        raise RuntimeError(f"MT5 order_send returned None: {m.last_error()}")
    return result
