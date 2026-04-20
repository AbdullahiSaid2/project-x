from __future__ import annotations

import os
import threading
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import databento as db
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

DATABENTO_LIVE_DATASET = "GLBX.MDP3"
DATABENTO_LIVE_SCHEMA = "ohlcv-1s"

LIVE_PARENT_MAP: Dict[str, str] = {
    "MNQ": "MNQ.FUT",
    "MES": "MES.FUT",
    "MYM": "MYM.FUT",
    "MGC": "MGC.FUT",
    "MCL": "MCL.FUT",
    "NQ": "NQ.FUT",
    "ES": "ES.FUT",
    "YM": "YM.FUT",
    "GC": "GC.FUT",
    "CL": "CL.FUT",
}

RESAMPLE_RULES = {
    "1s": None,
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}

_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_SCALE = 1e9


def _empty_ohlcv_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=_REQUIRED_COLS)
    df.index = pd.to_datetime(df.index)
    return df


def _utc_naive_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _to_utc_naive_ts(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.tz_convert("UTC").tz_localize(None)
    except Exception:
        pass
    try:
        ts = pd.to_datetime(int(value), unit="ns", utc=True)
        return ts.tz_convert("UTC").tz_localize(None)
    except Exception:
        return None


def _normalize_root(text: str) -> str:
    s = str(text).upper().strip()
    if s.endswith(".FUT"):
        s = s[:-4]
    if "." in s:
        s = s.split(".", 1)[0]
    return s


def _map_symbol_text_to_root(text: str) -> Optional[str]:
    s = _normalize_root(text)
    if s in LIVE_PARENT_MAP:
        return s
    for root in sorted(LIVE_PARENT_MAP.keys(), key=len, reverse=True):
        if s.startswith(root):
            return root
    return None


class DatabentoLiveOHLCVService:
    def __init__(
        self,
        api_key: str,
        *,
        dataset: str = DATABENTO_LIVE_DATASET,
        schema: str = DATABENTO_LIVE_SCHEMA,
        symbols: Optional[list[str]] = None,
        replay_minutes: int = 1440,
        max_rows_per_symbol: int = 50000,
    ):
        self.api_key = api_key
        self.dataset = dataset
        self.schema = schema
        self.symbols = symbols or list(LIVE_PARENT_MAP.keys())
        self.replay_minutes = max(0, int(replay_minutes))
        self.max_rows_per_symbol = max(1000, int(max_rows_per_symbol))

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._client = None
        self._instrument_to_symbol: Dict[int, str] = {}
        self._bars_1s: Dict[str, pd.DataFrame] = {root: _empty_ohlcv_df() for root in self.symbols}
        self._last_error: Optional[str] = None

        self._mapping_debug_seen: set[tuple[int, str]] = set()
        self._mcl_debug_count = 0
        self._mgc_bad_debug_count = 0

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self) -> "DatabentoLiveOHLCVService":
        if self.is_running:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="databento-live-ohlcv", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        client = self._client
        if client is not None:
            try:
                client.terminate()
            except Exception:
                try:
                    client.stop()
                except Exception:
                    pass

    def _subscribe_symbols(self) -> list[str]:
        out = []
        for root in self.symbols:
            parent = LIVE_PARENT_MAP.get(root.upper())
            if parent:
                out.append(parent)
        return out

    def _live_replay_start(self):
        if self.replay_minutes <= 0:
            return None
        start = datetime.now(timezone.utc) - timedelta(minutes=self.replay_minutes)
        return start.isoformat()

    def _run(self) -> None:
        try:
            self._client = db.Live(
                key=self.api_key,
                reconnect_policy="reconnect",
            )

            subscribe_kwargs = {
                "dataset": self.dataset,
                "schema": self.schema,
                "stype_in": "parent",
                "symbols": self._subscribe_symbols(),
            }
            replay_start = self._live_replay_start()
            if replay_start is not None:
                subscribe_kwargs["start"] = replay_start

            self._client.subscribe(**subscribe_kwargs)
            self._client.add_callback(self._on_record)

            print(
                f"[databento_live] started | dataset={self.dataset} | schema={self.schema} | "
                f"symbols={self.symbols} | replay_minutes={self.replay_minutes}"
            )

            self._client.start()
            self._client.block_for_close(timeout=None)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"[databento_live] fatal error: {self._last_error}")
            traceback.print_exc()
        finally:
            self._client = None

    def _on_record(self, record) -> None:
        if self._stop_event.is_set():
            return

        try:
            cls_name = type(record).__name__.lower()

            if "symbolmapping" in cls_name or hasattr(record, "stype_in_symbol") or hasattr(record, "stype_out_symbol"):
                self._handle_symbol_mapping(record)
                return

            if all(hasattr(record, attr) for attr in ("open", "high", "low", "close", "volume")):
                self._handle_ohlcv_record(record)
                return
        except Exception as exc:
            print(f"[databento_live] record handling error: {exc}")

    def _handle_symbol_mapping(self, record) -> None:
        instrument_id = getattr(record, "instrument_id", None)
        if instrument_id is None:
            return

        symbol_text = None
        for attr in ("stype_in_symbol", "stype_out_symbol", "raw_symbol", "symbol"):
            value = getattr(record, attr, None)
            if value:
                symbol_text = value
                break

        if not symbol_text:
            return

        root = _map_symbol_text_to_root(symbol_text)
        if root is None:
            return

        with self._lock:
            self._instrument_to_symbol[int(instrument_id)] = root

        debug_key = (int(instrument_id), root)
        if debug_key not in self._mapping_debug_seen:
            self._mapping_debug_seen.add(debug_key)
            print(
                f"[databento_live] symbol mapping | instrument_id={instrument_id} | "
                f"raw_symbol={symbol_text} | mapped_root={root}"
            )

    def _resolve_record_symbol(self, record) -> Optional[str]:
        for attr in ("symbol", "raw_symbol", "stype_in_symbol", "stype_out_symbol"):
            value = getattr(record, attr, None)
            if value:
                root = _map_symbol_text_to_root(value)
                if root is not None:
                    return root

        instrument_id = getattr(record, "instrument_id", None)
        if instrument_id is not None:
            return self._instrument_to_symbol.get(int(instrument_id))

        return None


    def _row_is_obviously_bad(self, root: str, o: float, h: float, l: float, c: float) -> bool:
        values = (o, h, l, c)
        if any(pd.isna(v) for v in values):
            return True
        if any(v <= 0 for v in values):
            return True

        scaled = (o / PRICE_SCALE, h / PRICE_SCALE, l / PRICE_SCALE, c / PRICE_SCALE)

        # MGC is currently the only feed that is repeatedly arriving with poisoned
        # zero/invalid OHLC packets in live mode. Drop those packets at source so
        # they never make it into the in-memory live bar store.
        if root == "MGC":
            if min(scaled) < 500.0 or max(scaled) > 10000.0:
                return True

        return False

    def _handle_ohlcv_record(self, record) -> None:
        root = self._resolve_record_symbol(record)
        if root is None:
            return

        ts = _to_utc_naive_ts(getattr(record, "ts_event", None))
        if ts is None:
            return

        raw_open = float(getattr(record, "open"))
        raw_high = float(getattr(record, "high"))
        raw_low = float(getattr(record, "low"))
        raw_close = float(getattr(record, "close"))

        if self._row_is_obviously_bad(root, raw_open, raw_high, raw_low, raw_close):
            if root == "MGC" and self._mgc_bad_debug_count < 10:
                self._mgc_bad_debug_count += 1
                print(
                    "[databento_live] dropping poisoned MGC row | "
                    f"ts={ts.isoformat()} | raw_open={raw_open} raw_high={raw_high} "
                    f"raw_low={raw_low} raw_close={raw_close} | "
                    f"scaled_close={raw_close / PRICE_SCALE}"
                )
            return

        row = pd.DataFrame(
            [{
                "Open": raw_open / PRICE_SCALE,
                "High": raw_high / PRICE_SCALE,
                "Low": raw_low / PRICE_SCALE,
                "Close": raw_close / PRICE_SCALE,
                "Volume": float(getattr(record, "volume")),
            }],
            index=[ts],
        )

        if root == "MCL" and self._mcl_debug_count < 10:
            self._mcl_debug_count += 1
            print(
                "[databento_live] MCL raw debug | "
                f"ts={ts.isoformat()} | raw_open={raw_open} raw_high={raw_high} "
                f"raw_low={raw_low} raw_close={raw_close} | "
                f"scaled_close={raw_close / PRICE_SCALE}"
            )

        with self._lock:
            existing = self._bars_1s.get(root)
            if existing is None or existing.empty:
                out = row
            else:
                out = pd.concat([existing, row])
                out = out[~out.index.duplicated(keep="last")]
                out = out.sort_index()

            if len(out) > self.max_rows_per_symbol:
                out = out.tail(self.max_rows_per_symbol)

            self._bars_1s[root] = out

    def _drop_incomplete_last_bar(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df.empty or timeframe == "1s":
            return df

        now = _utc_naive_now()
        tf = timeframe.strip()

        if tf == "1m":
            current_bucket = now.floor("min")
        elif tf == "3m":
            current_bucket = now.floor("3min")
        elif tf == "5m":
            current_bucket = now.floor("5min")
        elif tf == "15m":
            current_bucket = now.floor("15min")
        elif tf == "1H":
            current_bucket = now.floor("1h")
        elif tf == "4H":
            current_bucket = now.floor("4h")
        elif tf == "1D":
            current_bucket = now.floor("1d")
        else:
            return df

        if not df.empty and df.index[-1] >= current_bucket:
            return df.iloc[:-1]
        return df

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1m",
        *,
        tail_rows: int = 500,
    ) -> pd.DataFrame:
        root = _map_symbol_text_to_root(symbol)
        if root is None:
            return _empty_ohlcv_df()

        with self._lock:
            base = self._bars_1s.get(root, _empty_ohlcv_df()).copy()

        if base.empty:
            return base

        tf = timeframe.strip()
        rule = RESAMPLE_RULES.get(tf)
        if tf not in RESAMPLE_RULES:
            raise ValueError(f"Unsupported live timeframe: {timeframe}")

        if tf == "1s":
            out = base
        else:
            out = (
                base.resample(rule)
                .agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                })
                .dropna()
            )
            out = self._drop_incomplete_last_bar(out, tf)

        return out.tail(tail_rows)


_SERVICE: Optional[DatabentoLiveOHLCVService] = None
_SERVICE_LOCK = threading.Lock()


def get_live_service(symbols: Optional[list[str]] = None) -> DatabentoLiveOHLCVService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            api_key = os.getenv("DATABENTO_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("DATABENTO_API_KEY is not set")

            replay_minutes = int(os.getenv("DATABENTO_LIVE_REPLAY_MINUTES", "1440"))
            max_rows = int(os.getenv("DATABENTO_LIVE_MAX_ROWS", "50000"))

            _SERVICE = DatabentoLiveOHLCVService(
                api_key=api_key,
                symbols=symbols or list(LIVE_PARENT_MAP.keys()),
                replay_minutes=replay_minutes,
                max_rows_per_symbol=max_rows,
            ).start()

        return _SERVICE


def get_live_ohlcv(symbol: str, timeframe: str = "1m", tail_rows: int = 500) -> pd.DataFrame:
    service = get_live_service()
    return service.get_bars(symbol, timeframe=timeframe, tail_rows=tail_rows)


def get_latest_live_ohlcv(
    symbol: str,
    exchange: str = "tradovate",
    timeframe: str = "1m",
    days_back: int = 1,
    tail_rows: int = 500,
) -> pd.DataFrame:
    service = get_live_service()
    return service.get_bars(symbol, timeframe=timeframe, tail_rows=tail_rows)


def stop_live_service() -> None:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is not None:
            _SERVICE.stop()
            _SERVICE = None