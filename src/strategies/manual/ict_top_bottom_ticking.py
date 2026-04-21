from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import sys
from typing import List

import numpy as np
import pandas as pd
from backtesting import Strategy

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3] if len(ROOT.parents) >= 4 else ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_TICK_SIZE = 0.25
FORCE_FLAT_HOUR_ET = 16
FORCE_FLAT_MINUTE_ET = 50
GLOBEX_REOPEN_HOUR_ET = 18


@dataclass
class PendingSetup:
    direction: str = ""
    created_bar: int = -1
    expiry_bar: int = -1
    entry_ce: float = np.nan
    zone_high: float = np.nan
    zone_low: float = np.nan
    stop_price: float = np.nan
    target1: float = np.nan
    target2: float = np.nan
    target3: float = np.nan
    external_level: float = np.nan
    setup_type: str = ""
    entry_variant: str = ""
    internal_sweep: bool = False


def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ts = pd.DatetimeIndex(idx)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York")


def _session_date_for_et(ts: pd.Series) -> pd.Series:
    # CME-style session label: 18:00 ET belongs to next session date.
    return (
        ts.dt.tz_localize(None)
        + pd.to_timedelta((ts.dt.hour >= GLOBEX_REOPEN_HOUR_ET).astype(int), unit="D")
    ).dt.date


def build_model_frame(df: pd.DataFrame, tick_size: float = DEFAULT_TICK_SIZE) -> pd.DataFrame:
    m = df.copy()
    m.columns = [c.capitalize() for c in m.columns]
    et = _to_et(pd.DatetimeIndex(m.index))
    m["et_time"] = et
    m["et_date"] = et.date
    m["et_hour"] = et.hour
    m["et_minute"] = et.minute
    m["session_date"] = _session_date_for_et(pd.Series(et, index=m.index))

    session_summary = (
        m.groupby("session_date")
        .agg(session_high=("High", "max"), session_low=("Low", "min"))
        .shift(1)
        .rename(columns={"session_high": "prior_session_high", "session_low": "prior_session_low"})
    )
    m = m.join(session_summary, on="session_date")

    asia_rows = m[m["et_hour"] >= 18]
    asia_summary = (
        asia_rows.groupby("session_date")
        .agg(asia_high=("High", "max"), asia_low=("Low", "min"))
        .rename(columns={"asia_high": "asia_high", "asia_low": "asia_low"})
    )
    m = m.join(asia_summary, on="session_date")

    m["external_buyside"] = m[["prior_session_high", "asia_high"]].max(axis=1, skipna=True)
    m["external_sellside"] = m[["prior_session_low", "asia_low"]].min(axis=1, skipna=True)

    res = (
        m[["Open", "High", "Low", "Close", "Volume"]]
        .resample("15min", label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    res["range"] = res["High"] - res["Low"]
    res["body"] = (res["Close"] - res["Open"]).abs()
    res["bull"] = res["Close"] > res["Open"]
    res["bear"] = res["Close"] < res["Open"]
    res["swing_high_15"] = res["High"].rolling(5, center=True).max().eq(res["High"])
    res["swing_low_15"] = res["Low"].rolling(5, center=True).min().eq(res["Low"])
    context = res[["High", "Low", "Close", "swing_high_15", "swing_low_15"]].rename(
        columns={"High": "ctx_high_15", "Low": "ctx_low_15", "Close": "ctx_close_15"}
    )
    m = pd.merge_asof(
        m.sort_index(),
        context.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    m["atr14"] = (m["High"] - m["Low"]).rolling(14).mean()
    m["recent_high_6"] = m["High"].rolling(6).max().shift(1)
    m["recent_low_6"] = m["Low"].rolling(6).min().shift(1)
    m["recent_high_12"] = m["High"].rolling(12).max().shift(1)
    m["recent_low_12"] = m["Low"].rolling(12).min().shift(1)

    m["sweep_short"] = (
        pd.notna(m["external_buyside"])
        & (m["High"] >= m["external_buyside"] + tick_size)
        & (m["Close"] <= m["external_buyside"])
    )
    m["sweep_long"] = (
        pd.notna(m["external_sellside"])
        & (m["Low"] <= m["external_sellside"] - tick_size)
        & (m["Close"] >= m["external_sellside"])
    )

    m["internal_sweep_short"] = m["High"] >= (m["recent_high_6"] + tick_size)
    m["internal_sweep_long"] = m["Low"] <= (m["recent_low_6"] - tick_size)

    m["cos_short"] = m["Close"] < m["recent_low_6"]
    m["cos_long"] = m["Close"] > m["recent_high_6"]

    return m


class ICT_TOP_BOTTOM_TICKING(Strategy):
    fixed_size = 5
    min_warmup_bars = 150
    setup_expiry_bars = 18
    limit_touch_tolerance_ticks = 1
    require_cos_confirmation = True
    require_internal_sweep_filter = False
    symbol_name = "UNKNOWN"
    tick_size = DEFAULT_TICK_SIZE
    min_sweep_points = DEFAULT_TICK_SIZE
    retest_tolerance_points = DEFAULT_TICK_SIZE
    stop_buffer_points = DEFAULT_TICK_SIZE
    max_zone_width_points = np.inf

    partial_1_fraction = 2 / 5
    partial_2_fraction_of_remaining = 2 / 3
    move_stop_to_be_after_t1 = True

    target1_r = 1.0
    target2_r = 2.25
    target3_r = 4.25

    min_stop_points = 6.0
    max_stop_points = 30.0

    last_trade_log: List[dict] = []
    last_debug_counts: dict = {}

    def init(self):
        self.m = build_model_frame(self.data.df.copy(), tick_size=float(self.tick_size))
        self.pending = PendingSetup()
        self.partial1_taken = False
        self.partial2_taken = False
        self.be_moved = False
        self.active_risk = np.nan
        self.open_trade_meta = None
        self.prev_closed_count = 0
        self.debug_counts = {
            "sweep_short_seen": 0,
            "sweep_long_seen": 0,
            "pending_short_armed": 0,
            "pending_long_armed": 0,
            "entry_short": 0,
            "entry_long": 0,
            "partial1": 0,
            "partial2": 0,
            "be_move": 0,
            "forced_flat": 0,
            "expired_pending": 0,
            "blocked_internal_filter": 0,
            "reject_short_sweep_too_small": 0,
            "reject_long_sweep_too_small": 0,
            "reject_short_zone_too_wide": 0,
            "reject_long_zone_too_wide": 0,
            "reject_short_stop_out_of_bounds": 0,
            "reject_long_stop_out_of_bounds": 0,
            "reject_short_confirmation_missing": 0,
            "reject_long_confirmation_missing": 0,
            "pending_short_not_touched": 0,
            "pending_long_not_touched": 0,
        }
        self.__class__.last_trade_log = []
        self.__class__.last_debug_counts = {}
        self._sync_debug()
        self._emit_event("init", {"symbol": self.symbol_name})

    def _log_date_str(self) -> str:
        try:
            ts = self.m.iloc[self._i()]["et_time"]
            return pd.Timestamp(ts).strftime("%Y%m%d")
        except Exception:
            return datetime.utcnow().strftime("%Y%m%d")

    def _events_log_path(self) -> Path:
        return LOG_DIR / f"strategy_events_{self._log_date_str()}.jsonl"

    def _snapshot_log_path(self) -> Path:
        return LOG_DIR / f"strategy_debug_snapshot_{self._log_date_str()}.json"

    def _safe(self, value):
        if isinstance(value, (np.floating, np.integer)):
            if pd.isna(value):
                return None
            return value.item()
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _pending_dict(self) -> dict:
        return {k: self._safe(v) for k, v in asdict(self.pending).items()}

    def _current_row_context(self) -> dict:
        try:
            row = self.m.iloc[self._i()]
            keys = [
                "et_time",
                "Open",
                "High",
                "Low",
                "Close",
                "external_buyside",
                "external_sellside",
                "sweep_short",
                "sweep_long",
                "internal_sweep_short",
                "internal_sweep_long",
                "cos_short",
                "cos_long",
            ]
            out = {}
            for k in keys:
                if k in row.index:
                    out[k] = self._safe(row[k])
            return out
        except Exception:
            return {}

    def _emit_event(self, event: str, payload: dict | None = None):
        body = {
            "ts_utc": datetime.utcnow().isoformat(),
            "event": event,
            "symbol": self.symbol_name,
            "bar_index": self._i() if len(self.data) else -1,
            "debug_counts": {k: self._safe(v) for k, v in self.debug_counts.items()},
            "pending": self._pending_dict(),
            "row": self._current_row_context(),
        }
        if payload:
            body["payload"] = {k: self._safe(v) for k, v in payload.items()}
        try:
            with self._events_log_path().open("a", encoding="utf-8") as f:
                f.write(json.dumps(body, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _write_snapshot(self):
        body = {
            "ts_utc": datetime.utcnow().isoformat(),
            "symbol": self.symbol_name,
            "bar_index": self._i() if len(self.data) else -1,
            "debug_counts": {k: self._safe(v) for k, v in self.debug_counts.items()},
            "pending": self._pending_dict(),
            "position_open": bool(self.position) if hasattr(self, "position") else False,
            "partial1_taken": bool(self.partial1_taken),
            "partial2_taken": bool(self.partial2_taken),
            "be_moved": bool(self.be_moved),
            "active_risk": self._safe(self.active_risk),
            "open_trade_meta": self.open_trade_meta,
            "row": self._current_row_context(),
        }
        try:
            with self._snapshot_log_path().open("w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    def _sync_debug(self):
        self.__class__.last_debug_counts = dict(self.debug_counts)
        self._write_snapshot()

    def _i(self) -> int:
        return len(self.data) - 1

    def _latest_trade(self):
        try:
            if self.trades:
                return self.trades[-1]
        except Exception:
            pass
        return None

    def _clear_pending(self):
        old_pending = self._pending_dict()
        self.pending = PendingSetup()
        self._emit_event("pending_cleared", {"old_pending": old_pending})

    def _after_force_flat_cutoff(self, row: pd.Series) -> bool:
        hour = int(row.get("et_hour", -1))
        minute = int(row.get("et_minute", -1))
        return (hour > FORCE_FLAT_HOUR_ET) or (hour == FORCE_FLAT_HOUR_ET and minute >= FORCE_FLAT_MINUTE_ET)

    def _force_flat_if_needed(self, row: pd.Series):
        if self.position and self._after_force_flat_cutoff(row):
            try:
                self.position.close()
                self.debug_counts["forced_flat"] += 1
                self._sync_debug()
                self._emit_event("forced_flat", {"reason": "after_force_flat_cutoff"})
            except Exception as exc:
                self._emit_event("forced_flat_error", {"error": str(exc)})

    def _record_open_trade_meta(self, entry: float):
        self.open_trade_meta = {
            "setup_type": self.pending.setup_type,
            "entry_variant": self.pending.entry_variant,
            "external_level": self.pending.external_level,
            "zone_high": self.pending.zone_high,
            "zone_low": self.pending.zone_low,
            "planned_entry_price": entry,
            "planned_stop_price": self.pending.stop_price,
            "planned_target1_price": self.pending.target1,
            "planned_target2_price": self.pending.target2,
            "planned_target3_price": self.pending.target3,
            "internal_sweep": self.pending.internal_sweep,
        }

    def _log_newly_closed_trades(self):
        try:
            closed = list(self.closed_trades)
        except Exception:
            return

        if len(closed) <= self.prev_closed_count:
            return

        new_items = closed[self.prev_closed_count:]
        for t in new_items:
            meta = self.open_trade_meta or {}
            trade_row = {
                "side": "LONG" if float(t.size) > 0 else "SHORT",
                "setup_type": meta.get("setup_type", ""),
                "entry_variant": meta.get("entry_variant", ""),
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "pnl": float(getattr(t, "pl", np.nan)),
                "return_pct": float(getattr(t, "pl_pct", np.nan)) if hasattr(t, "pl_pct") else np.nan,
                **meta,
            }
            self.__class__.last_trade_log.append(trade_row)
            self._emit_event("trade_closed", trade_row)

        if new_items and not self.position:
            self.open_trade_meta = None
        self.prev_closed_count = len(closed)

    def _count(self, key: str):
        self.debug_counts[key] = self.debug_counts.get(key, 0) + 1
        self._sync_debug()
        self._emit_event("debug_count_increment", {"key": key, "value": self.debug_counts[key]})

    def _short_sweep_distance(self, row: pd.Series) -> float:
        ext = row.get("external_buyside", np.nan)
        if pd.isna(ext):
            return 0.0
        return max(0.0, float(row["High"]) - float(ext))

    def _long_sweep_distance(self, row: pd.Series) -> float:
        ext = row.get("external_sellside", np.nan)
        if pd.isna(ext):
            return 0.0
        return max(0.0, float(ext) - float(row["Low"]))

    def _short_zone(self, row: pd.Series) -> tuple[float, float, float, float]:
        zone_high = float(row["High"])
        zone_low = min(float(row["Open"]), float(row["Close"]))
        entry_ce = (zone_high + zone_low) / 2.0
        zone_width = zone_high - zone_low
        return zone_high, zone_low, entry_ce, zone_width

    def _long_zone(self, row: pd.Series) -> tuple[float, float, float, float]:
        zone_low = float(row["Low"])
        zone_high = max(float(row["Open"]), float(row["Close"]))
        entry_ce = (zone_high + zone_low) / 2.0
        zone_width = zone_high - zone_low
        return zone_low, zone_high, entry_ce, zone_width

    def _arm_short_from_sweep(self, row: pd.Series, i: int):
        self._emit_event("arm_short_attempt", {"bar_index": i})

        if self.require_internal_sweep_filter and not bool(row.get("internal_sweep_short", False)):
            self._count("blocked_internal_filter")
            self._emit_event("arm_short_blocked_internal_filter", {})
            return

        sweep_distance = self._short_sweep_distance(row)
        if sweep_distance < float(self.min_sweep_points):
            self._count("reject_short_sweep_too_small")
            self._emit_event(
                "arm_short_rejected_sweep_too_small",
                {"sweep_distance": sweep_distance, "min_sweep_points": self.min_sweep_points},
            )
            return

        zone_high, zone_low, entry_ce, zone_width = self._short_zone(row)
        if zone_width > float(self.max_zone_width_points):
            self._count("reject_short_zone_too_wide")
            self._emit_event(
                "arm_short_rejected_zone_too_wide",
                {"zone_width": zone_width, "max_zone_width_points": self.max_zone_width_points},
            )
            return

        stop = zone_high + float(self.stop_buffer_points)
        risk = stop - entry_ce
        if risk < float(self.min_stop_points) or risk > float(self.max_stop_points):
            self._count("reject_short_stop_out_of_bounds")
            self._emit_event(
                "arm_short_rejected_stop_out_of_bounds",
                {
                    "risk": risk,
                    "min_stop_points": self.min_stop_points,
                    "max_stop_points": self.max_stop_points,
                    "entry_ce": entry_ce,
                    "stop": stop,
                },
            )
            return

        self.pending = PendingSetup(
            direction="short",
            created_bar=i,
            expiry_bar=i + int(self.setup_expiry_bars),
            entry_ce=entry_ce,
            zone_high=zone_high,
            zone_low=zone_low,
            stop_price=stop,
            target1=entry_ce - (risk * self.target1_r),
            target2=entry_ce - (risk * self.target2_r),
            target3=entry_ce - (risk * self.target3_r),
            external_level=float(row.get("external_buyside", np.nan)),
            setup_type="TYPE2_SHORT_TOP_TICK",
            entry_variant="CE_LIMIT" if not self.require_cos_confirmation else "CE_PLUS_COS",
            internal_sweep=bool(row.get("internal_sweep_short", False)),
        )
        self._count("pending_short_armed")
        self._emit_event("pending_short_armed", self._pending_dict())

    def _arm_long_from_sweep(self, row: pd.Series, i: int):
        self._emit_event("arm_long_attempt", {"bar_index": i})

        if self.require_internal_sweep_filter and not bool(row.get("internal_sweep_long", False)):
            self._count("blocked_internal_filter")
            self._emit_event("arm_long_blocked_internal_filter", {})
            return

        sweep_distance = self._long_sweep_distance(row)
        if sweep_distance < float(self.min_sweep_points):
            self._count("reject_long_sweep_too_small")
            self._emit_event(
                "arm_long_rejected_sweep_too_small",
                {"sweep_distance": sweep_distance, "min_sweep_points": self.min_sweep_points},
            )
            return

        zone_low, zone_high, entry_ce, zone_width = self._long_zone(row)
        if zone_width > float(self.max_zone_width_points):
            self._count("reject_long_zone_too_wide")
            self._emit_event(
                "arm_long_rejected_zone_too_wide",
                {"zone_width": zone_width, "max_zone_width_points": self.max_zone_width_points},
            )
            return

        stop = zone_low - float(self.stop_buffer_points)
        risk = entry_ce - stop
        if risk < float(self.min_stop_points) or risk > float(self.max_stop_points):
            self._count("reject_long_stop_out_of_bounds")
            self._emit_event(
                "arm_long_rejected_stop_out_of_bounds",
                {
                    "risk": risk,
                    "min_stop_points": self.min_stop_points,
                    "max_stop_points": self.max_stop_points,
                    "entry_ce": entry_ce,
                    "stop": stop,
                },
            )
            return

        self.pending = PendingSetup(
            direction="long",
            created_bar=i,
            expiry_bar=i + int(self.setup_expiry_bars),
            entry_ce=entry_ce,
            zone_high=zone_high,
            zone_low=zone_low,
            stop_price=stop,
            target1=entry_ce + (risk * self.target1_r),
            target2=entry_ce + (risk * self.target2_r),
            target3=entry_ce + (risk * self.target3_r),
            external_level=float(row.get("external_sellside", np.nan)),
            setup_type="TYPE2_LONG_BOTTOM_TICK",
            entry_variant="CE_LIMIT" if not self.require_cos_confirmation else "CE_PLUS_COS",
            internal_sweep=bool(row.get("internal_sweep_long", False)),
        )
        self._count("pending_long_armed")
        self._emit_event("pending_long_armed", self._pending_dict())

    def _pending_short_ready(self, row: pd.Series) -> bool:
        touched = float(row["High"]) >= (self.pending.entry_ce - float(self.retest_tolerance_points))
        if not touched:
            self._count("pending_short_not_touched")
            self._emit_event(
                "pending_short_not_touched",
                {"high": float(row["High"]), "entry_ce": self.pending.entry_ce},
            )
            return False

        if self.require_cos_confirmation:
            ready = bool(row.get("cos_short", False)) and float(row["Close"]) < self.pending.entry_ce
            if not ready:
                self._count("reject_short_confirmation_missing")
                self._emit_event(
                    "pending_short_confirmation_missing",
                    {"close": float(row["Close"]), "entry_ce": self.pending.entry_ce},
                )
            else:
                self._emit_event("pending_short_ready", {"confirmed": True})
            return ready

        ready = float(row["Close"]) <= self.pending.entry_ce
        if ready:
            self._emit_event("pending_short_ready", {"confirmed": False})
        return ready

    def _pending_long_ready(self, row: pd.Series) -> bool:
        touched = float(row["Low"]) <= (self.pending.entry_ce + float(self.retest_tolerance_points))
        if not touched:
            self._count("pending_long_not_touched")
            self._emit_event(
                "pending_long_not_touched",
                {"low": float(row["Low"]), "entry_ce": self.pending.entry_ce},
            )
            return False

        if self.require_cos_confirmation:
            ready = bool(row.get("cos_long", False)) and float(row["Close"]) > self.pending.entry_ce
            if not ready:
                self._count("reject_long_confirmation_missing")
                self._emit_event(
                    "pending_long_confirmation_missing",
                    {"close": float(row["Close"]), "entry_ce": self.pending.entry_ce},
                )
            else:
                self._emit_event("pending_long_ready", {"confirmed": True})
            return ready

        ready = float(row["Close"]) >= self.pending.entry_ce
        if ready:
            self._emit_event("pending_long_ready", {"confirmed": False})
        return ready

    def _enter_short(self, row: pd.Series, i: int):
        entry = min(float(row["Close"]), float(self.pending.entry_ce))
        stop = float(self.pending.stop_price)
        risk = stop - entry

        self._emit_event(
            "enter_short_attempt",
            {
                "bar_index": i,
                "close": float(row["Close"]),
                "planned_entry_ce": float(self.pending.entry_ce),
                "computed_entry": entry,
                "stop": stop,
                "risk": risk,
            },
        )

        if risk <= 0 or not np.isfinite(risk):
            self._emit_event("enter_short_aborted_invalid_risk", {"risk": risk})
            self._clear_pending()
            return

        target1 = entry - (risk * self.target1_r)
        target2 = entry - (risk * self.target2_r)
        target3 = entry - (risk * self.target3_r)

        if not (np.isfinite(target1) and np.isfinite(target2) and np.isfinite(target3)):
            self._emit_event("enter_short_aborted_invalid_targets", {})
            self._clear_pending()
            return

        if not (target3 < entry < stop):
            self._emit_event(
                "enter_short_aborted_invalid_price_structure",
                {"target3": target3, "entry": entry, "stop": stop},
            )
            self._clear_pending()
            return

        self.pending.entry_ce = entry
        self.pending.target1 = target1
        self.pending.target2 = target2
        self.pending.target3 = target3

        self.active_risk = risk
        self.sell(size=self.fixed_size, sl=stop, tp=target3)
        self.partial1_taken = False
        self.partial2_taken = False
        self.be_moved = False
        self._record_open_trade_meta(entry)
        self.debug_counts["entry_short"] += 1
        self._sync_debug()
        self._emit_event(
            "entry_short",
            {
                "entry": entry,
                "stop": stop,
                "target1": target1,
                "target2": target2,
                "target3": target3,
                "risk": risk,
                "size": self.fixed_size,
            },
        )
        self._clear_pending()

    def _enter_long(self, row: pd.Series, i: int):
        entry = max(float(row["Close"]), float(self.pending.entry_ce))
        stop = float(self.pending.stop_price)
        risk = entry - stop

        self._emit_event(
            "enter_long_attempt",
            {
                "bar_index": i,
                "close": float(row["Close"]),
                "planned_entry_ce": float(self.pending.entry_ce),
                "computed_entry": entry,
                "stop": stop,
                "risk": risk,
            },
        )

        if risk <= 0 or not np.isfinite(risk):
            self._emit_event("enter_long_aborted_invalid_risk", {"risk": risk})
            self._clear_pending()
            return

        target1 = entry + (risk * self.target1_r)
        target2 = entry + (risk * self.target2_r)
        target3 = entry + (risk * self.target3_r)

        if not (np.isfinite(target1) and np.isfinite(target2) and np.isfinite(target3)):
            self._emit_event("enter_long_aborted_invalid_targets", {})
            self._clear_pending()
            return

        if not (stop < entry < target3):
            self._emit_event(
                "enter_long_aborted_invalid_price_structure",
                {"stop": stop, "entry": entry, "target3": target3},
            )
            self._clear_pending()
            return

        self.pending.entry_ce = entry
        self.pending.target1 = target1
        self.pending.target2 = target2
        self.pending.target3 = target3

        self.active_risk = risk
        self.buy(size=self.fixed_size, sl=stop, tp=target3)
        self.partial1_taken = False
        self.partial2_taken = False
        self.be_moved = False
        self._record_open_trade_meta(entry)
        self.debug_counts["entry_long"] += 1
        self._sync_debug()
        self._emit_event(
            "entry_long",
            {
                "entry": entry,
                "stop": stop,
                "target1": target1,
                "target2": target2,
                "target3": target3,
                "risk": risk,
                "size": self.fixed_size,
            },
        )
        self._clear_pending()

    def _manage_trade(self):
        trade = self._latest_trade()
        if trade is None or not self.position or not np.isfinite(self.active_risk):
            return

        price = float(self.data.Close[-1])

        if self.position.is_long:
            r_now = (price - float(trade.entry_price)) / self.active_risk

            if not self.partial1_taken and r_now >= self.target1_r:
                try:
                    self.position.close(portion=self.partial_1_fraction)
                    self.partial1_taken = True
                    self.debug_counts["partial1"] += 1
                    self._sync_debug()
                    self._emit_event("partial1_long", {"r_now": r_now, "price": price})
                except Exception as exc:
                    self._emit_event("partial1_long_error", {"error": str(exc)})

            if self.move_stop_to_be_after_t1 and self.partial1_taken and not self.be_moved:
                try:
                    if trade.sl is not None:
                        trade.sl = max(float(trade.sl), float(trade.entry_price))
                    self.be_moved = True
                    self.debug_counts["be_move"] += 1
                    self._sync_debug()
                    self._emit_event("be_move_long", {"new_sl": float(trade.entry_price)})
                except Exception as exc:
                    self._emit_event("be_move_long_error", {"error": str(exc)})

            if self.partial1_taken and not self.partial2_taken and r_now >= self.target2_r:
                try:
                    self.position.close(portion=self.partial_2_fraction_of_remaining)
                    self.partial2_taken = True
                    self.debug_counts["partial2"] += 1
                    self._sync_debug()
                    self._emit_event("partial2_long", {"r_now": r_now, "price": price})
                except Exception as exc:
                    self._emit_event("partial2_long_error", {"error": str(exc)})

        elif self.position.is_short:
            r_now = (float(trade.entry_price) - price) / self.active_risk

            if not self.partial1_taken and r_now >= self.target1_r:
                try:
                    self.position.close(portion=self.partial_1_fraction)
                    self.partial1_taken = True
                    self.debug_counts["partial1"] += 1
                    self._sync_debug()
                    self._emit_event("partial1_short", {"r_now": r_now, "price": price})
                except Exception as exc:
                    self._emit_event("partial1_short_error", {"error": str(exc)})

            if self.move_stop_to_be_after_t1 and self.partial1_taken and not self.be_moved:
                try:
                    if trade.sl is not None:
                        trade.sl = min(float(trade.sl), float(trade.entry_price))
                    self.be_moved = True
                    self.debug_counts["be_move"] += 1
                    self._sync_debug()
                    self._emit_event("be_move_short", {"new_sl": float(trade.entry_price)})
                except Exception as exc:
                    self._emit_event("be_move_short_error", {"error": str(exc)})

            if self.partial1_taken and not self.partial2_taken and r_now >= self.target2_r:
                try:
                    self.position.close(portion=self.partial_2_fraction_of_remaining)
                    self.partial2_taken = True
                    self.debug_counts["partial2"] += 1
                    self._sync_debug()
                    self._emit_event("partial2_short", {"r_now": r_now, "price": price})
                except Exception as exc:
                    self._emit_event("partial2_short_error", {"error": str(exc)})

    def next(self):
        self._log_newly_closed_trades()
        i = self._i()
        if i < self.min_warmup_bars:
            return

        row = self.m.iloc[i]
        self._manage_trade()
        self._force_flat_if_needed(row)

        if self.pending.expiry_bar >= 0 and i > self.pending.expiry_bar:
            self.debug_counts["expired_pending"] += 1
            self._sync_debug()
            self._emit_event("pending_expired", {"expired_bar": i})
            self._clear_pending()

        if self.position:
            return

        if self._after_force_flat_cutoff(row):
            self._emit_event("after_force_flat_cutoff_skip", {})
            self._clear_pending()
            return

        if self.pending.direction == "short":
            if self._pending_short_ready(row):
                self._enter_short(row, i)
                return

        if self.pending.direction == "long":
            if self._pending_long_ready(row):
                self._enter_long(row, i)
                return

        if self.pending.direction:
            return

        if bool(row.get("sweep_short", False)):
            self.debug_counts["sweep_short_seen"] += 1
            self._sync_debug()
            self._emit_event(
                "sweep_short_seen",
                {
                    "external_buyside": row.get("external_buyside", np.nan),
                    "high": row.get("High", np.nan),
                    "close": row.get("Close", np.nan),
                },
            )
            self._arm_short_from_sweep(row, i)
            return

        if bool(row.get("sweep_long", False)):
            self.debug_counts["sweep_long_seen"] += 1
            self._sync_debug()
            self._emit_event(
                "sweep_long_seen",
                {
                    "external_sellside": row.get("external_sellside", np.nan),
                    "low": row.get("Low", np.nan),
                    "close": row.get("Close", np.nan),
                },
            )
            self._arm_long_from_sweep(row, i)
            return