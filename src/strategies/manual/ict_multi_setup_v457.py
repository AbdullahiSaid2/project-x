from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd

from src.strategies.manual.ict_multi_setup_v455 import ICT_MULTI_SETUP_V455
from src.strategies.manual.ict_multi_setup_v454 import ICT_MULTI_SETUP_V454


@dataclass
class NewsWindow:
    event_time_et: pd.Timestamp
    title: str
    currency: str
    impact: str


class ICT_MULTI_SETUP_V457(ICT_MULTI_SETUP_V455):
    """
    V457 goals:
    - Keep the strong Apex intraday/session protections from the V455/V456 line.
    - Remove payout-day gating from execution logic.
    - Preserve daily PnL analytics.
    - Improve trade metadata capture so setup_type / bridge_type / tier / targets survive
      forced flatten, partials, BE exits, and delayed closes.
    """

    name = "ICT_MULTI_SETUP_V457"

    APEX_SESSION_START_ET = time(18, 0)
    APEX_SESSION_END_ET = time(16, 59)
    HARD_FLAT_ET = time(16, 55)
    GLOBAL_ENTRY_CUTOFF_ET = time(15, 45)
    NYPM_ENTRY_CUTOFF_ET = time(15, 15)

    NEWS_BLOCK_BEFORE_MIN = 5
    NEWS_BLOCK_AFTER_MIN = 5
    ENABLE_NEWS_BLOCK = True

    PROP_DAILY_LOSS_LIMIT = -1000.0
    PROP_MAX_DRAWDOWN_LIMIT = -2000.0

    TRADE_METADATA_LOG: List[Dict[str, Any]] = []
    DEBUG_COUNTERS: Dict[str, int] = defaultdict(int)

    def init(self) -> None:
        super().init()
        self._v457_news_windows: List[NewsWindow] = self._load_news_windows()
        self._v457_open_trade_session: Dict[int, Any] = {}
        self._v457_open_trade_meta_by_key: Dict[int, Dict[str, Any]] = {}
        self._v457_last_closed_count: int = 0
        self._v457_calendar_daily_realized_pnl = defaultdict(float)
        self._v457_apex_session_realized_pnl = defaultdict(float)
        self._v457_cycle_realized_pnl: float = 0.0
        self._v457_best_day_pnl_in_cycle: float = 0.0
        self._v457_equity_peak_est: float = 50000.0
        self._v457_account_balance_est: float = 50000.0
        self._v457_pending_meta_queue: Deque[Dict[str, Any]] = deque()
        self._v457_last_meta_snapshot: Dict[str, Any] | None = None
        type(self).TRADE_METADATA_LOG = []
        type(self).DEBUG_COUNTERS = defaultdict(int)

    def _current_timestamp_et(self) -> pd.Timestamp:
        idx = None
        if hasattr(self.data, "index"):
            try:
                idx = self.data.index[-1]
            except Exception:
                idx = None
        if idx is None and hasattr(self.data, "df"):
            try:
                idx = self.data.df.index[-1]
            except Exception:
                idx = None
        if idx is None:
            return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("America/New_York")
        ts = pd.Timestamp(idx)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("America/New_York")

    @staticmethod
    def _apex_session_date(ts_et: pd.Timestamp):
        if ts_et.time() >= time(18, 0):
            return ts_et.date()
        return (ts_et - pd.Timedelta(days=1)).date()

    def _is_after_hard_flat(self, ts_et: pd.Timestamp) -> bool:
        t = ts_et.time()
        return t >= self.HARD_FLAT_ET and t < self.APEX_SESSION_START_ET

    def _is_after_global_entry_cutoff(self, ts_et: pd.Timestamp) -> bool:
        t = ts_et.time()
        return self.GLOBAL_ENTRY_CUTOFF_ET <= t < self.APEX_SESSION_START_ET

    def _is_after_nypm_entry_cutoff(self, ts_et: pd.Timestamp) -> bool:
        t = ts_et.time()
        return self.NYPM_ENTRY_CUTOFF_ET <= t < self.APEX_SESSION_START_ET

    def _load_news_windows(self) -> List[NewsWindow]:
        if not self.ENABLE_NEWS_BLOCK:
            return []
        csv_path = Path(__file__).with_name("high_impact_news_et.csv")
        if not csv_path.exists():
            return []
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return []
        required = {"event_time_et", "title", "currency", "impact"}
        if not required.issubset(df.columns):
            return []
        df["event_time_et"] = pd.to_datetime(df["event_time_et"], errors="coerce")
        df = df.dropna(subset=["event_time_et"])
        out: List[NewsWindow] = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["event_time_et"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            else:
                ts = ts.tz_convert("America/New_York")
            out.append(NewsWindow(ts, str(row["title"]), str(row["currency"]), str(row["impact"]).upper()))
        return out

    def _in_news_block(self, ts_et: pd.Timestamp) -> bool:
        for event in self._v457_news_windows:
            start = event.event_time_et - pd.Timedelta(minutes=self.NEWS_BLOCK_BEFORE_MIN)
            end = event.event_time_et + pd.Timedelta(minutes=self.NEWS_BLOCK_AFTER_MIN)
            if start <= ts_et <= end:
                return True
        return False

    def _iter_open_trades(self) -> List[Any]:
        try:
            return list(self.trades)
        except Exception:
            return []

    def _close_all_open_trades(self, reason: str) -> None:
        open_trades = self._iter_open_trades()
        if reason == "news" and open_trades:
            type(self).DEBUG_COUNTERS["news_forced_flatten"] += 1
        for trade in open_trades:
            try:
                trade.close()
            except Exception:
                pass

    def _current_parent_setup_type(self) -> str:
        for attr in ("current_setup_type", "setup_type", "_setup_type", "active_setup_type"):
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    return str(val)
        return ""

    def _should_block_new_entries_for_time(self, now_et: pd.Timestamp) -> bool:
        setup_type = self._current_parent_setup_type().upper()
        if "NYPM" in setup_type:
            return self._is_after_nypm_entry_cutoff(now_et)
        return self._is_after_global_entry_cutoff(now_et)

    def _block_new_entries_and_run_parent(self, now_et: pd.Timestamp) -> None:
        orig_buy = getattr(self, "buy", None)
        orig_sell = getattr(self, "sell", None)

        def _blocked_buy(*args, **kwargs):
            if self._in_news_block(now_et):
                type(self).DEBUG_COUNTERS["prop_block_news"] += 1
            else:
                type(self).DEBUG_COUNTERS["prop_block_daily_trade_cap"] += 1
            return None

        def _blocked_sell(*args, **kwargs):
            if self._in_news_block(now_et):
                type(self).DEBUG_COUNTERS["prop_block_news"] += 1
            else:
                type(self).DEBUG_COUNTERS["prop_block_daily_trade_cap"] += 1
            return None

        try:
            if orig_buy is not None:
                self.buy = _blocked_buy
            if orig_sell is not None:
                self.sell = _blocked_sell
            # IMPORTANT: call the V454 core next() directly.
            # Calling super().next() here would re-enter V455.next(), which
            # calls self._block_new_entries_and_run_parent(...) again and
            # causes infinite recursion in V457.
            ICT_MULTI_SETUP_V454.next(self)
        finally:
            if orig_buy is not None:
                self.buy = orig_buy
            if orig_sell is not None:
                self.sell = orig_sell

    def _safe_get_trade_pnl(self, trade: Any) -> float:
        for attr in ("pl", "pnl", "profit_loss"):
            if hasattr(trade, attr):
                try:
                    return float(getattr(trade, attr))
                except Exception:
                    pass
        return 0.0

    def _safe_get_trade_size(self, trade: Any) -> float:
        if hasattr(trade, "size"):
            try:
                return float(trade.size)
            except Exception:
                pass
        return 0.0

    def _safe_get_trade_entry_price(self, trade: Any) -> Optional[float]:
        if hasattr(trade, "entry_price"):
            try:
                return float(trade.entry_price)
            except Exception:
                pass
        return None

    def _safe_get_trade_exit_price(self, trade: Any) -> Optional[float]:
        if hasattr(trade, "exit_price"):
            try:
                return float(trade.exit_price)
            except Exception:
                pass
        return None

    def _safe_get_trade_times(self, trade: Any):
        entry_time = None
        exit_time = None
        for attr in ("entry_time", "EntryTime"):
            if hasattr(trade, attr):
                try:
                    entry_time = pd.Timestamp(getattr(trade, attr))
                    break
                except Exception:
                    pass
        for attr in ("exit_time", "ExitTime"):
            if hasattr(trade, attr):
                try:
                    exit_time = pd.Timestamp(getattr(trade, attr))
                    break
                except Exception:
                    pass
        if entry_time is not None and entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC").tz_convert("America/New_York")
        elif entry_time is not None:
            entry_time = entry_time.tz_convert("America/New_York")
        if exit_time is not None and exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC").tz_convert("America/New_York")
        elif exit_time is not None:
            exit_time = exit_time.tz_convert("America/New_York")
        return entry_time, exit_time

    def _record_open_trade_meta(self, entry: float):
        super()._record_open_trade_meta(entry)
        snapshot = dict(getattr(self, "open_trade_meta", {}) or {})
        snapshot.setdefault("entry_variant", "PULLBACK_1M")
        snapshot["_entry_hint"] = float(entry)
        snapshot["_captured_at_et"] = self._current_timestamp_et()
        self._v457_pending_meta_queue.append(snapshot)
        self._v457_last_meta_snapshot = snapshot

    def _default_meta(self) -> Dict[str, Any]:
        return {
            "setup_type": self._current_parent_setup_type() or "",
            "bridge_type": getattr(self, "current_bridge_type", "") or "",
            "setup_tier": getattr(self, "current_setup_tier", "") or "",
            "entry_variant": getattr(self, "current_entry_variant", "PULLBACK_1M") or "PULLBACK_1M",
            "planned_entry_price": getattr(self, "current_planned_entry_price", None),
            "planned_stop_price": getattr(self, "current_planned_stop_price", None),
            "planned_target_price": getattr(self, "current_planned_target_price", None),
            "partial_target_price": getattr(self, "current_partial_target_price", None),
            "runner_target_price": getattr(self, "current_runner_target_price", None),
            "stop_points": getattr(self, "current_stop_points", None),
            "tp_points": getattr(self, "current_tp_points", None),
            "planned_rr": getattr(self, "current_planned_rr", None),
        }

    def _merge_meta(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base = self._default_meta()
        if meta:
            for k, v in meta.items():
                if k.startswith("_"):
                    continue
                if v is not None and v != "":
                    base[k] = v
        return base

    def _capture_new_open_trade_context(self, now_et: pd.Timestamp) -> None:
        open_trades = self._iter_open_trades()
        for trade in open_trades:
            key = id(trade)
            if key in self._v457_open_trade_session:
                continue
            self._v457_open_trade_session[key] = self._apex_session_date(now_et)

            entry_price = self._safe_get_trade_entry_price(trade)
            chosen_meta = None
            if self._v457_pending_meta_queue:
                if entry_price is not None:
                    best_idx = None
                    best_diff = float("inf")
                    for idx, candidate in enumerate(self._v457_pending_meta_queue):
                        hint = candidate.get("_entry_hint")
                        if hint is None:
                            continue
                        diff = abs(float(hint) - float(entry_price))
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = idx
                    if best_idx is not None and best_diff <= 8.0:
                        chosen_meta = self._v457_pending_meta_queue[best_idx]
                        del self._v457_pending_meta_queue[best_idx]
                if chosen_meta is None:
                    chosen_meta = self._v457_pending_meta_queue.popleft()
            elif self._v457_last_meta_snapshot is not None:
                chosen_meta = dict(self._v457_last_meta_snapshot)

            self._v457_open_trade_meta_by_key[key] = self._merge_meta(chosen_meta)

    def _lookup_trade_meta(self, trade: Any, entry_time_et: Optional[pd.Timestamp], entry_price: Optional[float]) -> Dict[str, Any]:
        best_meta: Optional[Dict[str, Any]] = None
        best_score = float("inf")

        for key, meta in list(self._v457_open_trade_meta_by_key.items()):
            hint = meta.get("planned_entry_price", meta.get("_entry_hint"))
            score = 999999.0
            if hint is not None and entry_price is not None:
                score = abs(float(hint) - float(entry_price))
            if score < best_score:
                best_score = score
                best_meta = meta

        if best_meta is None and self._v457_last_meta_snapshot is not None:
            best_meta = self._v457_last_meta_snapshot

        return self._merge_meta(best_meta)

    def _log_newly_closed_trades_v457(self) -> None:
        closed_trades = list(getattr(self, "closed_trades", []))
        if len(closed_trades) <= self._v457_last_closed_count:
            return
        new_trades = closed_trades[self._v457_last_closed_count:]
        self._v457_last_closed_count = len(closed_trades)

        for trade in new_trades:
            entry_time_et, exit_time_et = self._safe_get_trade_times(trade)
            pnl = self._safe_get_trade_pnl(trade)
            entry_price = self._safe_get_trade_entry_price(trade)
            exit_price = self._safe_get_trade_exit_price(trade)
            size = self._safe_get_trade_size(trade)

            if exit_time_et is not None:
                calendar_exit_day = exit_time_et.date()
                apex_exit_session = self._apex_session_date(exit_time_et)
                self._v457_calendar_daily_realized_pnl[calendar_exit_day] += pnl
                self._v457_apex_session_realized_pnl[apex_exit_session] += pnl
                self._v457_cycle_realized_pnl += pnl
                self._v457_account_balance_est = 50000.0 + self._v457_cycle_realized_pnl
                self._v457_equity_peak_est = max(self._v457_equity_peak_est, self._v457_account_balance_est)
                self._v457_best_day_pnl_in_cycle = max(
                    self._v457_best_day_pnl_in_cycle,
                    self._v457_calendar_daily_realized_pnl[calendar_exit_day],
                    self._v457_apex_session_realized_pnl[apex_exit_session],
                )

            realized_points = None
            if entry_price is not None and exit_price is not None:
                realized_points = (exit_price - entry_price) if size >= 0 else (entry_price - exit_price)

            meta = self._lookup_trade_meta(trade, entry_time_et, entry_price)
            planned_entry = meta.get("planned_entry_price")
            planned_stop = meta.get("planned_stop_price")
            planned_target = meta.get("planned_target_price")
            stop_points = meta.get("stop_points")
            tp_points = meta.get("tp_points")
            planned_rr = meta.get("planned_rr")
            if stop_points is None and planned_entry is not None and planned_stop is not None:
                stop_points = abs(float(planned_entry) - float(planned_stop))
            if tp_points is None and planned_entry is not None and planned_target is not None:
                tp_points = abs(float(planned_target) - float(planned_entry))
            if planned_rr is None and stop_points not in (None, 0) and tp_points is not None:
                try:
                    planned_rr = float(tp_points) / float(stop_points)
                except Exception:
                    planned_rr = None

            consistency_ratio = 0.0
            if abs(self._v457_cycle_realized_pnl) > 1e-9:
                consistency_ratio = self._v457_best_day_pnl_in_cycle / abs(self._v457_cycle_realized_pnl)

            row = {
                "side": "LONG" if size >= 0 else "SHORT",
                "setup_type": meta.get("setup_type", ""),
                "bridge_type": meta.get("bridge_type", ""),
                "setup_tier": meta.get("setup_tier", ""),
                "entry_variant": meta.get("entry_variant", "PULLBACK_1M") or "PULLBACK_1M",
                "planned_entry_price": planned_entry,
                "planned_stop_price": planned_stop,
                "planned_target_price": planned_target,
                "partial_target_price": meta.get("partial_target_price", None),
                "runner_target_price": meta.get("runner_target_price", None),
                "stop_points": stop_points,
                "tp_points": tp_points,
                "planned_rr": planned_rr,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "realized_points": realized_points,
                "realized_dollars_5_mnq": (realized_points * 10.0) if realized_points is not None else pnl,
                "return_pct": getattr(trade, "return_pct", getattr(trade, "pl_pct", None)),
                "pnl": pnl,
                "cycle_realized_pnl": self._v457_cycle_realized_pnl,
                "best_day_pnl_in_cycle": self._v457_best_day_pnl_in_cycle,
                "consistency_ratio": consistency_ratio,
                "prop_daily_loss_limit": self.PROP_DAILY_LOSS_LIMIT,
                "prop_max_drawdown_limit": self.PROP_MAX_DRAWDOWN_LIMIT,
                "entry_time_et": entry_time_et,
                "exit_time_et": exit_time_et,
                "entry_apex_session_date": self._apex_session_date(entry_time_et) if entry_time_et is not None else None,
                "exit_apex_session_date": self._apex_session_date(exit_time_et) if exit_time_et is not None else None,
                "calendar_exit_date_et": exit_time_et.date() if exit_time_et is not None else None,
            }
            type(self).TRADE_METADATA_LOG.append(row)

        if not self.position and hasattr(self, "open_trade_meta"):
            self.open_trade_meta = None

    def next(self) -> None:
        now_et = self._current_timestamp_et()

        if self._is_after_hard_flat(now_et):
            self._close_all_open_trades(reason="hard_flat")
            self._block_new_entries_and_run_parent(now_et)
            self._capture_new_open_trade_context(now_et)
            self._log_newly_closed_trades_v457()
            return

        for trade in self._iter_open_trades():
            key = id(trade)
            entry_session = self._v457_open_trade_session.get(key)
            if entry_session is None:
                self._v457_open_trade_session[key] = self._apex_session_date(now_et)
                continue
            current_session = self._apex_session_date(now_et)
            if current_session != entry_session:
                try:
                    trade.close()
                except Exception:
                    pass

        if self._in_news_block(now_et):
            self._close_all_open_trades(reason="news")
            self._block_new_entries_and_run_parent(now_et)
            self._capture_new_open_trade_context(now_et)
            self._log_newly_closed_trades_v457()
            return

        if self._should_block_new_entries_for_time(now_et):
            self._block_new_entries_and_run_parent(now_et)
        else:
            super().next()

        self._capture_new_open_trade_context(now_et)
        self._log_newly_closed_trades_v457()
