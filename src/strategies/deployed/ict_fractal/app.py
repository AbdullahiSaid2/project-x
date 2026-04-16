from __future__ import annotations

"""
How to run this
---------------
Place this file at:
    trading_system/src/strategies/deployed/ict_fractal/app.py

This app can be launched either directly or through:
    python src/strategies/deployed/run_models.py --models ict_fractal ...

Recommended usage through the launcher:
    python src/strategies/deployed/run_models.py --models ict_fractal --prop-profile apex_pa_50k --mode live --no-live-orders
    python src/strategies/deployed/run_models.py --models ict_fractal --prop-profile apex_pa_50k --mode live --live-orders

Direct usage:
    PYTHONPATH=. python src/strategies/deployed/ict_fractal/app.py

Environment variables
---------------------
Generic launcher-level vars:
    DEPLOY_MODE=dry|demo|live
    PROP_PROFILE=none|apex_pa_50k|apex_50k_eval|...
    LIVE_ORDERS=0|1
    MODEL_NAME=ict_fractal

Legacy / model-specific vars still supported:
    ICT_FRACTAL_EXECUTION_MODE=demo|live
    ICT_FRACTAL_* other settings remain supported through config.py

Source of truth precedence
--------------------------
1) Generic env vars from run_models.py
2) Legacy ICT_FRACTAL_* env vars
3) config.py defaults
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
MANUAL_DIR = PROJECT_ROOT / "src" / "strategies" / "manual"
DEPLOYED_DIR = PROJECT_ROOT / "src" / "strategies" / "deployed"

for p in (PROJECT_ROOT, MANUAL_DIR, DEPLOYED_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from config import (  # type: ignore
    EXECUTION_MODE as CONFIG_EXECUTION_MODE,
    FORCE_FLAT_ENABLED,
    FORCE_FLAT_HOUR_ET,
    FORCE_FLAT_MINUTE_ET,
    GLOBEX_REOPEN_HOUR_ET,
    GLOBEX_REOPEN_MINUTE_ET,
    LOOP_SECONDS,
    MODEL_NAME as CONFIG_MODEL_NAME,
    USE_TRADINGVIEW_BARS,
)
from execution import execute_signal, force_flat_all_positions  # type: ignore
from live_model import generate_live_actions  # type: ignore
from monitoring import log_monitor  # type: ignore
from state import load_state, save_state  # type: ignore

ET = ZoneInfo("America/New_York")


# -------- environment helpers --------

def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


GENERIC_MODE = _env_str("DEPLOY_MODE", "")
LEGACY_MODE = _env_str("ICT_FRACTAL_EXECUTION_MODE", "")
EXECUTION_MODE = (GENERIC_MODE or LEGACY_MODE or CONFIG_EXECUTION_MODE).strip().lower()

MODEL_NAME = _env_str("MODEL_NAME", CONFIG_MODEL_NAME or "ict_fractal")
PROP_PROFILE_NAME = _env_str("PROP_PROFILE", "none")
LIVE_ORDERS = _env_bool("LIVE_ORDERS", False)

# Alias map for convenience
PROP_PROFILE_ALIASES = {
    "apex_pa_50k": "apex_50k_eval",
    "apex_50k_pa": "apex_50k_eval",
    "apex_eval_50k": "apex_50k_eval",
}


def emit_log(event: str, **payload: Any) -> None:
    msg = {"event": event, **payload}
    print(json.dumps(msg, default=str))
    try:
        log_monitor(event, **payload)
    except Exception:
        pass


# -------- prop guard integration --------

class _NoopGuard:
    profile = type("NoopProfile", (), {"name": "none"})()

    def can_open_trade(self, trade_day):
        return type("Decision", (), {"allowed": True, "reason": None})()

    def on_trade_closed(self, pnl_dollars: float, trade_day):
        return None

    def export_state(self) -> dict[str, Any]:
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        return None


class PersistentPropGuard:
    def __init__(self, runtime_guard):
        self._guard = runtime_guard

    def can_open_trade(self, trade_day):
        return self._guard.can_open_trade(trade_day)

    def on_trade_closed(self, pnl_dollars: float, trade_day):
        self._guard.on_trade_closed(pnl_dollars, trade_day)

    def export_state(self) -> dict[str, Any]:
        return {
            "balance": getattr(self._guard, "balance", None),
            "high_watermark": getattr(self._guard, "high_watermark", None),
            "current_day": getattr(self._guard, "current_day", None),
            "day_realized": getattr(self._guard, "day_realized", None),
            "trades_today": getattr(self._guard, "trades_today", None),
            "consecutive_losses_today": getattr(self._guard, "consecutive_losses_today", None),
            "block_counts": getattr(self._guard, "block_counts", None),
        }

    def restore_state(self, saved: dict[str, Any]) -> None:
        if not saved:
            return
        for key in (
            "balance",
            "high_watermark",
            "current_day",
            "day_realized",
            "trades_today",
            "consecutive_losses_today",
            "block_counts",
        ):
            if key in saved:
                setattr(self._guard, key, saved[key])


def build_prop_guard(profile_name: str):
    normalized = PROP_PROFILE_ALIASES.get(profile_name, profile_name)
    if normalized == "none":
        return _NoopGuard(), "none"

    import_error = None
    try:
        from src.strategies.manual.prop_firm_profiles import get_prop_profile  # type: ignore
        from src.strategies.manual.prop_guard import PropFirmGuard  # type: ignore
    except Exception as exc1:
        import_error = exc1
        try:
            from prop_firm_profiles import get_prop_profile  # type: ignore
            from prop_guard import PropFirmGuard  # type: ignore
        except Exception as exc2:
            import_error = exc2
            emit_log(
                "prop_guard_unavailable",
                requested_profile=profile_name,
                normalized_profile=normalized,
                error=str(import_error),
            )
            return _NoopGuard(), "none"

    try:
        profile = get_prop_profile(normalized)
        guard = PersistentPropGuard(PropFirmGuard(profile))
        return guard, normalized
    except Exception as exc:
        emit_log(
            "prop_profile_load_failed",
            requested_profile=profile_name,
            normalized_profile=normalized,
            error=str(exc),
        )
        return _NoopGuard(), "none"


# -------- session logic --------

@dataclass(frozen=True)
class SessionState:
    allow_entries: bool
    force_flat: bool
    reason: str
    session_key: str


def now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(ET)


def _time_et(hour: int, minute: int) -> dt_time:
    return dt_time(hour=hour, minute=minute)


def _trading_session_key(ts: datetime) -> str:
    reopen_time = _time_et(GLOBEX_REOPEN_HOUR_ET, GLOBEX_REOPEN_MINUTE_ET)
    if ts.time().replace(tzinfo=None) >= reopen_time:
        return (ts.date() + timedelta(days=1)).isoformat()
    return ts.date().isoformat()


def get_session_state(ts: datetime) -> SessionState:
    force_flat_start = _time_et(FORCE_FLAT_HOUR_ET, FORCE_FLAT_MINUTE_ET)
    globex_reopen = _time_et(GLOBEX_REOPEN_HOUR_ET, GLOBEX_REOPEN_MINUTE_ET)
    weekday = ts.weekday()
    tod = ts.time().replace(tzinfo=None)
    session_key = _trading_session_key(ts)
    if weekday == 5:
        return SessionState(False, True, "weekend_closed", session_key)
    if weekday == 6:
        if tod < globex_reopen:
            return SessionState(False, True, "preopen_sunday", session_key)
        return SessionState(True, False, "session_open", session_key)
    if weekday in (0, 1, 2, 3):
        if force_flat_start <= tod < globex_reopen:
            return SessionState(False, True, "force_flat_window", session_key)
        return SessionState(True, False, "session_open", session_key)
    if weekday == 4:
        if tod < force_flat_start:
            return SessionState(True, False, "session_open", session_key)
        return SessionState(False, True, "weekend_closed", session_key)
    return SessionState(False, True, "unknown_session_state", session_key)


def _extract_trade_day(current_et: datetime, action: dict[str, Any]) -> str:
    raw_ts = (
        action.get("timestamp_et")
        or action.get("time_et")
        or action.get("bar_time_et")
        or current_et.isoformat()
    )
    try:
        ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ET)
        ts = ts.astimezone(ET)
        return ts.date().isoformat()
    except Exception:
        return current_et.date().isoformat()


def _extract_realized_pnl(result: dict[str, Any], action: dict[str, Any]) -> float | None:
    candidates = [
        result.get("realized_pnl_dollars"),
        result.get("pnl_dollars"),
        result.get("realized_pnl"),
        result.get("pnl"),
        action.get("realized_pnl_dollars"),
        action.get("pnl_dollars"),
        action.get("realized_pnl"),
        action.get("pnl"),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _is_closing_action(action: dict[str, Any]) -> bool:
    action_type = str(action.get("action_type", "")).lower()
    return action_type in {"exit", "close", "flatten", "force_flat", "reduce", "take_profit", "stop_loss"}


def maybe_force_flat(state: dict[str, Any], session_state: SessionState) -> dict[str, Any] | None:
    if not FORCE_FLAT_ENABLED or not session_state.force_flat:
        return None
    if state.get("last_force_flat_key") == session_state.session_key:
        return state.get("last_force_flat_result")

    if not LIVE_ORDERS:
        result = {
            "ok": True,
            "simulated": True,
            "reason": session_state.reason,
            "session_key": session_state.session_key,
            "live_orders": False,
        }
        emit_log(
            "force_flat_skipped_live_orders_disabled",
            reason=session_state.reason,
            session_key=session_state.session_key,
        )
    else:
        result = force_flat_all_positions(reason=session_state.reason, session_key=session_state.session_key)
        emit_log(
            "force_flat_executed",
            reason=session_state.reason,
            session_key=session_state.session_key,
            result=result,
        )

    state["last_force_flat_key"] = session_state.session_key
    state["last_force_flat_result"] = result
    return result


def run_once() -> dict[str, Any]:
    state = load_state()
    seen_ids = set(state.get("seen_signal_ids", []))
    current_et = now_et()
    session_state = get_session_state(current_et)
    force_flat_result = maybe_force_flat(state, session_state)

    guard, active_profile_name = build_prop_guard(PROP_PROFILE_NAME)
    guard.restore_state(state.get("prop_guard_state", {}))

    actions = generate_live_actions()
    sent = 0
    seen_now = 0
    blocked = 0

    for action in actions:
        action_id = action["signal_id"]
        seen_now += 1
        if action_id in seen_ids:
            continue

        action_type = str(action.get("action_type", "")).lower()
        trade_day = _extract_trade_day(current_et, action)

        if action_type == "entry":
            if not session_state.allow_entries:
                emit_log(
                    "pre_entry_blocked",
                    model_name=MODEL_NAME,
                    blocked_reason="session",
                    action=action,
                    session_reason=session_state.reason,
                    session_key=session_state.session_key,
                )
                seen_ids.add(action_id)
                blocked += 1
                continue

            decision = guard.can_open_trade(trade_day)
            if not getattr(decision, "allowed", False):
                emit_log(
                    "pre_entry_blocked",
                    model_name=MODEL_NAME,
                    blocked_reason=getattr(decision, "reason", "unknown"),
                    action=action,
                    prop_profile=active_profile_name,
                    session_key=session_state.session_key,
                    trade_day=trade_day,
                )
                seen_ids.add(action_id)
                blocked += 1
                continue

            emit_log(
                "pre_entry_allowed",
                model_name=MODEL_NAME,
                action=action,
                prop_profile=active_profile_name,
                session_key=session_state.session_key,
                trade_day=trade_day,
            )

        if not LIVE_ORDERS:
            result = {
                "ok": True,
                "simulated": True,
                "live_orders": False,
                "mode": EXECUTION_MODE,
            }
            emit_log(
                "action_skipped_live_orders_disabled",
                model_name=MODEL_NAME,
                action=action,
                result=result,
                prop_profile=active_profile_name,
                session_key=session_state.session_key,
            )
        else:
            result = execute_signal(action)
            emit_log(
                "action_processed",
                model_name=MODEL_NAME,
                action=action,
                result=result,
                prop_profile=active_profile_name,
                session_key=session_state.session_key,
            )

        if result.get("ok"):
            sent += 1
        else:
            unsupported = state.get("unsupported_actions", [])
            unsupported.append(
                {
                    "action": action,
                    "result": result,
                    "logged_at_et": current_et.isoformat(),
                }
            )
            state["unsupported_actions"] = unsupported[-200:]

        if _is_closing_action(action):
            pnl = _extract_realized_pnl(result, action)
            if pnl is not None:
                guard.on_trade_closed(pnl, trade_day)
                emit_log(
                    "prop_guard_trade_closed_update",
                    model_name=MODEL_NAME,
                    prop_profile=active_profile_name,
                    pnl_dollars=pnl,
                    trade_day=trade_day,
                )

        seen_ids.add(action_id)

    state["seen_signal_ids"] = sorted(seen_ids)[-10000:]
    state["prop_guard_state"] = guard.export_state()

    summary = {
        "model_name": MODEL_NAME,
        "mode": EXECUTION_MODE,
        "prop_profile": active_profile_name,
        "live_orders": LIVE_ORDERS,
        "cycle_time_et": current_et.isoformat(),
        "signals_seen": seen_now,
        "orders_sent": sent,
        "entries_blocked": blocked,
        "session_reason": session_state.reason,
        "session_key": session_state.session_key,
        "force_flat_result": force_flat_result,
        "data_source_mode": "tradingview" if USE_TRADINGVIEW_BARS else "fetcher",
    }
    state["last_cycle"] = summary
    save_state(state)
    print(summary)
    return summary


def main() -> None:
    print(
        f"[{MODEL_NAME}] starting loop | mode={EXECUTION_MODE} | "
        f"prop_profile={PROP_PROFILE_NAME} | live_orders={'on' if LIVE_ORDERS else 'off'} | "
        f"source={'tradingview bars' if USE_TRADINGVIEW_BARS else 'fetcher'}"
    )
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            print(f"[{MODEL_NAME}] stopped by user")
            break
        except Exception as exc:
            emit_log("cycle_error", model_name=MODEL_NAME, error=str(exc))
            print(f"[{MODEL_NAME}] cycle error: {exc}")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
