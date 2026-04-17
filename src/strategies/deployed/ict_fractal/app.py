from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

THIS_DIR = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def _load_local_module(module_name: str, filename: str):
    path = THIS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Force sibling modules by file path so src/config.py cannot shadow them
_config = _load_local_module("ict_fractal_local_config", "config.py")
_execution = _load_local_module("ict_fractal_local_execution", "execution.py")
_live_model = _load_local_module("ict_fractal_local_live_model", "live_model.py")
_monitoring = _load_local_module("ict_fractal_local_monitoring", "monitoring.py")
_state = _load_local_module("ict_fractal_local_state", "state.py")

EXECUTION_MODE = _config.EXECUTION_MODE
FORCE_FLAT_ENABLED = _config.FORCE_FLAT_ENABLED
FORCE_FLAT_HOUR_ET = _config.FORCE_FLAT_HOUR_ET
FORCE_FLAT_MINUTE_ET = _config.FORCE_FLAT_MINUTE_ET
GLOBEX_REOPEN_HOUR_ET = _config.GLOBEX_REOPEN_HOUR_ET
GLOBEX_REOPEN_MINUTE_ET = _config.GLOBEX_REOPEN_MINUTE_ET
LOOP_SECONDS = _config.LOOP_SECONDS
MODEL_NAME = _config.MODEL_NAME

execute_signal = _execution.execute_signal
force_flat_all_positions = _execution.force_flat_all_positions
generate_live_actions = _live_model.generate_live_actions
get_active_data_source_name = _live_model.get_active_data_source_name
log_monitor = _monitoring.log_monitor
load_state = _state.load_state
save_state = _state.save_state


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(str(val).strip())
    except ValueError:
        return default


DEPLOY_MODE = os.getenv("DEPLOY_MODE", "").strip().lower()
PROP_PROFILE = os.getenv("PROP_PROFILE", "none").strip()
LIVE_ORDERS = _env_bool("LIVE_ORDERS", False)
MODEL_NAME_ENV = os.getenv("MODEL_NAME", "").strip()

# New runtime controls
CLOSED_LOOP_SECONDS = _env_int("ICT_FRACTAL_CLOSED_LOOP_SECONDS", 60)
HEARTBEAT_EVERY_CLOSED_CYCLES = max(1, _env_int("ICT_FRACTAL_HEARTBEAT_EVERY_CLOSED_CYCLES", 5))
HEARTBEAT_EVERY_OPEN_CYCLES = max(1, _env_int("ICT_FRACTAL_HEARTBEAT_EVERY_OPEN_CYCLES", 1))

RUNTIME_MODE = DEPLOY_MODE or str(EXECUTION_MODE).strip().lower()
RUNTIME_MODEL_NAME = MODEL_NAME_ENV or MODEL_NAME
ALLOW_LIVE_SENDS = (RUNTIME_MODE == "live") and LIVE_ORDERS


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


def maybe_force_flat(state: dict[str, Any], session_state: SessionState) -> dict[str, Any] | None:
    if not FORCE_FLAT_ENABLED or not session_state.force_flat:
        return None

    if state.get("last_force_flat_key") == session_state.session_key:
        return state.get("last_force_flat_result")

    if not ALLOW_LIVE_SENDS:
        result = {
            "ok": True,
            "simulated": True,
            "live_orders_enabled": False,
            "reason": f"force_flat_{session_state.reason}_simulated_only",
            "session_key": session_state.session_key,
        }
    else:
        result = force_flat_all_positions(
            reason=session_state.reason,
            session_key=session_state.session_key,
        )

    state["last_force_flat_key"] = session_state.session_key
    state["last_force_flat_result"] = result
    return result


def _process_action(action: dict[str, Any], session_state: SessionState) -> dict[str, Any]:
    if action.get("action_type") == "entry":
        allowed_payload = {
            "pre_entry_allowed": True,
            "model_name": RUNTIME_MODEL_NAME,
            "mode": RUNTIME_MODE,
            "prop_profile": PROP_PROFILE,
            "live_orders_enabled": ALLOW_LIVE_SENDS,
            "symbol": action.get("symbol"),
            "side": action.get("side"),
            "qty": action.get("qty"),
            "signal_id": action.get("signal_id"),
        }
        print(allowed_payload)

    if not ALLOW_LIVE_SENDS:
        return {
            "ok": True,
            "simulated": True,
            "live_orders_enabled": False,
            "mode": RUNTIME_MODE,
            "prop_profile": PROP_PROFILE,
            "reason": "LIVE_ORDERS=0 or mode is not live",
        }

    return execute_signal(action)


def run_once() -> dict[str, Any]:
    state = load_state()
    seen_ids = set(state.get("seen_signal_ids", []))
    current_et = now_et()
    session_state = get_session_state(current_et)
    force_flat_result = maybe_force_flat(state, session_state)

    actions = generate_live_actions() if session_state.allow_entries else []
    sent = 0
    seen_now = 0
    blocked_by_session = 0

    for action in actions:
        action_id = action["signal_id"]
        seen_now += 1

        if action_id in seen_ids:
            continue

        if action.get("action_type") == "entry" and not session_state.allow_entries:
            blocked_by_session += 1
            blocked_payload = {
                "pre_entry_blocked": True,
                "blocked_reason": f"session_{session_state.reason}",
                "model_name": RUNTIME_MODEL_NAME,
                "mode": RUNTIME_MODE,
                "prop_profile": PROP_PROFILE,
                "live_orders_enabled": ALLOW_LIVE_SENDS,
                "symbol": action.get("symbol"),
                "side": action.get("side"),
                "qty": action.get("qty"),
                "signal_id": action.get("signal_id"),
                "session_key": session_state.session_key,
            }
            print(blocked_payload)
            log_monitor(
                "entry_blocked_by_session",
                action=action,
                session_reason=session_state.reason,
                session_key=session_state.session_key,
                runtime_mode=RUNTIME_MODE,
                prop_profile=PROP_PROFILE,
            )
            seen_ids.add(action_id)
            continue

        result = _process_action(action, session_state)

        log_monitor(
            "action_processed",
            action=action,
            result=result,
            session_key=session_state.session_key,
            runtime_mode=RUNTIME_MODE,
            prop_profile=PROP_PROFILE,
            live_orders_enabled=ALLOW_LIVE_SENDS,
        )

        if result.get("ok"):
            sent += 1

        seen_ids.add(action_id)

        if not result.get("ok"):
            unsupported = state.get("unsupported_actions", [])
            unsupported.append(
                {
                    "action": action,
                    "result": result,
                    "logged_at_et": current_et.isoformat(),
                }
            )
            state["unsupported_actions"] = unsupported[-200:]

    summary = {
        "model_name": RUNTIME_MODEL_NAME,
        "mode": RUNTIME_MODE,
        "prop_profile": PROP_PROFILE,
        "live_orders_enabled": ALLOW_LIVE_SENDS,
        "cycle_time_et": current_et.isoformat(),
        "signals_seen": seen_now,
        "orders_sent": sent,
        "blocked_by_session": blocked_by_session,
        "session_reason": session_state.reason,
        "session_key": session_state.session_key,
        "market_open": session_state.allow_entries,
        "force_flat_result": force_flat_result,
        "data_source_mode": get_active_data_source_name(),
        "next_sleep_seconds": LOOP_SECONDS if session_state.allow_entries else CLOSED_LOOP_SECONDS,
    }

    state["seen_signal_ids"] = sorted(seen_ids)[-10000:]
    state["last_cycle"] = summary
    save_state(state)

    return summary


def _should_print_summary(cycle_num: int, summary: dict[str, Any]) -> bool:
    if summary["market_open"]:
        return cycle_num % HEARTBEAT_EVERY_OPEN_CYCLES == 0
    return cycle_num % HEARTBEAT_EVERY_CLOSED_CYCLES == 0


def main() -> None:
    print(
        f"[{RUNTIME_MODEL_NAME}] starting exact-v473 loop in mode={RUNTIME_MODE} "
        f"prop_profile={PROP_PROFILE} live_orders={'on' if ALLOW_LIVE_SENDS else 'off'} "
        f"source={get_active_data_source_name()} open_sleep={LOOP_SECONDS}s closed_sleep={CLOSED_LOOP_SECONDS}s"
    )

    cycle_num = 0

    while True:
        try:
            cycle_num += 1
            summary = run_once()

            if _should_print_summary(cycle_num, summary):
                if summary["market_open"]:
                    print(summary)
                else:
                    print(
                        {
                            "model_name": summary["model_name"],
                            "mode": summary["mode"],
                            "cycle_time_et": summary["cycle_time_et"],
                            "market_open": summary["market_open"],
                            "session_reason": summary["session_reason"],
                            "data_source_mode": summary["data_source_mode"],
                            "next_sleep_seconds": summary["next_sleep_seconds"],
                            "heartbeat": "idle_closed_market",
                        }
                    )

            sleep_seconds = summary["next_sleep_seconds"]

        except KeyboardInterrupt:
            print(f"[{RUNTIME_MODEL_NAME}] stopped by user")
            break
        except Exception as exc:
            log_monitor(
                "cycle_error",
                error=str(exc),
                runtime_mode=RUNTIME_MODE,
                prop_profile=PROP_PROFILE,
                live_orders_enabled=ALLOW_LIVE_SENDS,
            )
            print(f"[{RUNTIME_MODEL_NAME}] cycle error: {exc}")
            sleep_seconds = LOOP_SECONDS

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
