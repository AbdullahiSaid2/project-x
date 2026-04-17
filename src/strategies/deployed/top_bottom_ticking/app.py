from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config import (
    EXECUTION_MODE,
    FORCE_FLAT_ENABLED,
    FORCE_FLAT_HOUR_ET,
    FORCE_FLAT_MINUTE_ET,
    GLOBEX_REOPEN_HOUR_ET,
    GLOBEX_REOPEN_MINUTE_ET,
    LOOP_SECONDS,
    MODEL_NAME,
)
from execution import execute_signal, force_flat_all_positions
from live_model import generate_live_signals
from state import load_state, save_state, mark_force_flat

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(ET)


def _minutes_since_midnight_et(ts: datetime) -> int:
    return ts.hour * 60 + ts.minute


def _force_flat_cutoff_minutes() -> int:
    return FORCE_FLAT_HOUR_ET * 60 + FORCE_FLAT_MINUTE_ET


def _globex_reopen_minutes() -> int:
    return GLOBEX_REOPEN_HOUR_ET * 60 + GLOBEX_REOPEN_MINUTE_ET


def is_blocked_by_flat_window(ts: datetime) -> bool:
    """
    Block only in the flat window between force-flat cutoff and Globex reopen.
    Example:
      cutoff  = 16:50 ET
      reopen  = 18:00 ET
    Then:
      16:50-17:59 => blocked
      18:00+      => allowed again
    """
    if not FORCE_FLAT_ENABLED:
        return False

    mins = _minutes_since_midnight_et(ts)
    cutoff = _force_flat_cutoff_minutes()
    reopen = _globex_reopen_minutes()

    return cutoff <= mins < reopen


def current_session_key(ts: datetime) -> str:
    return ts.date().isoformat()


def maybe_force_flat_once_per_session(ts: datetime) -> dict[str, Any] | None:
    if not FORCE_FLAT_ENABLED:
        return None

    if not is_blocked_by_flat_window(ts):
        return None

    state = load_state()
    last_force_flat = state.get("last_force_flat") or {}
    session_key = current_session_key(ts)

    if last_force_flat.get("session_key") == session_key:
        return {
            "ok": True,
            "action": "already_force_flatted_this_session",
            "session_key": session_key,
        }

    result = force_flat_all_positions(
        reason="force_flat_cutoff_window",
        session_key=session_key,
    )
    mark_force_flat(
        reason="force_flat_cutoff_window",
        session_key=session_key,
        extra={"result": result},
    )
    return result


def run_once() -> dict[str, Any]:
    state = load_state()
    seen_ids = set(state.get("seen_signal_ids", []))

    current_et = now_et()

    force_flat_result = maybe_force_flat_once_per_session(current_et)

    if is_blocked_by_flat_window(current_et):
        summary = {
            "model_name": MODEL_NAME,
            "mode": EXECUTION_MODE,
            "cycle_time_et": current_et.isoformat(),
            "signals_seen": 0,
            "orders_sent": 0,
            "skipped_reason": "force_flat_window",
            "force_flat_result": force_flat_result,
        }
        state["last_cycle"] = summary
        save_state(state)
        print(summary)
        return summary

    signals = generate_live_signals()
    sent = 0

    for signal in signals:
        signal_id = signal["signal_id"]
        if signal_id in seen_ids:
            continue

        result = execute_signal(signal)
        if result.get("ok"):
            sent += 1
            seen_ids.add(signal_id)

    state["seen_signal_ids"] = sorted(seen_ids)[-5000:]
    state["orders_sent"] = int(state.get("orders_sent", 0)) + sent
    state["last_cycle"] = {
        "model_name": MODEL_NAME,
        "mode": EXECUTION_MODE,
        "cycle_time_et": current_et.isoformat(),
        "signals_seen": len(signals),
        "orders_sent": sent,
        "force_flat_result": force_flat_result,
    }
    save_state(state)
    print(state["last_cycle"])
    return state["last_cycle"]


def main() -> None:
    print(f"[{MODEL_NAME}] starting live loop in mode={EXECUTION_MODE}")
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            print(f"[{MODEL_NAME}] stopped by user")
            break
        except Exception as exc:
            print(f"[{MODEL_NAME}] cycle error: {exc}")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()