from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
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
    USE_TRADINGVIEW_BARS,
)
from execution import execute_signal, force_flat_all_positions
from live_model import generate_live_signals
from state import load_state, save_state

ET = ZoneInfo('America/New_York')


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
        return ts.date().isoformat()
    previous_date = ts.date().fromordinal(ts.date().toordinal() - 1)
    return previous_date.isoformat()


def get_session_state(ts: datetime) -> SessionState:
    force_flat_start = _time_et(FORCE_FLAT_HOUR_ET, FORCE_FLAT_MINUTE_ET)
    globex_reopen = _time_et(GLOBEX_REOPEN_HOUR_ET, GLOBEX_REOPEN_MINUTE_ET)

    weekday = ts.weekday()
    time_of_day = ts.time().replace(tzinfo=None)
    session_key = _trading_session_key(ts)

    if weekday == 5:
        return SessionState(False, True, 'weekend_closed', session_key)
    if weekday == 6:
        if time_of_day < globex_reopen:
            return SessionState(False, True, 'preopen_sunday', session_key)
        return SessionState(True, False, 'session_open', session_key)
    if weekday in (0, 1, 2, 3):
        if force_flat_start <= time_of_day < globex_reopen:
            return SessionState(False, True, 'force_flat_window', session_key)
        return SessionState(True, False, 'session_open', session_key)
    if weekday == 4:
        if time_of_day < force_flat_start:
            return SessionState(True, False, 'session_open', session_key)
        return SessionState(False, True, 'weekend_closed', session_key)
    return SessionState(False, True, 'unknown_session_state', session_key)


def maybe_force_flat(state: dict[str, Any], session_state: SessionState) -> dict[str, Any] | None:
    if not FORCE_FLAT_ENABLED or not session_state.force_flat:
        return None

    last_force_flat_key = state.get('last_force_flat_key')
    if last_force_flat_key == session_state.session_key:
        return state.get('last_force_flat_result')

    result = force_flat_all_positions(
        reason=session_state.reason,
        session_key=session_state.session_key,
    )
    state['last_force_flat_key'] = session_state.session_key
    state['last_force_flat_result'] = result
    return result


def run_once() -> dict[str, Any]:
    state = load_state()
    seen_ids = set(state.get('seen_signal_ids', []))

    current_et = now_et()
    session_state = get_session_state(current_et)
    force_flat_result = maybe_force_flat(state, session_state)

    if not session_state.allow_entries:
        summary = {
            'model_name': MODEL_NAME,
            'mode': EXECUTION_MODE,
            'cycle_time_et': current_et.isoformat(),
            'signals_seen': 0,
            'orders_sent': 0,
            'skipped_reason': session_state.reason,
            'force_flat': session_state.force_flat,
            'force_flat_result': force_flat_result,
            'session_key': session_state.session_key,
            'data_source_mode': 'tradingview' if USE_TRADINGVIEW_BARS else 'fetcher',
        }
        state['last_cycle'] = summary
        save_state(state)
        print(summary)
        return summary

    signals = generate_live_signals()
    sent = 0
    for signal in signals:
        signal_id = signal['signal_id']
        if signal_id in seen_ids:
            continue
        result = execute_signal(signal)
        if result.get('ok'):
            sent += 1
            seen_ids.add(signal_id)

    state['seen_signal_ids'] = sorted(seen_ids)[-5000:]
    state['orders_sent'] = int(state.get('orders_sent', 0)) + sent
    state['last_cycle'] = {
        'model_name': MODEL_NAME,
        'mode': EXECUTION_MODE,
        'cycle_time_et': current_et.isoformat(),
        'signals_seen': len(signals),
        'orders_sent': sent,
        'session_reason': session_state.reason,
        'session_key': session_state.session_key,
        'data_source_mode': 'tradingview' if USE_TRADINGVIEW_BARS else 'fetcher',
    }
    save_state(state)
    print(state['last_cycle'])
    return state['last_cycle']


def main() -> None:
    source = 'tradingview bars' if USE_TRADINGVIEW_BARS else 'fetcher bars'
    print(f'[{MODEL_NAME}] starting exact-v473 loop in mode={EXECUTION_MODE} source={source}')
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            print(f'[{MODEL_NAME}] stopped by user')
            break
        except Exception as exc:
            print(f'[{MODEL_NAME}] cycle error: {exc}')
        time.sleep(LOOP_SECONDS)


if __name__ == '__main__':
    main()
