from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from zoneinfo import ZoneInfo

from config import (
    EXECUTION_MODE,
    FORCE_FLAT_HOUR_ET,
    FORCE_FLAT_MINUTE_ET,
    LOOP_SECONDS,
    MODEL_NAME,
)
from execution import execute_signal
from live_model import generate_live_signals
from state import load_state, save_state

ET = ZoneInfo('America/New_York')


def now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(ET)


def after_force_flat_cutoff(ts: datetime) -> bool:
    return (ts.hour > FORCE_FLAT_HOUR_ET) or (
        ts.hour == FORCE_FLAT_HOUR_ET and ts.minute >= FORCE_FLAT_MINUTE_ET
    )


def run_once() -> dict[str, Any]:
    state = load_state()
    seen_ids = set(state.get('seen_signal_ids', []))

    current_et = now_et()
    if after_force_flat_cutoff(current_et):
        summary = {
            'model_name': MODEL_NAME,
            'mode': EXECUTION_MODE,
            'cycle_time_et': current_et.isoformat(),
            'signals_seen': 0,
            'orders_sent': 0,
            'skipped_reason': 'after_force_flat_cutoff',
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
    }
    save_state(state)
    print(state['last_cycle'])
    return state['last_cycle']


def main() -> None:
    print(f'[{MODEL_NAME}] starting exact-v473 loop in mode={EXECUTION_MODE}')
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
