from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import STATE_DIR

STATE_FILE = STATE_DIR / 'runtime_state.json'


def _default_state() -> dict[str, Any]:
    return {
        'seen_signal_ids': [],
        'orders_sent': 0,
        'last_cycle': None,
        'last_force_flat': None,
        'last_error': None,
    }


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return _default_state()

    try:
        with STATE_FILE.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_state()
        base = _default_state()
        base.update(data)
        if not isinstance(base.get('seen_signal_ids'), list):
            base['seen_signal_ids'] = []
        return base
    except Exception as exc:
        state = _default_state()
        state['last_error'] = f'state_load_failed: {exc}'
        return state


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, sort_keys=True, default=str)
    tmp.replace(STATE_FILE)


def mark_force_flat(reason: str, session_key: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    state = load_state()
    payload = {
        'reason': reason,
        'session_key': session_key,
    }
    if extra:
        payload.update(extra)
    state['last_force_flat'] = payload
    save_state(state)
    return state
