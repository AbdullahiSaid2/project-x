from __future__ import annotations

import json
from typing import Any

from config import STATE_DIR

STATE_FILE = STATE_DIR / 'runtime_state.json'


def default_state() -> dict[str, Any]:
    return {
        'seen_signal_ids': [],
        'orders_sent': 0,
        'last_cycle': None,
    }


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return default_state()
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return default_state()


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')
