
from __future__ import annotations

import json
from typing import Any
from config import STATE_DIR

STATE_FILE = STATE_DIR / 'runtime_state.json'

DEFAULT_STATE = {
    'seen_signal_ids': [],
    'last_cycle': {},
    'last_force_flat_key': None,
    'last_force_flat_result': None,
    'unsupported_actions': [],
}

def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return dict(DEFAULT_STATE)
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        out = dict(DEFAULT_STATE)
        out.update(data)
        return out
    except Exception:
        return dict(DEFAULT_STATE)

def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
