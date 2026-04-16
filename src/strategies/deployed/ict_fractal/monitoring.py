
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo
from config import LOG_DIR, MODEL_NAME

ET = ZoneInfo('America/New_York')
MONITOR_LOG = LOG_DIR / 'monitor_log.jsonl'

def now_et_iso() -> str:
    return datetime.now(timezone.utc).astimezone(ET).isoformat()

def log_monitor(event: str, **payload: Any) -> None:
    item = {'event': event, 'model_name': MODEL_NAME, 'logged_at_et': now_et_iso(), **payload}
    with MONITOR_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, default=str) + '\n')
