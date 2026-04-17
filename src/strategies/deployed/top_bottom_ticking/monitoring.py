from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "top_bottom_ticking_monitor.jsonl"

def log_monitor(event: str, **payload: Any) -> None:
    row = {"ts_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
