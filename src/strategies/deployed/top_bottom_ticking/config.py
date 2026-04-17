from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[4]
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "top_bottom_ticking"
DEPLOY_DIR = BASE_DIR / "src" / "strategies" / "deployed" / MODEL_NAME
LOG_DIR = DEPLOY_DIR / "logs"
STATE_DIR = DEPLOY_DIR / "state"

for p in (LOG_DIR, STATE_DIR):
    p.mkdir(parents=True, exist_ok=True)

EXECUTION_MODE = os.getenv("TOP_BOTTOM_TICKING_EXECUTION_MODE", os.getenv("ICT_FRACTAL_EXECUTION_MODE", "paper")).strip().lower()
LOOP_SECONDS = int(os.getenv("TOP_BOTTOM_TICKING_LOOP_SECONDS", os.getenv("ICT_FRACTAL_LOOP_SECONDS", "20")))
DEFAULT_QTY = int(os.getenv("TOP_BOTTOM_TICKING_DEFAULT_QTY", os.getenv("ICT_FRACTAL_DEFAULT_QTY", "1")))
SYMBOLS = [s.strip().upper() for s in os.getenv("TOP_BOTTOM_TICKING_SYMBOLS", "MES").split(",") if s.strip()]
SIGNAL_LOOKBACK_BARS = int(os.getenv("TOP_BOTTOM_TICKING_SIGNAL_LOOKBACK_BARS", os.getenv("ICT_FRACTAL_SIGNAL_LOOKBACK_BARS", "1")))
BACKTEST_CASH = float(os.getenv("TOP_BOTTOM_TICKING_BACKTEST_CASH", os.getenv("ICT_FRACTAL_BACKTEST_CASH", "1000000")))

PICKMYTRADE_BASE_URL = os.getenv("PICKMYTRADE_BASE_URL", "https://api.pickmytrade.trade").strip()
PICKMYTRADE_TOKEN = os.getenv("PICKMYTRADE_TOKEN", "").strip()
PICKMYTRADE_ACCOUNT_ID = os.getenv("PICKMYTRADE_ACCOUNT_ID", "").strip()
PICKMYTRADE_STRATEGY_ID = os.getenv("TOP_BOTTOM_TICKING_PICKMYTRADE_STRATEGY_ID", MODEL_NAME).strip()
PICKMYTRADE_FORCE_FLAT_URL = os.getenv("PICKMYTRADE_FORCE_FLAT_URL", "").strip()

# NEW: exact webhook URL copied from PickMyTrade Generate Alert
PICKMYTRADE_WEBHOOK_URL = os.getenv("PICKMYTRADE_WEBHOOK_URL", "").strip()

FORCE_FLAT_HOUR_ET = int(os.getenv("TOP_BOTTOM_TICKING_FORCE_FLAT_HOUR_ET", os.getenv("ICT_FRACTAL_FORCE_FLAT_HOUR_ET", "16")))
FORCE_FLAT_MINUTE_ET = int(os.getenv("TOP_BOTTOM_TICKING_FORCE_FLAT_MINUTE_ET", os.getenv("ICT_FRACTAL_FORCE_FLAT_MINUTE_ET", "50")))
GLOBEX_REOPEN_HOUR_ET = int(os.getenv("TOP_BOTTOM_TICKING_GLOBEX_REOPEN_HOUR_ET", os.getenv("ICT_FRACTAL_GLOBEX_REOPEN_HOUR_ET", "18")))
GLOBEX_REOPEN_MINUTE_ET = int(os.getenv("TOP_BOTTOM_TICKING_GLOBEX_REOPEN_MINUTE_ET", os.getenv("ICT_FRACTAL_GLOBEX_REOPEN_MINUTE_ET", "0")))
FORCE_FLAT_ENABLED = os.getenv("TOP_BOTTOM_TICKING_FORCE_FLAT_ENABLED", os.getenv("ICT_FRACTAL_FORCE_FLAT_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}

ALLOW_UNSUPPORTED_V473_ACTIONS = os.getenv("TOP_BOTTOM_TICKING_ALLOW_UNSUPPORTED_ACTIONS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}