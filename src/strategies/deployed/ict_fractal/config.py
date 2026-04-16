
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[4]
load_dotenv(BASE_DIR / '.env')

MODEL_NAME = 'ict_fractal'
DEPLOY_DIR = BASE_DIR / 'src' / 'strategies' / 'deployed' / MODEL_NAME
LOG_DIR = DEPLOY_DIR / 'logs'
STATE_DIR = DEPLOY_DIR / 'state'
for p in (LOG_DIR, STATE_DIR):
    p.mkdir(parents=True, exist_ok=True)

EXECUTION_MODE = os.getenv('ICT_FRACTAL_EXECUTION_MODE', 'paper').strip().lower()
LOOP_SECONDS = int(os.getenv('ICT_FRACTAL_LOOP_SECONDS', '20'))
DEFAULT_QTY = int(os.getenv('ICT_FRACTAL_DEFAULT_QTY', '5'))
SYMBOLS = [s.strip().upper() for s in os.getenv('ICT_FRACTAL_SYMBOLS', 'NQ,MES,MYM,MGC').split(',') if s.strip()]
SIGNAL_LOOKBACK_BARS = int(os.getenv('ICT_FRACTAL_SIGNAL_LOOKBACK_BARS', '1'))
BACKTEST_CASH = float(os.getenv('ICT_FRACTAL_BACKTEST_CASH', '1000000'))

PICKMYTRADE_BASE_URL = os.getenv('PICKMYTRADE_BASE_URL', 'https://api.pickmytrade.trade').strip()
PICKMYTRADE_TOKEN = os.getenv('PICKMYTRADE_TOKEN', '').strip()
PICKMYTRADE_ACCOUNT_ID = os.getenv('PICKMYTRADE_ACCOUNT_ID', '').strip()
PICKMYTRADE_STRATEGY_ID = os.getenv('PICKMYTRADE_STRATEGY_ID', MODEL_NAME).strip()
PICKMYTRADE_FORCE_FLAT_URL = os.getenv('PICKMYTRADE_FORCE_FLAT_URL', '').strip()

FORCE_FLAT_HOUR_ET = int(os.getenv('ICT_FRACTAL_FORCE_FLAT_HOUR_ET', '16'))
FORCE_FLAT_MINUTE_ET = int(os.getenv('ICT_FRACTAL_FORCE_FLAT_MINUTE_ET', '50'))
GLOBEX_REOPEN_HOUR_ET = int(os.getenv('ICT_FRACTAL_GLOBEX_REOPEN_HOUR_ET', '18'))
GLOBEX_REOPEN_MINUTE_ET = int(os.getenv('ICT_FRACTAL_GLOBEX_REOPEN_MINUTE_ET', '0'))
FORCE_FLAT_ENABLED = os.getenv('ICT_FRACTAL_FORCE_FLAT_ENABLED', '1').strip().lower() in {'1','true','yes','on'}

USE_TRADINGVIEW_BARS = os.getenv('ICT_FRACTAL_USE_TRADINGVIEW_BARS', '1').strip().lower() in {'1','true','yes','on'}
TRADINGVIEW_FALLBACK_TO_FETCHER = os.getenv('ICT_FRACTAL_TV_FALLBACK_TO_FETCHER', '1').strip().lower() in {'1','true','yes','on'}
TRADINGVIEW_MIN_BARS = int(os.getenv('ICT_FRACTAL_TV_MIN_BARS', '50'))
TRADINGVIEW_BARS_DIR = BASE_DIR / 'src' / 'data' / 'tradingview_bars'
TRADINGVIEW_BARS_DIR.mkdir(parents=True, exist_ok=True)
TV_TIMEFRAME = os.getenv('ICT_FRACTAL_TV_TIMEFRAME', '1').strip()
_raw_map = os.getenv('ICT_FRACTAL_TRADINGVIEW_SYMBOL_MAP', 'NQ:MNQ1!,MES:MES1!,MYM:MYM1!,MGC:MGC1!')
TRADINGVIEW_SYMBOL_MAP = {}
for chunk in _raw_map.split(','):
    if ':' in chunk:
        k,v = chunk.split(':',1)
        TRADINGVIEW_SYMBOL_MAP[k.strip().upper()] = v.strip()

ALLOW_UNSUPPORTED_V473_ACTIONS = os.getenv('ICT_FRACTAL_ALLOW_UNSUPPORTED_V473_ACTIONS', '0').strip().lower() in {'1','true','yes','on'}
