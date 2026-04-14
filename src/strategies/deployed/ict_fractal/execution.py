from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests
from zoneinfo import ZoneInfo

from config import (
    EXECUTION_MODE,
    LOG_DIR,
    MODEL_NAME,
    PICKMYTRADE_ACCOUNT_ID,
    PICKMYTRADE_BASE_URL,
    PICKMYTRADE_STRATEGY_ID,
    PICKMYTRADE_TOKEN,
)

ET = ZoneInfo('America/New_York')
EXEC_LOG = LOG_DIR / 'execution_log.jsonl'


class ExecutionError(Exception):
    pass


def now_et_iso() -> str:
    return datetime.now(timezone.utc).astimezone(ET).isoformat()


def log_event(payload: dict[str, Any]) -> None:
    with EXEC_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload) + '\n')


def execute_signal(signal: dict[str, Any]) -> dict[str, Any]:
    if EXECUTION_MODE == 'paper':
        result = {
            'ok': True,
            'mode': 'paper',
            'signal_id': signal['signal_id'],
            'symbol': signal['symbol'],
            'side': signal['side'],
            'qty': signal['qty'],
            'logged_at_et': now_et_iso(),
        }
        log_event({'event': 'paper_order', 'model_name': MODEL_NAME, **result, 'signal': signal})
        return result
    return send_to_pickmytrade(signal)


def send_to_pickmytrade(signal: dict[str, Any]) -> dict[str, Any]:
    if not PICKMYTRADE_TOKEN:
        raise ExecutionError('Missing PICKMYTRADE_TOKEN in .env')
    if not PICKMYTRADE_ACCOUNT_ID:
        raise ExecutionError('Missing PICKMYTRADE_ACCOUNT_ID in .env')

    url = f"{PICKMYTRADE_BASE_URL.rstrip('/')}/api/v1/orders"
    headers = {
        'Authorization': f'Bearer {PICKMYTRADE_TOKEN}',
        'Content-Type': 'application/json',
    }
    payload = {
        'accountId': PICKMYTRADE_ACCOUNT_ID,
        'strategyId': PICKMYTRADE_STRATEGY_ID,
        'symbol': signal['symbol'],
        'side': signal['side'].upper(),
        'quantity': int(signal['qty']),
        'orderType': 'market',
        'clientOrderId': signal['signal_id'],
        'metadata': {
            'model_name': MODEL_NAME,
            'setup_type': signal['setup_type'],
            'setup_tier': signal['setup_tier'],
            'bridge_type': signal['bridge_type'],
            'entry_ref': signal['entry'],
            'stop_ref': signal['stop'],
            'target_ref': signal['target'],
            'timestamp_et': signal['timestamp_et'],
            'source': 'exact_v473_backtest_scan',
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    text = response.text[:4000]
    if response.status_code >= 300:
        log_event({
            'event': 'live_order_error',
            'logged_at_et': now_et_iso(),
            'status_code': response.status_code,
            'response_text': text,
            'signal': signal,
        })
        raise ExecutionError(f'PickMyTrade error {response.status_code}: {text}')

    try:
        data = response.json()
    except Exception:
        data = {'raw_response': text}

    result = {
        'ok': True,
        'mode': EXECUTION_MODE,
        'signal_id': signal['signal_id'],
        'symbol': signal['symbol'],
        'side': signal['side'],
        'qty': signal['qty'],
        'logged_at_et': now_et_iso(),
        'response': data,
    }
    log_event({'event': 'live_order_sent', **result, 'signal': signal})
    return result
