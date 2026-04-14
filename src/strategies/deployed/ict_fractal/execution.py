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
    PICKMYTRADE_FORCE_FLAT_URL,
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


def _auth_headers() -> dict[str, str]:
    if not PICKMYTRADE_TOKEN:
        raise ExecutionError('Missing PICKMYTRADE_TOKEN in .env')
    return {
        'Authorization': f'Bearer {PICKMYTRADE_TOKEN}',
        'Content-Type': 'application/json',
    }


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
    if not PICKMYTRADE_ACCOUNT_ID:
        raise ExecutionError('Missing PICKMYTRADE_ACCOUNT_ID in .env')

    url = f"{PICKMYTRADE_BASE_URL.rstrip('/')}/v2/add-trade-data-latest"
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

    response = requests.post(url, headers=_auth_headers(), json=payload, timeout=30)
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


def force_flat_all_positions(*, reason: str, session_key: str) -> dict[str, Any]:
    base_result = {
        'mode': EXECUTION_MODE,
        'reason': reason,
        'session_key': session_key,
        'logged_at_et': now_et_iso(),
    }

    if EXECUTION_MODE == 'paper':
        result = {
            'ok': True,
            **base_result,
            'action': 'paper_force_flat_logged',
        }
        log_event({'event': 'paper_force_flat', 'model_name': MODEL_NAME, **result})
        return result

    if not PICKMYTRADE_ACCOUNT_ID:
        raise ExecutionError('Missing PICKMYTRADE_ACCOUNT_ID in .env')

    if not PICKMYTRADE_FORCE_FLAT_URL:
        result = {
            'ok': False,
            **base_result,
            'action': 'force_flat_endpoint_not_configured',
        }
        log_event({'event': 'force_flat_skipped', 'model_name': MODEL_NAME, **result})
        return result

    url = PICKMYTRADE_FORCE_FLAT_URL
    if not url.lower().startswith('http://') and not url.lower().startswith('https://'):
        url = f"{PICKMYTRADE_BASE_URL.rstrip('/')}/{url.lstrip('/')}"

    payload = {
        'accountId': PICKMYTRADE_ACCOUNT_ID,
        'strategyId': PICKMYTRADE_STRATEGY_ID,
        'reason': reason,
        'metadata': {
            'model_name': MODEL_NAME,
            'session_key': session_key,
            'source': 'ict_fractal_force_flat_window',
        },
    }

    response = requests.post(url, headers=_auth_headers(), json=payload, timeout=30)
    text = response.text[:4000]
    if response.status_code >= 300:
        log_event({
            'event': 'force_flat_error',
            'model_name': MODEL_NAME,
            **base_result,
            'status_code': response.status_code,
            'response_text': text,
            'url': url,
        })
        raise ExecutionError(f'PickMyTrade force-flat error {response.status_code}: {text}')

    try:
        data = response.json()
    except Exception:
        data = {'raw_response': text}

    result = {
        'ok': True,
        **base_result,
        'action': 'force_flat_sent',
        'response': data,
        'url': url,
    }
    log_event({'event': 'force_flat_sent', 'model_name': MODEL_NAME, **result})
    return result
