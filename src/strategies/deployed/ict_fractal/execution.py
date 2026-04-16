
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
import requests
from zoneinfo import ZoneInfo

from config import (
    ALLOW_UNSUPPORTED_V473_ACTIONS,
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
        f.write(json.dumps(payload, default=str) + '\n')


def execute_signal(signal: dict[str, Any]) -> dict[str, Any]:
    action_type = str(signal.get('action_type', 'entry'))
    if EXECUTION_MODE == 'paper':
        result = {'ok': True, 'mode': 'paper', 'action_type': action_type, 'signal_id': signal['signal_id'], 'symbol': signal['symbol'], 'side': signal.get('side',''), 'qty': signal.get('qty',0), 'logged_at_et': now_et_iso()}
        log_event({'event': 'paper_action', 'model_name': MODEL_NAME, **result, 'signal': signal})
        return result
    if action_type == 'entry':
        return send_entry_to_pickmytrade(signal)
    if action_type == 'trade_closed':
        return send_exit_to_pickmytrade(signal, reason='v473_trade_closed')
    if action_type in {'partial_taken', 'stop_moved_to_be'}:
        result = {'ok': False, 'mode': EXECUTION_MODE, 'action_type': action_type, 'signal_id': signal['signal_id'], 'symbol': signal['symbol'], 'logged_at_et': now_et_iso(), 'reason': 'unsupported_by_current_pickmytrade_transport'}
        log_event({'event': 'unsupported_v473_action', **result, 'signal': signal})
        if not ALLOW_UNSUPPORTED_V473_ACTIONS:
            return result
    raise ExecutionError(f'Unsupported action_type: {action_type}')


def _build_multiple_accounts() -> list[dict[str, Any]]:
    if not PICKMYTRADE_TOKEN:
        raise ExecutionError('Missing PICKMYTRADE_TOKEN in .env')
    if not PICKMYTRADE_ACCOUNT_ID:
        raise ExecutionError('Missing PICKMYTRADE_ACCOUNT_ID in .env')
    return [{'token': PICKMYTRADE_TOKEN, 'account_id': PICKMYTRADE_ACCOUNT_ID, 'risk_percentage': 0, 'quantity_multiplier': 1}]


def _base_payload(signal: dict[str, Any], data_value: str) -> dict[str, Any]:
    return {
        'symbol': str(signal['symbol']).strip().upper(),
        'strategy_name': PICKMYTRADE_STRATEGY_ID or MODEL_NAME,
        'date': now_et_iso(),
        'data': data_value,
        'quantity': int(signal.get('qty', 0) or 0),
        'risk_percentage': 0,
        'price': round(float(signal.get('entry', 0) or 0), 4) if signal.get('entry') else 0,
        'tp': round(float(signal.get('target', 0) or 0), 4) if signal.get('target') else 0,
        'sl': round(float(signal.get('stop', 0) or 0), 4) if signal.get('stop') else 0,
        'percentage_tp': 0, 'dollar_tp': 0, 'dollar_sl': 0, 'percentage_sl': 0,
        'trail': 0, 'trail_stop': 0, 'trail_trigger': 0, 'trail_freq': 0,
        'update_tp': False, 'update_sl': False, 'breakeven': 0, 'breakeven_offset': 0,
        'token': PICKMYTRADE_TOKEN, 'pyramid': False, 'same_direction_ignore': False, 'reverse_order_close': False,
        'multiple_accounts': _build_multiple_accounts(),
    }


def send_entry_to_pickmytrade(signal: dict[str, Any]) -> dict[str, Any]:
    url = f"{PICKMYTRADE_BASE_URL.rstrip('/')}/v2/add-trade-data-latest"
    payload = _base_payload(signal, str(signal['side']).lower())
    payload['metadata'] = {
        'model_name': MODEL_NAME,
        'signal_id': signal['signal_id'],
        'setup_type': signal.get('setup_type', ''),
        'setup_tier': signal.get('setup_tier', ''),
        'bridge_type': signal.get('bridge_type', ''),
        'entry_ref': signal.get('entry'),
        'stop_ref': signal.get('stop'),
        'target_ref': signal.get('target'),
        'partial_target_ref': signal.get('partial_target'),
        'runner_target_ref': signal.get('runner_target'),
        'timestamp_et': signal.get('timestamp_et', now_et_iso()),
        'source': 'exact_v473_live_wrapper',
    }
    response = requests.post(url, json=payload, timeout=30)
    text = response.text[:4000]
    if response.status_code >= 300:
        log_event({'event': 'live_order_error', 'logged_at_et': now_et_iso(), 'status_code': response.status_code, 'response_text': text, 'signal': signal, 'url': url})
        raise ExecutionError(f'PickMyTrade error {response.status_code}: {text}')
    try:
        data = response.json()
    except Exception:
        data = {'raw_response': text}
    result = {'ok': True, 'mode': EXECUTION_MODE, 'action_type': 'entry', 'signal_id': signal['signal_id'], 'symbol': signal['symbol'], 'side': signal['side'], 'qty': signal['qty'], 'logged_at_et': now_et_iso(), 'response': data}
    log_event({'event': 'live_order_sent', **result, 'signal': signal, 'url': url})
    return result


def send_exit_to_pickmytrade(signal: dict[str, Any], reason: str) -> dict[str, Any]:
    url = f"{PICKMYTRADE_BASE_URL.rstrip('/')}/v2/add-trade-data-latest"
    payload = _base_payload(signal, 'exit')
    payload['metadata'] = {
        'model_name': MODEL_NAME,
        'signal_id': signal['signal_id'],
        'reason': reason,
        'timestamp_et': signal.get('timestamp_et', now_et_iso()),
        'source': 'exact_v473_live_wrapper',
    }
    response = requests.post(url, json=payload, timeout=30)
    text = response.text[:4000]
    if response.status_code >= 300:
        log_event({'event': 'live_exit_error', 'logged_at_et': now_et_iso(), 'status_code': response.status_code, 'response_text': text, 'signal': signal, 'url': url})
        raise ExecutionError(f'PickMyTrade error {response.status_code}: {text}')
    try:
        data = response.json()
    except Exception:
        data = {'raw_response': text}
    result = {'ok': True, 'mode': EXECUTION_MODE, 'action_type': 'trade_closed', 'signal_id': signal['signal_id'], 'symbol': signal['symbol'], 'logged_at_et': now_et_iso(), 'response': data}
    log_event({'event': 'live_exit_sent', **result, 'signal': signal, 'url': url})
    return result


def force_flat_all_positions(*, reason: str, session_key: str) -> dict[str, Any]:
    if EXECUTION_MODE == 'paper':
        result = {'ok': True, 'mode': 'paper', 'reason': reason, 'session_key': session_key, 'logged_at_et': now_et_iso()}
        log_event({'event': 'paper_force_flat', **result})
        return result
    if PICKMYTRADE_FORCE_FLAT_URL:
        url = PICKMYTRADE_FORCE_FLAT_URL if PICKMYTRADE_FORCE_FLAT_URL.startswith('http') else f"{PICKMYTRADE_BASE_URL.rstrip('/')}/{PICKMYTRADE_FORCE_FLAT_URL.lstrip('/')}"
        payload = {'token': PICKMYTRADE_TOKEN, 'account_id': PICKMYTRADE_ACCOUNT_ID, 'strategy_name': PICKMYTRADE_STRATEGY_ID or MODEL_NAME, 'reason': reason, 'session_key': session_key}
        response = requests.post(url, json=payload, timeout=30)
        text = response.text[:4000]
        if response.status_code >= 300:
            raise ExecutionError(f'PickMyTrade force-flat error {response.status_code}: {text}')
        try:
            data = response.json()
        except Exception:
            data = {'raw_response': text}
        result = {'ok': True, 'mode': EXECUTION_MODE, 'reason': reason, 'session_key': session_key, 'response': data, 'logged_at_et': now_et_iso()}
        log_event({'event': 'live_force_flat', **result})
        return result
    return send_exit_to_pickmytrade({'signal_id': f'force_flat_{session_key}', 'symbol': 'NQ', 'qty': 0, 'timestamp_et': now_et_iso()}, reason=reason)
