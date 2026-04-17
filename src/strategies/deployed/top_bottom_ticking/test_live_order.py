from execution import execute_signal
from config import EXECUTION_MODE

print("EXECUTION_MODE =", EXECUTION_MODE)

signal = {
    "signal_id": "connectivity-test-mes-buy-1",
    "model_name": "top_bottom_ticking",
    "symbol": "MES",
    "side": "BUY",
    "qty": 1,
    "entry": 0.0,
    "stop": 0.0,
    "target": 0.0,
    "timestamp_et": "manual_connectivity_test",
    "session_date_et": "manual_connectivity_test",
    "setup_type": "MANUAL_CONNECTIVITY_TEST",
    "setup_tier": "TEST",
    "bridge_type": "MANUAL",
}

result = execute_signal(signal)
print(result)