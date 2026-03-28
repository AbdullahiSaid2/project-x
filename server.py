# ============================================================
# 🌙 Live Dashboard Server
#
# Serves the live auto-refreshing dashboard at http://algotectrading
# Reads CSV log files in real-time as agents write to them.
#
# HOW TO RUN:
#   pip install flask
#   python server.py
#   Then open: http://algotectrading
# ============================================================

import csv
import json
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request

# ── paths ─────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "src" / "data"

CSV_FILES = {
    "backtest":    DATA_DIR / "rbi_results" / "backtest_stats.csv",
    "trades":      DATA_DIR / "trade_log.csv",
    "whale":       DATA_DIR / "whale_log.csv",
    "liquidation": DATA_DIR / "liquidation_log.csv",
    "sentiment":   DATA_DIR / "sentiment_log.csv",
    "risk":        DATA_DIR / "risk_log.csv",
    "swarm":       DATA_DIR / "swarm_log.csv",
    "ict":         DATA_DIR / "ict_scanner_log.csv",
    "ict_exec":    DATA_DIR / "ict_exec_log.csv",
    "ict_bt":      DATA_DIR / "ict_backtest" / "ict_backtest_summary.csv",
    "forward_test": DATA_DIR / "forward_test_log.csv",
    "funding_arb":  DATA_DIR / "funding_arb_log.csv",
    "copy_bot":     DATA_DIR / "copy_bot_log.csv",
    "regime":       DATA_DIR / "regime_log.csv",
    "monte_carlo":  DATA_DIR / "monte_carlo_results.csv",
    "vwap":         DATA_DIR / "vwap_log.csv",
    "psychology":   DATA_DIR / "psychology_log.csv",
}

app = Flask(__name__)


# ── CSV reader ────────────────────────────────────────────────
def read_csv(path: Path, max_rows: int = 500) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        return rows[-max_rows:]   # return most recent rows
    except Exception as e:
        return []


def get_file_age(path: Path) -> str:
    """Return human-readable age of a file."""
    if not path.exists():
        return "never"
    age_sec = time.time() - path.stat().st_mtime
    if age_sec < 60:
        return f"{int(age_sec)}s ago"
    if age_sec < 3600:
        return f"{int(age_sec/60)}m ago"
    return f"{int(age_sec/3600)}h ago"


# ── API endpoints ─────────────────────────────────────────────
@app.route("/api/data/<panel>")
def api_data(panel):
    path = CSV_FILES.get(panel)
    if not path:
        return jsonify({"error": "Unknown panel"}), 404
    rows = read_csv(path)
    return jsonify({
        "rows":      rows,
        "count":     len(rows),
        "last_update": get_file_age(path),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/status")
def api_status():
    status = {}
    for name, path in CSV_FILES.items():
        rows = read_csv(path, max_rows=1)
        status[name] = {
            "exists":      path.exists(),
            "rows":        read_csv(path).__len__() if path.exists() else 0,
            "last_update": get_file_age(path),
        }
    return jsonify(status)


@app.route("/api/summary")
def api_summary():
    """Key metrics for the header stat bar."""
    bt_rows  = read_csv(CSV_FILES["backtest"])
    tr_rows  = read_csv(CSV_FILES["trades"])
    wh_rows  = read_csv(CSV_FILES["whale"])
    lq_rows  = read_csv(CSV_FILES["liquidation"])
    se_rows  = read_csv(CSV_FILES["sentiment"])

    # Best backtest return
    returns = [float(r.get("return_pct", 0) or 0) for r in bt_rows]
    best_return = max(returns) if returns else 0

    # Latest sentiment
    latest_sentiment = se_rows[-1].get("sentiment", "—") if se_rows else "—"
    latest_score     = float(se_rows[-1].get("score", 0) or 0) if se_rows else 0

    # High whale alerts
    high_whale = sum(1 for r in wh_rows if r.get("alert_level") == "HIGH")

    # Recent trades
    recent_trades = len(tr_rows)

    return jsonify({
        "strategies_tested": len(bt_rows),
        "best_return":       round(best_return, 1),
        "total_trades":      recent_trades,
        "whale_alerts_high": high_whale,
        "liq_alerts":        len(lq_rows),
        "sentiment":         latest_sentiment,
        "sentiment_score":   round(latest_score, 2),
    })


# ── Main HTML page ────────────────────────────────────────────
HTML = open(ROOT / "src" / "dashboard_live.html").read()

@app.route("/")
def index():
    return HTML


if __name__ == "__main__":
    print("""
🌙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Algotec Trading Dashboard
   Open: http://algotectrading
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    app.run(host="0.0.0.0", port=8080, debug=False)
