# ============================================================
# 🌙 Algotec Dashboard Server
# ============================================================

import csv, json, time, subprocess, threading, os, socket
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "src" / "data"
DASHBOARD_HTML = ROOT / "dashboard_live.html"

CSV_FILES = {
    "backtest":      DATA_DIR / "rbi_results" / "backtest_stats.csv",
    "trades":        DATA_DIR / "trade_log.csv",
    "signals":       DATA_DIR / "signal_log.csv",
    "chart":         DATA_DIR / "chart_analysis_log.csv",
    "listing_arb":   DATA_DIR / "listing_arb_log.csv",
    "websearch":     DATA_DIR / "websearch_log.json",
    "research":      DATA_DIR / "research_agent_log.json",
    "prop_firm":     DATA_DIR / "prop_firm_log.csv",
    "forward_test":  DATA_DIR / "forward_test_log.csv",
}

VAULT_INDEX = ROOT / "src" / "strategies" / "vault" / "vault_index.json"
IDEAS_FILE = DATA_DIR / "ideas.txt"

DEPLOYMENTS = {
    "ict_fractal": {
        "name": "ICT Fractal",
        "folder": ROOT / "src" / "strategies" / "deployed" / "ict_fractal",
        "entry": ROOT / "src" / "strategies" / "deployed" / "ict_fractal" / "app.py",
        "process_name": "deploy_ict_fractal",
    },
    "top_bottom_ticking": {
        "name": "Top Bottom Ticking",
        "folder": ROOT / "src" / "strategies" / "deployed" / "top_bottom_ticking",
        "entry": ROOT / "src" / "strategies" / "deployed" / "top_bottom_ticking" / "app.py",
        "process_name": "deploy_top_bottom_ticking",
    },
}

app = Flask(__name__)

# ── Process manager ───────────────────────────────────────────
PROCESSES = {}
PROCESS_LOCK = threading.Lock()

def run_process(name: str, cmd: list, cwd: str = None):
    with PROCESS_LOCK:
        if name in PROCESSES:
            try:
                PROCESSES[name].terminate()
                PROCESSES[name].wait(timeout=3)
            except Exception:
                try:
                    PROCESSES[name].kill()
                except Exception:
                    pass
            del PROCESSES[name]

        proc = subprocess.Popen(
            cmd,
            cwd=cwd or str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        PROCESSES[name] = proc
        return proc.pid

def stop_process(name: str) -> bool:
    with PROCESS_LOCK:
        if name not in PROCESSES:
            return False
        try:
            PROCESSES[name].terminate()
            PROCESSES[name].wait(timeout=5)
        except Exception:
            try:
                PROCESSES[name].kill()
            except Exception:
                pass
        del PROCESSES[name]
        return True

def process_status(name: str) -> str:
    with PROCESS_LOCK:
        if name not in PROCESSES:
            return "stopped"
        poll = PROCESSES[name].poll()
        if poll is None:
            return "running"
        del PROCESSES[name]
        return "finished"

def read_csv(path: Path, max_rows: int = 500):
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-max_rows:]
    except Exception:
        return []

def file_age(path: Path):
    if not path.exists():
        return "never"
    s = time.time() - path.stat().st_mtime
    if s < 60:
        return f"{int(s)}s ago"
    if s < 3600:
        return f"{int(s/60)}m ago"
    return f"{int(s/3600)}h ago"

def tcp_check(host: str, port: int, timeout: float = 2.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "reachable"
    except Exception as exc:
        return False, str(exc)

def databento_health():
    key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not key:
        return {
            "service": "databento",
            "label": "Algotec > Live Data (Databento)",
            "status": "down",
            "message": "Missing DATABENTO_API_KEY",
        }
    ok, msg = tcp_check("hist.databento.com", 443)
    return {
        "service": "databento",
        "label": "Algotec > Live Data (Databento)",
        "status": "up" if ok else "degraded",
        "message": "API key present, endpoint reachable" if ok else f"API key present, network check failed: {msg}",
    }

def pickmytrade_health():
    token = os.getenv("PICKMYTRADE_TOKEN", "").strip()
    acct = os.getenv("PICKMYTRADE_ACCOUNT_ID", "").strip()
    base = os.getenv("PICKMYTRADE_BASE_URL", "").strip()
    health = os.getenv("PICKMYTRADE_HEALTH_URL", "").strip()

    if not token:
        return {
            "service": "pickmytrade",
            "label": "Algotec > PickMyTrade",
            "status": "down",
            "message": "Missing PICKMYTRADE_TOKEN",
        }

    target = None
    if health:
        target = health
    elif base:
        target = base

    if not target:
        return {
            "service": "pickmytrade",
            "label": "Algotec > PickMyTrade",
            "status": "degraded",
            "message": "Token present. Set PICKMYTRADE_BASE_URL or PICKMYTRADE_HEALTH_URL for live reachability checks.",
        }

    host = target.replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
    ok, msg = tcp_check(host, 443)
    extra = []
    if acct:
        extra.append("account id present")
    return {
        "service": "pickmytrade",
        "label": "Algotec > PickMyTrade",
        "status": "up" if ok else "degraded",
        "message": ("Endpoint reachable" if ok else f"Endpoint check failed: {msg}") + (f" ({', '.join(extra)})" if extra else ""),
    }

try:
    from src.webhooks.tradingview_webhook import webhook_bp
    app.register_blueprint(webhook_bp)
    print("✅ TradingView webhook blueprint loaded")
except Exception as exc:
    print(f"⚠️ Failed to load TradingView webhook blueprint: {exc}")

@app.route("/")
def index():
    html_path = DASHBOARD_HTML if DASHBOARD_HTML.exists() else ROOT / "src" / "dashboard_live.html"
    return html_path.read_text(encoding="utf-8")

@app.route("/api/data/<panel>")
def api_data(panel):
    path = CSV_FILES.get(panel)
    if not path:
        return jsonify({"error": "unknown"}), 404

    if str(path).endswith(".json"):
        if not path.exists():
            return jsonify({"rows": [], "count": 0, "last_update": "never"})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data[-100:] if isinstance(data, list) else [data]
            count = len(data) if isinstance(data, list) else 1
            return jsonify({"rows": rows, "count": count, "last_update": file_age(path)})
        except Exception:
            return jsonify({"rows": [], "count": 0, "last_update": file_age(path)})
    rows = read_csv(path)
    return jsonify({"rows": rows, "count": len(rows), "last_update": file_age(path)})

@app.route("/api/vault")
def api_vault():
    if not VAULT_INDEX.exists():
        return jsonify({"strategies": []})
    try:
        return jsonify(json.loads(VAULT_INDEX.read_text(encoding="utf-8")))
    except Exception:
        return jsonify({"strategies": []})

@app.route("/api/system")
def api_system():
    idea_count = 0
    if IDEAS_FILE.exists():
        idea_count = sum(1 for l in IDEAS_FILE.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#"))

    vault_count = 0
    if VAULT_INDEX.exists():
        try:
            data = json.loads(VAULT_INDEX.read_text(encoding="utf-8"))
            vault_count = len(data.get("strategies", []))
        except Exception:
            pass

    bt_rows = read_csv(CSV_FILES["backtest"])
    returns = [float(r.get("return_pct", 0) or 0) for r in bt_rows]
    sharpes = [float(r.get("sharpe", 0) or 0) for r in bt_rows if float(r.get("sharpe", 0) or 0) > 0]
    best_return = max(returns) if returns else 0
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0

    pending = []
    pf = DATA_DIR / "pending_signals.json"
    if pf.exists():
        try:
            pending = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            pass

    arb_rows = read_csv(CSV_FILES["listing_arb"])
    chart_rows = read_csv(CSV_FILES["chart"])
    long_sigs = sum(1 for r in chart_rows if r.get("signal") == "LONG")
    short_sigs = sum(1 for r in chart_rows if r.get("signal") == "SHORT")

    return jsonify({
        "idea_count": idea_count,
        "vault_count": vault_count,
        "backtest_count": len(bt_rows),
        "best_return": round(best_return, 1),
        "avg_sharpe": round(avg_sharpe, 2),
        "pending_signals": len(pending),
        "arb_candidates": len(arb_rows),
        "chart_analyses": len(chart_rows),
        "chart_longs": long_sigs,
        "chart_shorts": short_sigs,
        "last_backtest": file_age(CSV_FILES["backtest"]),
        "last_signal": file_age(CSV_FILES["signals"]),
        "last_chart": file_age(CSV_FILES["chart"]),
        "last_arb": file_age(CSV_FILES["listing_arb"]),
    })

@app.route("/api/signals/pending")
def pending_signals():
    pf = DATA_DIR / "pending_signals.json"
    if not pf.exists():
        return jsonify([])
    try:
        return jsonify(json.loads(pf.read_text(encoding="utf-8"))[-100:])
    except Exception:
        return jsonify([])

@app.route("/api/signals/approve/<sig_id>", methods=["POST"])
def approve(sig_id):
    try:
        from src.agents.signal_notifier import approve_signal, execute_signal
        sig = approve_signal(sig_id)
        if not sig:
            return jsonify({"error": "not found"}), 404
        return jsonify({"status": "approved", "result": execute_signal(sig)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/signals/reject/<sig_id>", methods=["POST"])
def reject(sig_id):
    try:
        from src.agents.signal_notifier import reject_signal
        return jsonify({"status": "rejected" if reject_signal(sig_id) else "not_found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

AGENT_COMMANDS = {
    "rbi_futures":       (["python3","src/agents/rbi_parallel.py","--market","futures"], "RBI Backtester — Futures"),
    "rbi_crypto":        (["python3","src/agents/rbi_parallel.py","--market","crypto"], "RBI Backtester — Crypto"),
    "rbi_all":           (["python3","src/agents/rbi_parallel.py","--market","all"], "RBI Backtester — All Markets"),
    "websearch":         (["python3","src/agents/websearch_agent.py","--queries","5"], "Websearch Agent"),
    "clear_cache":       (None, "Clear processed_ideas.json"),
    "vault_list":        (["python3","src/agents/vault_strategy.py","--list"], "List Vault Candidates"),
    "listing_arb":       (["python3","src/agents/listing_arb_agent.py","--once"], "Listing Arb Agent"),
    "forward_futures":   (["python3","src/agents/vault_forward_test.py","--mode","auto","--market","futures"], "Forward Test — Futures (auto)"),
    "forward_crypto":    (["python3","src/agents/vault_forward_test.py","--mode","auto","--market","crypto"], "Forward Test — Crypto (auto)"),
    "forward_notify":    (["python3","src/agents/vault_forward_test.py","--mode","notify","--market","all"], "Forward Test — Notify Mode (all)"),
    "notify":            (["python3","src/agents/signal_notifier.py","--mode","notify"], "ICT Signal Notifier (notify)"),
    "manual":            (["python3","src/agents/signal_notifier.py","--mode","manual"], "ICT Signal Notifier (manual)"),
    "apex_status":       (["python3","src/agents/apex_risk.py","--status"], "Apex Evaluation Status"),
    "apex_news":         (["python3","src/agents/apex_risk.py","--news"], "Check News Events"),
    "pmt_test":          (["python3","src/agents/apex_bridge.py","--test-pmt"], "Test PickMyTrade Connection"),
    "hl_test":           (["python3","src/agents/hyperliquid_setup.py"], "Test Hyperliquid Connection"),
    "handoff_now":       (["python3","src/agents/handoff_generator.py"], "Generate Handoff Doc Now"),
    "handoff_watch":     (["python3","src/agents/handoff_generator.py","--watch"], "Auto-Watch + Regenerate Handoff"),
    "handoff_diff":      (["python3","src/agents/handoff_generator.py","--diff"], "Show What Changed in Handoff"),
    "deploy_ict_fractal": (
        ["python3", "src/strategies/deployed/ict_fractal/app.py"],
        "Deploy — ICT Fractal",
    ),
    "deploy_top_bottom_ticking": (
        ["python3", "src/strategies/deployed/top_bottom_ticking/app.py"],
        "Deploy — Top Bottom Ticking",
    ),
}

@app.route("/api/control/run/<agent>", methods=["POST"])
def control_run(agent):
    if agent not in AGENT_COMMANDS:
        return jsonify({"error": f"Unknown agent: {agent}"}), 404

    cmd, desc = AGENT_COMMANDS[agent]
    if cmd is None:
        cache = ROOT / "src" / "data" / "processed_ideas.json"
        if cache.exists():
            cache.unlink()
            return jsonify({"status": "ok", "message": "Cache cleared — next RBI run will reprocess all ideas"})
        return jsonify({"status": "ok", "message": "Cache already empty"})

    try:
        pid = run_process(agent, cmd, cwd=str(ROOT))
        return jsonify({"status": "started", "agent": agent, "desc": desc, "pid": pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/control/stop/<agent>", methods=["POST"])
def control_stop(agent):
    stopped = stop_process(agent)
    return jsonify({"status": "stopped" if stopped else "not_running", "agent": agent})

@app.route("/api/control/status")
def control_status():
    return jsonify({name: process_status(name) for name in AGENT_COMMANDS})

@app.route("/api/control/stop_all", methods=["POST"])
def control_stop_all():
    stopped = []
    for name in list(PROCESSES.keys()):
        if stop_process(name):
            stopped.append(name)
    return jsonify({"stopped": stopped})

@app.route("/api/weekly_brief")
def api_weekly_brief():
    try:
        brief_file = ROOT / "src" / "data" / "weekly_brief.json"
        if not brief_file.exists():
            return jsonify({"available": False})
        brief = json.loads(brief_file.read_text(encoding="utf-8"))
        brief["available"] = True
        return jsonify(brief)
    except Exception as e:
        return jsonify({"error": str(e), "available": False})

@app.route("/api/apex")
def api_apex():
    try:
        state_file = ROOT / "src" / "data" / "apex_state.json"
        if not state_file.exists():
            return jsonify({"active": False})
        state = json.loads(state_file.read_text(encoding="utf-8"))
        from src.config import (APEX_DAILY_DRAWDOWN, APEX_MAX_DRAWDOWN, APEX_PROFIT_TARGET, PROP_FIRM_ACTIVE, PROP_FIRM_ACCOUNT_TYPE)
        daily_loss = abs(min(0, state.get("daily_pnl", 0)))
        total_dd = state.get("peak_equity", 50000) - state.get("current_equity", 50000)
        progress = state.get("total_pnl", 0) / APEX_PROFIT_TARGET * 100
        return jsonify({
            "active": PROP_FIRM_ACTIVE,
            "account_type": PROP_FIRM_ACCOUNT_TYPE,
            "equity": state.get("current_equity", 50000),
            "total_pnl": state.get("total_pnl", 0),
            "daily_pnl": state.get("daily_pnl", 0),
            "daily_dd": round(daily_loss, 2),
            "daily_dd_limit": APEX_DAILY_DRAWDOWN,
            "total_dd": round(total_dd, 2),
            "total_dd_limit": APEX_MAX_DRAWDOWN,
            "profit_target": APEX_PROFIT_TARGET,
            "progress_pct": round(progress, 1),
            "passed": state.get("passed", False),
            "blown": state.get("blown", False),
            "kill_switch": state.get("kill_switch", False),
            "trades_today": state.get("trades_today", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e), "active": False})

@app.route("/api/health/databento")
def api_health_databento():
    return jsonify(databento_health())

@app.route("/api/health/pickmytrade")
def api_health_pickmytrade():
    return jsonify(pickmytrade_health())

@app.route("/api/health/connections")
def api_health_connections():
    checks = [databento_health(), pickmytrade_health()]
    up = sum(1 for c in checks if c["status"] == "up")
    degraded = sum(1 for c in checks if c["status"] == "degraded")
    down = sum(1 for c in checks if c["status"] == "down")
    return jsonify({
        "checks": checks,
        "summary": {
            "up": up,
            "degraded": degraded,
            "down": down,
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }
    })

@app.route("/api/deployments")
def api_deployments():
    rows = []
    for key, dep in DEPLOYMENTS.items():
        rows.append({
            "key": key,
            "name": dep["name"],
            "folder": str(dep["folder"].relative_to(ROOT)) if dep["folder"].exists() else str(dep["folder"].relative_to(ROOT)),
            "entry": str(dep["entry"].relative_to(ROOT)),
            "folder_exists": dep["folder"].exists(),
            "entry_exists": dep["entry"].exists(),
            "status": process_status(dep["process_name"]),
            "last_modified": file_age(dep["entry"]) if dep["entry"].exists() else "missing",
        })
    return jsonify({"rows": rows, "count": len(rows)})

@app.route("/api/deployments/run/<key>", methods=["POST"])
def api_deploy_run(key):
    dep = DEPLOYMENTS.get(key)
    if not dep:
        return jsonify({"error": f"Unknown deployment: {key}"}), 404
    if not dep["entry"].exists():
        return jsonify({"error": f"Missing entry file: {dep['entry']}"}), 400
    try:
        pid = run_process(dep["process_name"], ["python3", str(dep["entry"].relative_to(ROOT))], cwd=str(ROOT))
        return jsonify({"status": "started", "deployment": key, "pid": pid})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/api/deployments/stop/<key>", methods=["POST"])
def api_deploy_stop(key):
    dep = DEPLOYMENTS.get(key)
    if not dep:
        return jsonify({"error": f"Unknown deployment: {key}"}), 404
    stopped = stop_process(dep["process_name"])
    return jsonify({"status": "stopped" if stopped else "not_running", "deployment": key})

if __name__ == "__main__":
    print("\n🌙 Algotec Dashboard → http://algotectrading\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
