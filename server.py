# ============================================================
# 🌙 Algotec Dashboard Server
# ============================================================

import csv, json, time, subprocess, threading, os, signal
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "src" / "data"

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
    "chart":         DATA_DIR / "chart_analysis_log.csv",
    "websearch":     DATA_DIR / "websearch_log.json",
    "listing_arb":   DATA_DIR / "listing_arb_log.csv",
}

VAULT_INDEX  = ROOT / "src" / "strategies" / "vault" / "vault_index.json"
IDEAS_FILE   = DATA_DIR / "ideas.txt"

app = Flask(__name__)

# ── Process manager ───────────────────────────────────────────
# Tracks running background processes so we can stop them
PROCESSES    = {}   # name → subprocess.Popen
PROCESS_LOCK = threading.Lock()

def run_process(name: str, cmd: list, cwd: str = None):
    """Start a background process. Kills existing one with same name first."""
    with PROCESS_LOCK:
        # Kill existing process with same name
        if name in PROCESSES:
            try:
                PROCESSES[name].terminate()
                PROCESSES[name].wait(timeout=3)
            except Exception:
                try: PROCESSES[name].kill()
                except: pass
            del PROCESSES[name]

        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd or str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        PROCESSES[name] = proc
        return proc.pid

def stop_process(name: str) -> bool:
    """Stop a named background process."""
    with PROCESS_LOCK:
        if name not in PROCESSES:
            return False
        try:
            PROCESSES[name].terminate()
            PROCESSES[name].wait(timeout=5)
        except Exception:
            try: PROCESSES[name].kill()
            except: pass
        del PROCESSES[name]
        return True

def process_status(name: str) -> str:
    """Return 'running', 'stopped', or 'finished'."""
    with PROCESS_LOCK:
        if name not in PROCESSES:
            return "stopped"
        poll = PROCESSES[name].poll()
        if poll is None:
            return "running"
        del PROCESSES[name]
        return "finished"

PYTHON = "python3"   # python executable

try:
    from src.webhooks.tradingview_webhook import webhook_bp
    app.register_blueprint(webhook_bp)
except Exception:
    pass

def read_csv(path, max_rows=500):
    if not path.exists(): return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-max_rows:]
    except: return []

def file_age(path):
    if not path.exists(): return "never"
    s = time.time() - path.stat().st_mtime
    if s < 60: return f"{int(s)}s ago"
    if s < 3600: return f"{int(s/60)}m ago"
    return f"{int(s/3600)}h ago"

@app.route("/")
def index():
    return open(ROOT / "src" / "dashboard_live.html").read()

@app.route("/api/data/<panel>")
def api_data(panel):
    path = CSV_FILES.get(panel)
    if not path: return jsonify({"error": "unknown"}), 404
    # JSON log files
    if str(path).endswith(".json"):
        if not path.exists(): return jsonify({"rows": [], "count": 0})
        try:
            data = json.loads(path.read_text())
            return jsonify({"rows": data[-100:] if isinstance(data, list) else [data],
                            "count": len(data) if isinstance(data, list) else 1})
        except: return jsonify({"rows": [], "count": 0})
    rows = read_csv(path)
    return jsonify({"rows": rows, "count": len(rows), "last_update": file_age(path)})

@app.route("/api/vault")
def api_vault():
    if not VAULT_INDEX.exists():
        return jsonify({"strategies": []})
    try:
        return jsonify(json.loads(VAULT_INDEX.read_text()))
    except: return jsonify({"strategies": []})

@app.route("/api/system")
def api_system():
    # Ideas count
    idea_count = 0
    if IDEAS_FILE.exists():
        idea_count = sum(1 for l in IDEAS_FILE.read_text().splitlines()
                        if l.strip() and not l.startswith("#"))
    # Vault count
    vault_count = 0
    if VAULT_INDEX.exists():
        try:
            data = json.loads(VAULT_INDEX.read_text())
            vault_count = len(data.get("strategies", []))
        except: pass
    # Backtest results
    bt_rows  = read_csv(CSV_FILES["backtest"])
    returns  = [float(r.get("return_pct", 0) or 0) for r in bt_rows]
    sharpes  = [float(r.get("sharpe", 0) or 0) for r in bt_rows
                if float(r.get("sharpe", 0) or 0) > 0]
    best_return  = max(returns) if returns else 0
    avg_sharpe   = sum(sharpes)/len(sharpes) if sharpes else 0
    # Pending signals
    pending = []
    try:
        pf = DATA_DIR / "pending_signals.json"
        if pf.exists():
            pending = json.loads(pf.read_text())
    except: pass
    # Listing arb
    arb_rows = read_csv(CSV_FILES["listing_arb"])
    # Chart analyses
    chart_rows = read_csv(CSV_FILES["chart"])
    long_sigs  = sum(1 for r in chart_rows if r.get("signal") == "LONG")
    short_sigs = sum(1 for r in chart_rows if r.get("signal") == "SHORT")
    return jsonify({
        "idea_count":       idea_count,
        "vault_count":      vault_count,
        "backtest_count":   len(bt_rows),
        "best_return":      round(best_return, 1),
        "avg_sharpe":       round(avg_sharpe, 2),
        "pending_signals":  len(pending),
        "arb_candidates":   len(arb_rows),
        "chart_analyses":   len(chart_rows),
        "chart_longs":      long_sigs,
        "chart_shorts":     short_sigs,
        "last_backtest":    file_age(CSV_FILES["backtest"]),
        "last_signal":      file_age(CSV_FILES["signals"]),
        "last_chart":       file_age(CSV_FILES["chart"]),
        "last_arb":         file_age(CSV_FILES["listing_arb"]),
    })

@app.route("/api/signals/pending")
def pending_signals():
    try:
        pf = DATA_DIR / "pending_signals.json"
        if not pf.exists(): return jsonify([])
        return jsonify(json.loads(pf.read_text())[-100:])
    except: return jsonify([])

@app.route("/api/signals/approve/<sig_id>", methods=["POST"])
def approve(sig_id):
    try:
        from src.agents.signal_notifier import approve_signal, execute_signal
        sig = approve_signal(sig_id)
        if not sig: return jsonify({"error": "not found"}), 404
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

# ── Control endpoints ─────────────────────────────────────────
AGENT_COMMANDS = {
    # ── Discovery ─────────────────────────────────────────────
    "rbi_futures":       (["python3","src/agents/rbi_parallel.py","--market","futures"],
                          "RBI Backtester — Futures"),
    "rbi_crypto":        (["python3","src/agents/rbi_parallel.py","--market","crypto"],
                          "RBI Backtester — Crypto"),
    "rbi_all":           (["python3","src/agents/rbi_parallel.py","--market","all"],
                          "RBI Backtester — All Markets"),
    "websearch":         (["python3","src/agents/websearch_agent.py","--queries","5"],
                          "Websearch Agent"),
    "clear_cache":       (None, "Clear processed_ideas.json"),

    # ── Vault ─────────────────────────────────────────────────
    "vault_list":        (["python3","src/agents/vault_strategy.py","--list"],
                          "List Vault Candidates"),
    "listing_arb":       (["python3","src/agents/listing_arb_agent.py","--once"],
                          "Listing Arb Agent"),

    # ── Forward Testing ───────────────────────────────────────
    "forward_futures":   (["python3","src/agents/vault_forward_test.py",
                           "--mode","auto","--market","futures"],
                          "Forward Test — Futures (auto)"),
    "forward_crypto":    (["python3","src/agents/vault_forward_test.py",
                           "--mode","auto","--market","crypto"],
                          "Forward Test — Crypto (auto)"),
    "forward_notify":    (["python3","src/agents/vault_forward_test.py",
                           "--mode","notify","--market","all"],
                          "Forward Test — Notify Mode (all)"),

    # ── Signal Notifier (legacy ICT) ──────────────────────────
    "notify":            (["python3","src/agents/signal_notifier.py","--mode","notify"],
                          "ICT Signal Notifier (notify)"),
    "manual":            (["python3","src/agents/signal_notifier.py","--mode","manual"],
                          "ICT Signal Notifier (manual)"),

    # ── Apex / Risk ───────────────────────────────────────────
    "apex_status":       (["python3","src/agents/apex_risk.py","--status"],
                          "Apex Evaluation Status"),
    "apex_news":         (["python3","src/agents/apex_risk.py","--news"],
                          "Check News Events"),
    "pmt_test":          (["python3","src/agents/apex_bridge.py","--test-pmt"],
                          "Test PickMyTrade Connection"),
    "hl_test":           (["python3","src/agents/hyperliquid_setup.py"],
                          "Test Hyperliquid Connection"),
}


@app.route("/api/control/run/<agent>", methods=["POST"])
def control_run(agent):
    if agent not in AGENT_COMMANDS:
        return jsonify({"error": f"Unknown agent: {agent}"}), 404

    cmd, desc = AGENT_COMMANDS[agent]

    # Special case: clear cache
    if cmd is None:
        cache = ROOT / "src" / "data" / "processed_ideas.json"
        if cache.exists():
            cache.unlink()
            return jsonify({"status": "ok", "message": "Cache cleared — next RBI run will reprocess all ideas"})
        return jsonify({"status": "ok", "message": "Cache already empty"})

    try:
        pid = run_process(agent, cmd, cwd=str(ROOT))
        return jsonify({"status": "started", "agent": agent,
                        "desc": desc, "pid": pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/control/stop/<agent>", methods=["POST"])
def control_stop(agent):
    stopped = stop_process(agent)
    return jsonify({"status": "stopped" if stopped else "not_running", "agent": agent})


@app.route("/api/control/status")
def control_status():
    return jsonify({
        name: process_status(name)
        for name in AGENT_COMMANDS
    })


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
        brief = json.loads(brief_file.read_text())
        brief["available"] = True
        return jsonify(brief)
    except Exception as e:
        return jsonify({"error": str(e), "available": False})


@app.route("/api/apex")
def api_apex():
    """Return current Apex evaluation state."""
    try:
        import json
        state_file = ROOT / "src" / "data" / "apex_state.json"
        if not state_file.exists():
            return jsonify({"active": False})
        state = json.loads(state_file.read_text())
        from src.config import (APEX_DAILY_DRAWDOWN, APEX_MAX_DRAWDOWN,
                                 APEX_PROFIT_TARGET, PROP_FIRM_ACTIVE,
                                 PROP_FIRM_ACCOUNT_TYPE)
        daily_loss  = abs(min(0, state.get("daily_pnl", 0)))
        total_dd    = state.get("peak_equity", 50000) - state.get("current_equity", 50000)
        progress    = state.get("total_pnl", 0) / APEX_PROFIT_TARGET * 100
        return jsonify({
            "active":          PROP_FIRM_ACTIVE,
            "account_type":    PROP_FIRM_ACCOUNT_TYPE,
            "equity":          state.get("current_equity", 50000),
            "total_pnl":       state.get("total_pnl", 0),
            "daily_pnl":       state.get("daily_pnl", 0),
            "daily_dd":        round(daily_loss, 2),
            "daily_dd_limit":  APEX_DAILY_DRAWDOWN,
            "total_dd":        round(total_dd, 2),
            "total_dd_limit":  APEX_MAX_DRAWDOWN,
            "profit_target":   APEX_PROFIT_TARGET,
            "progress_pct":    round(progress, 1),
            "passed":          state.get("passed", False),
            "blown":           state.get("blown", False),
            "kill_switch":     state.get("kill_switch", False),
            "trades_today":    state.get("trades_today", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e), "active": False})


if __name__ == "__main__":
    print("\n🌙 Algotec Dashboard → http://algotectrading\n")
    app.run(host="0.0.0.0", port=8080, debug=False)