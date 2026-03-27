# ============================================================
# 🌙 Main Orchestrator
# Runs all agents enabled in src/config.py
#
# Usage:
#   python src/main.py                         # all active agents
#   python src/agents/rbi_agent.py             # RBI only
#   python src/agents/trading_agent.py --once  # single live scan
#   python src/agents/whale_agent.py --once    # single whale scan
#   python src/agents/liquidation_agent.py     # liquidation tracker
#   python src/agents/sentiment_agent.py       # sentiment scanner
# ============================================================

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ACTIVE_AGENTS

print("""
🌙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   AI Trading System — Main Orchestrator
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def run_agent(name: str):
    if name == "rbi_backtester":
        from src.agents.rbi_agent import RBIAgent
        RBIAgent().run()
    elif name == "trading_agent":
        from src.agents.trading_agent import TradingAgent
        TradingAgent().run()
    elif name == "whale_monitor":
        from src.agents.whale_agent import WhaleMonitor
        WhaleMonitor().run()
    elif name == "liquidation":
        from src.agents.liquidation_agent import LiquidationAgent
        LiquidationAgent().run()
    elif name == "sentiment":
        from src.agents.sentiment_agent import SentimentAgent
        SentimentAgent().run()
    elif name == "risk_agent":
        print("ℹ️  Risk agent is embedded — no standalone process needed.")
    else:
        print(f"⚠️  Unknown agent: {name}")


active = [name for name, enabled in ACTIVE_AGENTS.items() if enabled]

if not active:
    print("⚠️  No agents enabled. Edit ACTIVE_AGENTS in src/config.py")
    sys.exit(0)

print(f"▶️  Starting agents: {active}\n")

if len(active) == 1:
    run_agent(active[0])
else:
    threads = []
    for name in active:
        t = threading.Thread(target=run_agent, args=(name,), daemon=True, name=name)
        t.start()
        threads.append(t)
        print(f"  🧵 Launched: {name}")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Orchestrator stopped.")
