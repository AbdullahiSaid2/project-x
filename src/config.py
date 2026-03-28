# ============================================================
# 🌙 Trading System Config
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()

# ── AI Model ────────────────────────────────────────────────
AI_MODEL    = "deepseek-chat"
AI_PROVIDER = "deepseek"

# ── Exchanges ───────────────────────────────────────────────
EXCHANGE = "hyperliquid"   # "hyperliquid" | "coinbase"

# ── Risk Management ─────────────────────────────────────────
MAX_POSITION_SIZE_USD  = 100     # max per trade
MAX_PORTFOLIO_RISK_PCT = 0.02    # 2% per trade
MAX_DAILY_LOSS_USD     = 500
STOP_LOSS_PCT          = 0.03    # 3% stop loss
TAKE_PROFIT_PCT        = 0.06    # 6% take profit

# ── Tokens to trade ─────────────────────────────────────────
HYPERLIQUID_TOKENS = ["BTC", "ETH", "SOL", "ARB", "AVAX"]
COINBASE_TOKENS    = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ── Futures (Tradovate) ──────────────────────────────────────
TRADOVATE_TOKENS   = ["MNQ", "MES", "MYM"]   # Micro contracts
# Start with 1 contract each. Margin: MNQ~$40, MES~$50, MYM~$50
DEFAULT_CONTRACTS  = 1

# TradingView backtest data directory
TV_DATA_DIR        = "src/data/tradingview"

# Webhook security (change this!)
WEBHOOK_SECRET    = os.getenv("WEBHOOK_SECRET", "algotec_secret_change_me")

# ── Data / Backtesting ──────────────────────────────────────
BACKTEST_INITIAL_CASH = 10_000
BACKTEST_COMMISSION   = 0.001    # 0.1%
DATA_TIMEFRAMES       = ["15m", "1H", "4H", "1D"]
DEFAULT_TIMEFRAME     = "1H"
DAYS_BACK             = 365

# ── Agent Loop ──────────────────────────────────────────────
SLEEP_BETWEEN_RUNS_SEC = 60

# ── Active Agents ───────────────────────────────────────────
# Recommended order to enable:
#  1. rbi_backtester  → find a working strategy first
#  2. sentiment       → understand market mood
#  3. whale_monitor   → track big money movements
#  4. liquidation     → spot reversal signals
#  5. trading_agent   → go live (only after steps 1-4!)
ACTIVE_AGENTS = {
    "rbi_backtester": True,    # standard RBI (3 datasets, sequential)
    "rbi_parallel":   False,   # ← flip to True for parallel RBI (25 datasets, 8x faster)
    "trading_agent":  False,   # flip to True for live trading
    "risk_agent":     False,   # embedded in trading_agent automatically
    "whale_monitor":  False,   # flip to True to track whales
    "liquidation":    False,   # flip to True to track liquidations
    "sentiment":      False,   # flip to True for news sentiment
}

# ── Execution Mode ─────────────────────────────────────────────
# Controls how ICT signals are handled
EXECUTION_MODE = "manual"   # "auto"   = execute immediately
                             # "manual" = wait for dashboard approval
                             # "notify" = alert only, never execute

# ── Prop Firm Settings ─────────────────────────────────────────
# Set these when trading with a prop firm account
PROP_FIRM_ACTIVE   = False          # set True when using prop account
PROP_FIRM_NAME     = "apex"         # apex | topstep | ftmo | fundednext
PROP_FIRM_ACCOUNT  = "50k"          # account size tier
# When PROP_FIRM_ACTIVE is True, risk limits are automatically
# loaded from the prop firm profile in prop_firm_monitor.py
