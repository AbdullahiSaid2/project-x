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
# Run both simultaneously — futures on Tradovate, crypto on Hyperliquid
EXCHANGE          = "hyperliquid"   # default for backward compat

TRADOVATE_ENABLED    = True
TRADOVATE_SIM        = True         # ← True = sim account, False = live
HYPERLIQUID_ENABLED  = True

TRADOVATE_API_URL_SIM  = "https://demo.tradovateapi.com/v1"
TRADOVATE_API_URL_LIVE = "https://live.tradovateapi.com/v1"

def get_tradovate_url():
    return TRADOVATE_API_URL_SIM if TRADOVATE_SIM else TRADOVATE_API_URL_LIVE

# ── Risk Management ─────────────────────────────────────────
# Account sizes
HYPERLIQUID_ACCOUNT_SIZE = 1000   # USD — your Hyperliquid balance
TRADOVATE_ACCOUNT_SIZE   = 50000  # USD — sim account size

# Per-trade risk — set below after PROP_FIRM_ACCOUNT_TYPE is defined

# Position sizing (auto-calculated from SL distance)
# Formula: position_size = (account * risk_pct) / (entry - stop_loss)
# This means if SL is tight, you take more contracts. If wide, fewer.
AUTO_SIZE_FROM_SL        = True   # True = calculate size from SL distance
MAX_POSITION_SIZE_USD    = 200    # hard cap regardless of SL calc

# Daily loss limits
MAX_DAILY_LOSS_PCT        = 0.03  # 3% of account = stop trading for the day
MAX_DAILY_LOSS_FUTURES    = 500   # USD hard limit for futures (prop firm safe)
MAX_DAILY_LOSS_CRYPTO     = 50    # USD hard limit for crypto

# Prop firm specific limits (auto-applied when PROP_FIRM_ACTIVE = True)
PROP_FIRM_MAX_DAILY_LOSS_PCT  = 0.025  # 2.5% daily loss limit
PROP_FIRM_MAX_TOTAL_LOSS_PCT  = 0.08   # 8% total drawdown limit
PROP_FIRM_PROFIT_TARGET_PCT   = 0.08   # 8% profit target

# Drawdown kill switch — stops ALL trading for the day
KILL_SWITCH_DAILY_LOSS_PCT    = 0.03   # 3% loss = kill switch activates

# Concurrent positions
MAX_CONCURRENT_FUTURES   = 1     # max open futures positions at once
MAX_CONCURRENT_CRYPTO    = 3     # max open crypto positions at once

# Minimum R:R to take a trade (from strategy signal)
MIN_RR_RATIO             = 1.5   # skip signals with R:R below this

# ── Tokens to trade ─────────────────────────────────────────

# ── Real-World Asset (RWA) Perps on Hyperliquid via Trade[XYZ] ──
# S&P 500 perpetual launched March 18 2026 — officially licensed.
# Trades 24/7 on Hyperliquid. Same index as MES/ES but no Apex rules.
# Symbol on Hyperliquid: "SPX" (via Trade[XYZ] market)
# Note: Outside NYSE hours, price constrained by Discovery Bounds.
#       Full price discovery during US market hours (9:30am-4pm EST).
HYPERLIQUID_RWA_TOKENS = [
    "SPX",          # S&P 500 perp — launched March 2026 (Trade[XYZ])
    # "NDX",        # Nasdaq 100 — expected next from Trade[XYZ]
    # "OIL",        # Crude oil — already $1B+ weekend volume
    # "GOLD",       # Gold — expected
]

# ── Crypto tokens — expanded universe ────────────────────────
HYPERLIQUID_TOKENS = [
    "BTC", "ETH", "SOL",           # Core
    "BNB", "XRP", "DOGE",          # Large cap
    "AVAX", "LINK", "ARB", "OP",   # L1/L2
]

# ── RBI backtest universe ─────────────────────────────────────
CRYPTO_BACKTEST_TOKENS = [
    "BTC", "ETH", "SOL",           # Core — always included
    "BNB", "AVAX", "LINK",         # Large cap alts
    "ARB", "OP", "UNI",            # DeFi/L2
    "PEPE", "WIF",                 # Meme (high volatility = more signals)
    "FET", "RNDR", "TAO",          # AI tokens (trending sector)
]

# SPX backtest — uses MES data as proxy (same underlying index).
# Once Hyperliquid SPX has 6+ months history, we'll use native data.
SPX_PROXY_SYMBOL = "MES"           # MES tracks S&P 500, same as SPX perp
SPX_POINT_VALUE  = 1.0             # SPX perp: $1 per point (vs $5 for MES)
SPX_TICK_SIZE    = 0.1             # 0.1 point minimum tick on HL

# ── Market maker token lists ──────────────────────────────────
# Best tokens for market making: high volume, tight natural spreads
MM_CRYPTO_TOKENS = ["TAO", "PEPE", "WIF", "SOL", "BNB"]
MM_RWA_TOKENS    = ["SPX"]        # S&P 500 perp — excellent MM opportunity

COINBASE_TOKENS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ── Futures (Tradovate / Apex) ────────────────────────────────
TRADOVATE_TOKENS   = ["MNQ", "MES", "MYM"]   # Micro contracts
# Start with 1 contract each. Margin: MNQ~$40, MES~$50, MYM~$50
DEFAULT_CONTRACTS  = 1

# TradingView backtest data directory
TV_DATA_DIR        = "src/data/tradingview"

# Webhook security (change this!)
WEBHOOK_SECRET    = os.getenv("WEBHOOK_SECRET", "algotec_secret_change_me")

# ── Data / Backtesting ──────────────────────────────────────
BACKTEST_INITIAL_CASH = 100_000
BACKTEST_COMMISSION   = 0.001    # 0.1%
DATA_TIMEFRAMES       = ["5m", "15m", "1H", "4H", "1D"]
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

# ── Data Sources ───────────────────────────────────────────────
# Priority: Databento > TradingView CSV > yfinance
# Set DATABENTO_API_KEY in .env to enable institutional data
# Futures symbols (MNQ/MES/MYM) auto-route to Databento if key present
DATABENTO_CACHE_HOURS  = 4      # hours before refreshing cached data
DATABENTO_DEFAULT_DAYS = 1825   # 5 years of history

# ── Execution Mode ─────────────────────────────────────────────
# Controls how ICT signals are handled
EXECUTION_MODE = "manual"   # "auto"   = execute immediately
                             # "manual" = wait for dashboard approval
                             # "notify" = alert only, never execute

# ── Prop Firm Settings (Apex $50k) ─────────────────────────────
PROP_FIRM_ACTIVE        = False     # ← flip to True when on Apex eval/performance
PROP_FIRM_NAME          = "apex"
PROP_FIRM_ACCOUNT_SIZE  = 50_000

# Account type changes which rules apply:
#   "eval"        — evaluation phase (trying to pass)
#   "performance" — funded account (after passing eval)
#
# Rule differences:
#   Eval:        No consistency rule, stricter drawdown
#   Performance: Consistency rule applies, profit splits begin
PROP_FIRM_ACCOUNT_TYPE  = "eval"    # "eval" | "performance"

# ── Kelly Criterion Position Sizing ───────────────────────────
# When True: uses each strategy's historical win rate + R:R to
# calculate optimal position size via Kelly Criterion (half-Kelly).
# When False: uses flat RISK_PER_TRADE_PCT for all trades.
# Prop firm: Kelly is auto-capped at eval/PA limits regardless.
# Personal account: Kelly can size more aggressively.
USE_KELLY_SIZING    = False  # ← set True for personal account, False for prop firm

# Risk per trade — 1% on eval (faster to target), 0.5% on performance (protect account)
RISK_PER_TRADE_PCT = 0.01 if PROP_FIRM_ACCOUNT_TYPE == "eval" else 0.005

# Rule 1 — Daily Drawdown Limit
# Max loss in a single trading day (from session open equity)
APEX_DAILY_DRAWDOWN     = 1_000    # $1,000

# Rule 2 — Maximum Trailing Drawdown
# Total account can never drop more than $2,000 from its highest point
APEX_MAX_DRAWDOWN       = 2_000    # $2,000 trailing from peak

# Rule 3 — Profit Target
# Must reach $3,000 profit to pass evaluation
APEX_PROFIT_TARGET      = 3_000    # $3,000

# Rule 4 — No News Trading
# Close ALL positions 5 minutes before high-impact news
# Do not open positions until 5 minutes after
APEX_NEWS_FILTER        = True
APEX_NEWS_CLOSE_BEFORE  = 5        # minutes before news
APEX_NEWS_OPEN_AFTER    = 5        # minutes after news

# Rule 5 — Consistency Rule (PERFORMANCE ACCOUNT ONLY)
# No single trading day can account for more than 50% of total profits.
# e.g. if total profit = $1,000, no day can have made > $500
# NOTE: This rule does NOT apply during evaluation.
#       Only enforced when PROP_FIRM_ACCOUNT_TYPE = "performance"
APEX_CONSISTENCY_PCT    = 0.50     # 50%

# Backward compat
PROP_FIRM_MAX_DAILY_LOSS_PCT  = APEX_DAILY_DRAWDOWN / PROP_FIRM_ACCOUNT_SIZE
PROP_FIRM_MAX_TOTAL_LOSS_PCT  = APEX_MAX_DRAWDOWN   / PROP_FIRM_ACCOUNT_SIZE
PROP_FIRM_PROFIT_TARGET_PCT   = APEX_PROFIT_TARGET  / PROP_FIRM_ACCOUNT_SIZE

# ── Apex Bridge Route ───────────────────────────────────────────
# How to get orders into your Apex-connected Tradovate account.
# Apex blocks direct API on evaluations — use a bridge instead.
#
# "tradovate_direct" → direct API (works on sim ONLY — blocked on Apex eval)
# "pickmytrade"      → PickMyTrade ($50/mo) — pickmytrade.trade
# "traderspost"      → TradersPost (~$30/mo) — traderspost.io  ← recommended
# "local"            → local relay (still needs API — only works on sim)
#
# For Apex eval: use "traderspost" or "pickmytrade"
# For Tradovate sim: use "tradovate_direct" or "local"
#
APEX_BRIDGE_ROUTE = "traderspost"   # ← recommended for Apex eval

# TradersPost settings (recommended — cleaner, cheaper than PickMyTrade)
# Get webhook URL from: traderspost.io → Strategies → your strategy → Webhook URL
TRADERSPOST_WEBHOOK_URL = os.getenv("TRADERSPOST_WEBHOOK_URL", "")

# PickMyTrade settings (alternative)
# Get from: pickmytrade.trade → Strategies → your strategy
PICKMYTRADE_TOKEN       = os.getenv("PICKMYTRADE_TOKEN", "")
PICKMYTRADE_STRATEGY_ID = os.getenv("PICKMYTRADE_STRATEGY_ID", "")