# ============================================================
# 🌙 Live Trading Agent
#
# HOW IT WORKS:
#   1. Fetches OHLCV + indicators for each token
#   2. Sends data to DeepSeek AI for a BUY / SELL / HOLD decision
#   3. Checks the Risk Agent before every trade
#   4. Executes via Hyperliquid or Coinbase
#   5. Logs every decision and trade to CSV
#
# HOW TO RUN:
#   python src/agents/trading_agent.py
#
# ⚠️  Only run after backtesting a strategy. Start with tiny size.
# ============================================================

import sys
import csv
import time
import json
import traceback
import pandas as pd
from src.models.ta_wrapper import ta
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config          import (EXCHANGE, SLEEP_BETWEEN_RUNS_SEC,
                                  HYPERLIQUID_TOKENS, COINBASE_TOKENS,
                                  MAX_POSITION_SIZE_USD, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
from src.models.llm_router import model
from src.exchanges.router      import get_price, buy, sell, close, active_symbols
from src.agents.risk_agent     import risk
from src.data.fetcher          import get_ohlcv

# ── paths ────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
TRADE_LOG  = REPO_ROOT / "src" / "data" / "trade_log.csv"
TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── per-run config ────────────────────────────────────────────
TRADE_SIZE_USD  = 50        # USD per trade (start small!)
TIMEFRAME       = "1H"
CANDLES_BACK    = 100       # how many candles to send to AI

# ── AI system prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert cryptocurrency trading analyst.
You will receive OHLCV candle data with technical indicators for a token.
Analyse the data and decide whether to BUY, SELL, or HOLD.

Respond ONLY with valid JSON — no markdown, no explanation:
{
  "decision":    "BUY" | "SELL" | "HOLD",
  "confidence":  0.0-1.0,
  "reasoning":   "One sentence max",
  "key_signals": ["signal1", "signal2"]
}

Rules:
- Only BUY or SELL when confidence >= 0.70
- Consider trend, momentum, and volume together
- When uncertain, return HOLD
- Never chase parabolic moves"""

USER_PROMPT = """Analyse this {symbol} {timeframe} data and give a trading decision.

Recent candles (newest last):
{candle_data}

Current indicators:
{indicators}

Current price: ${price:.2f}
Exchange: {exchange}"""


def _log_trade(row: dict):
    write_header = not TRADE_LOG.exists()
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


def _compute_indicators(df: pd.DataFrame) -> dict:
    """Calculate key technical indicators from OHLCV dataframe."""
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    indicators = {}

    # Trend
    indicators["ema_9"]   = round(float(ta.ema(close, 9).iloc[-1]),   4)
    indicators["ema_21"]  = round(float(ta.ema(close, 21).iloc[-1]),  4)
    indicators["ema_200"] = round(float(ta.ema(close, 200).iloc[-1]), 4)

    # Momentum
    rsi = ta.rsi(close, 14)
    indicators["rsi_14"] = round(float(rsi.iloc[-1]), 2)

    # MACD
    macd_df = ta.macd(close)
    if macd_df is not None and not macd_df.empty:
        indicators["macd"]        = round(float(macd_df.iloc[-1, 0]), 4)
        indicators["macd_signal"] = round(float(macd_df.iloc[-1, 2]), 4)
        indicators["macd_hist"]   = round(float(macd_df.iloc[-1, 1]), 4)

    # Bollinger Bands
    bb = ta.bbands(close, 20)
    if bb is not None and not bb.empty:
        indicators["bb_upper"] = round(float(bb.iloc[-1, 0]), 4)
        indicators["bb_mid"]   = round(float(bb.iloc[-1, 1]), 4)
        indicators["bb_lower"] = round(float(bb.iloc[-1, 2]), 4)

    # Volume
    avg_vol = float(volume.rolling(20).mean().iloc[-1])
    cur_vol = float(volume.iloc[-1])
    indicators["volume_ratio"] = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    # Trend direction
    indicators["trend"] = (
        "UPTREND"   if indicators["ema_9"] > indicators["ema_21"] > indicators["ema_200"]
        else "DOWNTREND" if indicators["ema_9"] < indicators["ema_21"] < indicators["ema_200"]
        else "SIDEWAYS"
    )

    return indicators


def _format_candles(df: pd.DataFrame, n: int = 20) -> str:
    """Format last N candles as a compact string for the AI."""
    tail = df.tail(n)
    lines = []
    for idx, row in tail.iterrows():
        lines.append(
            f"{idx.strftime('%Y-%m-%d %H:%M')} | "
            f"O:{row['Open']:.2f} H:{row['High']:.2f} "
            f"L:{row['Low']:.2f} C:{row['Close']:.2f} "
            f"V:{row['Volume']:.0f}"
        )
    return "\n".join(lines)


def analyse_symbol(symbol: str) -> dict | None:
    """Run full AI analysis for one symbol. Returns decision dict or None."""
    try:
        # 1 — fetch data
        df    = get_ohlcv(symbol, exchange=EXCHANGE, timeframe=TIMEFRAME, days_back=59)
        price = float(df["Close"].iloc[-1])

        # 2 — compute indicators
        indicators = _compute_indicators(df)

        # 3 — ask DeepSeek
        prompt = USER_PROMPT.format(
            symbol=symbol,
            timeframe=TIMEFRAME,
            candle_data=_format_candles(df, n=20),
            indicators=json.dumps(indicators, indent=2),
            price=price,
            exchange=EXCHANGE,
        )
        raw    = model.chat(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["symbol"] = symbol
        result["price"]  = price

        return result

    except Exception as e:
        print(f"  ❌ Analysis failed for {symbol}: {e}")
        return None


def execute_decision(decision: dict) -> bool:
    """Execute a trade based on AI decision. Returns True if trade placed."""
    symbol     = decision["symbol"]
    action     = decision["decision"]
    confidence = decision.get("confidence", 0)
    price      = decision["price"]

    print(f"\n  🤖 {symbol}: {action} (confidence: {confidence:.0%})")
    print(f"     Reasoning: {decision.get('reasoning', '—')}")
    print(f"     Signals:   {decision.get('key_signals', [])}")

    if action == "HOLD" or confidence < 0.70:
        print("     ⏸️  No trade (HOLD or low confidence)")
        return False

    direction = "buy" if action == "BUY" else "sell"
    allowed, reason = risk.check_trade(symbol, TRADE_SIZE_USD, direction)
    if not allowed:
        print(f"     🚫 Risk agent blocked: {reason}")
        return False

    sl = risk.stop_loss_price(price, "long" if action == "BUY" else "short")
    tp = risk.take_profit_price(price, "long" if action == "BUY" else "short")

    try:
        if action == "BUY":
            result = buy(symbol, TRADE_SIZE_USD)
        else:
            result = sell(symbol, TRADE_SIZE_USD)

        print(f"     ✅ Trade placed | SL: ${sl:.2f} | TP: ${tp:.2f}")

        _log_trade({
            "timestamp":  datetime.now().isoformat(),
            "exchange":   EXCHANGE,
            "symbol":     symbol,
            "action":     action,
            "usd_amount": TRADE_SIZE_USD,
            "price":      price,
            "stop_loss":  sl,
            "take_profit": tp,
            "confidence": confidence,
            "reasoning":  decision.get("reasoning", ""),
        })
        return True

    except Exception as e:
        print(f"     ❌ Trade execution failed: {e}")
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════
class TradingAgent:

    def __init__(self):
        self.symbols = active_symbols()
        print("🌙 Live Trading Agent initialised")
        print(f"   Exchange : {EXCHANGE}")
        print(f"   Symbols  : {self.symbols}")
        print(f"   Size/trade: ${TRADE_SIZE_USD}")
        print(f"   Timeframe: {TIMEFRAME}")
        print(f"   Loop interval: {SLEEP_BETWEEN_RUNS_SEC}s\n")

    def run_once(self):
        """Analyse all symbols and execute any qualifying trades."""
        print(f"\n{'═'*60}")
        print(f"🔍 Scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Print portfolio summary first
        risk.portfolio_summary()

        trades_placed = 0
        for symbol in self.symbols:
            print(f"\n── {symbol} ──")
            decision = analyse_symbol(symbol)
            if decision:
                placed = execute_decision(decision)
                if placed:
                    trades_placed += 1
            time.sleep(1)   # avoid rate limiting

        print(f"\n✅ Scan complete. Trades placed: {trades_placed}")

    def run(self):
        """Continuous loop — runs forever until Ctrl+C."""
        print("🚀 Starting live trading loop. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.run_once()
                print(f"\n😴 Sleeping {SLEEP_BETWEEN_RUNS_SEC}s...")
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)
        except KeyboardInterrupt:
            print("\n\n🛑 Trading agent stopped by user.")


# ── entrypoint ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🌙 Live Trading Agent")
    parser.add_argument("--once", action="store_true",
                        help="Run a single scan then exit (useful for testing)")
    args = parser.parse_args()

    agent = TradingAgent()
    if args.once:
        agent.run_once()
    else:
        agent.run()
