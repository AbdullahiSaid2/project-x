# ============================================================
# 🌙 Regime Detection Agent
#
# Detects the current market regime (Bull Trend, Bear Trend,
# High Volatility, Low Volatility, Sideways) and recommends
# which strategies to activate or pause accordingly.
#
# REGIMES:
#   BULL_TREND    → Run trend-following, momentum strategies
#   BEAR_TREND    → Run mean-reversion, short strategies
#   HIGH_VOL      → Reduce position sizes, widen stops
#   LOW_VOL       → Bollinger squeeze, breakout strategies
#   SIDEWAYS      → Mean reversion, funding arb
#
# HOW TO RUN:
#   python src/agents/regime_agent.py
#   python src/agents/regime_agent.py --once
# ============================================================

import sys
import csv
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import EXCHANGE, HYPERLIQUID_TOKENS
from src.data.fetcher          import get_ohlcv
from src.models.ta_wrapper     import ta
from src.models.llm_router import model

REPO_ROOT   = Path(__file__).resolve().parents[2]
REGIME_LOG  = REPO_ROOT / "src" / "data" / "regime_log.csv"
REGIME_FILE = REPO_ROOT / "src" / "data" / "current_regime.json"
REGIME_LOG.parent.mkdir(parents=True, exist_ok=True)


REGIME_PROMPT = """You are a market regime classification expert.
Analyse the market indicators below and classify the current regime.

DATA:
{data}

Respond ONLY with valid JSON:
{{
  "regime":       "BULL_TREND|BEAR_TREND|HIGH_VOLATILITY|LOW_VOLATILITY|SIDEWAYS",
  "confidence":   0.0-1.0,
  "sub_regime":   "e.g. BULL_PULLBACK, BEAR_BOUNCE, RANGE_BOUND",
  "strategies_activate":  ["list of strategy types to run now"],
  "strategies_pause":     ["list of strategy types to pause now"],
  "position_size_factor": 0.5-1.5,
  "reasoning":    "two sentences max",
  "key_signals":  ["signal1", "signal2", "signal3"]
}}

Strategy types: TREND_FOLLOWING, MEAN_REVERSION, BREAKOUT,
MOMENTUM, FUNDING_ARB, COPY_TRADING, ICT_SETUP"""


def compute_regime_indicators(symbol: str) -> dict:
    """Compute regime indicators across multiple timeframes."""
    try:
        df_d  = get_ohlcv(symbol, exchange=EXCHANGE, timeframe="1D", days_back=90)
        df_h  = get_ohlcv(symbol, exchange=EXCHANGE, timeframe="1H", days_back=30)

        close_d = df_d["Close"]
        close_h = df_h["Close"]

        # ── Trend indicators ───────────────────────────────────
        ema20  = float(ta.ema(close_d, 20).iloc[-1])
        ema50  = float(ta.ema(close_d, 50).iloc[-1])
        ema200 = float(ta.ema(close_d, 200).iloc[-1])
        price  = float(close_d.iloc[-1])

        # ── Volatility ─────────────────────────────────────────
        returns    = close_d.pct_change().dropna()
        vol_20     = float(returns.tail(20).std() * np.sqrt(365) * 100)  # annualised
        vol_5      = float(returns.tail(5).std()  * np.sqrt(365) * 100)
        atr_14     = float(ta.atr(df_d["High"], df_d["Low"], close_d, 14).iloc[-1])
        atr_pct    = atr_14 / price * 100

        # ── Momentum ───────────────────────────────────────────
        rsi_14     = float(ta.rsi(close_d, 14).iloc[-1])
        rsi_h      = float(ta.rsi(close_h, 14).iloc[-1])
        macd_df    = ta.macd(close_d)
        macd_hist  = float(macd_df.iloc[-1, 1]) if macd_df is not None else 0

        # ── Range analysis ─────────────────────────────────────
        high_20    = float(df_d["High"].tail(20).max())
        low_20     = float(df_d["Low"].tail(20).min())
        range_pct  = (high_20 - low_20) / low_20 * 100
        price_pos  = (price - low_20) / (high_20 - low_20) * 100  # 0-100%

        # ── Trend direction ────────────────────────────────────
        above_ema200 = price > ema200
        above_ema50  = price > ema50
        ema_aligned_bull = ema20 > ema50 > ema200
        ema_aligned_bear = ema20 < ema50 < ema200

        # ── Simple regime classifier ───────────────────────────
        if ema_aligned_bull and rsi_14 > 50:
            prelim = "BULL_TREND"
        elif ema_aligned_bear and rsi_14 < 50:
            prelim = "BEAR_TREND"
        elif vol_5 > vol_20 * 1.5:
            prelim = "HIGH_VOLATILITY"
        elif atr_pct < 1.5:
            prelim = "LOW_VOLATILITY"
        else:
            prelim = "SIDEWAYS"

        return {
            "symbol":           symbol,
            "price":            round(price, 2),
            "ema20":            round(ema20, 2),
            "ema50":            round(ema50, 2),
            "ema200":           round(ema200, 2),
            "above_ema200":     above_ema200,
            "ema_bull_aligned": ema_aligned_bull,
            "ema_bear_aligned": ema_aligned_bear,
            "rsi_daily":        round(rsi_14, 1),
            "rsi_hourly":       round(rsi_h, 1),
            "macd_histogram":   round(macd_hist, 4),
            "vol_20d_annual":   round(vol_20, 1),
            "vol_5d_annual":    round(vol_5, 1),
            "vol_expanding":    vol_5 > vol_20 * 1.2,
            "atr_pct":          round(atr_pct, 2),
            "range_20d_pct":    round(range_pct, 1),
            "price_position":   round(price_pos, 1),
            "preliminary_regime": prelim,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "preliminary_regime": "UNKNOWN"}


def log_regime(regime_data: dict):
    row = {
        "timestamp":        datetime.now().isoformat(),
        "symbol":           regime_data.get("symbol", "BTC"),
        "regime":           regime_data.get("regime", ""),
        "sub_regime":       regime_data.get("sub_regime", ""),
        "confidence":       regime_data.get("confidence", 0),
        "position_factor":  regime_data.get("position_size_factor", 1.0),
        "reasoning":        regime_data.get("reasoning", ""),
        "activate":         "|".join(regime_data.get("strategies_activate", [])),
        "pause":            "|".join(regime_data.get("strategies_pause", [])),
    }
    write_header = not REGIME_LOG.exists()
    with open(REGIME_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


REGIME_ICONS = {
    "BULL_TREND":      "🚀",
    "BEAR_TREND":      "🐻",
    "HIGH_VOLATILITY": "⚡",
    "LOW_VOLATILITY":  "😴",
    "SIDEWAYS":        "↔️",
    "UNKNOWN":         "❓",
}


class RegimeAgent:

    def __init__(self):
        self.primary_symbol = "BTC"
        print("🔭 Regime Detection Agent initialised")
        print(f"   Primary symbol: {self.primary_symbol}")

    def detect(self) -> dict:
        print(f"\n{'═'*60}")
        print(f"🔭 Regime scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Compute indicators for BTC (market leader)
        print(f"\n  📊 Computing indicators for {self.primary_symbol}...")
        indicators = compute_regime_indicators(self.primary_symbol)

        if "error" in indicators:
            print(f"  ❌ {indicators['error']}")
            return {}

        # AI classification
        print(f"  🤖 Classifying regime...")
        raw    = model.chat(
            system_prompt="You are a market regime expert. Return only valid JSON.",
            user_prompt=REGIME_PROMPT.format(data=json.dumps(indicators, indent=2)),
        )
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["symbol"]    = self.primary_symbol
        result["timestamp"] = datetime.now().isoformat()

        regime     = result.get("regime", "UNKNOWN")
        sub_regime = result.get("sub_regime", "")
        confidence = result.get("confidence", 0)
        factor     = result.get("position_size_factor", 1.0)
        reasoning  = result.get("reasoning", "")
        icon       = REGIME_ICONS.get(regime, "❓")

        print(f"\n  {icon} REGIME: {regime}")
        if sub_regime:
            print(f"     Sub-regime: {sub_regime}")
        print(f"     Confidence: {confidence:.0%}")
        print(f"     Position factor: {factor:.1f}x")
        print(f"     Reasoning: {reasoning}")
        print(f"\n  ✅ Activate: {result.get('strategies_activate', [])}")
        print(f"  ⏸️  Pause   : {result.get('strategies_pause', [])}")
        print(f"\n  📈 Indicators: RSI {indicators['rsi_daily']:.0f} | "
              f"Vol {indicators['vol_20d_annual']:.0f}% | "
              f"EMA {'🟢 Bull' if indicators['ema_bull_aligned'] else '🔴 Bear' if indicators['ema_bear_aligned'] else '⚪ Mixed'}")

        # Save current regime to JSON for other agents to read
        REGIME_FILE.write_text(json.dumps(result, indent=2))
        log_regime(result)

        return result

    def run(self):
        print("🚀 Regime Agent running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.detect()
                # Regime changes slowly — check every 4 hours
                print(f"\n😴 Next regime check in 4 hours...")
                time.sleep(4 * 3600)
        except KeyboardInterrupt:
            print("\n🛑 Regime Agent stopped.")


def get_current_regime() -> dict:
    """Read the latest regime from disk. Used by other agents."""
    if REGIME_FILE.exists():
        return json.loads(REGIME_FILE.read_text())
    return {"regime": "UNKNOWN", "position_size_factor": 1.0}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    agent = RegimeAgent()
    if args.once:
        agent.detect()
    else:
        agent.run()

from openbb import obb

# Fed funds rate
obb.economy.interest_rates()

# Jobs data
obb.economy.unemployment()

# Earnings calendar (useful for MES/ES trading around earnings)
obb.equity.calendar.earnings(start_date="2026-04-01")
