# ============================================================
# 🌙 VWAP Reversion Agent
#
# Trades mean reversion back to VWAP (Volume Weighted Average
# Price). Price tends to revert to VWAP especially intraday.
#
# LOGIC:
#   Price > VWAP + 1.5σ band → SHORT (overextended above)
#   Price < VWAP - 1.5σ band → LONG  (overextended below)
#   Target: VWAP (mean reversion)
#
# HOW TO RUN:
#   python src/agents/vwap_agent.py
#   python src/agents/vwap_agent.py --once
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

from src.config              import EXCHANGE, HYPERLIQUID_TOKENS, SLEEP_BETWEEN_RUNS_SEC
from src.data.fetcher         import get_ohlcv
from src.models.ta_wrapper    import ta
from src.models.llm_router import model
from src.exchanges.router     import get_price, buy, sell
from src.agents.risk_agent    import risk

REPO_ROOT = Path(__file__).resolve().parents[2]
VWAP_LOG  = REPO_ROOT / "src" / "data" / "vwap_log.csv"
VWAP_LOG.parent.mkdir(parents=True, exist_ok=True)

TRADE_SIZE_USD  = 50
VWAP_BAND_MULT  = 1.5    # standard deviations from VWAP to trigger
MIN_CONFIDENCE  = 0.65


def calculate_vwap_bands(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Calculate VWAP and bands.
    VWAP = Σ(Price × Volume) / Σ(Volume)
    Bands = VWAP ± multiplier × rolling std of price-VWAP
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vol           = df["Volume"]

    # Rolling VWAP
    cumulative_tp_vol = (typical_price * vol).rolling(window).sum()
    cumulative_vol    = vol.rolling(window).sum()
    vwap              = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)

    # Bands: rolling std of deviation from VWAP
    deviation = typical_price - vwap
    std_dev   = deviation.rolling(window).std()

    upper1 = vwap + (1.0 * std_dev)
    upper2 = vwap + (VWAP_BAND_MULT * std_dev)
    lower1 = vwap - (1.0 * std_dev)
    lower2 = vwap - (VWAP_BAND_MULT * std_dev)

    price  = float(df["Close"].iloc[-1])
    vwap_v = float(vwap.iloc[-1])
    upper2_v = float(upper2.iloc[-1])
    lower2_v = float(lower2.iloc[-1])
    upper1_v = float(upper1.iloc[-1])
    lower1_v = float(lower1.iloc[-1])

    # Signal
    if price > upper2_v:
        signal = "SHORT"
        distance_pct = (price - vwap_v) / vwap_v * 100
    elif price < lower2_v:
        signal = "LONG"
        distance_pct = (vwap_v - price) / vwap_v * 100
    else:
        signal = "HOLD"
        distance_pct = abs(price - vwap_v) / vwap_v * 100

    return {
        "price":       round(price, 4),
        "vwap":        round(vwap_v, 4),
        "upper2":      round(upper2_v, 4),
        "upper1":      round(upper1_v, 4),
        "lower1":      round(lower1_v, 4),
        "lower2":      round(lower2_v, 4),
        "signal":      signal,
        "distance_pct": round(distance_pct, 3),
        "above_vwap":  price > vwap_v,
    }


VWAP_PROMPT = """You are a VWAP mean reversion specialist.
Analyse this VWAP data and confirm if the trade signal is valid.

DATA:
{data}

Key questions:
1. Is price extended enough from VWAP to justify entry?
2. Is volume confirming the extension (low volume extension = weak)?
3. Any structural reason price might NOT revert to VWAP?

Respond ONLY with valid JSON:
{{"signal":"LONG"|"SHORT"|"HOLD","confidence":0.0-1.0,"reasoning":"one sentence"}}"""


def analyse_vwap(symbol: str, vwap_data: dict) -> dict:
    try:
        raw    = model.chat(
            system_prompt="You are a VWAP analyst. Return only valid JSON.",
            user_prompt=VWAP_PROMPT.format(data=json.dumps(vwap_data, indent=2)),
        )
        raw    = raw.replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        return {"signal": "HOLD", "confidence": 0, "reasoning": str(e)}


def log_vwap(row: dict):
    write_header = not VWAP_LOG.exists()
    with open(VWAP_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


class VWAPAgent:

    def __init__(self):
        self.symbols = HYPERLIQUID_TOKENS[:3]
        print("📊 VWAP Reversion Agent initialised")
        print(f"   Symbols   : {self.symbols}")
        print(f"   Band mult : {VWAP_BAND_MULT}σ")
        print(f"   Trade size: ${TRADE_SIZE_USD}")

    def scan(self):
        print(f"\n{'═'*60}")
        print(f"📊 VWAP scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for symbol in self.symbols:
            try:
                df        = get_ohlcv(symbol, exchange=EXCHANGE,
                                      timeframe="1H", days_back=10)
                vwap_data = calculate_vwap_bands(df)
                signal    = vwap_data["signal"]

                icon = {"LONG":"🟢","SHORT":"🔴","HOLD":"⚪"}.get(signal,"⚪")
                print(f"\n  {icon} {symbol}: {signal}")
                print(f"     Price  : ${vwap_data['price']:,.4f}")
                print(f"     VWAP   : ${vwap_data['vwap']:,.4f}")
                print(f"     Upper2 : ${vwap_data['upper2']:,.4f}")
                print(f"     Lower2 : ${vwap_data['lower2']:,.4f}")
                print(f"     Dist   : {vwap_data['distance_pct']:.3f}% from VWAP")

                if signal == "HOLD":
                    continue

                analysis = analyse_vwap(symbol, vwap_data)
                conf     = analysis.get("confidence", 0)
                print(f"     AI     : {analysis.get('reasoning','')} ({conf:.0%})")

                if conf < MIN_CONFIDENCE:
                    print(f"     ⏸️  Low confidence — skipping")
                    continue

                direction = "sell" if signal == "SHORT" else "buy"
                allowed, reason = risk.check_trade(symbol, TRADE_SIZE_USD, direction)
                if not allowed:
                    print(f"     🚫 Risk blocked: {reason}")
                    continue

                if signal == "LONG":
                    buy(symbol, TRADE_SIZE_USD)
                else:
                    sell(symbol, TRADE_SIZE_USD)

                print(f"     ✅ VWAP reversion trade placed!")
                log_vwap({
                    "timestamp": datetime.now().isoformat(),
                    "symbol":    symbol,
                    "signal":    signal,
                    "price":     vwap_data["price"],
                    "vwap":      vwap_data["vwap"],
                    "distance":  vwap_data["distance_pct"],
                    "confidence":conf,
                    "executed":  True,
                })

            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
            time.sleep(0.5)

    def run(self):
        print("🚀 VWAP Agent running. Ctrl+C to stop.\n")
        try:
            while True:
                self.scan()
                print(f"\n😴 Next scan in {SLEEP_BETWEEN_RUNS_SEC}s...")
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)
        except KeyboardInterrupt:
            print("\n🛑 VWAP Agent stopped.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    agent = VWAPAgent()
    if args.once:
        agent.scan()
    else:
        agent.run()
