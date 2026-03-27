# ============================================================
# 🌙 Liquidation Tracker Agent
#
# Monitors liquidation events across crypto markets.
# Large liquidation spikes often signal local price reversals —
# a cascade of longs being wiped = potential bounce incoming.
#
# HOW TO RUN:
#   python src/agents/liquidation_agent.py
#   python src/agents/liquidation_agent.py --once
# ============================================================

import sys
import csv
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import SLEEP_BETWEEN_RUNS_SEC, HYPERLIQUID_TOKENS
from src.models.llm_router import model

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
LIQ_LOG   = REPO_ROOT / "src" / "data" / "liquidation_log.csv"
LIQ_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── thresholds ────────────────────────────────────────────────
# Alert if liquidations in a window exceed these USD values
ALERT_THRESHOLDS = {
    "15m": 5_000_000,    # $5M in 15 mins
    "1h":  20_000_000,   # $20M in 1 hour
    "4h":  50_000_000,   # $50M in 4 hours
}

# ── CoinGlass liquidation data ────────────────────────────────
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"

def _cg_headers() -> dict:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("COINGLASS_API_KEY", "")
    return {"coinglassSecret": key} if key else {}


def get_liquidations_coinglass(symbol: str, interval: str = "h1") -> dict:
    """
    Fetch liquidation data from CoinGlass.
    interval: m15 | h1 | h4 | h12 | d1
    Returns dict with long_liq, short_liq, total in USD.
    Requires COINGLASS_API_KEY in .env (free tier available).
    """
    url    = f"{COINGLASS_BASE}/liquidation_history"
    params = {"symbol": symbol, "interval": interval, "limit": 48}
    try:
        r    = requests.get(url, headers=_cg_headers(), params=params, timeout=10)
        data = r.json()
        if data.get("success"):
            records = data.get("data", [])
            if records:
                latest = records[-1]
                return {
                    "symbol":        symbol,
                    "interval":      interval,
                    "long_liq_usd":  float(latest.get("longLiquidationUsd", 0)),
                    "short_liq_usd": float(latest.get("shortLiquidationUsd", 0)),
                    "total_liq_usd": float(latest.get("longLiquidationUsd", 0))
                                   + float(latest.get("shortLiquidationUsd", 0)),
                    "timestamp":     latest.get("createTime", ""),
                    "source":        "coinglass",
                }
    except Exception as e:
        pass
    return {}


def get_liquidations_hyperliquid(symbol: str) -> dict:
    """
    Fallback: estimate liquidation activity from Hyperliquid
    by looking at large forced trades (no direct liq endpoint on free tier).
    """
    try:
        r = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "recentTrades", "coin": symbol},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
        trades = r.json()

        # Heuristic: very large rapid trades in same direction = likely liquidations
        now_ms    = int(time.time() * 1000)
        window_ms = 15 * 60 * 1000   # 15 minutes
        recent    = [t for t in trades if now_ms - int(t.get("time", 0)) < window_ms]

        long_liq  = sum(
            float(t.get("px", 0)) * float(t.get("sz", 0))
            for t in recent if t.get("side") == "B"
        )
        short_liq = sum(
            float(t.get("px", 0)) * float(t.get("sz", 0))
            for t in recent if t.get("side") == "A"
        )

        return {
            "symbol":        symbol,
            "interval":      "15m",
            "long_liq_usd":  round(long_liq, 0),
            "short_liq_usd": round(short_liq, 0),
            "total_liq_usd": round(long_liq + short_liq, 0),
            "source":        "hyperliquid_estimate",
        }
    except Exception:
        return {}


def get_liquidations(symbol: str, interval: str = "h1") -> dict:
    """Try CoinGlass first, fall back to Hyperliquid estimate."""
    data = get_liquidations_coinglass(symbol, interval)
    if data:
        return data
    return get_liquidations_hyperliquid(symbol)


# ── AI analysis ───────────────────────────────────────────────
LIQ_ANALYSIS_PROMPT = """You are a crypto derivatives market expert.
Analyse the following liquidation event data and give a trading signal.

Large long liquidations  → price fell fast, possible bounce opportunity (longs wiped = selling pressure exhausted)
Large short liquidations → price rose fast, possible pullback opportunity (shorts wiped = buying pressure exhausted)

DATA:
{data}

Respond ONLY with valid JSON:
{{
  "signal":         "BUY_OPPORTUNITY" | "SELL_OPPORTUNITY" | "WATCH" | "NEUTRAL",
  "confidence":     0.0-1.0,
  "reasoning":      "One sentence explanation",
  "dominant_side":  "LONGS_LIQUIDATED" | "SHORTS_LIQUIDATED" | "MIXED",
  "alert_level":    "HIGH" | "MEDIUM" | "LOW",
  "suggested_action": "What a trader should consider doing right now"
}}"""


def analyse_liquidations(symbol: str, liq_data: dict, window: str) -> dict:
    """Ask DeepSeek to interpret liquidation spike."""
    payload = {
        "symbol":  symbol,
        "window":  window,
        "data":    liq_data,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        raw    = model.chat(
            system_prompt="You are a crypto derivatives analyst. Return only valid JSON.",
            user_prompt=LIQ_ANALYSIS_PROMPT.format(data=json.dumps(payload, indent=2)),
        )
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["symbol"] = symbol
        result["window"] = window
        return result
    except Exception as e:
        return {
            "symbol": symbol, "window": window,
            "signal": "NEUTRAL", "alert_level": "LOW",
            "reasoning": str(e),
        }


# ── logging ───────────────────────────────────────────────────
def _log(row: dict):
    write_header = not LIQ_LOG.exists()
    with open(LIQ_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ═══════════════════════════════════════════════════════════════
class LiquidationAgent:

    WINDOWS = [
        ("15m", "m15", ALERT_THRESHOLDS["15m"]),
        ("1h",  "h1",  ALERT_THRESHOLDS["1h"]),
        ("4h",  "h4",  ALERT_THRESHOLDS["4h"]),
    ]

    def __init__(self):
        self.symbols = HYPERLIQUID_TOKENS
        print("💦 Liquidation Tracker initialised")
        print(f"   Symbols : {self.symbols}")
        print(f"   Alerts  :")
        for label, _, thresh in self.WINDOWS:
            print(f"     {label}: > ${thresh/1e6:.0f}M")
        print()

    def scan(self) -> list[dict]:
        print(f"\n{'═'*60}")
        print(f"💦 Liquidation scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        all_alerts = []

        for symbol in self.symbols:
            print(f"\n── {symbol} ──")
            symbol_alerts = []

            for label, cg_interval, threshold in self.WINDOWS:
                try:
                    liq = get_liquidations(symbol, cg_interval)
                    if not liq:
                        continue

                    total     = liq.get("total_liq_usd", 0)
                    long_liq  = liq.get("long_liq_usd", 0)
                    short_liq = liq.get("short_liq_usd", 0)
                    source    = liq.get("source", "unknown")

                    bar_long  = "█" * min(int(long_liq  / threshold * 20), 20)
                    bar_short = "█" * min(int(short_liq / threshold * 20), 20)

                    print(f"   [{label}] Long : ${long_liq/1e6:>6.2f}M {bar_long}")
                    print(f"   [{label}] Short: ${short_liq/1e6:>6.2f}M {bar_short}")
                    print(f"          Total: ${total/1e6:.2f}M  (source: {source})")

                    if total >= threshold:
                        print(f"   🚨 SPIKE detected in {label} window!")
                        analysis = analyse_liquidations(symbol, liq, label)

                        signal   = analysis.get("signal", "NEUTRAL")
                        level    = analysis.get("alert_level", "LOW")
                        reason   = analysis.get("reasoning", "")
                        action   = analysis.get("suggested_action", "")
                        dominant = analysis.get("dominant_side", "MIXED")

                        icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(level, "ℹ️")
                        sig_icon = {
                            "BUY_OPPORTUNITY":  "🟢",
                            "SELL_OPPORTUNITY": "🔴",
                            "WATCH":            "👀",
                            "NEUTRAL":          "⚪",
                        }.get(signal, "⚪")

                        print(f"\n   {icon} {sig_icon} [{level}] {signal} — {dominant}")
                        print(f"   📌 {reason}")
                        print(f"   💡 {action}")

                        row = {
                            "timestamp":   datetime.now().isoformat(),
                            "symbol":      symbol,
                            "window":      label,
                            "total_liq_m": round(total / 1e6, 3),
                            "long_liq_m":  round(long_liq  / 1e6, 3),
                            "short_liq_m": round(short_liq / 1e6, 3),
                            "dominant":    dominant,
                            "signal":      signal,
                            "alert_level": level,
                            "reasoning":   reason,
                            "action":      action,
                        }
                        _log(row)
                        symbol_alerts.append(row)

                except Exception as e:
                    print(f"   ❌ Error [{label}]: {e}")

                time.sleep(0.3)

            if not symbol_alerts:
                print(f"   ✅ No liquidation spikes detected")
            else:
                all_alerts.extend(symbol_alerts)

        print(f"\n✅ Scan complete. {len(all_alerts)} spike alert(s).")
        return all_alerts

    def run(self):
        print("🚀 Liquidation Tracker running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.scan()
                print(f"\n😴 Next scan in {SLEEP_BETWEEN_RUNS_SEC}s...")
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)
        except KeyboardInterrupt:
            print("\n🛑 Liquidation Tracker stopped.")


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="💦 Liquidation Tracker")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    args = parser.parse_args()

    agent = LiquidationAgent()
    if args.once:
        agent.scan()
    else:
        agent.run()
