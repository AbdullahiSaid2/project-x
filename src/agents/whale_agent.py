# ============================================================
# 🌙 Whale Monitor Agent
#
# Tracks large wallet movements on-chain and via exchange OI.
# Alerts when a whale opens or closes a significant position.
#
# HOW TO RUN:
#   python src/agents/whale_agent.py
#   python src/agents/whale_agent.py --once   (single scan)
# ============================================================

import sys
import csv
import json
import time
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config              import EXCHANGE, SLEEP_BETWEEN_RUNS_SEC, HYPERLIQUID_TOKENS
from src.models.llm_router import model

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
WHALE_LOG  = REPO_ROOT / "src" / "data" / "whale_log.csv"
WHALE_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── thresholds ────────────────────────────────────────────────
WHALE_USD_THRESHOLD   = 500_000     # flag positions > $500k
OI_SPIKE_THRESHOLD    = 0.05        # flag if OI changes > 5% in one scan
FUNDING_ALERT_PCT     = 0.10        # flag funding rate > 0.1% per 8h

# ── Hyperliquid API helpers ───────────────────────────────────
HL_INFO = "https://api.hyperliquid.xyz/info"

def _hl_post(payload: dict) -> dict:
    r = requests.post(HL_INFO, json=payload,
                      headers={"Content-Type": "application/json"}, timeout=10)
    r.raise_for_status()
    return r.json()


def get_open_interest(symbol: str) -> dict:
    """Get open interest and funding rate from Hyperliquid."""
    data = _hl_post({"type": "metaAndAssetCtxs"})
    universe = data[0].get("universe", [])
    ctxs     = data[1]

    for i, asset in enumerate(universe):
        if asset.get("name") == symbol:
            ctx = ctxs[i]
            return {
                "symbol":        symbol,
                "open_interest": float(ctx.get("openInterest", 0)),
                "funding_rate":  float(ctx.get("funding", 0)),
                "mark_price":    float(ctx.get("markPx", 0)),
                "oi_usd":        float(ctx.get("openInterest", 0)) * float(ctx.get("markPx", 0)),
            }
    return {}


def get_top_traders(symbol: str, n: int = 20) -> list[dict]:
    """Get the largest open positions for a symbol."""
    data = _hl_post({"type": "clearinghouseState", "user": "0x0000000000000000000000000000000000000000"})
    # Public leaderboard endpoint
    try:
        leaderboard = _hl_post({"type": "leaderboard"})
        traders = []
        for entry in leaderboard[:n]:
            traders.append({
                "address":   entry.get("ethAddress", ""),
                "pnl":       float(entry.get("pnl", 0)),
                "volume":    float(entry.get("volume", 0)),
                "roi":       float(entry.get("roi", 0)),
            })
        return traders
    except Exception:
        return []


def get_large_trades(symbol: str, min_usd: float = 100_000) -> list[dict]:
    """Fetch recent trades above a USD threshold from Hyperliquid."""
    try:
        data  = _hl_post({"type": "recentTrades", "coin": symbol})
        large = []
        for trade in data:
            price = float(trade.get("px", 0))
            size  = float(trade.get("sz", 0))
            usd   = price * size
            if usd >= min_usd:
                large.append({
                    "symbol":    symbol,
                    "side":      trade.get("side", ""),
                    "price":     price,
                    "size":      size,
                    "usd_value": round(usd, 2),
                    "time":      trade.get("time", ""),
                })
        return large
    except Exception:
        return []


# ── AI analysis ───────────────────────────────────────────────
WHALE_ANALYSIS_PROMPT = """You are a crypto market analyst specialising in on-chain whale activity.
Analyse the following whale / open interest data and give a brief market impact assessment.

DATA:
{data}

Respond ONLY with valid JSON:
{{
  "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "key_observation": "One sentence describing the most important signal",
  "trading_implication": "One sentence on what this means for traders",
  "alert_level": "HIGH" | "MEDIUM" | "LOW"
}}"""


def analyse_whale_data(symbol: str, oi_data: dict, large_trades: list) -> dict:
    """Ask DeepSeek to interpret the whale activity."""
    payload = {
        "symbol":       symbol,
        "open_interest": oi_data,
        "large_trades":  large_trades[:10],
        "timestamp":     datetime.now().isoformat(),
    }
    try:
        raw    = model.chat(
            system_prompt="You are a crypto market analyst. Return only valid JSON.",
            user_prompt=WHALE_ANALYSIS_PROMPT.format(data=json.dumps(payload, indent=2)),
        )
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["symbol"] = symbol
        return result
    except Exception as e:
        return {"symbol": symbol, "alert_level": "LOW", "key_observation": str(e)}


# ── logging ───────────────────────────────────────────────────
def _log_whale(row: dict):
    write_header = not WHALE_LOG.exists()
    with open(WHALE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── previous OI store (in-memory, per session) ─────────────────
_prev_oi: dict[str, float] = {}


def _check_oi_spike(symbol: str, current_oi_usd: float) -> bool:
    prev = _prev_oi.get(symbol)
    if prev is None:
        _prev_oi[symbol] = current_oi_usd
        return False
    change_pct = abs(current_oi_usd - prev) / max(prev, 1)
    _prev_oi[symbol] = current_oi_usd
    return change_pct >= OI_SPIKE_THRESHOLD


# ═══════════════════════════════════════════════════════════════
class WhaleMonitor:

    def __init__(self):
        self.symbols = HYPERLIQUID_TOKENS
        print("🐋 Whale Monitor initialised")
        print(f"   Symbols          : {self.symbols}")
        print(f"   Whale threshold  : ${WHALE_USD_THRESHOLD:,.0f}")
        print(f"   OI spike trigger : {OI_SPIKE_THRESHOLD*100:.0f}%")
        print(f"   Funding alert    : {FUNDING_ALERT_PCT*100:.2f}% per 8h\n")

    def scan(self):
        print(f"\n{'═'*60}")
        print(f"🐋 Whale scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        alerts = []

        for symbol in self.symbols:
            print(f"\n── {symbol} ──")
            try:
                # 1 — Open Interest
                oi = get_open_interest(symbol)
                if not oi:
                    print(f"   ⚠️  No OI data")
                    continue

                oi_usd       = oi.get("oi_usd", 0)
                funding_rate = oi.get("funding_rate", 0)
                mark_price   = oi.get("mark_price", 0)
                oi_spike     = _check_oi_spike(symbol, oi_usd)

                print(f"   OI       : ${oi_usd:>14,.0f}")
                print(f"   Funding  : {funding_rate*100:>+.4f}% per 8h")
                print(f"   Mark px  : ${mark_price:,.2f}")

                # 2 — Large trades
                large_trades = get_large_trades(symbol, min_usd=WHALE_USD_THRESHOLD)
                if large_trades:
                    print(f"   🐋 {len(large_trades)} whale trade(s) detected!")
                    for t in large_trades[:3]:
                        print(f"      {t['side'].upper()} {t['size']} {symbol} "
                              f"@ ${t['price']:,.2f} = ${t['usd_value']:,.0f}")

                # 3 — Check alert conditions
                trigger = (
                    bool(large_trades)
                    or oi_spike
                    or abs(funding_rate) >= FUNDING_ALERT_PCT / 100
                )

                if trigger:
                    print(f"   🔔 Alert triggered — analysing with AI...")
                    analysis = analyse_whale_data(symbol, oi, large_trades)
                    alert_level = analysis.get("alert_level", "LOW")
                    sentiment   = analysis.get("market_sentiment", "NEUTRAL")
                    observation = analysis.get("key_observation", "")
                    implication = analysis.get("trading_implication", "")

                    icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(alert_level, "ℹ️")
                    print(f"\n   {icon} [{alert_level}] {sentiment}")
                    print(f"   📌 {observation}")
                    print(f"   💡 {implication}")

                    row = {
                        "timestamp":   datetime.now().isoformat(),
                        "symbol":      symbol,
                        "alert_level": alert_level,
                        "sentiment":   sentiment,
                        "oi_usd":      round(oi_usd, 0),
                        "funding_rate": round(funding_rate * 100, 4),
                        "large_trades": len(large_trades),
                        "oi_spike":    oi_spike,
                        "observation": observation,
                        "implication": implication,
                    }
                    _log_whale(row)
                    alerts.append(row)
                else:
                    print(f"   ✅ No whale activity detected")

            except Exception as e:
                print(f"   ❌ Error scanning {symbol}: {e}")

            time.sleep(0.5)

        print(f"\n✅ Whale scan complete. {len(alerts)} alert(s) fired.")
        return alerts

    def run(self):
        print("🚀 Whale Monitor running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.scan()
                print(f"\n😴 Next scan in {SLEEP_BETWEEN_RUNS_SEC}s...")
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)
        except KeyboardInterrupt:
            print("\n🛑 Whale Monitor stopped.")


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🐋 Whale Monitor")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    args = parser.parse_args()

    monitor = WhaleMonitor()
    if args.once:
        monitor.scan()
    else:
        monitor.run()
