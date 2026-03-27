# ============================================================
# 🌙 Copy Bot Agent
#
# Monitors top-performing wallets on Hyperliquid and mirrors
# their trades automatically. Uses the public leaderboard
# to identify consistently profitable traders.
#
# HOW TO RUN:
#   python src/agents/copy_bot_agent.py
#   python src/agents/copy_bot_agent.py --once
# ============================================================

import sys
import csv
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import SLEEP_BETWEEN_RUNS_SEC
from src.models.llm_router import model
from src.exchanges.router      import buy, sell, get_price
from src.agents.risk_agent     import risk

REPO_ROOT  = Path(__file__).resolve().parents[2]
COPY_LOG   = REPO_ROOT / "src" / "data" / "copy_bot_log.csv"
WALLET_DB  = REPO_ROOT / "src" / "data" / "tracked_wallets.json"
COPY_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
COPY_TRADE_SIZE_USD  = 25       # size per copied trade (start small)
MIN_TRADER_ROI       = 0.20     # only copy traders with 20%+ ROI
MIN_TRADER_VOLUME    = 100_000  # minimum $100k volume (proves active)
MAX_WALLETS_TO_TRACK = 5        # track top 5 wallets
COPY_DELAY_SECS      = 30       # wait 30s after detecting a trade
HL_INFO              = "https://api.hyperliquid.xyz/info"


def hl_post(payload: dict) -> dict:
    r = requests.post(HL_INFO, json=payload,
                      headers={"Content-Type": "application/json"}, timeout=10)
    r.raise_for_status()
    return r.json()


def get_leaderboard(top_n: int = 50) -> list[dict]:
    """Fetch top traders from Hyperliquid leaderboard."""
    try:
        data = hl_post({"type": "leaderboard"})
        traders = []
        for entry in data[:top_n]:
            roi    = float(entry.get("roi", 0))
            volume = float(entry.get("volume", 0))
            pnl    = float(entry.get("pnl", 0))
            addr   = entry.get("ethAddress", "")
            if roi >= MIN_TRADER_ROI and volume >= MIN_TRADER_VOLUME and addr:
                traders.append({
                    "address": addr,
                    "roi_pct": round(roi * 100, 2),
                    "volume":  round(volume, 0),
                    "pnl":     round(pnl, 2),
                    "rank":    entry.get("rank", 999),
                })
        traders.sort(key=lambda x: x["roi_pct"], reverse=True)
        return traders[:MAX_WALLETS_TO_TRACK]
    except Exception as e:
        print(f"  ❌ Leaderboard error: {e}")
        return []


def get_wallet_positions(address: str) -> list[dict]:
    """Get current open positions for a wallet."""
    try:
        data      = hl_post({"type": "clearinghouseState", "user": address})
        positions = []
        for pos in data.get("assetPositions", []):
            p    = pos.get("position", {})
            size = float(p.get("szi", 0))
            if size != 0:
                positions.append({
                    "symbol":      p.get("coin"),
                    "size":        size,
                    "side":        "LONG" if size > 0 else "SHORT",
                    "entry_price": float(p.get("entryPx", 0)),
                    "unrealised_pnl": float(p.get("unrealizedPnl", 0)),
                })
        return positions
    except Exception as e:
        return []


def load_wallet_db() -> dict:
    if WALLET_DB.exists():
        return json.loads(WALLET_DB.read_text())
    return {}


def save_wallet_db(db: dict):
    WALLET_DB.write_text(json.dumps(db, indent=2))


ANALYSIS_PROMPT = """You are a copy trading analyst.
Review these trader profiles and their current positions.
Decide which trades are worth copying RIGHT NOW.

TRADERS AND POSITIONS:
{data}

Consider:
- Is this a new position (not in our previous snapshot)?
- Is the position size significant (>$10k)?
- Does it align with current market conditions?
- Any red flags?

Respond ONLY with valid JSON:
{{
  "copy_trades": [
    {{
      "wallet": "0x...",
      "symbol": "BTC",
      "action": "BUY or SELL",
      "reason": "one sentence",
      "confidence": 0.0-1.0,
      "risk_flag": "any concern or null"
    }}
  ],
  "skip_reason": "why we're skipping others"
}}"""


def analyse_copy_opportunities(traders: list[dict],
                                prev_positions: dict) -> dict:
    """Ask DeepSeek which new positions are worth copying."""
    payload = []
    for t in traders:
        current = get_wallet_positions(t["address"])
        prev    = prev_positions.get(t["address"], [])
        prev_syms = {p["symbol"] for p in prev}
        new_pos   = [p for p in current if p["symbol"] not in prev_syms]

        if new_pos:
            payload.append({
                "address":  t["address"][:10] + "...",
                "roi_pct":  t["roi_pct"],
                "volume":   t["volume"],
                "new_positions": new_pos,
            })

    if not payload:
        return {"copy_trades": [], "skip_reason": "No new positions detected"}

    try:
        raw    = model.chat(
            system_prompt="You are a copy trading analyst. Return only valid JSON.",
            user_prompt=ANALYSIS_PROMPT.format(data=json.dumps(payload, indent=2)),
        )
        raw    = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"copy_trades": [], "skip_reason": f"Analysis error: {e}"}


def log_copy_trade(row: dict):
    write_header = not COPY_LOG.exists()
    with open(COPY_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


class CopyBotAgent:

    def __init__(self):
        self.wallet_db = load_wallet_db()
        print("🤖 Copy Bot Agent initialised")
        print(f"   Min ROI     : {MIN_TRADER_ROI*100:.0f}%")
        print(f"   Min volume  : ${MIN_TRADER_VOLUME:,}")
        print(f"   Max wallets : {MAX_WALLETS_TO_TRACK}")
        print(f"   Trade size  : ${COPY_TRADE_SIZE_USD}")

    def run_once(self):
        print(f"\n{'═'*60}")
        print(f"🤖 Copy Bot scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1 — Get top traders
        print(f"\n  📊 Fetching leaderboard...")
        traders = get_leaderboard()
        if not traders:
            print("  ❌ Could not fetch leaderboard")
            return

        print(f"  ✅ {len(traders)} qualifying traders found:")
        for t in traders:
            print(f"     {t['address'][:12]}... ROI: {t['roi_pct']:+.1f}% "
                  f"Vol: ${t['volume']/1e6:.1f}M")

        # 2 — Get previous positions snapshot
        prev_positions = self.wallet_db.get("positions", {})

        # 3 — Analyse for new copy opportunities
        print(f"\n  🔍 Checking for new positions...")
        analysis = analyse_copy_opportunities(traders, prev_positions)

        copy_trades = analysis.get("copy_trades", [])
        skip_reason = analysis.get("skip_reason", "")

        if not copy_trades:
            print(f"  ⏸️  No copy trades: {skip_reason}")
        else:
            print(f"\n  🎯 {len(copy_trades)} copy trade(s) identified:")
            for ct in copy_trades:
                confidence = ct.get("confidence", 0)
                risk_flag  = ct.get("risk_flag")
                print(f"\n  {'🟢' if ct['action']=='BUY' else '🔴'} "
                      f"{ct['symbol']} {ct['action']} — {confidence:.0%} confidence")
                print(f"     Reason: {ct.get('reason','')}")
                if risk_flag:
                    print(f"     ⚠️  Risk: {risk_flag}")

                if confidence < 0.65:
                    print(f"     ⏸️  Skipping — confidence too low")
                    continue

                # Risk check
                direction = ct["action"].lower()
                allowed, reason = risk.check_trade(
                    ct["symbol"], COPY_TRADE_SIZE_USD, direction
                )
                if not allowed:
                    print(f"     🚫 Risk blocked: {reason}")
                    continue

                # Execute
                try:
                    if ct["action"] == "BUY":
                        buy(ct["symbol"], COPY_TRADE_SIZE_USD)
                    else:
                        sell(ct["symbol"], COPY_TRADE_SIZE_USD)
                    print(f"     ✅ Copy trade executed!")
                    log_copy_trade({
                        "timestamp":  datetime.now().isoformat(),
                        "symbol":     ct["symbol"],
                        "action":     ct["action"],
                        "size_usd":   COPY_TRADE_SIZE_USD,
                        "confidence": confidence,
                        "reason":     ct.get("reason", ""),
                        "wallet":     ct.get("wallet", ""),
                        "executed":   True,
                    })
                except Exception as e:
                    print(f"     ❌ Execution failed: {e}")

        # 4 — Update position snapshot
        new_snapshot = {}
        for t in traders:
            new_snapshot[t["address"]] = get_wallet_positions(t["address"])
            time.sleep(0.5)

        self.wallet_db["positions"]    = new_snapshot
        self.wallet_db["last_updated"] = datetime.now().isoformat()
        self.wallet_db["tracked"]      = traders
        save_wallet_db(self.wallet_db)

    def run(self):
        print("🚀 Copy Bot running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.run_once()
                print(f"\n😴 Next scan in {SLEEP_BETWEEN_RUNS_SEC}s...")
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)
        except KeyboardInterrupt:
            print("\n🛑 Copy Bot stopped.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    agent = CopyBotAgent()
    if args.once:
        agent.run_once()
    else:
        agent.run()
