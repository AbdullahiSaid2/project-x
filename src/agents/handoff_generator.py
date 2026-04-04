#!/usr/bin/env python3
# ============================================================
# 🌙 Algotec Handoff Generator
#
# Automatically regenerates ALGOTEC_HANDOFF.md from the LIVE
# state of the system — vault strategies, config values,
# known errors in logs, recent activity, agent status.
#
# Run automatically: triggered by vault changes, on schedule,
# or manually before ending a Claude session.
#
# HOW TO USE:
#   python src/agents/handoff_generator.py          # generate once
#   python src/agents/handoff_generator.py --watch  # auto-regenerate on changes
#   python src/agents/handoff_generator.py --diff   # show what changed vs last
#
# AUTOMATIC TRIGGERS (configured in server.py):
#   - After every RBI run that vaults new strategies
#   - After every vault_forward_test cycle
#   - Every 6 hours via scheduler
#   - On demand from dashboard Controls panel
# ============================================================

import os, sys, json, time, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "src" / "data"
VAULT_DIR = ROOT / "src" / "strategies" / "vault"
OUT_FILE  = ROOT / "ALGOTEC_HANDOFF.md"
AGENTS_DIR = ROOT / "src" / "agents"


# ══════════════════════════════════════════════════════════════
# DATA COLLECTORS — read live system state
# ══════════════════════════════════════════════════════════════

def get_vault_strategies() -> dict:
    """Read current vault strategies from vault_index.json."""
    idx = VAULT_DIR / "vault_index.json"
    if not idx.exists():
        return {"strategies": [], "count": 0}
    data = json.loads(idx.read_text())
    strats = data.get("strategies", [])
    futures_syms = {"MES", "MNQ", "MYM"}
    return {
        "strategies": strats,
        "count":   len(strats),
        "futures": [s for s in strats if s.get("symbol") in futures_syms],
        "crypto":  [s for s in strats if s.get("symbol") not in futures_syms],
    }


def get_ideas_count() -> int:
    """Count trade signal lines in ideas.txt."""
    ideas = DATA_DIR / "ideas.txt"
    if not ideas.exists():
        return 0
    count = 0
    for line in ideas.read_text().splitlines():
        if line.strip().startswith(("Long ", "Short ")):
            count += 1
    return count


def get_apex_state() -> dict:
    """Read current Apex eval state."""
    state_file = DATA_DIR / "apex_state.json"
    if not state_file.exists():
        return {"status": "no_data"}
    try:
        return json.loads(state_file.read_text())
    except Exception:
        return {"status": "parse_error"}


def get_weekly_brief() -> dict:
    """Read latest weekly briefing."""
    brief_file = DATA_DIR / "weekly_brief.json"
    if not brief_file.exists():
        return {}
    try:
        return json.loads(brief_file.read_text())
    except Exception:
        return {}


def get_mm_stats() -> dict:
    """Read market maker log stats."""
    log_file = DATA_DIR / "mm_v2_log.csv"
    if not log_file.exists():
        return {"fills": 0, "pnl": 0.0}
    try:
        lines = log_file.read_text().strip().splitlines()
        if len(lines) < 2:
            return {"fills": 0, "pnl": 0.0}
        # Last line PnL
        last = dict(zip(lines[0].split(","), lines[-1].split(",")))
        return {
            "fills":    len(lines) - 1,
            "last_pnl": float(last.get("pnl", 0)),
            "symbol":   last.get("symbol", "—"),
        }
    except Exception:
        return {"fills": 0, "pnl": 0.0}


def get_pairs_state() -> dict:
    """Read pairs trading state."""
    pf = DATA_DIR / "pairs_state.json"
    if not pf.exists():
        return {}
    try:
        return json.loads(pf.read_text())
    except Exception:
        return {}


def get_recent_signals() -> list:
    """Read last 5 signals from signal log."""
    sig_file = DATA_DIR / "signal_log.csv"
    if not sig_file.exists():
        return []
    try:
        lines = sig_file.read_text().strip().splitlines()
        if len(lines) < 2:
            return []
        headers = lines[0].split(",")
        recent  = []
        for line in reversed(lines[-6:-1]):
            row = dict(zip(headers, line.split(",")))
            recent.append(row)
        return recent
    except Exception:
        return []


def get_pending_items() -> list:
    """Extract TODO items from all agent files (comments marked TODO or FIXME)."""
    todos = []
    for f in AGENTS_DIR.glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "TODO" in line or "FIXME" in line or "⚠️  KNOWN" in line:
                clean = line.strip().lstrip("# ").strip()
                if len(clean) > 10:
                    todos.append(f"{f.name}:{i} — {clean[:80]}")
    return todos[:10]   # max 10


def get_config_values() -> dict:
    """Read key config values."""
    cfg_file = ROOT / "src" / "config.py"
    if not cfg_file.exists():
        return {}
    text = cfg_file.read_text()
    vals = {}
    for line in text.splitlines():
        for key in ["PROP_FIRM_ACTIVE", "PROP_FIRM_ACCOUNT_TYPE",
                    "PROP_FIRM_ACCOUNT_SIZE", "RISK_PER_TRADE_PCT",
                    "USE_KELLY_SIZING", "APEX_BRIDGE_ROUTE"]:
            if line.strip().startswith(key):
                try:
                    vals[key] = line.split("=")[1].split("#")[0].strip()
                except Exception:
                    pass
    return vals


def get_agent_files() -> list:
    """List all agent files with their purpose (first docstring line)."""
    agents = []
    for f in sorted(AGENTS_DIR.glob("*.py")):
        text = f.read_text()
        # Find first non-empty docstring line
        purpose = "—"
        in_doc  = False
        for line in text.splitlines()[3:20]:
            stripped = line.strip().strip('"').strip("'").strip("#").strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = True
                stripped = stripped.strip('"""').strip("'''").strip()
                if stripped and len(stripped) > 5:
                    purpose = stripped[:70]
                    break
                continue
            if in_doc and stripped and len(stripped) > 5:
                purpose = stripped[:70]
                break
        agents.append((f.name, purpose))
    return agents


# ══════════════════════════════════════════════════════════════
# HANDOFF DOCUMENT GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_handoff() -> str:
    """Generate the complete handoff document from live system state."""

    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    vault      = get_vault_strategies()
    ideas_cnt  = get_ideas_count()
    apex       = get_apex_state()
    brief      = get_weekly_brief()
    mm         = get_mm_stats()
    config     = get_config_values()
    signals    = get_recent_signals()
    todos      = get_pending_items()
    agents     = get_agent_files()

    # Sort vault by EV
    futures_syms = {"MES", "MNQ", "MYM"}
    def ev(s):
        wr   = s.get("win_rate", 50) / 100
        risk = 500 if s.get("symbol") in futures_syms else 10
        return (wr * risk * 2) - ((1-wr) * risk)

    vault_sorted = sorted(vault["strategies"], key=ev, reverse=True)
    futures_strats = [s for s in vault_sorted if s.get("symbol") in futures_syms]
    crypto_strats  = [s for s in vault_sorted if s.get("symbol") not in futures_syms]

    # Apex state summary
    apex_section = ""
    if apex.get("status") == "no_data":
        apex_section = "No Apex state data yet — run apex_bridge.py to sync."
    else:
        apex_section = f"""
- Account type:    {config.get('PROP_FIRM_ACCOUNT_TYPE', '—')}
- Account size:    ${int(config.get('PROP_FIRM_ACCOUNT_SIZE','50000').replace('_','') if 'PROP_FIRM_ACCOUNT_SIZE' in config else 50000):,}
- Risk per trade:  {config.get('RISK_PER_TRADE_PCT', '0.01')}
- Bridge route:    {config.get('APEX_BRIDGE_ROUTE', 'pickmytrade')}
- Kelly sizing:    {config.get('USE_KELLY_SIZING', 'False')}"""

    # Weekly brief summary
    regime = brief.get("regime", {})
    brief_section = ""
    if regime:
        brief_section = f"""
- Macro regime:    {regime.get('regime', '—')} (size adj: {regime.get('size_adjustment', 1.0)})
- Generated:       {brief.get('generated_at', '—')}
- Caution tokens:  {', '.join(brief.get('caution_tokens', [])) or 'none'}"""
    else:
        brief_section = "\nNo weekly briefing yet — run weekly_briefing.py"

    # Recent signals
    sig_section = ""
    if signals:
        sig_section = "\n### Recent Signals\n```\n"
        for s in signals[:5]:
            sig_section += (f"  {s.get('timestamp','')[:16]}  "
                            f"{s.get('strategy',''):<20} "
                            f"{s.get('symbol',''):<6} "
                            f"{s.get('direction',''):<6} "
                            f"{s.get('status','')}\n")
        sig_section += "```\n"
    else:
        sig_section = "\nNo signals logged yet.\n"

    # Agent list
    agent_table = ""
    for fname, purpose in agents:
        agent_table += f"| `{fname}`{' '*(35-len(fname))} | {purpose} |\n"

    # Vault futures table
    fut_table = "| Strategy | Symbol | TF | Sharpe | WR | EV/trade | Status |\n"
    fut_table += "|---|---|---|---|---|---|---|\n"
    for s in futures_strats:
        ev_val = ev(s)
        status = "❌ SKIP" if ev_val < 0 else "✅ ACTIVE"
        fut_table += (f"| {s['name']} | {s['symbol']} | {s.get('timeframe','?')} | "
                      f"{s.get('sharpe',0):.2f} | {s.get('win_rate',0):.1f}% | "
                      f"${ev_val:+.0f} | {status} |\n")

    # Vault crypto table (top 10)
    crypto_table = "| Strategy | Symbol | TF | Sharpe | WR |\n"
    crypto_table += "|---|---|---|---|---|\n"
    for s in crypto_strats[:10]:
        crypto_table += (f"| {s['name']} | {s['symbol']} | {s.get('timeframe','?')} | "
                         f"{s.get('sharpe',0):.2f} | {s.get('win_rate',0):.1f}% |\n")
    if len(crypto_strats) > 10:
        crypto_table += f"| *...{len(crypto_strats)-10} more in vault_index.json* | | | | |\n"

    # TODO items
    todo_section = ""
    if todos:
        for t in todos:
            todo_section += f"- `{t}`\n"
    else:
        todo_section = "- No TODOs found in agent files\n"

    doc = f"""# ALGOTEC SYSTEM HANDOFF
## Auto-generated: {now}
## For any LLM to continue development — pick up where left off

---

## ⚡ CURRENT SYSTEM STATE

| Metric | Value |
|---|---|
| Vault strategies | **{vault['count']}** ({len(futures_strats)} futures + {len(crypto_strats)} crypto) |
| Strategy ideas | **{ideas_cnt}** signals in ideas.txt |
| Ideas file lines | {(DATA_DIR / 'ideas.txt').stat().st_size // 1024}KB |
| MM log entries | {mm.get('fills', 0)} |
| Generated | {now} |

---

## WHO YOU ARE WORKING WITH

**Trader:** Abdullahi (Nairobi, Kenya — EAT = UTC+3)
**Background:** ICT manual trader, ~1 year experience, passed multiple prop firm challenges
**Setup:** Mac, Python 3.10.9, venv, `~/trading_system/`
**Goal:** $1M+/month through Apex prop firm accounts + Hyperliquid personal account

---

## WHAT ALGOTEC IS

Three-stream automated trading system:

1. **Apex prop firm** — vaulted futures strategies via PickMyTrade → Tradovate → Apex $50k accounts
   - YOU RISK $0 — it's their capital, you keep 90% of profits
   - 20 account max = ~$154k/month ceiling
   - Current config: {config.get('PROP_FIRM_ACCOUNT_TYPE', '—')} account

2. **Hyperliquid personal** — crypto vault + market making (v2 WebSocket) + pairs trading
   - Your capital, no rules, no ceiling, 24/7
   - Market maker: BTC lead signal, real-time WebSocket, SPX perp support

3. **SPX perpetual** — S&P 500 perp launched March 18 2026 on Hyperliquid via Trade[XYZ]
   - Same MES strategies, no Apex restrictions, trades weekends

---

## APEX CONFIG
{apex_section}

---

## MACRO CONTEXT (Weekly Brief)
{brief_section}

---

## THE VAULT — {vault['count']} VAULTED STRATEGIES

### Futures ({len(futures_strats)}) — Apex via PickMyTrade
{fut_table}
**⚠️ SupplyClimax has negative EV — always skip it.**

### Crypto ({len(crypto_strats)}) — Hyperliquid direct
{crypto_table}
**Full list:** `src/strategies/vault/vault_index.json`

**Execution rules:**
- Strategies sorted by EV descending — best executes first
- MAX_CONCURRENT_FUTURES = 1 (so highest EV always wins the slot)
- All 28 scanned in parallel (8 threads via ThreadPoolExecutor)

---

## ACTIVE AGENTS

| File | Purpose |
|---|---|
{agent_table}

---

## KEY FILES

```
trading_system/
├── src/
│   ├── agents/           # 35 Python trading agents
│   ├── strategies/vault/ # 28 vaulted strategies + vault_index.json
│   ├── data/
│   │   ├── ideas.txt         # {ideas_cnt} trade signals
│   │   ├── processed_ideas.json  # RBI cache — DELETE to re-run all
│   │   ├── apex_state.json
│   │   ├── weekly_brief.json
│   │   ├── pairs_state.json
│   │   └── mm_v2_log.csv
│   ├── exchanges/
│   │   ├── tradovate.py      # Apex futures
│   │   └── hyperliquid.py    # HL crypto
│   ├── models/
│   │   └── llm_router.py     # Multi-LLM router (Claude→DeepSeek→Groq→Gemini)
│   └── dashboard_live.html   # Dashboard UI
├── server.py                 # Flask server at :8080
├── .env                      # API keys — NEVER COMMIT
└── ALGOTEC_HANDOFF.md        # This file (auto-generated)
```

---

## EXECUTION FLOW

```
ideas.txt
  ↓ rbi_parallel.py (Claude/DeepSeek generates Python strategy code)
  ↓ Backtest on historical data
  ↓ Sharpe > threshold → vault_index.json

vault_forward_test.py (hourly)
  ↓ Parallel scan (8 threads) — all {vault['count']} strategies
  ↓ EV-sorted execution — best first
  ↓ apex_risk.py + risk_manager.py (Kelly sizing)
  ↓
  ├─ FUTURES → PickMyTrade → Tradovate → Apex accounts
  ├─ CRYPTO  → Hyperliquid direct
  └─ SPX     → Hyperliquid RWA (Trade[XYZ])

market_maker_v2.py (24/7)
  ↓ WebSocket orderbook (<100ms reaction)
  ↓ BTC lead signal (pre-skews altcoins)
  ↓ 5-factor skew predictor
  ↓ Limit orders on Hyperliquid

pairs_trading_agent.py (5-min quick + 60-min deep)
  ↓ Kalman filter hedge ratio (dynamic)
  ↓ Dynamic Z-score thresholds
  ↓ Long laggard + short leader simultaneously
  ↓ Hyperliquid only (never Apex)
```

---

## .ENV REQUIREMENTS

```dotenv
# Apex / PickMyTrade
PICKMYTRADE_TOKEN=<token>          # ⚠️ REGENERATE — was exposed in chat
PICKMYTRADE_ACCOUNT_IDS=APEX...    # comma-sep for multiple PAs

# Hyperliquid
HYPERLIQUID_API_PRIVATE_KEY=0x...  # from app.hyperliquid.xyz/API
HYPERLIQUID_ACCOUNT_ADDRESS=0x...  # main Exodus ETH public address

# LLMs (at least one for RBI)
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=...
GROQ_API_KEY=...
GEMINI_KEY=...
```

---

## LLM ROUTER

**Default order:** DeepSeek → Groq (4 models) → Gemini → OpenAI
**RBI order:** Claude → DeepSeek → Groq

**Groq model chain (rate limit resilience):**
1. llama-3.3-70b-versatile
2. llama-3.1-8b-instant
3. gemma2-9b-it
4. mixtral-8x7b-32768

**Known fixed bugs:**
- ✅ `next_p` fallback bug fixed (showed wrong next provider)
- ✅ Groq multi-model retry added
- ✅ Gemini switched to gemini-1.5-flash-latest

---

## APEX RULES (post March 2026 update)

**Eval:**
- Profit target: $3,000 (on $50k account)
- Trailing drawdown: $2,500 from peak
- Daily loss limit (EOD only): $1,000 — pauses NOT fails
- 30-day time limit
- Close by 4:59pm ET

**Performance Account (PA):**
- 50% consistency rule: no single day > 50% of total profit
- 5 qualifying days before payout ($250+/day)

**Allowed:** Scalping (1m/5m), news trading, overnight holds, multiple accounts
**Banned:** HFT, simultaneous long/short same instrument

---

## INCOME MODEL

| Account | Monthly | Annual |
|---|---|---|
| 1 Apex PA | $7,707 | $92,484 |
| 10 Apex PA | $77,070 | $924,840 |
| 20 Apex PA (cap) | $154,139 | $1,849,668 |
| HL at $1k capital | $55 | $660 |
| HL at $100k capital | $5,500 | $66,000 |

**2026 trajectory (20% reinvestment compounding):**
- Month 5: Apex cap at $100k/month (5 PA accounts)
- Month 8: Apex cap hit (20 accounts, $157k/month)
- Dec 2026: $114k/month ($526k total earned Apr-Dec)
- 2027 run rate: $1.37M/year

---

## RECENT ACTIVITY
{sig_section}

---

## MARKET MAKER STATUS

- Log entries: {mm.get('fills', 0)}
- Last token: {mm.get('symbol', '—')}
- Last PnL: ${mm.get('last_pnl', 0.0):+.2f}

---

## PENDING / TODO ITEMS
{todo_section}

---

## HOW TO START THE SYSTEM

```bash
cd trading_system && source venv/bin/activate

# Dashboard
python server.py &

# Main income (futures — Apex)
python src/agents/vault_forward_test.py --mode auto --market futures

# Crypto (Hyperliquid)
python src/agents/vault_forward_test.py --mode auto --market crypto

# Market maker (paper first)
python src/agents/market_maker_v2.py --paper

# Pairs trading
python src/agents/pairs_trading_agent.py --live

# Find new strategies
rm src/data/processed_ideas.json
python src/agents/rbi_parallel.py --market futures
```

---

## INSTALLS NEEDED (if fresh setup)

```bash
pip install ccxt websockets statsmodels
pip install hyperliquid-python-sdk eth-account
pip install anthropic openai requests pandas numpy flask
```

---

## QUESTIONS TO ASK ABDULLAHI FIRST

1. What error are you seeing? (paste full traceback)
2. Which agent are you running?
3. Is this Apex eval or PA account? (different rules)
4. Is this Hyperliquid or Tradovate?
5. What's in your .env? (run `grep -v KEY .env` to check without exposing keys)

---

*Auto-generated by `src/agents/handoff_generator.py` — {now}*
*System by Abdullahi + Claude (Anthropic). Dashboard: http://algotectrading:8080*
"""
    return doc


# ══════════════════════════════════════════════════════════════
# FILE CHANGE WATCHER
# ══════════════════════════════════════════════════════════════

def get_system_hash() -> str:
    """Hash key system files to detect changes."""
    files_to_watch = [
        VAULT_DIR / "vault_index.json",
        DATA_DIR  / "ideas.txt",
        DATA_DIR  / "apex_state.json",
        DATA_DIR  / "weekly_brief.json",
        DATA_DIR  / "mm_v2_log.csv",
        ROOT / "src" / "config.py",
    ]
    h = hashlib.md5()
    for f in files_to_watch:
        if f.exists():
            h.update(f.stat().st_mtime_ns.to_bytes(8, 'big'))
            h.update(f.name.encode())
    return h.hexdigest()


def write_handoff():
    """Generate and write the handoff document."""
    doc = generate_handoff()
    OUT_FILE.write_text(doc)
    size = len(doc.splitlines())
    print(f"  ✅ ALGOTEC_HANDOFF.md updated — {size} lines — "
          f"{datetime.now().strftime('%H:%M:%S')}")
    return doc


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="🌙 Algotec Handoff Generator")
    p.add_argument("--watch",    action="store_true",
                   help="Watch for changes and auto-regenerate")
    p.add_argument("--interval", type=int, default=300,
                   help="Watch interval in seconds (default 300 = 5min)")
    p.add_argument("--diff",     action="store_true",
                   help="Show summary of what changed since last generation")
    args = p.parse_args()

    print("🌙 Algotec Handoff Generator")

    if args.diff:
        if OUT_FILE.exists():
            old_lines = OUT_FILE.read_text().splitlines()
            new_doc   = generate_handoff()
            new_lines = new_doc.splitlines()
            added   = [l for l in new_lines if l not in old_lines]
            removed = [l for l in old_lines if l not in new_lines]
            print(f"\n  Changes vs last handoff:")
            print(f"  +{len(added)} lines added, -{len(removed)} lines removed")
            if added[:5]:
                print(f"\n  Sample additions:")
                for l in added[:5]:
                    if l.strip():
                        print(f"    + {l[:80]}")
        else:
            print("  No existing handoff — generating first version")
        write_handoff()

    elif args.watch:
        print(f"  Watching for changes every {args.interval}s...")
        print(f"  Output: {OUT_FILE}")
        last_hash = ""
        while True:
            try:
                current_hash = get_system_hash()
                if current_hash != last_hash:
                    write_handoff()
                    last_hash = current_hash
                else:
                    print(f"\r  ⏱  No changes — next check in {args.interval}s   ",
                          end="", flush=True)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n  ⏹  Stopped")
                break

    else:
        # Generate once
        write_handoff()
        print(f"  Saved to: {OUT_FILE}")
