# ALGOTEC SYSTEM HANDOFF
## Auto-generated: 2026-04-04 14:08 UTC
## For any LLM to continue development — pick up where left off

---

## ⚡ CURRENT SYSTEM STATE

| Metric | Value |
|---|---|
| Vault strategies | **28** (5 futures + 23 crypto) |
| Strategy ideas | **233** signals in ideas.txt |
| Ideas file lines | 54KB |
| MM log entries | 0 |
| Generated | 2026-04-04 14:08 UTC |

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
   - Current config: "eval" account

2. **Hyperliquid personal** — crypto vault + market making (v2 WebSocket) + pairs trading
   - Your capital, no rules, no ceiling, 24/7
   - Market maker: BTC lead signal, real-time WebSocket, SPX perp support

3. **SPX perpetual** — S&P 500 perp launched March 18 2026 on Hyperliquid via Trade[XYZ]
   - Same MES strategies, no Apex restrictions, trades weekends

---

## APEX CONFIG
No Apex state data yet — run apex_bridge.py to sync.

---

## MACRO CONTEXT (Weekly Brief)

No weekly briefing yet — run weekly_briefing.py

---

## THE VAULT — 28 VAULTED STRATEGIES

### Futures (5) — Apex via PickMyTrade
| Strategy | Symbol | TF | Sharpe | WR | EV/trade | Status |
|---|---|---|---|---|---|---|
| ZScoreReversion | MES | 4H | 1.66 | 89.1% | $+836 | ✅ ACTIVE |
| HistogramFade | MES | 15m | 2.20 | 66.2% | $+493 | ✅ ACTIVE |
| RSIOverbought | MES | 15m | 1.61 | 62.9% | $+444 | ✅ ACTIVE |
| VolumeSpikeFade | MES | 15m | 1.62 | 61.7% | $+426 | ✅ ACTIVE |
| SupplyClimax | MES | 1H | 1.17 | 21.8% | $-173 | ❌ SKIP |

**⚠️ SupplyClimax has negative EV — always skip it.**

### Crypto (23) — Hyperliquid direct
| Strategy | Symbol | TF | Sharpe | WR |
|---|---|---|---|---|
| EarlyBreakoutLong | AVAX | 4H | 1.02 | 84.0% |
| RSI Reversal | ETH | 4H | 1.08 | 66.7% |
| GapSupportBollinger | FET | 4H | 1.06 | 58.9% |
| RSI Reversal | WIF | 15m | 1.15 | 55.3% |
| GapRetracementMidpoint | RNDR | 1H | 1.02 | 54.5% |
| HighRetraceShort | BNB | 4H | 1.03 | 50.0% |
| EMA Crossover | TAO | 4H | 1.09 | 50.0% |
| BollingerOversoldReversal | TAO | 4H | 1.41 | 45.5% |
| EMA_RSI_Reversion | PEPE | 15m | 1.44 | 44.2% |
| SAR Reversal | TAO | 4H | 1.07 | 44.0% |
| *...13 more in vault_index.json* | | | | |

**Full list:** `src/strategies/vault/vault_index.json`

**Execution rules:**
- Strategies sorted by EV descending — best executes first
- MAX_CONCURRENT_FUTURES = 1 (so highest EV always wins the slot)
- All 28 scanned in parallel (8 threads via ThreadPoolExecutor)

---

## ACTIVE AGENTS

| File | Purpose |
|---|---|
| `__init__.py`                         | — |
| `apex_bridge.py`                      | — |
| `apex_risk.py`                        | — |
| `chart_agent.py`                      | — |
| `copy_bot_agent.py`                   | — |
| `forward_test_logger.py`              | — |
| `funding_arb_agent.py`                | — |
| `handoff_generator.py`                | — |
| `hyperliquid_setup.py`                | — |
| `ict_backtester.py`                   | — |
| `ict_executor.py`                     | — |
| `ict_scanner.py`                      | — |
| `liquidation_agent.py`                | — |
| `listing_arb_agent.py`                | — |
| `market_maker_agent.py`               | — |
| `market_maker_v2.py`                  | — |
| `monte_carlo_agent.py`                | — |
| `pairs_trading_agent.py`              | — |
| `prop_firm_monitor.py`                | — |
| `psychology_guard.py`                 | — |
| `rbi_agent.py`                        | — |
| `rbi_parallel.py`                     | — |
| `regime_agent.py`                     | — |
| `research_agent.py`                   | — |
| `risk_agent.py`                       | — |
| `risk_manager.py`                     | — |
| `sentiment_agent.py`                  | — |
| `signal_notifier.py`                  | — |
| `swarm_agent.py`                      | — |
| `trading_agent.py`                    | — |
| `vault_forward_test.py`               | — |
| `vault_strategy.py`                   | — |
| `vwap_agent.py`                       | — |
| `websearch_agent.py`                  | — |
| `weekly_briefing.py`                  | — |
| `whale_agent.py`                      | — |


---

## KEY FILES

```
trading_system/
├── src/
│   ├── agents/           # 35 Python trading agents
│   ├── strategies/vault/ # 28 vaulted strategies + vault_index.json
│   ├── data/
│   │   ├── ideas.txt         # 233 trade signals
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
  ↓ Parallel scan (8 threads) — all 28 strategies
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

No signals logged yet.


---

## MARKET MAKER STATUS

- Log entries: 0
- Last token: —
- Last PnL: $+0.00

---

## PENDING / TODO ITEMS
- `handoff_generator.py:142 — """Extract TODO items from all agent files (comments marked TODO or FIXME)."""`
- `handoff_generator.py:146 — if "TODO" in line or "FIXME" in line or "⚠️  KNOWN" in line:`
- `handoff_generator.py:292 — todo_section = "- No TODOs found in agent files\n"`
- `handoff_generator.py:520 — PENDING / TODO ITEMS`


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

*Auto-generated by `src/agents/handoff_generator.py` — 2026-04-04 14:08 UTC*
*System by Abdullahi + Claude (Anthropic). Dashboard: http://algotectrading:8080*