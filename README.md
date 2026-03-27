# 🌙 AI Trading System
### Inspired by Moon Dev · Powered by DeepSeek · Trades on Hyperliquid & Coinbase

---

## What This Does

This system automatically:
1. **Reads** your trading ideas from a plain text file
2. **Researches** them using DeepSeek AI to define precise entry/exit rules
3. **Generates** backtesting Python code automatically
4. **Tests** each strategy across BTC, ETH, and SOL
5. **Saves** all results to CSV + JSON for you to review

---

## Project Structure

```
trading_system/
├── src/
│   ├── agents/
│   │   └── rbi_agent.py        ← Main RBI Backtester (run this!)
│   ├── models/
│   │   └── deepseek_model.py   ← DeepSeek AI client
│   ├── data/
│   │   ├── fetcher.py          ← OHLCV data from yfinance
│   │   ├── ideas.txt           ← YOUR trading ideas go here
│   │   └── rbi_results/        ← All backtest results saved here
│   └── config.py               ← All settings
├── dashboard.html               ← Open in browser to view results
├── requirements.txt
└── .env.example                 ← Copy to .env and add your keys
```

---

## Setup (5 steps)

### 1. Clone / download this project

### 2. Create a virtual environment
```bash
# Using conda (recommended):
conda create -n trading python=3.10.9
conda activate trading

# OR using venv:
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your DeepSeek API key
Copy `.env.example` to `.env` and fill it in:
```
DEEPSEEK_API_KEY=your_key_here
```
Get a free key at: https://platform.deepseek.com/

### 5. Add your trading ideas
Edit `src/data/ideas.txt` — one idea per line:
```
Buy when RSI drops below 30 and price is above 200 EMA
MACD crossover with high volume
Bollinger Band breakout
```

---

## Running the RBI Backtester

```bash
python src/agents/rbi_agent.py
```

That's it. The agent will:
- Pick up each idea from `ideas.txt`
- Use DeepSeek to research + generate a strategy
- Backtest it on BTC, ETH, SOL (Hyperliquid prices via yfinance)
- Save results to `src/data/rbi_results/backtest_stats.csv`

Processing time: ~60–90 seconds per idea.

---

## Viewing Results

**Option A — Dashboard (recommended)**
Open `dashboard.html` in your browser, click "Load CSV", and select
`src/data/rbi_results/backtest_stats.csv`

**Option B — Spreadsheet**
Open `src/data/rbi_results/backtest_stats.csv` in Excel / Google Sheets.

---

## Switching Exchange

In `src/config.py`:
```python
EXCHANGE = "hyperliquid"   # or "coinbase"
```

---

## Adding Live Trading (later)

Once you've found a strategy that backtests well (30+ days live before scaling):

1. Set `ACTIVE_AGENTS["trading_agent"] = True` in `src/config.py`
2. Add your exchange API keys to `.env`
3. Run `python src/agents/trading_agent.py`

> ⚠️ Always backtest thoroughly before using real money.
> Past performance does not guarantee future results.

---

## Security Reminders

- Never commit your `.env` file
- Never share your private keys
- Start with tiny position sizes
- This is educational — not financial advice

---

## Roadmap (next agents to add)

- [ ] Live Trading Agent (Hyperliquid + Coinbase)
- [ ] Risk Agent (enforces stop loss / daily loss limits)
- [ ] Whale Monitor (tracks large wallet movements)
- [ ] Liquidation Tracker (alerts on liquidation spikes)
- [ ] Sentiment Agent (Twitter/X sentiment scoring)
