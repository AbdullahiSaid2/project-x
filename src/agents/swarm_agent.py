# ============================================================
# 🌙 Swarm Trading Mode
#
# Instead of ONE AI deciding whether to trade, FIVE specialist
# agents each analyse a different dimension of the market and
# cast a weighted vote. The trade only executes if the swarm
# reaches a consensus above the confidence threshold.
#
# THE FIVE AGENTS:
#   1. 📈 Trend Agent      — reads price action & EMAs
#   2. 📊 Momentum Agent   — RSI, MACD, volume
#   3. 🐋 Whale Agent      — on-chain & OI data
#   4. 📰 Sentiment Agent  — news & social mood
#   5. ⚠️  Risk Agent      — checks portfolio exposure
#
# HOW TO RUN:
#   python src/agents/swarm_agent.py
#   python src/agents/swarm_agent.py --once
#   python src/agents/swarm_agent.py --symbol BTC
# ============================================================

import sys
import csv
import json
import time
import traceback
import concurrent.futures
import pandas as pd
from src.models.ta_wrapper import ta
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config              import (EXCHANGE, SLEEP_BETWEEN_RUNS_SEC,
                                      HYPERLIQUID_TOKENS, COINBASE_TOKENS,
                                      MAX_POSITION_SIZE_USD, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
from src.models.llm_router import model
from src.exchanges.router      import get_price, buy, sell, active_symbols
from src.agents.risk_agent     import risk
from src.data.fetcher          import get_ohlcv

# ── paths ────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
SWARM_LOG  = REPO_ROOT / "src" / "data" / "swarm_log.csv"
SWARM_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── swarm config ─────────────────────────────────────────────
SWARM_TRADE_SIZE_USD   = 50      # USD per trade
CONSENSUS_THRESHOLD    = 0.65    # need 65% weighted score to trade
MIN_AGENTS_AGREEING    = 3       # at least 3 of 5 must vote same direction
TIMEFRAME              = "1H"

# Agent weights (must sum to 1.0)
AGENT_WEIGHTS = {
    "trend":     0.25,
    "momentum":  0.25,
    "whale":     0.20,
    "sentiment": 0.15,
    "risk":      0.15,
}


# ── vote dataclass ────────────────────────────────────────────
@dataclass
class Vote:
    agent:      str
    signal:     str          # "BUY" | "SELL" | "HOLD"
    confidence: float        # 0.0 – 1.0
    reasoning:  str
    weight:     float = 0.0
    weighted_score: float = 0.0   # positive = buy, negative = sell


@dataclass
class SwarmDecision:
    symbol:          str
    timestamp:       str
    votes:           list = field(default_factory=list)
    consensus_score: float = 0.0   # -1.0 (strong sell) → +1.0 (strong buy)
    final_signal:    str = "HOLD"
    confidence:      float = 0.0
    agents_buy:      int = 0
    agents_sell:     int = 0
    agents_hold:     int = 0
    executed:        bool = False
    reasoning:       str = ""


# ════════════════════════════════════════════════════════════════
# SPECIALIST AGENTS
# Each returns a Vote after analysing one dimension of the market
# ════════════════════════════════════════════════════════════════

# ── 1. Trend Agent ────────────────────────────────────────────
TREND_PROMPT = """You are a trend-following trading specialist.
Analyse the price action and moving average data below.
Focus ONLY on trend direction — ignore everything else.

DATA:
{data}

Respond ONLY with valid JSON:
{{"signal":"BUY"|"SELL"|"HOLD","confidence":0.0-1.0,"reasoning":"one sentence"}}"""

def trend_agent(symbol: str, df: pd.DataFrame) -> Vote:
    close = df["Close"]
    ema9  = ta.ema(close, 9).iloc[-1]
    ema21 = ta.ema(close, 21).iloc[-1]
    ema50 = ta.ema(close, 50).iloc[-1]
    ema200= ta.ema(close, 200).iloc[-1]
    price = float(close.iloc[-1])

    # Price action: higher highs / lower lows over last 20 bars
    last20 = close.tail(20)
    highs  = df["High"].tail(20)
    lows   = df["Low"].tail(20)
    hh     = float(highs.max())
    ll     = float(lows.min())

    data = {
        "symbol": symbol, "price": price,
        "ema9": round(float(ema9),2), "ema21": round(float(ema21),2),
        "ema50": round(float(ema50),2), "ema200": round(float(ema200),2),
        "price_vs_ema200": "above" if price > float(ema200) else "below",
        "ema9_vs_ema21":   "above" if float(ema9) > float(ema21) else "below",
        "20bar_high": round(hh,2), "20bar_low": round(ll,2),
        "trend_structure": (
            "STRONG_UPTREND"   if float(ema9)>float(ema21)>float(ema50)>float(ema200)
            else "STRONG_DOWNTREND" if float(ema9)<float(ema21)<float(ema50)<float(ema200)
            else "MIXED"
        ),
    }
    raw = model.chat(
        system_prompt="You are a trend analyst. Return only valid JSON.",
        user_prompt=TREND_PROMPT.format(data=json.dumps(data, indent=2)),
    )
    raw  = raw.replace("```json","").replace("```","").strip()
    result = json.loads(raw)
    return Vote(
        agent="trend",
        signal=result.get("signal","HOLD"),
        confidence=float(result.get("confidence",0.5)),
        reasoning=result.get("reasoning",""),
        weight=AGENT_WEIGHTS["trend"],
    )


# ── 2. Momentum Agent ─────────────────────────────────────────
MOMENTUM_PROMPT = """You are a momentum trading specialist.
Analyse the technical indicators below. Focus on momentum signals ONLY.

DATA:
{data}

Respond ONLY with valid JSON:
{{"signal":"BUY"|"SELL"|"HOLD","confidence":0.0-1.0,"reasoning":"one sentence"}}"""

def momentum_agent(symbol: str, df: pd.DataFrame) -> Vote:
    close  = df["Close"]
    volume = df["Volume"]

    rsi    = float(ta.rsi(close, 14).iloc[-1])
    macd_df= ta.macd(close)
    macd   = float(macd_df.iloc[-1,0]) if macd_df is not None else 0
    macd_s = float(macd_df.iloc[-1,2]) if macd_df is not None else 0
    macd_h = float(macd_df.iloc[-1,1]) if macd_df is not None else 0

    bb     = ta.bbands(close, 20)
    bb_pct = float(bb.iloc[-1,3]) if bb is not None else 0.5   # %B

    vol_avg= float(volume.rolling(20).mean().iloc[-1])
    vol_now= float(volume.iloc[-1])
    vol_ratio = round(vol_now / max(vol_avg, 1), 2)

    # Stochastic
    stoch  = ta.stoch(df["High"], df["Low"], close)
    stoch_k= float(stoch.iloc[-1,0]) if stoch is not None else 50

    data = {
        "symbol": symbol,
        "rsi_14": round(rsi, 2),
        "rsi_zone": "oversold" if rsi<30 else "overbought" if rsi>70 else "neutral",
        "macd": round(macd,4), "macd_signal": round(macd_s,4),
        "macd_histogram": round(macd_h,4),
        "macd_crossover": "bullish" if macd>macd_s else "bearish",
        "bb_percent_b": round(bb_pct,3),
        "volume_ratio": vol_ratio,
        "stochastic_k": round(stoch_k,2),
        "momentum_bias": (
            "STRONG_BUY"  if rsi<35 and macd>macd_s and vol_ratio>1.2
            else "STRONG_SELL" if rsi>65 and macd<macd_s and vol_ratio>1.2
            else "NEUTRAL"
        ),
    }
    raw = model.chat(
        system_prompt="You are a momentum analyst. Return only valid JSON.",
        user_prompt=MOMENTUM_PROMPT.format(data=json.dumps(data, indent=2)),
    )
    raw    = raw.replace("```json","").replace("```","").strip()
    result = json.loads(raw)
    return Vote(
        agent="momentum",
        signal=result.get("signal","HOLD"),
        confidence=float(result.get("confidence",0.5)),
        reasoning=result.get("reasoning",""),
        weight=AGENT_WEIGHTS["momentum"],
    )


# ── 3. Whale Agent ────────────────────────────────────────────
WHALE_PROMPT = """You are an on-chain and derivatives market specialist.
Analyse the open interest, funding rate, and liquidation data below.
Give a signal based on smart money positioning ONLY.

DATA:
{data}

Respond ONLY with valid JSON:
{{"signal":"BUY"|"SELL"|"HOLD","confidence":0.0-1.0,"reasoning":"one sentence"}}"""

def whale_swarm_agent(symbol: str) -> Vote:
    try:
        import requests
        r    = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "metaAndAssetCtxs"},
            headers={"Content-Type": "application/json"},
            timeout=8,
        )
        data_raw = r.json()
        universe = data_raw[0].get("universe", [])
        ctxs     = data_raw[1]

        oi_usd = 0; funding = 0; mark_px = 0
        for i, asset in enumerate(universe):
            if asset.get("name") == symbol.replace("-USD",""):
                ctx     = ctxs[i]
                mark_px = float(ctx.get("markPx", 0))
                oi_usd  = float(ctx.get("openInterest", 0)) * mark_px
                funding = float(ctx.get("funding", 0))
                break

        data = {
            "symbol":       symbol,
            "oi_usd_millions": round(oi_usd / 1e6, 2),
            "funding_rate_8h": round(funding * 100, 5),
            "funding_bias":    "longs_paying" if funding > 0 else "shorts_paying",
            "mark_price":      mark_px,
            "signal_hint": (
                "BEARISH — overleveraged longs" if funding > 0.001
                else "BULLISH — shorts squeezable" if funding < -0.001
                else "NEUTRAL"
            ),
        }
        raw    = model.chat(
            system_prompt="You are an on-chain analyst. Return only valid JSON.",
            user_prompt=WHALE_PROMPT.format(data=json.dumps(data, indent=2)),
        )
        raw    = raw.replace("```json","").replace("```","").strip()
        result = json.loads(raw)
        return Vote(
            agent="whale",
            signal=result.get("signal","HOLD"),
            confidence=float(result.get("confidence",0.5)),
            reasoning=result.get("reasoning",""),
            weight=AGENT_WEIGHTS["whale"],
        )
    except Exception as e:
        return Vote(agent="whale", signal="HOLD", confidence=0.3,
                    reasoning=f"Data unavailable: {e}", weight=AGENT_WEIGHTS["whale"])


# ── 4. Sentiment Agent ────────────────────────────────────────
SENTIMENT_PROMPT = """You are a market sentiment specialist.
Read the news headlines and give a directional signal for {symbol}.

HEADLINES:
{headlines}

Respond ONLY with valid JSON:
{{"signal":"BUY"|"SELL"|"HOLD","confidence":0.0-1.0,"reasoning":"one sentence"}}"""

def sentiment_swarm_agent(symbol: str) -> Vote:
    try:
        import requests, xml.etree.ElementTree as ET
        feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
        ]
        headlines = []
        for url in feeds:
            try:
                r    = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=6)
                root = ET.fromstring(r.content)
                for item in root.findall(".//item")[:5]:
                    title = item.findtext("title","").strip()
                    if title:
                        headlines.append(title)
            except Exception:
                pass

        raw    = model.chat(
            system_prompt="You are a sentiment analyst. Return only valid JSON.",
            user_prompt=SENTIMENT_PROMPT.format(
                symbol=symbol,
                headlines="\n".join(headlines[:10]) or "No headlines available.",
            ),
        )
        raw    = raw.replace("```json","").replace("```","").strip()
        result = json.loads(raw)
        return Vote(
            agent="sentiment",
            signal=result.get("signal","HOLD"),
            confidence=float(result.get("confidence",0.4)),
            reasoning=result.get("reasoning",""),
            weight=AGENT_WEIGHTS["sentiment"],
        )
    except Exception as e:
        return Vote(agent="sentiment", signal="HOLD", confidence=0.3,
                    reasoning=f"Sentiment unavailable: {e}", weight=AGENT_WEIGHTS["sentiment"])


# ── 5. Risk Agent ─────────────────────────────────────────────
def risk_swarm_agent(symbol: str, price: float) -> Vote:
    """
    Not an AI call — pure rules-based risk assessment.
    Returns BUY/SELL/HOLD based on portfolio state.
    """
    try:
        allowed_buy, reason_buy   = risk.check_trade(symbol, SWARM_TRADE_SIZE_USD, "buy")
        allowed_sell, reason_sell = risk.check_trade(symbol, SWARM_TRADE_SIZE_USD, "sell")

        if allowed_buy and allowed_sell:
            # Portfolio has room — signal is permissive
            return Vote(
                agent="risk", signal="HOLD",
                confidence=0.6,
                reasoning="Portfolio within limits, no strong risk signal",
                weight=AGENT_WEIGHTS["risk"],
            )
        elif not allowed_buy and not allowed_sell:
            return Vote(
                agent="risk", signal="HOLD",
                confidence=0.9,
                reasoning=f"Risk limits prevent any trade: {reason_buy}",
                weight=AGENT_WEIGHTS["risk"],
            )
        elif not allowed_buy:
            return Vote(
                agent="risk", signal="SELL",
                confidence=0.8,
                reasoning=f"Buy blocked by risk: {reason_buy}",
                weight=AGENT_WEIGHTS["risk"],
            )
        else:
            return Vote(
                agent="risk", signal="BUY",
                confidence=0.7,
                reasoning="Sell blocked — only direction available is buy",
                weight=AGENT_WEIGHTS["risk"],
            )
    except Exception as e:
        return Vote(agent="risk", signal="HOLD", confidence=0.5,
                    reasoning=f"Risk check error: {e}", weight=AGENT_WEIGHTS["risk"])


# ════════════════════════════════════════════════════════════════
# SWARM CONSENSUS ENGINE
# ════════════════════════════════════════════════════════════════

def compute_consensus(votes: list[Vote]) -> SwarmDecision:
    """
    Aggregate all votes into a single weighted consensus decision.

    Scoring:
      BUY  vote → +confidence × weight
      SELL vote → -confidence × weight
      HOLD vote → 0

    Final score range: -1.0 (strong sell) to +1.0 (strong buy)
    """
    weighted_scores = []
    buy_count = sell_count = hold_count = 0

    for vote in votes:
        if vote.signal == "BUY":
            vote.weighted_score = vote.confidence * vote.weight
            buy_count += 1
        elif vote.signal == "SELL":
            vote.weighted_score = -(vote.confidence * vote.weight)
            sell_count += 1
        else:
            vote.weighted_score = 0
            hold_count += 1
        weighted_scores.append(vote.weighted_score)

    consensus = sum(weighted_scores)   # -1.0 to +1.0
    abs_cons  = abs(consensus)

    # Determine final signal
    if consensus >= CONSENSUS_THRESHOLD and buy_count >= MIN_AGENTS_AGREEING:
        signal = "BUY"
    elif consensus <= -CONSENSUS_THRESHOLD and sell_count >= MIN_AGENTS_AGREEING:
        signal = "SELL"
    else:
        signal = "HOLD"

    return SwarmDecision(
        symbol="",
        timestamp=datetime.now().isoformat(),
        votes=votes,
        consensus_score=round(consensus, 4),
        final_signal=signal,
        confidence=round(abs_cons, 4),
        agents_buy=buy_count,
        agents_sell=sell_count,
        agents_hold=hold_count,
        reasoning=(
            f"{buy_count} BUY / {sell_count} SELL / {hold_count} HOLD votes. "
            f"Weighted consensus: {consensus:+.3f}"
        ),
    )


def _print_vote(vote: Vote):
    icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
    icon  = icons.get(vote.signal, "⚪")
    bar   = "█" * int(vote.confidence * 10) + "░" * (10 - int(vote.confidence * 10))
    print(f"    {icon} [{vote.agent:10s}] {vote.signal:4s}  "
          f"conf:{vote.confidence:.0%}  [{bar}]  {vote.reasoning[:55]}")


def _print_decision(decision: SwarmDecision):
    score = decision.consensus_score
    bar_len = int(abs(score) * 20)
    if score > 0:
        bar = " " * 20 + "█" * bar_len
        col = "BUY  "
    else:
        bar = " " * (20 - bar_len) + "█" * bar_len + " " * 20
        col = "SELL "

    icons = {"BUY": "🚀", "SELL": "💀", "HOLD": "⏸️"}
    print(f"\n  {'─'*56}")
    print(f"  {icons.get(decision.final_signal,'⏸️')} SWARM VERDICT: {decision.final_signal}")
    print(f"     Score     : {score:+.3f}  (threshold ±{CONSENSUS_THRESHOLD})")
    print(f"     Votes     : 🟢{decision.agents_buy} BUY  "
          f"🔴{decision.agents_sell} SELL  ⚪{decision.agents_hold} HOLD")
    print(f"     Confidence: {decision.confidence:.0%}")
    print(f"  {'─'*56}\n")


# ── CSV logging ───────────────────────────────────────────────
def _log_decision(decision: SwarmDecision):
    row = {
        "timestamp":       decision.timestamp,
        "symbol":          decision.symbol,
        "final_signal":    decision.final_signal,
        "consensus_score": decision.consensus_score,
        "confidence":      decision.confidence,
        "agents_buy":      decision.agents_buy,
        "agents_sell":     decision.agents_sell,
        "agents_hold":     decision.agents_hold,
        "executed":        decision.executed,
        "reasoning":       decision.reasoning,
    }
    # Flatten individual votes
    for vote in decision.votes:
        row[f"vote_{vote.agent}"] = vote.signal
        row[f"conf_{vote.agent}"] = round(vote.confidence, 3)

    write_header = not SWARM_LOG.exists()
    with open(SWARM_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════════════
# SWARM AGENT MAIN CLASS
# ════════════════════════════════════════════════════════════════

class SwarmTradingAgent:

    def __init__(self):
        self.symbols = active_symbols()
        print("🐝 Swarm Trading Agent initialised")
        print(f"   Agents   : {list(AGENT_WEIGHTS.keys())}")
        print(f"   Threshold: {CONSENSUS_THRESHOLD:.0%} consensus")
        print(f"   Min agree: {MIN_AGENTS_AGREEING} of 5 agents")
        print(f"   Trade size: ${SWARM_TRADE_SIZE_USD}")
        print(f"   Symbols  : {self.symbols}\n")

    def analyse_symbol(self, symbol: str) -> SwarmDecision:
        print(f"\n  🔬 Running swarm analysis on {symbol}...")

        # Fetch OHLCV once — shared across trend & momentum agents
        try:
            df    = get_ohlcv(symbol.replace("-USD",""),
                              exchange=EXCHANGE, timeframe=TIMEFRAME, days_back=59)
            price = float(df["Close"].iloc[-1])
        except Exception as e:
            print(f"  ❌ Failed to fetch data for {symbol}: {e}")
            return None

        # Run all 5 specialist agents in parallel
        votes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(trend_agent,          symbol, df):    "trend",
                executor.submit(momentum_agent,       symbol, df):    "momentum",
                executor.submit(whale_swarm_agent,    symbol):        "whale",
                executor.submit(sentiment_swarm_agent,symbol):        "sentiment",
                executor.submit(risk_swarm_agent,     symbol, price): "risk",
            }
            for future in concurrent.futures.as_completed(futures):
                agent_name = futures[future]
                try:
                    vote = future.result(timeout=30)
                    votes.append(vote)
                except Exception as e:
                    print(f"    ⚠️  {agent_name} agent failed: {e}")
                    votes.append(Vote(
                        agent=agent_name, signal="HOLD",
                        confidence=0.3, reasoning=f"Error: {e}",
                        weight=AGENT_WEIGHTS.get(agent_name, 0.2),
                    ))

        # Print individual votes
        print(f"\n  🗳️  Votes for {symbol}:")
        for vote in sorted(votes, key=lambda v: v.agent):
            _print_vote(vote)

        # Compute consensus
        decision         = compute_consensus(votes)
        decision.symbol  = symbol
        _print_decision(decision)

        return decision

    def execute_decision(self, decision: SwarmDecision) -> bool:
        if decision.final_signal == "HOLD":
            print(f"  ⏸️  HOLD — no trade placed")
            return False

        direction = decision.final_signal.lower()   # "buy" or "sell"
        allowed, reason = risk.check_trade(
            decision.symbol, SWARM_TRADE_SIZE_USD, direction
        )
        if not allowed:
            print(f"  🚫 Risk agent blocked: {reason}")
            decision.executed = False
            return False

        try:
            price = get_price(decision.symbol)
            sl    = risk.stop_loss_price(price,   "long" if direction=="buy" else "short")
            tp    = risk.take_profit_price(price, "long" if direction=="buy" else "short")

            if direction == "buy":
                buy(decision.symbol, SWARM_TRADE_SIZE_USD)
            else:
                sell(decision.symbol, SWARM_TRADE_SIZE_USD)

            print(f"  ✅ Trade executed! Entry: ${price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f}")
            decision.executed = True
            return True

        except Exception as e:
            print(f"  ❌ Execution failed: {e}")
            traceback.print_exc()
            decision.executed = False
            return False

    def run_once(self):
        print(f"\n{'═'*60}")
        print(f"🐝 Swarm scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        risk.portfolio_summary()

        trades = 0
        for symbol in self.symbols:
            print(f"\n{'─'*60}")
            decision = self.analyse_symbol(symbol)
            if decision:
                placed = self.execute_decision(decision)
                if placed:
                    trades += 1
                _log_decision(decision)
            time.sleep(2)

        print(f"\n{'═'*60}")
        print(f"✅ Swarm scan complete. Trades placed: {trades}/{len(self.symbols)}")

    def run(self):
        print("🚀 Swarm Trading Agent running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.run_once()
                print(f"\n😴 Next swarm scan in {SLEEP_BETWEEN_RUNS_SEC}s...")
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)
        except KeyboardInterrupt:
            print("\n🛑 Swarm Agent stopped.")


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🐝 Swarm Trading Agent")
    parser.add_argument("--once",   action="store_true", help="Single scan then exit")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Analyse a single symbol only (e.g. BTC)")
    args = parser.parse_args()

    agent = SwarmTradingAgent()

    if args.symbol:
        # Single symbol mode
        sym      = args.symbol.upper()
        decision = agent.analyse_symbol(sym)
        if decision:
            agent.execute_decision(decision)
            _log_decision(decision)
    elif args.once:
        agent.run_once()
    else:
        agent.run()
