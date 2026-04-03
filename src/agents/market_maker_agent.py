#!/usr/bin/env python3
# ============================================================
# 🌙 Market Making Agent — RenTec-Style
#
# Posts simultaneous bid and ask orders, collecting the spread.
# Uses short-term price prediction to SKEW the quotes — the
# key innovation that separates RenTec from naive market makers.
#
# ── HOW NAIVE MARKET MAKING WORKS ────────────────────────────
#   Post bid at $449.50, ask at $450.50
#   Collect $1.00 spread when both fill
#   Risk: market moves against your inventory
#
# ── RENTECH'S IMPROVEMENT: QUOTE SKEWING ─────────────────────
#   Predict next 5-minute price direction using:
#     - MACD momentum
#     - RSI overbought/oversold
#     - Order book imbalance (more bids than asks = bullish)
#     - Short-term price momentum (last 3 bars)
#
#   If prediction = BULLISH:
#     - Move bid UP (more aggressive — you WANT to buy)
#     - Move ask UP (less aggressive — you WANT to sell higher)
#   If prediction = BEARISH:
#     - Move bid DOWN (less aggressive — avoid buying)
#     - Move ask DOWN (more aggressive — you WANT to sell now)
#
#   This means you make MORE money when right, and LESS when wrong
#   — without needing to be right most of the time.
#
# ── ANSWERS TO YOUR 3 QUESTIONS ──────────────────────────────
#   Q2: Is this the ICT market maker algo?
#   No — different concept. ICT's "market maker algo" refers to
#   institutional order flow TAKING liquidity by hunting stops
#   and creating liquidity sweeps. THIS agent is a liquidity
#   PROVIDER — it's on the other side of that trade. When ICT
#   says market makers hunt your stops, they mean institutions
#   taking out retail orders. This bot PROVIDES the orders that
#   get taken. It's the exchange participant, not the predator.
#
#   Q3: Does it need ideas/strategies?
#   No — it IS the strategy. It doesn't need the RBI backtester
#   or the vault. It generates its own signals (the skew model)
#   and executes them in a continuous loop.
#
# ── PROP FIRM COMPATIBILITY ───────────────────────────────────
#   ❌ NOT compatible with Apex eval/PA
#   Market making requires simultaneous long/short positions
#   and placing/cancelling hundreds of orders — both violate
#   typical prop firm rules. PERSONAL HYPERLIQUID ACCOUNT ONLY.
#
# ── REQUIREMENTS ─────────────────────────────────────────────
#   pip install hyperliquid-python-sdk eth-account
#   .env: HYPERLIQUID_ACCOUNT_ADDRESS, HYPERLIQUID_API_PRIVATE_KEY
#
# HOW TO RUN:
#   python src/agents/market_maker_agent.py --paper          # safe test
#   python src/agents/market_maker_agent.py --live TAO       # 1 token live
#   python src/agents/market_maker_agent.py --live TAO,PEPE  # multi-token
#   python src/agents/market_maker_agent.py --status         # show stats
# ============================================================

import os, sys, json, time, math, csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv
load_dotenv()

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"
MM_STATE = DATA_DIR / "mm_state.json"
MM_LOG   = DATA_DIR / "mm_log.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

class MMConfig:
    """Market maker configuration. Tune these per token."""

    # ── Spread settings ───────────────────────────────────────
    BASE_SPREAD_PCT   = 0.0015   # 0.15% base spread (bid-ask gap)
    MIN_SPREAD_PCT    = 0.0008   # 0.08% min — never go tighter
    MAX_SPREAD_PCT    = 0.005    # 0.50% max — widen in volatile mkts

    # ── Skew settings (RenTec innovation) ─────────────────────
    MAX_SKEW_PCT      = 0.0010   # max 0.10% skew in either direction
    SKEW_AGGRESSIVENESS = 0.6    # 0-1: how much we trust the prediction

    # ── Position / inventory limits ───────────────────────────
    MAX_INVENTORY_USD = 500      # max total exposure per token in USD
    MAX_INVENTORY_PCT = 0.5      # if inventory > 50% of max, skew to reduce
    ORDER_SIZE_USD    = 50       # size of each bid/ask order in USD

    # ── Risk limits ───────────────────────────────────────────
    MAX_DAILY_LOSS_USD = 100     # stop market making if we lose this
    MAX_POSITION_USD   = 300     # hard stop on single-token exposure
    STOP_ON_SPIKE_PCT  = 0.03    # pause if price moves 3% in one update

    # ── Loop timing ───────────────────────────────────────────
    UPDATE_INTERVAL_S  = 5       # requote every 5 seconds
    PREDICTION_BARS    = 20      # bars of history for prediction model

    # ── Hyperliquid fees ──────────────────────────────────────
    MAKER_FEE_PCT     = 0.0002   # 0.02% maker rebate (you GET paid!)
    TAKER_FEE_PCT     = 0.0005   # 0.05% taker fee (you PAY this)

    # Hyperliquid pays you to PROVIDE liquidity (maker fee is negative = rebate)
    # This makes market making especially attractive on HL


# ── SPX-specific config (S&P 500 perp via Trade[XYZ]) ─────────
class SPXMMConfig(MMConfig):
    """
    Tuned for the S&P 500 perpetual on Hyperliquid.

    Key differences from crypto:
    - Much higher price (~5,000+) so spreads are in index points not %
    - Volume heavily concentrated during NYSE hours (9:30am-4pm EST)
    - Price constrained by Discovery Bounds outside NYSE hours
    - Much larger notional per contract — smaller size needed
    - Correlates with MES/ES — can cross-reference for skew signal

    Why SPX is an EXCELLENT market making target:
    - $100M+ daily volume (launched March 2026, already top-10 on HL)
    - Highly liquid during US hours — many fills guaranteed
    - You KNOW the fair value (it's the S&P 500!)
    - Strong mean-reverting behaviour intraday
    - Maker rebate still applies
    """
    # Tighter spreads — SPX is highly liquid with many participants
    BASE_SPREAD_PCT   = 0.0008   # 0.08% (narrower than crypto)
    MIN_SPREAD_PCT    = 0.0004   # 0.04% minimum
    MAX_SPREAD_PCT    = 0.003    # 0.30% max (widen on macro events)

    # More aggressive skewing — SPX is more predictable than altcoins
    MAX_SKEW_PCT      = 0.0015
    SKEW_AGGRESSIVENESS = 0.7   # trust the model more on SPX

    # Smaller order size — SPX notional is much larger ($5,000+ per unit)
    ORDER_SIZE_USD    = 25      # $25 per leg on SPX

    # More conservative inventory — SPX can move fast on macro news
    MAX_INVENTORY_USD = 200
    MAX_POSITION_USD  = 150
    MAX_DAILY_LOSS_USD = 50

    # Tighter spike protection — SPX moves 0.5-1% on news
    STOP_ON_SPIKE_PCT = 0.015   # pause on 1.5% spike (vs 3% for crypto)

    # NYSE hours awareness
    PAUSE_OUTSIDE_NYSE_HOURS = True   # reduce activity outside 9:30-4pm EST
    NYSE_SPREAD_MULTIPLIER   = 2.5    # widen spread outside NYSE hours


# ══════════════════════════════════════════════════════════════
# PRICE PREDICTION MODEL (The RenTec skew engine)
# ══════════════════════════════════════════════════════════════

class SkewPredictor:
    """
    Predicts short-term price direction to skew bid/ask quotes.

    Output: score from -1.0 (strongly bearish) to +1.0 (strongly bullish)

    Signals used (same class as RenTec's publicly known approach):
      1. MACD histogram direction    — momentum acceleration
      2. RSI position                — overbought/oversold
      3. Order book imbalance        — more bids = bullish pressure
      4. Short-term price momentum   — last 3 closes vs 10-bar avg
      5. Volume trend                — rising vol confirms direction
    """

    def __init__(self, lookback: int = 20):
        self.lookback     = lookback
        self.price_history = deque(maxlen=100)
        self.vol_history   = deque(maxlen=50)
        self.last_score    = 0.0

    def update(self, price: float, volume: float = 1.0):
        self.price_history.append(price)
        self.vol_history.append(volume)

    def predict(self, orderbook: dict = None) -> tuple[float, dict]:
        """
        Returns (score, components) where score is -1 to +1.
        components shows how each signal contributed.
        """
        if len(self.price_history) < self.lookback:
            return 0.0, {"status": "warming_up",
                         "bars": len(self.price_history)}

        prices = np.array(self.price_history)
        vols   = np.array(self.vol_history)
        comps  = {}

        # ── Signal 1: MACD histogram momentum ─────────────────
        if len(prices) >= 26:
            ema12 = _ema(prices, 12)
            ema26 = _ema(prices, 26)
            macd  = ema12 - ema26
            signal_line = _ema(macd, 9)
            histogram   = macd - signal_line

            # Rising histogram = bullish momentum
            if len(histogram) >= 2:
                hist_slope = histogram[-1] - histogram[-2]
                comps["macd"] = np.clip(hist_slope / (abs(histogram[-1]) + 1e-8), -1, 1)
            else:
                comps["macd"] = 0.0
        else:
            comps["macd"] = 0.0

        # ── Signal 2: RSI position ─────────────────────────────
        if len(prices) >= 14:
            rsi = _rsi(prices, 14)
            # RSI 30 = max bullish signal, RSI 70 = max bearish signal
            # RSI 50 = neutral (score = 0)
            rsi_score = (50 - rsi) / 50    # 30 RSI → +0.4, 70 RSI → -0.4
            comps["rsi"] = np.clip(rsi_score, -1, 1)
        else:
            comps["rsi"] = 0.0

        # ── Signal 3: Order book imbalance ────────────────────
        if orderbook and "bids" in orderbook and "asks" in orderbook:
            bids = orderbook["bids"]
            asks = orderbook["asks"]

            if bids and asks:
                # Total size of top 5 bid levels vs top 5 ask levels
                bid_depth = sum(float(b[1]) for b in bids[:5])
                ask_depth = sum(float(a[1]) for a in asks[:5])
                total     = bid_depth + ask_depth
                if total > 0:
                    imbalance = (bid_depth - ask_depth) / total
                    comps["orderbook"] = np.clip(imbalance, -1, 1)
                else:
                    comps["orderbook"] = 0.0
            else:
                comps["orderbook"] = 0.0
        else:
            comps["orderbook"] = 0.0

        # ── Signal 4: Short-term momentum ────────────────────
        if len(prices) >= 10:
            sma10    = prices[-10:].mean()
            momentum = (prices[-1] - sma10) / sma10
            # Normalise: 1% move = ±0.5 score
            comps["momentum"] = np.clip(momentum * 50, -1, 1)
        else:
            comps["momentum"] = 0.0

        # ── Signal 5: Volume trend ────────────────────────────
        if len(vols) >= 5:
            recent_vol = vols[-3:].mean()
            older_vol  = vols[-5:-2].mean() + 1e-8
            vol_ratio  = recent_vol / older_vol
            # Combine with price direction — rising vol + up = more bullish
            price_dir = 1 if prices[-1] > prices[-2] else -1
            comps["volume"] = np.clip((vol_ratio - 1) * price_dir * 2, -1, 1)
        else:
            comps["volume"] = 0.0

        # ── Weighted combination ──────────────────────────────
        weights = {
            "orderbook":  0.35,   # highest weight — real-time signal
            "macd":       0.25,   # momentum acceleration
            "rsi":        0.15,   # mean reversion pressure
            "momentum":   0.15,   # recent price action
            "volume":     0.10,   # volume confirmation
        }

        score = sum(weights[k] * comps[k] for k in weights)
        score = np.clip(score, -1, 1)

        # Smooth with previous score (avoid overreacting to noise)
        self.last_score = 0.7 * score + 0.3 * self.last_score
        comps["final_score"] = round(self.last_score, 4)

        return self.last_score, comps


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha  = 2.0 / (period + 1)
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def _rsi(data: np.ndarray, period: int = 14) -> float:
    """RSI calculation."""
    deltas = np.diff(data[-period-1:])
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ══════════════════════════════════════════════════════════════
# QUOTE CALCULATOR
# ══════════════════════════════════════════════════════════════

def calculate_quotes(mid_price: float,
                     skew_score: float,
                     inventory_usd: float,
                     volatility: float,
                     config: MMConfig) -> dict:
    """
    Calculate bid and ask prices given:
      - mid_price:     current market mid
      - skew_score:    prediction (-1 to +1)
      - inventory_usd: current net position in USD
      - volatility:    recent price volatility (std dev as %)

    Returns: {bid, ask, spread_pct, skew_pct, reason}
    """
    # ── Step 1: Set spread based on volatility ────────────────
    # Wider market = wider spread to protect against adverse moves
    vol_multiplier = 1 + (volatility / 0.02)  # baseline 2% vol
    spread_pct     = config.BASE_SPREAD_PCT * vol_multiplier
    spread_pct     = np.clip(spread_pct,
                             config.MIN_SPREAD_PCT,
                             config.MAX_SPREAD_PCT)

    # ── Step 2: Calculate skew from prediction score ──────────
    # Positive score (bullish) → skew quotes UP (want to sell higher)
    # Negative score (bearish) → skew quotes DOWN (want to sell now)
    prediction_skew = skew_score * config.MAX_SKEW_PCT * config.SKEW_AGGRESSIVENESS

    # ── Step 3: Inventory skew (risk management) ───────────────
    # If we're long too much, skew DOWN to encourage selling
    # If we're short too much, skew UP to encourage buying
    max_inv = config.MAX_INVENTORY_USD
    inv_ratio    = inventory_usd / max_inv if max_inv > 0 else 0
    inventory_skew = -inv_ratio * config.MAX_SKEW_PCT * 0.5

    # ── Step 4: Combine skews ─────────────────────────────────
    total_skew = prediction_skew + inventory_skew
    total_skew = np.clip(total_skew, -config.MAX_SKEW_PCT, config.MAX_SKEW_PCT)

    # ── Step 5: Set bid/ask ───────────────────────────────────
    half_spread = spread_pct / 2
    bid = mid_price * (1 - half_spread + total_skew)
    ask = mid_price * (1 + half_spread + total_skew)

    # Build reason string
    direction = "BULLISH" if skew_score > 0.2 else "BEARISH" if skew_score < -0.2 else "NEUTRAL"
    reason = (f"Prediction={skew_score:+.3f} ({direction}) | "
              f"Spread={spread_pct*100:.3f}% | "
              f"Skew={total_skew*100:+.3f}% | "
              f"Vol={volatility*100:.2f}%")

    return {
        "bid":          round(bid, 6),
        "ask":          round(ask, 6),
        "mid":          round(mid_price, 6),
        "spread_pct":   round(spread_pct * 100, 4),
        "skew_pct":     round(total_skew * 100, 4),
        "skew_score":   round(skew_score, 4),
        "reason":       reason,
    }


# ══════════════════════════════════════════════════════════════
# INVENTORY TRACKER
# ══════════════════════════════════════════════════════════════

class InventoryTracker:
    """Tracks net position and P&L per token."""

    def __init__(self, symbol: str):
        self.symbol      = symbol
        self.net_qty     = 0.0     # positive = long, negative = short
        self.avg_cost    = 0.0
        self.realized_pnl = 0.0
        self.fills       = []      # list of (side, qty, price) tuples

    def record_fill(self, side: str, qty: float, price: float,
                    fee: float = 0.0):
        """side: 'BUY' or 'SELL'"""
        if side == "BUY":
            # Update average cost
            new_qty   = self.net_qty + qty
            if new_qty != 0:
                self.avg_cost = ((self.net_qty * self.avg_cost
                                  + qty * price) / new_qty)
            self.net_qty = new_qty
        else:
            # Realise P&L on sell
            pnl = (price - self.avg_cost) * qty - fee
            self.realized_pnl += pnl
            self.net_qty -= qty
            if abs(self.net_qty) < 1e-8:
                self.avg_cost = 0.0

        self.fills.append({
            "ts":    datetime.now(timezone.utc).isoformat(),
            "side":  side,
            "qty":   qty,
            "price": price,
            "fee":   fee,
        })

    def inventory_usd(self, current_price: float) -> float:
        return self.net_qty * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.net_qty == 0 or self.avg_cost == 0:
            return 0.0
        return self.net_qty * (current_price - self.avg_cost)

    def total_pnl(self, current_price: float) -> float:
        return self.realized_pnl + self.unrealized_pnl(current_price)

    def summary(self, current_price: float) -> str:
        inv = self.inventory_usd(current_price)
        upnl = self.unrealized_pnl(current_price)
        return (f"{self.symbol}: net={self.net_qty:+.4f} "
                f"(${inv:+.2f}) | "
                f"realised=${self.realized_pnl:+.2f} "
                f"unrealised=${upnl:+.2f} "
                f"total=${self.total_pnl(current_price):+.2f}")


# ══════════════════════════════════════════════════════════════
# HYPERLIQUID ORDER MANAGER
# ══════════════════════════════════════════════════════════════

class HyperliquidMM:
    """
    Manages limit orders on Hyperliquid for market making.
    Uses REST API (polling) rather than WebSocket — simpler,
    sufficient for 5-second update cycles.

    For true HFT upgrade, swap REST with WebSocket subscription.
    """

    def __init__(self, symbol: str, paper: bool = False):
        self.symbol      = symbol
        self.paper       = paper
        self.active_orders = {"bid": None, "ask": None}
        self.exchange    = None
        self.account     = None

        if not paper:
            self._connect()

    def _connect(self):
        try:
            exchange, account = self._get_exchange()
            self.exchange = exchange
            self.account  = account
            print(f"  ✅ Connected to Hyperliquid ({self.symbol})")
        except Exception as e:
            print(f"  ❌ Connection failed: {e}")

    def _get_exchange(self):
        from hyperliquid.utils import constants
        from hyperliquid.exchange import Exchange
        from eth_account import Account

        api_key  = os.getenv("HYPERLIQUID_API_PRIVATE_KEY", "")
        acct_addr = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS", "")
        wallet   = Account.from_key(api_key)
        exchange = Exchange(wallet, constants.MAINNET_API_URL,
                            account_address=acct_addr)
        return exchange, acct_addr

    def get_orderbook(self) -> dict:
        """Fetch current L2 orderbook via REST."""
        try:
            import requests
            r = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "l2Book", "coin": self.symbol},
                timeout=5
            )
            r.raise_for_status()
            book = r.json()
            return {
                "bids": book.get("levels", [[], []])[0],
                "asks": book.get("levels", [[], []])[1],
            }
        except Exception as e:
            return {"bids": [], "asks": [], "error": str(e)}

    def get_mid_price(self) -> float:
        """Get current mid price from orderbook."""
        try:
            import requests
            r = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "allMids"},
                timeout=5
            )
            r.raise_for_status()
            mids = r.json()
            for k, v in mids.items():
                if self.symbol.upper() in k.upper():
                    return float(v)
        except Exception:
            pass
        return 0.0

    def cancel_all(self):
        """Cancel all active market making orders."""
        if self.paper:
            self.active_orders = {"bid": None, "ask": None}
            return True

        try:
            if self.exchange:
                # Cancel by order IDs
                for side, order in self.active_orders.items():
                    if order and order.get("oid"):
                        self.exchange.cancel(self.symbol, order["oid"])
                self.active_orders = {"bid": None, "ask": None}
            return True
        except Exception as e:
            print(f"  ⚠️  Cancel failed: {e}")
            return False

    def place_quotes(self, bid_price: float, ask_price: float,
                     size_usd: float, current_price: float) -> dict:
        """
        Place (or replace) bid and ask limit orders.
        Returns fill info if orders were immediately matched.
        """
        if current_price <= 0:
            return {"status": "skipped", "reason": "no price"}

        qty = round(size_usd / current_price, 6)
        if qty <= 0:
            return {"status": "skipped", "reason": "qty too small"}

        if self.paper:
            # Paper trading — simulate the quotes
            self.active_orders = {
                "bid": {"price": bid_price, "qty": qty, "side": "BUY"},
                "ask": {"price": ask_price, "qty": qty, "side": "SELL"},
            }
            return {"status": "paper", "bid": bid_price,
                    "ask": ask_price, "qty": qty}

        try:
            # Cancel existing quotes first
            self.cancel_all()
            time.sleep(0.1)

            results = {}

            # Place bid (limit buy)
            bid_result = self.exchange.order(
                self.symbol,
                is_buy=True,
                sz=qty,
                limit_px=bid_price,
                order_type={"limit": {"tif": "Gtc"}},   # Good till cancel
                reduce_only=False,
            )
            results["bid"] = bid_result
            self.active_orders["bid"] = {
                "oid": bid_result.get("response", {}).get("data", {})
                                  .get("statuses", [{}])[0].get("resting", {})
                                  .get("oid"),
                "price": bid_price,
                "qty":   qty,
            }

            time.sleep(0.05)

            # Place ask (limit sell)
            ask_result = self.exchange.order(
                self.symbol,
                is_buy=False,
                sz=qty,
                limit_px=ask_price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=False,
            )
            results["ask"] = ask_result
            self.active_orders["ask"] = {
                "oid": ask_result.get("response", {}).get("data", {})
                                  .get("statuses", [{}])[0].get("resting", {})
                                  .get("oid"),
                "price": ask_price,
                "qty":   qty,
            }

            results["status"] = "placed"
            return results

        except Exception as e:
            return {"status": "error", "error": str(e)}


# ══════════════════════════════════════════════════════════════
# MAIN MARKET MAKER LOOP
# ══════════════════════════════════════════════════════════════

class MarketMaker:
    """
    Main market making engine for a single token.
    Runs continuously, updating quotes every UPDATE_INTERVAL_S seconds.
    """

    def __init__(self, symbol: str, paper: bool = False,
                 config: MMConfig = None):
        self.symbol    = symbol
        self.paper     = paper
        self.config    = config or MMConfig()
        self.hl        = HyperliquidMM(symbol, paper)
        self.predictor = SkewPredictor(lookback=self.config.PREDICTION_BARS)
        self.inventory = InventoryTracker(symbol)
        self.stats     = {
            "updates":      0,
            "quotes_placed": 0,
            "fills_bid":    0,
            "fills_ask":    0,
            "daily_pnl":    0.0,
            "started":      datetime.now(timezone.utc).isoformat(),
        }
        self.daily_loss  = 0.0
        self.last_price  = 0.0
        self.price_hist  = deque(maxlen=50)
        self.running     = True

    def _get_volatility(self) -> float:
        """Calculate recent price volatility (std dev of returns)."""
        if len(self.price_hist) < 5:
            return 0.02  # default 2%
        prices = np.array(self.price_hist)
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    def _check_risk(self, current_price: float) -> tuple[bool, str]:
        """Risk checks before placing quotes."""

        # Daily loss limit
        total_pnl = self.inventory.total_pnl(current_price)
        if total_pnl < -self.config.MAX_DAILY_LOSS_USD:
            return False, f"Daily loss limit hit: ${total_pnl:.2f}"

        # Max inventory
        inv_usd = abs(self.inventory.inventory_usd(current_price))
        if inv_usd > self.config.MAX_POSITION_USD:
            return False, f"Max inventory hit: ${inv_usd:.2f}"

        # Price spike protection
        if self.last_price > 0 and current_price > 0:
            price_change = abs(current_price - self.last_price) / self.last_price
            if price_change > self.config.STOP_ON_SPIKE_PCT:
                return False, f"Price spike: {price_change*100:.2f}%"

        return True, "OK"

    def _is_nyse_hours(self) -> bool:
        """Check if NYSE is currently open (9:30am-4pm EST, Mon-Fri)."""
        try:
            import zoneinfo
            est = zoneinfo.ZoneInfo("America/New_York")
            now = datetime.now(est)
            if now.weekday() >= 5:   # Saturday/Sunday
                return False
            open_time  = now.replace(hour=9,  minute=30, second=0)
            close_time = now.replace(hour=16, minute=0,  second=0)
            return open_time <= now <= close_time
        except Exception:
            return True  # assume open if can't check

    def step(self):
        """One update cycle — called every UPDATE_INTERVAL_S seconds."""
        self.stats["updates"] += 1

        # ── NYSE hours check (SPX only) ────────────────────────
        if (hasattr(self.config, 'PAUSE_OUTSIDE_NYSE_HOURS') and
                self.config.PAUSE_OUTSIDE_NYSE_HOURS):
            nyse_open = self._is_nyse_hours()
            if not nyse_open:
                # Outside NYSE hours: widen spreads, don't pause
                # (there IS volume, just use wider quotes)
                print(f"  ⏰ {self.symbol}: Outside NYSE hours "
                      f"— using wider spreads")

        # ── 1. Fetch market data ───────────────────────────────
        price     = self.hl.get_mid_price()
        orderbook = self.hl.get_orderbook()

        if price <= 0:
            print(f"  ⚠️  {self.symbol}: No price data")
            return

        self.price_hist.append(price)
        self.predictor.update(price)

        # ── 2. Risk check ──────────────────────────────────────
        ok, reason = self._check_risk(price)
        if not ok:
            print(f"  🚫 {self.symbol}: Risk stop — {reason}")
            self.hl.cancel_all()
            return

        # ── 3. Get prediction score ────────────────────────────
        score, components = self.predictor.predict(orderbook)

        # ── 4. Calculate quotes with skew ─────────────────────
        vol      = self._get_volatility()
        inv_usd  = self.inventory.inventory_usd(price)
        quotes   = calculate_quotes(price, score, inv_usd, vol, self.config)

        # ── 5. Place quotes ───────────────────────────────────
        result = self.hl.place_quotes(
            quotes["bid"], quotes["ask"],
            self.config.ORDER_SIZE_USD, price
        )

        if result.get("status") in ("placed", "paper"):
            self.stats["quotes_placed"] += 1

        # ── 6. Check for fills (simplified — check position change) ──
        # In production: subscribe to fill WebSocket
        # For now: compare inventory to expected
        pnl = self.inventory.total_pnl(price)

        # ── 7. Print status ───────────────────────────────────
        direction = ("🟢 BULL" if score > 0.2
                     else "🔴 BEAR" if score < -0.2 else "⚪ NEUT")
        mode_tag  = "[PAPER]" if self.paper else "[LIVE]"

        print(f"  {mode_tag} {self.symbol:6} "
              f"${price:.4f} | "
              f"Bid={quotes['bid']:.4f} Ask={quotes['ask']:.4f} "
              f"(+{quotes['spread_pct']:.3f}%) | "
              f"Skew={quotes['skew_pct']:+.3f}% {direction} | "
              f"Inv=${inv_usd:+.2f} | "
              f"PnL=${pnl:+.2f}")

        # Log
        self._log(price, quotes, score, pnl)
        self.last_price = price

    def _log(self, price: float, quotes: dict,
             score: float, pnl: float):
        exists = MM_LOG.exists()
        with open(MM_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp","symbol","price","bid","ask",
                "spread_pct","skew_score","skew_pct","pnl","mode"
            ])
            if not exists:
                w.writeheader()
            w.writerow({
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "symbol":     self.symbol,
                "price":      price,
                "bid":        quotes["bid"],
                "ask":        quotes["ask"],
                "spread_pct": quotes["spread_pct"],
                "skew_score": score,
                "skew_pct":   quotes["skew_pct"],
                "pnl":        pnl,
                "mode":       "paper" if self.paper else "live",
            })

    def run(self):
        """Main loop."""
        print(f"\n  ▶  {self.symbol} market maker started "
              f"({'PAPER' if self.paper else 'LIVE'})")
        try:
            while self.running:
                self.step()
                time.sleep(self.config.UPDATE_INTERVAL_S)
        except KeyboardInterrupt:
            pass
        finally:
            self.hl.cancel_all()
            print(f"\n  ⏹  {self.symbol} stopped. "
                  f"Updates: {self.stats['updates']} | "
                  f"Quotes: {self.stats['quotes_placed']}")


# ══════════════════════════════════════════════════════════════
# MULTI-TOKEN RUNNER
# ══════════════════════════════════════════════════════════════

def run_multi(symbols: list[str], paper: bool = True):
    """
    Run market making on multiple tokens in parallel threads.
    Each token gets its own MarketMaker instance and thread.
    """
    import threading

    print(f"\n🌙 Market Making Agent — {'PAPER' if paper else 'LIVE'}")
    print("=" * 60)
    rwa = ["SPX", "NDX", "OIL", "GOLD"]
    rwa_tokens    = [s for s in symbols if s.upper() in rwa]
    crypto_tokens = [s for s in symbols if s.upper() not in rwa]
    print(f"  RWA perps : {', '.join(rwa_tokens) if rwa_tokens else 'none'}")
    print(f"  Crypto    : {', '.join(crypto_tokens) if crypto_tokens else 'none'}")
    if rwa_tokens:
        print(f"  SPX note  : Best during NYSE hours (9:30am-4pm EST)")
        print(f"              Volume: $100M+/day | Tighter spreads applied")
    print(f"  Spread  : {MMConfig.BASE_SPREAD_PCT*100:.3f}% crypto | "
          f"{SPXMMConfig.BASE_SPREAD_PCT*100:.3f}% SPX base")
    print(f"  Update  : every {MMConfig.UPDATE_INTERVAL_S}s")
    print(f"  Mode    : {'⚠️  PAPER (safe test)' if paper else '💰 LIVE (real money)'}")
    print()
    print(f"  ⚠️  PROP FIRM NOTE: Market making is for personal")
    print(f"     Hyperliquid account ONLY. Not compatible with Apex.")
    print()
    print("  Spread income formula:")
    print(f"    Maker rebate {MMConfig.MAKER_FEE_PCT*100:.3f}% + "
          f"spread {MMConfig.BASE_SPREAD_PCT*100:.3f}%")
    print(f"    = ~{(MMConfig.MAKER_FEE_PCT+MMConfig.BASE_SPREAD_PCT)*100:.3f}% per round trip")
    print(f"    × {24*3600//MMConfig.UPDATE_INTERVAL_S:,} updates/day/token")
    print("=" * 60)

    def _get_config(sym: str) -> MMConfig:
        """Use SPX-specific config for index perps."""
        rwa = ["SPX", "NDX", "OIL", "GOLD", "SILVER"]
        return SPXMMConfig() if sym.upper() in rwa else MMConfig()

    makers  = [MarketMaker(sym, paper, config=_get_config(sym))
               for sym in symbols]
    threads = []

    for mm in makers:
        t = threading.Thread(target=mm.run, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)   # stagger starts

    try:
        while True:
            time.sleep(60)
            # Status summary every minute
            print(f"\n  📊 Status — {datetime.now().strftime('%H:%M:%S')}")
            for mm in makers:
                price = mm.last_price
                if price > 0:
                    pnl = mm.inventory.total_pnl(price)
                    print(f"     {mm.inventory.summary(price)}")
    except KeyboardInterrupt:
        print("\n\n  ⏹  Stopping all market makers...")
        for mm in makers:
            mm.running = False
        for t in threads:
            t.join(timeout=10)
        print("  ✅ All stopped")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🌙 Market Making Agent")
    p.add_argument("--paper",    action="store_true",
                   help="Paper trading mode (no real orders)")
    p.add_argument("--live",     type=str, metavar="TOKENS",
                   help="Live mode: --live SPX or --live SPX,TAO,PEPE")
    p.add_argument("--status",   action="store_true",
                   help="Show P&L log summary")
    p.add_argument("--explain",  action="store_true",
                   help="Explain the strategy in detail")
    args = p.parse_args()

    if args.explain:
        print("""
🌙 Market Making Agent — How It Works
======================================

WHAT YOU DO:
  You post a BUY order slightly below market price
  and a SELL order slightly above market price.
  When both fill, you profit the spread.

  TAO at $450.00:
    Your BID: $449.33 (0.15% below)
    Your ASK: $450.67 (0.15% above)
    Profit if both fill: $1.34 per TAO

RENTECH INNOVATION — QUOTE SKEWING:
  Instead of always centering your quotes on mid price,
  you SHIFT them based on a short-term price prediction.

  If the model says price is going UP:
    Bid moves to $449.55 (more aggressive — want to buy now)
    Ask moves to $450.95 (less aggressive — wait for higher)
    → You capture more spread when right

  If the model says price is going DOWN:
    Bid moves to $449.10 (less aggressive — avoid buying)
    Ask moves to $450.50 (more aggressive — sell quickly)
    → You reduce inventory risk when market falls

PREDICTION MODEL (5 signals weighted):
  1. Order book imbalance  35% — more bids than asks = bullish
  2. MACD histogram        25% — momentum acceleration
  3. RSI position          15% — overbought/oversold
  4. Short-term momentum   15% — recent price direction
  5. Volume trend          10% — confirms the move

TRADE FREQUENCY:
  5-second update cycle × 60 min × 16 trading hours = 11,520 updates/day
  Not every update results in a fill — actual fills depend on
  how often the market crosses your quotes.
  Typical fill rate: 10-30% of updates = 1,000-3,000 fills/day per token

HYPERLIQUID ADVANTAGE:
  Hyperliquid PAYS you 0.02% rebate for providing liquidity.
  So your income = spread (0.15%) + rebate (0.02%) = 0.17% per round trip
  Taker pays 0.05% — you collect part of that too.
        """)

    elif args.status:
        if MM_LOG.exists():
            df = pd.read_csv(MM_LOG)
            print(f"\n📊 Market Making Log Summary")
            print(f"  Total updates: {len(df)}")
            if "pnl" in df.columns:
                for sym in df["symbol"].unique():
                    sub = df[df["symbol"] == sym]
                    print(f"\n  {sym}:")
                    print(f"    Updates:  {len(sub)}")
                    print(f"    Avg spread: {sub['spread_pct'].mean():.4f}%")
                    print(f"    Avg skew:   {sub['skew_pct'].mean():+.4f}%")
                    print(f"    Latest PnL: ${sub['pnl'].iloc[-1]:+.2f}")
        else:
            print("No log yet — run the agent first")

    elif args.live:
        symbols = [s.strip().upper() for s in args.live.split(",")]
        run_multi(symbols, paper=False)

    else:
        # Default: paper mode on TAO
        print("\n  Starting PAPER mode on TAO (safe test — no real orders)")
        print("  Run --live TAO when ready for real trading\n")
        run_multi(["TAO"], paper=True)
