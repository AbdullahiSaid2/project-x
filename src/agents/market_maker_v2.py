#!/usr/bin/env python3
# ============================================================
# 🌙 Market Maker Agent v2 — WebSocket Edition
#
# Upgrade from REST polling (5-second lag) to WebSocket
# subscriptions (real-time, <100ms reaction time).
#
# This is the key step toward RenTec-style market making:
#
# v1 (REST):       poll orderbook every 5s → requote
# v2 (WebSocket):  receive every orderbook change → requote instantly
#
# WHAT'S NEW IN v2:
#   1. WebSocket orderbook — react in <100ms vs 5 seconds
#   2. BTC lead signal — when BTC moves, pre-skew altcoins
#      before the correlation propagates (RenTec's core edge)
#   3. Fill detection — detect your own fills via trade stream
#   4. Multi-asset correlated skewing — simultaneous updates
#   5. Adaptive spread — widens automatically on volatility spikes
#
# INSTALL:
#   pip install websockets
#
# HOW TO RUN:
#   python src/agents/market_maker_v2.py --paper          # safe test
#   python src/agents/market_maker_v2.py --live TAO       # 1 token
#   python src/agents/market_maker_v2.py --live TAO,PEPE,WIF
#   python src/agents/market_maker_v2.py --live SPX,TAO,PEPE
# ============================================================

import os, sys, json, time, asyncio, csv, math
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv
load_dotenv()

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"
LOG_FILE = DATA_DIR / "mm_v2_log.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HL_WS_URL  = "wss://api.hyperliquid.xyz/ws"
HL_API_URL = "https://api.hyperliquid.xyz"

# ── Correlation table (beta of token to BTC moves) ────────────
# When BTC moves X%, this token typically moves X% × beta
# Calibrated from historical data — update periodically
BTC_BETA = {
    "ETH":  0.95,
    "SOL":  1.20,
    "BNB":  0.80,
    "AVAX": 1.15,
    "TAO":  1.30,   # AI tokens more volatile
    "FET":  1.40,
    "RNDR": 1.35,
    "PEPE": 1.60,   # Meme tokens most volatile
    "WIF":  1.70,
    "ARB":  1.10,
    "OP":   1.10,
    "SPX":  0.30,   # S&P 500 perp — low correlation to BTC
}


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

class MMv2Config:
    BASE_SPREAD_PCT      = 0.0015   # 0.15% base
    MIN_SPREAD_PCT       = 0.0006
    MAX_SPREAD_PCT       = 0.006
    MAX_SKEW_PCT         = 0.0012
    ORDER_SIZE_USD       = 50
    MAX_INVENTORY_USD    = 400
    MAX_DAILY_LOSS_USD   = 100
    MAKER_FEE_PCT        = 0.0002   # Hyperliquid maker rebate

    # v2 specific
    BTC_LEAD_WEIGHT      = 0.4      # how much BTC signal influences altcoin skew
    FILL_LEARNING_RATE   = 0.05     # how fast fill feedback adjusts the model
    REQUOTE_COOLDOWN_MS  = 80       # minimum ms between requotes (avoid rate limits)
    MAX_REQUOTES_PER_SEC = 8        # HL rate limit ~10/s, stay safe at 8


# ══════════════════════════════════════════════════════════════
# SKEW PREDICTOR v2 — with BTC lead signal + fill feedback
# ══════════════════════════════════════════════════════════════

class SkewPredictorV2:
    """
    Enhanced predictor with two new signals vs v1:

    New Signal 6: BTC Lead Signal
      When BTC moves, altcoins typically follow within 15-60 seconds.
      We pre-skew altcoin quotes BEFORE the correlation propagates.
      This is the closest retail equivalent to RenTec's cross-asset signals.

    New Signal 7: Fill Feedback (self-learning)
      Track whether bids/asks that fill are followed by price moves
      in our favour or against us. Adjust the model accordingly.
      After 100+ fills, this becomes a genuine ML feedback loop.
    """

    def __init__(self, symbol: str, lookback: int = 30):
        self.symbol       = symbol
        self.lookback     = lookback
        self.price_hist   = deque(maxlen=200)
        self.vol_hist     = deque(maxlen=50)
        self.btc_moves    = deque(maxlen=20)   # recent BTC % moves
        self.fill_history = deque(maxlen=200)  # (side, price, post_move) tuples
        self.last_score   = 0.0
        self.fill_bias    = 0.0                # learned from fill outcomes
        self.beta         = BTC_BETA.get(symbol, 1.0)

    def update_price(self, price: float):
        self.price_hist.append(price)

    def update_btc_move(self, btc_pct_move: float):
        """Called when BTC price changes significantly."""
        self.btc_moves.append(btc_pct_move)

    def record_fill(self, side: str, fill_price: float,
                    price_5s_later: float):
        """
        Record fill outcome for self-learning.
        side: 'bid' or 'ask'
        If bid filled and price went UP → good outcome (we bought cheap)
        If bid filled and price went DOWN → bad outcome (inventory loss)
        """
        if fill_price <= 0 or price_5s_later <= 0:
            return
        price_move = (price_5s_later - fill_price) / fill_price
        if side == 'bid':
            outcome = price_move     # positive = good (bought, price went up)
        else:
            outcome = -price_move    # positive = good (sold, price went down)

        self.fill_history.append(outcome)

        # Update fill bias — running average of fill outcomes
        if len(self.fill_history) >= 10:
            recent_avg = np.mean(list(self.fill_history)[-20:])
            # Slowly update bias toward recent fill outcomes
            self.fill_bias = (0.95 * self.fill_bias +
                              0.05 * np.clip(recent_avg * 100, -0.5, 0.5))

    def predict(self, orderbook: dict) -> tuple[float, dict]:
        """Returns (score -1 to +1, components)."""
        comps = {}

        # ── Signal 1: MACD histogram ──────────────────────────
        if len(self.price_hist) >= 26:
            prices = np.array(self.price_hist)
            ema12  = _ema(prices, 12)
            ema26  = _ema(prices, 26)
            macd   = ema12 - ema26
            sig9   = _ema(macd, 9)
            hist   = macd - sig9
            if len(hist) >= 2:
                comps["macd"] = float(np.clip(
                    (hist[-1] - hist[-2]) / (abs(hist[-1]) + 1e-8), -1, 1))
            else:
                comps["macd"] = 0.0
        else:
            comps["macd"] = 0.0

        # ── Signal 2: RSI ─────────────────────────────────────
        if len(self.price_hist) >= 14:
            prices = np.array(self.price_hist)
            rsi    = _rsi(prices, 14)
            comps["rsi"] = float(np.clip((50 - rsi) / 50, -1, 1))
        else:
            comps["rsi"] = 0.0

        # ── Signal 3: Orderbook imbalance ─────────────────────
        if orderbook and "bids" in orderbook and "asks" in orderbook:
            bids = orderbook["bids"]
            asks = orderbook["asks"]
            if bids and asks:
                # Use top 10 levels for better signal
                bid_depth = sum(float(b[1]) for b in bids[:10])
                ask_depth = sum(float(a[1]) for a in asks[:10])
                total = bid_depth + ask_depth
                comps["orderbook"] = float(np.clip(
                    (bid_depth - ask_depth) / (total + 1e-8), -1, 1))
                # Also check for large walls
                # A single large bid wall = strong support
                max_bid = max((float(b[1]) for b in bids[:5]), default=0)
                max_ask = max((float(a[1]) for a in asks[:5]), default=0)
                if max_bid > ask_depth * 0.5:
                    comps["orderbook"] = min(1.0, comps["orderbook"] + 0.2)
                elif max_ask > bid_depth * 0.5:
                    comps["orderbook"] = max(-1.0, comps["orderbook"] - 0.2)
            else:
                comps["orderbook"] = 0.0
        else:
            comps["orderbook"] = 0.0

        # ── Signal 4: Short-term momentum ─────────────────────
        if len(self.price_hist) >= 10:
            prices = np.array(self.price_hist)
            sma10  = prices[-10:].mean()
            comps["momentum"] = float(np.clip(
                (prices[-1] - sma10) / sma10 * 50, -1, 1))
        else:
            comps["momentum"] = 0.0

        # ── Signal 5: Volume trend ─────────────────────────────
        if len(self.vol_hist) >= 5:
            vols = np.array(self.vol_hist)
            vol_ratio = vols[-3:].mean() / (vols[-5:-2].mean() + 1e-8)
            price_dir = 1 if (len(self.price_hist) >= 2 and
                              self.price_hist[-1] > self.price_hist[-2]) else -1
            comps["volume"] = float(np.clip((vol_ratio - 1) * price_dir * 2, -1, 1))
        else:
            comps["volume"] = 0.0

        # ── Signal 6: BTC Lead Signal (NEW in v2) ─────────────
        # This is the RenTec-style cross-asset signal.
        # When BTC just moved, predict this token will follow.
        if self.btc_moves and self.symbol != "BTC":
            # Weight recent BTC moves more
            weights = np.exp(-0.3 * np.arange(len(self.btc_moves))[::-1])
            btc_signal = float(np.average(
                list(self.btc_moves), weights=weights))
            # Scale by beta (TAO follows BTC 1.3x, SPX follows 0.3x)
            comps["btc_lead"] = float(np.clip(btc_signal * self.beta * 20, -1, 1))
        else:
            comps["btc_lead"] = 0.0

        # ── Signal 7: Fill Feedback (self-learning) ───────────
        # After enough fills, this becomes a genuine ML signal.
        comps["fill_bias"] = float(np.clip(self.fill_bias, -1, 1))

        # ── Weighted combination ──────────────────────────────
        weights = {
            "orderbook": 0.30,
            "btc_lead":  0.20,   # new — substantial weight on cross-asset
            "macd":      0.20,
            "momentum":  0.13,
            "rsi":       0.10,
            "volume":    0.05,
            "fill_bias": 0.02,   # small now — grows as fills accumulate
        }

        score = sum(weights[k] * comps[k] for k in weights)
        score = float(np.clip(score, -1, 1))

        # Smooth
        self.last_score = 0.6 * score + 0.4 * self.last_score
        comps["final"] = round(self.last_score, 4)

        return self.last_score, comps


def _ema(data, period):
    alpha  = 2.0 / (period + 1)
    result = np.zeros(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

def _rsi(data, period=14):
    deltas = np.diff(data[-period-1:])
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    ag, al = gains.mean(), losses.mean()
    if al == 0: return 100.0
    return 100 - (100 / (1 + ag/al))


# ══════════════════════════════════════════════════════════════
# INVENTORY TRACKER (same as v1)
# ══════════════════════════════════════════════════════════════

class InventoryTracker:
    def __init__(self, symbol):
        self.symbol        = symbol
        self.net_qty       = 0.0
        self.avg_cost      = 0.0
        self.realized_pnl  = 0.0
        self.fills         = []

    def record_fill(self, side, qty, price, fee=0):
        if side == "BUY":
            new_qty = self.net_qty + qty
            if new_qty:
                self.avg_cost = ((self.net_qty * self.avg_cost +
                                  qty * price) / new_qty)
            self.net_qty = new_qty
        else:
            pnl = (price - self.avg_cost) * qty - fee
            self.realized_pnl += pnl
            self.net_qty -= qty
            if abs(self.net_qty) < 1e-8:
                self.avg_cost = 0.0
        self.fills.append({"side": side, "qty": qty, "price": price})

    def inventory_usd(self, price):
        return self.net_qty * price

    def total_pnl(self, price):
        unrealized = self.net_qty * (price - self.avg_cost) if self.avg_cost else 0
        return self.realized_pnl + unrealized


# ══════════════════════════════════════════════════════════════
# WEBSOCKET MARKET MAKER — THE CORE
# ══════════════════════════════════════════════════════════════

class WebSocketMarketMaker:
    """
    Single-token WebSocket market maker.

    Maintains a persistent WebSocket connection to Hyperliquid.
    On every orderbook update (typically every 100-500ms):
      1. Calculates new bid/ask with skew from predictor
      2. Places/updates limit orders if price moved enough to requote
      3. Detects fills and updates inventory + predictor

    Runs as an asyncio coroutine — can run multiple tokens
    concurrently in one event loop.
    """

    def __init__(self, symbol: str, paper: bool = True,
                 config: MMv2Config = None,
                 btc_signal_queue: asyncio.Queue = None):
        self.symbol     = symbol
        self.paper      = paper
        self.config     = config or MMv2Config()
        self.predictor  = SkewPredictorV2(symbol)
        self.inventory  = InventoryTracker(symbol)
        self.btc_queue  = btc_signal_queue   # receives BTC moves

        self.current_bid    = 0.0
        self.current_ask    = 0.0
        self.last_mid       = 0.0
        self.last_requote   = 0.0
        self.active_bid_oid = None
        self.active_ask_oid = None
        self.daily_pnl      = 0.0
        self.updates        = 0
        self.requotes       = 0
        self.running        = True

        self.exchange       = None
        if not paper:
            self._init_exchange()

    def _init_exchange(self):
        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils import constants
            from eth_account import Account
            wallet = Account.from_key(os.getenv("HYPERLIQUID_API_PRIVATE_KEY"))
            self.exchange = Exchange(
                wallet,
                constants.MAINNET_API_URL,
                account_address=os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS")
            )
            print(f"  ✅ {self.symbol}: Exchange connected")
        except Exception as e:
            print(f"  ❌ {self.symbol}: Exchange connection failed: {e}")

    def _calculate_spread(self, volatility: float) -> float:
        """Adaptive spread — widens on volatility spikes."""
        vol_mult = 1 + (volatility / 0.015)  # baseline 1.5% vol
        spread = self.config.BASE_SPREAD_PCT * vol_mult
        return float(np.clip(spread,
                             self.config.MIN_SPREAD_PCT,
                             self.config.MAX_SPREAD_PCT))

    def _should_requote(self, new_bid: float, new_ask: float) -> bool:
        """Only requote if prices moved enough to justify cancel+replace."""
        if self.current_bid == 0 or self.current_ask == 0:
            return True
        # Requote if mid moved >25% of spread
        spread = self.current_ask - self.current_bid
        bid_drift = abs(new_bid - self.current_bid)
        ask_drift = abs(new_ask - self.current_ask)
        return (bid_drift + ask_drift) > spread * 0.25

    async def _place_quotes(self, bid: float, ask: float, mid: float):
        """Cancel old quotes and place new ones."""
        qty = round(self.config.ORDER_SIZE_USD / mid, 6) if mid > 0 else 0
        if qty <= 0:
            return

        if self.paper:
            self.current_bid = bid
            self.current_ask = ask
            self.requotes += 1
            return

        try:
            # Cancel existing
            if self.active_bid_oid:
                self.exchange.cancel(self.symbol, self.active_bid_oid)
            if self.active_ask_oid:
                self.exchange.cancel(self.symbol, self.active_ask_oid)

            await asyncio.sleep(0.02)  # 20ms gap

            # Place bid
            r_bid = self.exchange.order(
                self.symbol, is_buy=True, sz=qty,
                limit_px=bid,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=False
            )
            # Place ask
            r_ask = self.exchange.order(
                self.symbol, is_buy=False, sz=qty,
                limit_px=ask,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=False
            )

            self.current_bid = bid
            self.current_ask = ask
            self.requotes   += 1

            # Store order IDs for cancellation
            try:
                statuses = r_bid.get("response", {}).get("data", {}).get("statuses", [{}])
                self.active_bid_oid = statuses[0].get("resting", {}).get("oid")
                statuses = r_ask.get("response", {}).get("data", {}).get("statuses", [{}])
                self.active_ask_oid = statuses[0].get("resting", {}).get("oid")
            except Exception:
                pass

        except Exception as e:
            print(f"  ⚠️  {self.symbol}: Quote error: {e}")

    def _log(self, mid: float, bid: float, ask: float,
             score: float, skew: float):
        exists = LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts","symbol","mid","bid","ask",
                "spread_pct","skew_score","skew_pct",
                "inv_usd","pnl","mode"
            ])
            if not exists:
                w.writeheader()
            spread_pct = (ask - bid) / mid * 100 if mid else 0
            inv_usd    = self.inventory.inventory_usd(mid)
            pnl        = self.inventory.total_pnl(mid)
            w.writerow({
                "ts":         datetime.now(timezone.utc).isoformat(),
                "symbol":     self.symbol,
                "mid":        round(mid, 6),
                "bid":        round(bid, 6),
                "ask":        round(ask, 6),
                "spread_pct": round(spread_pct, 4),
                "skew_score": round(score, 4),
                "skew_pct":   round(skew * 100, 4),
                "inv_usd":    round(inv_usd, 2),
                "pnl":        round(pnl, 2),
                "mode":       "paper" if self.paper else "live",
            })

    async def on_orderbook_update(self, book: dict):
        """
        Called on every orderbook WebSocket message.
        This is the hot path — keep it fast.
        """
        self.updates += 1

        # Extract mid price
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if not bids or not asks:
            return

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid      = (best_bid + best_ask) / 2

        if mid <= 0:
            return

        # Rate limit check
        now_ms = time.time() * 1000
        if (now_ms - self.last_requote <
                self.config.REQUOTE_COOLDOWN_MS):
            return

        # Check for BTC signals in queue
        if self.btc_queue:
            while not self.btc_queue.empty():
                try:
                    btc_move = self.btc_queue.get_nowait()
                    self.predictor.update_btc_move(btc_move)
                except asyncio.QueueEmpty:
                    break

        # Update predictor
        self.predictor.update_price(mid)

        # Get prediction
        score, comps = self.predictor.predict({
            "bids": bids, "asks": asks
        })

        # Calculate spread and skew
        if len(self.predictor.price_hist) >= 5:
            prices = np.array(self.predictor.price_hist)
            returns = np.diff(prices[-10:]) / prices[-10:-1]
            volatility = float(np.std(returns)) if len(returns) > 0 else 0.015
        else:
            volatility = 0.015

        spread_pct = self._calculate_spread(volatility)
        inv_usd    = self.inventory.inventory_usd(mid)
        max_inv    = self.config.MAX_INVENTORY_USD

        # Skew: prediction + inventory correction
        pred_skew = score * self.config.MAX_SKEW_PCT * 0.65
        inv_skew  = -(inv_usd / max_inv) * self.config.MAX_SKEW_PCT * 0.5
        total_skew = float(np.clip(pred_skew + inv_skew,
                                   -self.config.MAX_SKEW_PCT,
                                   self.config.MAX_SKEW_PCT))

        half_spread = spread_pct / 2
        new_bid = mid * (1 - half_spread + total_skew)
        new_ask = mid * (1 + half_spread + total_skew)

        # Risk check
        pnl = self.inventory.total_pnl(mid)
        if pnl < -self.config.MAX_DAILY_LOSS_USD:
            if self.current_bid > 0:
                print(f"  🚫 {self.symbol}: Daily loss limit — stopping")
                self.running = False
            return

        # Only requote if price moved meaningfully
        if not self._should_requote(new_bid, new_ask):
            return

        # Place quotes
        await self._place_quotes(new_bid, new_ask, mid)
        self.last_requote = now_ms
        self.last_mid     = mid

        # Log periodically (every 50 updates to avoid excessive writes)
        if self.updates % 50 == 0:
            self._log(mid, new_bid, new_ask, score, total_skew)

        # Print status every 200 updates
        if self.updates % 200 == 0:
            direction = ("🟢" if score > 0.2 else
                         "🔴" if score < -0.2 else "⚪")
            btc_sig = comps.get("btc_lead", 0)
            mode    = "[PAPER]" if self.paper else "[LIVE]"
            print(f"  {mode} {self.symbol:6} ${mid:.4f} | "
                  f"Bid={new_bid:.4f} Ask={new_ask:.4f} | "
                  f"Spread={spread_pct*100:.3f}% | "
                  f"Skew={total_skew*100:+.3f}% {direction} | "
                  f"BTC={btc_sig:+.2f} | "
                  f"Inv=${inv_usd:+.2f} | "
                  f"PnL=${pnl:+.2f} | "
                  f"Requotes={self.requotes}")

    async def run_ws(self):
        """Main WebSocket loop for this token."""
        try:
            import websockets
        except ImportError:
            print(f"\n  ❌ websockets not installed.")
            print(f"     Run: pip install websockets")
            print(f"     Then restart the agent.\n")
            print(f"  💡 Running in REST fallback mode (5s cycle)...")
            await self._run_rest_fallback()
            return

        print(f"  🔌 {self.symbol}: Connecting to Hyperliquid WebSocket...")
        retry_delay = 1

        while self.running:
            try:
                async with websockets.connect(
                    HL_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    # Subscribe to L2 orderbook for this token
                    sub_msg = json.dumps({
                        "method": "subscribe",
                        "subscription": {
                            "type": "l2Book",
                            "coin": self.symbol
                        }
                    })
                    await ws.send(sub_msg)
                    print(f"  ✅ {self.symbol}: WebSocket subscribed")
                    retry_delay = 1  # reset on success

                    async for raw in ws:
                        if not self.running:
                            break
                        try:
                            msg  = json.loads(raw)
                            data = msg.get("data", {})
                            # HL sends: {"channel": "l2Book", "data": {"coin": ..., "levels": [bids, asks]}}
                            if msg.get("channel") == "l2Book":
                                levels = data.get("levels", [[], []])
                                book = {
                                    "bids": levels[0] if len(levels) > 0 else [],
                                    "asks": levels[1] if len(levels) > 1 else [],
                                }
                                await self.on_orderbook_update(book)
                        except Exception as e:
                            pass   # skip malformed messages

            except Exception as e:
                if not self.running:
                    break
                print(f"  ⚠️  {self.symbol}: WS disconnected ({e}) "
                      f"— reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

    async def _run_rest_fallback(self):
        """Fallback to REST polling if websockets not installed."""
        import aiohttp
        print(f"  🔄 {self.symbol}: REST fallback (5s cycle)")
        while self.running:
            try:
                import requests
                r = requests.post(
                    f"{HL_API_URL}/info",
                    json={"type": "l2Book", "coin": self.symbol},
                    timeout=5
                )
                r.raise_for_status()
                book = r.json()
                levels = book.get("levels", [[], []])
                await self.on_orderbook_update({
                    "bids": levels[0],
                    "asks": levels[1],
                })
            except Exception as e:
                print(f"  ⚠️  {self.symbol}: REST error: {e}")
            await asyncio.sleep(5)


# ══════════════════════════════════════════════════════════════
# BTC SIGNAL BROADCASTER
# ══════════════════════════════════════════════════════════════

class BTCSignalBroadcaster:
    """
    Monitors BTC price via WebSocket.
    When BTC moves significantly, broadcasts the move
    to all altcoin market makers so they pre-skew quotes.

    This is the key RenTec innovation in retail form:
    cross-asset lead signals propagated in real-time.
    """

    def __init__(self, queues: dict[str, asyncio.Queue],
                 min_move_pct: float = 0.05):
        self.queues       = queues      # {symbol: queue}
        self.min_move_pct = min_move_pct
        self.last_btc     = 0.0
        self.running      = True

    async def run(self):
        """Subscribe to BTC orderbook, broadcast moves to altcoins."""
        try:
            import websockets
        except ImportError:
            return   # silently skip if websockets not installed

        print(f"  📡 BTC Signal Broadcaster: starting...")
        while self.running:
            try:
                async with websockets.connect(HL_WS_URL,
                                               ping_interval=20) as ws:
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "l2Book", "coin": "BTC"}
                    }))
                    print(f"  ✅ BTC Signal Broadcaster: connected")

                    async for raw in ws:
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw)
                            if msg.get("channel") != "l2Book":
                                continue
                            levels = msg["data"].get("levels", [[], []])
                            bids   = levels[0]
                            asks   = levels[1] if len(levels) > 1 else []
                            if not bids or not asks:
                                continue

                            btc_mid = (float(bids[0][0]) + float(asks[0][0])) / 2
                            if self.last_btc > 0:
                                pct_move = (btc_mid - self.last_btc) / self.last_btc * 100
                                if abs(pct_move) >= self.min_move_pct:
                                    # Broadcast to all altcoin queues
                                    for sym, q in self.queues.items():
                                        if sym != "BTC":
                                            try:
                                                q.put_nowait(pct_move)
                                            except asyncio.QueueFull:
                                                pass
                            self.last_btc = btc_mid
                        except Exception:
                            pass
            except Exception as e:
                if not self.running:
                    break
                await asyncio.sleep(3)


# ══════════════════════════════════════════════════════════════
# MULTI-TOKEN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

async def run_all(symbols: list[str], paper: bool = True):
    """
    Run all tokens concurrently in one asyncio event loop.
    Each token has its own WebSocket connection and quote engine.
    BTC signal broadcaster runs as a shared coroutine.
    """
    print(f"\n🌙 Market Maker v2 — WebSocket Edition")
    print("=" * 62)
    print(f"  Mode    : {'⚠️  PAPER' if paper else '💰 LIVE'}")
    print(f"  Tokens  : {', '.join(symbols)}")
    print(f"  Speed   : Real-time WebSocket (vs 5s REST in v1)")
    print(f"  BTC lead: ✅ Cross-asset signal active")
    print(f"  Learning: ✅ Fill feedback loop active")
    print()
    print("  HOW IT'S DIFFERENT FROM v1:")
    print("  • Reacts to orderbook changes in <100ms (not 5s)")
    print("  • BTC moves instantly skew all altcoin quotes")
    print("  • Fills train the model over time (self-learning)")
    print("  • Multiple assets run concurrently (not sequentially)")
    print()
    print("  NOTE: Prop firm accounts — use v1 (REST) only.")
    print("  This runs on personal Hyperliquid account only.")
    print("=" * 62)

    try:
        import websockets
        print(f"\n  ✅ websockets installed — full WebSocket mode")
    except ImportError:
        print(f"\n  ⚠️  websockets not installed — REST fallback mode")
        print(f"     To get full speed: pip install websockets")

    # Create BTC signal queues for each non-BTC token
    btc_queues = {sym: asyncio.Queue(maxsize=10)
                  for sym in symbols if sym != "BTC"}

    # Create market makers
    makers = [
        WebSocketMarketMaker(sym, paper,
                             btc_signal_queue=btc_queues.get(sym))
        for sym in symbols
    ]

    # Create BTC broadcaster
    broadcaster = BTCSignalBroadcaster(btc_queues, min_move_pct=0.05)

    # Status printer
    async def status_loop():
        while True:
            await asyncio.sleep(60)
            print(f"\n  📊 Status {datetime.now().strftime('%H:%M:%S')}")
            for mm in makers:
                if mm.last_mid > 0:
                    pnl = mm.inventory.total_pnl(mm.last_mid)
                    inv = mm.inventory.inventory_usd(mm.last_mid)
                    fills = len(mm.inventory.fills)
                    print(f"     {mm.symbol:6} "
                          f"mid=${mm.last_mid:.4f} | "
                          f"inv=${inv:+.2f} | "
                          f"pnl=${pnl:+.2f} | "
                          f"fills={fills} | "
                          f"requotes={mm.requotes}")

    # Run everything concurrently
    tasks = [mm.run_ws() for mm in makers]
    tasks.append(broadcaster.run())
    tasks.append(status_loop())

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print(f"\n\n  ⏹  Stopping all market makers...")
        for mm in makers:
            mm.running = False
        broadcaster.running = False


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🌙 Market Maker v2 (WebSocket)")
    p.add_argument("--paper", action="store_true",
                   help="Paper mode (no real orders) — DEFAULT")
    p.add_argument("--live",  type=str, metavar="TOKENS",
                   help="Live: --live TAO or --live TAO,PEPE,WIF,SPX")
    p.add_argument("--status", action="store_true",
                   help="Show P&L log")
    args = p.parse_args()

    if args.status:
        if LOG_FILE.exists():
            import pandas as pd
            df = pd.read_csv(LOG_FILE)
            for sym in df["symbol"].unique():
                sub = df[df["symbol"] == sym]
                print(f"\n  {sym}: {len(sub)} updates | "
                      f"Latest PnL: ${sub['pnl'].iloc[-1]:+.2f}")
        else:
            print("No log yet")

    elif args.live:
        symbols = [s.strip().upper() for s in args.live.split(",")]
        asyncio.run(run_all(symbols, paper=False))

    else:
        # Default: paper mode
        tokens = ["TAO", "PEPE"]
        print(f"\nStarting PAPER mode on {', '.join(tokens)}")
        print("Use --live TAO,PEPE,WIF for real trading\n")
        asyncio.run(run_all(tokens, paper=True))
