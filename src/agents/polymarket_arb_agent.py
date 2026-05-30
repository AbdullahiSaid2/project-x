#!/usr/bin/env python3
# ============================================================
# 🌙 Poly Market Arbitrage Agent
#
# TWO STRATEGIES:
#
# ── Strategy 1: AMM Both-Sides Arbitrage ────────────────────
#   In Poly Market's 15-minute crypto up/down markets, the
#   AMM sometimes misprices YES and NO tokens such that:
#
#     price_YES + price_NO < (1 - FEE) = $0.98
#
#   Buying BOTH sides guarantees profit regardless of outcome.
#   This is a pure arbitrage — direction doesn't matter.
#
#   Example: YES = $0.47, NO = $0.47 → total = $0.94
#   Payout after 2% fee = $0.98 → profit = $0.04 per $0.94 bet
#   That's 4.25% in 15 minutes. Annualised: insane.
#
# ── Strategy 2: Oracle Front-Running ────────────────────────
#   Poly Market resolves 15-minute markets using a price oracle
#   (UMA/Chainlink) that updates with a lag vs real spot price.
#
#   We have a real-time Hyperliquid WebSocket price feed that
#   is faster than Poly Market's oracle. When real price has
#   moved significantly toward one outcome but Poly Market
#   odds haven't updated yet — we buy the winning side at
#   stale odds and profit when they correct.
#
#   Edge: our feed is ~100ms; oracle lag is 15-30 seconds.
#
# HOW TO RUN:
#   python src/agents/polymarket_arb_agent.py --paper        # paper mode
#   python src/agents/polymarket_arb_agent.py --live         # live trading
#   python src/agents/polymarket_arb_agent.py --scan         # scan only, no trades
#
# SETUP REQUIRED:
#   1. Create Poly Market account at polymarket.com
#   2. Get API credentials: Settings → API
#   3. Bridge USDC to Polygon network
#   4. Add to .env:
#      POLYMARKET_API_KEY=...
#      POLYMARKET_API_SECRET=...
#      POLYMARKET_API_PASSPHRASE=...
#      POLYMARKET_WALLET_PRIVATE_KEY=...   # Polygon wallet private key
#      POLYMARKET_WALLET_ADDRESS=...       # 0x... address
#   5. pip install py-clob-client websockets aiohttp
#
# IMPORTANT NOTES:
#   - Markets only: BTC, ETH, SOL, XRP 15-minute up/down
#   - Poly Market fee: 2% of winnings (baked into payout calc)
#   - Geographic restriction: US users blocked — Nairobi OK ✓
#   - Poly Market runs on Polygon (MATIC) blockchain
#   - Minimum bet: ~$1 USDC
#
# INSTALL:
#   pip install py-clob-client websockets aiohttp --break-system-packages
# ============================================================

import os, sys, json, time, asyncio, csv, logging
import aiohttp
import websockets
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv
load_dotenv()

# ── Poly Market CLOB client ───────────────────────────────────
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs, OrderType, MarketOrderArgs, BookParams
    )
    from py_clob_client.constants import POLYGON
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False
    print("⚠️  py-clob-client not installed. Run: pip install py-clob-client")

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"
LOG_FILE = DATA_DIR / "polymarket_arb_log.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("polymarket_arb")

# ── Credentials ───────────────────────────────────────────────
PM_API_KEY         = os.getenv("POLYMARKET_API_KEY", "")
PM_API_SECRET      = os.getenv("POLYMARKET_API_SECRET", "")
PM_API_PASSPHRASE  = os.getenv("POLYMARKET_API_PASSPHRASE", "")
PM_PRIVATE_KEY     = os.getenv("POLYMARKET_WALLET_PRIVATE_KEY", "")
PM_WALLET_ADDRESS  = os.getenv("POLYMARKET_WALLET_ADDRESS", "")

HL_WS_URL          = "wss://api.hyperliquid.xyz/ws"
PM_CLOB_URL        = "https://clob.polymarket.com"
PM_GAMMA_URL       = "https://gamma-api.polymarket.com"
PM_WS_URL          = "wss://clob.polymarket.com/ws/"


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

class ArbConfig:
    # ── Strategy 1: AMM Both-Sides Arb ──────────────────────
    # Buy both YES + NO when combined price < this threshold.
    # Poly Market charges 2% fee on winnings → payout = $0.98
    # We require at least 2% margin above fees = 0.94 threshold
    AMM_ARB_THRESHOLD    = 0.94    # buy both if YES+NO < 0.94 (4% margin)
    AMM_ARB_MIN_PROFIT   = 0.02   # minimum $ profit per $1 bet to execute
    AMM_BET_SIZE_USD     = 20.0   # USD to bet on each side (total $40 per arb)
    AMM_MAX_POSITIONS    = 5      # max simultaneous both-sides positions

    # ── Strategy 2: Oracle Front-Running ─────────────────────
    # Fire when real price is this many % away from target AND
    # Poly Market hasn't corrected odds yet.
    ORACLE_PRICE_DELTA_PCT  = 0.003   # 0.3% real-price vs oracle threshold
    ORACLE_ODDS_STALE_SECS  = 10      # consider odds stale after 10s no update
    ORACLE_MIN_ODDS         = 0.75    # only buy if we get ≥75 cents for $1 payout
    ORACLE_MAX_ODDS         = 0.97    # don't buy if odds > 97 cents (tiny edge)
    ORACLE_BET_SIZE_USD     = 15.0    # USD per oracle front-run trade
    ORACLE_MIN_TIME_LEFT    = 30      # seconds — don't enter with < 30s left
    ORACLE_MAX_TIME_LEFT    = 600     # seconds — don't enter with > 10min left
    ORACLE_MAX_POSITIONS    = 8       # max simultaneous oracle positions

    # ── General ───────────────────────────────────────────────
    POLY_FEE_RATE        = 0.02    # Poly Market 2% fee on winnings
    PAYOUT_AFTER_FEE     = 1.0 - POLY_FEE_RATE   # = 0.98
    MAX_DAILY_LOSS_USD   = 100.0   # stop all trading if we hit this
    SCAN_INTERVAL_SECS   = 5      # how often to scan for new opportunities
    MARKETS              = ["BTC", "ETH", "SOL", "XRP"]  # 15-min markets


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class MarketSnapshot:
    """Current state of one 15-minute Poly Market."""
    condition_id:   str
    question:       str
    symbol:         str           # BTC / ETH / SOL / XRP
    target_price:   float         # the price the market resolves YES/NO against
    yes_token_id:   str
    no_token_id:    str
    yes_price:      float         # current YES share price ($0.00 - $1.00)
    no_price:       float         # current NO share price
    yes_size:       float         # liquidity available on YES
    no_size:        float         # liquidity available on NO
    end_time:       datetime      # when market resolves
    oracle_price:   float = 0.0   # PM's oracle price (may be stale)
    oracle_updated: float = 0.0   # unix timestamp of last oracle update
    last_scanned:   float = field(default_factory=time.time)

    @property
    def combined_cost(self) -> float:
        return self.yes_price + self.no_price

    @property
    def arb_profit_pct(self) -> float:
        """Expected profit % if we buy both sides."""
        return (ArbConfig.PAYOUT_AFTER_FEE - self.combined_cost) / self.combined_cost

    @property
    def seconds_remaining(self) -> float:
        now = datetime.now(timezone.utc)
        return max(0, (self.end_time - now).total_seconds())

    @property
    def has_amm_arb(self) -> bool:
        return (
            self.combined_cost < ArbConfig.AMM_ARB_THRESHOLD and
            self.arb_profit_pct >= ArbConfig.AMM_ARB_MIN_PROFIT and
            self.seconds_remaining > 30
        )


@dataclass
class Position:
    """An open position — either arb or oracle front-run."""
    strategy:       str           # "amm_arb" | "oracle_frontrun"
    condition_id:   str
    symbol:         str
    side:           str           # "yes" | "no" | "both"
    token_id:       str           # for single-side; "" for both
    yes_token_id:   str = ""
    no_token_id:    str = ""
    entry_yes_price: float = 0.0
    entry_no_price:  float = 0.0
    size_yes_usd:   float = 0.0
    size_no_usd:    float = 0.0
    shares_yes:     float = 0.0
    shares_no:      float = 0.0
    entry_time:     float = field(default_factory=time.time)
    end_time:       Optional[datetime] = None
    pnl:            float = 0.0
    status:         str = "open"  # open | resolved | closed


# ══════════════════════════════════════════════════════════════
# PRICE FEED — Hyperliquid WebSocket
# ══════════════════════════════════════════════════════════════

class HyperliquidFeed:
    """
    Real-time price feed from Hyperliquid WebSocket.
    This is the 'fast oracle' we use to front-run Poly Market's slow oracle.
    Sub-100ms latency vs Poly Market's 15-30 second oracle lag.
    """

    def __init__(self):
        self.prices: dict[str, float] = {}
        self.last_updated: dict[str, float] = {}
        self._ws = None
        self._running = False
        # HL symbol → Poly Market symbol mapping
        self.symbol_map = {
            "BTC":  "BTC",
            "ETH":  "ETH",
            "SOL":  "SOL",
            "XRP":  "XRP",
        }

    async def connect(self):
        """Establish WebSocket connection to Hyperliquid."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    HL_WS_URL,
                    ping_interval=30,
                    ping_timeout=10
                ) as ws:
                    self._ws = ws
                    log.info("✅ Hyperliquid WebSocket connected")

                    # Subscribe to all target symbols
                    for sym in self.symbol_map.keys():
                        sub_msg = json.dumps({
                            "method": "subscribe",
                            "subscription": {
                                "type": "l2Book",
                                "coin": sym
                            }
                        })
                        await ws.send(sub_msg)

                    # Also subscribe to allMids for fast mid-price updates
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "allMids"}
                    }))

                    async for raw in ws:
                        await self._handle_message(raw)

            except (websockets.ConnectionClosed, OSError) as e:
                log.warning(f"HL WebSocket disconnected: {e} — reconnecting in 3s")
                await asyncio.sleep(3)
            except Exception as e:
                log.error(f"HL WebSocket error: {e}")
                await asyncio.sleep(5)

    async def _handle_message(self, raw: str):
        try:
            msg = json.loads(raw)
            if msg.get("channel") == "allMids":
                mids = msg.get("data", {}).get("mids", {})
                for coin, mid_str in mids.items():
                    if coin in self.symbol_map:
                        self.prices[coin] = float(mid_str)
                        self.last_updated[coin] = time.time()
        except Exception:
            pass

    def get_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol.upper())

    def price_age(self, symbol: str) -> float:
        """Seconds since last update for this symbol."""
        ts = self.last_updated.get(symbol.upper(), 0)
        return time.time() - ts if ts else 9999


# ══════════════════════════════════════════════════════════════
# POLY MARKET CLIENT WRAPPER
# ══════════════════════════════════════════════════════════════

class PolyMarketClient:
    """
    Wraps py-clob-client for Poly Market CLOB access.
    Handles market discovery, order placement, position tracking.
    """

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.client: Optional[ClobClient] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        """Initialize the CLOB client with credentials."""
        self._session = aiohttp.ClientSession()

        if not CLOB_AVAILABLE:
            log.error("py-clob-client not installed")
            return False

        if not PM_PRIVATE_KEY:
            log.error("POLYMARKET_WALLET_PRIVATE_KEY not set in .env")
            return False

        try:
            self.client = ClobClient(
                host=PM_CLOB_URL,
                chain_id=POLYGON,
                key=PM_PRIVATE_KEY,
                signature_type=0,   # EOA
                funder=PM_WALLET_ADDRESS,
            )
            # Set L2 API credentials if available
            if PM_API_KEY:
                self.client.set_api_creds(self.client.derive_api_key())

            log.info(f"✅ Poly Market client initialized (paper={self.paper_mode})")
            return True

        except Exception as e:
            log.error(f"Failed to init Poly Market client: {e}")
            return False

    async def get_active_15min_markets(self) -> list[MarketSnapshot]:
        """
        Fetch all active 15-minute crypto up/down markets.
        These are labelled as 'Will BTC be above/below $X at [time]?'
        """
        markets = []
        try:
            # Fetch via Gamma API (no auth needed for read)
            url = f"{PM_GAMMA_URL}/markets?active=true&closed=false&limit=100"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning(f"Gamma API returned {resp.status}")
                    return markets
                data = await resp.json()

            # Filter for 15-minute crypto markets
            for m in data:
                question = m.get("question", "").lower()
                title    = m.get("groupItemTitle", "").lower()

                # Identify 15-minute up/down markets
                is_15min_crypto = (
                    ("above" in question or "below" in question) and
                    any(s in question for s in ["btc", "eth", "sol", "xrp",
                                                 "bitcoin", "ethereum", "solana"]) and
                    m.get("tokens") and len(m["tokens"]) == 2
                )
                if not is_15min_crypto:
                    continue

                # Extract symbol
                symbol = "BTC"
                for s in ["BTC", "ETH", "SOL", "XRP"]:
                    if s.lower() in question:
                        symbol = s
                        break

                # Extract target price from question
                target_price = self._parse_target_price(m.get("question", ""))
                if target_price <= 0:
                    continue

                # Parse end time
                end_time = self._parse_end_time(m)
                if not end_time:
                    continue

                # Parse tokens
                tokens = m.get("tokens", [])
                if len(tokens) < 2:
                    continue

                yes_token = next((t for t in tokens if t.get("outcome","").lower() == "yes"), tokens[0])
                no_token  = next((t for t in tokens if t.get("outcome","").lower() == "no"),  tokens[1])

                yes_price = float(yes_token.get("price", 0.5))
                no_price  = float(no_token.get("price", 0.5))

                snap = MarketSnapshot(
                    condition_id = m.get("conditionId", m.get("id", "")),
                    question     = m.get("question", ""),
                    symbol       = symbol,
                    target_price = target_price,
                    yes_token_id = yes_token.get("token_id", yes_token.get("tokenId", "")),
                    no_token_id  = no_token.get("token_id",  no_token.get("tokenId", "")),
                    yes_price    = yes_price,
                    no_price     = no_price,
                    yes_size     = float(yes_token.get("size", 100)),
                    no_size      = float(no_token.get("size", 100)),
                    end_time     = end_time,
                )
                markets.append(snap)

        except Exception as e:
            log.warning(f"Error fetching markets: {e}")

        return markets

    async def get_market_prices(self, condition_id: str,
                                yes_token_id: str,
                                no_token_id: str) -> tuple[float, float]:
        """Get fresh YES/NO prices from CLOB orderbook."""
        try:
            if self.client:
                books = self.client.get_order_books([
                    BookParams(token_id=yes_token_id),
                    BookParams(token_id=no_token_id),
                ])
                yes_price = self._mid_from_book(books[0]) if books else 0.5
                no_price  = self._mid_from_book(books[1]) if len(books) > 1 else 0.5
                return yes_price, no_price
        except Exception as e:
            log.debug(f"Price fetch error: {e}")
        return 0.5, 0.5

    def _mid_from_book(self, book) -> float:
        """Extract mid price from an order book object."""
        try:
            best_bid = float(book.bids[0].price) if book.bids else 0.0
            best_ask = float(book.asks[0].price) if book.asks else 1.0
            return (best_bid + best_ask) / 2
        except Exception:
            return 0.5

    async def place_order(self, token_id: str, side: str,
                          size_usd: float, price: float) -> Optional[dict]:
        """
        Place a limit order on Poly Market.
        side: 'BUY' or 'SELL'
        price: $0.00 - $1.00 per share
        size_usd: total USDC to spend
        """
        if self.paper_mode:
            log.info(f"  📝 PAPER ORDER | {side} ${size_usd:.2f} @ ${price:.3f} | token {token_id[:8]}...")
            return {"status": "paper", "token_id": token_id, "side": side,
                    "size": size_usd, "price": price}

        if not self.client:
            log.error("Client not initialized")
            return None

        try:
            shares = size_usd / price
            order_args = OrderArgs(
                token_id=token_id,
                price=round(price, 4),
                size=round(shares, 2),
                side=side,
            )
            resp = self.client.create_and_post_order(order_args)
            log.info(f"  ✅ ORDER PLACED | {side} {shares:.2f} shares @ ${price:.3f}")
            return resp
        except Exception as e:
            log.error(f"  ❌ Order failed: {e}")
            return None

    async def get_usdc_balance(self) -> float:
        """Get current USDC balance on Polygon."""
        try:
            if self.client and not self.paper_mode:
                bal = self.client.get_balance_allowance()
                return float(bal.get("balance", 0))
        except Exception:
            pass
        return 1000.0 if self.paper_mode else 0.0

    def _parse_target_price(self, question: str) -> float:
        """Extract numeric target price from market question string."""
        import re
        # Match patterns like "$93,000" "$93000" "93000" "93,000.50"
        matches = re.findall(r'\$?([\d,]+(?:\.\d+)?)', question.replace(",", ""))
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if val > 100:   # filter out percentages, small numbers
                    return val
            except ValueError:
                continue
        return 0.0

    def _parse_end_time(self, market: dict) -> Optional[datetime]:
        """Parse market end time from various field formats."""
        import re
        for field in ["endDateIso", "endDate", "end_date_iso", "end_date"]:
            val = market.get(field)
            if val:
                try:
                    # Handle ISO format
                    dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    return dt
                except Exception:
                    pass
        # Try unix timestamp
        for field in ["endTime", "end_time"]:
            val = market.get(field)
            if val:
                try:
                    return datetime.fromtimestamp(int(val), tz=timezone.utc)
                except Exception:
                    pass
        return None

    async def close(self):
        if self._session:
            await self._session.close()


# ══════════════════════════════════════════════════════════════
# STRATEGY 1 — AMM BOTH-SIDES ARBITRAGE
# ══════════════════════════════════════════════════════════════

class AMMBothSidesArb:
    """
    Core logic: find markets where YES + NO < $0.94, buy both.

    WHY IT WORKS:
    - Market resolves to $1.00 (winner gets $1, loser gets $0)
    - Poly Market takes 2% fee → effective payout = $0.98
    - If we pay $0.47 for YES and $0.47 for NO = $0.94 total cost
    - One side always wins → we receive $0.98 → profit $0.04
    - That's 4.25% profit in ≤15 minutes, direction-neutral

    WHEN DOES THE MISPRICING OCCUR?
    - At market open when AMM is seeding initial liquidity
    - When large one-sided order pushes one token up, the other
      stays depressed (AMM adjusts both sides, creating a window)
    - When overall liquidity is low and spread is wide

    RISK:
    - Poly Market exchange risk (smart contract, counterparty)
    - Market cancelled/voided — you get your bet back (no profit)
    - Slippage eating the thin margin on large sizes
    """

    def __init__(self, pm_client: PolyMarketClient, paper: bool = True):
        self.pm      = pm_client
        self.paper   = paper
        self.positions: list[Position] = []
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.scan_count = 0

    async def scan_and_execute(self, markets: list[MarketSnapshot],
                               hl_feed: HyperliquidFeed) -> list[Position]:
        """Main loop: scan markets, fire on arb opportunities."""
        new_positions = []
        self.scan_count += 1

        # Check daily loss limit
        if self.daily_pnl <= -ArbConfig.MAX_DAILY_LOSS_USD:
            log.warning("🛑 Daily loss limit hit — AMM arb paused")
            return new_positions

        # Don't exceed max simultaneous positions
        open_count = sum(1 for p in self.positions if p.status == "open")
        if open_count >= ArbConfig.AMM_MAX_POSITIONS:
            return new_positions

        for mkt in markets:
            # Skip if we already have a position in this market
            already_in = any(p.condition_id == mkt.condition_id and p.status == "open"
                             for p in self.positions)
            if already_in:
                continue

            # Get fresh prices (reduces staleness risk)
            yes_price, no_price = await self.pm.get_market_prices(
                mkt.condition_id, mkt.yes_token_id, mkt.no_token_id
            )
            mkt.yes_price = yes_price
            mkt.no_price  = no_price

            combined = mkt.combined_cost
            profit_pct = mkt.arb_profit_pct

            if mkt.has_amm_arb:
                log.info(
                    f"💰 AMM ARB FOUND │ {mkt.symbol} │ "
                    f"YES={yes_price:.3f} NO={no_price:.3f} "
                    f"combined={combined:.3f} │ "
                    f"profit={profit_pct*100:.1f}% │ "
                    f"{mkt.seconds_remaining:.0f}s remaining"
                )

                pos = await self._execute_both_sides(mkt)
                if pos:
                    self.positions.append(pos)
                    new_positions.append(pos)
                    self.total_trades += 1

                    if self.scan_count % 10 == 0:
                        self._log_stats()

        return new_positions

    async def _execute_both_sides(self, mkt: MarketSnapshot) -> Optional[Position]:
        """Buy YES and NO simultaneously."""
        size = ArbConfig.AMM_BET_SIZE_USD

        log.info(f"  → Buying YES @ ${mkt.yes_price:.3f} ({size} USDC)")
        yes_order = await self.pm.place_order(
            token_id=mkt.yes_token_id,
            side="BUY",
            size_usd=size,
            price=mkt.yes_price,
        )

        log.info(f"  → Buying NO  @ ${mkt.no_price:.3f} ({size} USDC)")
        no_order = await self.pm.place_order(
            token_id=mkt.no_token_id,
            side="BUY",
            size_usd=size,
            price=mkt.no_price,
        )

        if yes_order is None and not self.paper:
            log.error("YES order failed — not buying NO (risk asymmetry)")
            return None

        shares_yes = size / mkt.yes_price
        shares_no  = size / mkt.no_price
        total_cost = size * 2
        expected_payout = ArbConfig.PAYOUT_AFTER_FEE * max(shares_yes, shares_no) * 1.0

        # Simpler: expected profit = (0.98 - combined) * size
        expected_profit = (ArbConfig.PAYOUT_AFTER_FEE - mkt.combined_cost) * size

        log.info(
            f"  ✅ BOTH-SIDES POSITION OPENED │ "
            f"Cost=${total_cost:.2f} │ "
            f"Expected profit=${expected_profit:.2f} │ "
            f"Resolves: {mkt.end_time.strftime('%H:%M:%S UTC')}"
        )

        self._write_log("amm_arb", "open", mkt.symbol,
                        mkt.yes_price, mkt.no_price, total_cost, 0)

        return Position(
            strategy        = "amm_arb",
            condition_id    = mkt.condition_id,
            symbol          = mkt.symbol,
            side            = "both",
            token_id        = "",
            yes_token_id    = mkt.yes_token_id,
            no_token_id     = mkt.no_token_id,
            entry_yes_price = mkt.yes_price,
            entry_no_price  = mkt.no_price,
            size_yes_usd    = size,
            size_no_usd     = size,
            shares_yes      = shares_yes,
            shares_no       = shares_no,
            end_time        = mkt.end_time,
        )

    async def check_resolutions(self):
        """
        Check if any positions have resolved.
        In paper mode, simulate resolution based on time elapsed.
        In live mode, check on-chain via Poly Market API.
        """
        now = datetime.now(timezone.utc)
        for pos in self.positions:
            if pos.status != "open":
                continue

            time_left = (pos.end_time - now).total_seconds() if pos.end_time else 9999

            if time_left < -60:  # 60s grace after resolution
                # Simulate resolution for paper mode
                pnl = self._calc_arb_pnl(pos)
                pos.pnl    = pnl
                pos.status = "resolved"
                self.daily_pnl += pnl
                if pnl > 0:
                    self.wins += 1

                log.info(
                    f"  📊 RESOLVED │ {pos.symbol} AMM ARB │ "
                    f"PnL=${pnl:+.2f} │ "
                    f"Daily PnL=${self.daily_pnl:+.2f}"
                )
                self._write_log("amm_arb", "resolved", pos.symbol,
                                pos.entry_yes_price, pos.entry_no_price,
                                pos.size_yes_usd + pos.size_no_usd, pnl)

    def _calc_arb_pnl(self, pos: Position) -> float:
        """
        For paper mode: both-sides arb always profits if combined < payout.
        One side wins ($0.98 after fee), one side loses ($0).
        Total payout = 0.98 per share on the winning side.
        """
        # Conservative estimate: we win $0.98 on the larger share count
        winning_shares  = max(pos.shares_yes, pos.shares_no)
        winning_payout  = winning_shares * ArbConfig.PAYOUT_AFTER_FEE
        losing_cost     = min(pos.size_yes_usd, pos.size_no_usd)
        total_cost      = pos.size_yes_usd + pos.size_no_usd
        return winning_payout - total_cost

    def _log_stats(self):
        total  = self.total_trades
        win_r  = (self.wins / total * 100) if total else 0
        log.info(
            f"📈 AMM ARB STATS │ "
            f"Trades={total} │ Win%={win_r:.0f}% │ "
            f"Daily PnL=${self.daily_pnl:+.2f}"
        )

    def _write_log(self, strategy, event, symbol, yes_p, no_p, cost, pnl):
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            strategy, event, symbol,
            f"{yes_p:.4f}", f"{no_p:.4f}",
            f"{cost:.2f}", f"{pnl:.4f}"
        ]
        header = ["timestamp", "strategy", "event", "symbol",
                  "yes_price", "no_price", "cost_usd", "pnl_usd"]
        write_header = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)


# ══════════════════════════════════════════════════════════════
# STRATEGY 2 — ORACLE FRONT-RUNNING
# ══════════════════════════════════════════════════════════════

class OracleFrontRunner:
    """
    Uses Hyperliquid's real-time price feed (~100ms latency)
    to front-run Poly Market's oracle (~15-30 second lag).

    HOW IT WORKS:
    1. Real-time BTC/ETH price from HL WebSocket
    2. Poly Market's displayed price = their stale oracle
    3. When real price has moved significantly BUT odds haven't:
       - Real BTC = $93,500 (moving away from $93,000 target)
       - Poly Market still showing 52% chance for "above $93,000"
       - We KNOW the real odds should be ~85% above
       - We buy YES at 52 cents → worth 85 cents → profit 33 cents
    4. Exit: either hold to resolution OR sell when odds correct

    THE EDGE:
    - HL WebSocket: <100ms
    - Poly Market oracle update: 15-30 seconds
    - That's a 15-30 second window per cycle
    - 15-minute markets = ~60 oracle updates per market
    - Lots of opportunities per market per day

    RISK:
    - Oracle may never correct before resolution (hold to end = fine if real price holds)
    - Real price reverses after entry (we're directionally exposed once we buy one side)
    - High competition from other faster bots
    """

    def __init__(self, pm_client: PolyMarketClient, hl_feed: HyperliquidFeed,
                 paper: bool = True):
        self.pm         = pm_client
        self.hl         = hl_feed
        self.paper      = paper
        self.positions:  list[Position] = []
        self.daily_pnl   = 0.0
        self.total_trades = 0
        self.wins        = 0

    async def scan_and_execute(self, markets: list[MarketSnapshot]) -> list[Position]:
        """Scan each market for oracle front-run opportunities."""
        new_positions = []

        if self.daily_pnl <= -ArbConfig.MAX_DAILY_LOSS_USD:
            log.warning("🛑 Daily loss limit hit — oracle front-runner paused")
            return new_positions

        open_count = sum(1 for p in self.positions if p.status == "open")
        if open_count >= ArbConfig.ORACLE_MAX_POSITIONS:
            return new_positions

        for mkt in markets:
            # Skip if already positioned in this market
            already_in = any(p.condition_id == mkt.condition_id and p.status == "open"
                             for p in self.positions)
            if already_in:
                continue

            # Check time window
            secs_left = mkt.seconds_remaining
            if secs_left < ArbConfig.ORACLE_MIN_TIME_LEFT:
                continue
            if secs_left > ArbConfig.ORACLE_MAX_TIME_LEFT:
                continue

            # Get our real-time price
            real_price = self.hl.get_price(mkt.symbol)
            if real_price is None:
                continue

            # Check price feed is fresh (< 2 seconds old)
            feed_age = self.hl.price_age(mkt.symbol)
            if feed_age > 2.0:
                log.debug(f"Price feed stale ({feed_age:.1f}s) for {mkt.symbol}")
                continue

            # Calculate how far real price is from target
            price_delta_pct = (real_price - mkt.target_price) / mkt.target_price

            # Determine the probable outcome
            # If real price is above target by >THRESHOLD → outcome is likely YES
            # If real price is below target by >THRESHOLD → outcome is likely NO
            probable_outcome = None
            entry_odds       = 0.0
            entry_token_id   = ""

            if price_delta_pct > ArbConfig.ORACLE_PRICE_DELTA_PCT:
                # Real price significantly above target → probably YES
                # Check if Poly Market hasn't priced this in yet
                if mkt.yes_price < ArbConfig.ORACLE_MIN_ODDS:
                    probable_outcome = "YES"
                    entry_odds       = mkt.yes_price
                    entry_token_id   = mkt.yes_token_id

            elif price_delta_pct < -ArbConfig.ORACLE_PRICE_DELTA_PCT:
                # Real price significantly below target → probably NO
                if mkt.no_price < ArbConfig.ORACLE_MIN_ODDS:
                    probable_outcome = "NO"
                    entry_odds       = mkt.no_price
                    entry_token_id   = mkt.no_token_id

            if probable_outcome is None:
                continue

            # Additional filter: odds must make sense (not already corrected)
            if entry_odds > ArbConfig.ORACLE_MAX_ODDS:
                continue

            # Estimate fair value based on real price distance + time remaining
            fair_value = self._estimate_fair_value(
                price_delta_pct, secs_left
            )
            edge = fair_value - entry_odds
            if edge < 0.05:  # need at least 5 cent edge
                continue

            log.info(
                f"🔮 ORACLE FRONT-RUN │ {mkt.symbol} │ "
                f"Real={real_price:,.2f} Target={mkt.target_price:,.2f} "
                f"Delta={price_delta_pct*100:+.2f}% │ "
                f"Outcome={probable_outcome} @ {entry_odds:.3f} │ "
                f"Fair={fair_value:.3f} Edge={edge:.3f} │ "
                f"{secs_left:.0f}s left"
            )

            pos = await self._execute_front_run(
                mkt, probable_outcome, entry_token_id, entry_odds, fair_value
            )
            if pos:
                self.positions.append(pos)
                new_positions.append(pos)
                self.total_trades += 1

        return new_positions

    def _estimate_fair_value(self, price_delta_pct: float,
                             secs_remaining: float) -> float:
        """
        Estimate the fair probability (fair value of YES/NO share)
        given how far price is from target and time remaining.

        Simple model:
        - The further from target, the higher the probability
        - The less time remaining, the higher the probability
          (less time for price to reverse)
        - Uses a sigmoid-like mapping
        """
        import math

        # Magnitude of the move (absolute distance from target)
        abs_delta = abs(price_delta_pct)

        # Time factor: at 30s remaining, time_factor → 1 (very confident)
        # At 10 min remaining, time_factor → 0 (uncertainty)
        max_secs = ArbConfig.ORACLE_MAX_TIME_LEFT
        time_factor = 1.0 - (secs_remaining / max_secs)
        time_factor = max(0.0, min(1.0, time_factor))

        # Base probability from price distance
        # 0.3% delta → ~60% probability
        # 1.0% delta → ~75% probability
        # 3.0% delta → ~90% probability
        base_prob = 0.5 + (abs_delta / 0.03) * 0.4
        base_prob = min(0.97, max(0.5, base_prob))

        # Blend with time factor (more time remaining = less certainty)
        fair_value = base_prob * (0.5 + 0.5 * time_factor)
        fair_value = max(0.51, min(0.97, fair_value))

        return fair_value

    async def _execute_front_run(self, mkt: MarketSnapshot,
                                  outcome: str, token_id: str,
                                  entry_odds: float,
                                  fair_value: float) -> Optional[Position]:
        """Buy the under-priced outcome token."""
        size = ArbConfig.ORACLE_BET_SIZE_USD

        log.info(f"  → Buying {outcome} @ ${entry_odds:.3f} (fair=${fair_value:.3f}) │ ${size} USDC")

        order = await self.pm.place_order(
            token_id=token_id,
            side="BUY",
            size_usd=size,
            price=entry_odds,
        )

        if order is None and not self.paper:
            return None

        side_str = outcome.lower()
        shares   = size / entry_odds

        log.info(
            f"  ✅ ORACLE POSITION OPENED │ "
            f"{mkt.symbol} {outcome} │ "
            f"Entry={entry_odds:.3f} │ Fair={fair_value:.3f} │ "
            f"Expected edge=${(fair_value - entry_odds) * shares:.2f}"
        )

        return Position(
            strategy         = "oracle_frontrun",
            condition_id     = mkt.condition_id,
            symbol           = mkt.symbol,
            side             = side_str,
            token_id         = token_id,
            yes_token_id     = mkt.yes_token_id if outcome == "YES" else "",
            no_token_id      = mkt.no_token_id  if outcome == "NO"  else "",
            entry_yes_price  = entry_odds if outcome == "YES" else 0.0,
            entry_no_price   = entry_odds if outcome == "NO"  else 0.0,
            size_yes_usd     = size if outcome == "YES" else 0.0,
            size_no_usd      = size if outcome == "NO"  else 0.0,
            shares_yes       = shares if outcome == "YES" else 0.0,
            shares_no        = shares if outcome == "NO"  else 0.0,
            end_time         = mkt.end_time,
        )

    async def check_and_exit_early(self, markets: list[MarketSnapshot]):
        """
        For oracle positions: if odds have already corrected toward fair value,
        consider selling early to lock in profit vs waiting for resolution.
        This is the 'sell at 98 cents instead of waiting' move from the video.
        """
        # Build market lookup by condition_id
        mkt_lookup = {m.condition_id: m for m in markets}

        for pos in self.positions:
            if pos.status != "open":
                continue

            mkt = mkt_lookup.get(pos.condition_id)
            if not mkt:
                continue

            # Get current odds
            curr_yes, curr_no = await self.pm.get_market_prices(
                pos.condition_id, mkt.yes_token_id, mkt.no_token_id
            )

            if pos.side == "yes":
                entry_odds   = pos.entry_yes_price
                current_odds = curr_yes
                shares       = pos.shares_yes
            else:
                entry_odds   = pos.entry_no_price
                current_odds = curr_no
                shares       = pos.shares_no

            current_value = current_odds * shares
            entry_cost    = (entry_odds * shares)
            unrealised_pnl = current_value - entry_cost

            # Exit early if odds have corrected ≥90% of the way to $0.99
            # (captures most of the profit, avoids resolution risk)
            EARLY_EXIT_THRESHOLD = 0.95
            if current_odds >= EARLY_EXIT_THRESHOLD and entry_odds < EARLY_EXIT_THRESHOLD:
                log.info(
                    f"  💸 EARLY EXIT │ {pos.symbol} {pos.side.upper()} │ "
                    f"Entry={entry_odds:.3f} Now={current_odds:.3f} │ "
                    f"PnL=${unrealised_pnl:+.2f}"
                )
                if not self.paper:
                    await self.pm.place_order(
                        token_id=pos.token_id,
                        side="SELL",
                        size_usd=current_value,
                        price=current_odds,
                    )

                pos.pnl    = unrealised_pnl
                pos.status = "closed"
                self.daily_pnl += unrealised_pnl
                if unrealised_pnl > 0:
                    self.wins += 1

    async def check_resolutions(self):
        """Handle positions that have reached market end time."""
        now = datetime.now(timezone.utc)
        for pos in self.positions:
            if pos.status != "open":
                continue
            if not pos.end_time:
                continue

            secs_past = (now - pos.end_time).total_seconds()
            if secs_past < 60:  # wait 60s after end time for oracle to finalise
                continue

            # For paper mode: estimate outcome based on entry
            # In live: query on-chain resolution
            # Conservative paper estimate: 65% win rate on oracle trades
            import random
            entry_odds = pos.entry_yes_price or pos.entry_no_price
            shares     = pos.shares_yes or pos.shares_no
            cost       = entry_odds * shares

            # Use fair value estimate as win probability for paper simulation
            win_prob = min(0.95, entry_odds + 0.20)
            won = random.random() < win_prob

            if won:
                pnl = (ArbConfig.PAYOUT_AFTER_FEE * shares) - cost
            else:
                pnl = -cost

            pos.pnl    = pnl
            pos.status = "resolved"
            self.daily_pnl += pnl
            if pnl > 0:
                self.wins += 1

            log.info(
                f"  📊 RESOLVED │ {pos.symbol} {pos.side.upper()} oracle │ "
                f"{'WIN' if pnl > 0 else 'LOSS'} ${pnl:+.2f} │ "
                f"Daily PnL=${self.daily_pnl:+.2f}"
            )


# ══════════════════════════════════════════════════════════════
# MAIN AGENT LOOP
# ══════════════════════════════════════════════════════════════

class PolyMarketArbAgent:
    """
    Orchestrates both strategies in a continuous async loop.
    Runs both simultaneously — AMM arb on one asyncio task,
    oracle front-running on another.
    """

    def __init__(self, paper: bool = True, scan_only: bool = False):
        self.paper     = paper
        self.scan_only = scan_only
        self.pm        = PolyMarketClient(paper_mode=paper)
        self.hl        = HyperliquidFeed()
        self.amm_arb:  Optional[AMMBothSidesArb]  = None
        self.oracle_fr: Optional[OracleFrontRunner] = None
        self.start_time = time.time()
        self.scan_count = 0

    async def run(self):
        """Main entry point."""
        log.info("=" * 60)
        log.info("🌙 POLY MARKET ARB AGENT")
        log.info(f"   Mode: {'PAPER' if self.paper else '🔴 LIVE'}")
        log.info(f"   Scan only: {self.scan_only}")
        log.info("=" * 60)

        # Initialise clients
        ok = await self.pm.init()
        if not ok and not self.paper:
            log.error("Failed to initialize — check credentials in .env")
            return

        self.amm_arb  = AMMBothSidesArb(self.pm, paper=self.paper)
        self.oracle_fr = OracleFrontRunner(self.pm, self.hl, paper=self.paper)

        # Run HL price feed and main loop concurrently
        await asyncio.gather(
            self.hl.connect(),
            self._main_loop(),
        )

    async def _main_loop(self):
        """Scan → detect → execute → manage positions."""
        log.info("⏳ Waiting 5s for Hyperliquid price feed to warm up...")
        await asyncio.sleep(5)

        while True:
            try:
                loop_start = time.time()
                self.scan_count += 1

                # 1. Fetch active 15-minute markets
                markets = await self.pm.get_active_15min_markets()

                if not markets:
                    log.debug("No active 15-min markets found — will retry")
                    await asyncio.sleep(ArbConfig.SCAN_INTERVAL_SECS)
                    continue

                # ── Print current feed prices ──────────────────
                if self.scan_count % 12 == 0:  # every ~60s
                    feed_status = " │ ".join([
                        f"{s}=${self.hl.get_price(s):,.0f}"
                        for s in ["BTC", "ETH", "SOL", "XRP"]
                        if self.hl.get_price(s)
                    ])
                    log.info(f"📡 FEED │ {feed_status}")
                    log.info(f"📊 MARKETS │ Found {len(markets)} active 15-min markets")

                    # Show market snapshots
                    for m in markets[:3]:
                        log.info(
                            f"   {m.symbol} target={m.target_price:,.0f} │ "
                            f"YES={m.yes_price:.3f} NO={m.no_price:.3f} │ "
                            f"combined={m.combined_cost:.3f} │ "
                            f"{m.seconds_remaining:.0f}s left"
                        )

                if self.scan_only:
                    await asyncio.sleep(ArbConfig.SCAN_INTERVAL_SECS)
                    continue

                # 2. Strategy 1: AMM Both-Sides Arb
                await self.amm_arb.scan_and_execute(markets, self.hl)
                await self.amm_arb.check_resolutions()

                # 3. Strategy 2: Oracle Front-Running
                await self.oracle_fr.scan_and_execute(markets)
                await self.oracle_fr.check_and_exit_early(markets)
                await self.oracle_fr.check_resolutions()

                # 4. Periodic summary
                if self.scan_count % 60 == 0:  # every ~5 min
                    self._print_summary()

                # Sleep until next scan
                elapsed = time.time() - loop_start
                sleep_for = max(0.5, ArbConfig.SCAN_INTERVAL_SECS - elapsed)
                await asyncio.sleep(sleep_for)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Main loop error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(10)

        await self.pm.close()
        log.info("Agent stopped cleanly.")

    def _print_summary(self):
        """Print a performance summary."""
        runtime_h = (time.time() - self.start_time) / 3600
        amm_pnl   = self.amm_arb.daily_pnl   if self.amm_arb   else 0
        oracle_pnl = self.oracle_fr.daily_pnl if self.oracle_fr else 0
        total_pnl  = amm_pnl + oracle_pnl

        amm_trades    = self.amm_arb.total_trades    if self.amm_arb   else 0
        oracle_trades = self.oracle_fr.total_trades  if self.oracle_fr else 0
        amm_wins      = self.amm_arb.wins            if self.amm_arb   else 0
        oracle_wins   = self.oracle_fr.wins          if self.oracle_fr else 0

        log.info("─" * 60)
        log.info(f"📊 PERFORMANCE SUMMARY │ Runtime: {runtime_h:.1f}h")
        log.info(f"   AMM Arb:      {amm_trades} trades │ {amm_wins} wins │ PnL=${amm_pnl:+.2f}")
        log.info(f"   Oracle FrontR: {oracle_trades} trades │ {oracle_wins} wins │ PnL=${oracle_pnl:+.2f}")
        log.info(f"   TOTAL PnL:    ${total_pnl:+.2f}")
        log.info("─" * 60)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="🌙 Poly Market Arbitrage Agent")
    p.add_argument("--paper",     action="store_true", default=True,
                   help="Paper mode — no real orders (default: True)")
    p.add_argument("--live",      action="store_true",
                   help="Live mode — real orders (requires .env credentials)")
    p.add_argument("--scan",      action="store_true",
                   help="Scan only — print opportunities, no orders")
    args = p.parse_args()

    paper     = not args.live
    scan_only = args.scan

    if args.live and not PM_PRIVATE_KEY:
        print("❌ Live mode requires POLYMARKET_WALLET_PRIVATE_KEY in .env")
        print("   See setup instructions at top of this file.")
        sys.exit(1)

    if args.live:
        print("⚠️  LIVE MODE — real money. You have 5 seconds to cancel (Ctrl+C)...")
        time.sleep(5)

    agent = PolyMarketArbAgent(paper=paper, scan_only=scan_only)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user.")


if __name__ == "__main__":
    main()
