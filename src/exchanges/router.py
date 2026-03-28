# ============================================================
# 🌙 Exchange Router
# Single interface regardless of which exchange you're using.
# Set EXCHANGE in src/config.py to switch between them.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import EXCHANGE, HYPERLIQUID_TOKENS, COINBASE_TOKENS


def get_price(symbol: str) -> float:
    """Get current market price for a symbol."""
    if EXCHANGE == "tradovate":
        from src.exchanges.tradovate import get_price as tv_price
        return tv_price(symbol)
    elif EXCHANGE == "hyperliquid":
        from src.exchanges.hyperliquid import get_price as hl_price
        return hl_price(symbol)
    else:
        from src.exchanges.coinbase import get_price as cb_price
        # Coinbase uses BTC-USD format; if bare symbol given, add -USD
        sym = symbol if "-" in symbol else f"{symbol}-USD"
        return cb_price(sym)


def get_balance() -> dict:
    """Get account balance."""
    if EXCHANGE == "tradovate":
        from src.exchanges.tradovate import get_price as tv_price
        return tv_price(symbol)
    elif EXCHANGE == "hyperliquid":
        import os
        from src.exchanges.hyperliquid import get_balance as hl_bal
        wallet = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
        return hl_bal(wallet)
    else:
        from src.exchanges.coinbase import get_balance as cb_bal
        usd = cb_bal("USD")
        return {"account_value": usd, "withdrawable": usd}


def get_positions() -> list[dict]:
    """Get all open positions."""
    if EXCHANGE == "tradovate":
        from src.exchanges.tradovate import get_price as tv_price
        return tv_price(symbol)
    elif EXCHANGE == "hyperliquid":
        import os
        from src.exchanges.hyperliquid import get_positions as hl_pos
        wallet = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
        return hl_pos(wallet)
    else:
        from src.exchanges.coinbase import get_positions as cb_pos
        return cb_pos()


def buy(symbol: str, usd_amount: float) -> dict:
    """Open a long / buy position."""
    if EXCHANGE == "tradovate":
        from src.exchanges.tradovate import get_price as tv_price
        return tv_price(symbol)
    elif EXCHANGE == "hyperliquid":
        from src.exchanges.hyperliquid import market_buy
        return market_buy(symbol, usd_amount)
    else:
        from src.exchanges.coinbase import market_buy
        sym = symbol if "-" in symbol else f"{symbol}-USD"
        return market_buy(sym, usd_amount)


def sell(symbol: str, usd_amount: float) -> dict:
    """Open a short / sell position."""
    if EXCHANGE == "tradovate":
        from src.exchanges.tradovate import get_price as tv_price
        return tv_price(symbol)
    elif EXCHANGE == "hyperliquid":
        from src.exchanges.hyperliquid import market_sell
        return market_sell(symbol, usd_amount)
    else:
        from src.exchanges.coinbase import get_price, market_sell
        sym       = symbol if "-" in symbol else f"{symbol}-USD"
        price     = get_price(sym)
        base_size = usd_amount / price
        return market_sell(sym, base_size)


def close(symbol: str) -> dict:
    """Close / exit a position."""
    if EXCHANGE == "tradovate":
        from src.exchanges.tradovate import get_price as tv_price
        return tv_price(symbol)
    elif EXCHANGE == "hyperliquid":
        from src.exchanges.hyperliquid import close_position
        return close_position(symbol)
    else:
        # For Coinbase spot: sell all held base currency
        from src.exchanges.coinbase import get_balance, market_sell
        sym  = symbol if "-" in symbol else f"{symbol}-USD"
        base = sym.split("-")[0]
        held = get_balance(base)
        if held > 0:
            return market_sell(sym, held)
        return {}


def active_symbols() -> list[str]:
    """Return the token list for the configured exchange."""
    return HYPERLIQUID_TOKENS if EXCHANGE == "hyperliquid" else COINBASE_TOKENS
