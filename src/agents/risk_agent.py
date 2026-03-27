# ============================================================
# 🌙 Risk Agent
# Monitors portfolio risk and blocks trades that violate limits.
# Import this in any other agent before placing trades.
# ============================================================

import sys
import json
import csv
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    MAX_POSITION_SIZE_USD,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_DAILY_LOSS_USD,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)
from src.exchanges.router import get_balance, get_positions

RISK_LOG = Path(__file__).resolve().parents[2] / "src" / "data" / "risk_log.csv"
RISK_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log(event: str, detail: str):
    ts = datetime.now().isoformat()
    row = [ts, event, detail]
    write_header = not RISK_LOG.exists()
    with open(RISK_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "event", "detail"])
        w.writerow(row)
    print(f"  📋 Risk log: [{event}] {detail}")


class RiskAgent:
    """
    Call check_trade() before every order.
    Returns (True, "") if trade is allowed, or (False, reason) if blocked.
    """

    def __init__(self):
        self.daily_loss_tracker: dict[str, float] = {}  # date → realised loss USD
        print("🛡️  Risk Agent initialised")
        print(f"   Max position : ${MAX_POSITION_SIZE_USD}")
        print(f"   Max daily loss: ${MAX_DAILY_LOSS_USD}")
        print(f"   Stop loss    : {STOP_LOSS_PCT*100:.1f}%")
        print(f"   Take profit  : {TAKE_PROFIT_PCT*100:.1f}%")

    # ── public interface ─────────────────────────────────────

    def check_trade(self, symbol: str, usd_amount: float,
                    direction: str = "buy") -> tuple[bool, str]:
        """
        Run all risk checks before a trade.
        direction: 'buy' | 'sell'
        Returns (allowed: bool, reason: str)
        """
        checks = [
            self._check_position_size(usd_amount),
            self._check_daily_loss(),
            self._check_portfolio_risk(usd_amount),
            self._check_existing_position(symbol, direction),
        ]
        for allowed, reason in checks:
            if not allowed:
                _log("BLOCKED", f"{direction.upper()} {symbol} ${usd_amount:.2f} — {reason}")
                return False, reason

        _log("APPROVED", f"{direction.upper()} {symbol} ${usd_amount:.2f}")
        return True, ""

    def record_loss(self, usd_loss: float):
        """Call this when a trade closes at a loss."""
        today = str(date.today())
        self.daily_loss_tracker[today] = (
            self.daily_loss_tracker.get(today, 0.0) + abs(usd_loss)
        )
        _log("LOSS", f"Recorded ${usd_loss:.2f} loss. Today total: ${self.daily_loss_tracker[today]:.2f}")

    def stop_loss_price(self, entry_price: float, direction: str = "long") -> float:
        """Calculate stop loss price from entry."""
        if direction == "long":
            return round(entry_price * (1 - STOP_LOSS_PCT), 6)
        return round(entry_price * (1 + STOP_LOSS_PCT), 6)

    def take_profit_price(self, entry_price: float, direction: str = "long") -> float:
        """Calculate take profit price from entry."""
        if direction == "long":
            return round(entry_price * (1 + TAKE_PROFIT_PCT), 6)
        return round(entry_price * (1 - TAKE_PROFIT_PCT), 6)

    def portfolio_summary(self) -> dict:
        """Print and return a summary of current risk exposure."""
        try:
            balance   = get_balance()
            positions = get_positions()
            account_value = balance.get("account_value", 0)
            total_exposure = sum(
                abs(p.get("size", 0)) * p.get("entry_price", 0)
                for p in positions
            )
            total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)

            summary = {
                "account_value_usd": round(account_value, 2),
                "open_positions":    len(positions),
                "total_exposure":    round(total_exposure, 2),
                "unrealized_pnl":    round(total_pnl, 2),
                "exposure_pct":      round(total_exposure / account_value * 100, 1)
                                     if account_value > 0 else 0,
                "positions": positions,
            }

            print("\n  📊 Portfolio Summary")
            print(f"     Account value : ${summary['account_value_usd']:,.2f}")
            print(f"     Open positions: {summary['open_positions']}")
            print(f"     Total exposure: ${summary['total_exposure']:,.2f} ({summary['exposure_pct']}%)")
            print(f"     Unrealized PnL: ${summary['unrealized_pnl']:+,.2f}")
            return summary

        except Exception as e:
            print(f"  ❌ Could not fetch portfolio summary: {e}")
            return {}

    # ── private checks ────────────────────────────────────────

    def _check_position_size(self, usd_amount: float) -> tuple[bool, str]:
        if usd_amount > MAX_POSITION_SIZE_USD:
            return False, f"Trade size ${usd_amount:.2f} exceeds max ${MAX_POSITION_SIZE_USD}"
        return True, ""

    def _check_daily_loss(self) -> tuple[bool, str]:
        today      = str(date.today())
        daily_loss = self.daily_loss_tracker.get(today, 0.0)
        if daily_loss >= MAX_DAILY_LOSS_USD:
            return False, f"Daily loss limit hit: ${daily_loss:.2f} / ${MAX_DAILY_LOSS_USD}"
        return True, ""

    def _check_portfolio_risk(self, usd_amount: float) -> tuple[bool, str]:
        try:
            balance       = get_balance()
            account_value = balance.get("account_value", 0)
            if account_value <= 0:
                return True, ""   # can't check, allow
            risk_pct = usd_amount / account_value
            if risk_pct > MAX_PORTFOLIO_RISK_PCT:
                return False, (
                    f"Trade is {risk_pct*100:.1f}% of portfolio, "
                    f"max allowed {MAX_PORTFOLIO_RISK_PCT*100:.1f}%"
                )
        except Exception:
            pass   # if exchange is unreachable, don't block
        return True, ""

    def _check_existing_position(self, symbol: str,
                                  direction: str) -> tuple[bool, str]:
        """Block opening the same direction twice."""
        try:
            positions = get_positions()
            for pos in positions:
                if pos.get("symbol") == symbol:
                    existing_side = "buy" if pos.get("size", 0) > 0 else "sell"
                    if existing_side == direction:
                        return False, f"Already have a {direction} position in {symbol}"
        except Exception:
            pass
        return True, ""


# Singleton
risk = RiskAgent()
