# ============================================================
# 🌙 Prop Firm Monitor Agent
#
# Tracks your progress through prop firm evaluations and
# funded accounts in real time. Monitors:
#
#   - Daily loss vs firm's daily loss limit
#   - Total drawdown vs firm's max drawdown
#   - Profit target progress
#   - Minimum trading days completed
#   - Auto-pauses Algotec if approaching breach limits
#
# SUPPORTED FIRMS:
#   Apex Trader Funding  (futures — Tradovate)
#   Topstep              (futures — Tradovate)
#   FTMO                 (forex/indices)
#   FundedNext           (crypto/forex)
#   Custom               (define your own rules)
#
# HOW TO RUN:
#   python src/agents/prop_firm_monitor.py
#   python src/agents/prop_firm_monitor.py --firm apex --account 50k
# ============================================================

import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT     = Path(__file__).resolve().parents[2]
PROP_LOG      = REPO_ROOT / "src" / "data" / "prop_firm_log.csv"
PROP_STATE    = REPO_ROOT / "src" / "data" / "prop_firm_state.json"
PAUSE_FILE    = REPO_ROOT / "src" / "data" / "trading_paused.json"
PROP_LOG.parent.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# PROP FIRM PROFILES
# All dollar values scale with account size
# ════════════════════════════════════════════════════════════════

PROP_FIRM_PROFILES = {

    "apex": {
        "name":              "Apex Trader Funding",
        "type":              "futures",
        "exchange":          "tradovate",
        "accounts": {
            "25k":  {"size": 25_000,  "daily_loss": 1_000, "max_dd": 1_500,  "profit_target": 1_500,  "min_days": 7},
            "50k":  {"size": 50_000,  "daily_loss": 2_000, "max_dd": 2_500,  "profit_target": 3_000,  "min_days": 7},
            "75k":  {"size": 75_000,  "daily_loss": 2_500, "max_dd": 3_500,  "profit_target": 4_250,  "min_days": 7},
            "100k": {"size": 100_000, "daily_loss": 3_000, "max_dd": 4_500,  "profit_target": 6_000,  "min_days": 7},
            "150k": {"size": 150_000, "daily_loss": 4_500, "max_dd": 6_500,  "profit_target": 9_000,  "min_days": 7},
            "300k": {"size": 300_000, "daily_loss": 9_000, "max_dd": 12_500, "profit_target": 18_000, "min_days": 7},
        },
        "profit_split":      0.90,
        "trailing_drawdown": True,    # drawdown trails your peak
        "news_trading":      True,    # allowed
        "overnight":         True,    # positions can be held overnight
        "weekend":           False,   # no weekend holding
        "platforms":         ["tradovate", "rithmic"],
        "evaluation_fee":    {"25k": 167, "50k": 187, "100k": 207},
        "website":           "apextraderfunding.com",
        "notes":             "Best for futures algos. 90% payout. Unlimited bots allowed.",
    },

    "topstep": {
        "name":              "Topstep",
        "type":              "futures",
        "exchange":          "tradovate",
        "accounts": {
            "50k":  {"size": 50_000,  "daily_loss": 1_000, "max_dd": 2_000,  "profit_target": 3_000,  "min_days": 5},
            "100k": {"size": 100_000, "daily_loss": 2_000, "max_dd": 3_000,  "profit_target": 6_000,  "min_days": 5},
            "150k": {"size": 150_000, "daily_loss": 3_000, "max_dd": 4_500,  "profit_target": 9_000,  "min_days": 5},
        },
        "profit_split":      0.90,
        "trailing_drawdown": False,
        "news_trading":      True,
        "overnight":         True,
        "weekend":           False,
        "platforms":         ["tradovate", "ninjatrader"],
        "evaluation_fee":    {"50k": 165, "100k": 325, "150k": 375},
        "website":           "topstep.com",
        "notes":             "Strong reputation. Good coaching tools. Strict drawdown.",
    },

    "ftmo": {
        "name":              "FTMO",
        "type":              "forex",
        "exchange":          "mt5",
        "accounts": {
            "10k":  {"size": 10_000,  "daily_loss": 500,   "max_dd": 1_000,  "profit_target": 1_000,  "min_days": 4},
            "25k":  {"size": 25_000,  "daily_loss": 1_250, "max_dd": 2_500,  "profit_target": 2_500,  "min_days": 4},
            "50k":  {"size": 50_000,  "daily_loss": 2_500, "max_dd": 5_000,  "profit_target": 5_000,  "min_days": 4},
            "100k": {"size": 100_000, "daily_loss": 5_000, "max_dd": 10_000, "profit_target": 10_000, "min_days": 4},
            "200k": {"size": 200_000, "daily_loss": 10_000,"max_dd": 20_000, "profit_target": 20_000, "min_days": 4},
        },
        "profit_split":      0.90,
        "trailing_drawdown": False,
        "news_trading":      False,   # restricted during news
        "overnight":         True,
        "weekend":           True,
        "platforms":         ["mt4", "mt5", "ctrader"],
        "evaluation_fee":    {"10k": 155, "25k": 250, "50k": 345, "100k": 540},
        "website":           "ftmo.com",
        "notes":             "Industry standard. EAs allowed. No news trading.",
    },

    "fundednext": {
        "name":              "FundedNext",
        "type":              "forex",
        "exchange":          "mt5",
        "accounts": {
            "6k":   {"size": 6_000,   "daily_loss": 300,   "max_dd": 600,   "profit_target": 600,   "min_days": 5},
            "15k":  {"size": 15_000,  "daily_loss": 750,   "max_dd": 1_500, "profit_target": 1_500, "min_days": 5},
            "25k":  {"size": 25_000,  "daily_loss": 1_250, "max_dd": 2_500, "profit_target": 2_500, "min_days": 5},
            "50k":  {"size": 50_000,  "daily_loss": 2_500, "max_dd": 5_000, "profit_target": 5_000, "min_days": 5},
            "100k": {"size": 100_000, "daily_loss": 5_000, "max_dd": 10_000,"profit_target": 10_000,"min_days": 5},
            "200k": {"size": 200_000, "daily_loss": 10_000,"max_dd": 20_000,"profit_target": 20_000,"min_days": 5},
        },
        "profit_split":      0.95,
        "trailing_drawdown": False,
        "news_trading":      True,
        "overnight":         True,
        "weekend":           True,
        "platforms":         ["mt5", "ctrader"],
        "evaluation_fee":    {"15k": 99, "25k": 149, "50k": 249, "100k": 449},
        "website":           "fundednext.com",
        "notes":             "Highest profit split (95%). Crypto + forex. EAs allowed.",
    },

    "custom": {
        "name":              "Custom Prop Firm",
        "type":              "custom",
        "exchange":          "any",
        "accounts": {
            "default": {
                "size":           50_000,
                "daily_loss":     2_000,
                "max_dd":         4_000,
                "profit_target":  5_000,
                "min_days":       5,
            }
        },
        "profit_split":      0.80,
        "trailing_drawdown": False,
        "notes":             "Edit this profile to match your firm's rules.",
    },
}


# ════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if PROP_STATE.exists():
        return json.loads(PROP_STATE.read_text())
    return {
        "firm":           None,
        "account_size":   None,
        "phase":          "evaluation",   # evaluation | funded
        "start_date":     None,
        "trading_days":   [],
        "starting_balance": 0,
        "peak_balance":   0,
        "current_balance": 0,
        "daily_pnl":      {},
        "total_pnl":      0,
    }


def save_state(state: dict):
    PROP_STATE.write_text(json.dumps(state, indent=2))


def reset_state(firm: str, account: str, starting_balance: float):
    state = {
        "firm":             firm,
        "account_size":     account,
        "phase":            "evaluation",
        "start_date":       date.today().isoformat(),
        "trading_days":     [],
        "starting_balance": starting_balance,
        "peak_balance":     starting_balance,
        "current_balance":  starting_balance,
        "daily_pnl":        {},
        "total_pnl":        0,
    }
    save_state(state)
    print(f"  ✅ State reset for {firm.upper()} {account} account")
    return state


# ════════════════════════════════════════════════════════════════
# BALANCE FETCHING
# ════════════════════════════════════════════════════════════════

def get_current_balance(exchange: str) -> float:
    """Fetch current account balance from broker."""
    try:
        if exchange == "tradovate":
            from src.exchanges.tradovate import get_balance
            data = get_balance()
            return float(data.get("account_value", 0))
        elif exchange in ("hyperliquid",):
            from src.exchanges.router import get_balance
            data = get_balance()
            return float(data.get("account_value", 0))
        else:
            return 0.0
    except Exception as e:
        print(f"  ⚠️  Could not fetch balance: {e}")
        return 0.0


# ════════════════════════════════════════════════════════════════
# RULE CHECKS
# ════════════════════════════════════════════════════════════════

def check_rules(state: dict, profile: dict, account_rules: dict) -> dict:
    """
    Check all prop firm rules against current state.
    Returns a dict of checks and whether trading should continue.
    """
    today         = date.today().isoformat()
    balance       = state.get("current_balance", 0)
    start_balance = state.get("starting_balance", balance)
    peak_balance  = state.get("peak_balance", balance)
    daily_pnl     = state.get("daily_pnl", {})
    trading_days  = state.get("trading_days", [])

    today_pnl    = float(daily_pnl.get(today, 0))
    total_pnl    = balance - start_balance
    days_traded  = len(set(trading_days))

    # Drawdown calculation
    if profile.get("trailing_drawdown"):
        drawdown = peak_balance - balance         # trailing from peak
    else:
        drawdown = start_balance - balance        # from starting balance

    # Limits
    daily_limit   = account_rules["daily_loss"]
    max_dd_limit  = account_rules["max_dd"]
    profit_target = account_rules["profit_target"]
    min_days      = account_rules["min_days"]

    # Safety buffers — pause trading at 80% of limits
    daily_buffer  = daily_limit  * 0.80
    max_dd_buffer = max_dd_limit * 0.80

    checks = {
        "daily_loss": {
            "current":   round(abs(min(today_pnl, 0)), 2),
            "limit":     daily_limit,
            "buffer":    daily_buffer,
            "breached":  today_pnl <= -daily_limit,
            "warning":   today_pnl <= -daily_buffer,
            "pct_used":  round(abs(min(today_pnl, 0)) / daily_limit * 100, 1),
        },
        "max_drawdown": {
            "current":   round(max(drawdown, 0), 2),
            "limit":     max_dd_limit,
            "buffer":    max_dd_buffer,
            "breached":  drawdown >= max_dd_limit,
            "warning":   drawdown >= max_dd_buffer,
            "pct_used":  round(max(drawdown, 0) / max_dd_limit * 100, 1),
        },
        "profit_target": {
            "current":   round(total_pnl, 2),
            "target":    profit_target,
            "achieved":  total_pnl >= profit_target,
            "pct_done":  round(max(total_pnl, 0) / profit_target * 100, 1),
        },
        "min_trading_days": {
            "current":   days_traded,
            "required":  min_days,
            "met":       days_traded >= min_days,
        },
        "can_trade":   True,
        "must_pause":  False,
        "eval_passed": False,
    }

    # Determine if trading should stop
    if checks["daily_loss"]["breached"]:
        checks["can_trade"]  = False
        checks["must_pause"] = True
        checks["pause_reason"] = f"Daily loss limit hit: ${checks['daily_loss']['current']:,.0f} / ${daily_limit:,.0f}"

    elif checks["max_drawdown"]["breached"]:
        checks["can_trade"]  = False
        checks["must_pause"] = True
        checks["pause_reason"] = f"Max drawdown breached: ${checks['max_drawdown']['current']:,.0f} / ${max_dd_limit:,.0f}"

    elif checks["daily_loss"]["warning"]:
        checks["warning"] = f"⚠️ Approaching daily loss limit: {checks['daily_loss']['pct_used']:.0f}% used"

    elif checks["max_drawdown"]["warning"]:
        checks["warning"] = f"⚠️ Approaching max drawdown: {checks['max_drawdown']['pct_used']:.0f}% used"

    # Check if evaluation is passed
    if (checks["profit_target"]["achieved"] and
            checks["min_trading_days"]["met"] and
            not checks["daily_loss"]["breached"] and
            not checks["max_drawdown"]["breached"]):
        checks["eval_passed"] = True

    return checks


def enforce_pause(reason: str):
    """Write the pause file to stop all trading agents."""
    from datetime import timedelta
    resume = (datetime.now() + timedelta(hours=24)).isoformat()
    PAUSE_FILE.write_text(json.dumps({
        "paused":    True,
        "reason":    f"PROP FIRM: {reason}",
        "paused_at": datetime.now().isoformat(),
        "resume_at": resume,
    }, indent=2))
    print(f"\n  🛑 TRADING PAUSED — {reason}")
    print(f"     Will resume at: {resume}")


def log_check(state: dict, checks: dict):
    row = {
        "timestamp":       datetime.now().isoformat(),
        "firm":            state.get("firm", ""),
        "account":         state.get("account_size", ""),
        "phase":           state.get("phase", ""),
        "balance":         state.get("current_balance", 0),
        "total_pnl":       checks["profit_target"]["current"],
        "daily_pnl":       -checks["daily_loss"]["current"],
        "drawdown":        checks["max_drawdown"]["current"],
        "daily_pct":       checks["daily_loss"]["pct_used"],
        "dd_pct":          checks["max_drawdown"]["pct_used"],
        "profit_pct":      checks["profit_target"]["pct_done"],
        "days_traded":     checks["min_trading_days"]["current"],
        "can_trade":       checks["can_trade"],
        "eval_passed":     checks["eval_passed"],
    }
    write_header = not PROP_LOG.exists()
    with open(PROP_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════════════
# DISPLAY
# ════════════════════════════════════════════════════════════════

def print_dashboard(firm_name: str, account_rules: dict,
                     state: dict, checks: dict):
    today = date.today().isoformat()
    print(f"\n{'═'*62}")
    print(f"🏢 PROP FIRM MONITOR — {firm_name.upper()}")
    print(f"   Account: ${account_rules['size']:,} | Phase: {state['phase'].upper()}")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*62}")

    bal   = state.get("current_balance", 0)
    start = state.get("starting_balance", bal)
    print(f"  💰 Balance    : ${bal:>10,.2f}")
    print(f"  📈 Total P&L  : ${checks['profit_target']['current']:>+10,.2f}  "
          f"({checks['profit_target']['pct_done']:.1f}% of target)")
    print(f"{'─'*62}")

    # Daily loss
    dl = checks["daily_loss"]
    icon = "🔴" if dl["breached"] else "🟡" if dl["warning"] else "🟢"
    bar  = "█" * int(dl["pct_used"] / 10) + "░" * (10 - int(dl["pct_used"] / 10))
    print(f"  {icon} Daily Loss  : ${dl['current']:>8,.0f} / ${dl['limit']:,}")
    print(f"     [{bar}] {dl['pct_used']:.0f}% used")

    # Max drawdown
    dd = checks["max_drawdown"]
    icon = "🔴" if dd["breached"] else "🟡" if dd["warning"] else "🟢"
    bar  = "█" * int(dd["pct_used"] / 10) + "░" * (10 - int(dd["pct_used"] / 10))
    print(f"  {icon} Drawdown    : ${dd['current']:>8,.0f} / ${dd['limit']:,}")
    print(f"     [{bar}] {dd['pct_used']:.0f}% used")

    # Profit target
    pt = checks["profit_target"]
    icon = "✅" if pt["achieved"] else "🎯"
    bar  = "█" * int(pt["pct_done"] / 10) + "░" * (10 - int(pt["pct_done"] / 10))
    print(f"  {icon} Profit Tgt  : ${pt['current']:>8,.0f} / ${pt['target']:,}")
    print(f"     [{bar}] {pt['pct_done']:.0f}% done")

    # Trading days
    td = checks["min_trading_days"]
    icon = "✅" if td["met"] else "📅"
    print(f"  {icon} Trading Days: {td['current']:>3} / {td['required']} required")

    print(f"{'─'*62}")
    if checks.get("eval_passed"):
        print(f"  🎉 EVALUATION PASSED! Ready to request funding.")
    elif checks.get("must_pause"):
        print(f"  🛑 TRADING HALTED: {checks.get('pause_reason','')}")
    elif checks.get("warning"):
        print(f"  {checks['warning']}")
    else:
        print(f"  ✅ All rules OK — trading can continue")
    print(f"{'═'*62}")


# ════════════════════════════════════════════════════════════════
# MAIN AGENT
# ════════════════════════════════════════════════════════════════

class PropFirmMonitor:

    def __init__(self, firm: str = "apex", account: str = "50k"):
        self.firm_key    = firm.lower()
        self.account_key = account.lower()
        self.profile     = PROP_FIRM_PROFILES.get(self.firm_key)

        if not self.profile:
            available = list(PROP_FIRM_PROFILES.keys())
            raise ValueError(f"Unknown firm '{firm}'. Available: {available}")

        self.account_rules = self.profile["accounts"].get(self.account_key)
        if not self.account_rules:
            available = list(self.profile["accounts"].keys())
            raise ValueError(
                f"Unknown account size '{account}' for {self.profile['name']}. "
                f"Available: {available}"
            )

        self.state = load_state()

        # Initialise state if not set or firm changed
        if (self.state.get("firm") != self.firm_key or
                self.state.get("account_size") != self.account_key or
                not self.state.get("starting_balance")):
            self.state = reset_state(
                self.firm_key,
                self.account_key,
                self.account_rules["size"],
            )

        print(f"🏢 Prop Firm Monitor initialised")
        print(f"   Firm    : {self.profile['name']}")
        print(f"   Account : ${self.account_rules['size']:,}")
        print(f"   Phase   : {self.state['phase']}")
        print(f"   Rules   : Daily loss ${self.account_rules['daily_loss']:,} | "
              f"Max DD ${self.account_rules['max_dd']:,} | "
              f"Target ${self.account_rules['profit_target']:,}")

    def check(self) -> dict:
        """Run a single rule check and update state."""
        exchange = self.profile.get("exchange", "tradovate")
        balance  = get_current_balance(exchange)

        today = date.today().isoformat()

        # Update state
        if balance > 0:
            self.state["current_balance"] = balance
            self.state["peak_balance"]    = max(
                self.state.get("peak_balance", balance), balance
            )
            # Track today's P&L
            start = self.state.get("starting_balance", balance)
            prev_days_pnl = sum(
                float(v) for k, v in self.state["daily_pnl"].items()
                if k != today
            )
            today_pnl = (balance - start) - prev_days_pnl
            self.state["daily_pnl"][today] = round(today_pnl, 2)
            self.state["total_pnl"]        = round(balance - start, 2)

        # Add today to trading days
        if today not in self.state["trading_days"]:
            self.state["trading_days"].append(today)

        save_state(self.state)

        # Run checks
        checks = check_rules(self.state, self.profile, self.account_rules)
        print_dashboard(self.profile["name"], self.account_rules,
                         self.state, checks)

        # Enforce pause if needed
        if checks.get("must_pause"):
            enforce_pause(checks.get("pause_reason", "Rule breach"))

        log_check(self.state, checks)
        return checks

    def run(self, interval: int = 300):
        """Run continuously, checking every N seconds."""
        print(f"🚀 Prop Firm Monitor running. Ctrl+C to stop.\n")
        try:
            while True:
                self.check()
                print(f"\n😴 Next check in {interval//60} minutes...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Prop Firm Monitor stopped.")

    @staticmethod
    def list_firms():
        """Print all available prop firm profiles."""
        print("\n🏢 Available Prop Firm Profiles:")
        print(f"{'─'*65}")
        for key, p in PROP_FIRM_PROFILES.items():
            accounts = list(p["accounts"].keys())
            print(f"  {key:<15} {p['name']:<25} Accounts: {accounts}")
            print(f"               Split: {p['profit_split']*100:.0f}% | "
                  f"Type: {p['type']} | {p.get('website','')}")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🏢 Prop Firm Monitor")
    parser.add_argument("--firm",    default="apex",
                        help="Prop firm key e.g. apex, topstep, ftmo")
    parser.add_argument("--account", default="50k",
                        help="Account size e.g. 25k, 50k, 100k")
    parser.add_argument("--once",    action="store_true",
                        help="Single check then exit")
    parser.add_argument("--list",    action="store_true",
                        help="List all available firms")
    parser.add_argument("--reset",   action="store_true",
                        help="Reset evaluation state")
    args = parser.parse_args()

    if args.list:
        PropFirmMonitor.list_firms()
    else:
        monitor = PropFirmMonitor(firm=args.firm, account=args.account)
        if args.reset:
            monitor.state = reset_state(
                args.firm, args.account,
                monitor.account_rules["size"]
            )
        if args.once:
            monitor.check()
        else:
            monitor.run()
