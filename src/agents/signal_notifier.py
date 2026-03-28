# ============================================================
# 🌙 Signal Notifier & Manual Confirmation System
#
# Bridges the gap between auto-execution and manual trading.
# When a signal fires, it can either:
#
#   AUTO mode  → Execute immediately (existing behaviour)
#   MANUAL mode → Send to dashboard for your approval
#                 You click APPROVE or REJECT
#                 Only approved trades get executed
#
# EXECUTION MODES (set in src/config.py):
#   EXECUTION_MODE = "auto"    # fires trades automatically
#   EXECUTION_MODE = "manual"  # waits for your approval
#   EXECUTION_MODE = "notify"  # alerts only, never executes
#
# HOW IT WORKS IN MANUAL MODE:
#   1. ICT Scanner detects A+ setup
#   2. Signal appears on dashboard "Pending Signals" tab
#   3. You see: symbol, direction, entry, SL, TP, grade
#   4. You click APPROVE → trade executes
#   5. You click REJECT  → signal discarded
#   6. Signals expire after 30 minutes if not acted on
#
# HOW TO RUN:
#   python src/agents/signal_notifier.py
#   python src/agents/signal_notifier.py --mode manual
#   python src/agents/signal_notifier.py --mode auto
# ============================================================

import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config            import (EXCHANGE, SLEEP_BETWEEN_RUNS_SEC,
                                    STOP_LOSS_PCT, TAKE_PROFIT_PCT)
from src.agents.ict_scanner import scan_symbol, SYMBOLS, check_kill_zone, GRADE_ICONS
from src.agents.risk_agent  import risk
from src.exchanges.router   import buy, sell, get_price

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
PENDING_FILE   = REPO_ROOT / "src" / "data" / "pending_signals.json"
SIGNAL_LOG     = REPO_ROOT / "src" / "data" / "signal_log.csv"
PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── config ────────────────────────────────────────────────────
SIGNAL_EXPIRY_MINUTES = 30     # signals expire if not acted on
TRADE_SIZE_USD        = 50     # default trade size
MIN_GRADE             = "A+"   # minimum grade to notify/execute


# ════════════════════════════════════════════════════════════════
# PENDING SIGNALS STORE
# ════════════════════════════════════════════════════════════════

def load_pending() -> list[dict]:
    """Load all pending signals from disk."""
    if not PENDING_FILE.exists():
        return []
    try:
        return json.loads(PENDING_FILE.read_text())
    except Exception:
        return []


def save_pending(signals: list[dict]):
    """Save pending signals to disk."""
    PENDING_FILE.write_text(json.dumps(signals, indent=2))


def add_pending_signal(signal: dict) -> str:
    """Add a new pending signal. Returns signal ID."""
    import hashlib
    sig_id = hashlib.md5(
        f"{signal['symbol']}{signal['timestamp']}".encode()
    ).hexdigest()[:8]

    signal["id"]      = sig_id
    signal["status"]  = "PENDING"
    signal["created"] = datetime.now().isoformat()
    signal["expires"] = (datetime.now() +
                         timedelta(minutes=SIGNAL_EXPIRY_MINUTES)).isoformat()

    pending = load_pending()

    # Don't add duplicates
    existing_ids = {s.get("id") for s in pending}
    if sig_id in existing_ids:
        return sig_id

    pending.append(signal)
    save_pending(pending)
    return sig_id


def approve_signal(sig_id: str) -> dict | None:
    """Mark a signal as approved and return it for execution."""
    pending = load_pending()
    for s in pending:
        if s.get("id") == sig_id:
            s["status"]      = "APPROVED"
            s["approved_at"] = datetime.now().isoformat()
            save_pending(pending)
            return s
    return None


def reject_signal(sig_id: str) -> bool:
    """Mark a signal as rejected."""
    pending = load_pending()
    for s in pending:
        if s.get("id") == sig_id:
            s["status"]     = "REJECTED"
            s["rejected_at"] = datetime.now().isoformat()
            save_pending(pending)
            return True
    return False


def expire_old_signals():
    """Remove signals that have passed their expiry time."""
    pending  = load_pending()
    now      = datetime.now()
    updated  = []
    expired  = 0

    for s in pending:
        if s.get("status") == "PENDING":
            try:
                exp = datetime.fromisoformat(s["expires"])
                if now > exp:
                    s["status"] = "EXPIRED"
                    expired += 1
            except Exception:
                pass
        updated.append(s)

    if expired:
        print(f"  ⏰ {expired} signal(s) expired")

    # Keep last 200 signals for history
    save_pending(updated[-200:])
    return expired


def get_pending_count() -> int:
    """Return number of signals waiting for approval."""
    return sum(1 for s in load_pending() if s.get("status") == "PENDING")


# ════════════════════════════════════════════════════════════════
# EXECUTION
# ════════════════════════════════════════════════════════════════

def execute_signal(signal: dict) -> dict:
    """Execute an approved signal through the exchange."""
    symbol    = signal.get("symbol")
    direction = signal.get("direction")
    size      = float(signal.get("size_usd", TRADE_SIZE_USD))

    # Final risk check before execution
    dir_str = "buy" if direction == "LONG" else "sell"
    allowed, reason = risk.check_trade(symbol, size, dir_str)
    if not allowed:
        print(f"  🚫 Risk blocked execution: {reason}")
        signal["status"]       = "BLOCKED"
        signal["block_reason"] = reason
        return signal

    try:
        if direction == "LONG":
            buy(symbol, size)
        else:
            sell(symbol, size)
        signal["status"]      = "EXECUTED"
        signal["executed_at"] = datetime.now().isoformat()
        print(f"  ✅ Signal executed: {symbol} {direction} ${size}")
    except Exception as e:
        signal["status"] = "FAILED"
        signal["error"]  = str(e)
        print(f"  ❌ Execution failed: {e}")

    # Update in pending store
    pending = load_pending()
    for i, s in enumerate(pending):
        if s.get("id") == signal.get("id"):
            pending[i] = signal
            break
    save_pending(pending)

    # Log to CSV
    log_signal(signal)
    return signal


def log_signal(signal: dict):
    """Log signal to CSV for dashboard."""
    row = {
        "timestamp":  signal.get("created", datetime.now().isoformat()),
        "id":         signal.get("id", ""),
        "symbol":     signal.get("symbol", ""),
        "grade":      signal.get("grade", ""),
        "direction":  signal.get("direction", ""),
        "entry":      signal.get("entry_price", 0),
        "stop_loss":  signal.get("stop_loss", 0),
        "take_profit":signal.get("take_profit", 0),
        "size_usd":   signal.get("size_usd", 0),
        "kill_zone":  signal.get("kill_zone", ""),
        "cisd":       signal.get("cisd_strength", ""),
        "d1_bias":    signal.get("d1_bias", ""),
        "status":     signal.get("status", ""),
        "mode":       signal.get("mode", ""),
        "approved_at":signal.get("approved_at", ""),
        "executed_at":signal.get("executed_at", ""),
    }
    write_header = not SIGNAL_LOG.exists()
    with open(SIGNAL_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════════════
# MAIN NOTIFIER AGENT
# ════════════════════════════════════════════════════════════════

class SignalNotifier:

    def __init__(self, mode: str = "manual"):
        """
        mode: "auto"   — execute immediately
              "manual" — wait for dashboard approval
              "notify" — alert only, never execute
        """
        valid = ("auto", "manual", "notify")
        if mode not in valid:
            raise ValueError(f"Mode must be one of {valid}")

        self.mode    = mode
        self.symbols = SYMBOLS
        self.seen    = set()   # track signals already queued this session

        icons = {"auto": "⚡", "manual": "👆", "notify": "🔔"}
        print(f"\n{icons[mode]} Signal Notifier — {mode.upper()} mode")
        print(f"   Symbols : {self.symbols}")
        print(f"   Min grade: {MIN_GRADE}")
        if mode == "manual":
            print(f"   → Signals appear on dashboard → you click APPROVE/REJECT")
        elif mode == "auto":
            print(f"   → Signals execute automatically with no confirmation")
        else:
            print(f"   → Signals alert only — no trades executed")

    def scan(self):
        """Scan all symbols and handle signals based on mode."""
        kz = check_kill_zone()
        print(f"\n{'═'*60}")
        print(f"🔍 Signal scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if kz.get("active"):
            print(f"   🔥 Kill Zone: {kz['name']}")
        else:
            print(f"   💤 Outside Kill Zone — Next: {kz.get('next_session','—')}")

        # Expire old pending signals
        expire_old_signals()

        grade_order = {"A+": 3, "A": 2, "B": 1, "NO SETUP": 0}
        min_order   = grade_order.get(MIN_GRADE, 3)

        for symbol in self.symbols:
            try:
                setup = scan_symbol(symbol)
                grade = setup.setup_grade
                icon  = GRADE_ICONS.get(grade, "⚪")

                print(f"\n  {icon} {symbol}: {grade}", end="")

                if grade_order.get(grade, 0) < min_order:
                    print(f" — below threshold")
                    continue

                print()  # newline after grade

                # Build signal dict
                price = setup.current_price
                bias  = setup.d1_bias
                pdh   = setup.prev_day_high
                pdl   = setup.prev_day_low

                if bias == "BULLISH":
                    entry = price
                    sl    = round(price * (1 - STOP_LOSS_PCT), 4)
                    tp    = round(pdh, 4) if pdh > price else round(price * (1 + TAKE_PROFIT_PCT), 4)
                else:
                    entry = price
                    sl    = round(price * (1 + STOP_LOSS_PCT), 4)
                    tp    = round(pdl, 4) if pdl < price else round(price * (1 - TAKE_PROFIT_PCT), 4)

                rr = abs(tp - entry) / max(abs(sl - entry), 0.0001)

                signal = {
                    "symbol":       symbol,
                    "timestamp":    setup.timestamp,
                    "grade":        grade,
                    "direction":    "LONG" if bias == "BULLISH" else "SHORT",
                    "d1_bias":      bias,
                    "entry_price":  round(entry, 4),
                    "stop_loss":    round(sl, 4),
                    "take_profit":  round(tp, 4),
                    "risk_reward":  round(rr, 2),
                    "size_usd":     TRADE_SIZE_USD,
                    "kill_zone":    setup.kill_zone_name or "",
                    "cisd_strength":setup.cisd_strength if hasattr(setup, "cisd_strength") else "",
                    "price_zone":   setup.price_zone,
                    "fvg_found":    setup.h1_fvg_found,
                    "ob_found":     setup.h1_ob_found,
                    "score":        setup.setup_score,
                    "mode":         self.mode,
                    "ai_notes":     setup.ai_notes,
                }

                sig_key = f"{symbol}_{setup.timestamp[:13]}"

                if self.mode == "auto":
                    if sig_key in self.seen:
                        print(f"     ⏭️  Already processed this session")
                        continue
                    self.seen.add(sig_key)
                    print(f"     ⚡ AUTO executing...")
                    signal["status"] = "APPROVED"
                    execute_signal(signal)

                elif self.mode == "manual":
                    if sig_key in self.seen:
                        print(f"     ⏭️  Already in pending queue")
                        continue
                    self.seen.add(sig_key)
                    sig_id = add_pending_signal(signal)
                    pending_count = get_pending_count()
                    print(f"     👆 Signal queued for approval (ID: {sig_id})")
                    print(f"     → Go to dashboard → Pending Signals tab → APPROVE or REJECT")
                    print(f"     → Expires in {SIGNAL_EXPIRY_MINUTES} minutes")
                    print(f"     Total pending: {pending_count}")

                elif self.mode == "notify":
                    if sig_key in self.seen:
                        continue
                    self.seen.add(sig_key)
                    add_pending_signal(signal)
                    print(f"     🔔 Signal logged (notify only — no execution)")
                    print(f"     Entry ${entry:,.4f} | SL ${sl:,.4f} | TP ${tp:,.4f}")
                    print(f"     R:R {rr:.1f} | {bias}")

            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
            time.sleep(0.5)

        # In auto mode, also check for newly approved signals from dashboard
        if self.mode == "manual":
            self._process_approvals()

    def _process_approvals(self):
        """Execute any signals approved via the dashboard."""
        pending = load_pending()
        for signal in pending:
            if signal.get("status") == "APPROVED" and "executed_at" not in signal:
                print(f"\n  ✅ Processing approved signal: "
                      f"{signal['symbol']} {signal['direction']}")
                execute_signal(signal)

    def run(self):
        print(f"🚀 Signal Notifier running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.scan()
                print(f"\n😴 Next scan in {SLEEP_BETWEEN_RUNS_SEC}s...")
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)
        except KeyboardInterrupt:
            print("\n🛑 Signal Notifier stopped.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🔔 Signal Notifier")
    p.add_argument("--mode", default="manual",
                   choices=["auto", "manual", "notify"],
                   help="Execution mode (default: manual)")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    notifier = SignalNotifier(mode=args.mode)
    if args.once:
        notifier.scan()
    else:
        notifier.run()
