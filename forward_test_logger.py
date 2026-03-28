# ============================================================
# 🌙 Forward Test Logger
#
# Automatically logs every ICT A/A+ setup signal and then
# checks back on each trade to record the actual outcome.
# No manual logging needed.
#
# WHAT IT DOES:
#   1. Runs ICT Scanner every 5 minutes
#   2. Logs every A+ and A grade setup automatically
#   3. Every hour checks all open signals and records:
#      - Did price hit the take profit? (WIN)
#      - Did price hit the stop loss?  (LOSS)
#      - Is it still running?          (OPEN)
#   4. Calculates live forward test statistics
#   5. Shows everything on the dashboard
#
# HOW TO RUN:
#   python src/agents/forward_test_logger.py
#   python src/agents/forward_test_logger.py --once
# ============================================================

import sys
import csv
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config           import EXCHANGE, SLEEP_BETWEEN_RUNS_SEC
from src.data.fetcher     import get_ohlcv
from src.agents.ict_scanner import scan_symbol, SYMBOLS, check_kill_zone

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
FT_LOG       = REPO_ROOT / "src" / "data" / "forward_test_log.csv"
FT_STATS     = REPO_ROOT / "src" / "data" / "forward_test_stats.json"
FT_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── config ────────────────────────────────────────────────────
MIN_GRADE        = "A"        # log A and A+ grades
CHECK_INTERVAL   = 300        # scan every 5 minutes
OUTCOME_LOOKBACK = 48         # check outcomes up to 48 hours later
TP_PCT           = 0.006      # 0.6% take profit (previous day high approx)
SL_PCT           = 0.005      # 0.5% stop loss


# ── CSV helpers ───────────────────────────────────────────────
FIELDNAMES = [
    "id", "timestamp", "symbol", "grade", "score",
    "d1_bias", "price", "pdh", "pdl", "price_zone",
    "fvg_found", "fvg_level", "ob_found", "ob_level",
    "displacement", "liq_swept", "kill_zone",
    "entry_price", "stop_loss", "take_profit",
    "outcome", "exit_price", "exit_time",
    "pnl_pct", "hours_to_outcome", "ai_notes",
]

def load_signals() -> list[dict]:
    if not FT_LOG.exists():
        return []
    try:
        with open(FT_LOG, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def save_signal(row: dict):
    write_header = not FT_LOG.exists()
    with open(FT_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        # Fill any missing fields with empty string
        filled = {k: row.get(k, "") for k in FIELDNAMES}
        w.writerow(filled)


def update_signal(signal_id: str, updates: dict):
    """Update an existing signal row by ID."""
    signals = load_signals()
    updated = False
    for s in signals:
        if s.get("id") == signal_id:
            s.update(updates)
            updated = True
            break
    if updated:
        with open(FT_LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(signals)


def generate_id(symbol: str, timestamp: str) -> str:
    import hashlib
    return hashlib.md5(f"{symbol}{timestamp}".encode()).hexdigest()[:8]


# ── Outcome checker ───────────────────────────────────────────
def check_outcome(signal: dict) -> dict | None:
    """
    Check if an open signal has hit its TP or SL yet.
    Returns updated fields or None if still open.
    """
    if signal.get("outcome") not in ("", "OPEN", None):
        return None   # already resolved

    try:
        entry      = float(signal.get("entry_price", 0))
        sl         = float(signal.get("stop_loss", 0))
        tp         = float(signal.get("take_profit", 0))
        direction  = signal.get("d1_bias", "BULLISH")
        symbol     = signal.get("symbol", "")
        sig_time   = datetime.fromisoformat(signal.get("timestamp", datetime.now().isoformat()))

        if entry == 0 or sl == 0 or tp == 0:
            return None

        # Check if signal is too old
        hours_elapsed = (datetime.now() - sig_time).total_seconds() / 3600
        if hours_elapsed > OUTCOME_LOOKBACK:
            return {
                "outcome":         "EXPIRED",
                "exit_time":       datetime.now().isoformat(),
                "hours_to_outcome": round(hours_elapsed, 1),
                "pnl_pct":         "0.0",
            }

        # Fetch recent candles to check if price hit TP or SL
        df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe="1H", days_back=3)
        if df is None or df.empty:
            return None

        # Only look at candles after the signal was logged
        df = df[df.index > sig_time]
        if df.empty:
            return None

        for idx, row in df.iterrows():
            high = float(row["High"])
            low  = float(row["Low"])

            if direction == "BULLISH":
                hit_tp = high >= tp
                hit_sl = low  <= sl
            else:
                hit_tp = low  <= tp
                hit_sl = high >= sl

            # Check SL first (conservative — if both hit same candle, count as loss)
            if hit_sl:
                hours = (idx - sig_time).total_seconds() / 3600
                pnl   = abs(entry - sl) / entry * -100
                return {
                    "outcome":          "LOSS",
                    "exit_price":       str(round(sl, 4)),
                    "exit_time":        str(idx),
                    "pnl_pct":          str(round(pnl, 3)),
                    "hours_to_outcome": str(round(hours, 1)),
                }
            if hit_tp:
                hours = (idx - sig_time).total_seconds() / 3600
                pnl   = abs(tp - entry) / entry * 100
                return {
                    "outcome":          "WIN",
                    "exit_price":       str(round(tp, 4)),
                    "exit_time":        str(idx),
                    "pnl_pct":          str(round(pnl, 3)),
                    "hours_to_outcome": str(round(hours, 1)),
                }

        # Still running
        return {"outcome": "OPEN"}

    except Exception as e:
        return None


# ── Stats calculator ──────────────────────────────────────────
def calculate_stats(signals: list[dict]) -> dict:
    resolved = [s for s in signals if s.get("outcome") in ("WIN","LOSS")]
    wins     = [s for s in resolved if s.get("outcome") == "WIN"]
    losses   = [s for s in resolved if s.get("outcome") == "LOSS"]
    open_s   = [s for s in signals  if s.get("outcome") in ("OPEN","","")]

    win_rate  = len(wins) / len(resolved) * 100 if resolved else 0
    avg_win   = sum(float(s.get("pnl_pct",0)) for s in wins)   / len(wins)   if wins   else 0
    avg_loss  = sum(float(s.get("pnl_pct",0)) for s in losses) / len(losses) if losses else 0

    # By grade
    ap_signals = [s for s in signals if s.get("grade") == "A+"]
    ap_resolved= [s for s in ap_signals if s.get("outcome") in ("WIN","LOSS")]
    ap_wins    = [s for s in ap_resolved if s.get("outcome") == "WIN"]
    ap_wr      = len(ap_wins) / len(ap_resolved) * 100 if ap_resolved else 0

    stats = {
        "total_signals":   len(signals),
        "resolved":        len(resolved),
        "open":            len(open_s),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        round(win_rate, 1),
        "avg_win_pct":     round(avg_win, 3),
        "avg_loss_pct":    round(avg_loss, 3),
        "aplus_signals":   len(ap_signals),
        "aplus_win_rate":  round(ap_wr, 1),
        "last_updated":    datetime.now().isoformat(),
    }
    FT_STATS.write_text(json.dumps(stats, indent=2))
    return stats


def print_stats(stats: dict):
    print(f"\n  📊 Forward Test Summary:")
    print(f"     Total signals : {stats['total_signals']}")
    print(f"     Resolved      : {stats['resolved']} "
          f"({stats['wins']}W / {stats['losses']}L)")
    print(f"     Open          : {stats['open']}")
    print(f"     Win rate      : {stats['win_rate']:.1f}%")
    print(f"     A+ win rate   : {stats['aplus_win_rate']:.1f}%")
    print(f"     Avg win       : +{stats['avg_win_pct']:.3f}%")
    print(f"     Avg loss      : {stats['avg_loss_pct']:.3f}%")


# ════════════════════════════════════════════════════════════════
class ForwardTestLogger:

    def __init__(self):
        self.logged_ids = {s.get("id") for s in load_signals()}
        print("📋 Forward Test Logger initialised")
        print(f"   Symbols      : {SYMBOLS}")
        print(f"   Min grade    : {MIN_GRADE}")
        print(f"   Scan interval: {CHECK_INTERVAL//60} minutes")
        print(f"   Existing logs: {len(self.logged_ids)} signals\n")

    def scan_and_log(self):
        """Run ICT scan and log any A/A+ setups."""
        kz     = check_kill_zone()
        kz_str = f"🔥 {kz['name']}" if kz.get("active") else f"Next: {kz.get('next_session','—')}"
        print(f"\n{'═'*60}")
        print(f"📋 Forward Test Scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   {kz_str}")

        new_signals = 0
        for symbol in SYMBOLS:
            try:
                setup = scan_symbol(symbol)
                grade = setup.setup_grade

                grade_order = {"A+": 3, "A": 2, "B": 1, "NO SETUP": 0}
                min_order   = grade_order.get(MIN_GRADE, 2)
                if grade_order.get(grade, 0) < min_order:
                    print(f"  ⏸️  {symbol}: {grade} — below threshold")
                    continue

                # Build signal ID to avoid duplicates
                sig_id = generate_id(symbol, setup.timestamp[:16])
                if sig_id in self.logged_ids:
                    print(f"  ⏭️  {symbol}: {grade} already logged")
                    continue

                # Calculate entry, SL, TP
                price    = setup.current_price
                pdh      = setup.prev_day_high
                pdl      = setup.prev_day_low
                bias     = setup.d1_bias

                if bias == "BULLISH":
                    entry = price
                    sl    = round(price * (1 - SL_PCT), 4)
                    tp    = round(pdh, 4) if pdh > price else round(price * (1 + TP_PCT), 4)
                else:
                    entry = price
                    sl    = round(price * (1 + SL_PCT), 4)
                    tp    = round(pdl, 4) if pdl < price else round(price * (1 - TP_PCT), 4)

                row = {
                    "id":          sig_id,
                    "timestamp":   setup.timestamp,
                    "symbol":      symbol,
                    "grade":       grade,
                    "score":       setup.setup_score,
                    "d1_bias":     setup.d1_bias,
                    "price":       round(price, 4),
                    "pdh":         round(pdh, 4),
                    "pdl":         round(pdl, 4),
                    "price_zone":  setup.price_zone,
                    "fvg_found":   setup.h1_fvg_found,
                    "fvg_level":   setup.h1_fvg_level,
                    "ob_found":    setup.h1_ob_found,
                    "ob_level":    setup.h1_ob_level,
                    "displacement":setup.h1_disp_strength,
                    "liq_swept":   setup.h1_swept_liq,
                    "kill_zone":   setup.kill_zone_name,
                    "entry_price": entry,
                    "stop_loss":   sl,
                    "take_profit": tp,
                    "outcome":     "OPEN",
                    "exit_price":  "",
                    "exit_time":   "",
                    "pnl_pct":     "",
                    "hours_to_outcome": "",
                    "ai_notes":    setup.ai_notes,
                }

                save_signal(row)
                self.logged_ids.add(sig_id)
                new_signals += 1

                icon = "🚨" if grade == "A+" else "⚡"
                print(f"\n  {icon} NEW SIGNAL LOGGED: {symbol} {grade}")
                print(f"     Bias   : {bias}")
                print(f"     Entry  : ${entry:,.4f}")
                print(f"     SL     : ${sl:,.4f}")
                print(f"     TP     : ${tp:,.4f}")
                print(f"     Zone   : {setup.price_zone}")
                print(f"     KZ     : {setup.kill_zone_name or 'Outside KZ'}")
                if setup.ai_notes:
                    print(f"     Notes  : {setup.ai_notes}")

            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
            time.sleep(1)

        if new_signals == 0:
            print(f"  ✅ No new A/A+ setups this scan")

        return new_signals

    def check_outcomes(self):
        """Check all open signals and update outcomes."""
        signals  = load_signals()
        open_sigs = [s for s in signals if s.get("outcome") in ("OPEN", "", None)]

        if not open_sigs:
            return

        print(f"\n  🔍 Checking outcomes for {len(open_sigs)} open signal(s)...")
        updated = 0
        for sig in open_sigs:
            result = check_outcome(sig)
            if result and result.get("outcome") != "OPEN":
                update_signal(sig["id"], result)
                outcome = result.get("outcome","")
                icon    = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "⏰"
                print(f"  {icon} {sig['symbol']} {sig['grade']} → {outcome} "
                      f"({result.get('pnl_pct','')}% in {result.get('hours_to_outcome','')}h)")
                updated += 1

        if updated:
            print(f"  📊 {updated} outcome(s) updated")

    def run_once(self):
        self.scan_and_log()
        self.check_outcomes()
        signals = load_signals()
        stats   = calculate_stats(signals)
        print_stats(stats)

    def run(self):
        print("🚀 Forward Test Logger running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.run_once()
                print(f"\n😴 Next scan in {CHECK_INTERVAL//60} minutes...")
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\n🛑 Forward Test Logger stopped.")
            signals = load_signals()
            stats   = calculate_stats(signals)
            print_stats(stats)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="📋 Forward Test Logger")
    p.add_argument("--once", action="store_true", help="Single scan then exit")
    args = p.parse_args()
    logger = ForwardTestLogger()
    if args.once:
        logger.run_once()
    else:
        logger.run()
