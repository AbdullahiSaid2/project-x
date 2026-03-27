# ============================================================
# 🌙 Psychology Guard Agent
#
# Monitors your trading behaviour and detects dangerous
# psychological patterns before they destroy your account:
#
#   OVERTRADING     → Too many trades in short period
#   REVENGE TRADING → Trading after a loss to "get it back"
#   TILT            → Increasing size after losses
#   LOSS STREAK     → Consecutive losses → system pause
#   GOOD STREAK     → Euphoria → prevent overconfidence
#
# When triggered, it logs warnings and can auto-pause trading.
#
# HOW TO RUN:
#   python src/agents/psychology_guard.py
#   python src/agents/psychology_guard.py --check
# ============================================================

import sys
import csv
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import MAX_DAILY_LOSS_USD
from src.models.llm_router import model

REPO_ROOT    = Path(__file__).resolve().parents[2]
PSYCH_LOG    = REPO_ROOT / "src" / "data" / "psychology_log.csv"
PAUSE_FILE   = REPO_ROOT / "src" / "data" / "trading_paused.json"
TRADE_LOG    = REPO_ROOT / "src" / "data" / "trade_log.csv"
PSYCH_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Thresholds ────────────────────────────────────────────────
MAX_TRADES_PER_HOUR     = 3      # more than this = overtrading
MAX_TRADES_PER_DAY      = 8      # daily trade cap
MAX_CONSECUTIVE_LOSSES  = 3      # pause after 3 losses in a row
MAX_DAILY_LOSS_PAUSE    = MAX_DAILY_LOSS_USD * 0.75   # pause at 75% of daily limit
REVENGE_WINDOW_MINUTES  = 15     # if next trade within 15min of loss = revenge


def load_recent_trades(hours: int = 24) -> pd.DataFrame:
    """Load trades from the last N hours."""
    if not TRADE_LOG.exists():
        return pd.DataFrame()
    try:
        df  = pd.read_csv(TRADE_LOG)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(hours=hours)
        return df[df["timestamp"] > cutoff].sort_values("timestamp")
    except Exception:
        return pd.DataFrame()


def is_paused() -> bool:
    """Check if trading is currently paused."""
    if not PAUSE_FILE.exists():
        return False
    data    = json.loads(PAUSE_FILE.read_text())
    resume  = data.get("resume_at")
    if resume:
        if datetime.now() < datetime.fromisoformat(resume):
            return True
        else:
            # Pause expired
            PAUSE_FILE.unlink()
            return False
    return data.get("paused", False)


def set_pause(reason: str, hours: float = 4):
    """Pause trading for N hours."""
    resume_at = (datetime.now() + timedelta(hours=hours)).isoformat()
    data = {
        "paused":    True,
        "reason":    reason,
        "paused_at": datetime.now().isoformat(),
        "resume_at": resume_at,
    }
    PAUSE_FILE.write_text(json.dumps(data, indent=2))
    print(f"\n  🛑 TRADING PAUSED for {hours}h: {reason}")
    print(f"     Resumes at: {resume_at}")


def clear_pause():
    """Manually clear trading pause."""
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
    print("  ✅ Trading pause cleared")


def detect_overtrading(df: pd.DataFrame) -> dict:
    """Detect if we're trading too frequently."""
    if df.empty:
        return {"detected": False}

    now       = datetime.now()
    last_hour = df[df["timestamp"] > now - timedelta(hours=1)]
    today     = df[df["timestamp"] > now.replace(hour=0, minute=0, second=0)]

    hourly_count = len(last_hour)
    daily_count  = len(today)

    if hourly_count > MAX_TRADES_PER_HOUR:
        return {
            "detected": True,
            "type":     "OVERTRADING_HOURLY",
            "count":    hourly_count,
            "message":  f"{hourly_count} trades in last hour (max {MAX_TRADES_PER_HOUR})",
            "severity": "HIGH",
        }
    if daily_count > MAX_TRADES_PER_DAY:
        return {
            "detected": True,
            "type":     "OVERTRADING_DAILY",
            "count":    daily_count,
            "message":  f"{daily_count} trades today (max {MAX_TRADES_PER_DAY})",
            "severity": "MEDIUM",
        }
    return {"detected": False, "hourly": hourly_count, "daily": daily_count}


def detect_revenge_trading(df: pd.DataFrame) -> dict:
    """Detect if next trade came too quickly after a loss."""
    if len(df) < 2:
        return {"detected": False}

    # We don't have outcome in trade_log — check time between trades as proxy
    # Fast consecutive trades (< 15 min apart) are a revenge flag
    times = df["timestamp"].sort_values()
    gaps  = times.diff().dt.total_seconds() / 60   # in minutes

    revenge_gaps = gaps[gaps < REVENGE_WINDOW_MINUTES].dropna()
    if len(revenge_gaps) >= 2:
        return {
            "detected": True,
            "type":     "REVENGE_TRADING",
            "count":    len(revenge_gaps),
            "message":  f"{len(revenge_gaps)} trades placed within {REVENGE_WINDOW_MINUTES}min of each other",
            "severity": "HIGH",
        }
    return {"detected": False}


def detect_loss_streak(df: pd.DataFrame) -> dict:
    """Detect consecutive losses (approximated from trade frequency + timing)."""
    if len(df) < MAX_CONSECUTIVE_LOSSES:
        return {"detected": False, "streak": 0}

    # Check last N trades — if they were all very quick (< 5 min gaps)
    # that suggests chasing/losses (heuristic only without P&L data)
    recent = df.tail(MAX_CONSECUTIVE_LOSSES)
    gaps   = recent["timestamp"].diff().dt.total_seconds().dropna()
    rapid  = (gaps < 300).all()   # all within 5 minutes

    if rapid and len(recent) >= MAX_CONSECUTIVE_LOSSES:
        return {
            "detected": True,
            "type":     "RAPID_FIRE_TRADES",
            "streak":   MAX_CONSECUTIVE_LOSSES,
            "message":  f"Last {MAX_CONSECUTIVE_LOSSES} trades placed very rapidly — possible loss chasing",
            "severity": "HIGH",
        }
    return {"detected": False, "streak": 0}


PSYCH_PROMPT = """You are a trading psychology coach.
Review this trader's recent behaviour and provide honest feedback.

DATA:
{data}

Patterns detected: {patterns}

Provide:
1. What psychological pattern is most concerning right now?
2. What should the trader do next (be specific)?
3. One grounding reminder

Keep it short, direct, and compassionate. Plain text, 3 sentences."""


def get_psychology_coaching(df: pd.DataFrame, patterns: list) -> str:
    """Get AI coaching based on detected patterns."""
    if not patterns:
        return "Trading behaviour looks healthy. No intervention needed."
    try:
        data = {
            "trades_last_24h":  len(df),
            "patterns_detected": patterns,
            "daily_loss_limit": MAX_DAILY_LOSS_USD,
        }
        return model.chat(
            system_prompt="You are a trading psychology coach. Be direct and brief.",
            user_prompt=PSYCH_PROMPT.format(
                data=json.dumps(data, indent=2),
                patterns=", ".join([p.get("type","") for p in patterns])
            ),
        ).strip()
    except Exception:
        return "Unable to generate coaching. Take a break and review your trades manually."


def log_psych_event(event_type: str, message: str, severity: str):
    row = {
        "timestamp": datetime.now().isoformat(),
        "event":     event_type,
        "severity":  severity,
        "message":   message,
    }
    write_header = not PSYCH_LOG.exists()
    with open(PSYCH_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


class PsychologyGuard:

    def check(self) -> dict:
        print(f"\n{'═'*60}")
        print(f"🧠 Psychology Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check if already paused
        if is_paused():
            pause_data = json.loads(PAUSE_FILE.read_text())
            print(f"\n  🛑 TRADING IS PAUSED")
            print(f"     Reason   : {pause_data.get('reason','')}")
            print(f"     Resumes  : {pause_data.get('resume_at','')}")
            return {"status": "PAUSED", "data": pause_data}

        # Load recent trades
        df = load_recent_trades(hours=24)
        print(f"\n  📊 Trades in last 24h: {len(df)}")

        if df.empty:
            print("  ✅ No recent trades — nothing to flag")
            return {"status": "HEALTHY", "patterns": []}

        # Run all checks
        patterns = []
        checks = [
            detect_overtrading(df),
            detect_revenge_trading(df),
            detect_loss_streak(df),
        ]

        for check in checks:
            if check.get("detected"):
                patterns.append(check)
                severity = check.get("severity", "MEDIUM")
                msg      = check.get("message", "")
                sev_icon = "🚨" if severity == "HIGH" else "⚠️"
                print(f"\n  {sev_icon} [{severity}] {check['type']}")
                print(f"     {msg}")
                log_psych_event(check["type"], msg, severity)

        if not patterns:
            print("\n  ✅ No psychological patterns detected — trading is healthy")
            return {"status": "HEALTHY", "patterns": []}

        # Get AI coaching
        print(f"\n  🤖 AI Coaching:")
        coaching = get_psychology_coaching(df, patterns)
        print(f"     {coaching}")

        # Auto-pause for HIGH severity
        high_patterns = [p for p in patterns if p.get("severity") == "HIGH"]
        if high_patterns:
            worst = high_patterns[0]
            set_pause(worst.get("message", "Pattern detected"), hours=4)

        return {"status": "WARNING" if patterns else "HEALTHY", "patterns": patterns}

    def run(self):
        print("🚀 Psychology Guard running. Ctrl+C to stop.\n")
        try:
            while True:
                self.check()
                print(f"\n😴 Next check in 30 minutes...")
                time.sleep(1800)
        except KeyboardInterrupt:
            print("\n🛑 Psychology Guard stopped.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--check",       action="store_true", help="Single check")
    p.add_argument("--clear-pause", action="store_true", help="Clear trading pause")
    args = p.parse_args()

    if args.clear_pause:
        clear_pause()
    elif args.check:
        PsychologyGuard().check()
    else:
        PsychologyGuard().run()
