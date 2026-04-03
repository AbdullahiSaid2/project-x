#!/usr/bin/env python3
# ============================================================
# 🌙 Apex Risk Monitor
#
# Enforces all Apex $50k evaluation rules:
#
#   Rule 1 — Daily Drawdown:    $1,000 max loss per day
#   Rule 2 — Max Drawdown:      $2,000 trailing from peak
#   Rule 3 — Profit Target:     $3,000 to pass
#   Rule 4 — No News Trading:   close 5 min before, open 5 min after
#   Rule 5 — Consistency:       no day > 50% of total profits
#
# USAGE:
#   from src.agents.apex_risk import ApexRisk
#   apex = ApexRisk()
#   ok, reason = apex.check(signal)
#   if ok: execute trade
#
#   apex.update_pnl(pnl_usd)        # call after each trade closes
#   apex.status()                    # print current standing
# ============================================================

import json
import csv
import time
import requests
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "src" / "data"
STATE_FILE = DATA_DIR / "apex_state.json"
LOG_FILE   = DATA_DIR / "apex_risk_log.csv"
NEWS_CACHE = DATA_DIR / "news_cache.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════

def _default_state() -> dict:
    return {
        "account_size":    50_000.0,
        "peak_equity":     50_000.0,   # for trailing drawdown
        "current_equity":  50_000.0,
        "total_pnl":       0.0,        # cumulative profit
        "daily_pnl":       0.0,        # today's profit
        "daily_peak":      0.0,        # today's highest profit (for daily DD calc)
        "best_day_pnl":    0.0,        # highest single day profit ever
        "today":           str(date.today()),
        "trades_today":    0,
        "total_trades":    0,
        "daily_history":   {},         # {date_str: pnl}
        "passed":          False,
        "blown":           False,
        "blow_reason":     "",
    }


def load_state() -> dict:
    if not STATE_FILE.exists():
        return _default_state()
    try:
        state = json.loads(STATE_FILE.read_text())
        today = str(date.today())
        # Reset daily fields on new day
        if state.get("today") != today:
            # Save yesterday's daily PnL to history before resetting
            if state.get("daily_pnl", 0) != 0:
                state["daily_history"][state["today"]] = state["daily_pnl"]
            # Track best single day
            state["best_day_pnl"] = max(
                state.get("best_day_pnl", 0),
                state.get("daily_pnl", 0)
            )
            state["daily_pnl"]    = 0.0
            state["daily_peak"]   = 0.0
            state["trades_today"] = 0
            state["today"]        = today
            save_state(state)
        return state
    except Exception:
        return _default_state()


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _log(event: str, detail: str, state: dict):
    exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","event","detail",
            "equity","daily_pnl","total_pnl","drawdown_from_peak"
        ])
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "event":              event,
            "detail":             detail,
            "equity":             state.get("current_equity", 0),
            "daily_pnl":          state.get("daily_pnl", 0),
            "total_pnl":          state.get("total_pnl", 0),
            "drawdown_from_peak": state.get("current_equity",0) - state.get("peak_equity",0),
        })


# ══════════════════════════════════════════════════════════════
# NEWS FILTER (Rule 4)
# ══════════════════════════════════════════════════════════════

def fetch_news_events(force_refresh: bool = False) -> list[dict]:
    """
    Fetch high-impact economic news events for today.
    Uses ForexFactory calendar (free, no API key needed).
    Falls back to empty list if unavailable.
    """
    cache = {}
    today_str = str(date.today())

    # Load cache
    if NEWS_CACHE.exists() and not force_refresh:
        try:
            cache = json.loads(NEWS_CACHE.read_text())
            if cache.get("date") == today_str:
                return cache.get("events", [])
        except Exception:
            pass

    events = []

    # Try ForexFactory JSON feed
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r   = requests.get(url, timeout=10,
                           headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()

        for item in data:
            # Filter: high impact + USD only + today
            if item.get("impact", "").lower() != "high":
                continue
            if item.get("country", "").upper() != "USD":
                continue

            try:
                raw_date = item.get("date", "")
                # ForexFactory format: "01-15-2026" or ISO
                for fmt in ["%m-%d-%Y", "%Y-%m-%dT%H:%M:%S%z",
                             "%Y-%m-%d %H:%M:%S"]:
                    try:
                        event_dt = datetime.strptime(raw_date[:len(fmt)-2], fmt[:len(raw_date)])
                        break
                    except Exception:
                        event_dt = None

                if event_dt and str(event_dt.date()) == today_str:
                    # ForexFactory time is EST
                    events.append({
                        "title":   item.get("title", "Unknown"),
                        "time":    item.get("date", ""),
                        "impact":  "HIGH",
                        "country": "USD",
                    })
            except Exception:
                continue

    except Exception as e:
        print(f"  ⚠️  News fetch failed: {e} — news filter disabled for safety")

    # Cache result
    try:
        NEWS_CACHE.write_text(json.dumps({
            "date":    today_str,
            "events":  events,
            "fetched": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    except Exception:
        pass

    return events


def get_news_windows(buffer_before: int = 5,
                     buffer_after: int  = 5) -> list[dict]:
    """
    Returns list of no-trade windows around high-impact news.
    Each window: {title, no_trade_start (UTC), no_trade_end (UTC)}
    """
    events  = fetch_news_events()
    windows = []

    for e in events:
        try:
            # Parse event time — may be EST or UTC
            raw = e.get("time", "")
            for fmt in ["%m-%d-%YT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%d %H:%M"]:
                try:
                    dt = datetime.strptime(raw[:19], fmt[:19])
                    break
                except Exception:
                    dt = None

            if not dt:
                continue

            # Assume EST if naive, convert to UTC
            if dt.tzinfo is None:
                import zoneinfo
                est = zoneinfo.ZoneInfo("America/New_York")
                dt  = dt.replace(tzinfo=est).astimezone(timezone.utc)

            windows.append({
                "title":          e["title"],
                "event_time_utc": dt,
                "no_trade_start": dt - timedelta(minutes=buffer_before),
                "no_trade_end":   dt + timedelta(minutes=buffer_after),
            })
        except Exception:
            continue

    return windows


def check_news_clear(signal_time: datetime = None) -> tuple[bool, str]:
    """
    Returns (is_clear, reason).
    is_clear = True means SAFE to trade.
    """
    from src.config import (APEX_NEWS_FILTER,
                            APEX_NEWS_CLOSE_BEFORE, APEX_NEWS_OPEN_AFTER)

    if not APEX_NEWS_FILTER:
        return True, "News filter disabled"

    now = signal_time or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    windows = get_news_windows(APEX_NEWS_CLOSE_BEFORE, APEX_NEWS_OPEN_AFTER)

    if not windows:
        return True, "No high-impact news scheduled today"

    for w in windows:
        start = w["no_trade_start"]
        end   = w["no_trade_end"]
        evt   = w["event_time_utc"]

        if start <= now <= end:
            mins_to_event = (evt - now).total_seconds() / 60
            if mins_to_event > 0:
                return False, (
                    f"🚫 NEWS BLOCK — {w['title']} in "
                    f"{mins_to_event:.0f} min. "
                    f"No trades until {end.strftime('%H:%M')} UTC"
                )
            else:
                mins_after = (now - evt).total_seconds() / 60
                return False, (
                    f"🚫 NEWS BLOCK — {w['title']} was "
                    f"{mins_after:.0f} min ago. "
                    f"No trades until {end.strftime('%H:%M')} UTC"
                )

    # Warn if news is coming soon (within 30 min)
    for w in windows:
        mins_away = (w["event_time_utc"] - now).total_seconds() / 60
        if 0 < mins_away <= 30:
            return True, (
                f"⚠️  News warning: {w['title']} in "
                f"{mins_away:.0f} min ({w['event_time_utc'].strftime('%H:%M')} UTC)"
            )

    return True, "Clear"


def should_close_for_news() -> tuple[bool, str]:
    """
    Returns (should_close, reason).
    Called every scan to check if open positions must be closed.
    """
    clear, reason = check_news_clear()
    if not clear:
        return True, reason
    return False, reason


# ══════════════════════════════════════════════════════════════
# MAIN APEX RISK CLASS
# ══════════════════════════════════════════════════════════════

class ApexRisk:

    def __init__(self):
        self.state = load_state()

    def reload(self):
        self.state = load_state()

    # ── Rule 1: Daily Drawdown ─────────────────────────────────
    def _check_daily_drawdown(self) -> tuple[bool, str]:
        from src.config import APEX_DAILY_DRAWDOWN
        # Daily drawdown measured from session-open equity
        # Session open equity = account_size + total_pnl at start of day
        session_open = self.state["account_size"] + (
            self.state["total_pnl"] - self.state["daily_pnl"]
        )
        daily_loss = min(0, self.state["daily_pnl"])
        loss_usd   = abs(daily_loss)

        if loss_usd >= APEX_DAILY_DRAWDOWN:
            return False, (
                f"🚫 DAILY DRAWDOWN — lost ${loss_usd:.0f} today "
                f"(limit: ${APEX_DAILY_DRAWDOWN}). No more trades today."
            )

        remaining = APEX_DAILY_DRAWDOWN - loss_usd
        if remaining < APEX_DAILY_DRAWDOWN * 0.25:
            return True, f"⚠️  Daily DD: ${loss_usd:.0f}/${APEX_DAILY_DRAWDOWN} — ${remaining:.0f} left"

        return True, f"Daily DD: ${loss_usd:.0f}/${APEX_DAILY_DRAWDOWN}"

    # ── Rule 2: Trailing Max Drawdown ─────────────────────────
    def _check_max_drawdown(self) -> tuple[bool, str]:
        from src.config import APEX_MAX_DRAWDOWN
        drawdown = self.state["peak_equity"] - self.state["current_equity"]

        if drawdown >= APEX_MAX_DRAWDOWN:
            return False, (
                f"🚫 MAX DRAWDOWN BLOWN — account dropped ${drawdown:.0f} "
                f"from peak ${self.state['peak_equity']:,.0f}. "
                f"Evaluation FAILED."
            )

        remaining = APEX_MAX_DRAWDOWN - drawdown
        if remaining < APEX_MAX_DRAWDOWN * 0.25:
            return True, f"⚠️  Max DD: ${drawdown:.0f}/${APEX_MAX_DRAWDOWN} — ${remaining:.0f} left"

        return True, f"Max DD: ${drawdown:.0f}/${APEX_MAX_DRAWDOWN}"

    # ── Rule 3: Profit Target ─────────────────────────────────
    def _check_profit_target(self) -> tuple[bool, str]:
        from src.config import APEX_PROFIT_TARGET
        total = self.state["total_pnl"]

        if total >= APEX_PROFIT_TARGET:
            return True, f"🎯 PROFIT TARGET REACHED! ${total:.0f} ≥ ${APEX_PROFIT_TARGET} — PASS!"

        remaining = APEX_PROFIT_TARGET - total
        return True, f"Profit: ${total:.0f}/${APEX_PROFIT_TARGET} (${remaining:.0f} to go)"

    # ── Rule 4: News Filter ────────────────────────────────────
    def _check_news(self) -> tuple[bool, str]:
        clear, reason = check_news_clear()
        return clear, reason

    # ── Rule 5: Consistency Rule (PERFORMANCE ONLY) ───────────
    def _check_consistency(self, projected_gain: float = 0) -> tuple[bool, str]:
        """
        No single day can be more than 50% of total profits.
        ONLY enforced on performance accounts — skipped during evaluation.
        """
        from src.config import APEX_CONSISTENCY_PCT, PROP_FIRM_ACCOUNT_TYPE

        # Skip entirely during evaluation
        if PROP_FIRM_ACCOUNT_TYPE != "performance":
            return True, "Consistency: N/A (eval — rule not active)"

        total_pnl = self.state["total_pnl"]

        # Only check if we're in profit overall
        if total_pnl <= 0:
            return True, "Consistency: N/A (no profit yet)"

        today_pnl = self.state["daily_pnl"]
        today_pct = today_pnl / total_pnl if total_pnl > 0 else 0
        limit_usd = total_pnl * APEX_CONSISTENCY_PCT

        # Already at limit
        if today_pnl >= limit_usd and total_pnl > 0:
            return False, (
                f"🚫 CONSISTENCY — today ${today_pnl:.0f} is "
                f"{today_pct*100:.0f}% of total ${total_pnl:.0f} "
                f"(max {APEX_CONSISTENCY_PCT*100:.0f}%). "
                f"Stop trading today to protect consistency."
            )

        # Warn if close
        if today_pnl >= limit_usd * 0.75 and total_pnl > 0:
            return True, (
                f"⚠️  Consistency: today ${today_pnl:.0f} = "
                f"{today_pct*100:.0f}% of total "
                f"(limit: {APEX_CONSISTENCY_PCT*100:.0f}%)"
            )

        return True, (
            f"Consistency: {today_pct*100:.0f}% "
            f"(${today_pnl:.0f}/${total_pnl:.0f})"
        )

    # ── Master check ──────────────────────────────────────────
    def check(self, signal: dict = None) -> tuple[bool, str]:
        """
        Run all 5 Apex rules. Returns (approved, reason).
        Call before every trade.
        """
        self.reload()

        # Check blown/passed first
        if self.state.get("blown"):
            return False, f"🚫 EVALUATION BLOWN: {self.state.get('blow_reason','')}"
        if self.state.get("passed"):
            return True, "🎯 EVALUATION PASSED — trading live account"

        results = []

        from src.config import PROP_FIRM_ACCOUNT_TYPE
        ok1, r1 = self._check_daily_drawdown()
        ok2, r2 = self._check_max_drawdown()
        ok3, r3 = self._check_profit_target()    # informational only
        ok4, r4 = self._check_news()
        ok5, r5 = self._check_consistency()      # skipped on eval

        # Mark blown if critical rules failed
        if not ok2:
            self.state["blown"]      = True
            self.state["blow_reason"] = r2
            save_state(self.state)
            _log("BLOWN", r2, self.state)

        # Aggregate result
        all_ok  = ok1 and ok2 and ok4 and ok5
        reasons = [r for ok, r in [(ok1,r1),(ok2,r2),(ok4,r4),(ok5,r5)]
                   if not ok]

        if not all_ok:
            return False, " | ".join(reasons)

        # Warnings (ok but close to limit)
        warnings = [r for ok, r in [(ok1,r1),(ok2,r2),(ok4,r4),(ok5,r5)]
                    if ok and ("⚠️" in r or "warning" in r.lower())]

        status_line = r3   # profit target progress
        if warnings:
            status_line += " | " + " | ".join(warnings)

        return True, status_line

    # ── PnL update ────────────────────────────────────────────
    def update_pnl(self, pnl_usd: float, trade_info: str = ""):
        """Call this after every trade closes with the realised P&L."""
        self.reload()

        self.state["daily_pnl"]    += pnl_usd
        self.state["total_pnl"]    += pnl_usd
        self.state["current_equity"] = (
            self.state["account_size"] + self.state["total_pnl"]
        )
        self.state["trades_today"] += 1
        self.state["total_trades"] += 1

        # Update peak equity for trailing drawdown
        if self.state["current_equity"] > self.state["peak_equity"]:
            self.state["peak_equity"] = self.state["current_equity"]

        # Update daily peak
        if self.state["daily_pnl"] > self.state["daily_peak"]:
            self.state["daily_peak"] = self.state["daily_pnl"]

        # Check if profit target hit
        from src.config import APEX_PROFIT_TARGET
        if (self.state["total_pnl"] >= APEX_PROFIT_TARGET
                and not self.state["passed"]):
            self.state["passed"] = True
            _log("PASSED", f"Profit target ${APEX_PROFIT_TARGET} reached!", self.state)
            print(f"\n  🎉🎉🎉 APEX EVALUATION PASSED! "
                  f"Total profit: ${self.state['total_pnl']:.0f} 🎉🎉🎉\n")

        save_state(self.state)
        _log("TRADE_CLOSED", f"PnL ${pnl_usd:+.2f} | {trade_info}", self.state)

    # ── Status display ─────────────────────────────────────────
    def status(self) -> dict:
        self.reload()
        s = self.state

        from src.config import (APEX_DAILY_DRAWDOWN, APEX_MAX_DRAWDOWN,
                                 APEX_PROFIT_TARGET, APEX_CONSISTENCY_PCT,
                                 PROP_FIRM_ACCOUNT_TYPE)

        daily_loss    = abs(min(0, s["daily_pnl"]))
        total_dd      = s["peak_equity"] - s["current_equity"]
        consistency   = (s["daily_pnl"] / s["total_pnl"] * 100
                         if s["total_pnl"] > 0 else 0)
        progress_pct  = s["total_pnl"] / APEX_PROFIT_TARGET * 100

        news_clear, news_reason = check_news_clear()

        print(f"""
  ┌─────────────────────────────────────────────────┐
  │  🏛️  APEX $50k — {PROP_FIRM_ACCOUNT_TYPE.upper()} ACCOUNT
  ├─────────────────────────────────────────────────┤
  │  Equity      : ${s['current_equity']:>10,.2f}
  │  Total P&L   : ${s['total_pnl']:>+10,.2f}
  │  Today P&L   : ${s['daily_pnl']:>+10,.2f}
  ├─────────────────────────────────────────────────┤
  │  Rule 1 — Daily DD     : ${daily_loss:>6,.0f} / $1,000
  │    {'🟢 OK' if daily_loss < APEX_DAILY_DRAWDOWN * 0.75 else '🟡 NEAR LIMIT' if daily_loss < APEX_DAILY_DRAWDOWN else '🔴 LIMIT HIT'}  ${APEX_DAILY_DRAWDOWN - daily_loss:.0f} remaining
  │
  │  Rule 2 — Max DD       : ${total_dd:>6,.0f} / $2,000
  │    {'🟢 OK' if total_dd < APEX_MAX_DRAWDOWN * 0.75 else '🟡 NEAR LIMIT' if total_dd < APEX_MAX_DRAWDOWN else '🔴 BLOWN'}  peak was ${s['peak_equity']:,.0f}
  │
  │  Rule 3 — Profit Target: ${s['total_pnl']:>6,.0f} / $3,000  ({progress_pct:.1f}%)
  │    {'🎯 PASSED!' if s['passed'] else '🟢 On track' if progress_pct > 0 else '⬜ Not started'}
  │
  │  Rule 4 — News Filter  : {'🟢 CLEAR' if news_clear else '🔴 BLOCKED'}
  │    {news_reason}
  │
  │  Rule 5 — Consistency  : {'⏭  N/A (eval account)' if PROP_FIRM_ACCOUNT_TYPE != 'performance' else f'{consistency:.1f}% today / 50% limit'}
  │    {'⏭  Only applies on performance account' if PROP_FIRM_ACCOUNT_TYPE != 'performance' else ('🟢 OK' if consistency < 40 else '🟡 NEAR LIMIT' if consistency < 50 else '🔴 LIMIT HIT')}
  ├─────────────────────────────────────────────────┤
  │  Trades today : {s['trades_today']}  |  Total: {s['total_trades']}
  │  Status       : {'💥 BLOWN' if s['blown'] else '🎉 PASSED' if s['passed'] else '⚡ ACTIVE'}
  └─────────────────────────────────────────────────┘""")

        return {
            "equity":        s["current_equity"],
            "total_pnl":     s["total_pnl"],
            "daily_pnl":     s["daily_pnl"],
            "daily_dd":      daily_loss,
            "total_dd":      total_dd,
            "consistency":   consistency,
            "news_clear":    news_clear,
            "passed":        s["passed"],
            "blown":         s["blown"],
        }

    # ── Manual PnL input (for manual trades) ─────────────────
    def record_trade(self, pnl_usd: float, symbol: str = "",
                     direction: str = ""):
        """
        Manually record a closed trade's P&L.
        Use this if you placed trades manually on Tradovate.

        python -c "
        from src.agents.apex_risk import ApexRisk
        ApexRisk().record_trade(+250, 'MES', 'LONG')   # winning trade
        ApexRisk().record_trade(-150, 'MNQ', 'SHORT')  # losing trade
        "
        """
        info = f"{direction} {symbol}".strip()
        self.update_pnl(pnl_usd, info)
        print(f"  ✅ Recorded: ${pnl_usd:+.2f} {info}")
        self.status()


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="🏛️ Apex Risk Monitor")
    p.add_argument("--status",  action="store_true", help="Show current status")
    p.add_argument("--news",    action="store_true", help="Check news events")
    p.add_argument("--record",  type=float, metavar="PNL",
                   help="Record a trade P&L (e.g. --record 250 or --record -150)")
    p.add_argument("--symbol",  default="", help="Symbol for record")
    p.add_argument("--dir",     default="", help="Direction for record")
    p.add_argument("--reset",   action="store_true", help="Reset state (new eval)")
    args = p.parse_args()

    apex = ApexRisk()

    if args.reset:
        STATE_FILE.write_text(json.dumps(_default_state(), indent=2))
        print("✅ Apex state reset — new evaluation started")

    elif args.news:
        events = fetch_news_events(force_refresh=True)
        clear, reason = check_news_clear()
        print(f"\n📰 High-impact USD news today: {len(events)} events")
        for e in events:
            print(f"  • {e['title']} @ {e['time']}")
        print(f"\nStatus: {'🟢 CLEAR' if clear else '🔴 BLOCKED'} — {reason}")

    elif args.record is not None:
        apex.record_trade(args.record, args.symbol, args.dir)

    else:
        apex.status()
        _, reason = apex.check()
        print(f"\n  Next trade: {reason}")
