from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pandas as pd

ET_TZ = "America/New_York"


def to_et(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET_TZ)


def et_minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute


def parse_hhmm(value: str) -> tuple[int, int]:
    h, m = value.split(":")
    return int(h), int(m)


def is_allowed_futures_time(et_ts: Any, flatten_time: str = "16:50", reopen_time: str = "18:00") -> bool:
    t = to_et(et_ts)
    wd = t.weekday()
    flat_h, flat_m = parse_hhmm(flatten_time)
    open_h, open_m = parse_hhmm(reopen_time)
    mins = et_minutes(t.hour, t.minute)
    flat_mins = et_minutes(flat_h, flat_m)
    open_mins = et_minutes(open_h, open_m)
    if wd == 5:
        return False
    if wd == 6:
        return mins >= open_mins
    if wd == 4 and mins >= flat_mins:
        return False
    if mins < flat_mins:
        return True
    if mins >= open_mins:
        return True
    return False


def should_force_flat(et_ts: Any, flatten_time: str = "16:50") -> bool:
    t = to_et(et_ts)
    wd = t.weekday()
    flat_h, flat_m = parse_hhmm(flatten_time)
    mins = et_minutes(t.hour, t.minute)
    flat_mins = et_minutes(flat_h, flat_m)
    return wd in (0, 1, 2, 3, 4) and flat_mins <= mins < et_minutes(18, 0)


def session_date(et_ts: Any, reopen_time: str = "18:00") -> Optional[str]:
    t = to_et(et_ts)
    if not is_allowed_futures_time(t, reopen_time=reopen_time):
        return None
    open_h, open_m = parse_hhmm(reopen_time)
    if et_minutes(t.hour, t.minute) >= et_minutes(open_h, open_m):
        return str((t + pd.Timedelta(days=1)).date())
    return str(t.date())


def load_news_events(path: Optional[str | Path]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["event_time_et", "event_name", "currency", "impact"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["event_time_et", "event_name", "currency", "impact"])
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["event_time_et", "event_name", "currency", "impact"])
    if "event_time_et" not in df.columns:
        raise ValueError("news_events.csv must contain event_time_et column")
    df = df.copy()
    df["event_dt_et"] = pd.to_datetime(df["event_time_et"], errors="coerce")
    return df.dropna(subset=["event_dt_et"]).sort_values("event_dt_et").reset_index(drop=True)


def news_blackout_status(et_ts: Any, news_events: pd.DataFrame, before_minutes: int, after_minutes: int) -> tuple[bool, str]:
    if news_events is None or news_events.empty:
        return False, ""
    t = to_et(et_ts).tz_localize(None)
    start = t - pd.Timedelta(minutes=after_minutes)
    end = t + pd.Timedelta(minutes=before_minutes)
    hit = news_events[(news_events["event_dt_et"] >= start) & (news_events["event_dt_et"] <= end)]
    if hit.empty:
        return False, ""
    return True, str(hit.iloc[0].get("event_name", "news_event"))
