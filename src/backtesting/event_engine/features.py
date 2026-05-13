from __future__ import annotations

import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(out["Close"], 20)
    out["ema50"] = ema(out["Close"], 50)
    out["ema200"] = ema(out["Close"], 200)
    out["atr14"] = atr(out, 14)
    out["roll_high_20"] = out["High"].rolling(20, min_periods=20).max().shift(1)
    out["roll_low_20"] = out["Low"].rolling(20, min_periods=20).min().shift(1)
    out["roll_high_60"] = out["High"].rolling(60, min_periods=60).max().shift(1)
    out["roll_low_60"] = out["Low"].rolling(60, min_periods=60).min().shift(1)
    out["body"] = (out["Close"] - out["Open"]).abs()
    out["range"] = out["High"] - out["Low"]
    out["close_pos"] = (out["Close"] - out["Low"]) / out["range"].replace(0, pd.NA)
    return out
