from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class FVGZone:
    direction: int
    top: float
    bottom: float
    midpoint: float
    birth_index: int


def detect_fvg(df: pd.DataFrame, min_gap_pct: float = 0.0) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float).replace(0, np.nan)

    bullish_gap = low > high.shift(2)
    bearish_gap = high < low.shift(2)

    bull_top = low.where(bullish_gap)
    bull_bottom = high.shift(2).where(bullish_gap)

    bear_top = low.shift(2).where(bearish_gap)
    bear_bottom = high.where(bearish_gap)

    if min_gap_pct > 0:
        bull_gap_size = (bull_top - bull_bottom) / close
        bear_gap_size = (bear_top - bear_bottom) / close
        bullish_gap = bullish_gap & (bull_gap_size >= min_gap_pct)
        bearish_gap = bearish_gap & (bear_gap_size >= min_gap_pct)
        bull_top = bull_top.where(bullish_gap)
        bull_bottom = bull_bottom.where(bullish_gap)
        bear_top = bear_top.where(bearish_gap)
        bear_bottom = bear_bottom.where(bearish_gap)

    out = pd.DataFrame(index=df.index)
    out["bullish_fvg"] = bullish_gap.fillna(False)
    out["bullish_fvg_top"] = bull_top
    out["bullish_fvg_bottom"] = bull_bottom
    out["bullish_fvg_mid"] = (bull_top + bull_bottom) / 2.0

    out["bearish_fvg"] = bearish_gap.fillna(False)
    out["bearish_fvg_top"] = bear_top
    out["bearish_fvg_bottom"] = bear_bottom
    out["bearish_fvg_mid"] = (bear_top + bear_bottom) / 2.0
    return out


def detect_displacement(
    df: pd.DataFrame,
    atr_period: int = 14,
    body_atr_mult: float = 1.0,
) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(atr_period).mean()
    body = (close - open_).abs()

    bullish = (close > open_) & (body > atr * body_atr_mult)
    bearish = (close < open_) & (body > atr * body_atr_mult)

    out = pd.DataFrame(index=df.index)
    out["atr"] = atr
    out["body"] = body
    out["bullish_displacement"] = bullish.fillna(False)
    out["bearish_displacement"] = bearish.fillna(False)
    out["displacement_mid"] = (high + low) / 2.0
    return out


def detect_swing_points(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)

    for i in range(left, len(df) - right):
        h = high.iloc[i]
        l = low.iloc[i]

        if h >= high.iloc[i - left:i].max() and h > high.iloc[i + 1:i + right + 1].max():
            swing_high.iloc[i] = True
        if l <= low.iloc[i - left:i].min() and l < low.iloc[i + 1:i + right + 1].min():
            swing_low.iloc[i] = True

    out = pd.DataFrame(index=df.index)
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    out["swing_high_price"] = high.where(swing_high)
    out["swing_low_price"] = low.where(swing_low)
    return out


def detect_structure_shift(
    df: pd.DataFrame,
    swing_left: int = 2,
    swing_right: int = 2,
) -> pd.DataFrame:
    swings = detect_swing_points(df, left=swing_left, right=swing_right)
    close = df["Close"].astype(float)

    last_swing_high = swings["swing_high_price"].ffill()
    last_swing_low = swings["swing_low_price"].ffill()

    bullish_shift = close > last_swing_high.shift(1)
    bearish_shift = close < last_swing_low.shift(1)

    out = pd.DataFrame(index=df.index)
    out["last_swing_high"] = last_swing_high
    out["last_swing_low"] = last_swing_low
    out["bullish_structure_shift"] = bullish_shift.fillna(False)
    out["bearish_structure_shift"] = bearish_shift.fillna(False)
    return out


def detect_cisd(
    df: pd.DataFrame,
    atr_period: int = 14,
    min_body_atr_mult: float = 0.8,
) -> pd.DataFrame:
    disp = detect_displacement(df, atr_period=atr_period, body_atr_mult=min_body_atr_mult)
    shift = detect_structure_shift(df)

    out = pd.DataFrame(index=df.index)
    out["bullish_cisd"] = (disp["bullish_displacement"] & shift["bullish_structure_shift"]).fillna(False)
    out["bearish_cisd"] = (disp["bearish_displacement"] & shift["bearish_structure_shift"]).fillna(False)
    return out


def detect_pd_array_context(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    range_high = high.rolling(lookback).max()
    range_low = low.rolling(lookback).min()
    eq = (range_high + range_low) / 2.0

    out = pd.DataFrame(index=df.index)
    out["range_high"] = range_high
    out["range_low"] = range_low
    out["eq"] = eq
    out["in_discount"] = (close < eq).fillna(False)
    out["in_premium"] = (close > eq).fillna(False)
    return out


def build_ict_feature_frame(
    df: pd.DataFrame,
    *,
    fvg_gap_pct: float = 0.0,
    atr_period: int = 14,
    displacement_atr_mult: float = 1.0,
    pd_lookback: int = 20,
) -> pd.DataFrame:
    fvg = detect_fvg(df, min_gap_pct=fvg_gap_pct)
    disp = detect_displacement(df, atr_period=atr_period, body_atr_mult=displacement_atr_mult)
    shift = detect_structure_shift(df)
    cisd = detect_cisd(df, atr_period=atr_period, min_body_atr_mult=max(0.8, displacement_atr_mult * 0.8))
    pd_ctx = detect_pd_array_context(df, lookback=pd_lookback)

    return pd.concat([fvg, disp, shift, cisd, pd_ctx], axis=1)


def parse_ict_flags_from_idea(idea: str) -> Dict[str, Any]:
    text = idea.lower()

    return {
        "uses_fvg": "fvg" in text or "fair value gap" in text,
        "uses_cisd": "cisd" in text or "change in state of delivery" in text,
        "uses_smt": "smt" in text,
        "uses_pd_array": "pd array" in text or "premium" in text or "discount" in text or "eq" in text,
        "uses_liquidity_sweep": "sweep" in text or "liquidity" in text,
        "uses_displacement": "displacement" in text,
        "requires_reclaim": "reclaim" in text,
        "requires_rejection": "reject" in text or "rejection" in text,
    }