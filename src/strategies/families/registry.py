from __future__ import annotations

import re


def _extract_timeframe(text: str) -> str:
    patterns = [
        (r"\b1\s*[- ]?\s*minute\b", "1m"),
        (r"\b5\s*[- ]?\s*minute\b", "5m"),
        (r"\b15\s*[- ]?\s*minute\b", "15m"),
        (r"\b30\s*[- ]?\s*minute\b", "30m"),
        (r"\b1\s*[- ]?\s*hour\b", "1h"),
        (r"\b4\s*[- ]?\s*hour\b", "4h"),
        (r"\b1\s*[- ]?\s*day\b", "1d"),
        (r"\b1m\b", "1m"),
        (r"\b5m\b", "5m"),
        (r"\b15m\b", "15m"),
        (r"\b30m\b", "30m"),
        (r"\b1h\b", "1h"),
        (r"\b4h\b", "4h"),
        (r"\b1d\b", "1d"),
    ]
    for pattern, tf in patterns:
        if re.search(pattern, text):
            return tf
    return "15m"


def _extract_lookback(text: str, default: int = 20) -> int:
    m = re.search(r"prior\s+(\d+)[-\s]*bar", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)[-\s]*bar", text)
    if m:
        return int(m.group(1))
    return default


def _extract_rr(text: str, default: float = 1.5) -> float:
    m = re.search(r"1\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    return default


def _extract_stop_atr_mult(text: str, default: float = 1.0) -> float:
    m = re.search(r"stop.*?([0-9]+(?:\.[0-9]+)?)\s*x?\s*atr", text)
    if m:
        return float(m.group(1))
    return default


def heuristic_schema_from_idea(idea: str):
    text = idea.lower()

    direction = "long_only"
    if any(x in text for x in ["short", "sell", "bearish"]):
        direction = "short_only"
    elif "both" in text or ("long" in text and "short" in text):
        direction = "both"

    if "fair value gap" in text or "fvg" in text:
        family = "ict_fvg"
    elif "liquidity" in text or "sweep" in text:
        family = "ict_liquidity_sweep"
    elif "inside bar" in text:
        family = "inside_bar"
    elif "three bar" in text or "three-bar" in text or "doji" in text:
        family = "three_bar"
    elif "double bottom" in text or "double top" in text or "two lows" in text or "two highs" in text:
        family = "double_bottom"
    elif any(x in text for x in ["breakout", "retest", "break below", "break above", "breaks below", "breaks above", "broken level"]):
        family = "breakout"
    else:
        family = "mean_reversion"

    tf = _extract_timeframe(text)
    lookback = _extract_lookback(text, 20)
    rr = _extract_rr(text, 1.5)
    sl_atr_mult = _extract_stop_atr_mult(text, 1.0)

    volume_confirmation = "volume" in text
    retest_required = "retest" in text or "retests" in text or "broken level" in text
    large_bar_confirmation = any(x in text for x in ["large red bar", "large green bar", "displacement", "impulsive"])
    rejection_confirmation = any(x in text for x in ["reverses", "rejects", "rejection", "closes back below", "closes back above"])
    close_confirmation = any(x in text for x in ["close below", "closes below", "close above", "closes above"])

    strict_mode = True
    if tf in {"1m", "5m"}:
        strict_mode = False

    indicator_params = {
        "rsi_window": 14,
        "atr_window": 14,
        "ema_fast": 20,
        "ema_slow": 50,
        "bb_window": 20,
        "bb_std": 2.0,
    }

    setup_params = {
        "lookback": lookback,
        "retest_required": retest_required,
        "volume_confirmation": volume_confirmation,
        "large_bar_confirmation": large_bar_confirmation,
        "rejection_confirmation": rejection_confirmation,
        "close_confirmation": close_confirmation,
        "max_retest_bars": 4 if tf in {"1m", "5m", "15m"} else 2,
        "price_tolerance_pct": 0.002,
        "break_buffer_pct": 0.0002,
        "retest_tolerance_pct": 0.001,
        "volume_multiplier": 1.05 if volume_confirmation else 1.0,
        "large_bar_atr_mult": 0.2 if large_bar_confirmation else 0.0,
        "use_session_filter": strict_mode,
        "use_volatility_filter": True,
        "use_trend_filter": strict_mode,
        "strict_mode": strict_mode,
        "use_regime_filter": family == "breakout",
        "regime_ema_window": 200,
        "min_breakout_range_mult": 1.0 if family == "breakout" else 0.0,
        "move_to_be_at_r": 1.0 if family == "breakout" else 0.0,
        "partial_tp_at_r": 1.0 if family == "breakout" else 0.0,
        "partial_tp_size": 0.5 if family == "breakout" else 0.0,
        "trail_atr_after_r": 1.0 if family == "breakout" else 0.0,
        "failure_exit_on_level_reclaim": family == "breakout",
    }

    risk_params = {
        "sl_atr_mult": sl_atr_mult,
        "tp_r_multiple": rr,
        "fixed_size": 0.1,
    }

    if family == "double_bottom":
        setup_params["price_tolerance_pct"] = 0.001

    if family == "ict_fvg":
        setup_params["retest_required"] = True
        setup_params["large_bar_confirmation"] = True
        setup_params["rejection_confirmation"] = False

    if family == "ict_liquidity_sweep":
        setup_params["retest_required"] = True
        setup_params["rejection_confirmation"] = True

    return {
        "family": family,
        "name": family.replace("_", " ").title().replace("Ict", "ICT"),
        "description": idea,
        "direction": direction,
        "timeframe_hint": tf,
        "trade_frequency_target": "medium",
        "indicator_params": indicator_params,
        "setup_params": setup_params,
        "risk_params": risk_params,
        "family_confidence": 0.84,
        "codifiability_score": 8.0,
        "ambiguity_score": 2.2,
        "normalized_idea": idea,
    }