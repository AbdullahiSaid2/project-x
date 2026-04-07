from __future__ import annotations

import textwrap

try:
    from src.strategies.families.schema import StrategySchema
except Exception:
    StrategySchema = object


RUNTIME_IMPORT_BLOCK = """
import numpy as np
import pandas as pd
from backtesting import Strategy
from src.strategies.families.wrappers import (
    ind_rsi,
    ind_atr,
    ind_ema,
    ind_sma,
    ind_macd,
    ind_macd_signal,
    ind_bb_low,
    ind_bb_high,
    ind_bb_mid,
)
""".strip()


def _append_class_methods(class_code: str, methods_code: str) -> str:
    methods = textwrap.dedent(methods_code).strip("\n")
    methods = textwrap.indent(methods, "    ")
    return class_code.rstrip() + "\n\n" + methods + "\n"


def _schema_attr(schema, path, default):
    try:
        root, key = path
        value = getattr(schema, root, {}).get(key, default)
        if value is None:
            return default
        return value
    except Exception:
        return default


def _schema_bool(schema, path, default):
    return "True" if bool(_schema_attr(schema, path, default)) else "False"


def _schema_num(schema, path, default):
    return repr(_schema_attr(schema, path, default))


def _common_header(name: str, schema) -> str:
    tf_hint = getattr(schema, "timeframe_hint", "15m").lower()
    is_htf = tf_hint in {"4h", "1d"}

    session_filter_default = _schema_attr(schema, ("setup_params", "use_session_filter"), True)
    trend_filter_default = _schema_attr(schema, ("setup_params", "use_trend_filter"), True)

    if is_htf:
        session_filter_default = False

    return textwrap.dedent(f"""
class {name}(Strategy):
    rsi_window = {_schema_num(schema, ("indicator_params", "rsi_window"), 14)}
    atr_window = {_schema_num(schema, ("indicator_params", "atr_window"), 14)}
    ema_fast = {_schema_num(schema, ("indicator_params", "ema_fast"), 20)}
    ema_slow = {_schema_num(schema, ("indicator_params", "ema_slow"), 50)}
    bb_window = {_schema_num(schema, ("indicator_params", "bb_window"), 20)}
    bb_std = {_schema_num(schema, ("indicator_params", "bb_std"), 2.0)}

    sl_atr_mult = {_schema_num(schema, ("risk_params", "sl_atr_mult"), 1.0)}
    tp_r_multiple = {_schema_num(schema, ("risk_params", "tp_r_multiple"), 1.5)}
    fixed_size = {_schema_num(schema, ("risk_params", "fixed_size"), 0.1)}

    lookback = {_schema_num(schema, ("setup_params", "lookback"), 20)}
    price_tolerance_pct = {_schema_num(schema, ("setup_params", "price_tolerance_pct"), 0.002)}
    break_buffer_pct = {_schema_num(schema, ("setup_params", "break_buffer_pct"), 0.0002)}
    retest_tolerance_pct = {_schema_num(schema, ("setup_params", "retest_tolerance_pct"), 0.001)}
    volume_multiplier = {_schema_num(schema, ("setup_params", "volume_multiplier"), 1.0)}
    large_bar_atr_mult = {_schema_num(schema, ("setup_params", "large_bar_atr_mult"), 0.0)}
    max_retest_bars = {_schema_num(schema, ("setup_params", "max_retest_bars"), 3)}

    retest_required = {_schema_bool(schema, ("setup_params", "retest_required"), False)}
    volume_confirmation = {_schema_bool(schema, ("setup_params", "volume_confirmation"), False)}
    large_bar_confirmation = {_schema_bool(schema, ("setup_params", "large_bar_confirmation"), False)}
    rejection_confirmation = {_schema_bool(schema, ("setup_params", "rejection_confirmation"), False)}
    close_confirmation = {_schema_bool(schema, ("setup_params", "close_confirmation"), False)}

    use_session_filter = {"True" if session_filter_default else "False"}
    use_volatility_filter = {_schema_bool(schema, ("setup_params", "use_volatility_filter"), True)}
    use_trend_filter = {"True" if trend_filter_default else "False"}

    direction = {repr(getattr(schema, "direction", "long_only"))}
    timeframe_hint = {repr(tf_hint)}

    def init(self):
        self.rsi = self.I(lambda x: ind_rsi(x, window=self.rsi_window), self.data.Close)
        self.atr = self.I(
            lambda h, l, c: ind_atr(h, l, c, window=self.atr_window),
            self.data.High, self.data.Low, self.data.Close
        )
        self.ema_fast_line = self.I(lambda x: ind_ema(x, window=self.ema_fast), self.data.Close)
        self.ema_slow_line = self.I(lambda x: ind_ema(x, window=self.ema_slow), self.data.Close)
        self.bb_low = self.I(lambda x: ind_bb_low(x, window=self.bb_window, window_dev=self.bb_std), self.data.Close)
        self.bb_high = self.I(lambda x: ind_bb_high(x, window=self.bb_window, window_dev=self.bb_std), self.data.Close)
        self.bb_mid = self.I(lambda x: ind_bb_mid(x, window=self.bb_window, window_dev=self.bb_std), self.data.Close)

        self.breakout_state = 0
        self.breakout_level = np.nan
        self.breakout_bar_index = -1
        self.breakout_direction = ""
        self.breakout_trigger_high = np.nan
        self.breakout_trigger_low = np.nan

    def _bar_index(self):
        return len(self.data) - 1

    def _hour_of_bar(self):
        try:
            idx = self.data.index[-1]
            if hasattr(idx, "hour"):
                return int(idx.hour)
        except Exception:
            pass
        return 12

    def _session_ok(self):
        if not self.use_session_filter:
            return True
        h = self._hour_of_bar()
        return h in (7, 8, 9, 10, 13, 14, 15, 16)

    def _volatility_ok(self):
        if not self.use_volatility_filter:
            return True
        if len(self.data) < 30:
            return False
        atr_now = float(self.atr[-1]) if self.atr[-1] == self.atr[-1] else 0.0
        close = float(self.data.Close[-1])
        if close <= 0:
            return False
        atr_pct = atr_now / close
        return 0.001 <= atr_pct <= 0.05

    def _trend_long_ok(self):
        if not self.use_trend_filter:
            return True
        return self.ema_fast_line[-1] > self.ema_slow_line[-1]

    def _trend_short_ok(self):
        if not self.use_trend_filter:
            return True
        return self.ema_fast_line[-1] < self.ema_slow_line[-1]

    def _allow_long(self):
        return self.direction in ("long_only", "both")

    def _allow_short(self):
        return self.direction in ("short_only", "both")

    def _base_ok(self):
        return len(self.data) >= max(60, self.lookback + 5) and self._session_ok() and self._volatility_ok()

    def _bar_body_large_bull(self):
        c = float(self.data.Close[-1])
        o = float(self.data.Open[-1])
        threshold = float(self.atr[-1]) * self.large_bar_atr_mult
        return c > o and ((c - o) > threshold if self.large_bar_confirmation else True)

    def _bar_body_large_bear(self):
        c = float(self.data.Close[-1])
        o = float(self.data.Open[-1])
        threshold = float(self.atr[-1]) * self.large_bar_atr_mult
        return c < o and ((o - c) > threshold if self.large_bar_confirmation else True)

    def _volume_ok(self):
        if not self.volume_confirmation:
            return True
        try:
            recent_avg = float(np.mean(self.data.Volume[-6:-1]))
            current = float(self.data.Volume[-1])
            if recent_avg <= 0:
                return current > float(self.data.Volume[-2])
            return current >= recent_avg * self.volume_multiplier
        except Exception:
            return True

    def _enter_long(self, sl, entry=None):
        entry_price = float(self.data.Close[-1] if entry is None else entry)
        risk = entry_price - sl
        if risk <= 0:
            return
        tp = entry_price + (risk * self.tp_r_multiple)
        if self.position.is_short:
            self.position.close()
        if not self.position:
            self.buy(size=self.fixed_size, sl=sl, tp=tp)

    def _enter_short(self, sl, entry=None):
        entry_price = float(self.data.Close[-1] if entry is None else entry)
        risk = sl - entry_price
        if risk <= 0:
            return
        tp = entry_price - (risk * self.tp_r_multiple)
        if self.position.is_long:
            self.position.close()
        if not self.position:
            self.sell(size=self.fixed_size, sl=sl, tp=tp)

    def _reset_breakout_state(self):
        self.breakout_state = 0
        self.breakout_level = np.nan
        self.breakout_bar_index = -1
        self.breakout_direction = ""
        self.breakout_trigger_high = np.nan
        self.breakout_trigger_low = np.nan

    def _breakout_state_expired(self):
        if self.breakout_state == 0:
            return False
        return (self._bar_index() - self.breakout_bar_index) > self.max_retest_bars
""").strip()


def _double_bottom(name: str, schema) -> str:
    body = """
def next(self):
    if not self._base_ok():
        return

    close = float(self.data.Close[-1])
    low_now = float(self.data.Low[-1])
    low_prev = float(min(self.data.Low[-self.lookback:-4]))
    mid_high = float(max(self.data.High[-8:-1]))
    tol = close * self.price_tolerance_pct

    double_bottom = abs(low_now - low_prev) <= tol
    rsi_confirm = self.rsi[-1] > self.rsi[-2] and self.rsi[-1] > 40

    if self._allow_long() and self._trend_long_ok() and double_bottom and rsi_confirm and close > mid_high:
        sl = low_now - (float(self.atr[-1]) * self.sl_atr_mult)
        self._enter_long(sl=sl, entry=close)

    if self._allow_short() and self._trend_short_ok():
        high_now = float(self.data.High[-1])
        high_prev = float(max(self.data.High[-self.lookback:-4]))
        mid_low = float(min(self.data.Low[-8:-1]))
        double_top = abs(high_now - high_prev) <= tol
        rsi_confirm_short = self.rsi[-1] < self.rsi[-2] and self.rsi[-1] < 60
        if double_top and rsi_confirm_short and close < mid_low:
            sl = high_now + (float(self.atr[-1]) * self.sl_atr_mult)
            self._enter_short(sl=sl, entry=close)
"""
    return _append_class_methods(_common_header(name, schema), body)


def _inside_bar(name: str, schema) -> str:
    body = """
def next(self):
    if not self._base_ok():
        return

    mother_high = float(self.data.High[-3])
    mother_low = float(self.data.Low[-3])
    inside_high = float(self.data.High[-2])
    inside_low = float(self.data.Low[-2])
    close = float(self.data.Close[-1])

    is_inside = inside_high <= mother_high and inside_low >= mother_low
    if not is_inside:
        return

    if self.large_bar_confirmation:
        if not (self._bar_body_large_bull() or self._bar_body_large_bear()):
            return

    if not self._volume_ok():
        return

    if self._allow_long() and self._trend_long_ok() and close > inside_high:
        sl = inside_low - (float(self.atr[-1]) * 0.25)
        self._enter_long(sl=sl, entry=close)
    elif self._allow_short() and self._trend_short_ok() and close < inside_low:
        sl = inside_high + (float(self.atr[-1]) * 0.25)
        self._enter_short(sl=sl, entry=close)
"""
    return _append_class_methods(_common_header(name, schema), body)


def _three_bar(name: str, schema) -> str:
    body = """
def next(self):
    if not self._base_ok():
        return

    o1, h1, l1, c1 = float(self.data.Open[-3]), float(self.data.High[-3]), float(self.data.Low[-3]), float(self.data.Close[-3])
    o2, h2, l2, c2 = float(self.data.Open[-2]), float(self.data.High[-2]), float(self.data.Low[-2]), float(self.data.Close[-2])
    o3, h3, l3, c3 = float(self.data.Open[-1]), float(self.data.High[-1]), float(self.data.Low[-1]), float(self.data.Close[-1])

    range2 = max(h2 - l2, 1e-9)
    doji = abs(c2 - o2) / range2 <= 0.35

    bullish = c1 < o1 and (h1 - l1) >= float(self.atr[-1]) and doji and c3 > h2
    bearish = c1 > o1 and (h1 - l1) >= float(self.atr[-1]) and doji and c3 < l2

    if self._allow_long() and self._trend_long_ok() and bullish and self._volume_ok():
        sl = l2 - (float(self.atr[-1]) * 0.25)
        self._enter_long(sl=sl, entry=c3)
    elif self._allow_short() and self._trend_short_ok() and bearish and self._volume_ok():
        sl = h2 + (float(self.atr[-1]) * 0.25)
        self._enter_short(sl=sl, entry=c3)
"""
    return _append_class_methods(_common_header(name, schema), body)


def _mean_reversion(name: str, schema) -> str:
    body = """
def next(self):
    if not self._base_ok():
        return

    close = float(self.data.Close[-1])

    long_signal = self._allow_long() and self._trend_long_ok() and self.rsi[-1] < 28 and close < self.bb_low[-1]
    short_signal = self._allow_short() and self._trend_short_ok() and self.rsi[-1] > 72 and close > self.bb_high[-1]

    if self.large_bar_confirmation:
        long_signal = long_signal and self._bar_body_large_bull()
        short_signal = short_signal and self._bar_body_large_bear()

    if not self._volume_ok():
        return

    if long_signal:
        sl = float(self.data.Low[-1]) - (float(self.atr[-1]) * self.sl_atr_mult)
        self._enter_long(sl=sl, entry=close)
    elif short_signal:
        sl = float(self.data.High[-1]) + (float(self.atr[-1]) * self.sl_atr_mult)
        self._enter_short(sl=sl, entry=close)
"""
    return _append_class_methods(_common_header(name, schema), body)


def _breakout(name: str, schema) -> str:
    body = """
def next(self):
    if not self._base_ok():
        return

    close = float(self.data.Close[-1])
    high = float(self.data.High[-1])
    low = float(self.data.Low[-1])

    lookback_high = float(max(self.data.High[-(self.lookback + 1):-1]))
    lookback_low = float(min(self.data.Low[-(self.lookback + 1):-1]))

    broke_above_now = close > lookback_high * (1 + self.break_buffer_pct)
    broke_below_now = close < lookback_low * (1 - self.break_buffer_pct)

    if self._breakout_state_expired():
        self._reset_breakout_state()

    if self.breakout_state == 0:
        if self._allow_long() and self._trend_long_ok() and broke_above_now:
            if (not self.large_bar_confirmation) or self._bar_body_large_bull():
                if self._volume_ok():
                    self.breakout_state = 1
                    self.breakout_level = lookback_high
                    self.breakout_bar_index = self._bar_index()
                    self.breakout_direction = "long"
                    self.breakout_trigger_high = high
                    self.breakout_trigger_low = low

        elif self._allow_short() and self._trend_short_ok() and broke_below_now:
            if (not self.large_bar_confirmation) or self._bar_body_large_bear():
                if self._volume_ok():
                    self.breakout_state = 1
                    self.breakout_level = lookback_low
                    self.breakout_bar_index = self._bar_index()
                    self.breakout_direction = "short"
                    self.breakout_trigger_high = high
                    self.breakout_trigger_low = low

        if not self.retest_required:
            if self._allow_long() and self._trend_long_ok() and broke_above_now and self._volume_ok():
                sl = min(low, lookback_high) - (float(self.atr[-1]) * self.sl_atr_mult)
                self._enter_long(sl=sl, entry=close)
            elif self._allow_short() and self._trend_short_ok() and broke_below_now and self._volume_ok():
                sl = max(high, lookback_low) + (float(self.atr[-1]) * self.sl_atr_mult)
                self._enter_short(sl=sl, entry=close)
            return

    if self.breakout_state != 1:
        return

    level = float(self.breakout_level)
    if self.breakout_direction == "long":
        touched = low <= level * (1 + self.retest_tolerance_pct)
        rejection = low <= level and close > level
        close_ok = close > level if self.close_confirmation else True
        signal = touched
        if self.rejection_confirmation:
            signal = signal and rejection
        signal = signal and close_ok and self._trend_long_ok() and self._volume_ok()
        if signal:
            sl = min(low, level) - (float(self.atr[-1]) * self.sl_atr_mult)
            self._enter_long(sl=sl, entry=close)
            self._reset_breakout_state()
        elif close < level * (1 - self.retest_tolerance_pct * 2):
            self._reset_breakout_state()

    elif self.breakout_direction == "short":
        touched = high >= level * (1 - self.retest_tolerance_pct)
        rejection = high >= level and close < level
        close_ok = close < level if self.close_confirmation else True
        signal = touched
        if self.rejection_confirmation:
            signal = signal and rejection
        signal = signal and close_ok and self._trend_short_ok() and self._volume_ok()
        if signal:
            sl = max(high, level) + (float(self.atr[-1]) * self.sl_atr_mult)
            self._enter_short(sl=sl, entry=close)
            self._reset_breakout_state()
        elif close > level * (1 + self.retest_tolerance_pct * 2):
            self._reset_breakout_state()
"""
    return _append_class_methods(_common_header(name, schema), body)


def _ict_fvg(name: str, schema) -> str:
    body = """
def next(self):
    if not self._base_ok():
        return

    h1 = float(self.data.High[-3])
    l1 = float(self.data.Low[-3])
    h3 = float(self.data.High[-1])
    l3 = float(self.data.Low[-1])
    close = float(self.data.Close[-1])

    bullish_gap_exists = l3 > h1
    bearish_gap_exists = h3 < l1

    bull_displacement = float(self.data.Close[-2]) > h1 and self._trend_long_ok() and self.rsi[-1] > 50
    bear_displacement = float(self.data.Close[-2]) < l1 and self._trend_short_ok() and self.rsi[-1] < 50

    if self._allow_long() and bullish_gap_exists and bull_displacement and self._volume_ok():
        sl = l3 - (float(self.atr[-1]) * 0.5)
        self._enter_long(sl=sl, entry=close)

    elif self._allow_short() and bearish_gap_exists and bear_displacement and self._volume_ok():
        sl = h3 + (float(self.atr[-1]) * 0.5)
        self._enter_short(sl=sl, entry=close)
"""
    return _append_class_methods(_common_header(name, schema), body)


def _ict_liquidity(name: str, schema) -> str:
    body = """
def next(self):
    if not self._base_ok():
        return

    prev_high = float(max(self.data.High[-15:-1]))
    prev_low = float(min(self.data.Low[-15:-1]))
    close = float(self.data.Close[-1])
    high = float(self.data.High[-1])
    low = float(self.data.Low[-1])

    bullish_sweep = low < prev_low and close > prev_low and self._trend_long_ok() and self.rsi[-1] > self.rsi[-2]
    bearish_sweep = high > prev_high and close < prev_high and self._trend_short_ok() and self.rsi[-1] < self.rsi[-2]

    if not self._volume_ok():
        return

    if self._allow_long() and bullish_sweep:
        sl = low - (float(self.atr[-1]) * 0.5)
        self._enter_long(sl=sl, entry=close)

    elif self._allow_short() and bearish_sweep:
        sl = high + (float(self.atr[-1]) * 0.5)
        self._enter_short(sl=sl, entry=close)
"""
    return _append_class_methods(_common_header(name, schema), body)


def compile_strategy_class(schema, class_name: str = "GeneratedStrategy") -> str:
    family = getattr(schema, "family", "mean_reversion")

    if family == "double_bottom":
        return _double_bottom(class_name, schema)
    if family == "inside_bar":
        return _inside_bar(class_name, schema)
    if family == "three_bar":
        return _three_bar(class_name, schema)
    if family == "breakout":
        return _breakout(class_name, schema)
    if family == "ict_fvg":
        return _ict_fvg(class_name, schema)
    if family == "ict_liquidity_sweep":
        return _ict_liquidity(class_name, schema)

    return _mean_reversion(class_name, schema)