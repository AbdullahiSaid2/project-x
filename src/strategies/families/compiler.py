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
from src.strategies.families.ict_features import build_ict_feature_frame
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
    regime_ema_window = {_schema_num(schema, ("setup_params", "regime_ema_window"), 200)}

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
    min_breakout_range_mult = {_schema_num(schema, ("setup_params", "min_breakout_range_mult"), 1.0)}
    move_to_be_at_r = {_schema_num(schema, ("setup_params", "move_to_be_at_r"), 0.0)}
    partial_tp_at_r = {_schema_num(schema, ("setup_params", "partial_tp_at_r"), 0.0)}
    partial_tp_size = {_schema_num(schema, ("setup_params", "partial_tp_size"), 0.0)}
    trail_atr_after_r = {_schema_num(schema, ("setup_params", "trail_atr_after_r"), 0.0)}
    fvg_gap_pct = {_schema_num(schema, ("setup_params", "fvg_gap_pct"), 0.0)}
    displacement_atr_mult = {_schema_num(schema, ("setup_params", "displacement_atr_mult"), 1.0)}
    pd_lookback = {_schema_num(schema, ("setup_params", "pd_lookback"), 20)}

    retest_required = {_schema_bool(schema, ("setup_params", "retest_required"), False)}
    volume_confirmation = {_schema_bool(schema, ("setup_params", "volume_confirmation"), False)}
    large_bar_confirmation = {_schema_bool(schema, ("setup_params", "large_bar_confirmation"), False)}
    rejection_confirmation = {_schema_bool(schema, ("setup_params", "rejection_confirmation"), False)}
    close_confirmation = {_schema_bool(schema, ("setup_params", "close_confirmation"), False)}
    use_regime_filter = {_schema_bool(schema, ("setup_params", "use_regime_filter"), False)}
    failure_exit_on_level_reclaim = {_schema_bool(schema, ("setup_params", "failure_exit_on_level_reclaim"), False)}

    use_session_filter = {"True" if session_filter_default else "False"}
    use_volatility_filter = {_schema_bool(schema, ("setup_params", "use_volatility_filter"), True)}
    use_trend_filter = {"True" if trend_filter_default else "False"}

    uses_fvg = {_schema_bool(schema, ("filters", "uses_fvg"), False)}
    uses_cisd = {_schema_bool(schema, ("filters", "uses_cisd"), False)}
    uses_smt = {_schema_bool(schema, ("filters", "uses_smt"), False)}
    uses_pd_array = {_schema_bool(schema, ("filters", "uses_pd_array"), False)}
    uses_liquidity_sweep = {_schema_bool(schema, ("filters", "uses_liquidity_sweep"), False)}
    uses_displacement = {_schema_bool(schema, ("filters", "uses_displacement"), False)}
    requires_reclaim = {_schema_bool(schema, ("filters", "requires_reclaim"), False)}
    requires_rejection = {_schema_bool(schema, ("filters", "requires_rejection"), False)}

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
        self.regime_ema = self.I(lambda x: ind_ema(x, window=self.regime_ema_window), self.data.Close)
        self.bb_low = self.I(lambda x: ind_bb_low(x, window=self.bb_window, window_dev=self.bb_std), self.data.Close)
        self.bb_high = self.I(lambda x: ind_bb_high(x, window=self.bb_window, window_dev=self.bb_std), self.data.Close)
        self.bb_mid = self.I(lambda x: ind_bb_mid(x, window=self.bb_window, window_dev=self.bb_std), self.data.Close)

        self.ict = build_ict_feature_frame(
            self.data.df,
            fvg_gap_pct=self.fvg_gap_pct,
            atr_period=self.atr_window,
            displacement_atr_mult=self.displacement_atr_mult,
            pd_lookback=self.pd_lookback,
        )

        self.breakout_state = 0
        self.breakout_level = np.nan
        self.breakout_bar_index = -1
        self.breakout_direction = ""
        self.breakout_trigger_high = np.nan
        self.breakout_trigger_low = np.nan

        self.active_breakout_level = np.nan
        self.active_risk_per_unit = np.nan

    def _bar_index(self):
        return len(self.data) - 1

    def _latest_trade(self):
        try:
            if self.trades:
                return self.trades[-1]
        except Exception:
            pass
        return None

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

    def _regime_long_ok(self):
        if not self.use_regime_filter:
            return True
        return float(self.data.Close[-1]) > float(self.regime_ema[-1])

    def _regime_short_ok(self):
        if not self.use_regime_filter:
            return True
        return float(self.data.Close[-1]) < float(self.regime_ema[-1])

    def _allow_long(self):
        return self.direction in ("long_only", "both")

    def _allow_short(self):
        return self.direction in ("short_only", "both")

    def _base_ok(self):
        return len(self.data) >= max(60, self.lookback + 5, self.regime_ema_window + 2) and self._session_ok() and self._volatility_ok()

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

    def _bar_range_ok(self):
        if self.min_breakout_range_mult <= 0:
            return True
        rng = float(self.data.High[-1]) - float(self.data.Low[-1])
        recent = []
        for i in range(2, 8):
            if len(self.data) > i:
                recent.append(float(self.data.High[-i]) - float(self.data.Low[-i]))
        median_rng = float(np.median(recent)) if recent else 0.0
        if median_rng <= 0:
            return rng > 0
        return rng >= median_rng * self.min_breakout_range_mult

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

    def _ict_long_ok(self):
        i = len(self.data.Close) - 1
        if i <= 0:
            return False

        long_ok = True

        bullish_fvg_ok = bool(self.ict["bullish_fvg"].iloc[i]) or bool(self.ict["bullish_fvg"].shift(1).iloc[i])
        bullish_cisd_ok = bool(self.ict["bullish_cisd"].iloc[i])
        discount_ok = bool(self.ict["in_discount"].iloc[i])
        bullish_disp_ok = bool(self.ict["bullish_displacement"].iloc[i])
        bullish_shift_ok = bool(self.ict["bullish_structure_shift"].iloc[i])

        if self.uses_fvg:
            long_ok = long_ok and bullish_fvg_ok
        if self.uses_cisd:
            long_ok = long_ok and bullish_cisd_ok
        if self.uses_pd_array:
            long_ok = long_ok and discount_ok
        if self.uses_displacement:
            long_ok = long_ok and bullish_disp_ok
        if self.requires_reclaim:
            long_ok = long_ok and bullish_shift_ok

        return long_ok

    def _ict_short_ok(self):
        i = len(self.data.Close) - 1
        if i <= 0:
            return False

        short_ok = True

        bearish_fvg_ok = bool(self.ict["bearish_fvg"].iloc[i]) or bool(self.ict["bearish_fvg"].shift(1).iloc[i])
        bearish_cisd_ok = bool(self.ict["bearish_cisd"].iloc[i])
        premium_ok = bool(self.ict["in_premium"].iloc[i])
        bearish_disp_ok = bool(self.ict["bearish_displacement"].iloc[i])
        bearish_shift_ok = bool(self.ict["bearish_structure_shift"].iloc[i])

        if self.uses_fvg:
            short_ok = short_ok and bearish_fvg_ok
        if self.uses_cisd:
            short_ok = short_ok and bearish_cisd_ok
        if self.uses_pd_array:
            short_ok = short_ok and premium_ok
        if self.uses_displacement:
            short_ok = short_ok and bearish_disp_ok
        if self.requires_rejection:
            short_ok = short_ok and bearish_shift_ok

        return short_ok

    def _enter_long(self, sl, entry=None):
        entry_price = float(self.data.Close[-1] if entry is None else entry)
        risk = entry_price - sl
        if risk <= 0:
            return
        tp = entry_price + (risk * self.tp_r_multiple)
        if self.position.is_short:
            self.position.close()
        self.buy(size=self.fixed_size, sl=sl, tp=tp)
        self.active_risk_per_unit = risk
        self.active_breakout_level = entry_price

    def _enter_short(self, sl, entry=None):
        entry_price = float(self.data.Close[-1] if entry is None else entry)
        risk = sl - entry_price
        if risk <= 0:
            return
        tp = entry_price - (risk * self.tp_r_multiple)
        if self.position.is_long:
            self.position.close()
        self.sell(size=self.fixed_size, sl=sl, tp=tp)
        self.active_risk_per_unit = risk
        self.active_breakout_level = entry_price

    def _manage_open_trade(self):
        trade = self._latest_trade()
        if trade is None or not self.position:
            return

        price = float(self.data.Close[-1])
        atr_now = float(self.atr[-1]) if self.atr[-1] == self.atr[-1] else 0.0

        if self.active_risk_per_unit and self.active_risk_per_unit == self.active_risk_per_unit:
            if self.position.is_long:
                r_now = (price - float(trade.entry_price)) / self.active_risk_per_unit
                if self.failure_exit_on_level_reclaim and self.active_breakout_level == self.active_breakout_level:
                    if price < self.active_breakout_level * (1 - self.retest_tolerance_pct):
                        self.position.close()
                        return
                if self.move_to_be_at_r > 0 and r_now >= self.move_to_be_at_r and trade.sl is not None:
                    try:
                        trade.sl = max(float(trade.sl), float(trade.entry_price))
                    except Exception:
                        pass
                if self.trail_atr_after_r > 0 and r_now >= self.trail_atr_after_r and atr_now > 0:
                    try:
                        trail_sl = price - atr_now * self.sl_atr_mult
                        trade.sl = max(float(trade.sl) if trade.sl is not None else -np.inf, trail_sl)
                    except Exception:
                        pass

            if self.position.is_short:
                r_now = (float(trade.entry_price) - price) / self.active_risk_per_unit
                if self.failure_exit_on_level_reclaim and self.active_breakout_level == self.active_breakout_level:
                    if price > self.active_breakout_level * (1 + self.retest_tolerance_pct):
                        self.position.close()
                        return
                if self.move_to_be_at_r > 0 and r_now >= self.move_to_be_at_r and trade.sl is not None:
                    try:
                        trade.sl = min(float(trade.sl), float(trade.entry_price))
                    except Exception:
                        pass
                if self.trail_atr_after_r > 0 and r_now >= self.trail_atr_after_r and atr_now > 0:
                    try:
                        trail_sl = price + atr_now * self.sl_atr_mult
                        trade.sl = min(float(trade.sl) if trade.sl is not None else np.inf, trail_sl)
                    except Exception:
                        pass

    def next(self):
        self._manage_open_trade()
""").strip()


def compile_strategy_class(schema: StrategySchema, class_name: str = "GeneratedStrategy") -> str:
    family = getattr(schema, "family", "mean_reversion")
    code = _common_header(class_name, schema)

    if family == "breakout":
        methods = f"""
def next(self):
    self._manage_open_trade()
    if not self._base_ok():
        return
    if len(self.data) < self.lookback + 3:
        return

    highs = np.array(self.data.High[-self.lookback-1:-1], dtype=float)
    lows = np.array(self.data.Low[-self.lookback-1:-1], dtype=float)

    breakout_high = float(np.max(highs))
    breakout_low = float(np.min(lows))

    close_now = float(self.data.Close[-1])
    high_now = float(self.data.High[-1])
    low_now = float(self.data.Low[-1])

    long_signal = (
        self._allow_long()
        and self._trend_long_ok()
        and self._regime_long_ok()
        and self._bar_body_large_bull()
        and self._bar_range_ok()
        and self._volume_ok()
        and close_now > breakout_high * (1 + self.break_buffer_pct)
    )

    short_signal = (
        self._allow_short()
        and self._trend_short_ok()
        and self._regime_short_ok()
        and self._bar_body_large_bear()
        and self._bar_range_ok()
        and self._volume_ok()
        and close_now < breakout_low * (1 - self.break_buffer_pct)
    )

    if self.retest_required:
        if long_signal:
            self.breakout_state = 1
            self.breakout_direction = "long"
            self.breakout_level = breakout_high
            self.breakout_bar_index = self._bar_index()
            self.breakout_trigger_high = high_now
            self.breakout_trigger_low = low_now

        if short_signal:
            self.breakout_state = 1
            self.breakout_direction = "short"
            self.breakout_level = breakout_low
            self.breakout_bar_index = self._bar_index()
            self.breakout_trigger_high = high_now
            self.breakout_trigger_low = low_now

        if self.breakout_state == 1 and (self._bar_index() - self.breakout_bar_index) > self.max_retest_bars:
            self.breakout_state = 0
            self.breakout_level = np.nan
            self.breakout_direction = ""

        if self.breakout_state == 1 and self.breakout_direction == "long":
            touched = low_now <= self.breakout_level * (1 + self.retest_tolerance_pct)
            confirm = close_now > self.breakout_level if self.close_confirmation else touched
            if self.rejection_confirmation:
                confirm = confirm and (close_now > float(self.data.Open[-1]))
            if confirm and self._ict_long_ok() and not self.position:
                sl = min(low_now, self.breakout_trigger_low) - float(self.atr[-1]) * self.sl_atr_mult * 0.25
                self._enter_long(sl=sl, entry=close_now)
                self.breakout_state = 0
                self.breakout_direction = ""

        if self.breakout_state == 1 and self.breakout_direction == "short":
            touched = high_now >= self.breakout_level * (1 - self.retest_tolerance_pct)
            confirm = close_now < self.breakout_level if self.close_confirmation else touched
            if self.rejection_confirmation:
                confirm = confirm and (close_now < float(self.data.Open[-1]))
            if confirm and self._ict_short_ok() and not self.position:
                sl = max(high_now, self.breakout_trigger_high) + float(self.atr[-1]) * self.sl_atr_mult * 0.25
                self._enter_short(sl=sl, entry=close_now)
                self.breakout_state = 0
                self.breakout_direction = ""
        return

    if long_signal and self._ict_long_ok() and not self.position:
        sl = low_now - float(self.atr[-1]) * self.sl_atr_mult
        self._enter_long(sl=sl, entry=close_now)
        return

    if short_signal and self._ict_short_ok() and not self.position:
        sl = high_now + float(self.atr[-1]) * self.sl_atr_mult
        self._enter_short(sl=sl, entry=close_now)
        return
"""
        return _append_class_methods(code, methods)

    methods = """
def next(self):
    self._manage_open_trade()
    if not self._base_ok():
        return
    if len(self.data) < max(self.lookback + 3, 30):
        return

    close_now = float(self.data.Close[-1])
    low_now = float(self.data.Low[-1])
    high_now = float(self.data.High[-1])

    if self.direction in ("long_only", "both"):
        long_signal = (
            self._trend_long_ok()
            and self._regime_long_ok()
            and close_now < float(self.bb_low[-1]) * (1 + self.price_tolerance_pct)
        )
        if long_signal and self._ict_long_ok() and not self.position:
            sl = low_now - float(self.atr[-1]) * self.sl_atr_mult
            self._enter_long(sl=sl, entry=close_now)
            return

    if self.direction in ("short_only", "both"):
        short_signal = (
            self._trend_short_ok()
            and self._regime_short_ok()
            and close_now > float(self.bb_high[-1]) * (1 - self.price_tolerance_pct)
        )
        if short_signal and self._ict_short_ok() and not self.position:
            sl = high_now + float(self.atr[-1]) * self.sl_atr_mult
            self._enter_short(sl=sl, entry=close_now)
            return
"""
    return _append_class_methods(code, methods)