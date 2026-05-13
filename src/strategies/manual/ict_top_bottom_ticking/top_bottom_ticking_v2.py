import pandas as pd
from datetime import time


# =========================================================
# CONFIG
# =========================================================

TARGET_R = 10.0

STOP_BUFFER = 0.10

CE_TOLERANCE = 0.10

MIN_DISPLACEMENT_ATR = 1.5
MIN_BODY_RATIO = 0.60

MIN_RB_DISPLACEMENT = 1.2
MIN_RB_BODY_RATIO = 0.55

MIN_VOLATILITY_RATIO = 0.75
EXPANSION_THRESHOLD = 1.0

MAX_TRADES_PER_DAY = 3
MAX_CONSECUTIVE_LOSSES = 2
MAX_DAILY_R_LOSS = -1.5
DAILY_PROFIT_LOCK = 3.0

PARTIAL_R = 2.0
PARTIAL_CLOSE = 0.50

LONDON_KILLZONE = (
    time(8, 0),
    time(11, 30),
)

NY_AM = (
    time(13, 30),
    time(16, 0),
)


# =========================================================
# PAYOUT RISK MANAGER
# =========================================================

class PayoutRiskManager:

    def __init__(self):
        self.daily_r = 0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.locked = False

    def reset_day(self):
        self.daily_r = 0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.locked = False

    def can_trade(self):
        return not self.locked

    def register_trade(self, realized_r):

        self.daily_trades += 1
        self.daily_r += realized_r

        if realized_r < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.daily_trades >= MAX_TRADES_PER_DAY:
            self.locked = True

        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.locked = True

        if self.daily_r <= MAX_DAILY_R_LOSS:
            self.locked = True

        if self.daily_r >= DAILY_PROFIT_LOCK:
            self.locked = True


# =========================================================
# SESSION FILTERS
# =========================================================

def allowed_session(timestamp_et):

    t = timestamp_et.time()

    return (
        LONDON_KILLZONE[0] <= t <= LONDON_KILLZONE[1]
        or
        NY_AM[0] <= t <= NY_AM[1]
    )


# =========================================================
# VOLATILITY FILTERS
# =========================================================

def volatility_ok(current_atr, rolling_atr):

    return current_atr >= (
        rolling_atr * MIN_VOLATILITY_RATIO
    )


def expansion_regime(current_atr, rolling_atr):

    return current_atr > (
        rolling_atr * EXPANSION_THRESHOLD
    )


# =========================================================
# LIQUIDITY SWEEPS
# =========================================================

def bullish_sweep(current_low, previous_low):

    return current_low < previous_low


def bearish_sweep(current_high, previous_high):

    return current_high > previous_high


# =========================================================
# MSS LOGIC
# =========================================================

def displacement_strength(candle, atr):

    return (
        (candle.high - candle.low)
        / max(atr, 0.01)
    )


def candle_body_ratio(candle):

    rng = candle.high - candle.low

    if rng <= 0:
        return 0

    return abs(
        candle.close - candle.open
    ) / rng


def valid_displacement(candle, atr):

    return (
        displacement_strength(candle, atr)
        >= MIN_DISPLACEMENT_ATR
        and
        candle_body_ratio(candle)
        >= MIN_BODY_RATIO
    )


def bullish_mss(candle, prior_high, atr):

    return (
        valid_displacement(candle, atr)
        and
        candle.close > prior_high
    )


def bearish_mss(candle, prior_low, atr):

    return (
        valid_displacement(candle, atr)
        and
        candle.close < prior_low
    )


# =========================================================
# REJECTION BLOCKS
# =========================================================

def calculate_ce(high, low):

    return (high + low) / 2


def valid_rejection_block(candle, atr):

    body_ratio = candle_body_ratio(candle)

    displacement = (
        candle.high - candle.low
    ) / max(atr, 0.01)

    return (
        displacement >= MIN_RB_DISPLACEMENT
        and
        body_ratio >= MIN_RB_BODY_RATIO
    )


def build_rejection_block(candle, direction, atr):

    if not valid_rejection_block(candle, atr):
        return None

    return {
        "direction": direction,
        "high": candle.high,
        "low": candle.low,
        "open": candle.open,
        "close": candle.close,
        "ce": calculate_ce(
            candle.high,
            candle.low,
        ),
        "timestamp": candle.name,
    }


# =========================================================
# FAIR VALUE GAPS
# =========================================================

def bullish_fvg(c1, c2, c3):

    if c1.high < c3.low:

        return {
            "direction": "bullish",
            "upper": c3.low,
            "lower": c1.high,
            "midpoint": (
                c3.low + c1.high
            ) / 2,
        }

    return None


def bearish_fvg(c1, c2, c3):

    if c1.low > c3.high:

        return {
            "direction": "bearish",
            "upper": c1.low,
            "lower": c3.high,
            "midpoint": (
                c1.low + c3.high
            ) / 2,
        }

    return None


# =========================================================
# PARTIALS
# =========================================================

def should_take_partial(current_r):

    return current_r >= PARTIAL_R


def partial_size(position_size):

    return max(
        1,
        int(position_size * PARTIAL_CLOSE)
    )


# =========================================================
# ENTRY HELPERS
# =========================================================

def near_ce(price, ce_level, rb_range):

    tolerance = rb_range * CE_TOLERANCE

    return abs(price - ce_level) <= tolerance


# =========================================================
# MAIN STRATEGY ENGINE
# =========================================================

def generate_top_bottom_ticking_v2_signals(df):

    trades = []

    risk_manager = PayoutRiskManager()

    if "atr" not in df.columns:

        df["atr"] = (
            (df["high"] - df["low"])
            .rolling(14)
            .mean()
        )

    df["rolling_atr"] = (
        df["atr"]
        .rolling(50)
        .mean()
    )

    current_day = None

    for i in range(50, len(df) - 3):

        candle = df.iloc[i]

        timestamp = candle.name

        # =====================================
        # RESET DAY
        # =====================================

        if current_day != timestamp.date():

            current_day = timestamp.date()

            risk_manager.reset_day()

        # =====================================
        # PAYOUT RISK LOCK
        # =====================================

        if not risk_manager.can_trade():
            continue

        # =====================================
        # SESSION FILTER
        # =====================================

        if not allowed_session(timestamp):
            continue

        atr = candle.atr
        rolling_atr = candle.rolling_atr

        if pd.isna(atr) or pd.isna(rolling_atr):
            continue

        # =====================================
        # VOLATILITY FILTERS
        # =====================================

        if not volatility_ok(
            atr,
            rolling_atr,
        ):
            continue

        if not expansion_regime(
            atr,
            rolling_atr,
        ):
            continue

        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        # =====================================
        # BUILD RB
        # =====================================

        bullish_rb = build_rejection_block(
            prev,
            "bullish",
            atr,
        )

        bearish_rb = build_rejection_block(
            prev,
            "bearish",
            atr,
        )

        # =====================================
        # SWEEP + MSS
        # =====================================

        bullish_setup = (
            bullish_sweep(
                candle.low,
                prev.low,
            )
            and
            bullish_mss(
                candle,
                prev2.high,
                atr,
            )
        )

        bearish_setup = (
            bearish_sweep(
                candle.high,
                prev.high,
            )
            and
            bearish_mss(
                candle,
                prev2.low,
                atr,
            )
        )

        # =====================================
        # LONG SETUP
        # =====================================

        if bullish_setup:

            fvg = bullish_fvg(
                prev2,
                prev,
                candle,
            )

            if fvg and bullish_rb:

                rb_range = (
                    bullish_rb["high"]
                    - bullish_rb["low"]
                )

                if near_ce(
                    candle.close,
                    bullish_rb["ce"],
                    rb_range,
                ):

                    entry = candle.close

                    stop = (
                        bullish_rb["low"]
                        - STOP_BUFFER
                    )

                    risk = entry - stop

                    if risk > 0:

                        target = (
                            entry
                            + (risk * TARGET_R)
                        )

                        trades.append(
                            {
                                "timestamp": timestamp,
                                "side": "LONG",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "risk": risk,
                                "setup": "bullish_rb_mss",
                            }
                        )

                        risk_manager.register_trade(
                            1.0
                        )

        # =====================================
        # SHORT SETUP
        # =====================================

        if bearish_setup:

            fvg = bearish_fvg(
                prev2,
                prev,
                candle,
            )

            if fvg and bearish_rb:

                rb_range = (
                    bearish_rb["high"]
                    - bearish_rb["low"]
                )

                if near_ce(
                    candle.close,
                    bearish_rb["ce"],
                    rb_range,
                ):

                    entry = candle.close

                    stop = (
                        bearish_rb["high"]
                        + STOP_BUFFER
                    )

                    risk = stop - entry

                    if risk > 0:

                        target = (
                            entry
                            - (risk * TARGET_R)
                        )

                        trades.append(
                            {
                                "timestamp": timestamp,
                                "side": "SHORT",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "risk": risk,
                                "setup": "bearish_rb_mss",
                            }
                        )

                        risk_manager.register_trade(
                            1.0
                        )

    return pd.DataFrame(trades)