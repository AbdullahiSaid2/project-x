from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from backtesting import Backtest

from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_top_bottom_ticking import (
    ICTTopBottomTickingType1Sniper,
    ICTTopBottomTickingType2,
    ICTTopBottomTickingType2Strict,
    ICTTopBottomTickingType2StrictSession,
)


OUT_DIR = Path('.')
CONTRACT_MULTIPLIER = 2.0  # approx $2 per point per MNQ micro; first-pass report scaling placeholder
BACKTEST_CASH = 50_000


@dataclass
class RunConfig:
    symbol: str
    exchange: str = 'tradovate'
    timeframe: str = '5m'
    days_back: int = 365


VARIANTS = {
    'type2_baseline': ICTTopBottomTickingType2,
    'type2_internal_required': ICTTopBottomTickingType2Strict,
    'type2_internal_session': ICTTopBottomTickingType2StrictSession,
    'type1_sniper': ICTTopBottomTickingType1Sniper,
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if 'datetime' in out.columns:
            out['datetime'] = pd.to_datetime(out['datetime'], utc=False)
            out = out.set_index('datetime')
        elif 'Date' in out.columns:
            out['Date'] = pd.to_datetime(out['Date'], utc=False)
            out = out.set_index('Date')
    rename_map = {}
    for c in out.columns:
        low = c.lower()
        if low == 'open':
            rename_map[c] = 'Open'
        elif low == 'high':
            rename_map[c] = 'High'
        elif low == 'low':
            rename_map[c] = 'Low'
        elif low == 'close':
            rename_map[c] = 'Close'
        elif low == 'volume':
            rename_map[c] = 'Volume'
    out = out.rename(columns=rename_map)
    return out[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().sort_index()


def run_symbol(cfg: RunConfig, variant_name: str):
    print(f"\n=== {cfg.symbol} | {variant_name} ===")
    print(f"Loading {cfg.symbol} {cfg.timeframe} data...")
    df = get_ohlcv(cfg.symbol, exchange=cfg.exchange, timeframe=cfg.timeframe, days_back=cfg.days_back)
    df = _normalize_df(df)
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

    strategy_cls = VARIANTS[variant_name]
    bt = Backtest(df, strategy_cls, cash=BACKTEST_CASH, commission=0.0, exclusive_orders=True, trade_on_close=False)
    stats = bt.run()
    trades = stats['_trades'].copy()
    trades['symbol'] = cfg.symbol
    trades['variant'] = variant_name
    if len(trades):
        trades['entry_time'] = pd.to_datetime(trades['EntryTime'])
        trades['month'] = trades['entry_time'].dt.to_period('M').astype(str)
        trades['day'] = trades['entry_time'].dt.date.astype(str)
        trades['pnl'] = trades['PnL']
    return stats, trades


def summarize_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    monthly = trades.groupby(['variant', 'month'], as_index=False).agg(
        trades=('pnl', 'count'),
        gross_pnl=('pnl', 'sum'),
        avg_trade=('pnl', 'mean'),
        win_rate=('pnl', lambda s: (s > 0).mean()),
    )

    daily = trades.groupby(['variant', 'day'], as_index=False).agg(
        trades=('pnl', 'count'),
        gross_pnl=('pnl', 'sum'),
        avg_trade=('pnl', 'mean'),
    )

    losses = trades[trades['pnl'] < 0].copy()
    if losses.empty:
        loss_clusters = pd.DataFrame(columns=['variant', 'start_day', 'end_day', 'loss_trades', 'gross_loss'])
    else:
        losses = losses.sort_values(['variant', 'entry_time']).reset_index(drop=True)
        clusters = []
        for variant, grp in losses.groupby('variant'):
            grp = grp.sort_values('entry_time').reset_index(drop=True)
            cluster_start = grp.loc[0, 'entry_time']
            cluster_end = grp.loc[0, 'entry_time']
            count = 1
            gross_loss = grp.loc[0, 'pnl']
            for i in range(1, len(grp)):
                gap = grp.loc[i, 'entry_time'] - grp.loc[i-1, 'entry_time']
                if gap <= pd.Timedelta(days=2):
                    cluster_end = grp.loc[i, 'entry_time']
                    count += 1
                    gross_loss += grp.loc[i, 'pnl']
                else:
                    clusters.append({
                        'variant': variant,
                        'start_day': cluster_start.date().isoformat(),
                        'end_day': cluster_end.date().isoformat(),
                        'loss_trades': count,
                        'gross_loss': gross_loss,
                    })
                    cluster_start = grp.loc[i, 'entry_time']
                    cluster_end = grp.loc[i, 'entry_time']
                    count = 1
                    gross_loss = grp.loc[i, 'pnl']
            clusters.append({
                'variant': variant,
                'start_day': cluster_start.date().isoformat(),
                'end_day': cluster_end.date().isoformat(),
                'loss_trades': count,
                'gross_loss': gross_loss,
            })
        loss_clusters = pd.DataFrame(clusters)
    return monthly, daily, loss_clusters


def run_all_symbols(symbols: Iterable[str] | None = None):
    symbols = list(symbols or ['MNQ'])
    all_trades = []
    stats_rows = []
    for raw_symbol in symbols:
        symbol = raw_symbol.replace('MNQ', 'MNQ').replace('NQ', 'MNQ')
        cfg = RunConfig(symbol=symbol)
        for variant_name in VARIANTS:
            stats, trades = run_symbol(cfg, variant_name)
            stats_rows.append({
                'symbol': cfg.symbol,
                'variant': variant_name,
                'trades': int(stats['# Trades']) if '# Trades' in stats else len(trades),
                'return_pct': float(stats['Return [%]']) if 'Return [%]' in stats else None,
                'win_rate_pct': float(stats['Win Rate [%]']) if 'Win Rate [%]' in stats else None,
                'profit_factor': float(stats['Profit Factor']) if 'Profit Factor' in stats else None,
                'max_dd_pct': float(stats['Max. Drawdown [%]']) if 'Max. Drawdown [%]' in stats else None,
            })
            all_trades.append(trades)

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    monthly, daily, clusters = summarize_trades(combined)
    stats_df = pd.DataFrame(stats_rows)

    stats_df.to_csv(OUT_DIR / 'top_bottom_ticking_variant_summary.csv', index=False)
    combined.to_csv(OUT_DIR / 'top_bottom_ticking_all_trades.csv', index=False)
    monthly.to_csv(OUT_DIR / 'top_bottom_ticking_monthly_summary.csv', index=False)
    daily.to_csv(OUT_DIR / 'top_bottom_ticking_daily_summary.csv', index=False)
    clusters.to_csv(OUT_DIR / 'top_bottom_ticking_losing_clusters.csv', index=False)

    print('\nSaved:')
    print('- top_bottom_ticking_variant_summary.csv')
    print('- top_bottom_ticking_all_trades.csv')
    print('- top_bottom_ticking_monthly_summary.csv')
    print('- top_bottom_ticking_daily_summary.csv')
    print('- top_bottom_ticking_losing_clusters.csv')
    return stats_df, combined, clusters


if __name__ == '__main__':
    run_all_symbols(['MNQ'])
