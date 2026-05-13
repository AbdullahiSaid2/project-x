
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetcher import get_ohlcv
from src.backtesting.event_engine.defaults import SYMBOL_SPECS, PROP_PROFILES
from src.backtesting.event_engine.models import Position, PropProfile
from src.backtesting.event_engine.time_rules import to_et, is_allowed_futures_time, should_force_flat, session_date, load_news_events, news_blackout_status
from src.strategies.adapters.ict_payout_cycle_event_adapter import ICTPayoutCycleAdapter

OUT_DIR = Path('src/backtesting/event_engine/outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MAX_CONTRACTS = {'MNQ': 6, 'MES': 6, 'MYM': 8, 'MGC': 4, 'MCL': 4}

def parse_contract_caps(value: str) -> dict[str, int]:
    caps = dict(DEFAULT_MAX_CONTRACTS)
    if value:
        for part in value.split(','):
            if not part.strip(): continue
            sym, raw = part.split(':')
            caps[sym.strip().upper()] = int(raw)
    return caps

def apply_profile_overrides(profile: PropProfile, args) -> PropProfile:
    updates = {}
    for arg, field, cast in [('risk_per_trade','risk_per_trade',float), ('daily_profit_target','daily_profit_target',float), ('daily_soft_loss_stop','daily_soft_loss_stop',float), ('max_trades_per_day','max_trades_per_day',int), ('pause_after_consecutive_losses','pause_after_consecutive_losses',int)]:
        val = getattr(args, arg)
        if val is not None: updates[field] = cast(val)
    return replace(profile, **updates) if updates else profile

def calculate_contracts(profile: PropProfile, spec, symbol: str, entry: float, stop: float, caps: dict[str, int]):
    risk_points = abs(entry - stop)
    if risk_points <= 0: return 0, 0.0, 0.0
    risk_per_contract = risk_points * spec.dollars_per_point
    if risk_per_contract <= 0: return 0, risk_per_contract, 0.0
    contracts = min(int(profile.risk_per_trade // risk_per_contract), int(caps.get(symbol, 1)))
    if contracts < 1: return 0, risk_per_contract, 0.0
    return contracts, risk_per_contract, risk_per_contract * contracts

def close_position(pos: Position, spec, ts, row, exit_price: float, reason: str, commission: float, same_bar=False):
    points = exit_price - pos.entry_price if pos.side == 'LONG' else pos.entry_price - exit_price
    gross = points * spec.dollars_per_point * pos.size
    comm = commission * 2.0 * pos.size
    return {'strategy_name': pos.strategy_name, 'symbol': pos.symbol, 'side': pos.side, 'size': pos.size, 'entry_time_et': to_et(pos.entry_time), 'exit_time_et': to_et(ts), 'entry_price': pos.entry_price, 'exit_price': exit_price, 'realized_points': points, 'dollars_per_point': spec.dollars_per_point, 'gross_pnl_dollars': gross, 'commissions_dollars': comm, 'net_pnl_dollars': gross-comm, 'trade_type': pos.trade_type, 'exit_reason': reason, 'planned_risk_dollars': pos.planned_risk_dollars, 'planned_target_dollars': pos.planned_target_dollars, 'same_bar_exit': same_bar}

def position_unrealized(pos: Position, spec, mark: float) -> float:
    points = mark - pos.entry_price if pos.side == 'LONG' else pos.entry_price - mark
    return points * spec.dollars_per_point * pos.size

def total_open_unrealized(positions, data, ts):
    total = 0.0
    for sym, pos in positions.items():
        if ts in data[sym].index:
            total += position_unrealized(pos, SYMBOL_SPECS[sym], float(data[sym].loc[ts]['Close']))
    return total

def flatten_all(positions, data, ts, reason, commission):
    out=[]
    for sym in list(positions.keys()):
        if ts not in data[sym].index: continue
        row = data[sym].loc[ts]
        out.append(close_position(positions[sym], SYMBOL_SPECS[sym], ts, row, float(row['Close']), reason, commission, False))
        del positions[sym]
    return out

def maybe_exit_position(pos, spec, ts, idx, row, commission):
    if idx <= pos.entry_bar_index: return None
    high, low = float(row['High']), float(row['Low'])
    if pos.side == 'LONG':
        if low <= pos.stop_price: return close_position(pos, spec, ts, row, pos.stop_price, 'stop_loss', commission, False)
        if high >= pos.target_price: return close_position(pos, spec, ts, row, pos.target_price, 'take_profit', commission, False)
    else:
        if high >= pos.stop_price: return close_position(pos, spec, ts, row, pos.stop_price, 'stop_loss', commission, False)
        if low <= pos.target_price: return close_position(pos, spec, ts, row, pos.target_price, 'take_profit', commission, False)
    return None

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--symbols', nargs='+', default=['MNQ','MES','MYM','MGC'])
    p.add_argument('--prop-profile', default='apex_50k_pa', choices=list(PROP_PROFILES.keys()))
    p.add_argument('--days-back', type=int, default=365)
    p.add_argument('--timeframe', default='1m')
    p.add_argument('--tail-rows', type=int, default=180000)
    p.add_argument('--no-tail', action='store_true')
    p.add_argument('--commission-per-contract-side', type=float, default=2.0)
    p.add_argument('--target-r', type=float, default=5.0)
    p.add_argument('--min-planned-target-dollars', type=float, default=250.0)
    p.add_argument('--news-events', default='')
    p.add_argument('--output-prefix', default='ict_payout_cycle')
    p.add_argument('--min-ict-payout-score', type=float, default=3.0)
    p.add_argument('--fvg-expiry-bars', type=int, default=30)
    p.add_argument('--stop-buffer-atr', type=float, default=0.25)
    p.add_argument('--entry-mode', default='ce', choices=['ce','close'])
    p.add_argument('--warmup-bars', type=int, default=500)
    p.add_argument('--relaxed-mode', action='store_true')
    p.add_argument('--risk-per-trade', type=float, default=150)
    p.add_argument('--daily-profit-target', type=float, default=650)
    p.add_argument('--daily-soft-loss-stop', type=float, default=350)
    p.add_argument('--max-trades-per-day', type=int, default=6)
    p.add_argument('--pause-after-consecutive-losses', type=int, default=2)
    p.add_argument('--max-contracts', default='MNQ:6,MES:6,MYM:8,MGC:4')
    p.add_argument('--disable-unrealized-daily-lock', action='store_true')
    return p.parse_args()

def main():
    args=parse_args(); adapter=ICTPayoutCycleAdapter(); profile=apply_profile_overrides(PROP_PROFILES[args.prop_profile], args); caps=parse_contract_caps(args.max_contracts); unrealized_lock=not args.disable_unrealized_daily_lock
    news_events=load_news_events(args.news_events) if profile.news_blackout_enabled else pd.DataFrame()
    symbols=[s.upper() for s in args.symbols]; data={}; maps={}
    print('Loading data and building ICT Payout Cycle v2 features...')
    for sym in symbols:
        print(f'\n=== loading {sym} ===')
        df=get_ohlcv(sym, exchange='tradovate', timeframe=args.timeframe, days_back=args.days_back)
        if not args.no_tail: df=df.tail(args.tail_rows)
        df=adapter.build_features(sym, df); data[sym]=df; maps[sym]={ts:i for i, ts in enumerate(df.index)}
        print(f'{sym}: rows={len(df)} start={df.index.min()} end={df.index.max()}')
    all_ts=sorted(set().union(*[set(df.index) for df in data.values()])); print(f'\nRunning {adapter.name} over {len(all_ts)} timestamps...')
    positions={}; trades=[]; rejected=[]; daily_rows=[]; balance=profile.account_size; peak=balance; eod_peak=balance; floor=profile.account_size-profile.max_drawdown
    current=None; daily_pnl=0.0; daily_trades=0; losses=0; locked=False; lock_reason=''
    def record(trade):
        nonlocal balance,daily_pnl,peak,losses
        trades.append(trade); pnl=float(trade['net_pnl_dollars']); balance+=pnl; daily_pnl+=pnl; peak=max(peak,balance); losses=losses+1 if pnl<0 else 0
    def close_session(next_session):
        nonlocal eod_peak,floor,current,daily_pnl,daily_trades,losses,locked,lock_reason
        if current is not None:
            eod_peak=max(eod_peak,balance); new_floor=eod_peak-profile.max_drawdown
            if profile.drawdown_stop_level is not None: new_floor=min(new_floor, profile.drawdown_stop_level)
            floor=max(floor,new_floor)
            daily_rows.append({'session_date':current,'balance':balance,'daily_net_pnl':daily_pnl,'daily_trades':daily_trades,'eod_peak_balance':eod_peak,'drawdown_floor':floor,'day_locked':locked,'day_lock_reason':lock_reason})
        current=next_session; daily_pnl=0.0; daily_trades=0; losses=0; locked=False; lock_reason=''
    for ts in all_ts:
        et=to_et(ts); sess=session_date(et, reopen_time=profile.reopen_time_et)
        if sess!=current: close_session(sess)
        in_news,_=news_blackout_status(et, news_events, profile.news_minutes_before, profile.news_minutes_after) if profile.news_blackout_enabled else (False,'')
        force=should_force_flat(et, flatten_time=profile.flatten_time_et) or not is_allowed_futures_time(et, flatten_time=profile.flatten_time_et, reopen_time=profile.reopen_time_et)
        if in_news and profile.flatten_before_news: force=True
        if force and positions:
            for tr in flatten_all(positions,data,ts,'force_flat_news' if in_news else 'force_flat_session',args.commission_per_contract_side): record(tr)
        for sym in list(positions.keys()):
            if ts not in data[sym].index: continue
            row=data[sym].loc[ts]; tr=maybe_exit_position(positions[sym],SYMBOL_SPECS[sym],ts,maps[sym][ts],row,args.commission_per_contract_side)
            if tr: record(tr); del positions[sym]
        equity_pnl=daily_pnl+(total_open_unrealized(positions,data,ts) if positions else 0.0)
        if unrealized_lock and positions:
            if profile.daily_profit_target is not None and equity_pnl>=profile.daily_profit_target:
                for tr in flatten_all(positions,data,ts,'daily_equity_profit_lock',args.commission_per_contract_side): record(tr)
                locked=True; lock_reason='daily_equity_profit_lock'
            elif profile.daily_soft_loss_stop is not None and equity_pnl<=-abs(profile.daily_soft_loss_stop):
                for tr in flatten_all(positions,data,ts,'daily_equity_soft_loss_lock',args.commission_per_contract_side): record(tr)
                locked=True; lock_reason='daily_equity_soft_loss_lock'
        if profile.daily_loss_limit is not None and daily_pnl<=-abs(profile.daily_loss_limit): locked=True; lock_reason='daily_loss_limit'
        if profile.daily_soft_loss_stop is not None and daily_pnl<=-abs(profile.daily_soft_loss_stop): locked=True; lock_reason='daily_soft_loss_stop'
        if profile.daily_profit_target is not None and daily_pnl>=profile.daily_profit_target: locked=True; lock_reason='daily_profit_target'
        if profile.max_trades_per_day is not None and daily_trades>=profile.max_trades_per_day: locked=True; lock_reason='max_trades_per_day'
        if profile.pause_after_consecutive_losses is not None and losses>=profile.pause_after_consecutive_losses: locked=True; lock_reason='consecutive_losses'
        if balance<=floor: locked=True; lock_reason='max_drawdown_breach'
        if force or locked or in_news: continue
        for sym in symbols:
            if sym in positions or ts not in data[sym].index: continue
            df=data[sym]; idx=maps[sym][ts]; order=adapter.signal_for_row(sym,df.iloc[idx],df.iloc[:idx+1],SYMBOL_SPECS[sym],profile,args)
            if order is None: continue
            contracts,rpc,planned=calculate_contracts(profile,SYMBOL_SPECS[sym],sym,order.entry_price,order.stop_price,caps)
            if contracts<1:
                rejected.append({'timestamp_et':et,'symbol':sym,'reject_reason':'risk_too_large_or_stop_invalid','trade_type':order.trade_type,'risk_per_contract':rpc,'entry_price':order.entry_price,'stop_price':order.stop_price,'setup_score':order.setup_score}); continue
            planned_target=abs(order.target_price-order.entry_price)*SYMBOL_SPECS[sym].dollars_per_point*contracts
            if planned_target<args.min_planned_target_dollars:
                rejected.append({'timestamp_et':et,'symbol':sym,'reject_reason':'planned_target_too_small','trade_type':order.trade_type,'planned_target_dollars':planned_target,'setup_score':order.setup_score}); continue
            positions[sym]=Position(sym,order.side,contracts,order.entry_price,order.stop_price,order.target_price,ts,idx,order.trade_type,order.strategy_name,planned,planned_target); daily_trades+=1
    for sym in list(positions.keys()):
        df=data[sym]; ts=df.index[-1]; row=df.iloc[-1]; record(close_position(positions[sym],SYMBOL_SPECS[sym],ts,row,float(row['Close']),'end_of_test',args.commission_per_contract_side,False)); del positions[sym]
    close_session(None)
    trades_df=pd.DataFrame(trades); rej_df=pd.DataFrame(rejected); daily_df=pd.DataFrame(daily_rows)
    prefix=args.output_prefix or 'ict_payout_cycle'; out_trade=OUT_DIR/f'{prefix}_event_trade_log.csv'; out_rej=OUT_DIR/f'{prefix}_event_rejected_signals.csv'; out_daily=OUT_DIR/f'{prefix}_event_daily_summary.csv'
    trades_df.to_csv(out_trade,index=False); rej_df.to_csv(out_rej,index=False); daily_df.to_csv(out_daily,index=False)
    gross=trades_df['gross_pnl_dollars'].sum() if not trades_df.empty else 0.0; net=trades_df['net_pnl_dollars'].sum() if not trades_df.empty else 0.0; comm=trades_df['commissions_dollars'].sum() if not trades_df.empty else 0.0
    print('\n================ ICT PAYOUT CYCLE V2 FINAL REPORT ================')
    print(f'Strategy:                 {adapter.name}'); print(f'Profile:                  {profile.name}'); print(f'Risk per trade:            ${profile.risk_per_trade:,.2f}'); print(f'Max contracts:             {caps}'); print(f'Daily profit target:       {profile.daily_profit_target}'); print(f'Daily soft loss stop:      {profile.daily_soft_loss_stop}'); print(f'Unrealized daily lock:     {unrealized_lock}'); print(f'Target R:                  {args.target_r}'); print(f'Min setup score:           {args.min_ict_payout_score}'); print(f'Relaxed mode:              {args.relaxed_mode}'); print(f'Trades:                   {len(trades_df)}'); print(f'Gross PnL:                ${gross:,.2f}'); print(f'Commissions:              ${comm:,.2f}'); print(f'Net PnL:                  ${net:,.2f}'); print(f'Final balance:            ${profile.account_size+net:,.2f}'); print(f'Rejected signals:         {len(rej_df)}')
    if not trades_df.empty:
        by=trades_df.groupby('symbol').agg(trades=('net_pnl_dollars','size'),net_pnl_dollars=('net_pnl_dollars','sum'),avg_trade=('net_pnl_dollars','mean'),median_trade=('net_pnl_dollars','median'),win_rate_pct=('net_pnl_dollars',lambda s:(s>0).mean()*100),worst_trade=('net_pnl_dollars','min'),best_trade=('net_pnl_dollars','max')).reset_index().sort_values('net_pnl_dollars',ascending=False); print('\nBy symbol:'); print(by.to_string(index=False))
    if not daily_df.empty:
        print('\nDaily lock counts:'); print(daily_df['day_lock_reason'].fillna('').replace('', 'none').value_counts().to_string())
    print('\nWrote files:'); print(f'  {out_trade}'); print(f'  {out_rej}'); print(f'  {out_daily}')
if __name__=='__main__': main()
