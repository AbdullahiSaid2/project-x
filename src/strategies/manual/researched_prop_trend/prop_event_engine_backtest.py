
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse, math, sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
try:
    import yaml
except Exception as exc:
    raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) >= 5 else ROOT
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.data.fetcher import get_ohlcv
OUT_TRADE_LOG=ROOT/'event_trade_log.csv'; OUT_DAILY=ROOT/'event_daily_summary.csv'; OUT_MONTHLY=ROOT/'event_monthly_summary.csv'; OUT_SYMBOL=ROOT/'event_symbol_summary.csv'; OUT_TRADE_TYPE=ROOT/'event_trade_type_summary.csv'; OUT_RISK_BREACHES=ROOT/'event_risk_breaches.csv'; OUT_SAME_BAR=ROOT/'event_same_bar_trades.csv'; OUT_FORCED_FLAT=ROOT/'event_forced_flat_trades.csv'; OUT_REJECTED=ROOT/'event_rejected_signals.csv'; OUT_ACCOUNT_CURVE=ROOT/'event_account_curve.csv'
@dataclass(frozen=True)
class SymbolSpec:
    symbol:str; exchange:str; asset_class:str; dollars_per_point:float; tick_size:float; tick_value:float; min_stop_points:float; max_stop_points:float; atr_stop_mult:float; target_r:float; max_contracts:int; breakout_lookback:int; pullback_lookback:int
@dataclass(frozen=True)
class PropProfile:
    name:str; account_size:float; profit_target:Optional[float]; max_drawdown:float; drawdown_type:str; daily_loss_limit:Optional[float]; risk_per_trade:float; internal_daily_stop:float; internal_daily_profit_lock:float; max_open_risk:float; max_trades_per_day:int; max_losses_per_day:int; max_equity_index_positions:int; flatten_time_et:str; reopen_time_et:str; mode:str='eval'; news_blackout_enabled:bool=False; news_blackout_minutes_before:int=5; news_blackout_minutes_after:int=5; flatten_before_news:bool=False
@dataclass
class Position:
    symbol:str; side:str; size:int; entry_time:pd.Timestamp; entry_time_et:pd.Timestamp; entry_price:float; stop_price:float; target_price:float; planned_risk_points:float; planned_risk_dollars:float; planned_target_dollars:float; trade_type:str; entry_bar:int; session_label:str
@dataclass
class AccountState:
    start_balance:float; balance:float; realised_pnl:float=0.0; daily_net_pnl:float=0.0; current_date:Any=None; daily_locked:bool=False; peak_balance:float=0.0; max_drawdown:float=0.0
    def __post_init__(self): self.peak_balance=self.start_balance

def load_yaml(path:Path)->Dict[str,Any]:
    with path.open('r',encoding='utf-8') as f: return yaml.safe_load(f) or {}
def load_symbol_specs(path:Path=ROOT/'symbol_specs.yaml')->Dict[str,SymbolSpec]:
    raw=load_yaml(path); out={}
    for symbol,cfg in raw.items():
        out[symbol.upper()]=SymbolSpec(symbol.upper(),str(cfg.get('exchange','tradovate')),str(cfg['asset_class']),float(cfg['dollars_per_point']),float(cfg['tick_size']),float(cfg['tick_value']),float(cfg['min_stop_points']),float(cfg['max_stop_points']),float(cfg['atr_stop_mult']),float(cfg['target_r']),int(cfg['max_contracts']),int(cfg['breakout_lookback']),int(cfg['pullback_lookback']))
    return out
def load_prop_profiles(path:Path=ROOT/'prop_profiles.yaml')->Dict[str,PropProfile]:
    raw=load_yaml(path); out={}
    for name,cfg in raw.items():
        out[name]=PropProfile(name,float(cfg['account_size']),float(cfg['profit_target']) if cfg.get('profit_target') is not None else None,float(cfg['max_drawdown']),str(cfg['drawdown_type']),float(cfg['daily_loss_limit']) if cfg.get('daily_loss_limit') is not None else None,float(cfg['risk_per_trade']),float(cfg['internal_daily_stop']),float(cfg.get('internal_daily_profit_lock',999999)),float(cfg.get('max_open_risk',999999)),int(cfg.get('max_trades_per_day',999999)),int(cfg.get('max_losses_per_day',999999)),int(cfg.get('max_equity_index_positions',999999)),str(cfg.get('flatten_time_et','16:50')),str(cfg.get('reopen_time_et','18:00')),str(cfg.get('mode','eval')),bool(cfg.get('news_blackout_enabled',False)),int(cfg.get('news_blackout_minutes_before',5)),int(cfg.get('news_blackout_minutes_after',5)),bool(cfg.get('flatten_before_news',False)))
    return out
def to_et(ts:Any)->pd.Timestamp:
    t=pd.Timestamp(ts)
    if t.tzinfo is None: t=t.tz_localize('UTC')
    return t.tz_convert('America/New_York')
def et_minutes(h:int,m:int)->int: return int(h)*60+int(m)
def is_allowed_futures_trading_time(et_ts:Any)->bool:
    t=to_et(et_ts); dow=int(t.dayofweek); now=et_minutes(t.hour,t.minute); open18=1080; close1659=1019
    if dow==5: return False
    if dow==6: return now>=open18
    if dow in (0,1,2,3): return now<close1659 or now>=open18
    if dow==4: return now<close1659
    return False
def should_force_flat(et_ts:Any)->bool:
    t=to_et(et_ts); dow=int(t.dayofweek); now=et_minutes(t.hour,t.minute)
    if dow in (0,1,2,3,4) and now>=1010: return True
    if dow==5: return True
    if dow==6 and now<1080: return True
    return False
def futures_session_label(et_ts:Any)->Optional[str]:
    if pd.isna(et_ts): return None
    t=to_et(et_ts)
    if not is_allowed_futures_trading_time(t): return None
    return str((t+pd.Timedelta(days=1)).date()) if et_minutes(t.hour,t.minute)>=1080 else str(t.date())

def load_news_events(path:Path)->pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=['event_time_et','event_name','currency','impact','source','event_dt_et'])
    df=pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=['event_time_et','event_name','currency','impact','source','event_dt_et'])
    if 'event_time_et' not in df.columns:
        raise ValueError('news_events.csv must contain event_time_et column')
    df=df.copy()
    df['event_dt_et']=pd.to_datetime(df['event_time_et'],errors='coerce')
    df=df.dropna(subset=['event_dt_et']).sort_values('event_dt_et').reset_index(drop=True)
    if 'event_name' not in df.columns: df['event_name']='news_event'
    if 'currency' not in df.columns: df['currency']='USD'
    if 'impact' not in df.columns: df['impact']='high'
    return df

def news_blackout_status(et_ts:Any,news_events:pd.DataFrame,before_minutes:int,after_minutes:int)->Tuple[bool,str]:
    if news_events is None or news_events.empty:
        return False,''
    t=to_et(et_ts).tz_localize(None)
    # Event is in blackout if now is between event-before and event+after.
    start=t-pd.Timedelta(minutes=after_minutes)
    end=t+pd.Timedelta(minutes=before_minutes)
    hit=news_events[(news_events['event_dt_et']>=start)&(news_events['event_dt_et']<=end)]
    if hit.empty:
        return False,''
    return True,str(hit.iloc[0].get('event_name','news_event'))
def normalize_ohlcv(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy(); rename={}
    for col in df.columns:
        l=str(col).lower()
        if l=='open': rename[col]='Open'
        elif l=='high': rename[col]='High'
        elif l=='low': rename[col]='Low'
        elif l=='close': rename[col]='Close'
        elif l=='volume': rename[col]='Volume'
    df=df.rename(columns=rename)
    for c in ['Open','High','Low','Close']:
        if c not in df.columns: raise ValueError(f'Missing required column: {c}')
    if 'Volume' not in df.columns: df['Volume']=0
    return df.sort_index()
def atr(df:pd.DataFrame,n:int=14)->pd.Series:
    pc=df['Close'].shift(1); tr=pd.concat([df['High']-df['Low'],(df['High']-pc).abs(),(df['Low']-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def resample_features(df:pd.DataFrame, rule:str, prefix:str)->pd.DataFrame:
    agg=df[['Open','High','Low','Close','Volume']].resample(rule).agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    out=pd.DataFrame(index=agg.index); out[f'{prefix}_close']=agg['Close']; out[f'{prefix}_ema20']=agg['Close'].ewm(span=20,adjust=False,min_periods=20).mean(); out[f'{prefix}_ema50']=agg['Close'].ewm(span=50,adjust=False,min_periods=50).mean()
    out[f'{prefix}_bull']=(out[f'{prefix}_close']>out[f'{prefix}_ema20'])&(out[f'{prefix}_ema20']>out[f'{prefix}_ema50']); out[f'{prefix}_bear']=(out[f'{prefix}_close']<out[f'{prefix}_ema20'])&(out[f'{prefix}_ema20']<out[f'{prefix}_ema50'])
    return out.reindex(df.index, method='ffill')
def build_features(df:pd.DataFrame,spec:SymbolSpec)->pd.DataFrame:
    df=normalize_ohlcv(df); out=df.copy(); out['atr14']=atr(out,14)
    out=out.join(resample_features(out,'1h','h1')).join(resample_features(out,'4h','h4')).join(resample_features(out,'1D','d1'))
    dc=out['Close'].resample('1D').last().dropna(); d=pd.DataFrame(index=dc.index); d['d_ma5']=dc.rolling(5,min_periods=5).mean(); d['d_ma25']=dc.rolling(25,min_periods=25).mean(); d['d5_gt_d25']=d['d_ma5']>d['d_ma25']; d['d5_lt_d25']=d['d_ma5']<d['d_ma25']; out=out.join(d.reindex(out.index,method='ffill'))
    out['recent_high']=out['High'].shift(1).rolling(spec.pullback_lookback,min_periods=spec.pullback_lookback).max(); out['recent_low']=out['Low'].shift(1).rolling(spec.pullback_lookback,min_periods=spec.pullback_lookback).min(); out['channel_high']=out['High'].shift(1).rolling(spec.breakout_lookback,min_periods=spec.breakout_lookback).max(); out['channel_low']=out['Low'].shift(1).rolling(spec.breakout_lookback,min_periods=spec.breakout_lookback).min()
    out['bull_reclaim']=(out['Low']<out['recent_low'])&(out['Close']>out['recent_low']); out['bear_reclaim']=(out['High']>out['recent_high'])&(out['Close']<out['recent_high']); out['breakout_long']=out['Close']>out['channel_high']; out['breakout_short']=out['Close']<out['channel_low']
    out['trend_score']=0
    for p in ['h1','h4','d1']:
        out.loc[out[f'{p}_bull'].fillna(False),'trend_score']+=1; out.loc[out[f'{p}_bear'].fillna(False),'trend_score']-=1
    out.loc[out['d5_gt_d25'].fillna(False),'trend_score']+=1; out.loc[out['d5_lt_d25'].fillna(False),'trend_score']-=1
    et=pd.Series(out.index,index=out.index).apply(to_et); out['et']=et; out['et_date']=et.apply(lambda x:x.date()); out['session_label']=et.apply(futures_session_label); out['trading_hours_ok']=et.apply(is_allowed_futures_trading_time); out['force_flat_now']=et.apply(should_force_flat)
    idx=pd.Series(out.index,index=out.index); out['next_timestamp']=idx.shift(-1); out['minutes_to_next_bar']=(out['next_timestamp']-idx).dt.total_seconds()/60; out['large_forward_gap']=out['minutes_to_next_bar'].fillna(0)>60
    return out
def round_to_tick(price:float,tick:float)->float: return round(price/tick)*tick if tick>0 else price
def planned_order(symbol,spec,prop,row,side,trade_type,target_r,min_target):
    close=float(row['Close']); atr_now=float(row.get('atr14',0) or 0)
    if not math.isfinite(atr_now) or atr_now<=0: return None,'invalid_atr'
    if side=='LONG':
        structure=min(float(row['Low']),float(row['recent_low'])) if trade_type=='trend_pullback_long' else max(float(row['channel_low']), close-atr_now*spec.atr_stop_mult)
        stop=round_to_tick(min(structure, close-atr_now*spec.atr_stop_mult), spec.tick_size); stop_points=close-stop; target=round_to_tick(close+stop_points*target_r,spec.tick_size)
    else:
        structure=max(float(row['High']),float(row['recent_high'])) if trade_type=='trend_pullback_short' else min(float(row['channel_high']), close+atr_now*spec.atr_stop_mult)
        stop=round_to_tick(max(structure, close+atr_now*spec.atr_stop_mult), spec.tick_size); stop_points=stop-close; target=round_to_tick(close-stop_points*target_r,spec.tick_size)
    if stop_points<spec.min_stop_points or stop_points>spec.max_stop_points: return None,'stop_bounds'
    rpc=stop_points*spec.dollars_per_point; size=int(math.floor(prop.risk_per_trade/rpc)) if rpc>0 else 0; size=max(0,min(size,spec.max_contracts))
    if size<1: return None,'size_zero'
    planned_risk=stop_points*spec.dollars_per_point*size; planned_target=stop_points*target_r*spec.dollars_per_point*size
    if planned_target<min_target: return None,'min_planned_target'
    return {'symbol':symbol,'side':side,'size':size,'entry_price':round_to_tick(close,spec.tick_size),'stop_price':stop,'target_price':target,'planned_risk_points':stop_points,'planned_risk_dollars':planned_risk,'planned_target_dollars':planned_target,'trade_type':trade_type},''
def signal_for_row(symbol,spec,prop,row,min_score,target_r,min_target,allowed):
    if not bool(row.get('trading_hours_ok',False)): return None,'outside_trading_hours'
    if bool(row.get('force_flat_now',False)) or bool(row.get('large_forward_gap',False)): return None,'entry_before_close_or_gap'
    score=int(row.get('trend_score',0) or 0); candidates=[]
    if score>=min_score:
        if bool(row.get('bull_reclaim',False)): candidates.append(('LONG','trend_pullback_long'))
        if bool(row.get('breakout_long',False)): candidates.append(('LONG','turtle_breakout_long'))
    elif score<=-min_score:
        if bool(row.get('bear_reclaim',False)): candidates.append(('SHORT','trend_pullback_short'))
        if bool(row.get('breakout_short',False)): candidates.append(('SHORT','turtle_breakout_short'))
    for side,tt in candidates:
        if tt not in allowed: continue
        return planned_order(symbol,spec,prop,row,side,tt,target_r,min_target)
    return None,'no_signal'
def close_position(pos,spec,exit_time,row,exit_price,reason,comm_side,same_bar,collision):
    pts=(exit_price-pos.entry_price) if pos.side=='LONG' else (pos.entry_price-exit_price); gross=pts*spec.dollars_per_point*pos.size; comm=pos.size*comm_side*2; net=gross-comm; loss_r=abs(gross)/pos.planned_risk_dollars if gross<0 and pos.planned_risk_dollars>0 else 0.0
    return {'symbol':pos.symbol,'side':pos.side,'size':pos.size,'entry_time':pos.entry_time,'exit_time':exit_time,'entry_time_et':pos.entry_time_et,'exit_time_et':to_et(exit_time),'entry_price':pos.entry_price,'exit_price':exit_price,'stop_price':pos.stop_price,'target_price':pos.target_price,'planned_risk_points':pos.planned_risk_points,'planned_risk_dollars':pos.planned_risk_dollars,'planned_target_dollars':pos.planned_target_dollars,'realized_points':pts,'dollars_per_point':spec.dollars_per_point,'gross_pnl_dollars':gross,'round_turn_commission_dollars':comm,'net_pnl_dollars':net,'actual_loss_vs_planned_risk':loss_r,'trade_type':pos.trade_type,'exit_reason':reason,'same_bar_exit':same_bar,'intrabar_collision':collision,'entry_session_label':pos.session_label,'exit_session_label':futures_session_label(to_et(exit_time))}
def evaluate_exit(pos,spec,ts,row):
    high=float(row['High']); low=float(row['Low']); close=float(row['Close'])
    if bool(row.get('force_flat_now',False)) or bool(row.get('large_forward_gap',False)): return close,'force_flat_or_gap',False
    if pos.side=='LONG':
        sl=low<=pos.stop_price; tp=high>=pos.target_price
        if sl and tp: return pos.stop_price,'sl_intrabar_collision',True
        if sl: return pos.stop_price,'stop_loss',False
        if tp: return pos.target_price,'take_profit',False
    else:
        sl=high>=pos.stop_price; tp=low<=pos.target_price
        if sl and tp: return pos.stop_price,'sl_intrabar_collision',True
        if sl: return pos.stop_price,'stop_loss',False
        if tp: return pos.target_price,'take_profit',False
    return None
def summarize(df,groups):
    if df.empty: return pd.DataFrame()
    return df.groupby(groups,dropna=False).agg(trades=('net_pnl_dollars','size'),gross_pnl_dollars=('gross_pnl_dollars','sum'),commissions_dollars=('round_turn_commission_dollars','sum'),net_pnl_dollars=('net_pnl_dollars','sum'),avg_net_trade=('net_pnl_dollars','mean'),median_net_trade=('net_pnl_dollars','median'),win_rate_pct=('net_pnl_dollars',lambda s:(s>0).mean()*100),profit_factor=('net_pnl_dollars',lambda s:s[s>0].sum()/abs(s[s<0].sum()) if abs(s[s<0].sum())>0 else float('inf')),worst_trade=('net_pnl_dollars','min'),best_trade=('net_pnl_dollars','max')).reset_index().sort_values('net_pnl_dollars',ascending=False)
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--symbols',nargs='+',default=['MNQ','MES','MYM','MGC']); p.add_argument('--prop-profile',default='apex_50k_eod_eval'); p.add_argument('--days-back',type=int,default=365); p.add_argument('--timeframe',default='1m'); p.add_argument('--tail-rows',type=int,default=180000); p.add_argument('--no-tail',action='store_true'); p.add_argument('--commission-per-contract-side',type=float,default=2.0); p.add_argument('--min-trend-score',type=int,default=3); p.add_argument('--target-r',type=float,default=10.0); p.add_argument('--min-planned-target-dollars',type=float,default=500.0); p.add_argument('--allowed-trade-types',default='trend_pullback_long,trend_pullback_short,turtle_breakout_long,turtle_breakout_short'); p.add_argument('--max-actual-loss-r-multiple',type=float,default=1.5); p.add_argument('--risk-per-trade',type=float,default=None); p.add_argument('--news-events',default=str(ROOT/'news_events.csv')); p.add_argument('--enable-news-blackout',action='store_true'); p.add_argument('--disable-news-blackout',action='store_true'); p.add_argument('--news-minutes-before',type=int,default=None); p.add_argument('--news-minutes-after',type=int,default=None); p.add_argument('--flatten-before-news',action='store_true'); return p.parse_args()
def main():
    args=parse_args(); specs=load_symbol_specs(); prop=load_prop_profiles()[args.prop_profile];
    if args.risk_per_trade is not None: prop=PropProfile(**{**prop.__dict__,'risk_per_trade':float(args.risk_per_trade)})
    if args.enable_news_blackout: prop=PropProfile(**{**prop.__dict__,'news_blackout_enabled':True})
    if args.disable_news_blackout: prop=PropProfile(**{**prop.__dict__,'news_blackout_enabled':False})
    if args.news_minutes_before is not None: prop=PropProfile(**{**prop.__dict__,'news_blackout_minutes_before':int(args.news_minutes_before)})
    if args.news_minutes_after is not None: prop=PropProfile(**{**prop.__dict__,'news_blackout_minutes_after':int(args.news_minutes_after)})
    if args.flatten_before_news: prop=PropProfile(**{**prop.__dict__,'flatten_before_news':True})
    news_events=load_news_events(Path(args.news_events)) if prop.news_blackout_enabled else pd.DataFrame()
    symbols=[s.upper() for s in args.symbols]; allowed={x.strip() for x in args.allowed_trade_types.split(',') if x.strip()}; data={}
    print('Loading and building features...')
    for sym in symbols:
        spec=specs[sym]; print(f'\n=== loading {sym} ==='); df=get_ohlcv(sym,exchange=spec.exchange,timeframe=args.timeframe,days_back=args.days_back); df=normalize_ohlcv(df); df=df if args.no_tail else df.tail(args.tail_rows); feat=build_features(df,spec); data[sym]=feat; print(f'{sym}: rows={len(feat)} start={feat.index.min()} end={feat.index.max()}')
    master=sorted(set().union(*[set(df.index) for df in data.values()])); acct=AccountState(prop.account_size,prop.account_size); positions={}; trades=[]; rejected=[]; curve=[]
    print(f'\nProfile mode: {prop.mode} | risk_per_trade=${prop.risk_per_trade:,.2f} | news_blackout={prop.news_blackout_enabled}')
    if prop.news_blackout_enabled: print(f'News file: {args.news_events} | rows={len(news_events)} | window={prop.news_blackout_minutes_before}m before/{prop.news_blackout_minutes_after}m after | flatten={prop.flatten_before_news}')
    print(f'\nRunning event loop over {len(master)} timestamps...')
    for step,ts in enumerate(master):
        et=to_et(ts); d=et.date()
        if acct.current_date is None or d!=acct.current_date:
            acct.current_date=d; acct.daily_net_pnl=0.0; acct.daily_locked=False
        in_news_blackout,news_event_name=news_blackout_status(et,news_events,prop.news_blackout_minutes_before,prop.news_blackout_minutes_after) if prop.news_blackout_enabled else (False,'')
        if in_news_blackout and prop.flatten_before_news:
            for sym in list(positions.keys()):
                if ts not in data[sym].index: continue
                spec=specs[sym]; row=data[sym].loc[ts]; pos=positions[sym]; same=int(pos.entry_bar)==int(data[sym].index.get_loc(ts)); tr=close_position(pos,spec,ts,row,float(row['Close']),'news_blackout_flat_'+news_event_name,args.commission_per_contract_side,same,False); trades.append(tr); acct.realised_pnl+=tr['net_pnl_dollars']; acct.daily_net_pnl+=tr['net_pnl_dollars']; acct.balance+=tr['net_pnl_dollars']; acct.peak_balance=max(acct.peak_balance,acct.balance); acct.max_drawdown=min(acct.max_drawdown,acct.balance-acct.peak_balance); del positions[sym]
        for sym in list(positions.keys()):
            if ts not in data[sym].index: continue
            spec=specs[sym]; row=data[sym].loc[ts]; pos=positions[sym]; ev=evaluate_exit(pos,spec,ts,row)
            if ev:
                price,reason,coll=ev; same=int(pos.entry_bar)==int(data[sym].index.get_loc(ts)); tr=close_position(pos,spec,ts,row,price,reason,args.commission_per_contract_side,same,coll); trades.append(tr); acct.realised_pnl+=tr['net_pnl_dollars']; acct.daily_net_pnl+=tr['net_pnl_dollars']; acct.balance+=tr['net_pnl_dollars']; acct.peak_balance=max(acct.peak_balance,acct.balance); acct.max_drawdown=min(acct.max_drawdown,acct.balance-acct.peak_balance); del positions[sym]
        if acct.daily_net_pnl<=-abs(prop.internal_daily_stop): acct.daily_locked=True
        if not acct.daily_locked:
            for sym in symbols:
                if sym in positions or ts not in data[sym].index: continue
                row=data[sym].loc[ts]; spec=specs[sym]
                if in_news_blackout:
                    rejected.append({'timestamp':ts,'timestamp_et':et,'symbol':sym,'reject_reason':'news_blackout','news_event_name':news_event_name}); continue
                order,reason=signal_for_row(sym,spec,prop,row,args.min_trend_score,args.target_r,args.min_planned_target_dollars,allowed)
                if not order:
                    if reason not in ('no_signal','outside_trading_hours','entry_before_close_or_gap'): rejected.append({'timestamp':ts,'timestamp_et':et,'symbol':sym,'reject_reason':reason})
                    continue
                sess=futures_session_label(et)
                if sess is None:
                    rejected.append({'timestamp':ts,'timestamp_et':et,'symbol':sym,'reject_reason':'entry_outside_trading_hours'}); continue
                positions[sym]=Position(sym,order['side'],int(order['size']),ts,et,float(order['entry_price']),float(order['stop_price']),float(order['target_price']),float(order['planned_risk_points']),float(order['planned_risk_dollars']),float(order['planned_target_dollars']),str(order['trade_type']),int(data[sym].index.get_loc(ts)),sess)
        if step%1000==0 or step==len(master)-1: curve.append({'timestamp':ts,'timestamp_et':et,'balance':acct.balance,'realised_pnl':acct.realised_pnl,'daily_net_pnl':acct.daily_net_pnl,'daily_locked':acct.daily_locked,'open_positions':len(positions),'max_drawdown':acct.max_drawdown})
    for sym,pos in list(positions.items()):
        spec=specs[sym]; row=data[sym].iloc[-1]; ts=data[sym].index[-1]; tr=close_position(pos,spec,ts,row,float(row['Close']),'end_of_test',args.commission_per_contract_side,False,False); trades.append(tr)
    df=pd.DataFrame(trades); rej=pd.DataFrame(rejected); curve=pd.DataFrame(curve)
    if not df.empty:
        df['exit_date_et']=pd.to_datetime(df['exit_time_et'],errors='coerce').dt.tz_localize(None).dt.date; df['exit_month_et']=pd.to_datetime(df['exit_time_et'],errors='coerce').dt.tz_localize(None).dt.to_period('M').astype(str); df['risk_breach']=(df['gross_pnl_dollars']<0)&(df['actual_loss_vs_planned_risk']>args.max_actual_loss_r_multiple)
    df.to_csv(OUT_TRADE_LOG,index=False); rej.to_csv(OUT_REJECTED,index=False); curve.to_csv(OUT_ACCOUNT_CURVE,index=False)
    if not df.empty:
        summarize(df,['symbol']).to_csv(OUT_SYMBOL,index=False); summarize(df,['symbol','trade_type']).to_csv(OUT_TRADE_TYPE,index=False); summarize(df,['exit_date_et','symbol']).to_csv(OUT_DAILY,index=False); summarize(df,['exit_month_et','symbol']).to_csv(OUT_MONTHLY,index=False); df[df['risk_breach']].to_csv(OUT_RISK_BREACHES,index=False); df[df['same_bar_exit']].to_csv(OUT_SAME_BAR,index=False); df[df['exit_reason'].astype(str).str.contains('force_flat',na=False)].to_csv(OUT_FORCED_FLAT,index=False)
    gross=float(df['gross_pnl_dollars'].sum()) if not df.empty else 0; comm=float(df['round_turn_commission_dollars'].sum()) if not df.empty else 0; net=float(df['net_pnl_dollars'].sum()) if not df.empty else 0
    print('\n================ EVENT ENGINE FINAL REPORT ================'); print(f'Trades:             {len(df)}'); print(f'Gross PnL:          ${gross:,.2f}'); print(f'Commissions:        ${comm:,.2f}'); print(f'Net realised PnL:   ${net:,.2f}'); print(f'Final balance:      ${prop.account_size+net:,.2f}'); print(f'Max drawdown:       ${acct.max_drawdown:,.2f}'); print(f'Rejected signals:   {len(rej)}')
    if not df.empty:
        print(f"Risk breaches:      {int(df['risk_breach'].sum())}"); print(f"Same-bar exits:     {int(df['same_bar_exit'].sum())}"); print('\nBy symbol:'); print(summarize(df,['symbol']).to_string(index=False))
    print('\nWrote files:')
    for p in [OUT_TRADE_LOG,OUT_DAILY,OUT_MONTHLY,OUT_SYMBOL,OUT_TRADE_TYPE,OUT_RISK_BREACHES,OUT_SAME_BAR,OUT_FORCED_FLAT,OUT_REJECTED,OUT_ACCOUNT_CURVE]: print(f'  {p}')
if __name__=='__main__': main()
