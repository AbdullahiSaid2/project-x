
from pathlib import Path
import argparse
import pandas as pd

def summarize(df, groups):
    if df.empty: return pd.DataFrame()
    col='net_pnl_dollars' if 'net_pnl_dollars' in df.columns else 'gross_pnl_dollars'
    return df.groupby(groups,dropna=False).agg(trades=(col,'size'),gross_pnl_dollars=('gross_pnl_dollars','sum') if 'gross_pnl_dollars' in df.columns else (col,'sum'),net_pnl_dollars=(col,'sum'),avg_trade=(col,'mean'),median_trade=(col,'median'),win_rate_pct=(col,lambda s:(s>0).mean()*100),profit_factor=(col,lambda s:s[s>0].sum()/abs(s[s<0].sum()) if abs(s[s<0].sum())>0 else float('inf')),worst_trade=(col,'min'),best_trade=(col,'max')).reset_index().sort_values('net_pnl_dollars',ascending=False)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--trade-log',required=True); p.add_argument('--planned-risk',type=float,default=250); p.add_argument('--risk-breach-multiple',type=float,default=1.5); args=p.parse_args()
    df=pd.read_csv(args.trade_log); col='net_pnl_dollars' if 'net_pnl_dollars' in df.columns else 'gross_pnl_dollars'
    print('\n================ DIAGNOSTICS REPORT ================'); print(f'File: {args.trade_log}'); print(f'Trades: {len(df)}');
    if 'gross_pnl_dollars' in df.columns: print(f"Gross PnL: ${df['gross_pnl_dollars'].sum():,.2f}")
    print(f"Net PnL:   ${df[col].sum():,.2f}")
    if 'round_turn_commission_dollars' in df.columns: print(f"Commissions: ${df['round_turn_commission_dollars'].sum():,.2f}")
    print('\nBy symbol:'); print(summarize(df,['symbol']).to_string(index=False))
    if 'trade_type' in df.columns:
        print('\nBy symbol/trade_type:'); print(summarize(df,['symbol','trade_type']).to_string(index=False))
    if 'same_bar_exit' in df.columns: same=df[df['same_bar_exit'].astype(str).str.lower().isin(['true','1'])]
    elif 'entry_time_et' in df.columns and 'exit_time_et' in df.columns: same=df[pd.to_datetime(df['entry_time_et'],errors='coerce')==pd.to_datetime(df['exit_time_et'],errors='coerce')]
    else: same=pd.DataFrame()
    print(f'\nSame-bar exits: {len(same)}')
    if not same.empty: print(f"Same-bar net PnL: ${same[col].sum():,.2f}")
    if 'actual_loss_vs_planned_risk' in df.columns: breaches=df[(df['gross_pnl_dollars']<0)&(df['actual_loss_vs_planned_risk']>args.risk_breach_multiple)]
    else: breaches=df[df[col]<-(args.planned_risk*args.risk_breach_multiple)]
    print(f'\nRisk breaches > {args.risk_breach_multiple}R: {len(breaches)}')
    if not breaches.empty:
        print(f"Risk breach net PnL: ${breaches[col].sum():,.2f}"); cols=[c for c in ['symbol','side','size','entry_time_et','exit_time_et','entry_price','exit_price','stop_price','target_price','planned_risk_dollars','gross_pnl_dollars','net_pnl_dollars','actual_loss_vs_planned_risk','trade_type','exit_reason'] if c in breaches.columns]; print(breaches.sort_values(col).head(20)[cols].to_string(index=False))
    if 'exit_reason' in df.columns:
        print('\nExit reasons:'); print(df['exit_reason'].value_counts().to_string())
        forced=df[df['exit_reason'].astype(str).str.contains('force_flat',case=False,na=False)]; print(f'\nForced-flat trades: {len(forced)}');
        if not forced.empty: print(f"Forced-flat net PnL: ${forced[col].sum():,.2f}")
    if 'entry_session_label' in df.columns and 'exit_session_label' in df.columns:
        cross=df[df['entry_session_label'].astype(str)!=df['exit_session_label'].astype(str)]; print(f'\nCross-session accepted trades: {len(cross)}')
if __name__=='__main__': main()
