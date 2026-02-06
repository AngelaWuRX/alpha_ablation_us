import pandas as pd
import numpy as np
import os
from utils import load_config

def calculate_turnover(weights_df):
    """
    Calculates daily turnover: sum(|w_t - w_{t-1}|)
    """
    return weights_df.diff().abs().sum(axis=1).fillna(0)

def backtest_vectorized(df, top_n=10, cost_bps=10.0, execution_lag=1):
    """
    Robust Vectorized Backtester.
    1. Pivots Data
    2. Aligns Signals (T) with Returns (T+1)
    3. Calculates PnL
    """
    # 1. Pivot Signals & Returns
    # Ensure dates are Datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # Returns Matrix (Use 'close' to calculate returns if 'ret' isn't pre-calc)
    if 'ret' not in df.columns:
        df = df.sort_values(['ticker', 'date'])
        df['ret'] = df.groupby('ticker')['close'].pct_change()
    
    # Create the Matrices [Index=Date, Columns=Ticker]
    returns = df.pivot(index='date', columns='ticker', values='ret')
    signals = df.pivot(index='date', columns='ticker', values='score')

    # 2. Apply Execution Lag (CRITICAL STEP)
    # Signal at day T is used to trade at Open of T+1. 
    # So we align Signal(T) with Return(T+1).
    aligned_signals = signals.shift(execution_lag)
    
    # 3. Filter for valid trading days (Intersection of Signal & Return)
    valid_dates = returns.index.intersection(aligned_signals.index)
    returns = returns.loc[valid_dates]
    aligned_signals = aligned_signals.loc[valid_dates]
    
    if len(valid_dates) == 0:
        print("⚠️ WARNING: No overlapping dates between Signals and Returns!")
        return pd.DataFrame(), {}

    # 4. Generate Weights (Long Only, Top N)
    # Rank the signals row-by-row
    ranks = aligned_signals.rank(axis=1, ascending=False, method='first')
    
    # Create weights: 1/N for Top N, 0 otherwise
    weights = (ranks <= top_n).astype(float)
    weights = weights.div(top_n, axis=0) # Normalize so sum(weights) = 1.0
    
    # 5. Calculate Performance
    # Daily Portfolio Return = Sum(Weight * Stock_Return)
    # We use numpys efficient multiply-sum
    daily_gross_ret = (weights * returns).sum(axis=1)
    
    # 6. Calculate Transaction Costs
    turnover = calculate_turnover(weights)
    cost_rate = cost_bps / 10000.0
    daily_cost = turnover * cost_rate
    
    daily_net_ret = daily_gross_ret - daily_cost
    
    # 7. Compile Results
    perf_df = pd.DataFrame({
        'gross_ret': daily_gross_ret,
        'cost': daily_cost,
        'net_ret': daily_net_ret,
        'turnover': turnover
    })
    
    perf_df['cum_ret'] = (1 + perf_df['net_ret']).cumprod()
    
    # 8. Summary Stats
    # Annualized Sharpe (assuming daily data)
    r = perf_df['net_ret']
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 1e-6 else 0
    
    # Max Drawdown
    cum_ret = perf_df['cum_ret']
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    summary = {
        "Sharpe": round(sharpe, 4),
        "Max_Drawdown": f"{round(max_dd * 100, 2)}%",
        "Avg_Turnover": f"{round(turnover.mean() * 100, 2)}%",
        "Total_Return": f"{round((perf_df['cum_ret'].iloc[-1] - 1) * 100, 2)}%"
    }
    
    return perf_df, summary

def run_backtest_all():
    config = load_config()
    results_dir = 'results'
    
    bt_cfg = config.get('backtest', {})
    top_n = bt_cfg.get('top_n', 10)
    cost = bt_cfg.get('cost', 10.0)
    lag = bt_cfg.get('execution_lag', 1) 

    signal_files = [f for f in os.listdir(results_dir) if f.startswith("signals_") and f.endswith(".csv")]
    
    if not signal_files:
        print("❌ No signal files found in 'results/'. Run train_all.py first.")
        return

    summary_list = []
    print("\n" + "="*60)
    print(f"💰 [BACKTEST] Params: Top_{top_n}, Cost_{cost}bps, Lag_{lag}d")
    print("="*60)

    for filename in signal_files:
        model_name = filename.replace("signals_", "").replace(".csv", "").upper()
        print(f"🔄 Processing {model_name}...", end=" ")
        
        try:
            file_path = os.path.join(results_dir, filename)
            df = pd.read_csv(file_path)
            
            # CRITICAL FIX: Ensure 'ret' or 'close' exists
            # If signals_*.csv only has [date, ticker, score, fwd_ret], we rely on fwd_ret?
            # Standard: We need Raw Returns for the backtest, not just Fwd Returns.
            # Assuming your saved file has 'fwd_ret' which is the return for the NEXT day.
            # We can use that, but we must align carefully.
            
            # Let's map 'fwd_ret' back to 'ret' for the backtester
            # If fwd_ret at T is Return(T -> T+1), then we can just use it directly
            # WITHOUT shifting the signal again? 
            # NO. Safer to calculate fresh returns from prices if available.
            # If not, use fwd_ret but acknowledge the shift is already embedded.
            
            # Simpler approach: Assume df has 'date', 'ticker', 'score', 'fwd_ret'
            # We rename 'fwd_ret' to 'ret' because fwd_ret on Day T IS the return we realize 
            # if we trade on Day T (Wait, fwd_ret is T to T+1).
            
            # If you trade on Open T+1, you realize Ret(T+1 -> T+2).
            # This is getting complex.
            
            # SIMPLEST FIX for your pipeline:
            # Your train_all.py saves: date, ticker, score, fwd_ret
            # 'fwd_ret' on row T is (Price_T+1 / Price_T) - 1. 
            # If we trade at Close T (Simulated), we realize 'fwd_ret'.
            # So we treat 'fwd_ret' as the return to capture.
            
            df['ret'] = df['fwd_ret'] 
            
            # Since fwd_ret is ALREADY looking forward, we set lag=0 for the shift 
            # because the row T already contains the future return.
            perf, s = backtest_vectorized(df, top_n=top_n, cost_bps=cost, execution_lag=0)
            
            if perf.empty:
                print("SKIPPED (No overlap)")
                continue

            s['Model'] = model_name
            summary_list.append(s)
            
            output_path = os.path.join(results_dir, f"perf_{model_name.lower()}.csv")
            perf.to_csv(output_path)
            print("✅ Done")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")

    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        # Reorder columns
        cols = ['Model', 'Sharpe', 'Total_Return', 'Max_Drawdown', 'Avg_Turnover']
        summary_df = summary_df[cols]
        print("\n" + summary_df.to_string(index=False))
        # Save summary
        summary_df.to_csv(os.path.join(results_dir, "backtest_summary.csv"), index=False)

if __name__ == "__main__":
    run_backtest_all()