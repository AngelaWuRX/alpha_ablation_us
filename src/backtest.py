import pandas as pd
import math
import os
from utils import load_config

def backtest_topn(df, top_n=10, cost_bps=10.0, execution_lag=1):
    """
    Simulates a portfolio buying Top N stocks based on model scores.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Calculate Daily Returns
    df = df.sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker")["close"].pct_change()

    # 2. Prevent Look-ahead Bias (The Execution Lag)
    # This aligns the signal with the day the trade actually happens
    df["date_for_trade"] = df.groupby("ticker")["date"].shift(-execution_lag)
    
    # 3. Pivot into Matrices for vectorized math
    returns_matrix = df.pivot(index="date", columns="ticker", values="ret").fillna(0)
    scores_matrix = df.pivot(index="date_for_trade", columns="ticker", values="score")
    
    # Only trade on days where we have both a signal and a return
    dates = scores_matrix.index.intersection(returns_matrix.index).sort_values()
    
    last_w = pd.Series(0.0, index=returns_matrix.columns)
    performance = []
    cost_rate = cost_bps / 10000.0

    # 4. Step-through Backtest
    for d in dates:
        day_scores = scores_matrix.loc[d].dropna()
        
        if len(day_scores) > 0:
            top_tickers = day_scores.nlargest(top_n).index
            new_w = pd.Series(0.0, index=returns_matrix.columns)
            new_w.loc[top_tickers] = 1.0 / top_n
        else:
            new_w = last_w 
            
        # Turnover calculation (Difference between yesterday's holdings and today's)
        turnover = (new_w - last_w).abs().sum()
        cost = turnover * cost_rate
        
        # Performance: Yesterday's weights * Today's realized returns
        gross_ret = (last_w * returns_matrix.loc[d]).sum()
        net_ret = gross_ret - cost
        
        performance.append({
            "date": d,
            "net_ret": net_ret,
            "turnover": turnover
        })
        last_w = new_w

    perf_df = pd.DataFrame(performance).set_index("date")
    perf_df["cum_ret"] = (1 + perf_df["net_ret"]).cumprod()
    
    # 5. Professional Metrics
    r = perf_df["net_ret"]
    sharpe = (r.mean() / r.std() * math.sqrt(252)) if r.std() != 0 else 0
    cum_max = perf_df["cum_ret"].cummax()
    mdd = ((perf_df["cum_ret"] - cum_max) / cum_max).min()

    summary = {
        "Sharpe": round(sharpe, 4),
        "Max_Drawdown": f"{round(mdd * 100, 2)}%",
        "Avg_Turnover": f"{round(perf_df['turnover'].mean() * 100, 2)}%"
    }
    return perf_df, summary

def run_backtest_all():
    # Now config is actually used to drive the backtest settings
    config = load_config()
    results_dir = 'results'
    
    bt_cfg = config.get('backtest', {})
    top_n = bt_cfg.get('top_n', 10)
    cost = bt_cfg.get('cost', 10.0)
    # You can add 'execution_lag' to your config.yaml as well
    lag = bt_cfg.get('execution_lag', 1) 

    signal_files = [f for f in os.listdir(results_dir) if f.startswith("signals_") and f.endswith(".csv")]
    
    summary_list = []
    print("\n" + "="*60)
    print(f"💰 [BACKTEST] Params: Top_{top_n}, Cost_{cost}bps, Lag_{lag}d")
    print("="*60)

    for filename in signal_files:
        model_name = filename.replace("signals_", "").replace(".csv", "").upper()
        df = pd.read_csv(os.path.join(results_dir, filename))
        
        perf, s = backtest_topn(df, top_n=top_n, cost_bps=cost, execution_lag=lag)
        s['Model'] = model_name
        summary_list.append(s)
        perf.to_csv(os.path.join(results_dir, f"perf_{model_name.lower()}.csv"))
        print(f"✅ Completed: {model_name}")

    summary_df = pd.DataFrame(summary_list)
    print("\n" + summary_df.to_string(index=False))
    return summary_df

if __name__ == "__main__":
    run_backtest_all()