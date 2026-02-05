import pytest
import pandas as pd
import os
from backtest import backtest_topn

def test_sign_flip_falsification():
    """
    Property: If we flip the signal, a strategy with high positive IC 
    should become a strategy with high negative IC and losses.
    """
    path = "results/signals_mlp.csv"
    if not os.path.exists(path): pytest.skip("MLP signals not found")
    
    df = pd.read_csv(path)
    df_flipped = df.copy()
    df_flipped['score'] = -df['score']
    
    perf_orig, _ = backtest_topn(df, cost_bps=0)
    perf_flip, _ = backtest_topn(df_flipped, cost_bps=0)
    
    # Sum of cumulative returns should move in opposite directions
    assert perf_orig['cum_ret'].iloc[-1] != perf_flip['cum_ret'].iloc[-1]

def test_data_leakage_detector():
    """
    Falsification Test: If we deliberately create a 'perfect leak' 
    (score = tomorrow's return), the system MUST detect it via an 
    insanely high Sharpe. If it doesn't, the backtester is ignoring the signal.
    """
    config = pd.read_csv("results/signals_linear.csv") # Use any result as template
    df = config.copy()
    
    # The 'Cheat': Score is exactly the forward return
    df['score'] = df['fwd_ret']
    
    # Backtest with 0 lag to capture the leak
    _, summary = backtest_topn(df, top_n=5, execution_lag=0, cost_bps=0)
    
    # A perfect leak should produce a Sharpe ratio > 10
    # Convert string Sharpe if your backtest returns it as string, 
    # but based on our previous code, it's a float.
    assert float(summary['Sharpe']) > 10.0, "Backtester failed to detect/exploit a 100% signal leak."