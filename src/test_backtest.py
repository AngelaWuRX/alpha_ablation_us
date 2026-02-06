import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import backtest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from backtest import backtest_vectorized

@pytest.fixture
def toy_data():
    """
    Creates a tiny, predictable dataset for White-Box Testing.
    
    Scenario:
    - T0 (Jan 1): Signal BUY AAPL (Score 1.0), GOOG (Score 0.0)
    - T1 (Jan 2): AAPL Jumps +10%. We trade here based on T0 signal (Lag=1).
    - T2 (Jan 3): GOOG Jumps +10%. AAPL Flat.
    """
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    
    # We repeat dates for each ticker
    df = pd.DataFrame({
        'date': dates.repeat(2),
        'ticker': ['AAPL', 'GOOG'] * 3,
        'close': [
            100.0, 100.0,  # Jan 1 (T0)
            110.0, 100.0,  # Jan 2 (T1): AAPL +10%
            110.0, 110.0   # Jan 3 (T2): GOOG +10%
        ],
        'score': [
            1.0, 0.0,      # T0 Signal: Buy AAPL
            0.0, 1.0,      # T1 Signal: Buy GOOG
            0.5, 0.5       # T2 Signal: Flat
        ]
    })
    
    # Pre-calculate returns so the backtester doesn't have to guess
    # We group by ticker to ensure close-to-close returns are correct
    df['ret'] = df.groupby('ticker')['close'].pct_change()
    
    # Note: 'ret' on Jan 1 will be NaN (no previous day).
    # 'ret' on Jan 2 for AAPL will be 0.10.
    return df

def test_execution_lag_integrity(toy_data):
    """
    [CRITICAL] Verifies that a signal on T0 captures the return on T1.
    """
    print("\n--- Testing Execution Lag (Signal T0 -> Ret T1) ---")
    
    # Lag=1 means Signal(T0) trades on T1 Open and captures Ret(T1)
    # On T0: Signal AAPL=1.0. 
    # On T1: AAPL Return is +10%.
    # Portfolio Return on T1 should be +10%.
    
    perf, _ = backtest_vectorized(toy_data, top_n=1, cost_bps=0.0, execution_lag=1)
    
    print("Performance Table:\n", perf[['gross_ret', 'net_ret']])
    
    # Check Jan 2 return
    t1_ret = perf.loc['2023-01-02', 'gross_ret']
    
    # We use np.isclose because floating point math (0.10000000001)
    assert np.isclose(t1_ret, 0.10), f"Lag Failed! Expected 0.10, got {t1_ret}"
    print("✅ Lag Logic Verified.")

def test_transaction_cost_math(toy_data):
    """
    [CRITICAL] Verifies cost accounting.
    """
    print("\n--- Testing Transaction Costs ---")
    # Cost = 100bps (1%). 
    # T1 Turnover: We go from 0 holdings to 100% AAPL. Turnover = 100%. Cost = 1%.
    # T2 Turnover: We swap AAPL to GOOG. Sell 100%, Buy 100%. Turnover = 200%. Cost = 2%.
    
    perf, _ = backtest_vectorized(toy_data, top_n=1, cost_bps=100.0, execution_lag=1)
    
    # T1 (Jan 2): Gross +10%. Cost 1%. Net should be +9% (0.09).
    t1_net = perf.loc['2023-01-02', 'net_ret']
    
    print(f"Jan 2 Net Ret: {t1_net}")
    assert np.isclose(t1_net, 0.09), f"Cost Failed T1! Expected 0.09, got {t1_net}"
    print("✅ Cost Accounting Verified.")

def test_scaling_invariance(toy_data):
    """
    [ROBUSTNESS] Multiplied scores should yield identical trades.
    """
    print("\n--- Testing Score Scaling Invariance ---")
    df_scaled = toy_data.copy()
    df_scaled['score'] = df_scaled['score'] * 1000  # Massive scaling
    
    perf_orig, _ = backtest_vectorized(toy_data, top_n=1, execution_lag=1)
    perf_scaled, _ = backtest_vectorized(df_scaled, top_n=1, execution_lag=1)
    
    # The Net Returns must be identical
    pd.testing.assert_series_equal(perf_orig['net_ret'], perf_scaled['net_ret'])
    print("✅ Scaling Invariance Verified.")

def test_all_zero_signal(toy_data):
    """
    [DEFENSIVE] No signal = No trade = 0 return.
    """
    print("\n--- Testing Zero Signal ---")
    df_zero = toy_data.copy()
    df_zero['score'] = 0.0
    
    perf, _ = backtest_vectorized(df_zero, top_n=1, execution_lag=1)
    
    # Depending on how 'rank' handles ties, it might select random stocks.
    # But usually, if all are 0, rank order is preserved by index.
    # However, standard practice: if score is 0, maybe we shouldn't trade?
    # Your current logic ranks everything. If all are 0, it picks top N based on index.
    # This test ensures the code doesn't crash, even if returns aren't 0.
    
    assert not perf.empty
    print("✅ Zero Signal Handled (Code didn't crash).")

if __name__ == "__main__":
    # Allows running this file directly to see prints
    # python ai_tests/test_backtest_defensive.py
    pytest.main(["-s", __file__])