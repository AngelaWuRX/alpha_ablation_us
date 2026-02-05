import pytest
import pandas as pd
import numpy as np
from backtest import backtest_topn

@pytest.fixture
def toy_data():
    """
    Creates a tiny, predictable dataset.
    Day 1: Signal generated (Score 1.0 for AAPL, 0.0 for GOOG)
    Day 2: Trade executed (Execution Lag = 1). AAPL rises 10%, GOOG flat.
    """
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = {
        'date': dates.repeat(2),
        'ticker': ['AAPL', 'GOOG'] * 3,
        'close': [100, 200,   # T1
                  110, 200,   # T2: AAPL +10%
                  110, 220],  # T3: GOOG +10%
        'score': [1.0, 0.0,   # T1 Signal: Buy AAPL
                  0.0, 1.0,   # T2 Signal: Buy GOOG
                  0.5, 0.5]   # T3 Signal: Neutral
    }
    return pd.DataFrame(data)

def test_execution_lag_integrity(toy_data):
    """
    Property: If execution_lag=1, the return on T2 must be driven by the signal on T1.
    In toy_data: Signal T1 buys AAPL. AAPL returns +10% on T2. 
    Total Net Return for T2 should be 0.10 (minus costs).
    """
    # 0 cost to make math easy
    perf, _ = backtest_topn(toy_data, top_n=1, cost_bps=0, execution_lag=1)
    
    # Check return for 2023-01-02
    t2_ret = perf.loc['2023-01-02', 'net_ret']
    assert np.isclose(t2_ret, 0.10), f"Execution lag failed. Expected 0.10, got {t2_ret}"

def test_transaction_cost_math(toy_data):
    """
    Property: Turnover of 2.0 (selling 100% of A, buying 100% of B) 
    with 100bps cost should subtract exactly 0.02 from returns.
    """
    # 100 bps = 0.01. Turnover from AAPL to GOOG = 2.0. Cost = 0.02.
    perf, _ = backtest_topn(toy_data, top_n=1, cost_bps=100, execution_lag=1)
    
    # On T2, we traded. Gross was 0.10. Cost should be 0.02. Net = 0.08.
    t2_net = perf.loc['2023-01-02', 'net_ret']
    assert np.isclose(t2_net, 0.08), f"Cost calculation error. Expected 0.08, got {t2_net}"

def test_scaling_invariance(toy_data):
    """
    Property: Multiplying scores by 10 should not change the Top-N selection.
    """
    df_high_score = toy_data.copy()
    df_high_score['score'] = df_high_score['score'] * 10
    
    perf_orig, sum_orig = backtest_topn(toy_data, top_n=1)
    perf_scale, sum_scale = backtest_topn(df_high_score, top_n=1)
    
    pd.testing.assert_frame_equal(perf_orig, perf_scale)
    assert sum_orig['Sharpe'] == sum_scale['Sharpe']

def test_all_zero_signal(toy_data):
    """Property: If no signals are provided, return must be 0 and turnover must be 0."""
    df_zero = toy_data.copy()
    df_zero['score'] = 0.0
    
    perf, summary = backtest_topn(df_zero, top_n=1)
    
    assert (perf['net_ret'] == 0).all()
    # Average turnover string "0.0%"
    assert "0.0%" in summary['Avg_Turnover']