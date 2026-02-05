import pytest
import pandas as pd
import numpy as np
from utils import load_config

def test_no_future_labels():
    """Property: A label at time T must not be known at time T."""
    config = load_config()
    df = pd.read_csv(config['universe']['processed_path'])
    df['date'] = pd.to_datetime(df['date'])
    
    # We use np.abs to ensure correlation isn't perfectly 1 or -1
    # Perfect correlation with the 'close' price usually means the label is 
    # the current price, not the future price (leakage).
    for ticker, group in df.groupby('ticker'):
        corr = group['close'].corr(group['fwd_ret'])
        assert np.abs(corr) < 0.99, f"Leak detected in {ticker}: Close/Label correlation is {corr}"

def test_timestamp_monotonicity():
    """Property: Time must only move forward with no duplicates."""
    config = load_config()
    df = pd.read_csv(config['universe']['processed_path'])
    df['date'] = pd.to_datetime(df['date'])
    
    for ticker, group in df.groupby('ticker'):
        # Using np.diff to check that the time delta is always positive
        time_diffs = np.diff(group['date'].values).astype(float)
        assert np.all(time_diffs > 0), f"Non-monotonic time index or duplicate found for {ticker}"

# Using pytest.mark.parametrize to iterate through all factors defined in config
config_data = load_config()
@pytest.mark.parametrize("factor", config_data['features']['list'])
def test_feature_sanity(factor):
    """Property: Every factor must be numerically stable (no Inf, limited NaNs)."""
    df = pd.read_csv(config_data['universe']['processed_path'])
    
    # Use np.isinf and np.isnan for rigorous numerical check
    has_inf = np.isinf(df[factor]).any()
    null_ratio = df[factor].isnull().mean()
    
    assert not has_inf, f"Factor {factor} contains infinite values."
    assert null_ratio < 0.25, f"Factor {factor} has too many missing values ({null_ratio:.2%})."

def test_shuffle_invariance():
    """
    Property: Shuffling the global dataframe rows must NOT change 
    the relationship between a row's features and its label. 
    (Ensures labels aren't calculated based on row index).
    """
    config = load_config()
    df = pd.read_csv(config['universe']['processed_path'])
    
    row_0_orig = df.iloc[0].copy()
    
    # Use np to shuffle indices
    shuffled_df = df.sample(frac=1, random_state=42)
    
    # Find that original first row in the shuffled df
    row_0_new = shuffled_df[(shuffled_df['ticker'] == row_0_orig['ticker']) & 
                            (shuffled_df['date'] == row_0_orig['date'])].iloc[0]
    
    # Check that the label (fwd_ret) stayed with its features
    assert np.isclose(row_0_orig['fwd_ret'], row_0_new['fwd_ret']), "Label-Feature mapping broken by shuffle."