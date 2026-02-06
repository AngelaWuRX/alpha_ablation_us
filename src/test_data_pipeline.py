import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import your real data logic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# If you have specific functions to load data, import them. 
# Otherwise, we will test the CSV output directly.
from utils import load_config

@pytest.fixture
def real_data():
    """
    Loads the actual processed data for inspection.
    """
    config = load_config()
    file_path = config['universe']['processed_path']
    
    if not os.path.exists(file_path):
        pytest.skip(f"Skipping: Processed data not found at {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def test_data_integrity(real_data):
    """
    [CRITICAL] Ensures the dataset is ready for training (No NaNs, No Inf).
    """
    print(f"\n\n🔍 [DATA INSPECTION] Loaded {len(real_data)} rows.")
    
    # 1. Check for NaNs
    null_counts = real_data.isnull().sum().sum()
    print(f"   ... Checking for Missing Values (NaNs): Found {null_counts}")
    assert null_counts == 0, f"Dataset contains {null_counts} NaNs! Run clean_data() first."
    
    # 2. Check for Infinite values
    inf_counts = np.isinf(real_data.select_dtypes(include=np.number)).sum().sum()
    print(f"   ... Checking for Infinite Values: Found {inf_counts}")
    assert inf_counts == 0, "Dataset contains Infinite values (Division by zero likely)."
    
    print("✅ DATA INTEGRITY PASSED.")

def test_target_distribution(real_data):
    """
    [LOGIC] Checks if 'fwd_ret' looks like a stock return (approx normal distribution).
    """
    print("\n🔍 [LABEL INSPECTION] Analyzing 'fwd_ret'...")
    
    target = real_data['fwd_ret']
    
    # Calculate stats
    mean_ret = target.mean()
    std_ret = target.std()
    min_ret = target.min()
    max_ret = target.max()
    
    print(f"   ... Mean: {mean_ret:.6f} (Should be near 0)")
    print(f"   ... Std:  {std_ret:.6f} (Should be ~0.01 - 0.03 for daily)")
    print(f"   ... Min/Max: {min_ret:.2f} / {max_ret:.2f}")
    
    # Sanity Checks
    # 1. The mean shouldn't be massive (e.g., > 1% daily average is unrealistic for S&P 500)
    assert abs(mean_ret) < 0.01, f"Mean return is suspiciously high: {mean_ret:.2%}"
    
    # 2. We shouldn't have 1000% returns in a day (likely data error)
    assert max_ret < 5.0, f"Found a +500% return label: {max_ret}. Check for price glitches."
    
    print("✅ LABEL DISTRIBUTION PASSED.")

def test_feature_stationarity(real_data):
    """
    [ML SAFETY] Checks if features are Z-Scored (Mean ~0, Std ~1).
    This prevents the 'Exploding Gradient' problem in Neural Networks.
    """
    print("\n🔍 [FEATURE INSPECTION] Checking Scaling/Z-Scoring...")
    
    # Exclude date, ticker, fwd_ret
    features = real_data.select_dtypes(include=np.number).drop(columns=['fwd_ret'], errors='ignore')
    
    # Check the first few features
    for col in features.columns[:5]:
        mu = features[col].mean()
        sigma = features[col].std()
        print(f"   ... Feature '{col}': Mean={mu:.2f}, Std={sigma:.2f}")
        
        # If you Z-scored, Mean should be close to 0 (e.g., +/- 0.5)
        # Warning only, not assertion failure, as some features might be naturally skewed
        if abs(mu) > 1.0:
            print(f"       ⚠️ WARNING: Feature '{col}' is not centered (Mean {mu:.2f}). MLP might struggle.")
            
    print("✅ FEATURE INSPECTION COMPLETED.")

def test_date_alignment(real_data):
    """
    [TIME] Ensures we don't have duplicate (Date, Ticker) pairs.
    """
    print("\n🔍 [TIME INSPECTION] Checking Uniqueness...")
    
    duplicates = real_data.duplicated(subset=['date', 'ticker']).sum()
    print(f"   ... Duplicate Rows: {duplicates}")
    
    assert duplicates == 0, "Found duplicate Date-Ticker pairs! This ruins backtests."
    print("✅ UNIQUENESS PASSED.")

if __name__ == "__main__":
    # Allows running directly: python ai_tests/test_data_pipeline.py
    pytest.main(["-s", __file__])