import pandas as pd
from utils import load_config

def run_sanity_check():
    config = load_config()
    file_path = config['universe']['processed_path']
    
    print(f"🔍 [Sanity Check] Auditing processed data: {file_path}")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # --- Check 1: Look-ahead Bias (The Golden Rule) ---
    # Principle: 'fwd_ret' must NOT be perfectly correlated with today's return.
    # If correlation is near 1.0, you are likely predicting "today" instead of "tomorrow".
    df['today_ret'] = df.groupby('ticker')['close'].pct_change()
    leakage_corr = df['today_ret'].corr(df['fwd_ret'])
    
    print(f"\n1️⃣ Look-ahead Bias Detection:")
    if leakage_corr > 0.95:
        print(f"   ❌ CRITICAL WARNING: High Look-ahead Bias detected! Corr={leakage_corr:.4f}")
    else:
        print(f"   ✅ PASS: Correlation between Today's Ret and Label is {leakage_corr:.4f} (Low is expected)")

    # --- Check 2: Missing Values ---
    null_counts = df.isnull().sum()
    print(f"\n2️⃣ Missing Value Check:")
    if null_counts.any():
        print(f"   ❌ ERROR: Found NaN values in the following columns!\n{null_counts[null_counts > 0]}")
    else:
        print(f"   ✅ PASS: No missing values found across all fields.")

    # --- Check 3: Data Coverage ---
    print(f"\n3️⃣ Universe & Timeline Coverage:")
    n_tickers = df['ticker'].nunique()
    n_dates = df['date'].nunique()
    print(f"   💡 Total Tickers: {n_tickers}")
    print(f"   💡 Total Timesteps: {n_dates}")
    
    # --- Check 4: Feature Distribution (Winsorization Check) ---
    print(f"\n4️⃣ Feature Distribution Statistics:")
    # We aggregate min/max/mean to ensure features are z-scored or normalized correctly
    stats = df[config['features']['list'] + ['fwd_ret']].agg(['min', 'max', 'mean'])
    print(stats.to_string())

    # --- Check 5: Logical Alignment Spot-Check ---
    # Randomly sample one ticker to verify lag alignment visually.
    sample_ticker = df['ticker'].iloc[0]
    sample_df = df[df['ticker'] == sample_ticker].sort_values('date').head(3)
    
    print(f"\n5️⃣ Alignment Logic Spot-Check (Ticker: {sample_ticker}):")
    print(sample_df[['date', 'close', 'fwd_ret']])
    print("   👉 VISUAL VERIFY: The 'fwd_ret' in Row 1 should equal ((Row 2 'close' / Row 1 'close') - 1)")

if __name__ == "__main__":
    run_sanity_check()