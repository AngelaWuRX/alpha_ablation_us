import pandas as pd
import numpy as np
from utils import load_config

def run_linear_baseline(data_path, output_path):
    config = load_config()
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    factor_cols = config['features']['list']
    split_date = config['models']['train_split_date']
    
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    ic_values = {}
    for col in factor_cols:
        daily_ic = train_df.groupby('date').apply(
            lambda x: x[col].rank().corr(x['fwd_ret'].rank())
        )
        ic_values[col] = daily_ic.mean()
    
    weights = np.array([ic_values[col] for col in factor_cols])
    test_df['score'] = test_df[factor_cols].values @ weights
    
    test_df[['date', 'ticker', 'close', 'score', 'fwd_ret']].to_csv(output_path, index=False)
    print(f"📊 Linear Baseline ICs: {ic_values}")