import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
from utils import load_config

def calculate_ic_metrics(signal_path, model_name, config):
    if not os.path.exists(signal_path):
        print(f"⚠️  Warning: File not found {signal_path}")
        return None
    
    # 1. Use config to get the correct column names
    target_col = config['features']['label'] # e.g., 'fwd_ret'
    oos_start = config['models']['train_split_date']
    
    df = pd.read_csv(signal_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Filter for Out-of-Sample period only
    # We only care about how the model predicts on "unseen" data
    df = df[df['date'] >= oos_start].copy()
    
    if df.empty:
        print(f"⚠️  Warning: No OOS data for {model_name} after {oos_start}")
        return None

    # Calculate Daily Rank IC (Spearman Correlation)
    def daily_rank_ic(group):
        if len(group) < 2: 
            return np.nan
        # Spearman is the industry standard for Rank IC
        ic, _ = spearmanr(group['score'], group[target_col])
        return ic

    daily_ic = df.groupby('date').apply(daily_rank_ic).dropna()
    
    metrics = {
        "Model": model_name,
        "Mean_IC": daily_ic.mean(),
        "IC_Std": daily_ic.std(),
        "ICIR": daily_ic.mean() / daily_ic.std() if daily_ic.std() != 0 else 0,
        "IC_Hit_Rate": (daily_ic > 0).mean()
    }
    return metrics

def run_evaluation():
    config = load_config() # Load config here
    results_dir = 'results'
    
    # Find signal files
    signal_files = [f for f in os.listdir(results_dir) if f.startswith("signals_") and f.endswith(".csv")]
    
    if not signal_files:
        print("⚠️  No signal files found in results directory.")
        return
    print(f"🔍 Found {len(signal_files)} signal files for evaluation.")

    summary = []
    print("\n" + "="*60)
    print(f"🧪 [METRICS] Evaluating OOS Rank IC (Post-{config['models']['train_split_date']})")
    print("="*60)
    
    for filename in signal_files:
        model_name = filename.replace("signals_", "").replace(".csv", "").upper()
        path = os.path.join(results_dir, filename)
        
        print(f"➡️  Evaluating Model: {model_name}")
        # Pass config into the calculator
        m = calculate_ic_metrics(path, model_name, config)
        if m:
            summary.append(m)
            print(f"✅ Evaluated: {model_name}")
            print(f"   Mean IC: {m['Mean_IC']:.4f}, IC Std: {m['IC_Std']:.4f}, ICIR: {m['ICIR']:.4f}, IC Hit Rate: {m['IC_Hit_Rate']:.4f}")
        else:
            print(f"❌ Skipped: {model_name} due to insufficient data.")
            
    summary_df = pd.DataFrame(summary).sort_values("Mean_IC", ascending=False)
    print("\n" + summary_df.to_string(index=False))
    print("="*60)
    return summary_df

if __name__ == "__main__":
    run_evaluation()