import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import load_config

def plot_equity_curves():
    config = load_config()
    results_dir = 'results'
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create two subplots: Top for NAV, Bottom for Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    perf_files = [f for f in os.listdir(results_dir) if f.startswith("perf_") and f.endswith(".csv")]
    
    for filename in perf_files:
        model_name = filename.replace("perf_", "").replace(".csv", "").upper()
        df = pd.read_csv(os.path.join(results_dir, filename))
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 1. Plot Equity Curve
        ax1.plot(df['cum_ret'], label=f"{model_name}")
        
        # 2. Plot Drawdown
        # Calculate drawdown: (Current / Running Max) - 1
        cum_max = df['cum_ret'].cummax()
        dd = (df['cum_ret'] / cum_max) - 1
        ax2.fill_between(dd.index, dd, 0, alpha=0.3, label=f"{model_name} DD")

    # Styling Top Plot
    ax1.set_title(f"Strategy Performance Comparison (Cost: {config['backtest']['cost']}bps)", fontsize=14)
    ax1.set_ylabel("Net Asset Value (NAV)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Styling Bottom Plot
    ax2.set_ylabel("Drawdown")
    ax2.set_ylim(-0.3, 0) # Adjust based on how much it crashes
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "final_performance.png"))
    print(f"📈 Chart saved to: {plots_dir}/final_performance.png")
    plt.show()

if __name__ == "__main__":
    plot_equity_curves()