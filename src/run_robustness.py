import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_config, set_seed
from backtest import backtest_topn
# Import your training functions to re-run with different seeds
from models_mlp import run_mlp_model 

def run_robustness_test():
    config = load_config()
    results_dir = 'results'
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- 1. Seed Sensitivity (Testing Model Stability) ---
    # We use different seeds to see if the MLP's performance is consistent
    seeds = [42, 7, 2026, 88, 123]
    seed_sharpes = []
    
    print("🔬 Phase 1: Seed Sensitivity (Re-training Models)...")
    for s in seeds:
        set_seed(s) # Now using the seed!
        temp_output = f"{results_dir}/signals_robust_seed_{s}.csv"
        
        # Re-run training (using MLP as an example)
        run_mlp_model(config['universe']['processed_path'], temp_output)
        
        # Backtest the new signal
        df_sig = pd.read_csv(temp_output)
        _, summary = backtest_topn(df_sig, top_n=config['backtest']['top_n'])
        seed_sharpes.append(summary['Sharpe'])
    
    # Calculate Variance using Numpy
    sharpe_std = np.std(seed_sharpes)
    sharpe_mean = np.mean(seed_sharpes)

    # --- 2. Cost Sensitivity (Testing Alpha Decay) ---
    print("\n🔬 Phase 2: Cost Sensitivity Analysis...")
    signal_path = os.path.join(results_dir, "signals_transformer.csv")
    df_trans = pd.read_csv(signal_path)
    
    costs = np.linspace(0, 50, 6) # [0, 10, 20, 30, 40, 50] bps
    cost_metrics = []
    for c in costs:
        _, summary = backtest_topn(df_trans, top_n=10, cost_bps=c)
        cost_metrics.append(summary['Sharpe'])

    # --- 3. Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Seed Distribution Plot
    ax1.hist(seed_sharpes, bins=5, color='skyblue', edgecolor='black')
    ax1.axvline(sharpe_mean, color='red', linestyle='dashed', label=f'Mean: {sharpe_mean:.2f}')
    ax1.set_title(f"Seed Stability (Std Dev: {sharpe_std:.4f})")
    ax1.set_xlabel("Annualized Sharpe")
    ax1.legend()

    # Cost Decay Plot
    ax2.plot(costs, cost_metrics, marker='s', linestyle='-', color='darkorange')
    ax2.set_title("Alpha Decay vs. Transaction Costs")
    ax2.set_xlabel("Cost (BPS)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/robustness_results.png")
    print(f"\n📊 Robustness report saved to {plots_dir}/robustness_results.png")
    
    print(f"💡 Statistical Verdict: Mean Sharpe {sharpe_mean:.2f} with stability {1 - sharpe_std:.2%}")

if __name__ == "__main__":
    run_robustness_test()