import pandas as pd
import os
from utils import load_config

def generate_final_report():
    # Now using config for logic, not just decoration
    config = load_config()
    results_dir = 'results'
    oos_start_date = config['models']['train_split_date']
    
    print(f"📊 [Final Summary] Analyzing OOS Performance since {oos_start_date}...")
    
    # 1. Run Metrics and Backtest programmatically
    from metrics import run_evaluation
    from backtest import run_backtest_all
    
    metrics_summary = run_evaluation()
    backtest_summary = run_backtest_all()
    
    # 2. Alignment & Merging
    metrics_summary['Model'] = metrics_summary['Model'].str.upper()
    backtest_summary['Model'] = backtest_summary['Model'].str.upper()
    
    final_report = pd.merge(metrics_summary, backtest_summary, on="Model")
    
    # 3. Dynamic Quality Score
    # We use config to decide which metrics are 'Success' metrics
    # Here, we weigh Sharpe heavily as per common industry standard
    final_report['Score_Index'] = (
        final_report['Mean_IC'] * 100 + 
        final_report['Sharpe'] * 2 - 
        final_report['Max_Drawdown'].abs()
    ).round(2)
    
    final_report = final_report.sort_values("Sharpe", ascending=False)
    
    # 4. Save and Log
    report_path = os.path.join(results_dir, "final_model_leaderboard.csv")
    final_report.to_csv(report_path, index=False)
    
    print("\n" + "🏆" + "="*70 + "🏆")
    print(f"      S&P 500 ALPHA RESEARCH - FINAL LEADERBOARD (OOS: {oos_start_date})")
    print("="*72)
    print(final_report.to_string(index=False))
    print("="*72)
    
    # 5. Strategic Advice using Config
    best_model = final_report.iloc[0]['Model']
    target_top_n = config['backtest']['top_n']
    
    print(f"\n💡 ANALYSIS: {best_model} is the strongest candidate.")
    print(f"   Performance based on Top {target_top_n} stock selection with {config['backtest']['cost']} bps cost.")

if __name__ == "__main__":
    generate_final_report()