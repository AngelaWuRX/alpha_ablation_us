import pandas as pd
import numpy as np
from utils import load_config

def run_sanity_check():
    config = load_config()
    file_path = config['universe']['processed_path']
    
    print(f"🔍 [Sanity Check] 正在深度审计数据: {file_path}")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # --- Check 1: 未来函数漏斗 (The Golden Rule) ---
    # 原理：fwd_ret 绝不能与当天的涨幅 (close/open-1) 完全一样
    # 如果相关性接近 1.0，说明你预测的是“今天”，而不是“明天”
    df['today_ret'] = df.groupby('ticker')['close'].pct_change()
    leakage_corr = df['today_ret'].corr(df['fwd_ret'])
    
    print(f"\n1️⃣ 未来函数检测:")
    if leakage_corr > 0.9:
        print(f"   ❌ 警告：检测到严重的未来函数！Corr={leakage_corr:.4f}")
    else:
        print(f"   ✅ 通过：今日收益与目标标签相关性为 {leakage_corr:.4f} (低相关性是正常的)")

    # --- Check 2: 数据缺失与空值 ---
    null_counts = df.isnull().sum()
    print(f"\n2️⃣ 空值检测:")
    if null_counts.any():
        print(f"   ❌ 错误：发现空值！\n{null_counts[null_counts > 0]}")
    else:
        print(f"   ✅ 通过：全字段无空值")

    # --- Check 3: 股票与时间戳的完整性 ---
    print(f"\n3️⃣ 样本覆盖率:")
    n_tickers = df['ticker'].nunique()
    n_dates = df['date'].nunique()
    print(f"   💡 当前股票池数量: {n_tickers}")
    print(f"   💡 时间步总数: {n_dates}")
    
    # --- Check 4: 因子分布审计 ---
    print(f"\n4️⃣ 因子数值范围 (Winsorization 检查):")
    stats = df[config['features']['list'] + ['fwd_ret']].agg(['min', 'max', 'mean'])
    print(stats.to_string())

    # --- Check 5: 逻辑一致性 ---
    # 随机抽样一只股票，检查时间是否连续，fwd_ret 是否真的对应下一天
    sample_ticker = df['ticker'].iloc[0]
    sample_df = df[df['ticker'] == sample_ticker].sort_values('date').head(3)
    print(f"\n5️⃣ 逻辑对齐抽样 (Ticker: {sample_ticker}):")
    print(sample_df[['date', 'close', 'fwd_ret']])
    print("   👉 请肉眼确认：第一行的 fwd_ret 是否等于 (第二行close / 第一行close - 1)")

if __name__ == "__main__":
    run_sanity_check()