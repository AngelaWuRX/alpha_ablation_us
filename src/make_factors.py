import pandas as pd
import os
from utils import load_config

def generate_factors():
    config = load_config()
    raw_path = config['universe']['raw_data_path']
    output_path = config['universe']['processed_path']
    
    print(f"🏗️  正在从原始数据构建因子特征: {raw_path}")
    
    if not os.path.exists(raw_path):
        print(f"❌ 错误：找不到原始数据文件 {raw_path}。请确保路径正确。")
        return

    # 1. 加载数据
    df = pd.read_csv(raw_path)
    
    # 2. 统一字段名（处理 Kaggle 大小写问题）
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. 核心步骤：按股票分组并排序（防止数据串户）
    df = df.sort_values(['ticker', 'date'])
    grouped = df.groupby('ticker')

    print("🧪 正在计算 4 大核心技术因子...")

    # --- 因子工程开始 ---
    
    # Factor 1: mom_20 (20日动量 - 过去一个月的累计收益)
    df['mom_20'] = grouped['close'].pct_change(20)
    
    # Factor 2: rev_5 (5日反转 - 短期内是否涨过头了)
    df['rev_5'] = grouped['close'].pct_change(5)
    
    # Factor 3: vol_20 (20日波动率 - 风险指标)
    # 先计算日收益率，再算滚动标准差
    df['daily_ret'] = grouped['close'].pct_change()
    df['vol_20'] = grouped['daily_ret'].transform(lambda x: x.rolling(20).std())
    
    # Factor 4: ma_gap_20 (20日均线偏离度 - 衡量价格是否回归)
    df['ma_20'] = grouped['close'].transform(lambda x: x.rolling(20).mean())
    df['ma_gap_20'] = (df['close'] / df['ma_20']) - 1

    # --- 因子工程结束 ---

    # 4. 清理：去掉计算滚动窗口时产生的 NaN 值
    before_len = len(df)
    df = df.dropna(subset=['mom_20', 'rev_5', 'vol_20', 'ma_gap_20'])
    print(f"🧹 已清理滚动窗口产生的空值: {before_len - len(df)} 行")

    # 5. 只保留必要的列，节省内存和后续处理速度
    final_cols = ['date', 'ticker', 'close'] + config['features']['list']
    df = df[final_cols]

    # 6. 保存到 data/factors.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ 因子构建完成！共有 {len(df)} 行有效样本。")
    print(f"📂 结果保存至: {output_path}")

if __name__ == "__main__":
    generate_factors()