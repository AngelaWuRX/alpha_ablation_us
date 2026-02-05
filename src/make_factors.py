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

    df = pd.read_csv(raw_path)
    
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values(['ticker', 'date'])
    grouped = df.groupby('ticker')

    print("🧪 正在计算 4 大核心技术因子...")

    
    df['mom_20'] = grouped['close'].pct_change(20)
    
    df['rev_5'] = grouped['close'].pct_change(5)
    
    df['daily_ret'] = grouped['close'].pct_change()
    df['vol_20'] = grouped['daily_ret'].transform(lambda x: x.rolling(20).std())
    
    df['ma_20'] = grouped['close'].transform(lambda x: x.rolling(20).mean())
    df['ma_gap_20'] = (df['close'] / df['ma_20']) - 1


    before_len = len(df)
    df = df.dropna(subset=['mom_20', 'rev_5', 'vol_20', 'ma_gap_20'])
    print(f"🧹 已清理滚动窗口产生的空值: {before_len - len(df)} 行")

    final_cols = ['date', 'ticker', 'close'] + config['features']['list']
    df = df[final_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ 因子构建完成！共有 {len(df)} 行有效样本。")
    print(f"📂 结果保存至: {output_path}")

if __name__ == "__main__":
    generate_factors()