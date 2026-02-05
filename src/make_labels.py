import pandas as pd
import os

def generate_labels(input_path, output_path):
    print(f"🏷️  正在生成标签 (Labels): {input_path}")
    df = pd.read_csv(input_path)
    
    # 确保按股票和时间排序，这是计算错位（Shift）的前提
    df = df.sort_values(['ticker', 'date'])
    
    # --- 核心逻辑：计算 Fwd_Ret ---
    # 我们预测的是：下一期的收盘价相对于这一期收盘价的涨幅
    # pct_change(-1) 的意思是：(下期价格 - 本期价格) / 本期价格
    df['fwd_ret'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1
    
    # 处理异常值（去极值）：金融数据里常有异常波动，进行 Winsorize 处理
    # 限制在 -10% 到 +10% 之间，防止极端噪音带偏模型
    df['fwd_ret'] = df['fwd_ret'].clip(-0.1, 0.1)
    
    # 删掉最后一行（因为最后一行没有下一期收益了）
    df = df.dropna(subset=['fwd_ret'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ 标签生成完毕，包含 fwd_ret 的数据已保存至: {output_path}")

if __name__ == "__main__":
    generate_labels('data/factors.csv', 'data/factors.csv') # 直接覆盖原文件