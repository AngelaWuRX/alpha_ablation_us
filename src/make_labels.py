import pandas as pd
import os

def generate_labels(input_path, output_path):
    print(f"🏷️  正在生成标签 (Labels): {input_path}")
    df = pd.read_csv(input_path)
    
    df = df.sort_values(['ticker', 'date'])
    

    df['fwd_ret'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1
    

    df['fwd_ret'] = df['fwd_ret'].clip(-0.1, 0.1)
    
    df = df.dropna(subset=['fwd_ret'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ 标签生成完毕，包含 fwd_ret 的数据已保存至: {output_path}")

if __name__ == "__main__":
    generate_labels('data/factors.csv', 'data/factors.csv')