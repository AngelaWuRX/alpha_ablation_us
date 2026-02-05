# src/train_all.py
import sys
import os
from utils import load_config, set_seed
import models_linear
import models_xgb
import models_mlp
import models_transformer

# 自动处理路径，防止导入失败
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)



def run_pipeline():
    config = load_config()
    # 1. 锁死全局种子
    set_seed(config['models'].get('seed', 42))
    
    data_path = config['universe']['processed_path']
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 任务配置：展示名 | 模块 | 真实函数名
    tasks = [
        ("Linear", models_linear, "run_linear_baseline"),
        ("XGBoost", models_xgb, "run_xgb_model"),
        ("MLP", models_mlp, "run_mlp_model"),
        ("Transformer", models_transformer, "run_transformer_model")
    ]

    print(f"\n🚀 启动全链路实验: {config['project_name']}")
    
    for name, module, func in tasks:
        output_file = f"{results_dir}/signals_{name.lower()}.csv"
        print(f"▶️  正在训练: {name} ...")
        
        # 2. 在每个模型开始前再次校准种子（防止上个模型对随机状态的干扰）
        set_seed(42) 
        
        try:
            worker = getattr(module, func)
            worker(data_path, output_file)
            print(f"   ✅ 信号已产出至: {output_file}")
        except Exception as e:
            print(f"   ❌ {name} 失败: {e}")

if __name__ == "__main__":
    run_pipeline()