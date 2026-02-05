import pandas as pd
import xgboost as xgb
from utils import load_config, set_seed

def run_xgb_model(data_path, output_path):
    config = load_config()
    xgb_cfg = config['models']['xgb']
    set_seed(config['models']['seed'])
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    factor_cols = config['features']['list']
    train_df = df[df['date'] < config['models']['train_split_date']].dropna()
    test_df = df[df['date'] >= config['models']['train_split_date']].copy()
    
    # 使用 config 参数
    model = xgb.XGBRegressor(
        n_estimators=xgb_cfg['n_estimators'],
        max_depth=xgb_cfg['max_depth'],
        learning_rate=xgb_cfg['learning_rate'],
        random_state=config['models']['seed'],
        n_jobs=-1
    )
    
    model.fit(train_df[factor_cols], train_df['fwd_ret'])
    test_df['score'] = model.predict(test_df[factor_cols])
    test_df[['date', 'ticker', 'close', 'score', 'fwd_ret']].to_csv(output_path, index=False)