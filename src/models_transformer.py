import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
from utils import load_config, set_seed

class TS_Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, n_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        x = self.embedding(x)
        x = self.transformer(x)
        # Aggregation: take the last time step
        return self.fc(x[:, -1, :])

def run_transformer_model(data_path, output_path):
    config = load_config()
    tf_cfg = config['models']['transformer']
    feat_cols = config['features']['list']
    target_col = config['features']['label']
    set_seed(config['models'].get('seed', 42))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🤖 Transformer using device: {device}")

    # 1. Load
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] < config['models']['train_split_date']].dropna()
    test_df = df[df['date'] >= config['models']['train_split_date']].copy()
    
    # 2. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feat_cols])
    X_test = scaler.transform(test_df[feat_cols])
    
    # 3. Reshape for Transformer: (Samples, Seq_Len=1, Features)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.FloatTensor(train_df[target_col].values).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)

    # 4. Loader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=tf_cfg['batch_size'], shuffle=True)

    # 5. Build
    model = TS_Transformer(
        input_dim=len(feat_cols),
        d_model=tf_cfg['d_model'],
        nhead=tf_cfg['n_heads'],
        n_layers=tf_cfg['n_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=tf_cfg['lr'])
    criterion = nn.MSELoss()

    # 6. Train
    model.train()
    print(f"🚀 Training Transformer: Heads={tf_cfg['n_heads']}, D_Model={tf_cfg['d_model']}")
    for epoch in range(tf_cfg['epochs']):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/len(train_loader):.6f}")

    # 7. Eval
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t.to(device)).cpu().numpy().flatten()
    
    test_df['score'] = test_preds
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df[['date', 'ticker', 'close', 'score', 'fwd_ret']].to_csv(output_path, index=False)