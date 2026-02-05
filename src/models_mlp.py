import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
from utils import load_config, set_seed

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(DynamicMLP, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def run_mlp_model(data_path, output_path):
    config = load_config()
    mlp_cfg = config['models']['mlp']
    feat_cols = config['features']['list']
    target_col = config['features']['label']
    set_seed(config['models'].get('seed', 42))

    # Device selection (Mac MPS, Nvidia CUDA, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  MLP using device: {device}")

    # 1. Load and Split
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] < config['models']['train_split_date']].dropna()
    test_df = df[df['date'] >= config['models']['train_split_date']].copy()
    
    # 2. Preprocess
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feat_cols])
    X_test = scaler.transform(test_df[feat_cols])
    y_train = train_df[target_col].values.astype(np.float32)

    # 3. Create DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    train_loader = DataLoader(train_ds, batch_size=mlp_cfg['batch_size'], shuffle=True)

    # 4. Initialize Model
    model = DynamicMLP(len(feat_cols), mlp_cfg['hidden_layers']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=mlp_cfg['lr'])

    # 5. Training Loop
    model.train()
    print(f"🧠 Training MLP for {mlp_cfg['epochs']} epochs...")
    for epoch in range(mlp_cfg['epochs']):
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
            print(f"Epoch [{epoch+1}/{mlp_cfg['epochs']}], Loss: {epoch_loss/len(train_loader):.6f}")

    # 6. Predict
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        test_preds = model(X_test_t).cpu().numpy().flatten()
    
    test_df['score'] = test_preds
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df[['date', 'ticker', 'close', 'score', 'fwd_ret']].to_csv(output_path, index=False)