import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedLinearUnit(nn.Module):
    """
    The 'Gate' of the model. 
    GLU(x) = x * sigmoid(gate(x))
    It allows the model to control how much information flows through.
    """
    def __init__(self, input_size):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_size, input_size * 2)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    """
    GRN: The fundamental building block of TFT.
    It combines Skip Connections (ResNet) with Gating (LSTM-like logic).
    Structure: Input -> Linear -> ELU -> Linear -> GLU -> Add -> Norm
    """
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
        # Project residual if sizes don't match
        self.project_residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def forward(self, x, context=None):
        residual = self.project_residual(x) if self.project_residual else x
        
        # Standard GRN path
        x = self.fc1(x)
        if context is not None:
            x = x + context # Context injection (optional)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        
        # Residual Connection + Normalization
        return self.norm(x + residual)

class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer (Simplified for Alpha Prediction).
    
    Key Features from Paper:
    1. Gated Residual Networks (GRN) for filtering noise.
    2. LSTM Layer for "Local" processing (momentum/volatility).
    3. Self-Attention for "Global" dependencies (long-term patterns).
    4. Gating mechanisms everywhere to handle non-stationary markets.
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.1, output_dim=1):
        super(TFTModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Input Processing (Variable Selection Simulation)
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # 2. Local Processing (LSTM)
        # Captures immediate market moves (e.g., 3-day momentum)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_gate = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # 3. Global Processing (Attention)
        # Captures long-term regimes (e.g., 200-day trends)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.att_gate = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # 4. Fusion Layer (Combining Local + Global)
        self.fusion_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # 5. Output Layer
        self.final_projector = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len, Features]
        Returns:
            out: [Batch, 1] (Prediction for the final timestamp)
        """
        # --- Step 1: Input Encoding ---
        x_enc = self.input_encoder(x) # [B, T, H]
        x_enc = self.input_grn(x_enc) # Apply gating to suppress noisy features
        
        # --- Step 2: Local Temporal Processing (LSTM) ---
        # The paper argues LSTM is better for local context than Attention
        lstm_out, _ = self.lstm(x_enc)
        # Apply GRN/Skip connection over the LSTM output
        x_local = self.lstm_gate(lstm_out + x_enc) # [B, T, H]
        
        # --- Step 3: Global Temporal Processing (Attention) ---
        # Attention needs to look at the whole history
        # We query with the LOCAL context, Key/Value is also LOCAL context
        attn_out, _ = self.attention(x_local, x_local, x_local)
        # Apply GRN/Skip connection over Attention output
        x_global = self.att_gate(attn_out + x_local) # [B, T, H]
        
        # --- Step 4: Final Fusion ---
        x_fused = self.fusion_grn(x_global)
        
        # We only care about the prediction at the last timestep T
        last_step = x_fused[:, -1, :] # [B, H]
        
        # --- Step 5: Output ---
        out = self.final_projector(last_step)
        return out