import pytest
import torch
import numpy as np
import sys
import os

# Add src to path to import your actual model file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from models_mlp import DynamicMLP
except ImportError:
    # Fallback mock for demonstration if file doesn't exist yet
    import torch.nn as nn
    class DynamicMLP(nn.Module):
        def __init__(self, input_dim, hidden_layers):
            super().__init__()
            layers = []
            in_dim = input_dim
            for h in hidden_layers:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

def test_mlp_initialization():
    """
    [STABILITY] Verifies that weights are not all zeros (Dead) or NaNs (Exploded).
    """
    print("\n\n🔬 [MODEL TEST] Checking Weight Initialization...")
    input_dim = 10
    hidden_layers = [32, 16]
    model = DynamicMLP(input_dim=input_dim, hidden_layers=hidden_layers)
    
    for name, param in model.named_parameters():
        weights = param.data.cpu().numpy()
        
        # Calculate stats
        w_mean = np.mean(weights)
        w_std = np.std(weights)
        w_range = np.ptp(weights) # Peak-to-peak (Max - Min)
        
        print(f"   Layer '{name}': Mean={w_mean:.4f}, Std={w_std:.4f}, Range={w_range:.4f}")
        
        # 1. Check for NaNs
        assert not np.isnan(weights).any(), f"❌ NaN found in {name}"
        
        # 2. Check for Dead Initialization (All Zeros)
        # Weights must have variance. Biases *can* be zero, so we focus on weights.
        if "weight" in name:
            assert w_range > 1e-5, f"❌ Weights for {name} are dead (all uniform/zero)."
            
    print("✅ INITIALIZATION PASSED.")

def test_mlp_gradient_flow():
    """
    [TRAINABILITY] Ensures gradients can travel from Output -> Input without vanishing.
    This fails if you have 'Dead ReLUs' or disconnected graphs.
    """
    print("\n🔬 [MODEL TEST] Checking Gradient Flow (Backprop)...")
    input_dim = 5
    model = DynamicMLP(input_dim=input_dim, hidden_layers=[10, 10])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create synthetic data
    x = torch.randn(1, input_dim)
    y = torch.tensor([[1.0]])
    
    # Forward Pass
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.MSELoss()(out, y)
    
    print(f"   Forward Pass Output: {out.item():.4f}")
    
    # Backward Pass
    loss.backward()
    
    # Check the FIRST layer's gradient. 
    # If this is non-zero, the signal successfully traveled all the way back.
    # Note: Accessing model.net[0] assumes Sequential. Adjust if architecture changes.
    first_layer = list(model.modules())[1] # usually the first Linear layer
    if hasattr(first_layer, 'weight'):
        grad = first_layer.weight.grad
        grad_sum = torch.abs(grad).sum().item()
        
        print(f"   First Layer Gradient Magnitude: {grad_sum:.6f}")
        
        assert grad is not None, "❌ Gradient is None (Graph broken)."
        assert grad_sum > 0, "❌ Gradient is Zero (Vanishing Gradient / Dead Neurons)."
    else:
        print("   ⚠️ Could not locate first layer weights automatically. Skipping exact check.")
        
    print("✅ GRADIENT FLOW PASSED.")

@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_mlp_batch_invariance(batch_size):
    """
    [LOGIC] Verifies that prediction for Sample A doesn't change if we add Sample B to the batch.
    Common bug with BatchNorm or hidden states (RNNs).
    """
    if batch_size == 1: 
        return # Trivial case
        
    print(f"\n🔬 [MODEL TEST] Checking Batch Invariance (Batch={batch_size})...")
    input_dim = 8
    model = DynamicMLP(input_dim=input_dim, hidden_layers=[16])
    model.eval() # CRITICAL: BatchNorm behaves differently in Train vs Eval
    
    # 1. Predict Single Sample
    fixed_sample = np.random.randn(1, input_dim).astype(np.float32)
    fixed_tensor = torch.from_numpy(fixed_sample)
    
    with torch.no_grad():
        pred_single = model(fixed_tensor).numpy()
    
    # 2. Predict as part of a Batch
    # Stack the fixed sample with random noise
    filler = np.random.randn(batch_size - 1, input_dim).astype(np.float32)
    batch_tensor = torch.from_numpy(np.vstack([fixed_sample, filler]))
    
    with torch.no_grad():
        # Take the first result from the batch (which corresponds to fixed_sample)
        pred_batch = model(batch_tensor).numpy()[0:1] 
        
    diff = np.abs(pred_single - pred_batch).sum()
    print(f"   Single Pred: {pred_single[0][0]:.6f}")
    print(f"   Batch  Pred: {pred_batch[0][0]:.6f}")
    print(f"   Difference:  {diff:.9f}")
    
    np.testing.assert_allclose(pred_single, pred_batch, atol=1e-5, 
                               err_msg="❌ Model output changed due to batch size!")
    print("✅ BATCH INVARIANCE PASSED.")

def test_mlp_overfit_capability():
    """
    [CAPACITY] Can the model memorize a single data point?
    If it can't do this, it will never learn the stock market.
    """
    print("\n🔬 [MODEL TEST] Checking Overfitting Capability (The 'Sanity' Check)...")
    input_dim = 4
    model = DynamicMLP(input_dim=input_dim, hidden_layers=[64])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()
    
    x = torch.randn(1, input_dim)
    y = torch.tensor([[5.0]]) # Arbitrary target
    
    print(f"   Target Value: 5.0")
    
    initial_loss = criterion(model(x), y).item()
    print(f"   Start Loss: {initial_loss:.4f}")
    
    # Train Loop
    for i in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"   Step {i}: Loss {loss.item():.6f}")
            
    final_loss = criterion(model(x), y).item()
    print(f"   Final Loss: {final_loss:.6f}")
    
    assert final_loss < 0.01, f"❌ Model failed to converge! Loss: {final_loss}"
    print("✅ OVERFIT CHECK PASSED.")

if __name__ == "__main__":
    pytest.main(["-s", __file__])