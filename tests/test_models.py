import pytest
import torch
import numpy as np
from models_mlp import DynamicMLP

def test_mlp_initialization():
    """Property: Weights should not be initialized to zero, NaN, or Inf."""
    input_dim = 10
    hidden_layers = [32, 16]
    model = DynamicMLP(input_dim=input_dim, hidden_layers=hidden_layers)
    
    for name, param in model.named_parameters():
        weights = param.data.cpu().numpy()
        
        # Check for numerical stability
        assert not np.isnan(weights).any(), f"NaN found in initialized weights of {name}"
        assert not np.isinf(weights).any(), f"Inf found in initialized weights of {name}"
        
        # Check for 'Dead Initialization' (weights all zero)
        # Using np.ptp (peak-to-peak) to ensure there is variance in weights
        if "weight" in name:
            assert np.ptp(weights) > 1e-5, f"Weights for {name} are too uniform/dead."

def test_mlp_gradient_flow():
    """Property: Loss should be able to propagate to the first layer (no vanishing gradients)."""
    input_dim = 5
    model = DynamicMLP(input_dim=input_dim, hidden_layers=[10, 10])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create synthetic input and target
    x = torch.randn(1, input_dim)
    y = torch.tensor([[1.0]])
    
    # Forward and Backward pass
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.MSELoss()(out, y)
    loss.backward()
    
    # Check if the weight gradient in the very first layer is non-zero
    # The first layer in self.net is index 0
    first_layer_grad = model.net[0].weight.grad
    
    assert first_layer_grad is not None, "Gradient was not calculated for the first layer."
    assert torch.abs(first_layer_grad).sum().item() > 0, "Gradients are zero; signal is lost in the network."

@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_mlp_batch_invariance(batch_size):
    """Property: Predictions for a sample should be identical regardless of batch size."""
    input_dim = 8
    model = DynamicMLP(input_dim=input_dim, hidden_layers=[16])
    model.eval() # Switch to eval mode to fix any potential BatchNorm/Dropout behavior
    
    # Create a single fixed sample
    fixed_sample = np.random.randn(1, input_dim).astype(np.float32)
    fixed_tensor = torch.from_numpy(fixed_sample)
    
    # Prediction 1: Single sample batch
    with torch.no_grad():
        pred_single = model(fixed_tensor).numpy()
    
    # Prediction 2: Part of a larger batch
    filler = np.random.randn(batch_size - 1, input_dim).astype(np.float32)
    batch_tensor = torch.from_numpy(np.vstack([fixed_sample, filler]))
    
    with torch.no_grad():
        pred_batch = model(batch_tensor).numpy()[0:1] # Extract the first sample result
        
    np.testing.assert_allclose(pred_single, pred_batch, atol=1e-6, 
                               err_msg="Model output changes based on batch size (Leakage/State error).")

def test_mlp_overfit_capability():
    """Property: On a tiny dataset, the model must be able to drive loss near zero."""
    input_dim = 4
    model = DynamicMLP(input_dim=input_dim, hidden_layers=[64])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # High LR for speed
    criterion = torch.nn.MSELoss()
    
    # 1 sample dataset
    x = torch.randn(1, input_dim)
    y = torch.tensor([[5.0]])
    
    # Train for 50 steps
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    
    final_loss = criterion(model(x), y).item()
    assert final_loss < 1e-3, f"Model failed to overfit a single data point. Loss: {final_loss}"