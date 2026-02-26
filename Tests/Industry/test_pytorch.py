import sys
import os
import time

# Add parent directory to path to reach gmem package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gmem.core import GMemContext

def test_pytorch_integration():
    print("  [TEST] PyTorch Integration (Tensor Hydration)...")
    try:
        import torch
    except ImportError:
        print("    [SKIP] PyTorch not installed in this environment.")
        return True

    ctx = GMemContext(seed=0xCAFE)
    n = 1024 * 1024  # 1M elements
    
    # 1. Hydrate Tensor from GMem
    start_time = time.time()
    # Direct creation from list is fine for verification
    raw_data = ctx.fetch_bulk(0, n)
    tensor = torch.tensor(raw_data, dtype=torch.float32)
    end_time = time.time()
    
    print(f"    Hydrated {n:,} elements into PyTorch Tensor in {end_time - start_time:.3f}s")
    
    # 2. Basic Tensor Ops
    assert tensor.ndim == 1
    assert tensor.shape[0] == n
    
    mean_val = torch.mean(tensor)
    print(f"    Tensor Mean: {mean_val.item():.6f}")
    assert 0.4 <= mean_val <= 0.6  # Law of large numbers for uniform [0,1]
    
    # 3. Gradient Flow (Mock)
    # Even if data is synthetic, we can use it as weights
    tensor.requires_grad = True
    loss = (tensor ** 2).sum()
    loss.backward()
    assert tensor.grad is not None
    print(f"    Gradient flow verified. Loss: {loss.item():.2f}")
    
    print("    -> PASS: GMem integrates seamlessly with PyTorch tensors.")
    return True

if __name__ == "__main__":
    print("=== PyTorch Industry Integration Tests ===")
    try:
        test_pytorch_integration()
        print("\nALL PYTORCH INTEGRATION TESTS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
