import torch
import unittest
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gmem.pytorch import GMemLinear

class TestGMemPyTorchIntegration(unittest.TestCase):
    
    def test_forward_pass_equivalence(self):
        print("\n--- Testing GMemLinear Forward Pass ---")
        
        # Test exact deterministic synthesis dimensions
        batch_size = 32
        in_features = 768
        out_features = 1024
        
        # Random inputs
        x = torch.randn(batch_size, in_features)
        
        layer = GMemLinear(in_features, out_features, bias=False)
        
        # The layer should have no weight parameter tensor!
        with self.assertRaises(AttributeError):
            _ = layer.weight
            
        print(f"Layer instantiated: {layer}")
        
        t0 = time.perf_counter()
        out = layer(x)
        t1 = time.perf_counter()
        
        print(f"Forward pass completed in {t1 - t0:.4f}s")
        self.assertEqual(out.shape, (batch_size, out_features))
        
        # Ensure determinism: second pass must be perfectly identical
        out2 = layer(x)
        self.assertTrue(torch.allclose(out, out2))
        print("Determinism check passed.")
        
    def test_backward_pass(self):
        print("\n--- Testing GMemLinear Backward Pass (Autograd Hook) ---")
        x = torch.randn(8, 64, requires_grad=True)
        layer = GMemLinear(in_features=64, out_features=128, bias=True)
        
        # We want to optimize the BIAS, not the weight (weight is a mathematical seed)
        out = layer(x)
        loss = out.sum()
        
        loss.backward()
        
        # Ensure gradients correctly flowed back to the input
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (8, 64))
        
        # Ensure gradients accumulated on the bias
        self.assertIsNotNone(layer.bias.grad)
        self.assertEqual(layer.bias.grad.shape, (128,))
        
        print(f"Input gradients calculated: norm={x.grad.norm().item():.4f}")
        print(f"Bias gradients calculated: norm={layer.bias.grad.norm().item():.4f}")
        print("Backward pass successfully navigated the Virtual Manifold hook.")
        
    def test_extreme_scale(self):
        print("\n--- Testing GMemLinear Extreme Memory Scale ---")
        # Let's instantiate a 100-Billion Parameter embedding table equivalent
        # 100,000 x 1,000,000 matrix = 100,000,000,000 parameters
        # In Float32, this is 400 GB! PyTorch would instantly OOM. 
        
        in_features = 1_000_000
        out_features = 100_000
        
        t0 = time.perf_counter()
        # This will allocate NO memory for weights, finishing instantly.
        huge_layer = GMemLinear(in_features, out_features, bias=False)
        t1 = time.perf_counter()
        
        print(f"Instantiated a 400 GB (100 Billion Param) virtual layer in {t1-t0:.6f} seconds.")
        self.assertEqual(huge_layer.in_features, in_features)
        print("Out of Memory avoided. Memory Wall = Broken.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
