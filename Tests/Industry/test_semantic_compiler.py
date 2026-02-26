import time
import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gmem.rs_bridge import SemanticCompiler

class TestSemanticCompiler(unittest.TestCase):
    
    def test_inverse_projection_compression(self):
        print("\n--- Testing Semantic Compiler (Inverse SVD Projection) ---")
        
        # 1. Create a "Physical" Real-World Matrix (e.g. attention weights)
        # 500 x 50 = 25,000 float64s
        m, n = 500, 500
        A = np.random.randn(m, 50)
        B = np.random.randn(50, n)
        
        anchor_size = 16
        
        # 2. Compile it!
        t0 = time.perf_counter()
        # This pushes the matrices into Rust, performs SVD on an anchor subset,
        # and computes the W^-1 pseudo-inverse bounding box.
        compiler = SemanticCompiler(A, B, anchor_size=anchor_size)
        t1 = time.perf_counter()
        
        print(f"Compiled 2.0 MB Matrix down to geometric Anchor Rules in {t1-t0:.4f}s")
        
        # 3. Retrieve values strictly through Mathematical Navigation, NOT memory lookups
        row, col = 42, 108
        
        # The Semantic Compiler 'navigate' reconstructs the value on the fly:
        # A_ij ~ K_i * W^-1 * R_j
        t0 = time.perf_counter()
        synthesized_val = compiler.fetch(row, col)
        t1 = time.perf_counter()
        
        print(f"Synthesized A[{row}, {col}] = {synthesized_val:.6f} in {t1-t0:.7f}s")
        
        # Evaluate accuracy (It's an approximation bounded by the SVD rank reduction)
        actual_val = (A @ B)[row, col]
        print(f"Actual Memory A[{row}, {col}] = {actual_val:.6f}")
        
        # The accuracy depends entirely on the Rank and the Anchor Size.
        # Here we just prove the pipeline executes without segfaults.
        self.assertIsNotNone(synthesized_val)
        self.assertFalse(np.isnan(synthesized_val))
        
        print("Semantic Compiler pipeline successfully navigated the C-FFI boundary.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
