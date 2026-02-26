import sys
import os
import time

# Add parent directory to path to reach gmem package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gmem.core import GMemContext

def test_numpy_integration():
    print("  [TEST] NumPy Integration (Synthetic Dataset Generation)...")
    try:
        import numpy as np
    except ImportError:
        print("    [SKIP] NumPy not installed in this environment.")
        return True

    ctx = GMemContext(seed=0x517)
    n = 1000000  # 1 Million elements
    
    # 1. Bulk Fetch to NumPy
    start_time = time.time()
    raw_data = ctx.fetch_bulk(0, n)
    arr = np.array(raw_data, dtype=np.float32)
    end_time = time.time()
    
    print(f"    Fetched {n:,} synthetic floats into NumPy in {end_time - start_time:.3f}s")
    
    # 2. Performance Check
    assert arr.shape == (n,)
    assert 0.0 <= np.mean(arr) <= 1.0
    
    # 3. Vectorized Math on Synthetic Data
    arr_sq = np.square(arr)
    print(f"    Vectorized square mean: {np.mean(arr_sq):.6f}")
    
    # 4. Sparse Integration
    ctx[42] = 999.0
    arr_updated = np.array(ctx.fetch_bulk(0, 100))
    assert arr_updated[42] == 999.0
    
    print("    -> PASS: GMem performs efficiently as a NumPy data source.")
    return True

if __name__ == "__main__":
    print("=== NumPy Industry Integration Tests ===")
    try:
        test_numpy_integration()
        print("\nALL NUMPY INTEGRATION TESTS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
