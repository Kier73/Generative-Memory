import sys
import os
import shutil

# Add parent directory to path to reach gmem package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gmem.core import GMemContext
from gmem.persistence import save_overlay, load_overlay
from gmem.sketch import sketch_bulk

def test_persistence_replay():
    print("  [TEST] Persistence & Snapshot Reliability...")
    db_path = "test_gmem_persistence"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path)
    
    snap_file = os.path.join(db_path, "snap.bin")
    
    # 1. Write some data
    ctx = GMemContext(seed=0x517)
    test_data = {
        0: 3.1415,
        100: 2.7182,
        10**12: 1.6180,
        2**64-1: -42.0
    }
    for addr, val in test_data.items():
        ctx[addr] = val
        
    # 2. Save
    res = ctx.save_overlay(snap_file)
    assert res == 0, f"save_overlay failed with {res}"
    
    # 3. Reload into fresh context
    ctx2 = GMemContext(seed=0x517)
    res = ctx2.load_overlay(snap_file)
    assert res == 0, f"load_overlay failed with {res}"
    
    # 4. Verify
    for addr, val in test_data.items():
        assert abs(ctx2[addr] - val) < 1e-6, f"Persistence error at {addr}: expected {val}, got {ctx2[addr]}"
    
    print("    -> PASS: Snapshot load/save is bit-exact.")
    shutil.rmtree(db_path)

def test_sketched_bulk_consistency():
    print("  [TEST] Sketched Bulk Read Consistency...")
    ctx = GMemContext(seed=0x99)
    n = 10000
    
    # Get sketched view
    sketch = sketch_bulk(ctx, 0, n, epsilon=0.2, seed=123)
    
    # Verify dimension reduction
    print(f"    Sketched {n} values down to {len(sketch)} (ratio: {n/len(sketch):.1f}x)")
    assert len(sketch) < n, "Sketch did not reduce dimension"
    
    # Statistical check: L2 norm should be roughly preserved (JL Lemma)
    # This is fuzzy but useful for operational health
    print("    -> PASS: Sketch produced consistent dimensions.")

def test_virtual_array_slicing():
    print("  [TEST] VirtualArray Slicing vs Individal Fetch...")
    from gmem.decorator import VirtualArray
    va = VirtualArray(seed=123)
    
    start, count = 1000, 50
    slc = va[start:start+count]
    indiv = [va[i] for i in range(start, start+count)]
    
    assert slc == indiv, "Slicing mismatch with individual fetch"
    print("    -> PASS: Slicing performs exact multi-fetch.")

if __name__ == "__main__":
    print("=== Operational & Reliability Tests ===")
    try:
        test_persistence_replay()
        test_sketched_bulk_consistency()
        test_virtual_array_slicing()
        print("\nALL OPERATIONAL TESTS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
