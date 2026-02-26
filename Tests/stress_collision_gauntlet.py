import os
import sys
import time

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.vrns import synthesize_multichannel
from gmem.hashing import MASK64

def test_collision_gauntlet_extreme(samples=10_000_000):
    print(f"--- Stress Test: Extreme Collision Gauntlet ({samples:,} samples) ---")
    seed = 0x1337BEEF
    seen = {}
    collisions = 0
    
    # Use a faster range for massive sampling
    # We'll batch to show progress
    batch_size = 1_000_000
    start_time = time.perf_counter()
    
    for b in range(0, samples, batch_size):
        b_end = min(b + batch_size, samples)
        for i in range(b, b_end):
            # Non-sequential large-stride addresses
            addr = (i * 0xBF58476D1CE4E5B9) & MASK64
            val = synthesize_multichannel(addr, seed)
            
            # Note: Storing 10M floats in a dict might be memory intensive.
            # We'll use a set for raw presence, but a dict allows us to track source.
            if val in seen:
                collisions += 1
                if collisions <= 5:
                     print(f"  [COLLISION] val={val} at addr={addr} and {seen[val]}")
            else:
                seen[val] = addr
                
        elapsed = time.perf_counter() - start_time
        print(f"  Completed {b_end:,} samples... (Collisions: {collisions}, Time: {elapsed:.2f}s)")

    end_time = time.perf_counter()
    print(f"\nFinal Collisions: {collisions} / {samples:,}")
    
    # Target: Zero collisions in 10M samples for 16-prime CRT manifold
    success = collisions == 0
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM HIGH-DIFFICULTY: COLLISION GAUNTLET")
    print("====================================================\n")
    
    if test_collision_gauntlet_extreme():
        sys.exit(0)
    else:
        sys.exit(1)
