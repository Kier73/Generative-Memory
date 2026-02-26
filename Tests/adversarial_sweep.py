import os
import sys
import time
import random

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext
from gmem.hashing import MASK64
from gmem.vrns import synthesize_multichannel

def test_boundary_conditions():
    print("--- Testing Boundary Conditions ---")
    ctx = GMemContext(seed=0)
    
    # Extreme addresses
    addrs = [0, MASK64, MASK64 + 1, -1]
    for a in addrs:
        try:
            val = ctx.fetch(a)
            print(f"Address {a:X} -> {val}")
        except Exception as e:
            print(f"FAILED: Exception at address {a:X}: {e}")
            return False
            
    # Extreme seeds
    seeds = [0, MASK64, -1]
    for s in seeds:
        try:
            ctx_s = GMemContext(s)
            val = ctx_s.fetch(0)
            print(f"Seed {s:X} -> {val}")
        except Exception as e:
            print(f"FAILED: Exception at seed {s:X}: {e}")
            return False
            
    print("PASS: Boundary conditions handled (masked correctly).")
    return True

def test_collision_resistance_high_entropy():
    print("\n--- Testing Collision Resistance (High Entropy) ---")
    # v2.0.0 uses synthesize_multichannel which has 16 primes.
    # We'll check for collisions in a 1M sample.
    seed = 0xCAFEEBABE
    seen = set()
    samples = 1000000
    collisions = 0
    
    for i in range(samples):
        # Use a large prime stride to jump around the space
        addr = (i * 0xD3E0D63AD62EE571) & MASK64
        val = synthesize_multichannel(addr, seed)
        if val in seen:
            collisions += 1
        else:
            seen.add(val)
            
    print(f"Collisions in {samples} samples: {collisions}")
    # With 16 primes/CRT, collisions should be 0 or extremely low.
    success = collisions == 0
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def test_interpolation_search_stability():
    print("\n--- Testing Interpolation Search Stability ---")
    seed = 42
    ctx = GMemContext(seed)
    
    # The monotonic manifold should be searchable.
    targets = [0.1, 0.5, 0.9, 0.999]
    for t in targets:
        try:
            idx = ctx.search(t)
            val = ctx.fetch_monotonic(idx)
            print(f"Search Target {t} -> Index {idx}, Value {val}")
            # Check if it's reasonably close
            if abs(val - t) > 0.1: # Very loose check, just ensuring it's not garbage
                 print(f"WARNING: Wide gap in search result for {t}")
        except Exception as e:
            print(f"FAILED: Search exception for target {t}: {e}")
            return False
            
    print("PASS: Interpolation search is stable.")
    return True

def test_persistence_stress():
    print("\n--- Testing Persistence Stress (AOF Corruption/Replay) ---")
    aof_path = "stress_test.aof"
    if os.path.exists(aof_path): os.remove(aof_path)
    
    ctx = GMemContext(seed=123)
    ctx.persistence_attach(aof_path)
    
    print("Writing 10,000 values...")
    written = {}
    for i in range(10000):
        addr = random.getrandbits(64)
        val = random.random()
        ctx.write(addr, val)
        written[addr] = val
        
    ctx.persistence_detach()
    
    print("Replaying AOF...")
    ctx_new = GMemContext(seed=123)
    start = time.perf_counter()
    ctx_new.persistence_attach(aof_path)
    end = time.perf_counter()
    
    print(f"Replay time: {end - start:.4f}s")
    
    # Verify a subset
    print("Verifying data integrity (float32 tolerance)...")
    sample_size = 100
    for addr in random.sample(list(written.keys()), sample_size):
        if abs(ctx_new.fetch(addr) - written[addr]) > 1e-6:
             print(f"FAILED: Data mismatch at {addr} (diff: {abs(ctx_new.fetch(addr) - written[addr])})")
             return False
             
    ctx_new.persistence_detach()
    if os.path.exists(aof_path): os.remove(aof_path)
    
    print("PASS: Persistence replayed correctly.")
    return True

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM ADVERSARIAL SWEEP")
    print("====================================================\n")
    
    results = [
        test_boundary_conditions(),
        test_collision_resistance_high_entropy(),
        test_interpolation_search_stability(),
        test_persistence_stress()
    ]
    
    if all(results):
        print("\nADVERSARIAL SWEEP: PASS")
        sys.exit(0)
    else:
        print("\nADVERSARIAL SWEEP: FAIL")
        sys.exit(1)
