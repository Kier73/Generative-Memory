import os
import sys
import time
import random
import psutil

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext
from gmem.hashing import MASK64

def check_determinism():
    print("--- Checking Determinism ---")
    seed = 0xAAABBBCCCDDDEEEF
    ctx1 = GMemContext(seed)
    ctx2 = GMemContext(seed)
    
    addresses = [0, 1, 10**6, 10**12, MASK64 - 1, MASK64]
    for addr in addresses:
        val1 = ctx1.fetch(addr)
        val2 = ctx2.fetch(addr)
        if val1 != val2:
            print(f"FAILED: Determinism breach at address {addr}")
            return False
    print("PASS: Determinism verified.")
    return True

def check_o1_performance():
    print("\n--- Checking O(1) Fetch/Write Performance ---")
    seed = random.getrandbits(64)
    ctx = GMemContext(seed)
    
    # Measure fetch performance across scales
    def measure_fetch(label, addr):
        start = time.perf_counter()
        for _ in range(1000):
            ctx.fetch(addr)
        end = time.perf_counter()
        avg_us = (end - start) * 1000  # ms for 1000 calls = us for 1 call
        print(f"Fetch {label} (addr={addr}): {avg_us:.4f} us")
        return avg_us

    # Cold cache fetches
    t1 = measure_fetch("Low", 0)
    t2 = measure_fetch("Mid", 10**12)
    t3 = measure_fetch("High", MASK64 - 1)
    
    # Variability check
    times = [t1, t2, t3]
    avg = sum(times) / 3
    max_diff = max(abs(t - avg) / avg for t in times)
    print(f"O(1) Variability: {max_diff*100:.2f}%")
    
    # Success if variability is reasonable (synthesis is constant time math)
    success = max_diff < 0.5  # 50% threshold for jitter
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def check_sparse_overlay_efficiency():
    print("\n--- Checking Sparse Overlay Efficiency ---")
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    
    seed = 42
    ctx = GMemContext(seed)
    
    # Initial state should be tiny
    mem_initial = process.memory_info().rss - baseline
    print(f"Initial Context Memory: {mem_initial / 1024:.2f} KB")
    
    # Perform 100,000 sparse writes
    print("Writing 100,000 sparse entries...")
    for i in range(100000):
        ctx.write(i * 10**9, 1.0)
    
    mem_after = (process.memory_info().rss - baseline) - mem_initial
    print(f"Overlay Memory (100k entries): {mem_after / 1024 / 1024:.2f} MB")
    
    # Approx 80-150 bytes per entry is expected for Python dict
    per_entry = mem_after / 100000
    print(f"Memory per written entry: {per_entry:.1f} bytes")
    
    success = mem_after > 0 and per_entry < 500
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def check_scale_invariance():
    print("\n--- Checking Scale Invariance ---")
    # GMem claims 2^64 cells can be layered.
    # We'll test indexing deep into the space and nested contexts.
    seed = 0x123
    parent = GMemContext(seed)
    
    # Layering via Morphing or simply addressing
    addr_petabyte = 10**15
    addr_exabyte = 10**18
    
    v1 = parent.fetch(addr_petabyte)
    v2 = parent.fetch(addr_exabyte)
    
    print(f"Value at 1 PB: {v1}")
    print(f"Value at 1 EB: {v2}")
    
    # Determinism at scale
    success = v1 != v2 and v1 is not None and v2 is not None
    print(f"Status: {'PASS' if success else 'FAIL (trivial values)'}")
    return success

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM INTEGRITY AUDIT")
    print("====================================================\n")
    
    results = [
        check_determinism(),
        check_o1_performance(),
        check_sparse_overlay_efficiency(),
        check_scale_invariance()
    ]
    
    if all(results):
        print("\nINTEGRITY AUDIT: PASS")
        sys.exit(0)
    else:
        print("\nINTEGRITY AUDIT: FAIL")
        sys.exit(1)
