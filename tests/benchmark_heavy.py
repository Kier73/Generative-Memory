import sys
import os
import time
import random
import multiprocessing
import ctypes

# Ensure we can import the wrapper
sys.path.append(os.path.abspath("python"))
from gvm import GVM

def benchmark_writes(count=1000000):
    print(f"\n[STRESS] 1. Massive Ingestion: {count:,} Random Writes")
    gvm = GVM(seed=0xAAAA)
    
    # Pre-generate addresses to measure raw write speed, not Python random speed
    print("  Generating address table...")
    addrs = [random.randint(0, 10**14) for _ in range(count)]
    val = 1.0
    
    print("  Executing writes...")
    start = time.time()
    for addr in addrs:
        gvm.write(addr, val)
    end = time.time()
    
    duration = end - start
    ops = count / duration
    print(f"  Result: {ops:,.0f} OPS (Writes/Sec)")
    print(f"  Time: {duration:.2f}s")
    
    # Verify a few
    print("  Verifying random sample...")
    for _ in range(5):
        idx = random.randint(0, count-1)
        res = gvm.read(addrs[idx])
        if res != val:
            print(f"  [FAIL] Mismatch at index {idx}")
            return
    print("  [PASS] Integrity verified.")

def benchmark_bandwidth(size_gb=1.0):
    print(f"\n[STRESS] 2. Sustained Throughput: {size_gb} GB Sequential Read")
    gvm = GVM(seed=0xBBBB)
    
    element_count = int((size_gb * 1024**3) / 4) # 4 bytes per float
    chunk_size = 10 * 1024 * 1024 # 10M elements per chunk to stay cache friendly
    
    total_floats = 0
    start = time.time()
    
    # Read in chunks
    cursor = 0
    while total_floats < element_count:
        # We don't convert to list to avoid Python overhead, just touch the ctypes buffer
        # This calls the raw C bulk fetch
        remaining = element_count - total_floats
        batch = min(remaining, chunk_size)
        
        # Use low-level api to avoid python list conversion overhead for valid benchmark
        # We want to test ENGINE speed, not Python GC speed
        buffer = (ctypes.c_float * batch)()
        # Access private lib for raw speed test
        gvm._lib.gmem_fetch_bulk_f32(gvm.ctx, cursor, buffer, batch)
        
        cursor += batch
        total_floats += batch
        
        # progress
        if total_floats % (50 * 1024 * 1024) == 0:
            print(f"  Processed {total_floats*4 / 1024**3:.2f} GB...", end='\r')
            
    end = time.time()
    duration = end - start
    gb_per_sec = size_gb / duration
    
    print(f"\n  Result: {gb_per_sec:.2f} GB/s")
    print(f"  Time: {duration:.2f}s")
    if gb_per_sec > 5.0:
        print("  [IMPRESSIVE] faster than NVMe SSDs.")

def worker_stress(seed):
    gvm = GVM(seed)
    # Read 100MB
    buffer = (ctypes.c_float * 25000000)()
    gvm._lib.gmem_fetch_bulk_f32(gvm.ctx, 0, buffer, 25000000)
    return True

def benchmark_concurrency():
    print(f"\n[STRESS] 3. Concurrency: 4 Processes x 100MB Generation")
    start = time.time()
    
    with multiprocessing.Pool(4) as p:
        p.map(worker_stress, [0x1, 0x2, 0x3, 0x4])
        
    end = time.time()
    print(f"  Total Time: {end - start:.2f}s")
    print("  [PASS] No deadlocks or crashes.")

if __name__ == "__main__":
    try:
        print("=== GVM HEAVY STRESS SUITE ===")
        # No monkey patching needed if we access _lib from within GVM instances or properly export it
        import gvm
        
        benchmark_writes(1000000) 
        benchmark_bandwidth(1.0)
        benchmark_concurrency()
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
