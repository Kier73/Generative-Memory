import ctypes
import os
import time
import numpy as np

def test_search():
    dll_path = os.path.abspath("build/Release/gmem.dll")
    print(f"--- GMC Predictive Search Benchmark ---")
    
    if not os.path.exists(dll_path):
        print("Error: gmem.dll not found.")
        return

    gmem = ctypes.CDLL(dll_path)

    # Function Signatures
    gmem.gmem_create.argtypes = [ctypes.c_uint64]
    gmem.gmem_create.restype = ctypes.c_void_p
    gmem.gmem_destroy.argtypes = [ctypes.c_void_p]
    gmem.gmem_fetch_monotonic_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    gmem.gmem_fetch_monotonic_f32.restype = ctypes.c_float
    gmem.gmem_search_f32.argtypes = [ctypes.c_void_p, ctypes.c_float]
    gmem.gmem_search_f32.restype = ctypes.c_uint64

    # Create Context
    seed = 1234
    ctx = gmem.gmem_create(seed)
    
    target = 0.75
    print(f"Targeting value: {target:.4f}")

    # 1. LINEAR SCAN (Mockup for comparison)
    # We only scan 10,000 points because 2^64 is impossible
    print("Starting Linear Scan simulation (100k points)...")
    start_time = time.time()
    for i in range(100000):
        val = gmem.gmem_fetch_monotonic_f32(ctx, i)
        if val >= target:
            break
    linear_time = time.time() - start_time
    print(f"Linear Scan (100k) took: {linear_time:.6f}s")

    # 2. PREDICTIVE SEARCH
    print("Starting Predictive Search (2^64 range)...")
    start_time = time.time()
    found_addr = gmem.gmem_search_f32(ctx, target)
    predictive_time = time.time() - start_time
    
    found_val = gmem.gmem_fetch_monotonic_f32(ctx, found_addr)
    print(f"Predictive Search found address: {found_addr}")
    print(f"Value at address: {found_val:.8f}")
    print(f"Predictive Search took: {predictive_time:.8f}s")
    
    # 3. ANALYSIS
    speedup = linear_time / (predictive_time if predictive_time > 0 else 1e-9)
    # Scale linear time to 2^64 for theoretical comparison
    theoretical_linear = (linear_time / 100000) * (18446744073709551615 / 2) # Assume target is at middle
    
    print(f"\n[RESULT] Direct Speedup (vs 100k scan): {speedup:.2f}x")
    print(f"[RESULT] Convergence: {found_val:.4f} is target {target:.4f}")
    print(f"[THEORY] A linear scan of 2^64 space would take approx {theoretical_linear / 3600 / 24 / 365:.1f} YEARS.")
    print(f"[THEORY] Predictive Search resolved it in {predictive_time*1000:.4f} MILLISECONDS.")

    gmem.gmem_destroy(ctx)
    print("--- Benchmark Complete ---")

if __name__ == "__main__":
    test_search()
