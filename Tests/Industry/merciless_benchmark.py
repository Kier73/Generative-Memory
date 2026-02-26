import os
import sys
import time
import hashlib
import random
import multiprocessing as mp

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gmem.core import GMemContext
from gmem.vrns import synthesize_multichannel
from gmem.ntt import synthesize_product_law, inverse_product_law
from gmem.hashing import fmix64, vl_mask, MASK64

def time_it(func, iters, *args):
    start = time.perf_counter()
    for _ in range(iters):
        func(*args)
    return time.perf_counter() - start

def speed_sha256(addr, seed_bytes):
    # Simulated SHA-256 noise generation to compare
    h = hashlib.sha256()
    h.update(addr.to_bytes(8, 'little') + seed_bytes)
    return h.digest()

def benchmark_throughput():
    print("====================================================")
    print("   GMEM MERCILESS BENCHMARK: SPEED & ACCURACY")
    print("====================================================\n")
    
    iters = 500_000
    addr = 0x1337BEEFCAFEBABE
    seed = 0x9E3779B97F4A7C15
    seed_bytes = seed.to_bytes(8, 'little')
    
    print(f"--- 1. Pure Synthesis Throughput ({iters:,} iterations) ---")
    
    t_sha = time_it(speed_sha256, iters, addr, seed_bytes)
    print(f"SHA-256 (Baseline): {t_sha:.4f}s ({iters/t_sha:,.0f} ops/sec)")
    
    t_vRNS = time_it(synthesize_multichannel, iters, addr, seed)
    print(f"vRNS 16-Prime + fmix64: {t_vRNS:.4f}s ({iters/t_vRNS:,.0f} ops/sec)")
    print(f"Speedup vs SHA-256: {t_sha / t_vRNS:.2f}x")

    t_fmix = time_it(fmix64, iters, addr ^ seed)
    print(f"fmix64 raw mixing: {t_fmix:.4f}s ({iters/t_fmix:,.0f} ops/sec)")

    t_vl = time_it(vl_mask, iters, addr, seed)
    print(f"Feistel vl_mask (Bijective): {t_vl:.4f}s ({iters/t_vl:,.0f} ops/sec)")

    print(f"\n--- 2. Core Context Overhead ({iters // 5:,} iterations) ---")
    ctx = GMemContext(seed)
    
    # Pre-populate some cache/overlay
    ctx.write(addr, 0.42)
    
    iters_ctx = iters // 5
    t_fetch_clean = time_it(ctx.fetch, iters_ctx, addr + 1)
    print(f"Context Fetch (Clean/Synthesis): {t_fetch_clean:.4f}s ({iters_ctx/t_fetch_clean:,.0f} ops/sec)")
    
    t_fetch_dirty = time_it(ctx.fetch, iters_ctx, addr)
    print(f"Context Fetch (Dirty/Overlay): {t_fetch_dirty:.4f}s ({iters_ctx/t_fetch_dirty:,.0f} ops/sec)")
    
    print(f"\n--- 3. Law Composition & NTT Field ({iters:,} iterations) ---")
    
    t_ntt = time_it(synthesize_product_law, iters, seed, addr)
    print(f"Goldilocks Multiply (Composition): {t_ntt:.4f}s ({iters/t_ntt:,.0f} ops/sec)")
    
    t_ntt_inv = time_it(inverse_product_law, iters, seed, addr)
    print(f"Goldilocks Inverse (Recovery): {t_ntt_inv:.4f}s ({iters/t_ntt_inv:,.0f} ops/sec)")

def test_accuracy_collisions():
    print(f"\n--- 4. Accuracy & Scale Invariance (Collision Resistance) ---")
    samples = 1_000_000
    seed = 0x8BADF00D
    seen = set()
    collisions = 0
    t0 = time.perf_counter()
    
    for i in range(samples):
        # High stride address access
        addr = (i * 0x94D049BB133111EB) & MASK64
        val = synthesize_multichannel(addr, seed)
        if val in seen:
            collisions += 1
        seen.add(val)
        
    t1 = time.perf_counter()
    print(f"Sampled {samples:,} widely distributed points in {t1-t0:.2f}s.")
    print(f"Collisions found: {collisions} (Target: 0)")
    
    if collisions == 0:
        print("-> Scale Invariance & Orthogonality: PASSED")
    else:
        print("-> Scale Invariance & Orthogonality: FAILED")


if __name__ == '__main__':
    benchmark_throughput()
    test_accuracy_collisions()
