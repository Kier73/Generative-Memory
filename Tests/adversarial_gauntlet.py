import os
import sys
import time
import random
import numpy as np
import scipy.stats as stats
import psutil

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext
from gmem.hashing import vl_mask, MASK64
from gmem.vrns import synthesize_multichannel

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def test_avalanche_effect(samples=1000):
    print(f"--- Avalanche Effect Audit ({samples} samples) ---")
    distances = []
    seed = random.getrandbits(64)
    for _ in range(samples):
        addr = random.getrandbits(64)
        h1 = vl_mask(addr, seed)
        # Flip a random bit
        bit = random.randint(0, 63)
        addr_flipped = addr ^ (1 << bit)
        h2 = vl_mask(addr_flipped, seed)
        distances.append(hamming_distance(h1, h2))
    
    avg_dist = sum(distances) / samples
    print(f"Average Hamming Distance: {avg_dist:.2f} bits (Target: 32)")
    # Success if within 10% of 32 bits
    success = 28 <= avg_dist <= 36
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def test_uniformity(samples=1000000):
    print(f"\n--- Uniformity Audit ({samples} samples) ---")
    seed = random.getrandbits(64)
    values = []
    # Using a fast range for sampling
    for i in range(samples):
        values.append(synthesize_multichannel(i, seed))
    
    # Chi-Squared test
    # Divide [0, 1) into 100 bins
    bins = 100
    observed, _ = np.histogram(values, bins=bins, range=(0, 1))
    expected = samples / bins
    chi_sq, p_val = stats.chisquare(observed, f_exp=[expected]*bins)
    
    print(f"Chi-Squared: {chi_sq:.2f}, P-Value: {p_val:.4f}")
    # Success if p-value > 0.01 (not significantly non-uniform)
    success = p_val > 0.01
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def test_collision_resistance(samples=1000000):
    print(f"\n--- Collision Resistance Audit ({samples} samples) ---")
    seed = random.getrandbits(64)
    seen = {}
    collisions = 0
    for i in range(samples):
        # Sample non-contiguous addresses to avoid simple RNS patterns
        addr = (i * 0xBF58476D1CE4E5B9) & MASK64
        val = synthesize_multichannel(addr, seed)
        if val in seen:
            collisions += 1
        else:
            seen[val] = addr
            
    print(f"Collisions: {collisions} / {samples}")
    # Success if collision rate is extremely low
    success = collisions < (samples * 0.0001)
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def test_scaling_latency():
    print(f"\n--- O(1) Scaling Verification ---")
    seed = random.getrandbits(64)
    ctx = GMemContext(seed)
    
    # Scale 1: Small (10 entries)
    for i in range(10):
        ctx.write(i, float(i))
    
    start_time = time.perf_counter()
    for i in range(10000):
        ctx.fetch(i)
    latency_small = (time.perf_counter() - start_time) / 10000
    
    # Scale 2: Large (1M entries)
    print(f"Populating context with 1M entries...")
    for i in range(10000, 1010000):
        ctx.write(i, float(i))
    
    start_time = time.perf_counter()
    for i in range(10000):
        ctx.fetch(i)
    latency_large = (time.perf_counter() - start_time) / 10000
    
    ratio = latency_large / latency_small
    print(f"Latency (Small): {latency_small*1e6:.2f} us")
    print(f"Latency (Large): {latency_large*1e6:.2f} us")
    print(f"Scaling Ratio: {ratio:.2f}x")
    
    # Success if ratio is near 1.0 (some overhead is expected but not linear)
    success = ratio < 2.0 
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def test_memory_footprint():
    print(f"\n--- Memory Footprint Audit ---")
    process = psutil.Process(os.getpid())
    
    # Baseline
    baseline = process.memory_info().rss
    
    # 1M entries in GMem
    print("Writing 1M entries to GMem...")
    seed = random.getrandbits(64)
    ctx = GMemContext(seed)
    for i in range(1000000):
        ctx.write(i, 1.0)
    
    mem_gmem = process.memory_info().rss - baseline
    
    # 1M entries in dict
    print("Writing 1M entries to standard dict...")
    d = {}
    for i in range(1000000):
        d[i] = 1.0
    
    mem_dict = (process.memory_info().rss - baseline) - mem_gmem
    
    print(f"GMem Memory (1M writes): {mem_gmem / 1024 / 1024:.2f} MB")
    print(f"Dict Memory (1M writes): {mem_dict / 1024 / 1024:.2f} MB")
    
    # Success if GMem is roughly comparable or better (it uses Overlay which is a wrapped dict, so should be similar)
    success = mem_gmem < (mem_dict * 1.5)
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM ADVERSARIAL GAUNTLET")
    print("====================================================\n")
    
    results = [
        test_avalanche_effect(),
        test_uniformity(),
        test_collision_resistance(),
        test_scaling_latency(),
        test_memory_footprint()
    ]
    
    if all(results):
        print("\n====================================================")
        print("   GAUNTLET RESULT: FULL PASS")
        print("   Claims of O(1) synthesis, entropy, and sparse efficacy are VALID.")
        print("====================================================")
        sys.exit(0)
    else:
        print("\n====================================================")
        print("   GAUNTLET RESULT: FAIL")
        print("   System failed one or more rigor checks.")
        print("====================================================")
        sys.exit(1)
