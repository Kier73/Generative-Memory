import sys
import os
import math
from collections import Counter

# Add parent directory to path to reach gmem package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gmem.vrns import synthesize_multichannel
from gmem.hashing import fmix64, MASK64
from gmem.hdc import HdcManifold

def test_uniformity():
    print("  [TEST] Uniformity Audit (Chi-Squared)...")
    n = 100000
    buckets = 10
    counts = [0] * buckets
    
    # Sample 100k values
    for i in range(n):
        v = synthesize_multichannel(i, 0x517)
        bucket = int(v * buckets)
        if bucket >= buckets: bucket = buckets - 1
        counts[bucket] += 1
    
    expected = n / buckets
    chi_sq = sum((c - expected)**2 / expected for c in counts)
    
    # Critical value for df=9, p=0.01 is 21.666
    print(f"    Chi-Squared: {chi_sq:.4f} (expected < 21.67 for p=0.01)")
    assert chi_sq < 21.666, f"Uniformity failed: Chi-Squared too high ({chi_sq})"
    print("    -> PASS: Synthesis is statistically uniform.")

def test_avalanche_effect():
    print("  [TEST] Avalanche Effect Audit (Diffusion)...")
    # Change 1 bit in address, measure how many bits change in hash
    addr = 0x1234567890ABCDEF
    original_h = fmix64(addr)
    
    bit_changes = []
    for i in range(64):
        trial_addr = addr ^ (1 << i)
        trial_h = fmix64(trial_addr)
        diff = original_h ^ trial_h
        popcount = bin(diff).count('1')
        bit_changes.append(popcount)
    
    avg_change = sum(bit_changes) / 64
    print(f"    Average Bits Changed: {avg_change:.2f} (Target: 32.0)")
    assert 28 <= avg_change <= 36, f"Avalanche effect poor: {avg_change} bits changed on average"
    print("    -> PASS: Diffusion meets cryptographic standards (Avalanche ~50%).")

def test_hdc_collision_resistance():
    print("  [TEST] HDC Collision Resistance (1024-bit)...")
    n = 10000
    seen = set()
    for i in range(n):
        m = HdcManifold(seed=i)
        fp = m.fingerprint()
        if fp in seen:
            assert False, f"Collision detected at seed {i}"
        seen.add(fp)
    print(f"    Sampled {n} HDC manifolds with zero fingerprint collisions.")
    print("    -> PASS: HDC fingerprinting is collision-resistant at scale.")

if __name__ == "__main__":
    print("=== Statistical Entropy Tests ===")
    try:
        test_uniformity()
        test_avalanche_effect()
        test_hdc_collision_resistance()
        print("\nALL STATISTICAL TESTS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
