import os
import sys
import time

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.ntt import multiply_mod, add_mod, verify_law_roundtrip
from gmem.hdc import HdcManifold, HDC_DIM

def test_ntt_algebraic_integrity(samples=1000):
    print(f"--- Stress Test: NTT Algebraic Integrity ({samples} laws) ---")
    success_count = 0
    # Start from 1 to avoid degenerate mod_inverse(0) case
    for i in range(1, samples + 1):
        if verify_law_roundtrip(i, i + 100):
            success_count += 1
            
    print(f"NTT Roundtrips: {success_count}/{samples}")
    success = success_count == samples
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

def test_hdc_manifold_stability():
    print("\n--- Stress Test: HDC Manifold Stability ---")
    # HDC Dim is 1024-bit. We test bind/bundle identity.
    # (A * B) * B' approx A
    manifold_a = HdcManifold(seed=42, label="A")
    manifold_b = HdcManifold(seed=100, label="B")
    
    # Bind (XOR)
    bound = manifold_a.bind(manifold_b)
    
    # Unbind (XOR is its own inverse)
    recovered_a = bound.bind(manifold_b)
    
    sim = manifold_a.similarity(recovered_a)
    print(f"Cosine Similarity (A vs recovered A): {sim:.6f}")
    
    # Bundling multiple vectors
    vectors = [HdcManifold(seed=i) for i in range(1000, 1030)] # 30 vectors
    head = vectors[0]
    bundled = head.bundle(vectors[1:])
    
    # Verify each original is "present" in bundle
    all_present = True
    for v in vectors:
        sim_v = bundled.similarity(v)
        if sim_v < 0.05: # High volume bundling reduces individual similarities
             print(f"  [WEAK] Low similarity for vector in bundle: {sim_v:.4f}")
             all_present = False
             
    success = sim > 0.99 and all_present
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM HIGH-DIFFICULTY: ALGEBRAIC INTEGRITY")
    print("====================================================\n")
    
    res1 = test_ntt_algebraic_integrity()
    res2 = test_hdc_manifold_stability()
    
    if res1 and res2:
        sys.exit(0)
    else:
        sys.exit(1)
