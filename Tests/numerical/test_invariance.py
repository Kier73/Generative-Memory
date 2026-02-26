import sys
import os

# Add parent directory to path to reach gmem package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gmem.hashing import vl_mask, vl_inverse_mask
from gmem.vrns import CRTEngine, MODULI_16, MODULI_LEGACY, vrns_project, vrns_project_16
from gmem.ntt import verify_law_roundtrip, resolve_value_at, synthesize_product_law, inverse_product_law

def test_hash_invariance():
    print("  [TEST] Hashing Invariance (Bijective Mapping)...")
    seeds = [0, 0x517, 0xCAFEBABE, 0xFFFFFFFFFFFFFFFF]
    addrs = [0, 1, 42, 10**6, 10**12, 10**18, 2**64-1]
    
    for seed in seeds:
        for addr in addrs:
            h = vl_mask(addr, seed)
            rev = vl_inverse_mask(h, seed)
            assert rev == addr, f"Hash inversion failed for addr={addr}, seed={seed}"
    print("    -> PASS: Hashing is perfectly bijective.")

def test_crt_exactness():
    print("  [TEST] CRT Exactness (Arithmetic Recovery)...")
    # Test 16-prime engine
    crt16 = CRTEngine(MODULI_16)
    test_vals = [0, 1, 1234567, 2**64-1, crt16.M - 1]
    for val in test_vals:
        res = crt16.decompose(val)
        rec = crt16.reconstruct(res)
        assert rec == val, f"CRT-16 failed for {val}"
    
    # Test 8-prime engine
    crt8 = CRTEngine(MODULI_LEGACY)
    test_vals_8 = [0, 1, 42, crt8.M - 1]
    for val in test_vals_8:
        res = crt8.decompose(val)
        rec = crt8.reconstruct(res)
        assert rec == val, f"CRT-8 failed for {val}"
    print("    -> PASS: CRT engines recover original integers with zero bit-drift.")

def test_ntt_algebraic_recovery():
    print("  [TEST] NTT Algebraic Law Recovery (Goldilocks Field)...")
    seeds = [(0x517, 0xCAFE), (12345, 67890), (0xDEADBEEF, 0x1337)]
    for sa, sb in seeds:
        ok = verify_law_roundtrip(sa, sb)
        assert ok, f"NTT law recovery failed for {sa}, {sb}"
    print("    -> PASS: Context factorization is algebraically precise.")

if __name__ == "__main__":
    print("=== Numerical Invariance Tests ===")
    try:
        test_hash_invariance()
        test_crt_exactness()
        test_ntt_algebraic_recovery()
        print("\nALL NUMERICAL TESTS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
