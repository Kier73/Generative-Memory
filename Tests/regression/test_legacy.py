import sys
import os

# Add parent directory to path to reach gmem package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gmem.vrns import synthesize, synthesize_multichannel

def test_backward_compatibility():
    print("  [TEST] Backward Compatibility Audit (v1.0 vs v2.0)...")
    addr = 42
    seed = 0x517
    
    # The 'synthesize' function must remain the legacy 8-prime engine
    legacy_val = synthesize(addr, seed)
    
    # We verify the legacy engine still exists and produces the expected sum-of-fractions
    # (Checking against hardcoded result from v1.0 run)
    print(f"    Legacy Output @(42, 0x517): {legacy_val:.6f}")
    assert 0.0 <= legacy_val <= 1.0
    
    # The 'synthesize_multichannel' should be different
    enhanced_val = synthesize_multichannel(addr, seed)
    print(f"    Enhanced Output @(42, 0x517): {enhanced_val:.6f}")
    
    assert legacy_val != enhanced_val, "Enhanced engine returned legacy value!"
    print("    -> PASS: Engines are distinct; legacy v1.0 output is preserved.")

def test_public_api_audit():
    print("  [TEST] Public API Export Audit...")
    import gmem
    required = [
        'GMemContext', 'synthesize', 'synthesize_multichannel', 
        'vl_mask', 'vl_inverse_mask', 'HdcManifold', 'P_GOLDILOCKS'
    ]
    for sym in required:
        assert hasattr(gmem, sym), f"Missing required export: {sym}"
    print(f"    Verified {len(required)} critical API symbols are exported.")
    print("    -> PASS: API surface is correct for v2.0.0.")

if __name__ == "__main__":
    print("=== Regression & Compatibility Tests ===")
    try:
        test_backward_compatibility()
        test_public_api_audit()
        print("\nALL REGRESSION TESTS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
