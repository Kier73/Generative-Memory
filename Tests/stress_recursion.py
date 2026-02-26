import os
import sys
import time

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext

def test_recursive_resolution(depth=100):
    print(f"--- Stress Test: Recursive Resolution ({depth} layers) ---")
    
    # Create a chain of contexts
    contexts = [GMemContext(seed=i) for i in range(depth + 1)]
    
    # Layer 0 is the base.
    # We'll attach them in a chain: contexts[1] mirrors contexts[0], etc.
    # Wait, GMem resolution order: Overlay -> Morph -> Mirror -> Synthesis.
    # To test recursion, we'll use mirror_attach.
    
    start_time = time.perf_counter()
    for i in range(1, depth + 1):
        contexts[i].mirror_attach(contexts[i-1])
    end_time = time.perf_counter()
    print(f"Chain setup time: {end_time - start_time:.4f}s")
    
    # Test fetch at the end of the chain (Layer 100)
    # This should trigger 100 recursive calls ifLayer 100 has no overlay.
    test_addr = 0xDEADBEEF
    
    print(f"Executing fetch at depth {depth}...")
    start_time = time.perf_counter()
    val = contexts[depth].fetch(test_addr)
    end_time = time.perf_counter()
    
    latency = (end_time - start_time) * 1e6 # us
    print(f"Value: {val}")
    print(f"Fetch latency (100 layers): {latency:.2f} us")
    
    # Verification: value should match base context (seed 0)
    expected_val = contexts[0].fetch(test_addr)
    if abs(val - expected_val) > 1e-15:
        print(f"FAILED: Value mismatch! Expected {expected_val}, got {val}")
        return False
        
    print("PASS: Recursive resolution stable.")
    return True

def test_morph_recursion(depth=10):
    # Morphing is more expensive than mirroring.
    print(f"\n--- Stress Test: Morph Recursion ({depth} layers) ---")
    contexts = [GMemContext(seed=i) for i in range(depth + 1)]
    
    for i in range(1, depth + 1):
        # mode 1 = linear (y = 2x + 0)
        contexts[i].morph_attach(contexts[i-1], mode=1, a=2.0, b=0.0)
        
    test_addr = 42
    base_val = contexts[0].fetch(test_addr)
    expected_val = base_val * (2**depth)
    
    print(f"Executing deep morph fetch...")
    start_time = time.perf_counter()
    val = contexts[depth].fetch(test_addr)
    end_time = time.perf_counter()
    
    latency = (end_time - start_time) * 1e6 # us
    print(f"Base Value: {base_val}")
    print(f"Morphed Value: {val}")
    print(f"Expected Value: {expected_val}")
    print(f"Fetch latency ({depth} layers): {latency:.2f} us")
    
    if abs(val - expected_val) > 1e-9:
        print("FAILED: Morph recursion error.")
        return False
        
    print("PASS: Morph recursion stable.")
    return True

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM HIGH-DIFFICULTY: RECURSIVE RESOLUTION")
    print("====================================================\n")
    
    res1 = test_recursive_resolution(depth=100)
    res2 = test_morph_recursion(depth=10)
    
    if res1 and res2:
        sys.exit(0)
    else:
        sys.exit(1)
