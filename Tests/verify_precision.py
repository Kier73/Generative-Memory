import os
import sys
import random

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext

def test_high_precision_roundtrip():
    print("--- Testing High Precision Persistence Round-trip ---")
    aof_path = "high_precision.aof"
    if os.path.exists(aof_path): os.remove(aof_path)
    
    seed = 999
    ctx = GMemContext(seed)
    
    # Use float64 precision
    print("Attaching high-precision AOF...")
    ctx.persistence_attach(aof_path, high_precision=True)
    
    # A high-precision value that float32 would truncate
    # f32 has ~7 decimal digits, f64 has ~15-17.
    pi_high = 3.14159265358979323846
    addr = 12345
    
    print(f"Writing high-precision value: {pi_high}")
    ctx.write(addr, pi_high)
    
    ctx.persistence_detach()
    
    # Verify file size (8 + 8 = 16 bytes per entry)
    file_size = os.path.getsize(aof_path)
    print(f"AOF File Size: {file_size} bytes (Expected 16 for 1 entry)")
    if file_size != 16:
        print("FAILED: File size mismatch. Expected 16 bytes.")
        return False

    # Recover
    print("Recovering high-precision state...")
    ctx_new = GMemContext(seed)
    ctx_new.persistence_attach(aof_path, high_precision=True)
    
    val_recovered = ctx_new.fetch(addr)
    print(f"Recovered value: {val_recovered}")
    
    diff = abs(val_recovered - pi_high)
    print(f"Difference: {diff}")
    
    # float32 drift would be around 1e-7. float64 should be much closer.
    if diff < 1e-15:
        print("PASS: High-precision recovery exact.")
        success = True
    else:
        print("FAILED: High-precision recovery drift found.")
        success = False
        
    ctx_new.persistence_detach()
    if os.path.exists(aof_path): os.remove(aof_path)
    return success

if __name__ == "__main__":
    if test_high_precision_roundtrip():
        print("\nPRECISION FIX VERIFIED.")
        sys.exit(0)
    else:
        print("\nPRECISION FIX FAILED.")
        sys.exit(1)
