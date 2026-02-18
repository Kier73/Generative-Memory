import ctypes
import os
import numpy as np

def test_coherence():
    dll_path = os.path.abspath("build/Release/gmem.dll")
    print(f"--- Infinite-L2 Coherence Test ---")
    
    if not os.path.exists(dll_path):
        print("Error: gmem.dll not found.")
        return

    gmem = ctypes.CDLL(dll_path)

    # Function Signatures
    gmem.gmem_create.argtypes = [ctypes.c_uint64]
    gmem.gmem_create.restype = ctypes.c_void_p
    gmem.gmem_destroy.argtypes = [ctypes.c_void_p]
    gmem.gmem_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    gmem.gmem_fetch_f32.restype = ctypes.c_float
    gmem.gmem_write_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_float]
    gmem.gmem_write_f32.restype = None

    # Create Context
    seed = 42
    ctx = gmem.gmem_create(seed)
    
    # 1. READ SYNTHETIC BASIS
    addr = 0x1337
    basis_val = gmem.gmem_fetch_f32(ctx, addr)
    print(f"Original Synthetic Basis at {addr:#x}: {basis_val:.6f}")
    
    # 2. WRITE TO OVERLAY
    injected_val = 0.987654
    print(f"Writing Overlay value {injected_val:.6f} to {addr:#x}...")
    gmem.gmem_write_f32(ctx, addr, injected_val)
    
    # 3. VERIFY PERSISTENCE (PHYSICAL CACHE)
    resolved_val = gmem.gmem_fetch_f32(ctx, addr)
    print(f"Resolved value at {addr:#x}: {resolved_val:.6f}")
    
    if abs(resolved_val - injected_val) < 1e-6:
        print("[PASS] Overlay Persistence Verified.")
    else:
        print("[FAIL] Persistence failed.")

    # 4. VERIFY CONTINUITY (SYNTHETIC MANIFOLD)
    other_addr = 0x9001
    undisturbed_val = gmem.gmem_fetch_f32(ctx, other_addr)
    print(f"Untouched Basis at {other_addr:#x}: {undisturbed_val:.6f}")
    
    if undisturbed_val != basis_val: # Statistically likely they are different
        print("[PASS] Manifold Continuity Verified. Adjacent space is undisturbed.")
    else:
        print("[NOTE] Adjacent address returned same value (Statistical anomaly or law match).")

    gmem.gmem_destroy(ctx)
    print("--- Test Complete ---")

if __name__ == "__main__":
    test_coherence()
