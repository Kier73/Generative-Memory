import ctypes
import os
import numpy as np

def test_gmem():
    dll_path = os.path.abspath("build/Release/gmem.dll")
    print(f"Loading GMC Library: {dll_path}")
    
    if not os.path.exists(dll_path):
        print("Error: gmem.dll not found. Please build the project first.")
        return

    gmem = ctypes.CDLL(dll_path)

    # Function Signatures
    gmem.gmem_create.argtypes = [ctypes.c_uint64]
    gmem.gmem_create.restype = ctypes.c_void_p
    gmem.gmem_destroy.argtypes = [ctypes.c_void_p]
    gmem.gmem_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    gmem.gmem_fetch_f32.restype = ctypes.c_float
    gmem.gmem_fetch_bulk_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]

    # Create Context
    seed = 0xDEADC0DE
    ctx = gmem.gmem_create(seed)
    print(f"GMC Context Created with seed: {seed:#x}")

    # Test 1: Determinism
    addr = 1024
    val1 = gmem.gmem_fetch_f32(ctx, addr)
    val2 = gmem.gmem_fetch_f32(ctx, addr)
    print(f"Address {addr}: {val1:.6f}")
    
    if val1 == val2:
        print("[PASS] Deterministic read verified.")
    else:
        print("[FAIL] Deterministic read failed.")

    # Test 2: Bulk Materialization
    count = 1000
    buffer = (ctypes.c_float * count)()
    gmem.gmem_fetch_bulk_f32(ctx, 0, buffer, count)
    
    print(f"Materialized {count} points. First 5: {[round(buffer[i], 4) for i in range(5)]}")
    
    # Test 3: Synthetic Addressing (Large Address)
    large_addr = 0xFFFFFFFFFFFFFFFF // 2
    large_val = gmem.gmem_fetch_f32(ctx, large_addr)
    print(f"High-Address Synthesis ({large_addr}): {large_val:.6f}")

    gmem.gmem_destroy(ctx)
    print("GMC Context Destroyed.")

if __name__ == "__main__":
    test_gmem()
