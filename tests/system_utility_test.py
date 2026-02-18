import ctypes
import os
import time

def test_system_utility():
    # Correct pathing: script is in tests/, dll is in build/Release/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(script_dir, "..", "generative_memory", "build", "Release", "gmem.dll")
    
    print("--- GMC System Utility Verification (Phase 5) ---")
    
    if not os.path.exists(dll_path):
        # Try local path if being run from different directory
        dll_path = os.path.abspath("build/Release/gmem.dll")
        if not os.path.exists(dll_path):
            print(f"Error: gmem.dll not found at {dll_path}")
            return

    gmem = ctypes.CDLL(dll_path)

    # Function Signatures
    gmem.gmem_create.argtypes = [ctypes.c_uint64]
    gmem.gmem_create.restype = ctypes.c_void_p
    gmem.gmem_destroy.argtypes = [ctypes.c_void_p]
    gmem.gmem_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    gmem.gmem_fetch_f32.restype = ctypes.c_float
    gmem.gmem_write_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_float]
    gmem.gmem_save_overlay.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    gmem.gmem_save_overlay.restype = ctypes.c_int
    gmem.gmem_load_overlay.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    gmem.gmem_load_overlay.restype = ctypes.c_int
    
    gmem.g_malloc.argtypes = [ctypes.c_size_t]
    gmem.g_malloc.restype = ctypes.c_void_p
    gmem.g_free.argtypes = [ctypes.c_void_p]
    gmem.g_get_f32.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    gmem.g_get_f32.restype = ctypes.c_float

    # 1. VERIFY PERSISTENCE
    print("\nSection 1: Overlay Persistence Test")
    seed = 0xAAAA
    state_path = b"overlay_test.bin"
    
    # Create and modify
    ctx1 = gmem.gmem_create(seed)
    gmem.gmem_write_f32(ctx1, 100, ctypes.c_float(0.123))
    gmem.gmem_write_f32(ctx1, 2000, ctypes.c_float(0.456))
    print(f"  Saving state to {state_path.decode()}...")
    gmem.gmem_save_overlay(ctx1, state_path)
    gmem.gmem_destroy(ctx1)

    # Load and verify
    ctx2 = gmem.gmem_create(0) # Seed will be overridden by load
    print(f"  Loading state from {state_path.decode()}...")
    gmem.gmem_load_overlay(ctx2, state_path)
    val1 = gmem.gmem_fetch_f32(ctx2, 100)
    val2 = gmem.gmem_fetch_f32(ctx2, 2000)
    print(f"  Restored values: {val1:.3f}, {val2:.3f}")
    
    if abs(val1 - 0.123) < 1e-4 and abs(val2 - 0.456) < 1e-4:
        print("  [PASS] Persistence Verified.")
    else:
        print("  [FAIL] Persistence Integrity Violation.")
    gmem.gmem_destroy(ctx2)
    if os.path.exists(state_path): os.remove(state_path)

    # 2. VERIFY G_MALLOC (Sparse Allocator)
    print("\nSection 2: Sparse Allocator (g_malloc) Test")
    size_1gb = 1024 * 1024 * 1024
    print(f"  Allocating 1GB Virtual Sparse Buffer...")
    ptr = gmem.g_malloc(size_1gb)
    
    if not ptr:
        print("  [FAIL] g_malloc returned NULL.")
        return
        
    # Check synthetic basis
    idx = 123456
    val_synthetic = gmem.g_get_f32(ptr, idx)
    print(f"  Synthetic value at index {idx}: {val_synthetic:.6f}")
    
    # Free
    gmem.g_free(ptr)
    print("  [PASS] g_malloc/g_free lifecycle complete.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_system_utility()
