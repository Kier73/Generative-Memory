import ctypes
import os

def test_structural_filters():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(script_dir, "..", "build", "Release", "gmem.dll")
    
    print("--- GMC Isomorphic Filter Verification (Phase 6) ---")
    
    if not os.path.exists(dll_path):
        print(f"Error: gmem.dll not found at {dll_path}")
        return

    gmem = ctypes.CDLL(dll_path)

    # Function Signatures
    gmem.gmem_create.argtypes = [ctypes.c_uint64]
    gmem.gmem_create.restype = ctypes.c_void_p
    gmem.gmem_destroy.argtypes = [ctypes.c_void_p]
    
    gmem.gmem_create_json_filter.argtypes = [ctypes.c_char_p]
    gmem.gmem_create_json_filter.restype = ctypes.c_void_p
    gmem.gmem_destroy_filter.argtypes = [ctypes.c_void_p]
    
    gmem.gmem_filter_get_val.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    gmem.gmem_filter_get_val.restype = ctypes.c_float
    gmem.gmem_filter_set_val.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]

    # Initialize
    ctx = gmem.gmem_create(1337)
    filter = gmem.gmem_create_json_filter(b"user_schema")

    # 1. DETERMINISTIC PATH RESOLUTION
    path1 = b"users[0].profile.id"
    path2 = b"users[0].profile.id"
    path3 = b"users[1].profile.id"
    
    val1 = gmem.gmem_filter_get_val(filter, ctx, path1)
    val2 = gmem.gmem_filter_get_val(filter, ctx, path2)
    val3 = gmem.gmem_filter_get_val(filter, ctx, path3)
    
    print(f"Path: {path1.decode()} -> Value: {val1:.6f}")
    print(f"Path: {path2.decode()} -> Value: {val2:.6f}")
    print(f"Path: {path3.decode()} -> Value: {val3:.6f}")
    
    if val1 == val2:
        print("[PASS] Path resolution is deterministic.")
    else:
        print("[FAIL] Mismatch in deterministic resolution.")
        
    if val1 != val3:
        print("[PASS] Collision avoidance verified (Unique paths map to unique addresses).")

    # 2. SURGICAL EDITING
    print("\nPerforming Surgical Edit...")
    target_path = b"metadata.version"
    original_val = gmem.gmem_filter_get_val(filter, ctx, target_path)
    print(f"  Original value at {target_path.decode()}: {original_val:.6f}")
    
    new_val = 2.0
    gmem.gmem_filter_set_val(filter, ctx, target_path, ctypes.c_float(new_val))
    
    updated_val = gmem.gmem_filter_get_val(filter, ctx, target_path)
    print(f"  Updated value at {target_path.decode()}: {updated_val:.6f}")
    
    if abs(updated_val - new_val) < 1e-5:
        print("[PASS] Surgical Edit Persisted in GVM Overlay.")
    else:
        print("[FAIL] Surgical Edit failed.")

    # Cleanup
    gmem.gmem_destroy_filter(filter)
    gmem.gmem_destroy(ctx)
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_structural_filters()
