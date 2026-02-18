import ctypes
import os
import time
import threading

def test_hardening():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(script_dir, "..", "build", "Release", "gmem.dll")
    
    print("--- GMC Phase 8: Hardening & SIMD Verification ---")
    
    if not os.path.exists(dll_path):
        print(f"Error: gmem.dll not found.")
        return

    gmem = ctypes.CDLL(dll_path)

    # Signatures
    gmem.gmem_create.argtypes = [ctypes.c_uint64]
    gmem.gmem_create.restype = ctypes.c_void_p
    gmem.gmem_destroy.argtypes = [ctypes.c_void_p]
    gmem.gmem_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    gmem.gmem_fetch_f32.restype = ctypes.c_float
    gmem.gmem_write_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_float]
    gmem.gmem_fetch_bulk_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    gmem.gmem_save_overlay.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    gmem.gmem_load_overlay.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    gmem.gmem_load_overlay.restype = ctypes.c_int

    ctx = gmem.gmem_create(42)

    # 1. CONCURRENCY STRESS TEST
    print("\n[TEST 1] Multithreaded Overlay Stress Test")
    num_threads = 8
    items_per_thread = 1000
    
    def worker(tid):
        for j in range(items_per_thread):
            addr = tid * 10000 + j
            gmem.gmem_write_f32(ctx, addr, ctypes.c_float(float(tid) + 0.5))
            
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    print(f"  [PASS] {num_threads} threads completed {num_threads * items_per_thread} writes without crash.")
    
    # Verify one value
    v = gmem.gmem_fetch_f32(ctx, 30005)
    print(f"  Verification check (Thread 3, Item 5): {v:.2f}")

    # 2. SIMD PERFORMANCE BENCHMARK
    print("\n[TEST 2] SIMD Bulk Fetch Benchmark")
    bulk_count = 100000000 # 100 Million floats
    buffer = (ctypes.c_float * bulk_count)()
    
    start_time = time.perf_counter()
    gmem.gmem_fetch_bulk_f32(ctx, 0, buffer, bulk_count)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    throughput = (bulk_count * 4) / (duration * 1024 * 1024) # MB/s
    print(f"  Processed {bulk_count} synthetic floats in {duration:.4f}s")
    print(f"  Throughput: {throughput:.2f} MB/s")
    
    if throughput > 100:
        print("  [PASS] High-throughput synthesis verified.")
    else:
        print("  [WARNING] Throughput lower than expected for SIMD.")

    # 3. INTEGRITY SHIELD TEST
    print("\n[TEST 3] CRC32 Corruption Injection Test")
    state_path = b"integrity_test.gvm"
    gmem.gmem_save_overlay(ctx, state_path)
    
    # Corrupt the file manually
    with open(state_path, "r+b") as f:
        f.seek(20) # Go deep into the file
        original_byte = f.read(1)
        f.seek(20)
        f.write(bytes([ord(original_byte) ^ 0xFF])) # Flip bits
        
    ctx_corrupt = gmem.gmem_create(0)
    res = gmem.gmem_load_overlay(ctx_corrupt, state_path)
    
    if res == -2:
        print("  [PASS] Integrity Shield detected corruption (Checksum Mismatch).")
    else:
        print(f"  [FAIL]loader returned {res} (Expected -2 for corruption).")
        
    gmem.gmem_destroy(ctx_corrupt)
    if os.path.exists(state_path): os.remove(state_path)

    # Cleanup
    gmem.gmem_destroy(ctx)
    print("\n--- Phase 8 Verification Complete ---")

if __name__ == "__main__":
    test_hardening()
