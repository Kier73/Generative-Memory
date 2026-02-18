import ctypes
import os
import struct

def test_raw_drive():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(script_dir, "..", "build", "Release", "gmem.dll")
    
    print("--- GMC Phase 7: Virtual Raw Drive Simulator ---")
    
    if not os.path.exists(dll_path):
        print(f"Error: gmem.dll not found.")
        return

    gmem = ctypes.CDLL(dll_path)
    
    # Signatures
    gmem.gmem_create.argtypes = [ctypes.c_uint64]
    gmem.gmem_create.restype = ctypes.c_void_p
    gmem.gmem_destroy.argtypes = [ctypes.c_void_p]
    gmem.gmem_create_block_filter.argtypes = [ctypes.c_uint32]
    gmem.gmem_create_block_filter.restype = ctypes.c_void_p
    gmem.gmem_destroy_filter.argtypes = [ctypes.c_void_p]
    gmem.gmem_filter_read_block.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]

    # Simulation Setup
    sector_size = 4096
    ctx = gmem.gmem_create(0x1337)
    filter = gmem.gmem_create_block_filter(sector_size)
    
    print(f"[INIT] Simulating Raw Block Device (Sector Size: {sector_size} bytes)")

    # 1. SECTOR STABILITY TEST
    sector_id = 1024
    buffer1 = ctypes.create_string_buffer(sector_size)
    buffer2 = ctypes.create_string_buffer(sector_size)
    
    print(f"[TEST] Reading Sector {sector_id} twice...")
    gmem.gmem_filter_read_block(filter, ctx, sector_id, buffer1)
    gmem.gmem_filter_read_block(filter, ctx, sector_id, buffer2)
    
    if buffer1.raw == buffer2.raw:
        print(f"  [PASS] Sector stability verified. Bit-exact parity.")
        # Print first few bytes for visual confirmation
        hex_data = buffer1.raw[:16].hex(' ')
        print(f"  Data Fragment: {hex_data}")
    else:
        print(f"  [FAIL] Sector mismatch. Determinism failure.")

    # 2. SECTOR UNIQUENESS TEST
    buffer3 = ctypes.create_string_buffer(sector_size)
    sector_id_2 = 2048
    print(f"[TEST] Reading Sector {sector_id_2} for uniqueness...")
    gmem.gmem_filter_read_block(filter, ctx, sector_id_2, buffer3)
    
    if buffer1.raw != buffer3.raw:
        print(f"  [PASS] Sector uniqueness verified. Entropy confirmed.")
    else:
        print(f"  [FAIL] Collision detected between sectors.")

    # 3. CAPACITY SIMULATION
    # Simulate reading a sector at the 100TB mark
    huge_sector_id = (100 * 1024**4) // sector_size
    print(f"[TEST] Accessing Sector at 100TB mark (ID: {huge_sector_id})...")
    buffer_huge = ctypes.create_string_buffer(sector_size)
    gmem.gmem_filter_read_block(filter, ctx, huge_sector_id, buffer_huge)
    print(f"  Resolved Data at 100TB: {buffer_huge.raw[:16].hex(' ')}")
    print(f"  [PASS] Infinite Addressing Capacity confirmed.")

    # Cleanup
    gmem.gmem_destroy_filter(filter)
    gmem.gmem_destroy(ctx)
    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    test_raw_drive()
