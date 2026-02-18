import struct
import os

def verify_mount():
    file_path = "G:\\gmem_100tb.raw"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Verifying GVM Mount: {file_path}")
    
    # 1. READ SYNTHETIC DATA
    # Read first 8 floats
    with open(file_path, "rb") as f:
        data = f.read(32)
        floats = struct.unpack("8f", data)
        print(f"  Synthetic (Offset 0): {floats}")

    # 2. WRITE TO GVM (Coherence Test)
    # Write a specific float to offset 1024
    offset = 1024 * 4
    magic_value = 123.456
    print(f"  Writing {magic_value} to offset {offset}...")
    with open(file_path, "r+b") as f:
        f.seek(offset)
        f.write(struct.pack("f", magic_value))
    
    # 3. VERIFY COHERENCE
    with open(file_path, "rb") as f:
        f.seek(offset)
        verify_data = f.read(4)
        verify_val = struct.unpack("f", verify_data)[0]
        print(f"  Readback (Offset {offset}): {verify_val}")
        
        if abs(verify_val - magic_value) < 1e-4:
            print("  [PASS] Coherence Verified through OS layer.")
        else:
            print("  [FAIL] Coherence check failed.")

if __name__ == "__main__":
    verify_mount()
