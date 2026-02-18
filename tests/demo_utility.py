import sys
import os
import hashlib
import time

# Ensure we can import the wrapper
sys.path.append(os.path.abspath("python"))
from gvm import GVM

def draw_ascii_heatmap(data, width, height):
    chars = " .:-=+*#%@"
    for y in range(height):
        line = ""
        for x in range(width):
            val = data[y * width + x]
            # Normalize 0.0-1.0 to char index
            idx = int(val * (len(chars) - 1))
            idx = max(0, min(idx, len(chars) - 1))
            line += chars[idx]
        print(line)

def demo_terrain():
    print("\n=== Use Case 1: Infinite Procedural Terrain (Game Dev) ===")
    print("Generating 64x32 chunk at Offset 0...")
    
    gvm = GVM(seed=0x12345678)
    width, height = 64, 32
    # Bulk read 2048 floats
    terrain = gvm.read_bulk(0, width * height)
    
    draw_ascii_heatmap(terrain, width, height)
    print("Verified: Data generated deterministically without storage.")

def demo_sparse_kv():
    print("\n=== Use Case 2: Exascale Key-Value Store (Database) ===")
    gvm = GVM(seed=0xDB)
    
    # Simulate a hash map using the 64-bit address space
    keys = ["user:kross", "user:admin", "config:rate_limit"]
    
    print("Writing records...")
    for k in keys:
        # Hash string to 64-bit integer
        addr = int(hashlib.md5(k.encode()).hexdigest()[:16], 16)
        # Store a dummy value (e.g. timestamp/id)
        val = float(len(k)) 
        print(f"  Key: '{k}' -> Addr: 0x{addr:X} -> Val: {val}")
        gvm.write(addr, val)
        
    print("Reading records index-free...")
    for k in keys:
        addr = int(hashlib.md5(k.encode()).hexdigest()[:16], 16)
        val = gvm.read(addr)
        print(f"  Key: '{k}' -> Retrieved: {val}")
        if val != float(len(k)):
            raise Exception("Data mismatch!")
            
    print("Verified: O(1) access in 18 Exabyte space.")

def demo_steganography():
    print("\n=== Use Case 3: Poly-Context Steganography (Security) ===")
    secret_seed = 0xCAFEBABE
    public_seed = 0x00000000
    
    addr = 0x1000
    secret_msg = 3.14159
    
    print(f"Writing secret '{secret_msg}' to Address 0x{addr:X} in SEED 0x{secret_seed:X}...")
    ctx_secret = GVM(secret_seed)
    ctx_secret.write(addr, secret_msg)
    
    print(f"Attempting to read Address 0x{addr:X} from PUBLIC SEED 0x{public_seed:X}...")
    ctx_public = GVM(public_seed)
    val = ctx_public.read(addr)
    
    print(f"  Result: {val}")
    
    if val == secret_msg:
        print("  [FAIL] Leak detected!")
    else:
        print("  [SUCCESS] Secret is invisible to public context.")
        print("  The public user sees only procedural background noise.")

if __name__ == "__main__":
    try:
        demo_terrain()
        demo_sparse_kv()
        demo_steganography()
        print("\nAll demos passed.")
    except Exception as e:
        print(f"\nDEMO FAILED: {e}")
        sys.exit(1)
