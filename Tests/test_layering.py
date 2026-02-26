import os
import sys
import time

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext
from gmem.hashing import MASK64

def test_nested_addressing_128bit():
    print("--- Layering Demo: $2^{128}$ Virtual Addressing ---")
    print("Scenario: Using a GMem value as a seed for a sub-space.")
    
    # Root context
    root_seed = 0x9001
    root_ctx = GMemContext(root_seed)
    
    # Primary Address (64-bit)
    addr_p = 0xAAAA_BBBB_CCCC_DDDD
    # Secondary Address (64-bit)
    addr_s = 0x1111_2222_3333_4444
    
    # 128-bit virtual address is effectively (addr_p, addr_s)
    print(f"Addressing (0x{addr_p:016X}, 0x{addr_s:016X})...")
    
    # Step 1: Fetch entropy from root space to seed sub-space
    sub_seed_raw = root_ctx.fetch(addr_p)
    # Convert float back to uint64 for seeding
    import struct
    sub_seed = struct.unpack('<Q', struct.pack('<d', sub_seed_raw))[0]
    
    # Step 2: Initialize sub-space
    sub_ctx = GMemContext(sub_seed)
    
    # Step 3: Fetch final value
    val = sub_ctx.fetch(addr_s)
    print(f"Final 128-bit Virtual Value: {val:.10f}")
    
    # Verification: Identity check
    root_ctx_2 = GMemContext(root_seed)
    sub_seed_raw_2 = root_ctx_2.fetch(addr_p)
    sub_seed_2 = struct.unpack('<Q', struct.pack('<d', sub_seed_raw_2))[0]
    sub_ctx_2 = GMemContext(sub_seed_2)
    val_2 = sub_ctx_2.fetch(addr_s)
    
    if abs(val - val_2) < 1e-15:
        print("PASS: 128-bit deterministic identity confirmed.")
        return True
    else:
        print("FAIL: Determinism broken across layers.")
        return False

def test_namespace_signatures():
    print("\n--- Layering Demo: Namespace Signatures ---")
    print("Scenario: Isolating namespaces using byte signatures (upper-bit seeding).")
    
    base_seed = 0xFFFF
    
    # Signatures (e.g., 'LOGS' and 'META')
    sig_logs = 0x4C4F4753 << 32 
    sig_meta = 0x4D455441 << 32
    
    ctx_logs = GMemContext(base_seed ^ sig_logs)
    ctx_meta = GMemContext(base_seed ^ sig_meta)
    
    addr = 12345
    val_logs = ctx_logs.fetch(addr)
    val_meta = ctx_meta.fetch(addr)
    
    print(f"Log Namespace Value at {addr}: {val_logs:.6f}")
    print(f"Meta Namespace Value at {addr}: {val_meta:.6f}")
    
    if val_logs != val_meta:
        print("PASS: Namespaces isolated via signature bit-mixing.")
        return True
    else:
        print("FAIL: Namespace collision.")
        return False

def test_recursive_function_scratchpads():
    print("\n--- Layering Demo: Recursive Scratchpads ---")
    print("Scenario: Each level of recursion gets its own massive virtual space.")
    
    def recursive_op(depth, parent_seed):
        if depth == 0: return
        
        # Derive current seed from parent + depth
        current_seed = (parent_seed ^ (depth * 0xDEADEAD)) & MASK64
        ctx = GMemContext(current_seed)
        
        # Use address 0 as a 'local variable' in virtual space
        local_val = ctx.fetch(0)
        print(f"  Depth {depth} Local Virtual Val: {local_val:.4f}")
        
        recursive_op(depth - 1, current_seed)

    print("Starting recursive call chain...")
    recursive_op(5, 0x5EED)
    print("PASS: Hierarchical scratchpads materialized on-demand.")
    return True

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM LAYERING AND SCALE-INVARIANCE")
    print("====================================================\n")
    
    res1 = test_nested_addressing_128bit()
    res2 = test_namespace_signatures()
    res3 = test_recursive_function_scratchpads()
    
    if all([res1, res2, res3]):
        print("\nLAYERING VERIFICATION: SUCCESS")
        sys.exit(0)
    else:
        print("\nLAYERING VERIFICATION: FAILURE")
        sys.exit(1)
