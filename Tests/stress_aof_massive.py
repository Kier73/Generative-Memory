import os
import sys
import time
import random

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext

def test_aof_massive_replay(count=1_000_000):
    print(f"--- Stress Test: Massive AOF Replay ({count:,} entries, High Precision) ---")
    aof_path = "massive_stress.aof"
    if os.path.exists(aof_path): os.remove(aof_path)
    
    seed = 0xAAFF
    ctx = GMemContext(seed)
    
    print(f"Writing {count:,} high-precision entries...")
    ctx.persistence_attach(aof_path, high_precision=True)
    
    start_time = time.perf_counter()
    for i in range(count):
        # We'll use semi-random addresses to stress the overlay dict
        addr = (i * 0x7FFFFFFFFFFFFFFF) & 0xFFFFFFFFFFFFFFFF
        ctx.write(addr, 1.234567890123456)
    
    write_duration = time.perf_counter() - start_time
    print(f"Write throughput: {count / write_duration:.2f} ops/s")
    
    ctx.persistence_detach()
    
    file_size = os.path.getsize(aof_path)
    print(f"AOF File Size: {file_size / 1024 / 1024:.2f} MB")
    
    # Replay
    print("\nReplaying massive AOF...")
    ctx_new = GMemContext(seed)
    start_time = time.perf_counter()
    ctx_new.persistence_attach(aof_path, high_precision=True)
    replay_duration = time.perf_counter() - start_time
    
    print(f"Replay throughput: {count / replay_duration:.2f} ops/s")
    
    # Integrity check
    print("Verifying final sample...")
    test_addr = ((count-1) * 0x7FFFFFFFFFFFFFFF) & 0xFFFFFFFFFFFFFFFF
    val = ctx_new.fetch(test_addr)
    if abs(val - 1.234567890123456) < 1e-15:
        print("PASS: Data integrity confirmed.")
        success = True
    else:
        print(f"FAILED: Data corruption! Got {val}")
        success = False
        
    ctx_new.persistence_detach()
    if os.path.exists(aof_path): os.remove(aof_path)
    return success

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM HIGH-DIFFICULTY: MASSIVE AOF REPLAY")
    print("====================================================\n")
    
    if test_aof_massive_replay(count=1_000_000):
        sys.exit(0)
    else:
        sys.exit(1)
