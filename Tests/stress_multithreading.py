import os
import sys
import threading
import time
import random

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext

def worker_task(ctx, thread_id, num_ops, stats):
    """Worker thread performing random writes and fetches."""
    writes = 0
    fetches = 0
    errors = 0
    
    for _ in range(num_ops):
        try:
            op = random.random()
            addr = random.getrandbits(64)
            
            if op < 0.3: # 30% Writes
                ctx.write(addr, random.random())
                writes += 1
            elif op < 0.8: # 50% Fetches
                ctx.fetch(addr)
                fetches += 1
            else: # 20% Bulk Fetches
                ctx.fetch_bulk(addr & ~0xFF, 16)
                fetches += 1
        except Exception as e:
            errors += 1
            
    stats[thread_id] = (writes, fetches, errors)

def test_multithreading_stress(num_threads=50, ops_per_thread=2000):
    print(f"--- Stress Test: Massive Multi-threading ({num_threads} threads, {ops_per_thread} ops each) ---")
    ctx = GMemContext(seed=0xACE)
    
    threads = []
    stats = {}
    
    start_time = time.perf_counter()
    for i in range(num_threads):
        t = threading.Thread(target=worker_task, args=(ctx, i, ops_per_thread, stats))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    
    total_writes = sum(s[0] for s in stats.values())
    total_fetches = sum(s[1] for s in stats.values())
    total_errors = sum(s[2] for s in stats.values())
    
    duration = end_time - start_time
    throughput = (total_writes + total_fetches) / duration
    
    print(f"Total Ops: {total_writes + total_fetches}")
    print(f"Total Errors: {total_errors}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} ops/s")
    print(f"Final Overlay Count: {ctx.overlay_count}")
    
    success = total_errors == 0 and ctx.overlay_count > 0
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM HIGH-DIFFICULTY: MULTI-THREADING")
    print("====================================================\n")
    
    if test_multithreading_stress():
        sys.exit(0)
    else:
        sys.exit(1)
