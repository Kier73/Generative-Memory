import sys
import os
import time
import threading
import multiprocessing
import ctypes

sys.path.append(os.path.abspath("python"))
from gvm import GVM

# Global instance for shared test
shared_gvm = None

def worker_write_shared(count):
    # Writes to the GLOBAL shared_gvm instance
    # This tests Lock Contention
    val = 1.0
    for i in range(count):
        # Use a random-ish address to provoke collisions/resizes
        addr = (i * 997) + (id(threading.current_thread()) % 100)
        shared_gvm.write(addr, val)

def worker_write_sharded(count, seed):
    # Creates its OWN local GVM instance
    # This tests Parallel Scalability
    local_gvm = GVM(seed)
    val = 1.0
    for i in range(count):
        addr = i * 997
        local_gvm.write(addr, val)

def run_benchmark():
    print("=== GVM MULTITHREADED BENCHMARK ===")
    
    total_ops = 200000
    thread_counts = [1, 2, 4, 8]
    
    # 1. Shared Context (Lock Contention)
    print("\n--- Scenario A: Shared Context (Single Seed) ---")
    print("Expectation: Performance flatlines due to internal GMEM_LOCK serialization.")
    
    global shared_gvm
    shared_gvm = GVM(seed=0x5555)
    
    for t_count in thread_counts:
        ops_per_thread = total_ops // t_count
        threads = []
        
        start = time.time()
        for i in range(t_count):
            t = threading.Thread(target=worker_write_shared, args=(ops_per_thread,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        end = time.time()
        duration = end - start
        ops_sec = total_ops / duration
        print(f"  Threads: {t_count} | Duration: {duration:.4f}s | Throughput: {ops_sec:,.0f} OPS")

    # 2. Sharded Context (Parallel Tenants)
    print("\n--- Scenario B: Sharded Contexts (Unique Seeds) ---")
    print("Expectation: Performance scales linearly (limited only by CPU/OS).")
    
    for t_count in thread_counts:
        ops_per_thread = total_ops // t_count
        threads = []
        
        start = time.time()
        for i in range(t_count):
            # Pass unique seed per thread
            t = threading.Thread(target=worker_write_sharded, args=(ops_per_thread, 0x1000 + i))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        end = time.time()
        duration = end - start
        ops_sec = total_ops / duration
        print(f"  Threads: {t_count} | Duration: {duration:.4f}s | Throughput: {ops_sec:,.0f} OPS")
        
    print("\n[CONCLUSION]")
    print("To scale GVM, use multiple Tenants (Sharding).")
    print("Accessing a single Tenant is thread-safe but serialized.")

if __name__ == "__main__":
    run_benchmark()
