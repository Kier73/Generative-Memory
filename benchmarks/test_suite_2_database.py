import os
import sys
import time
import concurrent.futures
import numpy as np
import random

# Add the parent directory to Python path to import gmem modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings-python')))

from gmem.rs_bridge import FastGMemContext

class DatabaseKVProxyTest:
    def __init__(self, seed: int = 1999):
        self.seed = seed
        self.ctx = FastGMemContext(seed)
        
    def test_concurrent_writes(self, num_threads: int = 16, ops_per_thread: int = 25000):
        """
        Simulates a massive localized concurrent load (e.g., Redis or high-traffic Memcached)
        by hammering the ZMask and DashMap `Physical Overlay` simultaneously from 16 cores.
        """
        print(f"\\n--- Running Concurrent Key-Value Lock-Free Test ({num_threads} Threads, {ops_per_thread:,} Ops/Thread) ---")
        
        # Pre-generate workloads to avoid Python RNG overhead affecting the raw C-FFI benchmark
        print("Pre-generating parallel workloads...")
        workloads = []
        for t in range(num_threads):
            start_addr = t * ops_per_thread
            addrs = np.arange(start_addr, start_addr + ops_per_thread, dtype=np.uint64)
            vals = np.random.rand(ops_per_thread).astype(np.float64)
            workloads.append((addrs, vals))
            
        def worker_write(addrs, vals):
            for i in range(len(addrs)):
                self.ctx.write(int(addrs[i]), float(vals[i]))
                
        def worker_read(addrs, expected_vals):
            errors = 0
            for i in range(len(addrs)):
                val = self.ctx.fetch(int(addrs[i]))
                if abs(val - expected_vals[i]) > 1e-7:
                    errors += 1
            return errors
            
        # 1. Concurrent Write Phase
        print("Initiating Write Phase...")
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_write, w[0], w[1]) for w in workloads]
            concurrent.futures.wait(futures)
            
        write_elapsed = time.perf_counter() - start_time
        total_ops = num_threads * ops_per_thread
        
        print(f"Write Time:         {write_elapsed:.4f}s")
        print(f"Write Ops/sec:      {total_ops / write_elapsed:,.0f} ops/s")
        
        # 2. Concurrent Read Phase (Verification)
        print("\\nInitiating Read Phase...")
        start_time = time.perf_counter()
        
        total_errors = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_read, w[0], w[1]) for w in workloads]
            for f in concurrent.futures.as_completed(futures):
                total_errors += f.result()
                
        read_elapsed = time.perf_counter() - start_time
        
        print(f"Read Time:          {read_elapsed:.4f}s")
        print(f"Read Ops/sec:       {total_ops / read_elapsed:,.0f} ops/s")
        print(f"Data Errors:        {total_errors} (Ideal is 0)")
        
        assert total_errors == 0, f"Database Integrity Failed! {total_errors} concurrent race conditions detected."
        print("[SUCCESS] Lock-Free Atomic ZMask & DashMap Concurrency Verified.")

if __name__ == "__main__":
    print("=========================================================")
    print(" UNIVERSAL TEST SUITE 2: DATABASE & KV PROXY             ")
    print("=========================================================")
    tester = DatabaseKVProxyTest(seed=9999)
    tester.test_concurrent_writes(num_threads=16, ops_per_thread=50000)
    
    print("\\n[SUCCESS] All Database & KV Proxy Suite Tests Passed!")
