import os
import sys
import time
import struct
import tempfile
import numpy as np
import pandas as pd

# Add the parent directory to Python path to import gmem modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings-python')))

from gmem.rs_bridge import FastGMemContext

def test_ssd_saturation():
    print("===========================================")
    print(" SUITE 8: SSD SEQUENTIAL WRITE DRAG RACE   ")
    print("===========================================")
    
    # 5 Million Key-Value database inserts
    num_writes = 5000000
    
    print(f"\\n[~] Generating {num_writes:,} random database entries...")
    addresses = np.random.randint(0, 2**48, size=num_writes, dtype=np.int64)
    values = np.random.randn(num_writes)
    
    # Temporary files strictly for testing I/O boundaries
    csv_path = os.path.join(tempfile.gettempdir(), 'standard_database.csv')
    aof_path = os.path.join(tempfile.gettempdir(), 'gmem_mathematics.gvm_delta')
    
    # --- Test A: Standard NoSQL/Pandas Sequential Write (CSV/Parquet simulation) ---
    print("\\n--- Test A: Standard Pandas Disk Flush ---")
    df = pd.DataFrame({'address': addresses, 'value': values})
    
    start_time = time.perf_counter()
    # to_csv is a baseline standard for Python sequential disk logging (simulating slow structured IO)
    df.to_csv(csv_path, index=False)
    python_io_time = time.perf_counter() - start_time
    
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"Time to Flush (Pandas):  {python_io_time:.4f} seconds")
    print(f"Throughput:              {num_writes // python_io_time:,} ops/sec")
    print(f"Disk Space Consumed:     {csv_size:.2f} MB")
    
    # --- Test B: Generative Memory AOF (Mmap Append) ---
    print("\\n--- Test B: Generative Memory Persistence (.gvm_delta) ---")
    ctx = FastGMemContext(seed=1337)
    
    # Attach tracking logs directly to our physical hard drive via mmap
    ctx.persistence_attach(aof_path)
    
    start_time = time.perf_counter()
    
    # This invokes standard atomic caching + Rust Append-Only File `memmap2` logic
    for i in range(num_writes):
        ctx.write(int(addresses[i]), float(values[i]))
        
    gmem_io_time = time.perf_counter() - start_time
    
    aof_size = os.path.getsize(aof_path) / (1024 * 1024)
    print(f"Time to Sync (GMem):     {gmem_io_time:.4f} seconds")
    print(f"Throughput:              {num_writes // gmem_io_time:,} ops/sec")
    print(f"Disk Space Consumed:     {aof_size:.2f} MB")
    
    print("\\n=== CONCLUSION ===")
    if gmem_io_time < python_io_time:
        print(f"Generative Memory Zero-Copy AOF is {python_io_time / gmem_io_time:.1f}x FASTER than dumping structured Python data.")
    else:
        print(f"Standard Python IO is faster by {gmem_io_time / python_io_time:.1f}x (likely due to pure Pandas C-optimised dump vs FFI loop overhead).")
        
    try:
        os.remove(csv_path)
        os.remove(aof_path)
    except Exception:
        pass
    
if __name__ == "__main__":
    test_ssd_saturation()
