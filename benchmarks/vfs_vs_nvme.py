import os
import time
import struct
import statistics

# Configuration
GVM_PATH = "G:\\gmem_100tb.raw"
NVME_PATH = "C:\\Users\\kross\\Downloads\\gvm_nvme_benchmark.tmp"
BLOCK_SIZE = 1 * 1024 * 1024  # 1MB blocks
TEST_SIZE_MB = 1024           # 1GB test
NUM_RUNS = 3

def measure_throughput(path, mode="rb", size_mb=TEST_SIZE_MB):
    blocks = (size_mb * 1024 * 1024) // BLOCK_SIZE
    throughputs = []
    
    # Pre-allocate buffer for write tests if needed
    buffer = b'\x00' * BLOCK_SIZE if mode == "wb" else None

    for run in range(NUM_RUNS):
        print(f"  Run {run+1}/{NUM_RUNS} on {path}...", end="\r")
        start_time = time.perf_counter()
        
        try:
            with open(path, mode) as f:
                for _ in range(blocks):
                    if "r" in mode:
                        data = f.read(BLOCK_SIZE)
                        if not data: break
                    else:
                        f.write(buffer)
                f.flush()
                # Ensure write is synced to disk for physical drive
                if os.name == 'nt' and "w" in mode:
                    os.fsync(f.fileno())
        except Exception as e:
            print(f"\n  Error during benchmark: {e}")
            return None
            
        end_time = time.perf_counter()
        duration = end_time - start_time
        mb_per_sec = size_mb / duration
        throughputs.append(mb_per_sec)
        
    avg_throughput = statistics.mean(throughputs)
    return avg_throughput

def run_benchmark():
    print("=== GVM Inductive Substrate vs Physical NVMe Benchmark ===")
    print(f"Test Parameters: {TEST_SIZE_MB}MB sequential, {BLOCK_SIZE//1024}KB blocks\n")

    # 1. NVMe Read
    print("Measuring NVMe Sequential Read (C:)...")
    nvme_read = measure_throughput(NVME_PATH.replace(".tmp", "_src.tmp"), "rb")
    if nvme_read: print(f"  NVMe Read: {nvme_read:.2f} MB/s")

    # 2. GVM Read
    print("Measuring GVM Inductive Read (G:)...")
    gvm_read = measure_throughput(GVM_PATH, "rb")
    if gvm_read: print(f"  GVM Read: {gvm_read:.2f} MB/s")

    # 3. NVMe Write
    print("Measuring NVMe Sequential Write (C:)...")
    nvme_write = measure_throughput(NVME_PATH, "wb")
    if nvme_write: print(f"  NVMe Write: {nvme_write:.2f} MB/s")

    # 4. GVM Write
    print("Measuring GVM Sparse Write (G:)...")
    gvm_write = measure_throughput(GVM_PATH, "r+b") # r+b to avoid truncating 100TB file
    if gvm_write: print(f"  GVM Write: {gvm_write:.2f} MB/s")

    print("\n=== Performance Delta Report ===")
    if gvm_read and nvme_read:
        diff_read = (gvm_read / nvme_read) 
        print(f"Inductive Read Speedup: {diff_read:.2f}x faster than physical hardware")
    
    if gvm_write and nvme_write:
        diff_write = (gvm_write / nvme_write)
        print(f"Sparse Write Efficiency: {diff_write:.2f}x physical throughput")

    # Cleanup NVMe temp files
    if os.path.exists(NVME_PATH): os.remove(NVME_PATH)
    src_tmp = NVME_PATH.replace(".tmp", "_src.tmp")
    if os.path.exists(src_tmp): os.remove(src_tmp)

if __name__ == "__main__":
    # Create a dummy source file on NVMe for read testing
    src_tmp = NVME_PATH.replace(".tmp", "_src.tmp")
    if not os.path.exists(src_tmp):
        print(f"Preparing {TEST_SIZE_MB}MB source file on NVMe...")
        with open(src_tmp, "wb") as f:
            f.write(b'\x00' * (TEST_SIZE_MB * 1024 * 1024))
            
    run_benchmark()
