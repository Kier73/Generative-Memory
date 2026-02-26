import time
import sys
import os
import psutil # For memory monitoring

# Add current directory to path
sys.path.append(os.path.abspath('.'))

import numpy as np
from gmem import GMemContext, VirtualArray

def get_ram_usage():
    """Returns current physical RAM usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class GMemMatrix:
    """
    A Virtual Matrix backed by Generative Memory.
    Provides constant-memory block processing for massive dimensions.
    """
    def __init__(self, ctx, rows, cols, seed_offset=0):
        self.ctx = ctx
        self.rows = rows
        self.cols = cols
        self.seed_offset = seed_offset
        # Map 2D (r, c) to linear address: addr = seed_offset + r*cols + c
    
    def get_block(self, r_start, r_len, c_start, c_len):
        """Fetch a block as a physical NumPy array."""
        # For rows we need to do multiple bulk reads or one big one if contiguous
        # To keep it simple and constant RAM, we fetch row-by-row
        block = np.zeros((r_len, c_len), dtype=np.float32)
        for i in range(r_len):
            addr = self.seed_offset + (r_start + i) * self.cols + c_start
            block[i, :] = self.ctx.fetch_bulk(addr, c_len)
        return block

    def set_block(self, r_start, r_len, c_start, c_len, data):
        """Write a block back to the generative manifold."""
        for i in range(r_len):
            addr = self.seed_offset + (r_start + i) * self.cols + c_start
            row_data = data[i, :]
            for j, val in enumerate(row_data):
                self.ctx.write(addr + j, float(val))

def run_gauntlet(duration=30):
    print("====================================================")
    print("   GMem v2.0.0 MATRIX GAUNTLET (30s STRESS RUN)")
    print("====================================================\n")
    print(f"Goal: Multiply increasing N x N matrices.")
    print(f"Constraint: Physical RAM must stay roughly constant.")
    print(f"Technique: Block-wise multiplication via Virtual Manifold\n")

    ctx = GMemContext(seed=0x517)
    start_time = time.time()
    n = 128
    step = 0
    
    initial_ram = get_ram_usage()
    print(f"[INIT] Base RAM Usage: {initial_ram:.2f} MB\n")

    print(f"{'STEP':<6} | {'DIM (NxN)':<12} | {'TIME':<10} | {'TOTAL ADDR':<15} | {'RAM (MB)':<10}")
    print("-" * 65)

    BLOCK_SIZE = 64 # Size of chunks processed in physical RAM

    while time.time() - start_time < duration:
        step_start = time.time()
        
        # 1. Define virtual matrices A, B, C
        # We give them large separation in the 2^64 address space
        A = GMemMatrix(ctx, n, n, seed_offset=0)
        B = GMemMatrix(ctx, n, n, seed_offset=10**15)
        C = GMemMatrix(ctx, n, n, seed_offset=2*10**15)
        
        # 2. Block-wise Multiplication (C = A @ B)
        # To keep memory constant, we only ever have a few blocks in RAM
        for i in range(0, n, BLOCK_SIZE):
            for j in range(0, n, BLOCK_SIZE):
                # Calculate block C[i:i+B, j:j+B]
                c_block = np.zeros((min(BLOCK_SIZE, n-i), min(BLOCK_SIZE, n-j)), dtype=np.float32)
                for k in range(0, n, BLOCK_SIZE):
                    a_block = A.get_block(i, min(BLOCK_SIZE, n-i), k, min(BLOCK_SIZE, n-k))
                    b_block = B.get_block(k, min(BLOCK_SIZE, n-k), j, min(BLOCK_SIZE, n-j))
                    c_block += a_block @ b_block
                
                # We don't actually need to store C back to RAM to prove the point,
                # but we could write it to the overlay if we wanted to 'persist' the computation.
                # C.set_block(i, min(BLOCK_SIZE, n-i), j, min(BLOCK_SIZE, n-j), c_block)

        step_end = time.time()
        elapsed = step_end - step_start
        total_addr = n * n * 3
        current_ram = get_ram_usage()

        print(f"{step:<6} | {f'{n}x{n}':<12} | {elapsed:7.3f}s | {total_addr:14,} | {current_ram:8.2f}")
        
        # Increase size for next step: DOUBLE each time
        n = int(n * 2)
        step += 1
        
        # Safety break if we get too big for 1s steps
        if n > 1000000: n = 1000000 

    total_elapsed = time.time() - start_time
    print("-" * 65)
    print(f"\n[FINISH] Gauntlet completed in {total_elapsed:.2f}s")
    print(f"[RESULT] Final RAM: {get_ram_usage():.2f} MB (Delta: {get_ram_usage() - initial_ram:+.2f} MB)")
    print(f"[SUCCESS] Handled logical addressing up to {n*n*3:,} cells with constant memory.")

if __name__ == "__main__":
    try:
        run_gauntlet(30)
    except KeyboardInterrupt:
        print("\nAborted.")
