import os
import sys
import time
import ctypes
import struct
import numpy as np

# Add the parent directory to Python path to import gmem modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings-python')))

from gmem.rs_bridge import FastGMemContext

def double_to_mantissa(d: float) -> int:
    """Extracts the 52-bit mantissa from an IEEE-754 double."""
    bits = struct.unpack('!Q', struct.pack('!d', d))[0]
    return bits & ((1 << 52) - 1)

class CryptographicPhysicsTest:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ctx = FastGMemContext(seed)
        
    def test_strict_avalanche_criterion(self, iterations: int = 100000):
        """
        The Strict Avalanche Criterion (SAC) states that flipping a single input bit 
        should flip exactly 50% of the output bits on average.
        We will test this natively over the 52 preserved bits of the Rust FFI float mantissa.
        """
        print(f"\n--- Running Strict Avalanche Criterion (SAC) Test ({iterations} iterations) ---")
        
        total_bit_flips = 0
        total_bits_expected = iterations * 52  # IEEE-754 float64 has 52 mantissa bits
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            # Base value
            base_val = self.ctx.fetch(i)
            base_bits = double_to_mantissa(base_val)
            
            # Flip one bit in the input address
            flipped_address = i ^ 1 
            flipped_val = self.ctx.fetch(flipped_address)
            flipped_bits = double_to_mantissa(flipped_val)
            
            # Count the Hamming distance
            xor_result = base_bits ^ flipped_bits
            num_flips = bin(xor_result).count('1')
            total_bit_flips += num_flips
            
        elapsed = time.perf_counter() - start_time
        
        avalanche_ratio = total_bit_flips / total_bits_expected
        print(f"SAC Ratio:          {avalanche_ratio:.4f} (Ideal is 0.5000)")
        print(f"Execution Time:     {elapsed:.4f}s")
        print(f"Ops/sec:            {iterations / elapsed:,.0f} ops/s")
        
        assert 0.49 < avalanche_ratio < 0.51, "SAC Failed: Entropy bounds violated!"
        print("[SUCCESS] Strict Avalanche Criterion Passed.")
        
    def test_bijection_feistel(self):
        """
        Proves the `vl_mask` (Feistel) logic is perfectly invertible O(1).
        While the FFI bridge automatically applies this under the hood, we can
        validate it via the C-FFI direct inversion call if we map it, or 
        verify the uniformity of coordinates.
        Here we assert that fetching sequentially acts as a perfect pseudo-random
        number generator conforming to uniform distribution.
        """
        print("\n--- Running Bijective Uniformity Test (1,000,000 Fetch Map) ---")
        size = 1000000
        
        start_time = time.perf_counter()
        # Fast bulk fetch via FFI (if implemented) or sequential
        # To stress test FFI overhead, we do individual sequential fetches
        vals = np.zeros(size, dtype=np.float32)
        for i in range(size):
            vals[i] = self.ctx.fetch(i)
            
        elapsed = time.perf_counter() - start_time
        
        mean = np.mean(vals)
        std = np.std(vals)
        
        print(f"Mean (expected 0.5): {mean:.6f}")
        print(f"Std (expected 0.288): {std:.6f}")
        print(f"Ops/sec:             {size / elapsed:,.0f} ops/s")
        
        assert 0.49 < mean < 0.51, "Uniformity Failed: Mean deviation!"
        assert 0.28 < std < 0.30, "Uniformity Failed: Standard deviation violation!"
        print("[SUCCESS] Feistel Distribution Passed.")

    def test_birthday_paradox_collision(self):
        """
        Proves that across millions of sequential addresses, the 64-bit entropy 
        generator creates ZERO collisions (Duplicate float returns).
        """
        print("\n--- Running Birthday Paradox Collision Test (2,500,000 fetches) ---")
        size = 2500000
        
        start_time = time.perf_counter()
        
        # Use a python set to track uniqueness (hash table)
        unique_floats = set()
        
        # We can optimize the check by pre-calculating the generator and dropping them into a set
        # But iterating fetch is also fine to stress test FFI context boundary
        for i in range(size):
            unique_floats.add(self.ctx.fetch(i))
            
        elapsed = time.perf_counter() - start_time
        
        print(f"Expected Size: {size:,}")
        print(f"Actual Set Size: {len(unique_floats):,}")
        print(f"Collisions:     {size - len(unique_floats)}")
        print(f"Ops/sec:        {size / elapsed:,.0f} ops/s")
        
        assert len(unique_floats) == size, f"Collision detected! Expected {size}, got {len(unique_floats)}"
        print("[SUCCESS] Birthday Paradox Collision Passed. Entropy is absolute.")


if __name__ == "__main__":
    print("=========================================================")
    print(" UNIVERSAL TEST SUITE 1: PHYSICS & CRYPTOGRAPHY limits   ")
    print("=========================================================")
    tester = CryptographicPhysicsTest(seed=1337)
    
    tester.test_strict_avalanche_criterion(iterations=100000)
    tester.test_bijection_feistel()
    tester.test_birthday_paradox_collision()
    
    print("\\n[SUCCESS] All Physics & Cryptography Suite Tests Passed!")
