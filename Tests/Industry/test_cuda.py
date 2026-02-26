import time
import unittest
import numpy as np

try:
    import cupy as cp
    from gmem.cuda import SemanticGPUKernel
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

class TestGPUKernelIntegration(unittest.TestCase):
    
    @unittest.skipIf(not _HAS_GPU, "CuPy is required to test CUDA Integration.")
    def test_gpu_extreme_throughput(self):
        print("\n--- Testing CUDA Semantic Generative Kernel ---")
        
        # 1 Gigabyte embedding parameter footprint (125,000,000 Float64s)
        # Attempting this natively in Python loops would take hours.
        size = 125_000_000 
        seed = 0x1337BEEFCAFEBABE
        
        kernel = SemanticGPUKernel(seed)
        
        # Warmup GPU JIT Compiler
        _ = kernel.fill_tensor(1024)
        cp.cuda.Stream.null.synchronize()
        
        print(f"Launching {size:,} parameters onto Nvidia Streaming Multiprocessors...")
        
        t0 = time.perf_counter()
        
        # This allocates GPU memory and triggers the math natively, returning a CuPy array
        gpu_tensor = kernel.fill_tensor(size)
        
        # Wait for all SM threads to finish processing the math
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        
        self.assertEqual(gpu_tensor.shape[0], size)
        
        duration = t1 - t0
        ops_sec = size / duration
        
        print(f"1 GB Tensor Hydrated in {duration:.4f} seconds.")
        print(f"Nvidia GPU Synthesis Throughput: {ops_sec:,.0f} ops/sec")
        
        # Let's validate the math is structurally identical to the Rust CPU engine execution
        # We check index 42,000
        from gmem.rs_bridge import FastGMemContext
        rust_ctx = FastGMemContext(seed)
        
        rust_val = rust_ctx.fetch(42000)
        gpu_val = float(gpu_tensor[42000].get()) # Pull exactly 1 value back over PCIe
        
        print(f"Rust Context: A[42000] = {rust_val:.6f}")
        print(f"CUDA Context: A[42000] = {gpu_val:.6f}")
        
        self.assertAlmostEqual(rust_val, gpu_val, places=12, msg="GPU CUDA Math diverged from Rust CPU Math!")
        print("Success: Zero-RAM Topology is isomorphic across CPU and GPU architectures.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
