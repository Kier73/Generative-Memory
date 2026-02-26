import os
import ctypes
import numpy as np

# Prefer cupy for seamless GPU ndarray tracking, fallback to CPU simulation if no GPU
try:
    import cupy as cp
    _HAS_GPU = True
except ImportError:
    print("Warning: CuPy not found. CUDA Integration unavailable.")
    _HAS_GPU = False

class SemanticGPUKernel:
    """
    Directly compiles the gmem_kernel.cu file into raw PTX instructions and launches
    it over multi-threaded streaming multiprocessors (SM).
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._compiled_kernel = None
        
        if _HAS_GPU:
            kernel_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "Generative_Memory", "gmem_kernel.cu"
            ))
            
            with open(kernel_path, 'r') as f:
                cuda_code = f.read()
                
            # Compile JIT directly onto Nvidia hardware
            self._compiled_kernel = cp.RawKernel(cuda_code, 'gmem_hydrate_tensor')
            
    def fill_tensor(self, tensor_size: int, start_addr: int = 0):
        """
        Dynamically provisions a GPU memory buffer and triggers thousands of CUDA 
        threads simultaneously to mathematically synthesize the topological values.
        """
        if not _HAS_GPU:
            raise RuntimeError("Cupy is required to execute Generative Memory GPU Kernels.")
            
        # 1. Allocate raw device memory
        gpu_buffer = cp.empty(tensor_size, dtype=cp.float64)
        
        # 2. Configure 1D Block / Grid Thread Topology
        threads_per_block = 256
        blocks_per_grid = (tensor_size + threads_per_block - 1) // threads_per_block
        
        # 3. Launch kernel on the device
        # Arguments must match extern "C" gmem_hydrate_tensor signature:
        # (double* out_tensor, uint64_t size, uint64_t start_addr, uint64_t seed)
        self._compiled_kernel(
            (blocks_per_grid,), (threads_per_block,), 
            (gpu_buffer, cp.uint64(tensor_size), cp.uint64(start_addr), cp.uint64(self.seed))
        )
        
        # Returns the CuPy device array natively (0 bytes transferred over PCIe!)
        return gpu_buffer
