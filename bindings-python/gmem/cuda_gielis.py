# gmem/cuda_gielis.py
import numpy as np
import math

try:
    import cupy as cp
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

class GielisGPUCompiler:
    """
    Phase 8G: Semantic Autoencoder CUDA Gielis GPU Compiler.
    Replaces the PyTorch SuperShape CPU meshgrid projector with a blazingly fast 
    VRAM native kernel computing Lattice Locks over 80,000 Ghost Topologies simultaneously.
    """
    def __init__(self):
        self.active = _HAS_GPU
        if not self.active:
            print("  [WARN] CuPy not found. Gielis Lattice Lock GPU Compiler disabled. Returning empty anchors.")
            return
            
        with open("Generative_Memory/gielis_kernel.cu", "r") as f:
            kernel_code = f.read()
            
        # JIT Compile the PTX kernel using Nvidia NVrtc
        self.module = cp.RawModule(code=kernel_code)
        self.gielis_lattice_lock = self.module.get_function("gielis_lattice_lock")

    def search(self, target_signal: np.ndarray, freq: float) -> tuple:
        """
        Executes a mass-parallel geometric search.
        Takes a 1D scalar vector (the flattened video/image or timeseries) and 
        returns the optimal 4 Gielis Anchors that reconstruct its manifold shape.
        """
        if not self.active:
            return (0.0, 0.0, 0.0, 0.0)

        L = target_signal.shape[0]
        
        # Rigorous normalization
        t_mean = np.mean(target_signal)
        t_std = np.std(target_signal) + 1e-6
        target_norm = (target_signal - t_mean) / t_std
        
        # Calculate raw phi angles
        t_arr = np.linspace(-1, 1, L, dtype=np.float32)
        phi_arr = 2 * math.pi * freq * t_arr
        
        # Phase 1: The Macro-Net Search Grid
        m_vals = np.arange(1, 11, 1, dtype=np.float32)                   # 10 
        n1_vals = np.logspace(-1.0, 1.0, 20, dtype=np.float32)             # 20
        n2_vals = np.logspace(-1.0, 1.0, 20, dtype=np.float32)             # 20
        n3_vals = np.logspace(-1.0, 1.0, 20, dtype=np.float32)             # 20
        
        # 10 * 20 * 20 * 20 = 80,000 total configurations
        num_topologies = len(m_vals) * len(n1_vals) * len(n2_vals) * len(n3_vals)
        
        # We need contiguous 1D parameter arrays for the GPU kernel
        # np.meshgrid does this cleanly
        M_grid, N1_grid, N2_grid, N3_grid = np.meshgrid(m_vals, n1_vals, n2_vals, n3_vals, indexing='ij')
        
        M_flat = M_grid.ravel()
        N1_flat = N1_grid.ravel()
        N2_flat = N2_grid.ravel()
        N3_flat = N3_grid.ravel()
        
        # Move pointers to VRAM
        d_target = cp.asarray(target_norm, dtype=cp.float32)
        d_phi = cp.asarray(phi_arr, dtype=cp.float32)
        d_m = cp.asarray(M_flat, dtype=cp.float32)
        d_n1 = cp.asarray(N1_flat, dtype=cp.float32)
        d_n2 = cp.asarray(N2_flat, dtype=cp.float32)
        d_n3 = cp.asarray(N3_flat, dtype=cp.float32)
        d_mse_out = cp.zeros(num_topologies, dtype=cp.float32)
        
        # Launch limits
        threads_per_block = 256
        blocks_per_grid = (num_topologies + threads_per_block - 1) // threads_per_block
        
        # Fire 80,000 SM compute threads simultaneously
        self.gielis_lattice_lock(
            (blocks_per_grid,), (threads_per_block,),
            (d_target, d_phi, d_m, d_n1, d_n2, d_n3, d_mse_out, cp.int32(L), cp.int32(num_topologies))
        )
        
        cp.cuda.Stream.null.synchronize()
        
        # Pull minimum MSE back to CPU
        mse_cpu = d_mse_out.get()
        best_idx = np.argmin(mse_cpu)
        
        return float(M_flat[best_idx]), float(N1_flat[best_idx]), float(N2_flat[best_idx]), float(N3_flat[best_idx])
