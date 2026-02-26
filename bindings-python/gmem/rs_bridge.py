import ctypes
import os
import sys

# Load the compiled Rust Dynamic Library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core-rust", "target", "release", "gmem_rs.dll"))

if not os.path.exists(lib_path):
    # Try linux extension if on different os, though system is Windows based on info
    alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core-rust", "target", "release", "libgmem_rs.so"))
    if os.path.exists(alt_path):
        lib_path = alt_path
    else:
        raise FileNotFoundError(f"Generative Memory Rust binding not found at {lib_path}. Did you run `cargo build --release`?")

gmem_lib = ctypes.CDLL(lib_path)

# -------------------------------------
# Define Argument & Return Types (ABI)
# -------------------------------------

# gmem_context_new(seed: u64) -> *mut CGMemContext
gmem_lib.gmem_context_new.argtypes = [ctypes.c_uint64]
gmem_lib.gmem_context_new.restype = ctypes.c_void_p

# gmem_fetch(ctx_ptr: *const CGMemContext, addr: u64) -> f64
gmem_lib.gmem_fetch.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
gmem_lib.gmem_fetch.restype = ctypes.c_double

# gmem_write(ctx_ptr: *mut CGMemContext, addr: u64, value: f64)
gmem_lib.gmem_write.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_double]
gmem_lib.gmem_write.restype = None

# gmem_overlay_count(ctx_ptr: *const CGMemContext) -> usize
gmem_lib.gmem_overlay_count.argtypes = [ctypes.c_void_p]
gmem_lib.gmem_overlay_count.restype = ctypes.c_size_t

# gmem_context_free(ctx_ptr: *mut CGMemContext)
gmem_lib.gmem_context_free.argtypes = [ctypes.c_void_p]
gmem_lib.gmem_context_free.restype = None

# gmem_persistence_attach(ctx_ptr: *mut CGMemContext, path: *const c_char) -> i32
gmem_lib.gmem_persistence_attach.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
gmem_lib.gmem_persistence_attach.restype = ctypes.c_int

# gmem_fill_tensor(ctx_ptr: *const CGMemContext, out_ptr: *mut f64, size: usize, start_addr: u64)
import numpy as np
from numpy.ctypeslib import ndpointer
gmem_lib.gmem_fill_tensor.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_uint64]
gmem_lib.gmem_fill_tensor.restype = None

# -------------------------------------
# Semantic Compiler FFI (Inverse Projection)
# -------------------------------------

import numpy as np
from numpy.ctypeslib import ndpointer

# gmem_anchor_new(a_ptr: *const f64, a_rows: usize, a_cols: usize,
#                 b_ptr: *const f64, b_rows: usize, b_cols: usize,
#                 s: usize) -> *mut AnchorNavigator
gmem_lib.gmem_anchor_new.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_size_t,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t
]
gmem_lib.gmem_anchor_new.restype = ctypes.c_void_p

# gmem_anchor_navigate(nav_ptr: *const AnchorNavigator, i: usize, j: usize) -> f64
gmem_lib.gmem_anchor_navigate.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
gmem_lib.gmem_anchor_navigate.restype = ctypes.c_double

# gmem_anchor_free(nav_ptr: *mut AnchorNavigator)
gmem_lib.gmem_anchor_free.argtypes = [ctypes.c_void_p]
gmem_lib.gmem_anchor_free.restype = None

# -------------------------------------
# Python Object Wrapper
# -------------------------------------
class FastGMemContext:
    """Seamless Rust-backed Generative Memory Context for extreme performance."""
    
    def __init__(self, seed: int):
        self._ptr = gmem_lib.gmem_context_new(seed)
        if not self._ptr:
            raise MemoryError("Failed to allocate Generative Memory Context via FFI in Rust.")
            
    def fetch(self, addr: int) -> float:
        """Fetch value from lock-free virtual space."""
        return gmem_lib.gmem_fetch(self._ptr, addr)
        
    def write(self, addr: int, value: float):
        """Write explicit physical override to the mathematical manifold."""
        gmem_lib.gmem_write(self._ptr, addr, float(value))
        
    def fill_tensor(self, tensor: np.ndarray, start_addr: int = 0):
        """
        Extremely high speed FFI bulk allocation. 
        Takes a pre-allocated numpy float64 array and hydrates it linearly from the manifold
        using the native C-backend, eliminating the python scalar loop completely.
        """
        if tensor.dtype != np.float64:
            raise ValueError("Tensor must be float64 contiguous precision.")
        # NumPy/PyTorch arrays must be flattened before taking size
        flat = tensor.reshape(-1)
        gmem_lib.gmem_fill_tensor(self._ptr, flat, flat.size, ctypes.c_uint64(start_addr))
        
    def persistence_attach(self, filepath: str):
        """Mount physical overlay edits dynamically to disk AOF."""
        encoded_path = filepath.encode('utf-8')
        success = gmem_lib.gmem_persistence_attach(self._ptr, encoded_path)
        if success == 0:
            raise IOError(f"Failed to attach AOF persistence at path: {filepath}")
        
    @property
    def overlay_count(self) -> int:
        """Count of explicitly allocated pages/overlay entries."""
        return gmem_lib.gmem_overlay_count(self._ptr)
        
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            gmem_lib.gmem_context_free(self._ptr)
            self._ptr = None

    # Magic Methods (Dictionary like mapped addressing)
    def __getitem__(self, addr: int) -> float:
        return self.fetch(addr)
        
    def __setitem__(self, addr: int, value: float):
        self.write(addr, value)

class SemanticCompiler:
    """
    Inverse Matrix Projection (The Semantic Compiler).
    
    Compresses massive external datasets (like NumPy matrices) into a 
    pure mathematical topology bound by pseudo-inverses ($W^{-1}$).
    """
    def __init__(self, target_matrix: np.ndarray, basis_matrix: np.ndarray, anchor_size: int = 16):
        # Ensure 64-bit precision floats for Rust
        self.target = np.ascontiguousarray(target_matrix, dtype=np.float64)
        self.basis = np.ascontiguousarray(basis_matrix, dtype=np.float64)
        self.anchor_size = anchor_size
        
        a_rows, a_cols = self.target.shape
        b_rows, b_cols = self.basis.shape
        
        self._ptr = gmem_lib.gmem_anchor_new(
            self.target, a_rows, a_cols,
            self.basis, b_rows, b_cols,
            self.anchor_size
        )
        if not self._ptr:
            raise MemoryError("Failed to allocate Semantic Compiler via FFI in Rust.")

    def fetch(self, row: int, col: int) -> float:
        """Dynamically retrieve the compressed data point without storing it."""
        return gmem_lib.gmem_anchor_navigate(self._ptr, row, col)

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            gmem_lib.gmem_anchor_free(self._ptr)
            self._ptr = None
