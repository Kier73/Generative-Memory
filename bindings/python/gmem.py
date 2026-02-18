import ctypes
import os
import platform
import time

# Locate the DLL
# Start by checking common locations
_lib_path = ""
_system = platform.system()

if _system == "Windows":
    _lib_name = "libgmem.dll"
    # Search specific paths for convenience in dev/release
    # We are in bindings/python, so we need to go up 2 levels
    _base = os.path.dirname(os.path.abspath(__file__))
    _search_paths = [
        os.path.abspath("."),
        os.path.join(_base, "../../build_test"),
        os.path.join(_base, "../../build/Debug"),
        os.path.join(_base, "../../release/bin"),
        "C:/Windows/System32"
    ]
else:
    _lib_name = "libgmem.so"
    _search_paths = [".", "./lib", "../build"]

for path in _search_paths:
    candidate = os.path.join(path, _lib_name)
    if os.path.exists(candidate):
        _lib_path = candidate
        break

if not _lib_path:
    raise RuntimeError(f"Could not find {_lib_name}. Please ensure it is in the path or build directory.")

# Python 3.8+ on Windows requires adding the DLL directory to the search path
if _system == "Windows" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(_lib_path))
    # Also add MinGW bin if present, just in case of runtime dependencies
    mingw_bin = "C:/LLVM_Polly/mingw64/bin"
    if os.path.exists(mingw_bin):
        os.add_dll_directory(mingw_bin)

# Load Library
_gmem = ctypes.CDLL(_lib_path)

# Define Types
gmem_ctx_t = ctypes.c_void_p

# Function Signatures
_gmem.gmem_create.argtypes = [ctypes.c_uint64]
_gmem.gmem_create.restype = gmem_ctx_t

_gmem.gmem_destroy.argtypes = [gmem_ctx_t]
_gmem.gmem_destroy.restype = None

_gmem.gmem_fetch_f32.argtypes = [gmem_ctx_t, ctypes.c_uint64]
_gmem.gmem_fetch_f32.restype = ctypes.c_float

_gmem.gmem_fetch_bulk_f32.argtypes = [gmem_ctx_t, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
_gmem.gmem_fetch_bulk_f32.restype = None

class GenerativeMemory:
    def __init__(self, seed=0):
        self.ctx = _gmem.gmem_create(seed)
        if not self.ctx:
            raise MemoryError("Failed to create GVM context")
            
    def __del__(self):
        if self.ctx:
            _gmem.gmem_destroy(self.ctx)
            self.ctx = None
            
    def get(self, address: int) -> float:
        """Fetch a single float. High overhead for bulk ops."""
        return _gmem.gmem_fetch_f32(self.ctx, address)
        
    def get_range(self, start_addr: int, count: int):
        """Fetch a range of floats. Zero-copy implementation (Fast)."""
        buffer = (ctypes.c_float * count)()
        _gmem.gmem_fetch_bulk_f32(self.ctx, start_addr, buffer, count)
        # Convert to list or numpy array depending on user need
        # For raw speed, return the ctypes array (memory view)
        return buffer

if __name__ == "__main__":
    print(f"Loaded GVM from: {_lib_path}")
    print("Initializing 1PB Context...")
    mem = GenerativeMemory(0xCAFEBABE)
    
    start_t = time.perf_counter()
    val = mem.get(1024 * 1024 * 1024 * 1024) # 1 TB
    end_t = time.perf_counter()
    
    print(f"Fetch (Single): {val:.6f} in {(end_t - start_t)*1000:.4f} ms")
    
    # Bulk Test
    count = 1000000 # 1 Million floats (4MB)
    print(f"Bulk Fetch ({count} items)...")
    start_t = time.perf_counter()
    data = mem.get_range(0, count)
    end_t = time.perf_counter()
    
    dt = end_t - start_t
    print(f"Fetch (Bulk):   {dt*1000:.4f} ms")
    print(f"Throughput:     {(count * 4 / 1024 / 1024) / dt:.2f} MB/s")
    
    print("Success.")
