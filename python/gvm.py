import ctypes
import os
import sys

# Load the DLL
# Try to find it in the current directory or build directory
_paths = [
    "./gmem.dll",
    "./build/gmem.dll",
    "./build/Release/gmem.dll",
    "../build/gmem.dll"
]
_lib = None
for p in _paths:
    if os.path.exists(p):
        try:
            _lib = ctypes.CDLL(p)
            break
        except Exception as e:
            print(f"Failed to load {p}: {e}")

if not _lib:
    raise RuntimeError("Could not find gmem.dll. Please build the project.")

# Type Definitions
c_float_p = ctypes.POINTER(ctypes.c_float)
c_void_p = ctypes.c_void_p
c_uint64 = ctypes.c_uint64
c_size_t = ctypes.c_size_t

# struct gmem_context (Opaque)
class GMemContext(ctypes.Structure):
    pass
GMemContextP = ctypes.POINTER(GMemContext)

# Function Signatures
_lib.gmem_create.argtypes = [c_uint64]
_lib.gmem_create.restype = GMemContextP

_lib.gmem_destroy.argtypes = [GMemContextP]
_lib.gmem_destroy.restype = None

_lib.gmem_fetch_f32.argtypes = [GMemContextP, c_uint64]
_lib.gmem_fetch_f32.restype = ctypes.c_float

_lib.gmem_write_f32.argtypes = [GMemContextP, c_uint64, ctypes.c_float]
_lib.gmem_write_f32.restype = None

_lib.gmem_fetch_bulk_f32.argtypes = [GMemContextP, c_uint64, c_float_p, c_size_t]
_lib.gmem_fetch_bulk_f32.restype = None

# Python API Class
class GVM:
    _lib = _lib # Expose for raw access
    def __init__(self, seed=0x1337):
        self.ctx = _lib.gmem_create(seed)
        if not self.ctx:
            raise MemoryError("Failed to create GVM context")
            
    def __del__(self):
        if self.ctx:
            _lib.gmem_destroy(self.ctx)
            
    def read(self, addr):
        return _lib.gmem_fetch_f32(self.ctx, addr)
        
    def write(self, addr, value):
        _lib.gmem_write_f32(self.ctx, addr, value)
        
    def read_bulk(self, start_addr, count):
        buffer = (ctypes.c_float * count)()
        _lib.gmem_fetch_bulk_f32(self.ctx, start_addr, buffer, count)
        return list(buffer)

if __name__ == "__main__":
    print("GVM Python Wrapper")
    gvm = GVM(0xDEADBEEF)
    print("Reading Addr 0:", gvm.read(0))
    print("Writing 42.0 to Addr 0...")
    gvm.write(0, 42.0)
    print("Reading Addr 0:", gvm.read(0))
