import ctypes
import math
import numpy as np
from PIL import Image

# Import DLL binding base
from gmem.rs_bridge import gmem_lib

# ----------------------------------------------------------------------
# Expose the FFI Space-Filling Curves (Phase 1)
# ----------------------------------------------------------------------
gmem_lib.gmem_hilbert_encode.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]
gmem_lib.gmem_hilbert_encode.restype = ctypes.c_uint64

gmem_lib.gmem_hilbert_decode.argtypes = [
    ctypes.c_uint64, 
    ctypes.c_uint32, 
    ctypes.POINTER(ctypes.c_uint64), 
    ctypes.POINTER(ctypes.c_uint64)
]
gmem_lib.gmem_hilbert_decode.restype = None


class MediaBridge:
    """
    Phase 1: Generative Memory Media Dimensional Flattening.
    Translates physical 2D image matrices into strict 1D Generative Topologies
    via Hilbert Space-Filling Curves, preserving spatial locality.
    """
    
    @staticmethod
    def encode_coord(i: int, j: int, order: int) -> int:
        """Bijective map: 2D (i, j) -> 1D address."""
        return gmem_lib.gmem_hilbert_encode(i, j, order)

    @staticmethod
    def decode_coord(d: int, order: int) -> tuple:
        """Bijective map: 1D address -> 2D (i, j)."""
        out_i = ctypes.c_uint64(0)
        out_j = ctypes.c_uint64(0)
        gmem_lib.gmem_hilbert_decode(d, order, ctypes.byref(out_i), ctypes.byref(out_j))
        return (out_i.value, out_j.value)

    @staticmethod
    def calculate_order(width: int, height: int) -> int:
        """Calculates the necessary power-of-2 Hilbert order to fit the image."""
        max_dim = max(width, height)
        # Find next power of 2
        p = 1
        order = 0
        while p < max_dim:
            p *= 2
            order += 1
        return order
        
    @staticmethod
    def image_to_topology(image_path: str, grayscale: bool = True) -> tuple:
        """
        Parses an image and flattens it along the Hilbert curve trajectory.
        Returns:
            flat_array (np.ndarray): The 1D topology stream.
            order (int): The structural order needed for Decoding.
            original_size (tuple): (width, height).
        """
        img = Image.open(image_path)
        if grayscale:
            img = img.convert('L')
            
        width, height = img.size
        order = MediaBridge.calculate_order(width, height)
        max_dim = 1 << order
        
        # We process it as a max_dim x max_dim square to satisfy Hilbert requirements
        padded_img = Image.new('L' if grayscale else img.mode, (max_dim, max_dim), color=0)
        padded_img.paste(img, (0, 0))
        
        arr2d = np.array(padded_img)
        flat_array = np.zeros(max_dim * max_dim, dtype=arr2d.dtype)
        
        # High-performance native conversion to 1D based on the C-FFI hook
        # For industrial scale, this loop itself would move to C, but Python suffices for POC.
        for i in range(max_dim):
            for j in range(max_dim):
                addr = MediaBridge.encode_coord(i, j, order)
                flat_array[addr] = arr2d[i, j]
                
        return flat_array, order, (width, height)

    @staticmethod
    def topology_to_image(flat_array: np.ndarray, order: int, original_size: tuple) -> Image.Image:
        """
        Takes a 1D Generator stream and inflates it geometrically back into a 2D image.
        """
        max_dim = 1 << order
        arr2d = np.zeros((max_dim, max_dim), dtype=flat_array.dtype)
        
        for d in range(len(flat_array)):
            i, j = MediaBridge.decode_coord(d, order)
            arr2d[i, j] = flat_array[d]
            
        # Crop back to original
        width, height = original_size
        img = Image.fromarray(arr2d)
        return img.crop((0, 0, width, height))

if __name__ == "__main__":
    print("Testing Media Bridge Space-Filling Bijectivity...")
    order = 8 # 256x256
    
    # Test strict biodecoding over native Rust layer
    s = MediaBridge.encode_coord(100, 150, order)
    a, b = MediaBridge.decode_coord(s, order)
    assert (a, b) == (100, 150), f"Decoder failed. Got ({a}, {b})"
    print("[+] Hilbert C-FFI Topology Passed Bijectivity Checks.")
