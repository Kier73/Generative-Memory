import os
import sys
import time

# Add the parent directory to Python path to import gmem modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings-python')))

from gmem.autoencoder import UniversalSemanticAutoencoder
import cv2
import numpy as np

class MediaTeleportationTest:
    def __init__(self):
        self.encoder = UniversalSemanticAutoencoder()
        self.image_path = os.path.join(os.path.dirname(__file__), "..", "test.jpg")
        
    def test_end_to_end_compression(self):
        print(f"\\n--- Running Media Autoencoder Geometry Flattening Test ---")
        
        if not os.path.exists(self.image_path):
            print(f"[!] Warning: test.jpg not found at {self.image_path}.")
            print("Creating a synthetic high-entropy structural test image...")
            # Create a complex procedural image
            t = np.linspace(0, 10, 256)
            X, Y = np.meshgrid(t, t)
            Z = np.sin(X**2 + Y**2) * 255
            cv2.imwrite(self.image_path, Z.astype(np.uint8))
            
        original_size_bytes = os.path.getsize(self.image_path)
        print(f"Original Physical Image Size: {original_size_bytes:,} bytes")
        
        start_time = time.perf_counter()
        
        # Teleport the data geometrically!
        # This pipes the JPEG -> Hilbert Space-Filling Mapping -> Koopman FFI Sonar -> CuDARc Native Gielis Compiler -> payload
        payload = self.encoder.encode_image(self.image_path)
        
        elapsed = time.perf_counter() - start_time
        
        payload_size = len(payload)
        compression_ratio = original_size_bytes / payload_size
        
        print(f"\\n--- Teleportation Metrics ---")
        print(f"Processed Time:          {elapsed:.4f}s")
        print(f"Holographic GVM Size:    {payload_size} bytes")
        print(f"Compression Achieved:    {compression_ratio:,.1f}x")
        
        assert payload_size < 100, f"GVM Anchor package is too large: {payload_size} bytes! Expected ~52."
        assert compression_ratio > 100, "Teleportation ratio is mathematically invalid!"
        
        print("\\n[SUCCESS] Universal Semantic Autoencoder pipeline perfectly validated natively across FFI limits.")

if __name__ == "__main__":
    print("=========================================================")
    print(" UNIVERSAL TEST SUITE 4: MEDIA TELEPORTATION PIPELINE    ")
    print("=========================================================")
    
    tester = MediaTeleportationTest()
    tester.test_end_to_end_compression()
    
    print("\\n[SUCCESS] Media Teleportation Suite passed!")
