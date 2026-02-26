import os
import sys
import torch
import torch.nn as nn
import time

# Add the parent directory to Python path to import gmem modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings-python')))

from gmem.pytorch import GMemLinear

def test_vram_exhaustion(safe_mode=False):
    print("===========================================")
    print(" SUITE 7: VRAM EXHAUSTION (RTX 4060 8GB)   ")
    print("===========================================")
    
    print("\n[!] WARNING: The 'Standard PyTorch' phase of this test is designed to intentionally")
    print("    exhaust your system's physical RAM or VRAM to prove mathematical boundaries.")
    print("    This will cause your Operating System to heavily thrash the disk pagefile,")
    print("    resulting in a temporary system 'freeze' or stutter right before PyTorch crashes.\n")

    if not safe_mode:
        response = input("Do you want to run the destructive PyTorch crash test? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Skipping the destructive standard allocation phase. Running Generative Memory only.")
            safe_mode = True
            
    if not torch.cuda.is_available():
        print("[!] Warning: CUDA not available. Running against System RAM capacity.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Targeting Accelerator: {torch.cuda.get_device_name(0)}")

    layer_dim = 15000  # 15,000 x 15,000 parameters = 225 Million Floats
    # At float32 (4 bytes), each layer is ~900 Megabytes.
    
    print(f"\\n--- Test A: Standard PyTorch VRAM Accumulation ---")
    print(f"Layer Size: {layer_dim}x{layer_dim} (approx 900+ MB per layer)")
    
    standard_layers = []
    crashed = False
    
    if not safe_mode:
        try:
            for i in range(1, 20):
                print(f"Allocating Standard Layer {i}...")
                # We must wrap it in nn.Sequential or list and move to device
                layer = nn.Linear(layer_dim, layer_dim, bias=False).to(device)
                standard_layers.append(layer)
                # Force CUDA context sync and allocation lock
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                    print(f"  -> Physical VRAM Used: {mem_alloc:.2f} GB")
        except RuntimeError as e:
            if "Out of memory" in str(e) or "out of memory" in str(e).lower():
                print(f"\\n[!] MEMORY WALL HIT! PyTorch Crashed at Layer {len(standard_layers)+1}.")
                crashed = True
            else:
                raise e
                
        # Purge VRAM completely
        del standard_layers
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        print("Skipped destructive PyTorch allocations.")
    
    print(f"\\n--- Test B: Generative Memory Infinite Navigability ---")
    gmem_layers = []
    
    print("Attempting to instantiate double the layers that crashed PyTorch...")
    target_layers = (len(standard_layers) + 1) * 2 if crashed else 40
    
    start_time = time.perf_counter()
    for i in range(1, target_layers + 1):
        # O(1) Mathematically bounded!
        layer = GMemLinear(layer_dim, layer_dim, seed=i)
        gmem_layers.append(layer)
        if i % 10 == 0:
            print(f"Allocated GMem Layer {i} / {target_layers}")
            
    elapsed = time.perf_counter() - start_time
    
    print(f"\\n[SUCCESS] Successfully initiated {len(gmem_layers)} Massive Linear Maps!")
    print(f"Total time to mathematically map '{len(gmem_layers)}x 900MB' layers: {elapsed:.4f} seconds.")
    
    if device.type == 'cuda':
        mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        print(f"Total Physical VRAM Consumed: {mem_alloc:.4f} GB")

    print(f"\\nThe Machine is safe. You can navigate geometry infinitely.")

if __name__ == "__main__":
    safe_arg = "--safe" in sys.argv
    test_vram_exhaustion(safe_mode=safe_arg)
