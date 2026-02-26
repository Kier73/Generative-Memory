import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext
from gmem.decorator import gmem_cached

def demo_virtual_initialization():
    print("--- PyTorch Demo: Virtual Weight Initialization ---")
    print("Goal: Initialize a large LayerNorm or Weight matrix from the GMem manifold.")
    
    seed = 42
    ctx = GMemContext(seed)
    
    # 1. Define a standard PyTorch Linear layer
    hidden_dim = 1024
    layer = nn.Linear(hidden_dim, hidden_dim)
    
    # 2. Hydrate weights from GMem (deterministically synthesized)
    # We fetch hidden_dim^2 values from the manifold
    print(f"Fetching {hidden_dim**2} weights from GMem...")
    start = time.perf_counter()
    weights_flat = ctx.fetch_bulk(0, hidden_dim**2)
    end = time.perf_counter()
    print(f"Fetch Time: {end - start:.4f}s")
    
    # Convert to tensor and reshape
    gmem_weights = torch.tensor(weights_flat, dtype=torch.float32).view(hidden_dim, hidden_dim)
    
    # Assign to layer weights
    with torch.no_grad():
        layer.weight.copy_(gmem_weights)
        
    print(f"Layer weight mean: {layer.weight.mean().item():.6f}")
    print(f"First 5 weights: {layer.weight[0, :5].tolist()}")
    
    # Verification: create new context with same seed, weights should match
    ctx2 = GMemContext(seed)
    val_check = ctx2.fetch(0)
    print(f"Verification Check: {val_check:.6f} (Matches {layer.weight[0, 0].item():.6f}?)")
    
    return True

def demo_virtual_embedding_overrides():
    print("\n--- PyTorch Demo: Sparse Virtual Embedding Table ---")
    print("Goal: A massive embedding table (1B entries) where only overrides cost RAM.")
    
    vocab_size = 1_000_000_000
    embed_dim = 1  # Simplified for demo
    seed = 123
    ctx = GMemContext(seed)
    
    # Initial state: everything is synthesized
    id_rare = 999_999_999
    val_original = ctx.fetch(id_rare)
    print(f"Rare token {id_rare} base weight: {val_original:.6f}")
    
    # Attachment: AOF for persistence
    aof_path = "embeddings.aof"
    if os.path.exists(aof_path): os.remove(aof_path)
    ctx.persistence_attach(aof_path, high_precision=True)
    
    # Fine-tuning: Override a rare token
    print("Fine-tuning: Overriding weight for rare token...")
    new_weight = 0.5
    ctx.write(id_rare, new_weight)
    
    print(f"Rare token {id_rare} current weight (GMem): {ctx.fetch(id_rare):.6f}")
    
    ctx.persistence_detach()
    
    # In PyTorch, we can wrap this in a custom Embedding module
    class GMemEmbedding(nn.Module):
        def __init__(self, context):
            super().__init__()
            self.ctx = context
            
        def forward(self, indices):
            # Batch fetch from GMem
            # (In a real impl, this would be highly optimized)
            results = [self.ctx.fetch(idx.item()) for idx in indices.flatten()]
            return torch.tensor(results, dtype=torch.float32).view_as(indices)

    embed = GMemEmbedding(ctx)
    input_ids = torch.tensor([0, id_rare, 42])
    output = embed(input_ids)
    print(f"Embed Output for {input_ids.tolist()}: {output.tolist()}")
    
    if os.path.exists(aof_path): os.remove(aof_path)
    return True

@gmem_cached(seed=777, namespace="torch_cache")
def expensive_tensor_op(x_val):
    print(f"  [Running Expensive Torch Op for x={x_val}...]")
    # Simulate heavy compute (e.g., massive matrix power)
    t = torch.tensor([x_val], dtype=torch.float32)
    for _ in range(100):
        t = torch.pow(t, 1.0001)
        torch.sin_(t)
    time.sleep(0.5)
    return t.item()

def demo_persistent_memoization():
    print("\n--- PyTorch Demo: Persistent Activation Memoization ---")
    print("Goal: Use @gmem_cached to avoid re-computing identical activations.")
    
    x = 1.234
    
    start = time.perf_counter()
    r1 = expensive_tensor_op(x)
    t1 = time.perf_counter() - start
    print(f"Run 1: {r1:.6f} (Time: {t1:.2f}s)")
    
    start = time.perf_counter()
    r2 = expensive_tensor_op(x)
    t2 = time.perf_counter() - start
    print(f"Run 2: {r2:.6f} (Time: {t2:.2f}s)")
    
    if t2 < 0.1:
        print("PASS: Activation retrieved from GMem manifold.")
        return True
    else:
        print("FAIL: Memoization failed.")
        return False

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM PYTORCH UTILITY SUITE")
    print("====================================================\n")
    
    res1 = demo_virtual_initialization()
    res2 = demo_virtual_embedding_overrides()
    res3 = demo_persistent_memoization()
    
    if all([res1, res2, res3]):
        print("\nPYTORCH INTEGRATION: SUCCESS")
        sys.exit(0)
    else:
        print("\nPYTORCH INTEGRATION: FAILURE")
        sys.exit(1)
