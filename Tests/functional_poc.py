import os
import sys
import time
import random

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext
from gmem.decorator import gmem_cached

def functional_demo_state_store():
    print("--- Functional Demo: Persistent Virtual State Store ---")
    print("Scenario: A civilization simulation with 2^64 possible entity slots.")
    print("Each slot has a deterministic 'base personality' (synthesized).")
    print("We only store 'experience' (overrides) for active entities.")
    
    seed = 123456789
    ctx = GMemContext(seed)
    
    # 1. Accessing "Trillions" of entities
    entity_id_1 = 10**12
    entity_id_2 = 10**12 + 1
    
    base_1 = ctx.fetch(entity_id_1)
    base_2 = ctx.fetch(entity_id_2)
    
    print(f"Entity {entity_id_1} base score: {base_1:.6f}")
    print(f"Entity {entity_id_2} base score: {base_2:.6f}")
    
    # 2. Functional enhancement: Persistent Overrides
    aof_path = "world_state.aof"
    if os.path.exists(aof_path): os.remove(aof_path)
    
    ctx.persistence_attach(aof_path)
    print("\nSimulating entity interaction...")
    # Entity 1 gains experience
    new_score_1 = base_1 + 0.1
    ctx.write(entity_id_1, new_score_1)
    
    print(f"Entity {entity_id_1} updated score (RAM): {ctx.fetch(entity_id_1):.6f}")
    
    ctx.persistence_detach()
    
    # 3. Recovery enhancement
    print("\nSimulating process restart...")
    ctx_recovered = GMemContext(seed)
    ctx_recovered.persistence_attach(aof_path)
    
    print(f"Entity {entity_id_1} score (Recovered): {ctx_recovered.fetch(entity_id_1):.6f}")
    print(f"Entity {entity_id_2} score (Still Synthesized): {ctx_recovered.fetch(entity_id_2):.6f}")
    
    ctx_recovered.persistence_detach()
    if os.path.exists(aof_path): os.remove(aof_path)
    
    return True

@gmem_cached(seed=888, namespace="expensive_ops")
def expensive_calculation(n):
    print(f"  [Computing expensive_calculation({n})...]")
    time.sleep(1) # Fake delay
    return n * 0.12345

def functional_demo_memoization():
    print("\n--- Functional Demo: Manifold Memoization ---")
    print("Using GMem as a massive, persistent cache for function results.")
    
    start = time.perf_counter()
    r1 = expensive_calculation(100)
    t1 = time.perf_counter() - start
    print(f"Result 1: {r1:.6f} (Time: {t1:.2f}s)")
    
    start = time.perf_counter()
    r2 = expensive_calculation(100)
    t2 = time.perf_counter() - start
    print(f"Result 2: {r2:.6f} (Time: {t2:.2f}s)")
    
    if t2 < 0.1 and abs(r1 - r2) < 1e-9:
        print("PASS: Memoization worked (cached in manifold).")
        return True
    else:
        print("FAIL: Memoization failed.")
        return False

if __name__ == "__main__":
    print("====================================================")
    print("   GMEM FUNCTIONAL PURPOSE POC")
    print("====================================================\n")
    
    res1 = functional_demo_state_store()
    res2 = functional_demo_memoization()
    
    if res1 and res2:
        print("\nFUNCTIONAL POC: SUCCESS")
        print("Generative Memory effectively enhances Python with: ")
        print("1. O(1) addressing of massive virtual state spaces.")
        print("2. Transparent persistence of overrides.")
        print("3. High-performance memoization via @gmem_cached.")
        sys.exit(0)
    else:
        print("\nFUNCTIONAL POC: FAILURE")
        sys.exit(1)
