import os
import sys
import time

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gmem.core import GMemContext
from gmem.rs_bridge import FastGMemContext

def time_it(func, iters, *args):
    start = time.perf_counter()
    for _ in range(iters):
        func(*args)
    return time.perf_counter() - start

def benchmark_bridge_throughput():
    print("====================================================")
    print("   GMEM PYTHON VS RUST (FFI) BRIDGE BENCHMARK")
    print("====================================================\n")
    
    iters = 1_000_000
    addr = 0x1337BEEFCAFEBABE
    seed = 0x9E3779B97F4A7C15
    
    print(f"Instantiating Contexts and Warming Up System...")
    ctx_py = GMemContext(seed)
    ctx_rs = FastGMemContext(seed)
    
    # Pre-populate caches/overlays to stress the lookup vs mathematical paths
    ctx_py.write(addr, 0.42)
    ctx_rs.write(addr, 0.42)
    
    print(f"\n--- GMemContext Fetch Throughput ({iters:,} iterations) ---")
    
    # 1. Pure Python Context
    t_py_clean = time_it(ctx_py.fetch, iters, addr + 1)
    print(f"Python Context (Clean Math): {t_py_clean:.4f}s ({iters/t_py_clean:,.0f} ops/sec)")
    
    t_py_dirty = time_it(ctx_py.fetch, iters, addr)
    print(f"Python Context (Dirty Overlay): {t_py_dirty:.4f}s ({iters/t_py_dirty:,.0f} ops/sec)")

    print("-" * 50)
    
    # 2. Rust FFI Context
    t_rs_clean = time_it(ctx_rs.fetch, iters, addr + 1)
    print(f"Rust FFI Context (Clean Math): {t_rs_clean:.4f}s ({iters/t_rs_clean:,.0f} ops/sec)")
    
    t_rs_dirty = time_it(ctx_rs.fetch, iters, addr)
    print(f"Rust FFI Context (Dirty Overlay): {t_rs_dirty:.4f}s ({iters/t_rs_dirty:,.0f} ops/sec)")

    print("=" * 50)
    print(f"Math Routing Speedup (Rust vs Py): {t_py_clean / t_rs_clean:.2f}x")
    print(f"Overlay Routing Speedup (Rust vs Py): {t_py_dirty / t_rs_dirty:.2f}x")
    from gmem.vrns import synthesize_multichannel
    # Comparing Rust Context lookup speed vs direct Python vRNS (without Context rules routing)
    t_py_vrns = time_it(synthesize_multichannel, iters, addr+1, seed)
    print(f"Rust Context Fetch vs Raw Python synth Speedup: {t_py_vrns / t_rs_clean:.2f}x")
    print("=" * 50)

    # Validate Equivalence
    assert abs(ctx_rs.fetch(addr) - ctx_py.fetch(addr)) < 1e-12, "Overlay logic failure!"
    assert abs(ctx_rs.fetch(addr+1) - synthesize_multichannel(addr+1, seed)) < 1e-12, "Math logic failure!"
    print("\n-> Validation: Python AND Rust Contexts return equivalent topology results.")

if __name__ == '__main__':
    benchmark_bridge_throughput()
