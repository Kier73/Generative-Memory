import os
import sys

# Ensure gmem is in your PYTHONPATH or run from the project root
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gmem import (
    GMemContext, JSONFilter, BlockFilter, 
    hilbert_xy_to_d, phi, psi_read, psi_write
)

def demonstrate_navigational_space():
    print("=== GMem: Practical Navigational Space Demo ===")
    ctx = GMemContext(seed=12345)

    # 1. Semantic Navigation (JSON Filter)
    # This turns the 16-EB space into a "Virtual Object Store"
    print("\n[1] Semantic Navigation (Virtual Metadata Tree)")
    js = JSONFilter("enterprise_vfs")
    
    paths = [
        "root/users/admin/permissions",
        "root/users/guest/limits",
        "root/config/network/mtu",
        "root/data/shards/0xC0FFEE/last_modified"
    ]
    
    for p in paths:
        addr = js.resolve_path(p)
        val = js.get_val(ctx, p)
        print(f"  Path: {p:<40} | Addr: 0x{addr:016X} | Val: {val:.6f}")

    # 2. Spatial Navigation (Hilbert Surface)
    # Mapping a 2D coordinate system into the sparse manifold
    print("\n[2] Spatial Navigation (Hilbert Virtual Grid)")
    grid_res = 1024
    test_coords = [(0, 0), (0, 1), (512, 512), (1023, 1023)]
    
    for x, y in test_coords:
        d_index = hilbert_xy_to_d(grid_res, x, y)
        val = ctx.fetch(d_index)
        print(f"  Coord: ({x:>4}, {y:>4}) | Hilbert Index: {d_index:>10} | Val: {val:.6f}")

    # 3. Linear System Bridging (Phi/Psi)
    # Treating the manifold as a virtual swap or linear memory space
    print("\n[3] Linear Addressing Bridge (Phi/Psi Mapping)")
    linear_addrs = [0x00400000, 0x00400004, 0x00401000] # Page-boundary awareness
    
    for l_addr in linear_addrs:
        v_addr = phi(l_addr, ctx.seed)
        # Note: Psi read/write ensures that nearby linear addresses 
        # map to the same virtual page relative offset.
        val = psi_read(ctx, l_addr)
        print(f"  Linear: 0x{l_addr:08X} | Manifold Addr: 0x{v_addr:016X} | Val: {val:.6f}")

    print("\n[Summary]")
    print("The address space is not 'noise'—it's a deterministic 'Coordinate System'.")
    print("If you know the Seed and the Path/Coordinate, you can always find the same data.")
    print("This allows for 16-EB virtual file systems, 2D holographic grids, and mapping")
    print("entire system memory maps into a single sparse context.")

if __name__ == "__main__":
    demonstrate_navigational_space()
