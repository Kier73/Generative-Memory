# Generative Memory

A dependency-free, portable C library for **Procedural Data Synthesis** and **Synthetic Addressing**. GVM projects a virtual 1-Petabyte data space using deterministic mathematical functions, requiring nearly zero disk space while providing bit-exact uniformity.

##  Key Features
- **Large Scale**: Universal addressing from `0` to `2^64 - 1` ($2^{64}$ bytes).
- **Procedural Origin**: Deterministic data generation using a 4-round Feistel Shuffle and Variety Reverse Number System (VRNS).
- **Materiality Layer**: High-performance sparse overlay with **AVX2 SIMD** acceleration.
- **Persistence**: Append-Only File (AOF) logging for durable data modifications.
- **Mesh Coherence**: Autonomous node discovery and real-time synchronization via UDP.
- **Functional Morphing**: Real-time derivations ($V_{derived} = f(V_{source})$) without data duplication.
- **OS Integration**: Ready-to-use WinFSP/FUSE bridge for mounting 100TB+ virtual drives.

> [!NOTE]
> **System Requirements**: This library is optimized with **AVX2** instructions by default. It requires a CPU from ~2013 or later (Intel Haswell / AMD Excavator). Pre-Haswell systems will need to compile without the `-mavx2` flag.

## Physical Efficiency (1PB Context)
RAM usage is proportional to modifications ($O(M)$), not the virtual size ($O(V)$).

| Modification Count | Total RAM Usage | Efficiency Ratio (P:V) |
| :--- | :--- | :--- |
| **0** (Pure Generative) | ~64 MB | 1 : 16,000,000 |
| **1 Million** Writes | ~88 MB | 1 : 11,000,000 |
| **100 Million** Writes | ~2.4 GB | 1 : 400,000 |

## Quick Start (5 Minutes)

**Goal**: Create a 1 Petabyte virtual drive and read/write data in under 5 minutes.

1.  **Compile the Project**:
    ```bash
    mkdir build && cd build
    cmake ..
    cmake --build . --config Release
    ```

2.  **Run the Diagnostic**:
    The build produces `gmem_diag` (Diagnostic Tool). Run it to verify the system is working:
    ```bash
    ./Debug/gmem_diag.exe  # Windows
    ./gmem_diag            # Linux/macOS
    ```
    *Output should show "SUCCESS: GVM Context created" and "Resolved Value".*

3.  **Try the API (Mini-Demo)**:
    Create a file named `demo.c`:
    ```c
    #include "gmem.h"
    #include <stdio.h>

    int main() {
        // Create a space with seed 42
        gmem_ctx_t ctx = gmem_create(42);
        
        // Read from address 1 Trillion (far beyond RAM limits)
        float val = gmem_fetch_f32(ctx, 1000000000000ULL);
        printf("Procedural Value at 1TB: %f\n", val);
        
        // Write a value (stored in sparse overlay)
        gmem_write_f32(ctx, 1000000000000ULL, 999.99f);
        printf("New Value at 1TB: %f\n", gmem_fetch_f32(ctx, 1000000000000ULL));
        
        gmem_destroy(ctx);
        return 0;
    }
    ```
    Compile and run it linked against `gmem`.

## Build & Installation
GVM is built using CMake and requires a C99-compliant compiler.

### Standard Build
```bash
git clone https://github.com/Kier73/Generative-Memory.git
cd generative_memory
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Portability Note
When moving the `generative_memory` folder to a new location, always delete the `build/` folder first. CMake stores absolute paths that must be regenerated in the new environment.

## C API Integration
```c
#include "gmem.h"

int main() {
    // 1. Initialize 1PB Context
    gmem_ctx_t ctx = gmem_create(0x1337BEEF);

    // 2. Fetch Procedural Data (Deterministic)
    float val = gmem_fetch_f32(ctx, 1024ULL * 1024 * 1024); // at 1GB offset

    // 3. Materialize a Delta
    gmem_write_f32(ctx, 42, 3.1415f);

    // 4. Cleanup
    gmem_destroy(ctx);
    return 0;
}
```

## Python Integration (Zero-Dependency)
The `bindings/python/gmem.py` script provides a direct `ctypes` wrapper.

```python
from bindings.python.gmem import GenerativeMemory

# 1. Initialize 1PB Space
mem = GenerativeMemory(seed=0xCAFEBABE)

# 2. Bulk Fetch (Zero-Copy Speed: ~2.2 GB/s)
data = mem.get_range(start_addr=0, count=1_000_000)

print(f"First Value: {data[0]}")
```

## Testing & Verification
The `tests/` directory contains a comprehensive suite of Python audits and C-level "Gauntlet" stressors.
```bash
python tests/gauntlet_exascale_stress.py
python tests/gauntlet_material_stress.py
```

## License
Dual licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
