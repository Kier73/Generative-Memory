import os
import time
import numpy as np

# ARO Signatures from gmem_aro.c
SIG_CONSTANT_FILL = 0x1234567887654321

def benchmark_aro():
    print("=== GVM Phase 14: ARO Benchmark (Algebraic Rewrite) ===")
    
    # 1. Standard Synthesis (Seed = 0xDEADBEEF)
    # This won't trigger ARO shunting (GMEM_ARR_NONE)
    print("  Measuring Standard Page-Run Synthesis (No ARO)...")
    os.system("taskkill /F /IM gmem_mount.exe >nul 2>&1")
    # Using drive Z: for benchmark
    cmd_standard = f'powershell "$env:PATH += \";C:\\Program Files (x86)\\WinFsp\\bin\"; .\\build\\Release\\gmem_mount.exe Z: -f -s 0xDEADBEEF"'
    # Note: I need to start the service in background or use a diagnostic for raw throughput
    
    # Alternative: Use direct FFI/C-diag for more accurate results
    print("  [INFO] Running internal diagnostic for raw throughput...")
    
def run_diag(seed):
    import subprocess
    # Create a temporary diag source with Windows timers
    with open("src/aro_bench.c", "w") as f:
        f.write(f"""
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmem.h"

int main() {{
    uint64_t seed = {seed};
    gmem_ctx_t ctx = gmem_create(seed);
    float *buf = malloc(1024 * 1024 * 1024);
    
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    gmem_fetch_bulk_f32(ctx, 0, buf, 256 * 1024 * 1024);
    
    QueryPerformanceCounter(&end);
    double elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("%.3f", elapsed);
    
    free(buf);
    gmem_destroy(ctx);
    return 0;
}}
""")
    # Compile and run
    subprocess.run(["gcc", "-O3", "-mavx2", "-Iinclude", "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", "src/gmem_aro.c", "src/aro_bench.c", "-o", "build/Release/aro_bench.exe"], check=True, stderr=subprocess.PIPE)
    result = subprocess.run(["./build/Release/aro_bench.exe"], capture_output=True, text=True)
    # The output might contain compiler messages if not careful, but diag.exe should only print elapsed
    return float(result.stdout.strip())

def run_raw_baseline():
    import subprocess
    with open("src/raw_bench.c", "w") as f:
        f.write(f"""
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

int main() {{
    float *buf = malloc(1024 * 1024 * 1024);
    __m256 v = _mm256_set1_ps(1.0f);
    
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    for (size_t k = 0; k < 256 * 1024 * 1024; k += 8) {{
        _mm256_storeu_ps(&buf[k], v);
    }}
    
    QueryPerformanceCounter(&end);
    double elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("%.3f", elapsed);
    
    free(buf);
    return 0;
}}
""")
    subprocess.run(["gcc", "-O3", "-mavx2", "src/raw_bench.c", "-o", "build/Release/raw_bench.exe"], check=True)
    result = subprocess.run(["./build/Release/raw_bench.exe"], capture_output=True, text=True)
    return float(result.stdout.strip())

if __name__ == "__main__":
    t_raw = run_raw_baseline()
    print(f"  System Raw Peak (SIMD Fill): {t_raw:.3f}s (Throughput: {(1.0/t_raw):.2f} GB/s)")
    
    t_standard = run_diag("0xDEADBEEF")
    print(f"  Standard Synthesis: {t_standard:.3f}s (Throughput: {(1.0/t_standard):.2f} GB/s)")
    
    t_aro = run_diag("0x1234567887654321")
    print(f"  ARO-Optimized (Constant Shunt): {t_aro:.3f}s (Throughput: {(1.0/t_aro):.2f} GB/s)")
    
    print(f"\n[CONCLUSION] ARO efficiency vs Peak: { (t_raw / t_aro) * 100 :.2f}%")
    speedup = t_standard / t_aro
    print(f"ARO Speedup: {speedup:.2f}x")
