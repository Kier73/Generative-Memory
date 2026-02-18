import subprocess
import os
import time

def run_exascale_gauntlet():
    print("=== GVM Gauntlet: Exascale Throughput & Parity Stress ===")
    
    with open("src/exascale_gauntlet.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gmem.h"

int main() {{
    uint64_t seed = 0x1337BEEF;
    uint64_t exascale_offset = 1152921504606846976ULL; // 1 Exabyte (10^18)
    size_t fetch_count = 10000000; // 10 million floats (40MB)
    float *buffer = (float *)malloc(fetch_count * sizeof(float));
    
    gmem_ctx_t ctx = gmem_create(seed);
    
    printf("  1. Testing 1EB Offset Read (Scalar Fallback)...\\n");
    clock_t start = clock();
    for(size_t i=0; i<1000; i++) {{
        gmem_fetch_f32(ctx, exascale_offset + i);
     acorns: ; // prevent optimization
    }}
    clock_t end = clock();
    double scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("    Scalar sample complete.\\n");

    printf("  2. Testing 1EB Offset Bulk Read (AVX2 Path)...\\n");
    start = clock();
    gmem_fetch_bulk_f32(ctx, exascale_offset, buffer, fetch_count);
    end = clock();
    double bulk_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    double throughput = (fetch_count * sizeof(float)) / (bulk_time * 1024 * 1024 * 1024);
    printf("    Bulk read complete. Throughput: %.2f GB/s\\n", throughput);

    printf("  3. Verifying Bit-Parity at Exascale Edge...\\n");
    for(size_t i=0; i<100; i++) {{
        float scalar = gmem_fetch_f32(ctx, exascale_offset + i);
        if (buffer[i] != scalar) {{
            printf("  [FAIL] Parity mismatch at offset + %zu! (Bulk: %f, Scalar: %f)\\n", i, buffer[i], scalar);
            return 1;
        }}
    }}
    printf("    Exascale Integrity Verified.\\n");

    printf("  4. Testing 500TB Offset Bulk Read (Inside Z-Mask Range)...\\n");
    uint64_t inside_offset = 500ULL * 1024 * 1024 * 1024 * 1024 / sizeof(float);
    start = clock();
    gmem_fetch_bulk_f32(ctx, inside_offset, buffer, fetch_count);
    end = clock();
    double mid_time = (double)(end - start) / CLOCKS_PER_SEC;
    double mid_throughput = (fetch_count * sizeof(float)) / (mid_time * 1024 * 1024 * 1024);
    printf("    In-range throughput: %.2f GB/s\\n", mid_throughput);
    
    gmem_destroy(ctx);
    free(buffer);
    return 0;
}}
""")
    
    print("  Compiling Exascale Gauntlet...")
    sources = [
        "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", 
        "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", 
        "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", 
        "src/gmem_archetype.c", "src/exascale_gauntlet.c"
    ]
    
    # Force AVX2
    cmd = ["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc"] + sources + ["-lws2_32", "-o", "build/Release/exascale_gauntlet.exe"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return

    # Run
    result = subprocess.run(["./build/Release/exascale_gauntlet.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Exascale Gauntlet Passed.")
    else:
        print(f"[ERROR] Exascale Gauntlet Failed with Exit Code {result.returncode}")
        print(result.stderr)

if __name__ == "__main__":
    run_exascale_gauntlet()
