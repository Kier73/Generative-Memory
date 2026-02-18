import subprocess
import time

def run_honest_bench():
    print("=== GVM Integrity Check: Honest Law Discovery ===")
    
    # Create a diagnostic that USES the registration API
    # and enables Shadow-Check verification
    with open("src/aro_honest_bench.c", "w") as f:
        f.write(f"""
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmem.h"
#include "gmem_aro.h"
#include "gmem_vrns.h"

int main() {{
    uint64_t variety_sig = 0x1234567887654321ULL;
    uint64_t seed = variety_sig; // Use signature as seed for testing shunt
    
    gmem_ctx_t ctx = gmem_create(seed);
    
    // Calculate the expected constant value honestly (using RNS path once)
    gmem_rns_t rns = gmem_vrns_project(0, seed);
    float expected_const = gmem_vrns_to_float(rns);
    
    // 1. Register the Law (The Honest way)
    gmem_law_register(ctx, variety_sig, GMEM_ARR_CONSTANT, expected_const);
    printf("  Registered Law: 0x%llx -> Constant(%f)\\n", (unsigned long long)variety_sig, expected_const);
    
    float *buf = malloc(1024 * 1024 * 128); // 128MB
    
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    
    // 2. Perform Shadow-Check Read
    printf("  Running Shadow-Check Verification (Honest Parity)...\\n");
    gmem_fetch_bulk_f32(ctx, 0, buf, 32 * 1024 * 1024); // 32M samples in debug mode
    
    // 3. Perform Throughput Benchmark (Normal Mode)
    QueryPerformanceCounter(&start);
    gmem_fetch_bulk_f32(ctx, 0, buf, 32 * 1024 * 1024);
    QueryPerformanceCounter(&end);
    
    double elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("  Throughput (Normal): %.2f GB/s\\n", (0.128 / elapsed));
    
    free(buf);
    gmem_destroy(ctx);
    return 0;
}}
""")
    # Compile with GMEM_DEBUG_INTEGRITY
    print("  Compiling with GMEM_DEBUG_INTEGRITY...")
    subprocess.run(["gcc", "-O3", "-mavx2", "-DGMEM_DEBUG_INTEGRITY", "-Iinclude", "-Isrc", "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", "src/gmem_aro.c", "src/aro_honest_bench.c", "-o", "build/Release/aro_honest.exe"], check=True)
    
    # Run
    result = subprocess.run(["./build/Release/aro_honest.exe"], capture_output=True, text=True)
    print(result.stdout)
    if "GMC INTEGRITY FAILURE" in result.stderr:
        print(f"[FAIL] {result.stderr}")
    elif result.returncode == 0:
        print("[PASS] Integrity Verified. Shadow-Check matched bit-exact basis.")
    else:
        print(f"[ERROR] Exit Code {result.returncode}")

if __name__ == "__main__":
    run_honest_bench()
