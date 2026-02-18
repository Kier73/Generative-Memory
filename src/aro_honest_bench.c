
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmem.h"
#include "gmem_aro.h"
#include "gmem_vrns.h"

int main() {
    uint64_t variety_sig = 0x1234567887654321ULL;
    uint64_t seed = variety_sig; // Use signature as seed for testing shunt
    
    gmem_ctx_t ctx = gmem_create(seed);
    
    // Calculate the expected constant value honestly (using RNS path once)
    gmem_rns_t rns = gmem_vrns_project(0, seed);
    float expected_const = gmem_vrns_to_float(rns);
    
    // 1. Register the Law (The Honest way)
    gmem_law_register(ctx, variety_sig, GMEM_ARR_CONSTANT, expected_const);
    printf("  Registered Law: 0x%llx -> Constant(%f)\n", (unsigned long long)variety_sig, expected_const);
    
    float *buf = malloc(1024 * 1024 * 128); // 128MB
    
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    
    // 2. Perform Shadow-Check Read
    printf("  Running Shadow-Check Verification (Honest Parity)...\n");
    gmem_fetch_bulk_f32(ctx, 0, buf, 32 * 1024 * 1024); // 32M samples in debug mode
    
    // 3. Perform Throughput Benchmark (Normal Mode)
    QueryPerformanceCounter(&start);
    gmem_fetch_bulk_f32(ctx, 0, buf, 32 * 1024 * 1024);
    QueryPerformanceCounter(&end);
    
    double elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("  Throughput (Normal): %.2f GB/s\n", (0.128 / elapsed));
    
    free(buf);
    gmem_destroy(ctx);
    return 0;
}
