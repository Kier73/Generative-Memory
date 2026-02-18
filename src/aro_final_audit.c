
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmem.h"
#include "gmem_aro.h"
#include "gmem_vrns.h"

int main() {
    uint64_t variety_sig = 0x1234567887654321ULL;
    uint64_t seed = 0xAAABBCCCDDDULL;
    
    gmem_ctx_t ctx = gmem_create(seed);
    float *buf = malloc(1024 * 64); // 64KB
    
    // TEST 1: INVALID CONSTANT REGISTRATION
    printf("  Test 1: Registering INVALID Constant Law on Variable Manifold...\n");
    gmem_law_register(ctx, seed, GMEM_ARR_CONSTANT, 0.5f, 0.0f);
    
    printf("  Running Shadow-Check (Expect Integrity Failure)...\n");
    // This should trigger exit(1) in the shadow-check loop in gmem.c
    gmem_fetch_bulk_f32(ctx, 0, buf, 1024);
    
    printf("  [FAIL] Test 1 did not catch the mismatch!\n");
    free(buf);
    gmem_destroy(ctx);
    return 1;
}
