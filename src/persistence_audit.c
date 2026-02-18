
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gmem.h"

int main() {
    uint64_t seed = 0xABCDE;
    uint64_t addr1 = 1000;
    float val1 = 123.456f;
    uint64_t addr2 = 2000;
    float val2 = 789.012f;

    printf("  1. Creating Session 1 and writing to AOF...\n");
    gmem_ctx_t ctx1 = gmem_create(seed);
    if (gmem_persistence_attach(ctx1, "test_persistence.gvm_delta") != 0) {
        printf("  [FAIL] Could not attach persistence file\n");
        return 1;
    }
    
    gmem_write_f32(ctx1, addr1, val1);
    gmem_write_f32(ctx1, addr2, val2);
    gmem_destroy(ctx1);
    
    printf("  2. Creating Session 2 and re-attaching AOF...\n");
    gmem_ctx_t ctx2 = gmem_create(seed);
    
    // Before attach, should be generative noise
    float noise = gmem_fetch_f32(ctx2, addr1);
    printf("  Baseline (Noise) at addr %llu: %f\n", (unsigned long long)addr1, noise);

    if (gmem_persistence_attach(ctx2, "test_persistence.gvm_delta") != 0) {
        printf("  [FAIL] Could not re-attach persistence file\n");
        return 1;
    }
    
    // After attach, should match session 1
    float read1 = gmem_fetch_f32(ctx2, addr1);
    float read2 = gmem_fetch_f32(ctx2, addr2);
    
    printf("  Readout 1: %f (Expected %f)\n", read1, val1);
    printf("  Readout 2: %f (Expected %f)\n", read2, val2);
    
    if (read1 == val1 && read2 == val2) {
        printf("  [PASS] Persistence Verified: Materialized changes survived session reboot.\n");
    } else {
        printf("  [FAIL] Parity Mismatch: Log replay failed.\n");
        return 1;
    }
    
    gmem_destroy(ctx2);
    return 0;
}
