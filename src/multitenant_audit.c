
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmem.h"
#include "gmem_manager.h"

int main() {
    gmem_manager_t mgr = gmem_manager_init();
    
    // 1. Isolation Test: Different seeds should produce different values
    gmem_ctx_t ctx1 = gmem_manager_get_by_seed(mgr, 0x111);
    gmem_ctx_t ctx2 = gmem_manager_get_by_seed(mgr, 0x222);
    
    float v1 = gmem_fetch_f32(ctx1, 100);
    float v2 = gmem_fetch_f32(ctx2, 100);
    
    printf("  Isolation: Seed 0x111 -> %f, Seed 0x222 -> %f\n", v1, v2);
    if (v1 == v2) {
        printf("  [FAIL] Isolation Mismatch: Seeds produced identical values!\n");
        return 1;
    }
    printf("  [PASS] Isolation: Contexts are independent.\n");
    
    // 2. Leakage Test: Writing to one should NOT affect the other
    gmem_write_f32(ctx1, 500, 0.99f);
    float l1 = gmem_fetch_f32(ctx1, 500);
    float l2 = gmem_fetch_f32(ctx2, 500);
    
    printf("  Leakage: Write 0.99 to t1. t1 -> %f, t2 -> %f\n", l1, l2);
    if (l1 == l2) {
        printf("  [FAIL] Leakage Detected: Write to Tenant A leaked to Tenant B!\n");
        return 1;
    }
    printf("  [PASS] Leakage: Overlays are isolated.\n");
    
    // 3. Quota Test: Enforcement of entry limits
    gmem_ctx_t ctx_quota = gmem_manager_get_by_seed_with_quota(mgr, 0x333, 10);
    printf("  Quota: Testing 10-entry limit on Seed 0x333...\n");
    
    for(int i=0; i<20; i++) {
        gmem_write_f32(ctx_quota, i*10, 0.5f);
    }
    
    // Check how many actually stuck (we don't have a count API yet, so we check the values)
    int count = 0;
    for(int i=0; i<20; i++) {
        if(gmem_fetch_f32(ctx_quota, i*10) == 0.5f) count++;
    }
    
    printf("  Quota: Attempted 20 writes, %d persisted.\n", count);
    if (count > 10) {
        printf("  [FAIL] Quota bypassed! entries: %d, limit: 10\n", count);
        return 1;
    }
    printf("  [PASS] Quota enforced.\n");
    
    gmem_manager_shutdown(mgr);
    return 0;
}
