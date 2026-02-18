
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmem.h"
#include "gmem_manager.h"

int main() {
    gmem_manager_t mgr = gmem_manager_init();
    
    // 1. Create Source and Shadow
    gmem_ctx_t source = gmem_manager_get_by_seed(mgr, 0xABC);
    gmem_ctx_t shadow = gmem_manager_get_by_seed(mgr, 0xDEF); // Seed is different, but will be shadowed
    
    printf("  Initial Fetch (Pre-Mirror): source[100]=%f, shadow[100]=%f\n", 
           gmem_fetch_f32(source, 100), gmem_fetch_f32(shadow, 100));
    
    // Attach Shadow
    printf("  Attaching Shadow to Source (GMEM_MIRROR_IDENTITY)...\n");
    gmem_mirror_attach(shadow, source, GMEM_MIRROR_IDENTITY);
    
    float v_src = gmem_fetch_f32(source, 100);
    float v_shd = gmem_fetch_f32(shadow, 100);
    printf("  Fetch (Post-Mirror): source[100]=%f, shadow[100]=%f\n", v_src, v_shd);
    
    if (v_src != v_shd) {
        printf("  [FAIL] Mirror Mismatch: Shadow did not pull from Source!\n");
        return 1;
    }
    printf("  [PASS] Shadow reflects Source generative layer.\n");
    
    // 2. Write to Source -> Should be visible in Shadow
    printf("  Writing 1.23 to Source[500]...\n");
    gmem_write_f32(source, 500, 1.23f);
    if (gmem_fetch_f32(shadow, 500) != 1.23f) {
        printf("  [FAIL] Mirror Mismatch: Shadow did not see Source overlay write!\n");
        return 1;
    }
    printf("  [PASS] Shadow reflects Source overlay writes.\n");
    
    // 3. Write to Shadow -> Should NOT be visible in Source (Delta Isolation)
    printf("  Writing 4.56 to Shadow[600]...\n");
    gmem_write_f32(shadow, 600, 4.56f);
    if (gmem_fetch_f32(source, 600) == 4.56f) {
        printf("  [FAIL] Delta Leakage: Shadow write leaked back to Source!\n");
        return 1;
    }
    printf("  [PASS] Shadow has isolated Delta Overlay (Coherence Check Passed).\n");
    
    // 4. Bulk Fetch Parity
    float buf_src[16], buf_shd[16];
    gmem_fetch_bulk_f32(source, 1000, buf_src, 16);
    gmem_fetch_bulk_f32(shadow, 1000, buf_shd, 16);
    
    for(int i=0; i<16; i++) {
        if(buf_src[i] != buf_shd[i]) {
            printf("  [FAIL] Bulk Parity Mismatch at index %d!\n", i);
            return 1;
        }
    }
    printf("  [PASS] Bulk Mirror Parity Verified.\n");
    
    gmem_manager_shutdown(mgr);
    return 0;
}
