
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "gmem.h"
#include "gmem_manager.h"

int main() {
    uint64_t seed = 0xFACEFEED;
    uint64_t addr = 0x2000;
    float expected = 123.456f;
    
    gmem_manager_t mgr = gmem_manager_init();
    gmem_manager_net_start(mgr);
    
    // Provision the context so we can receive updates for it
    gmem_ctx_t ctx = gmem_manager_get_by_seed(mgr, seed);
    
    printf("Listing...\n");
    fflush(stdout);
    
    // Poll for update (timeout 5s)
    for(int i=0; i<50; i++) {
        float val = gmem_fetch_f32(ctx, addr);
        if (val == expected) {
            printf("Received Sync: %f\n", val);
            gmem_manager_shutdown(mgr);
            return 0;
        }
        Sleep(100);
    }
    
    printf("Timeout waiting for sync.\n");
    gmem_manager_shutdown(mgr);
    return 1;
}
