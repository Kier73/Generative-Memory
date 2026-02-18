
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "gmem.h"
#include "gmem_manager.h"

int main() {
    uint64_t seed = 0xFACEFEED;
    uint64_t addr = 0x2000;
    float val = 123.456f;
    
    gmem_manager_t mgr = gmem_manager_init();
    gmem_manager_net_start(mgr);
    
    gmem_ctx_t ctx = gmem_manager_get_by_seed(mgr, seed);
    
    printf("Broadcasting write...\n");
    // Write triggers broadcast
    gmem_write_f32(ctx, addr, val);
    
    // Keep alive briefly to ensure packet sends
    Sleep(1000);
    
    gmem_manager_shutdown(mgr);
    return 0;
}
