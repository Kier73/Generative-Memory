
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gmem.h"

int main(int argc, char* argv[]) {
    uint64_t seed = 0x12345;
    const char* filename = "test_persist.gvm";
    uint64_t addr = 0x1000;
    float val_test = 3.14159f;

    if (argc > 1 && strcmp(argv[1], "write") == 0) {
        // CLEANUP
        remove(filename);
        
        gmem_ctx_t ctx = gmem_create(seed);
        if(gmem_persistence_attach(ctx, filename) != 0) {
            return 1;
        }
        gmem_write_f32(ctx, addr, val_test);
        gmem_destroy(ctx);
        printf("Written\n");
        return 0;
    } 
    
    if (argc > 1 && strcmp(argv[1], "read") == 0) {
        gmem_ctx_t ctx = gmem_create(seed);
        if(gmem_persistence_attach(ctx, filename) != 0) {
            return 2;
        }
        float val_out = gmem_fetch_f32(ctx, addr);
        gmem_destroy(ctx);
        
        if (fabs(val_out - val_test) < 0.0001) {
             printf("Verified: %f\n", val_out);
             return 0;
        } else {
             printf("Failed: %f != %f\n", val_out, val_test);
             return 3;
        }
    }
    return 0;
}
