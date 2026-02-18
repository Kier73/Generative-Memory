
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmem.h"

int main() {
    uint64_t seed = 0x1234567887654321;
    gmem_ctx_t ctx = gmem_create(seed);
    float *buf = malloc(1024 * 1024 * 1024);
    
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    gmem_fetch_bulk_f32(ctx, 0, buf, 256 * 1024 * 1024);
    
    QueryPerformanceCounter(&end);
    double elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("%.3f", elapsed);
    
    free(buf);
    gmem_destroy(ctx);
    return 0;
}
