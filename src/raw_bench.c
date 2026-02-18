
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

int main() {
    float *buf = malloc(1024 * 1024 * 1024);
    __m256 v = _mm256_set1_ps(1.0f);
    
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    for (size_t k = 0; k < 256 * 1024 * 1024; k += 8) {
        _mm256_storeu_ps(&buf[k], v);
    }
    
    QueryPerformanceCounter(&end);
    double elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("%.3f", elapsed);
    
    free(buf);
    return 0;
}
