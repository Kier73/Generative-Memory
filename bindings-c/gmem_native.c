// gmem_native.c
// Generative Memory: Bare-Metal C Integration Proof
//
// This script bypasses all Python bytecode limits and instantiates
// the GMem contextual math directly from the exact C-FFI used by OS kernels.

#include "gmem_native.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main() {
  printf("==========================================\n");
  printf("   Generative Memory natively in C\n");
  printf("==========================================\n");

  uint64_t SEED = 0x1337BEEFCAFEBABE;

  // 1. Allocate the pure mathematical lock-free context manifold.
  // Memory overhead: O(1) < 1 Kilobyte.
  CGMemContext *ctx = gmem_context_new(SEED);
  if (!ctx) {
    printf(
        "Failed to allocate CGMemContext. Is gmem_rs.dll linked properly?\n");
    return 1;
  }

  printf("[+] Allocated O(1) Manifold bounded by Seed: 0x%llX\n",
         (unsigned long long)SEED);

  // 2. Resolve native topological coordinates
  // Mathematically: H = \bigoplus_{i=0}^{15} fmix64( (addr \pmod{p_i}) \lor
  // (p_i \ll 16) \lor (i \ll 32) ) / (2^{64}-1)
  printf("\n--- Topological Coordinates ---\n");
  for (uint64_t i = 1000; i < 1005; ++i) {
    double val = gmem_fetch(ctx, i);
    printf("  Fetch[%llu] = %.6f\n", (unsigned long long)i, val);
  }

  // 3. Write physical dirty pages (Physical Override mapping)
  printf("\n--- Physical Overrides ---\n");
  gmem_write(ctx, 1002, 3.14159);
  printf("  Write[1002] -> 3.14159\n");
  printf("  Fetch[1002] = %.6f (Expected: 3.14159)\n", gmem_fetch(ctx, 1002));

  size_t dirty_pages = gmem_overlay_count(ctx);
  printf("  Total Physics Overlay Count: %zu pages consumed.\n", dirty_pages);

  // 4. Bulk tensor initialization benchmark
  // Simulating the instantaneous hydration of a large parametric layer
  printf("\n--- C-Native Vectorized Hydration ---\n");
  size_t layer_size = 10000000; // 10 million parameters
  double *native_layer = (double *)malloc(layer_size * sizeof(double));
  if (!native_layer) {
    printf("Failed to allocate test buffer.\n");
    return 1;
  }

  clock_t start = clock();
  gmem_fill_tensor(ctx, native_layer, layer_size, 0);
  clock_t end = clock();

  double exec_time = (double)(end - start) / CLOCKS_PER_SEC;
  double ops_sec = (double)layer_size / exec_time;

  printf("Hydrated %zu floats directly into C++ allocation in %.4f seconds.\n",
         layer_size, exec_time);
  printf("Achieved Throughput: %.0f ops/sec (Single thread bare-metal)\n",
         ops_sec);

  free(native_layer);
  gmem_context_free(ctx);

  printf("\nSuccess. C memory pipeline freed perfectly.\n");
  return 0;
}
