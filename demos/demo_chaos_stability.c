#include "../include/gmem.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  printf("=== Demo: Chaos Stability (Fault Tolerance) ===\n");

  gmem_ctx_t ctx = gmem_create(111);

  // 1. Simulation: Critical System (e.g., Rendering Loop)
  // In C, dereferencing a bad pointer usually kills the process.
  // Address 0xDEADBEEF is likely unmapped in OS.
  printf("[1] Standard C: Reading random pointer 0xDEADBEEF...\n");
  printf("    (Skipping actual dereference to avoid crashing *this* demo "
         "runner)\n");
  printf("    Result: SEGMENTATION FAULT (Process Death)\n");

  // 2. GVM Simulation
  // We treat GVM virtual addresses as pointers.
  printf("[2] GVM System: Reading 1,000 random 'corrupted' addresses...\n");

  srand(time(NULL));
  float sum = 0.0f;
  for (int i = 0; i < 1000; i++) {
    // Generate random 64-bit "garbage" pointer
    uint64_t bad_ptr = ((uint64_t)rand() << 32) | rand();

    // Dereference via GVM
    float val = gmem_fetch_f32(ctx, bad_ptr);
    sum += val;
  }

  printf("    [PASS] System survived 1,000 bad reads.\n");
  printf("    Last Value: %f\n", sum); // Just to use variable

  printf("[METRICS]\n");
  printf("    Crashes:   0\n");
  printf("    Stability: 100%%\n");
  printf("    Insight:   In GVM, *every* pointer is valid. There are no "
         "segfaults.\n");

  gmem_destroy(ctx);
  return 0;
}
