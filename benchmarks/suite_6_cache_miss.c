// suite_6_cache_miss.c
// Test 1: CPU L3 Cache Eviction vs Mathematical Generation
// Compiled natively to benchmark MSI Cyborg 15 A12V cache architecture

#include "../Generative_Memory/gmem_native.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// 1 Gigabyte array of doubles to guarantee L3 Cache (24MB Intel) eviction
#define ARRAY_SIZE 134217728
#define NUM_LOOKUPS 10000000

// Super fast pseudo-random generator for array indexes
uint64_t xorshift64(uint64_t *state) {
  uint64_t x = *state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return *state = x;
}

int main() {
  printf("==========================================\n");
  printf(" SUITE 5: L3 CACHE EVICTION vs GMEM MATH \n");
  printf("==========================================\n");

  uint64_t rng_state = 0xDEADBEEFC0DECAFE;

  // ---------------------------------------------------------
  // TEST A: Physical RAM Random Access (Forcing Cache Misses)
  // ---------------------------------------------------------
  printf("\n[~] Allocating 1.0 GB Physical Double Array to saturate L3 "
         "Cache...\n");

  double *huge_ram_array = (double *)malloc(ARRAY_SIZE * sizeof(double));
  if (!huge_ram_array) {
    printf("[!] Malloc failed. Out of system memory.\n");
    return 1;
  }

  // Hydrate array to avoid OS page faults during the read benchmark
  for (size_t i = 0; i < ARRAY_SIZE; i++) {
    huge_ram_array[i] = (double)(i % 100) / 100.0;
  }

  printf(
      "    -> Allocation complete. Beginning 10 Million random RAM lookups.\n");

  clock_t start = clock();
  double dummy_sum = 0.0;

  for (size_t i = 0; i < NUM_LOOKUPS; i++) {
    uint64_t random_index = xorshift64(&rng_state) % ARRAY_SIZE;
    dummy_sum +=
        huge_ram_array[random_index]; // Forces a Cache Miss into main DDR5 RAM
  }

  clock_t end = clock();
  double ram_time = (double)(end - start) / CLOCKS_PER_SEC;
  double ram_ops = (double)NUM_LOOKUPS / ram_time;

  printf("\n--- Test A: Pure RAM Random Access ---\n");
  printf("Execute Time:    %.4f seconds\n", ram_time);
  printf("Throughput:      %.0f ops/sec\n", ram_ops);
  printf("Check Sum:       %.4f\n", dummy_sum);

  free(huge_ram_array);

  // ---------------------------------------------------------
  // TEST B: Generative Virtual Memory Math (O(1) ALU only)
  // ---------------------------------------------------------
  printf("\n[~] Instantiating Zero-Byte Rust Virtual Context...\n");
  CGMemContext *ctx = gmem_context_new(0x1337BEEFCAFEBABE);

  rng_state = 0xDEADBEEFC0DECAFE; // Reset RNG so paths match
  dummy_sum = 0.0;

  printf(
      "    -> Context mapped. Beginning 10 Million random Math Navigations.\n");

  start = clock();

  for (size_t i = 0; i < NUM_LOOKUPS; i++) {
    uint64_t random_index = xorshift64(&rng_state) % ARRAY_SIZE;
    dummy_sum += gmem_fetch(
        ctx, random_index); // Mathematically synthesizes the address instantly
  }

  end = clock();
  double gmem_time = (double)(end - start) / CLOCKS_PER_SEC;
  double gmem_ops = (double)NUM_LOOKUPS / gmem_time;

  printf("\n--- Test B: Generative Memory Math Synthesis ---\n");
  printf("Execute Time:    %.4f seconds\n", gmem_time);
  printf("Throughput:      %.0f ops/sec\n", gmem_ops);
  printf("Check Sum:       %.4f\n", dummy_sum);

  gmem_context_free(ctx);

  // ---------------------------------------------------------
  // Conclusion
  // ---------------------------------------------------------
  printf("\n=== CONCLUSION ===\n");
  if (gmem_time < ram_time) {
    printf("Generative Memory achieved absolute superiority.\n");
    printf("Math calculation is %.2fX FASTER than retrieving a physical byte "
           "missing the L3 Cache.\n",
           ram_time / gmem_time);
  } else {
    printf("DDR5 RAM is %.2fX FASTER than math synthesis.\n",
           gmem_time / ram_time);
  }

  return 0;
}
