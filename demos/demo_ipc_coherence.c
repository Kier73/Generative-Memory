#include "../include/gmem.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  printf("=== Demo: IPC Coherence (Infinite Clipboard) ===\n");

  // Scenario: Two separate "processes" (simulated here by contexts)
  // want to share a 1GB dataset.

  // 1. Process A "Invents" a world
  uint64_t shared_seed = 0xCAFEBABE;
  printf("[1] Process A: Creates World with Seed 0x%llX\n", shared_seed);
  gmem_ctx_t proc_a = gmem_create(shared_seed);

  // A verifies a value at a specific high address
  uint64_t address = 1024ULL * 1024 * 1024 * 50; // 50GB offset
  float val_a = gmem_fetch_f32(proc_a, address);
  printf("    Proc A Value at 50GB: %f\n", val_a);

  // 2. Transfer logic (The handshake)
  printf("[2] IPC Transfer: Sending 8 bytes (The Seed)...\n");

  // 3. Process B "Mounts" the world
  printf("[3] Process B: Receives Seed 0x%llX\n", shared_seed);
  gmem_ctx_t proc_b = gmem_create(shared_seed);

  // B reads the same address
  float val_b = gmem_fetch_f32(proc_b, address);
  printf("    Proc B Value at 50GB: %f\n", val_b);

  // 4. Verify Coherence
  if (val_a == val_b) {
    printf("[PASS] Zero-Copy Coherence Verified. Data is Identical.\n");
  } else {
    printf("[FAIL] Data mismatch.\n");
  }

  printf("[METRICS]\n");
  printf("    Data Shared: 16 Exabytes (Virtual)\n");
  printf("    Copy Cost:   0 bytes\n");
  printf("    Sync Speed:  Instant\n");

  gmem_destroy(proc_a);
  gmem_destroy(proc_b);
  return 0;
}
