#include "gmem.h"
#include "gmem_trinity.h"
#include <stdio.h>
#include <stdlib.h>


int main() {
  printf("Initializing GVM Context (Seed: 0xDEADBEEF)...\n");
  gmem_ctx_t ctx = gmem_create(0xDEADBEEF);
  if (!ctx) {
    printf("FAILED to create GVM context.\n");
    return 1;
  }
  printf("SUCCESS: GVM Context created.\n");

  printf("Testing Trinity Inductive Resolve (X=100, Y=200, N=1024)...\n");
  float val = g_inductive_resolve_sorted(ctx, 100, 200, 1024);
  printf("Resolved Value: %f\n", val);

  printf("Testing Hierarchical Z-Mask Write (Addr: 1,000,000)...\n");
  gmem_write_f32(ctx, 1000000, 123.456f);
  printf("Write Successful.\n");

  printf("Destroying Context...\n");
  gmem_destroy(ctx);
  printf("SUCCESS: Diagnostic Complete.\n");

  return 0;
}
