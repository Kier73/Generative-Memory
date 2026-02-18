#include "../include/gmem.h"
#include "../include/gmem_aro.h"
#include <assert.h>
#include <stdio.h>


int main() {
  printf("[TEST] ARO Symbolic Engine...\n");

  gmem_ctx_t ctx = gmem_create(12345);

  // Test 1: Null Rule (Seed 0) -> x * 0
  // gmem_aro.c detects seed 0 and builds (x * 0) + 0 which simplifies to
  // CONST(0)

  float p1 = -1.0f, p2 = -1.0f;
  gmem_arr_rule_t rule = gmem_aro_simplify(ctx, 0, &p1, &p2);

  if (rule == GMEM_ARR_CONST && p1 == 0.0f) {
    printf("[PASS] Null Identity (Seed 0 -> CONST 0)\n");
  } else {
    printf("[FAIL] Null Identity Failed. Rule: %d, p1: %f\n", rule, p1);
    return 1;
  }

  // Test 2: Linear Identity (Seed 0xDEADBEEF) -> x * 1
  // gmem_aro.c detects seed 0xDEADBEEF and builds x * 1 which simplifies to
  // VAR_X

  rule = gmem_aro_simplify(ctx, 0xDEADBEEF, &p1, &p2);
  if (rule == GMEM_ARR_LINEAR && p2 == 1.0f && p1 == 0.0f) {
    printf("[PASS] Linear Identity (Seed DEADBEEF -> 1*x + 0)\n");
  } else {
    printf("[FAIL] Linear Identity Failed. Rule: %d, p1: %f, p2: %f\n", rule,
           p1, p2);
    return 1;
  }

  printf("[SUCCESS] All ARO Tests Passed.\n");
  gmem_destroy(ctx);
  return 0;
}
