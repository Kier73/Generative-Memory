#ifndef GMEM_ARO_H
#define GMEM_ARO_H

#include "gmem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Algebraic Rewrite Rule (ARR) Type.
 */
typedef enum {
  GMEM_ARR_IDENTITY,   // a + 0 = a, a * 1 = a
  GMEM_ARR_ZERO,       // a * 0 = 0
  GMEM_ARR_CONST,      // c1 + c2 = c3
  GMEM_ARR_LINEAR,     // a * x + b
  GMEM_ARR_ISOMORPHIC, // Complex graph simplified to Linear/Scalar
  GMEM_ARR_NONE        // No optimization found
} gmem_arr_rule_t;

/**
 * Perform Isomorphic Simplification on a variety signature.
 * Returns the simplified rule and the pre-computed constant if applicable.
 */
gmem_arr_rule_t gmem_aro_simplify(gmem_ctx_t ctx, uint64_t variety_sig,
                                  float *p1_out, float *p2_out);

/**
 * Register an Algebraic Rewrite Rule (ARR) for a specific variety signature.
 * p1: Constant/Intercept, p2: Slope (if applicable)
 */
void gmem_law_register(gmem_ctx_t ctx, uint64_t variety_sig,
                       gmem_arr_rule_t rule, float p1, float p2);

#ifdef __cplusplus
}
#endif

#endif // GMEM_ARO_H
