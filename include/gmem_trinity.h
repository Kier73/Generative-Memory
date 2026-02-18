#ifndef GMEM_TRINITY_H
#define GMEM_TRINITY_H

#include "gmem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Trinity Moduli: Industrial Goldilocks primes.
 */
#define GMEM_MOD_GOLDILOCKS 0xFFFFFFFF00000001ULL
#define GMEM_MOD_SAFE 0xFFFFFFFF7FFFFFFFULL
#define GMEM_MOD_ULTRA 0xFFFFFFFFFFFFFFC5ULL

/**
 * Solve a synthetic manifold at coordinate (x, y) using Trinity Synergy.
 * V_final = V_law ^ V_choice ^ V_event
 * @return Bit-exact 128-bit residue reconstruction.
 */
unsigned __int128 g_trinity_solve_rns(gmem_ctx_t ctx, const char *intention,
                                      const char *law, uint64_t x, uint64_t y,
                                      uint64_t event_sig);

/**
 * Resolve a sorted manifold position into a float variety [0, 1].
 * O(1) Inductive Sort Implementation.
 */
float g_inductive_resolve_sorted(gmem_ctx_t ctx, uint64_t x, uint64_t y,
                                 uint64_t n);

#ifdef __cplusplus
}
#endif

#endif // GMEM_TRINITY_H
