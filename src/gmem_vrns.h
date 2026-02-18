#ifndef GMEM_VRNS_H
#define GMEM_VRNS_H

#include <stddef.h>
#include <stdint.h>


/**
 * Project a 64-bit value into 8 parallel residues.
 * This provides O(1) bit-exact variety across multiple bases.
 */
typedef struct {
  uint16_t residues[8];
} gmem_rns_t;

/**
 * Generate a residue-based variety for a given address and seed.
 */
gmem_rns_t gmem_vrns_project(uint64_t addr, uint64_t seed);

/**
 * Convert RNS residues to a normalized 32-bit float (Torus Projection).
 */
float gmem_vrns_to_float(gmem_rns_t rns);

#endif // GMEM_VRNS_H
