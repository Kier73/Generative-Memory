#include "gmem_vrns.h"

static const uint16_t MODULI[8] = {251, 257, 263, 269, 271, 277, 281, 283};

gmem_rns_t gmem_vrns_project(uint64_t addr, uint64_t seed) {
  uint64_t x = addr ^ seed;
  gmem_rns_t rns;
  for (int i = 0; i < 8; i++) {
    rns.residues[i] = (uint16_t)(x % MODULI[i]);
  }
  return rns;
}

float gmem_vrns_to_float(gmem_rns_t rns) {
  // Torus Projection: Combining residues into a single high-entropy float.
  // We use a simplified fractional summation (Analytic Induction).
  double accumulator = 0.0;
  for (int i = 0; i < 8; i++) {
    accumulator += (double)rns.residues[i] / (double)MODULI[i];
  }
  // Return fractional part
  return (float)(accumulator - (long)accumulator);
}
