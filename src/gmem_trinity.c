#include "../include/gmem_trinity.h"
#include "gmem_internal.h"
#include <string.h>

// --- INTERNAL HELPERS ---

// 2D to 1D Hilbert Curve Mapping
static uint64_t hilbert_xy_to_d(uint64_t n, uint64_t x, uint64_t y) {
  uint64_t d = 0;
  for (uint64_t s = n / 2; s > 0; s /= 2) {
    uint64_t rx = (x & s) > 0;
    uint64_t ry = (y & s) > 0;
    d += s * s * ((3 * rx) ^ ry);

    // Rotate/Flip
    if (ry == 0) {
      if (rx == 1) {
        x = n - 1 - x;
        y = n - 1 - y;
      }
      uint64_t t = x;
      x = y;
      y = t;
    }
  }
  return d;
}

// Simple DjB2 Hash for string signatures
static uint64_t hash_signature(const char *str) {
  uint64_t hash = 5381;
  int c;
  while ((c = *str++))
    hash = ((hash << 5) + hash) + c;
  return hash;
}

// Extended Euclidean Algorithm for Mod Inverse
static __int128 mod_inverse(__int128 a, __int128 m) {
  __int128 m0 = m, t, q;
  __int128 x0 = 0, x1 = 1;
  if (m == 1)
    return 0;
  while (a > 1) {
    q = a / m;
    t = m;
    m = a % m, a = t;
    t = x0;
    x0 = x1 - q * x0;
    x1 = t;
  }
  if (x1 < 0)
    x1 += m0;
  return x1;
}

// --- TRINITY API ---

float g_inductive_resolve_sorted(gmem_ctx_t ctx, uint64_t x, uint64_t y,
                                 uint64_t n) {
  // 1. Hilbert Mapping
  uint64_t d = hilbert_xy_to_d(n, x, y);

  // 2. Monotonic Resolve (vGPU Inductive Sort logic)
  double total_size = (double)n * (double)n;
  double base_ramp = (double)d / total_size;

  // 3. Subtle variety based on seed
  uint64_t variety_seed = d ^ ctx->seed;
  variety_seed *= 0x517cc1b727220a95ULL;
  variety_seed ^= variety_seed >> 31;
  double variety = ((double)variety_seed / 18446744073709551615.0) / total_size;

  return (float)(base_ramp + variety);
}

unsigned __int128 g_trinity_solve_rns(gmem_ctx_t ctx, const char *intention,
                                      const char *law, uint64_t x, uint64_t y,
                                      uint64_t event_sig) {
  uint64_t i_sig = hash_signature(intention);
  uint64_t l_sig = hash_signature(law);
  uint64_t result_sig = i_sig ^ l_sig ^ event_sig ^ ctx->seed;

  // Moduli
  const uint64_t m[] = {GMEM_MOD_GOLDILOCKS, GMEM_MOD_SAFE, GMEM_MOD_ULTRA};
  const int count = 3;
  unsigned __int128 residues[3];

  // 1. Generate Residues
  for (int i = 0; i < count; i++) {
    uint64_t mix = result_sig;
    mix ^= (x ^ (y << 13));
    mix *= m[i];
    residues[i] = mix % m[i];
  }

  // 2. Chinese Remainder Theorem (CRT) Reconstruction
  unsigned __int128 m_prod = (unsigned __int128)m[0] * m[1] * m[2];
  unsigned __int128 sum = 0;

  for (int i = 0; i < count; i++) {
    unsigned __int128 Mi = m_prod / m[i];
    unsigned __int128 yi = mod_inverse((__int128)Mi, (__int128)m[i]);

    sum = (sum + residues[i] * Mi * yi) % m_prod;
  }

  return sum;
}
