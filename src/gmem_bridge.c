#include "../include/gmem.h"
#include <stdio.h>

#define PAGE_SIZE 4096 // Standard 4KB Page

/**
 * Bridge Implementation of Projective Page Mapping (Phi)
 * Maps a linear system address (L) to a synthetic manifold address (M).
 * Ensures page-alignment and locality preservation.
 */
static uint64_t gmem_bridge_phi(uint64_t linear_addr, uint64_t seed) {
  // 1. Align to page boundary
  uint64_t page_index = linear_addr / PAGE_SIZE;
  uint64_t offset = linear_addr % PAGE_SIZE;

  // 2. Project Page Index into Manifold
  // Use a simple but deterministic projection for the prototype
  uint64_t manifold_base = page_index ^ seed;
  manifold_base *= 0xBF58476D1CE4E5B9ULL; // MurmurHash3-style mixer
  manifold_base ^= manifold_base >> 33;

  // 3. Re-combine with offset to maintain intra-page continuity
  return (manifold_base & ~(uint64_t)(PAGE_SIZE - 1)) | offset;
}

/**
 * Bridge Implementation of Coherence Operator (Psi)
 * Fetches data from either the physical overlay or the synthetic basis.
 */
float gmem_bridge_fetch_psi(gmem_ctx_t ctx, uint64_t linear_addr) {
  if (!ctx)
    return 0.0f;

  // Project linear address to synthetic manifold address
  // We use a fixed seed or the context seed
  uint64_t m_addr = gmem_bridge_phi(linear_addr, 0x1337BEEF);

  // Delegate to core GMC fetch (which handles overlay coherence internally)
  return gmem_fetch_f32(ctx, m_addr);
}

/**
 * Bridge Implementation for Writable Coherence
 */
void gmem_bridge_write_psi(gmem_ctx_t ctx, uint64_t linear_addr, float value) {
  if (!ctx)
    return;

  // Project linear address to synthetic manifold address
  uint64_t m_addr = gmem_bridge_phi(linear_addr, 0x1337BEEF);

  // Delegate to core GMC write
  gmem_write_f32(ctx, m_addr, value);
}
